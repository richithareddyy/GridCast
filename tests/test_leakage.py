import numpy as np
import pandas as pd
import pytest

from gridcast.eda import summarize_training
from gridcast.features import available_history, calendar_features
from gridcast.validation import ValidationError


def study():
    idx = pd.date_range("2022-01-01", "2024-01-01", inclusive="left", freq="h", tz="UTC")
    frame = calendar_features(idx)
    frame["demand_mw"] = 12000 + np.sin(np.arange(len(idx)) / 24) * 1000
    frame["weather_temperature_c"] = np.linspace(-10, 35, len(idx))
    cfg = {
        "periods": {"train": {"start": "2022-01-01T00:00Z", "end": "2023-01-01T00:00Z"}},
        "conditions": {
            "extremes": {"quantile": 0.95},
            "extreme_temperature": {"lower_quantile": 0.05, "upper_quantile": 0.95},
        },
    }
    return frame, cfg


def test_future_targets_weather_do_not_change_training_eda_or_thresholds():
    frame, config = study()
    before = summarize_training(frame, config)
    frame.loc[frame.index >= "2023-01-01", ["demand_mw", "weather_temperature_c"]] = 1e12
    assert summarize_training(frame, config) == before


def test_history_cutoff_respects_hour_end_and_reporting_delay():
    frame, _ = study()
    origin = pd.Timestamp("2022-01-02T00:00Z")
    history = available_history(frame, origin)
    assert history.index.max() == origin - pd.Timedelta(hours=2)
    assert (history.index + pd.Timedelta(hours=2) <= origin).all()
    assert origin - pd.Timedelta(hours=1) not in history.index


def test_future_target_mutation_cannot_change_available_history():
    frame, _ = study()
    origin = pd.Timestamp("2022-02-01T00:00Z")
    before = available_history(frame, origin)
    frame.loc[frame.index >= origin - pd.Timedelta(hours=1), "demand_mw"] = -999999
    pd.testing.assert_frame_equal(before, available_history(frame, origin))


def test_naive_cutoff_and_negative_latency_rejected():
    frame, _ = study()
    with pytest.raises(ValidationError):
        available_history(frame, "2022-02-01")
    with pytest.raises(ValidationError):
        available_history(frame, "2022-02-01T00:00Z", -1)


def test_calendar_is_deterministic_and_independent_of_targets():
    frame, _ = study()
    pd.testing.assert_frame_equal(calendar_features(frame.index), calendar_features(frame.index))


def test_24_hour_path_alignment_and_daily_baseline_availability():
    origin = pd.Timestamp("2023-01-01T00:00Z")
    target_ends = pd.date_range(origin + pd.Timedelta(hours=1), periods=24, freq="h")
    available_cutoff = origin - pd.Timedelta(hours=1)
    daily_candidates = target_ends - pd.Timedelta(hours=24)
    assert (daily_candidates <= available_cutoff).sum() == 23
    assert daily_candidates[-1] > available_cutoff
    assert target_ends[-1] - pd.Timedelta(hours=48) <= available_cutoff
