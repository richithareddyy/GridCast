import copy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from gridcast.anomalies import anomaly_events, rolling_reliability
from gridcast.data import load_config
from gridcast.features import calendar_features
from gridcast.reliability import evaluate_reliability, forecast_reserved, load_reserved_study
from gridcast.uncertainty import interval_metrics, residual_intervals
from gridcast.validation import ValidationError

ROOT = Path(__file__).parents[1]


def calibration():
    origins = pd.date_range("2023-11-01", periods=40, freq="D", tz="UTC")
    rows = []
    for n, origin in enumerate(origins):
        for h in [1, 24]:
            start = origin + pd.Timedelta(hours=h - 1)
            rows.append(
                {
                    "origin_utc": origin,
                    "target_start_utc": start,
                    "target_end_utc": start + pd.Timedelta(hours=1),
                    "horizon": h,
                    "prediction": 100.0,
                    "actual": 101.0 + n,
                }
            )
    history = pd.DataFrame(rows)
    origin = origins[-1] + pd.Timedelta(days=1)
    forecast = history.iloc[-2:].copy()
    forecast["origin_utc"] = origin
    forecast["target_start_utc"] = origin + pd.to_timedelta(forecast.horizon - 1, unit="h")
    forecast["target_end_utc"] = forecast.target_start_utc + pd.Timedelta(hours=1)
    return history, forecast


def test_empirical_rank_and_delay_are_hand_calculated():
    history, forecast = calibration()
    result = residual_intervals(forecast, history)
    first = result.loc[result.horizon.eq(1)].set_index("level")
    assert first.loc[0.5, "upper"] == 121
    assert first.loc[0.8, "upper"] == 133
    assert first.loc[0.95, "upper"] == 139
    assert (first.calibration_n == 40).all()
    assert (result.loc[result.horizon.eq(24), "calibration_n"] == 39).all()
    assert (result.latest_residual_available_at <= result.origin_utc).all()
    for _, group in result.groupby("horizon"):
        ordered = group.sort_values("level")
        assert ordered.lower.is_monotonic_decreasing and ordered.upper.is_monotonic_increasing


def test_interval_future_mutation_and_new_forecast_actual_do_not_change_bounds():
    history, forecast = calibration()
    before = residual_intervals(forecast, history)
    changed = history.copy()
    changed.loc[changed.target_end_utc + pd.Timedelta(hours=1) > forecast.origin_utc.iloc[0], "actual"] = 1e12
    future = forecast.copy()
    future["actual"] = 1e15
    after = residual_intervals(future, pd.concat([changed, future], ignore_index=True))
    pd.testing.assert_frame_equal(before, after)


def test_interval_expires_old_residuals_and_rejects_insufficient_history():
    history, forecast = calibration()
    with pytest.raises(ValidationError, match="Insufficient"):
        residual_intervals(forecast, history, window_days=20)
    with pytest.raises(ValidationError, match="Insufficient"):
        residual_intervals(forecast, history, levels=[0.999])
    with pytest.raises(ValidationError, match="Duplicate"):
        residual_intervals(forecast, pd.concat([history, history.iloc[:1]], ignore_index=True))


def test_interval_score_penalizes_width_and_misses():
    frame = pd.DataFrame({"actual": [5.0, 12.0], "lower": [0.0, 0.0], "upper": [10.0, 10.0], "level": [0.8, 0.8]})
    metrics = interval_metrics(frame)
    assert metrics["coverage"] == 0.5
    assert metrics["mean_width_mw"] == 10
    assert metrics["interval_score_mw"] == pytest.approx(20)
    frame.loc[0, "upper"] = -1
    with pytest.raises(ValidationError):
        interval_metrics(frame)


def test_anomalies_are_strict_violations_grouped_by_elapsed_hours():
    idx = pd.date_range("2024-01-01", periods=5, freq="h", tz="UTC")
    frame = pd.DataFrame(
        {
            "target_start_utc": idx,
            "level": 0.95,
            "actual": [10, 12, 13, 0, -1],
            "prediction": 5.0,
            "lower": 0.0,
            "upper": 10.0,
        }
    )
    events, episodes = anomaly_events(frame)
    assert events.target_start_utc.tolist() == idx[[1, 2, 4]].tolist()
    assert episodes.hours.tolist() == [2, 1]
    assert events.direction.tolist() == ["above_interval", "above_interval", "below_interval"]


def test_rolling_monitor_does_not_use_unpublished_or_future_labels():
    idx = pd.date_range("2024-01-01", periods=24 * 12, freq="h", tz="UTC")
    frame = pd.DataFrame(
        {
            "target_start_utc": idx,
            "target_end_utc": idx + pd.Timedelta(hours=1),
            "origin_utc": idx.floor("D"),
            "level": 0.95,
            "actual": 100.0,
            "prediction": 90.0,
            "lower": 80.0,
            "upper": 110.0,
        }
    )
    before = rolling_reliability(frame)
    cutoff = pd.Timestamp("2024-01-10T00:00Z")
    frame.loc[frame.target_end_utc + pd.Timedelta(hours=1) > cutoff, "actual"] = 1e9
    after = rolling_reliability(frame)
    pd.testing.assert_frame_equal(before.loc[before.as_of_utc <= cutoff], after.loc[after.as_of_utc <= cutoff])
    assert (before.latest_label_available_at <= before.as_of_utc).all()
    assert (before.MAE == 10).all() and (before.residual_variance_mw2 == 0).all()


def test_reserved_forecast_future_mutation_preserves_first_origin_and_fit():
    cfg = load_config(ROOT, "evaluation.yaml")
    cfg["periods"]["calibration"] = {"start": "2023-07-01T00:00:00Z", "end": "2023-07-02T00:00:00Z"}
    cfg["periods"]["test"] = {"start": "2023-07-02T00:00:00Z", "end": "2023-07-03T00:00:00Z"}
    idx = pd.date_range("2023-01-01", "2023-07-03", inclusive="left", freq="h", tz="UTC")
    frame = calendar_features(idx)
    frame["demand_mw"] = 12000 + 500 * np.sin(np.arange(len(idx)) / 24) + np.arange(len(idx)) / 10
    frame["weather_temperature_c"] = 10.0
    selection = {
        "frozen_lightgbm_parameters": cfg["learning"]["lightgbm"]["parameters"],
        "frozen_boost_rounds": 5,
        "seed": 42,
        "lookback_hours": 168,
        "reporting_delay_hours": 1,
    }
    first, fits = forecast_reserved(frame, cfg, selection)
    origin = pd.Timestamp("2023-07-01T00:00Z")
    changed = copy.deepcopy(frame)
    changed.loc[changed.index >= origin - pd.Timedelta(hours=1), "demand_mw"] = 1e9
    changed.loc[changed.index >= origin, "weather_temperature_c"] = -1e6
    second, fits2 = forecast_reserved(changed, cfg, selection)
    assert fits == fits2
    assert len(first) == 2 * 24 * 2
    pd.testing.assert_series_equal(
        first.loc[first.origin_utc.eq(origin), "prediction"], second.loc[second.origin_utc.eq(origin), "prediction"]
    )
    assert fits[0]["boost_rounds"] == 5


def test_selection_guard_precedes_reading_reserved_partitions(tmp_path, monkeypatch):
    predictions = tmp_path / "reports/predictions"
    predictions.mkdir(parents=True)
    (predictions / "validation.parquet").write_bytes(b"changed")

    def forbid(*args, **kwargs):
        raise AssertionError("Reserved outcomes must not be read")

    monkeypatch.setattr(pd, "read_parquet", forbid)
    with pytest.raises(ValidationError, match="selection evidence"):
        load_reserved_study(tmp_path, {}, {"selected_model": "lightgbm", "validation_predictions_sha256": "incorrect"})


def test_reliability_joins_context_after_intervals_and_preserves_condition_counts():
    history, forecast = calibration()
    history["partition"] = "calibration"
    forecast["partition"] = "test"
    points = pd.concat([history, forecast], ignore_index=True)
    points["model"] = "lightgbm"
    points["mase_scale"] = 10.0
    points["fold"] = "2023-12"
    points["weekend"] = False
    points["holiday"] = False
    points["peak_hour"] = False
    points["month"] = 12
    points["weather_temperature_c"] = np.nan
    daily = points.copy()
    daily["model"] = "daily"
    cfg = load_config(ROOT, "evaluation.yaml")
    cfg["uncertainty"].update(
        minimum_samples_per_horizon=30, anomaly_level=0.95, monitor_window_days=30, monitor_minimum_hours=168
    )
    intervals, outputs = evaluate_reliability(
        pd.concat([points, daily], ignore_index=True),
        cfg,
        {"extreme_demand_threshold_mw": 130, "extreme_temperature_low_c": 0, "extreme_temperature_high_c": 30},
    )
    assert len(intervals) == 6
    conditions = outputs["interval_conditions"]
    assert (conditions.loc[conditions.condition.eq("weather_unknown"), "n"] == 2).all()
    assert (conditions.loc[conditions.condition.eq("extreme_temperature"), "n"] == 0).all()
    assert (intervals.latest_residual_available_at <= intervals.origin_utc).all()
