from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from gridcast.backtesting import backtest, load_development, monthly_folds
from gridcast.baselines import predict_baselines
from gridcast.data import load_config
from gridcast.evaluation import condition_scores, mase_scale, metrics
from gridcast.features import calendar_features, forecast_features
from gridcast.models import HarmonicAR1, harmonic_design
from gridcast.validation import ValidationError

ROOT = Path(__file__).parents[1]


def series():
    index = pd.date_range("2022-01-01", periods=600, freq="h", tz="UTC")
    return pd.Series(1000 + np.arange(len(index)), index=index, name="demand_mw")


def test_baselines_hand_calculated_and_last_horizon_latency():
    y = series()
    origin = y.index[400]
    result = predict_baselines(y, origin)
    np.testing.assert_array_equal(result.naive, [1398] * 24)
    np.testing.assert_array_equal(result.daily.iloc[:23], np.arange(1376, 1399))
    assert result.daily.iloc[23] == 1375  # 423 - 48, not unpublished 423 - 24.
    np.testing.assert_array_equal(result.weekly, np.arange(1232, 1256))
    assert result.index[0] == origin and result.index[-1] == origin + pd.Timedelta(hours=23)
    for column in ["daily_source_utc", "weekly_source_utc", "naive_source_utc"]:
        assert (result[column] + pd.Timedelta(hours=2) <= origin).all()


def test_baselines_zero_delay_uses_all_24_previous_day_values():
    y = series()
    result = predict_baselines(y, y.index[400], delay=0)
    np.testing.assert_array_equal(result.daily, np.arange(1376, 1400))


def test_features_mask_unknown_lag_and_use_only_trailing_history():
    y = series()
    result = forecast_features(y, y.index[400])
    assert result.lag_24.iloc[:23].notna().all()
    assert pd.isna(result.lag_24.iloc[-1]) and not result.lag_24_available.iloc[-1]
    assert (result.origin_last_demand == 1398).all()
    tail = y.iloc[375:399]
    assert (result.rolling_24_mean == tail.mean()).all()
    assert (result.rolling_24_std == tail.std()).all()
    assert (result.rolling_24_min == 1375).all() and (result.rolling_24_max == 1398).all()
    assert result.horizon.tolist() == list(range(1, 25))


def test_mutating_unpublished_and_future_targets_changes_neither_features_nor_baselines():
    y = series()
    origin = y.index[400]
    before_features = forecast_features(y, origin)
    before_baselines = predict_baselines(y, origin)
    y.loc[y.index >= origin - pd.Timedelta(hours=1)] = 1e12
    pd.testing.assert_frame_equal(before_features, forecast_features(y, origin))
    pd.testing.assert_frame_equal(before_baselines, predict_baselines(y, origin))


def test_missing_history_is_not_silently_skipped_by_rolling_or_baseline():
    y = series()
    y.iloc[398] = np.nan
    features = forecast_features(y, y.index[400])
    assert features.rolling_24_mean.isna().all()
    with pytest.raises(ValidationError, match="Latest available"):
        predict_baselines(y, y.index[400])


@pytest.mark.parametrize(
    "transform", [lambda y: y.iloc[::-1], lambda y: y.drop(y.index[4]), lambda y: pd.concat([y, y.iloc[[-1]]])]
)
def test_bad_time_axes_rejected(transform):
    y = series()
    with pytest.raises(ValidationError):
        forecast_features(transform(y), y.index[400])


def test_metrics_known_values_and_nonfinite_rejection():
    frame = pd.DataFrame({"actual": [10.0, 20.0], "prediction": [8.0, 24.0], "mase_scale": [2.0, 4.0]})
    result = metrics(frame)
    assert result["MAE"] == 3
    assert result["RMSE"] == pytest.approx(np.sqrt(10))
    assert result["WAPE"] == 20
    assert result["MASE"] == 1  # Mean row-level scaled losses, not loss / mean scale.
    frame.loc[0, "actual"] = np.nan
    with pytest.raises(ValidationError):
        metrics(frame)
    assert metrics(frame.iloc[:0])["n"] == 0


def test_mase_scale_is_training_only_and_rejects_degenerate_scale():
    assert mase_scale(series().iloc[:400]) == 168
    with pytest.raises(ValidationError, match="zero seasonal"):
        mase_scale(pd.Series(np.ones(400)))


def test_fold_boundaries_have_no_gaps_and_do_not_touch_holdouts():
    cfg = load_config(ROOT, "evaluation.yaml")
    folds = list(monthly_folds(cfg))
    assert len(folds) == 9
    assert folds[0][1] == pd.Timestamp(cfg["periods"]["validation"]["start"])
    assert folds[-1][2] == pd.Timestamp(cfg["periods"]["calibration"]["start"])
    assert all(a[2] == b[1] for a, b in zip(folds, folds[1:], strict=False))
    assert sum(int((end - start).total_seconds() / 3600) for _, start, end in folds) == 6552


def test_development_loader_never_opens_calibration_or_test(monkeypatch):
    cfg = load_config(ROOT, "evaluation.yaml")
    calls = []

    def read(path):
        name = Path(path).stem
        assert name in ["train", "validation"]
        calls.append(name)
        bounds = cfg["periods"][name]
        idx = pd.date_range(bounds["start"], bounds["end"], inclusive="left", freq="h")
        return pd.DataFrame({"demand_mw": 1000.0}, index=idx)

    monkeypatch.setattr(pd, "read_parquet", read)
    data = load_development(ROOT, cfg)
    assert calls == ["train", "validation"]
    assert data.index.max() < pd.Timestamp(cfg["periods"]["calibration"]["start"])


def test_ar_residual_forecast_bridges_unpublished_hour():
    cfg = load_config(ROOT, "evaluation.yaml")["statistical_model"]
    y = series()
    origin = y.index[400]
    model = HarmonicAR1(cfg)
    model.coefficients = harmonic_design(y.index[:2], cfg).iloc[0] * 0
    model.rho = 0.5
    model.fit_end = y.index[300]
    result = model.predict(y, origin)
    assert result.iloc[0] == 1398 * 0.5**2
    assert result.iloc[-1] == pytest.approx(1398 * 0.5**25)
    y.iloc[399:] = 1e15
    pd.testing.assert_series_equal(result, model.predict(y, origin))
    model.fit_end = origin
    with pytest.raises(ValidationError, match="unavailable"):
        model.predict(y, origin)


def synthetic_study():
    cfg = load_config(ROOT, "evaluation.yaml")
    cfg["periods"]["validation"] = {"start": "2023-01-01T00:00Z", "end": "2023-01-03T00:00Z"}
    idx = pd.date_range("2022-01-01", "2023-01-03", inclusive="left", freq="h", tz="UTC")
    frame = calendar_features(idx)
    design = harmonic_design(idx, cfg["statistical_model"])
    rng = np.random.default_rng(42)
    innovation = rng.normal(0, 40, len(idx))
    residual = np.zeros(len(idx))
    for n in range(1, len(idx)):
        residual[n] = 0.7 * residual[n - 1] + innovation[n]
    beta = np.arange(design.shape[1]) * 3 + 10.0
    beta[0] = 12000
    frame["demand_mw"] = design.to_numpy() @ beta + residual
    frame["weather_temperature_c"] = 10.0
    return frame, cfg


def test_statistical_fit_and_prediction_are_deterministic():
    frame, cfg = synthetic_study()
    train = frame.loc[frame.index < "2022-12-31", "demand_mw"]
    first = HarmonicAR1(cfg["statistical_model"]).fit(train)
    second = HarmonicAR1(cfg["statistical_model"]).fit(train)
    assert 0 < first.rho < 1 and first.fit_end == train.index[-1]
    pd.testing.assert_series_equal(first.coefficients, second.coefficients)
    assert first.rho == second.rho


def test_end_to_end_future_mutation_preserves_first_origin_and_fit():
    frame, cfg = synthetic_study()
    p1, f1, audit, fits1 = backtest(frame, cfg)
    origin = pd.Timestamp("2023-01-01T00:00Z")
    frame.loc[frame.index >= origin - pd.Timedelta(hours=1), "demand_mw"] *= 5
    frame.loc[frame.index >= origin, "weather_temperature_c"] = -100
    p2, f2, _, fits2 = backtest(frame, cfg)
    assert fits1 == fits2
    assert len(p1) == 2 * 24 * 4
    pd.testing.assert_series_equal(
        p1.loc[p1.origin_utc.eq(origin), "prediction"], p2.loc[p2.origin_utc.eq(origin), "prediction"]
    )
    pd.testing.assert_frame_equal(f1.loc[f1.origin_utc.eq(origin)], f2.loc[f2.origin_utc.eq(origin)])
    assert (audit.latest_history_available_at <= audit.origin_utc).all()
    assert (p1.target_end_utc - p1.origin_utc == pd.to_timedelta(p1.horizon, unit="h")).all()
    assert not p1.duplicated(["model", "origin_utc", "horizon"]).any()


def test_unknown_weather_is_not_counted_as_an_extreme_temperature():
    p = pd.DataFrame(
        {
            "actual": [100.0, 200.0],
            "prediction": [90.0, 220.0],
            "mase_scale": [20.0, 20.0],
            "weekend": [False, False],
            "holiday": [False, True],
            "peak_hour": [True, False],
            "month": [1, 1],
            "weather_temperature_c": [np.nan, -10.0],
            "model": ["a", "a"],
        }
    )
    scores = condition_scores(
        p, {"extreme_demand_threshold_mw": 150, "extreme_temperature_low_c": 0, "extreme_temperature_high_c": 30}
    )
    assert scores.set_index("condition").loc["weather_unknown", "n"] == 1
    assert scores.set_index("condition").loc["extreme_temperature", "n"] == 1
