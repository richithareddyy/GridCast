"""Comparable metrics and condition slices, with training-only scaling."""

import numpy as np
import pandas as pd

from gridcast.validation import ValidationError


def mase_scale(history: pd.Series, seasonal_lag=168):
    if seasonal_lag < 1 or len(history) <= seasonal_lag or not np.isfinite(history).all():
        raise ValidationError("MASE needs finite training history longer than the seasonal lag")
    scale = float((history - history.shift(seasonal_lag)).abs().iloc[seasonal_lag:].mean())
    if scale <= 0:
        raise ValidationError("MASE undefined for a zero seasonal training scale")
    return scale


def metrics(frame: pd.DataFrame):
    if frame.empty:
        return {"n": 0, "MAE": np.nan, "RMSE": np.nan, "WAPE": np.nan, "MASE": np.nan}
    if not np.isfinite(frame[["actual", "prediction", "mase_scale"]].to_numpy()).all():
        raise ValidationError("Nonfinite metric inputs; define and report missing-label policy")
    if (frame.mase_scale <= 0).any():
        raise ValidationError("MASE denominator must be positive")
    error = frame.actual - frame.prediction
    denominator = frame.actual.abs().sum()
    return {
        "n": len(frame),
        "MAE": float(error.abs().mean()),
        "RMSE": float(np.sqrt((error**2).mean())),
        "WAPE": float(100 * error.abs().sum() / denominator) if denominator else np.nan,
        "MASE": float((error.abs() / frame.mase_scale).mean()),
    }


def score_groups(predictions: pd.DataFrame, groups):
    rows = []
    for keys, frame in predictions.groupby(groups, observed=True, sort=True):
        keys = keys if isinstance(keys, tuple) else (keys,)
        rows.append({**dict(zip(groups, keys, strict=True)), **metrics(frame)})
    return pd.DataFrame(rows)


def condition_scores(predictions: pd.DataFrame, thresholds: dict):
    p = predictions
    weather = p.weather_temperature_c
    masks = {
        "normal_weekday": ~p.weekend & ~p.holiday,
        "weekend": p.weekend,
        "holiday": p.holiday,
        "peak_hours": p.peak_hour,
        "extreme_demand": p.actual.ge(thresholds["extreme_demand_threshold_mw"]),
        "nonextreme_demand": p.actual.lt(thresholds["extreme_demand_threshold_mw"]),
        "extreme_temperature": weather.le(thresholds["extreme_temperature_low_c"])
        | weather.ge(thresholds["extreme_temperature_high_c"]),
        "weather_unknown": weather.isna(),
        "winter": p.month.isin([12, 1, 2]),
        "spring": p.month.isin([3, 4, 5]),
        "summer": p.month.isin([6, 7, 8]),
        "autumn": p.month.isin([9, 10, 11]),
    }
    return pd.DataFrame(
        [
            {"condition": name, "model": model, **metrics(p.loc[mask & p.model.eq(model)])}
            for name, mask in masks.items()
            for model in sorted(p.model.unique())
        ]
    )
