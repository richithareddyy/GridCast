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


def condition_masks(predictions: pd.DataFrame, thresholds: dict):
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
    return masks


def condition_scores(predictions: pd.DataFrame, thresholds: dict):
    p = predictions
    masks = condition_masks(p, thresholds)
    return pd.DataFrame(
        [
            {"condition": name, "model": model, **metrics(p.loc[mask & p.model.eq(model)])}
            for name, mask in masks.items()
            for model in sorted(p.model.unique())
        ]
    )


def paired_comparisons(predictions, max_lags=7):
    """Daily paired MAE differences vs the predeclared daily baseline, HAC + Holm."""
    import statsmodels.api as sm
    from scipy.stats import norm
    from statsmodels.stats.multitest import multipletests

    p = predictions.copy()
    p["absolute_error"] = (p.actual - p.prediction).abs()
    losses = p.groupby(["origin_utc", "model"]).absolute_error.mean().unstack("model")
    if losses.isna().any().any() or "daily" not in losses:
        raise ValidationError("Paired comparison requires matched origins and the daily baseline")
    rows = []
    for model in losses.columns.drop("daily"):
        delta = (losses[model] - losses.daily).to_numpy()
        if len(delta) <= max_lags + 1:
            raise ValidationError("Too few origins for the requested HAC lag")
        fitted = sm.OLS(delta, np.ones((len(delta), 1))).fit(cov_type="HAC", cov_kwds={"maxlags": max_lags})
        mean, se = float(fitted.params[0]), float(fitted.bse[0])
        p_value = float(2 * norm.sf(abs(mean / se))) if se > 0 else float(mean == 0)
        rows.append(
            {
                "model": model,
                "reference": "daily",
                "daily_origins": len(delta),
                "hac_lags": max_lags,
                "mean_mae_difference_mw": mean,
                "hac_standard_error": se,
                "ci95_low": mean - 1.959963984540054 * se,
                "ci95_high": mean + 1.959963984540054 * se,
                "p_value": p_value,
            }
        )
    result = pd.DataFrame(rows)
    result["p_holm"] = multipletests(result.p_value, method="holm")[1]
    return result
