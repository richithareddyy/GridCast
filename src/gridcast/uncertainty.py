"""Empirical horizon-wise intervals using only matured rolling forecast residuals."""

import numpy as np
import pandas as pd

from gridcast.validation import ValidationError


def residual_intervals(forecasts, history, levels=(0.5, 0.8, 0.95), window_days=90, min_samples=30, delay=1):
    """Issue intervals for one origin; actuals in forecasts are never consulted.

    Rank ceil((n+1)*coverage), without clipping. This order statistic borrows the
    split-conformal finite-sample rank but does not claim exchangeable coverage.
    """
    if forecasts.origin_utc.nunique() != 1:
        raise ValidationError("Interval batch must contain exactly one forecast origin")
    if any(not 0 < level < 1 for level in levels) or window_days < 1 or min_samples < 1 or delay < 0:
        raise ValidationError("Invalid residual calibration settings")
    origin = forecasts.origin_utc.iloc[0]
    eligible = history.loc[
        (history.target_end_utc + pd.Timedelta(hours=delay) <= origin)
        & (history.target_start_utc >= origin - pd.Timedelta(days=window_days))
        & (history.origin_utc < origin)
    ]
    if eligible.duplicated(["origin_utc", "horizon"]).any():
        raise ValidationError("Duplicate residual calibration keys")
    rows = []
    for row in forecasts.itertuples(index=False):
        past = eligible.loc[eligible.horizon.eq(row.horizon)]
        residual = (past.actual - past.prediction).abs().to_numpy()
        if not np.isfinite(residual).all():
            raise ValidationError("Nonfinite matured calibration residual")
        n = len(residual)
        for level in levels:
            rank = int(np.ceil((n + 1) * level))
            if n < min_samples or rank > n:
                raise ValidationError("Insufficient residual history for finite interval")
            radius = float(np.partition(residual, rank - 1)[rank - 1])
            rows.append(
                {
                    "origin_utc": origin,
                    "target_start_utc": row.target_start_utc,
                    "horizon": row.horizon,
                    "level": level,
                    "prediction": row.prediction,
                    "lower": row.prediction - radius,
                    "upper": row.prediction + radius,
                    "calibration_n": n,
                    "quantile_rank": rank,
                    "oldest_residual_target": past.target_start_utc.min(),
                    "latest_residual_available_at": (past.target_end_utc + pd.Timedelta(hours=delay)).max(),
                }
            )
    return pd.DataFrame(rows)


def interval_metrics(frame):
    if frame.empty:
        return {"n": 0, "coverage": np.nan, "mean_width_mw": np.nan, "interval_score_mw": np.nan}
    if not np.isfinite(frame[["actual", "lower", "upper", "level"]].to_numpy()).all():
        raise ValidationError("Nonfinite interval scoring input")
    if (frame.lower > frame.upper).any() or not frame.level.between(0, 1, inclusive="neither").all():
        raise ValidationError("Invalid interval bounds or nominal level")
    width = frame.upper - frame.lower
    score = width + 2 / (1 - frame.level) * (
        (frame.lower - frame.actual).clip(lower=0) + (frame.actual - frame.upper).clip(lower=0)
    )
    return {
        "n": len(frame),
        "coverage": float(frame.actual.between(frame.lower, frame.upper).mean()),
        "mean_width_mw": float(width.mean()),
        "interval_score_mw": float(score.mean()),
    }


def interval_tables(intervals, thresholds):
    from gridcast.evaluation import condition_masks

    groups = {"overall": np.ones(len(intervals), dtype=bool), **condition_masks(intervals, thresholds)}
    condition_rows = []
    for level, frame in intervals.groupby("level"):
        for name, mask in groups.items():
            selected = pd.Series(mask, index=intervals.index).reindex(frame.index)
            condition_rows.append({"condition": name, "level": level, **interval_metrics(frame.loc[selected])})

    def grouped(columns):
        return pd.DataFrame(
            [
                {
                    **dict(zip(columns, key if isinstance(key, tuple) else (key,), strict=True)),
                    **interval_metrics(group),
                }
                for key, group in intervals.groupby(columns)
            ]
        )

    return {
        "interval_conditions": pd.DataFrame(condition_rows),
        "interval_horizons": grouped(["level", "horizon"]),
        "interval_months": grouped(["level", "fold"]),
    }
