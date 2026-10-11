"""Ex-post forecast anomalies and availability-aware trailing reliability."""

import numpy as np
import pandas as pd


def anomaly_events(intervals, level=0.95, delay=1):
    frame = intervals.loc[intervals.level.eq(level)].copy().sort_values("target_start_utc")
    frame["residual_mw"] = frame.actual - frame.prediction
    radius = (frame.upper - frame.lower) / 2
    frame["residual_to_radius"] = frame.residual_mw.abs() / radius.replace(0, np.nan)
    frame["is_anomaly"] = ~frame.actual.between(frame.lower, frame.upper)
    frame["direction"] = np.where(frame.actual > frame.upper, "above_interval", "below_interval")
    events = frame.loc[frame.is_anomaly].copy()
    if "target_end_utc" in events:
        events["detected_at_utc"] = events.target_end_utc + pd.Timedelta(hours=delay)
    # Consecutive flagged UTC hours form an episode; no inferred physical event label.
    starts = events.target_start_utc.diff().ne(pd.Timedelta(hours=1))
    events["episode_id"] = starts.cumsum()
    episodes = events.groupby("episode_id").agg(
        start_utc=("target_start_utc", "min"),
        end_utc=("target_start_utc", "max"),
        hours=("target_start_utc", "size"),
        max_absolute_residual_mw=("residual_mw", lambda x: x.abs().max()),
        max_residual_to_radius=("residual_to_radius", "max"),
    )
    return events, episodes.reset_index()


def rolling_reliability(intervals, window_days=30, min_hours=168, delay=1):
    frame = intervals.loc[intervals.level.eq(0.95)].copy().sort_values("target_start_utc")
    rows = []
    for origin in sorted(frame.origin_utc.unique()):
        known = frame.loc[
            (frame.target_end_utc + pd.Timedelta(hours=delay) <= origin)
            & (frame.target_start_utc >= origin - pd.Timedelta(days=window_days))
        ]
        if len(known) < min_hours:
            continue
        error = known.actual - known.prediction
        rows.append(
            {
                "as_of_utc": origin,
                "n": len(known),
                "MAE": float(error.abs().mean()),
                "RMSE": float(np.sqrt((error**2).mean())),
                "residual_mean_mw": float(error.mean()),
                "residual_variance_mw2": float(error.var(ddof=1)),
                "coverage_95": float(known.actual.between(known.lower, known.upper).mean()),
                "latest_label_available_at": (known.target_end_utc + pd.Timedelta(hours=delay)).max(),
            }
        )
    return pd.DataFrame(rows)
