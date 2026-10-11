"""Persistence and strong seasonal baselines with explicit reporting latency."""

import numpy as np
import pandas as pd

from gridcast.features import available_history, validate_history
from gridcast.validation import ValidationError


def predict_baselines(series: pd.Series, origin, horizon=24, delay=1):
    validate_history(series)
    origin = pd.Timestamp(origin)
    if horizon < 1 or delay < 0 or origin.tz is None:
        raise ValidationError("Invalid forecast request")
    history = available_history(series.to_frame("y"), origin, delay).y
    if history.empty or pd.isna(history.iloc[-1]):
        raise ValidationError("Latest available observation is absent; no silent imputation")
    targets = pd.date_range(origin, periods=horizon, freq="h")
    daily_sources = targets - pd.Timedelta(hours=24)
    # Roll back whole seasonal cycles until the source is published, never to a future label.
    while (daily_sources > history.index[-1]).any():
        daily_sources = daily_sources.where(daily_sources <= history.index[-1], daily_sources - pd.Timedelta(hours=24))
    weekly_sources = targets - pd.Timedelta(hours=168)
    if (weekly_sources > history.index[-1]).any():
        raise ValidationError("Weekly baseline unavailable for requested horizon")
    result = pd.DataFrame(
        {
            "naive": np.repeat(history.iloc[-1], horizon),
            "daily": history.reindex(daily_sources).to_numpy(),
            "weekly": history.reindex(weekly_sources).to_numpy(),
            "daily_source_utc": daily_sources,
            "weekly_source_utc": weekly_sources,
            "naive_source_utc": history.index[-1],
        },
        index=targets,
    )
    if not np.isfinite(result[["naive", "daily", "weekly"]].to_numpy()).all():
        raise ValidationError("Required baseline source missing/nonfinite; no fallback imputation")
    return result
