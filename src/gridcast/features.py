"""Calendar and forecast-origin predictors with explicit temporal availability."""

import holidays
import pandas as pd

from gridcast.validation import ValidationError


def calendar_features(index: pd.DatetimeIndex, timezone: str = "America/New_York"):
    if index.tz is None:
        raise ValidationError("Calendar requires timezone-aware timestamps")
    local = index.tz_convert(timezone)
    calendar = holidays.US(years=range(local.year.min() - 1, local.year.max() + 2), observed=True)
    result = pd.DataFrame(index=index)
    result["local_time"] = local.astype(str)
    result["utc_offset_hours"] = [v.utcoffset().total_seconds() / 3600 for v in local]
    result["hour"] = local.hour
    result["weekday"] = local.dayofweek
    result["month"] = local.month
    result["year"] = local.year
    result["weekend"] = local.dayofweek >= 5
    result["holiday_name"] = [calendar.get(d, "") for d in local.date]
    result["holiday"] = result["holiday_name"].ne("")
    result["leap_day"] = (local.month == 2) & (local.day == 29)
    result["peak_hour"] = (local.hour >= 16) & (local.hour <= 20)
    return result


def partition(frame: pd.DataFrame, bounds: dict) -> pd.DataFrame:
    return frame.loc[
        (frame.index >= pd.Timestamp(bounds["start"])) & (frame.index < pd.Timestamp(bounds["end"]))
    ].copy()


def available_history(frame: pd.DataFrame, origin, reporting_delay_hours: int = 1):
    """Interval-start index; an observation matures at interval end + reporting delay."""
    origin = pd.Timestamp(origin)
    if origin.tz is None or frame.index.tz is None:
        raise ValidationError("History cutoff requires timezone-aware timestamps")
    if reporting_delay_hours < 0:
        raise ValidationError("Reporting delay cannot be negative")
    return frame.loc[frame.index + pd.Timedelta(hours=1 + reporting_delay_hours) <= origin].copy()


def validate_history(series: pd.Series):
    """Require a real hourly axis before positional lags or autoregressive fitting."""
    if not isinstance(series.index, pd.DatetimeIndex) or str(series.index.tz) != "UTC":
        raise ValidationError("Demand history must have a UTC DatetimeIndex")
    if series.empty or series.index.has_duplicates or not series.index.is_monotonic_increasing:
        raise ValidationError("History must be nonempty, sorted and unique")
    if not series.index.equals(pd.date_range(series.index[0], series.index[-1], freq="h")):
        raise ValidationError("History must be an exact hourly grid; represent gaps explicitly")


def forecast_features(series: pd.Series, origin, horizon=24, delay=1):
    """A forecast-origin feature matrix; target observations never enter predictors.

    origin_last_demand represents the latest available history. Target lag_24 can be unknown;
    leave it null and record availability rather than reading across the cutoff.
    """
    validate_history(series)
    origin = pd.Timestamp(origin)
    if horizon < 1 or delay < 0 or origin.tz is None:
        raise ValidationError("Invalid forecast horizon, origin or reporting delay")
    history = available_history(series.to_frame("y"), origin, delay).y
    if history.empty:
        raise ValidationError("No demand is available at this forecast origin")
    targets = pd.date_range(origin, periods=horizon, freq="h")
    features = calendar_features(targets).drop(columns=["local_time", "holiday_name"])
    features["horizon"] = range(1, horizon + 1)
    features["origin_last_demand"] = history.iloc[-1]
    for lag in [24, 48, 168]:
        sources = targets - pd.Timedelta(hours=lag)
        features[f"lag_{lag}"] = history.reindex(sources).to_numpy()
        features[f"lag_{lag}_available"] = features[f"lag_{lag}"].notna()
    for window in [24, 168]:
        tail = history.iloc[-window:]
        complete = len(tail) == window and tail.notna().all()
        for stat in ["mean", "std", "min", "max"]:
            features[f"rolling_{window}_{stat}"] = getattr(tail, stat)() if complete else float("nan")
    features.attrs["latest_history_start"] = str(history.index[-1])
    features.attrs["latest_history_available_at"] = str(history.index[-1] + pd.Timedelta(hours=delay + 1))
    return features
