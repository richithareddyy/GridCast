"""Known calendar context. Target lags and forecasting arrive in Milestone 2."""

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
    """Half-open interval-start partition; never include a neighboring split's label."""
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
