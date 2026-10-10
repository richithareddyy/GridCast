"""Explicit data contracts: preserve gaps; reject ambiguous structural corruption."""

import numpy as np
import pandas as pd


class ValidationError(ValueError):
    """The source violates a documented contract."""


def require_columns(frame: pd.DataFrame, required: list[str]) -> None:
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValidationError(f"Missing required columns: {missing}")


def utc_timestamps(values, *, naive_is_utc: bool = False) -> pd.DatetimeIndex:
    """Naive dates are accepted only for sources explicitly documented as UTC."""
    if not naive_is_utc:
        parsed_values = [pd.Timestamp(value) for value in values]
        if any(pd.isna(value) for value in parsed_values):
            raise ValidationError("Missing timestamps are not allowed")
        if any(value.tz is None for value in parsed_values):
            raise ValidationError("Timezone-naive timestamp without a source UTC contract")
        parsed = pd.DatetimeIndex(pd.to_datetime(parsed_values, utc=True))
    else:
        parsed = pd.DatetimeIndex(pd.to_datetime(values, errors="raise", utc=True))
    if parsed.hasnans:
        raise ValidationError("Missing timestamps are not allowed")
    return parsed


def normalize_hourly(frame: pd.DataFrame, start: str, end: str):
    """Validate interval-start UTC keys and insert flagged, unfilled missing rows."""
    require_columns(frame, ["demand_mwh_reported"])
    idx = frame.index
    if not isinstance(idx, pd.DatetimeIndex) or str(idx.tz) != "UTC":
        raise ValidationError("Expected a timezone-aware UTC DatetimeIndex")
    if idx.hasnans:
        raise ValidationError("Missing timestamps")
    duplicates = idx[idx.duplicated(keep=False)]
    if len(duplicates):
        raise ValidationError(f"Duplicate UTC timestamps: {duplicates.astype(str).tolist()[:20]}")
    if not idx.equals(idx.floor("h")):
        raise ValidationError("Off-grid timestamps: expected exact hourly boundaries")
    expected = pd.date_range(start, end, freq="h", inclusive="left")
    if str(expected.tz) != "UTC":
        raise ValidationError("Configured boundaries must be UTC")
    if len(idx.difference(expected)):
        raise ValidationError("Source rows outside configured study period")
    missing = expected.difference(idx)
    out = frame.sort_index().reindex(expected)
    out.index.name = "timestamp_utc"
    out["missing_interval"] = ~out.index.isin(idx)
    raw = pd.to_numeric(out["demand_mwh_reported"], errors="raise")
    invalid = raw.notna() & (~np.isfinite(raw) | raw.le(0))
    out["invalid_demand"] = invalid
    out["missing_demand"] = raw.isna()
    out["demand_mw"] = raw.mask(invalid)  # One-hour MWh / 1 hour = average MW.
    local = out.index.tz_convert("America/New_York")
    day_counts = pd.Series(1, index=local).groupby(local.date).sum()
    full_days = day_counts.iloc[1:-1]  # UTC boundary days can be partial local days.
    audit = {
        "source_rows_in_study": len(frame),
        "normalized_rows": len(out),
        "was_sorted": bool(idx.is_monotonic_increasing),
        "duplicate_utc_timestamps": 0,
        "missing_intervals": len(missing),
        "missing_interval_timestamps": missing.astype(str).tolist(),
        "missing_demand": int(raw.isna().sum()),
        "invalid_demand": int(invalid.sum()),
        "invalid_demand_timestamps": out.index[invalid].astype(str).tolist(),
        "start_utc": str(out.index.min()),
        "last_interval_start_utc": str(out.index.max()),
        "end_exclusive_utc": str(pd.Timestamp(end)),
        "local_wall_clock_repeated_hours": int(local.tz_localize(None).duplicated().sum()),
        "local_23_hour_dates": [str(d) for d in full_days.index[full_days.eq(23)]],
        "local_25_hour_dates": [str(d) for d in full_days.index[full_days.eq(25)]],
        "partial_local_boundary_dates": {str(d): int(n) for d, n in day_counts.iloc[[0, -1]].items()},
        "leap_day_hours_utc": int(((out.index.month == 2) & (out.index.day == 29)).sum()),
    }
    return out, audit


def validate_periods(periods: dict, start: str, end: str) -> None:
    cursor = pd.Timestamp(start)
    for name in ["train", "validation", "calibration", "test"]:
        bounds = periods[name]
        lo, hi = pd.Timestamp(bounds["start"]), pd.Timestamp(bounds["end"])
        if str(lo.tz) != "UTC" or str(hi.tz) != "UTC" or lo != cursor or hi <= lo:
            raise ValidationError(f"Noncontiguous, unordered, or non-UTC period: {name}")
        cursor = hi
    if cursor != pd.Timestamp(end):
        raise ValidationError("Evaluation periods do not cover the study period")


def align_weather(demand: pd.DataFrame, weather: pd.DataFrame) -> pd.DataFrame:
    if str(weather.index.tz) != "UTC" or weather.index.has_duplicates:
        raise ValidationError("Weather keys must be unique and UTC")
    if not weather.index.equals(weather.index.floor("h")):
        raise ValidationError("Weather must be aggregated to interval-start hourly UTC")
    aligned = demand.join(weather, how="left", validate="one_to_one")
    if len(aligned) != len(demand):
        raise ValidationError("Weather join changed the number of demand intervals")
    return aligned
