import numpy as np
import pandas as pd
import pytest

from gridcast.features import calendar_features, partition
from gridcast.validation import (
    ValidationError,
    align_weather,
    normalize_hourly,
    require_columns,
    utc_timestamps,
    validate_periods,
)


def frame(start="2024-01-01T00:00Z", periods=5):
    return pd.DataFrame(
        {"demand_mwh_reported": np.arange(periods) + 100.0},
        index=pd.date_range(start, periods=periods, freq="h"),
    )


def test_naive_dates_require_explicit_source_contract():
    with pytest.raises(ValidationError, match="Timezone-naive"):
        utc_timestamps(["2024-01-01 00:00"])
    assert str(utc_timestamps(["2024-01-01 00:00"], naive_is_utc=True).tz) == "UTC"


def test_offset_conversion_and_invalid_dates():
    assert utc_timestamps(["2024-01-01T00:00-05:00"])[0] == pd.Timestamp("2024-01-01T05:00Z")
    with pytest.raises(ValueError):
        utc_timestamps(["not-a-timestamp"], naive_is_utc=True)
    with pytest.raises(ValidationError, match="Missing"):
        utc_timestamps([None], naive_is_utc=True)


def test_duplicate_utc_rows_are_rejected_even_if_identical():
    raw = frame()
    with pytest.raises(ValidationError, match="Duplicate UTC"):
        normalize_hourly(pd.concat([raw, raw.iloc[[0]]]), raw.index[0], raw.index[-1])


def test_missing_intervals_are_inserted_and_flagged_not_imputed():
    raw = frame().drop(frame().index[2])
    out, audit = normalize_hourly(raw, "2024-01-01T00:00Z", "2024-01-01T05:00Z")
    assert len(out) == 5 and audit["missing_intervals"] == 1
    assert out.iloc[2].missing_interval and out.iloc[2].missing_demand
    assert pd.isna(out.iloc[2].demand_mw)


@pytest.mark.parametrize("bad", [0, -5, np.inf, -np.inf])
def test_invalid_target_preserves_source_but_masks_model_value(bad):
    raw = frame()
    raw.iloc[1, 0] = bad
    out, audit = normalize_hourly(raw, "2024-01-01T00:00Z", "2024-01-01T05:00Z")
    assert out.iloc[1].demand_mwh_reported == bad
    assert pd.isna(out.iloc[1].demand_mw) and audit["invalid_demand"] == 1


def test_sorting_and_one_hour_unit_conversion():
    out, audit = normalize_hourly(frame().iloc[::-1], "2024-01-01T00:00Z", "2024-01-01T05:00Z")
    assert not audit["was_sorted"] and out.index.is_monotonic_increasing
    np.testing.assert_array_equal(out.demand_mw, np.arange(5) + 100)


def test_off_grid_timezone_and_schema_are_rejected():
    raw = frame()
    raw.index += pd.Timedelta(minutes=30)
    with pytest.raises(ValidationError, match="Off-grid"):
        normalize_hourly(raw, "2024-01-01T00:00Z", "2024-01-01T05:00Z")
    raw.index = raw.index.tz_localize(None)
    with pytest.raises(ValidationError, match="UTC"):
        normalize_hourly(raw, "2024-01-01T00:00Z", "2024-01-01T05:00Z")
    with pytest.raises(ValidationError, match="Missing required"):
        require_columns(pd.DataFrame({"wrong": []}), ["Demand"])


def test_spring_dst_has_23_real_hours_and_no_synthetic_hour():
    idx = pd.date_range("2024-03-10T05:00Z", "2024-03-11T04:00Z", inclusive="left", freq="h")
    cal = calendar_features(idx)
    assert len(cal) == 23 and 2 not in cal.hour.values
    assert set(cal.utc_offset_hours) == {-5, -4}


def test_fall_dst_preserves_both_repeated_local_hours():
    idx = pd.date_range("2024-11-03T04:00Z", "2024-11-04T05:00Z", inclusive="left", freq="h")
    cal = calendar_features(idx)
    repeated = cal.loc[cal.hour.eq(1)]
    assert len(cal) == 25 and len(repeated) == 2 and idx.is_unique
    assert set(repeated.utc_offset_hours) == {-5, -4}


def test_leap_day_and_observed_holiday_calendar():
    idx = pd.date_range("2024-02-29T05:00Z", periods=24, freq="h")
    assert calendar_features(idx).leap_day.sum() == 24
    cal = calendar_features(pd.DatetimeIndex(["2021-07-05T12:00Z", "2021-07-06T12:00Z"]))
    assert cal.holiday.tolist() == [True, False]
    assert "observed" in cal.holiday_name.iloc[0]


def test_partition_boundaries_are_half_open():
    raw = frame()
    a = partition(raw, {"start": raw.index[0], "end": raw.index[2]})
    b = partition(raw, {"start": raw.index[2], "end": raw.index[-1] + pd.Timedelta(hours=1)})
    assert len(a.index.intersection(b.index)) == 0
    pd.testing.assert_frame_equal(pd.concat([a, b]), raw)


def test_chronological_config_has_no_overlap_or_gap():
    periods = {}
    for n, key in enumerate(["train", "validation", "calibration", "test"]):
        periods[key] = {"start": f"202{n}-01-01T00:00Z", "end": f"202{n + 1}-01-01T00:00Z"}
    validate_periods(periods, "2020-01-01T00:00Z", "2024-01-01T00:00Z")
    periods["test"]["start"] = "2022-12-31T00:00Z"
    with pytest.raises(ValidationError):
        validate_periods(periods, "2020-01-01T00:00Z", "2024-01-01T00:00Z")


def test_weather_left_join_preserves_gaps_and_cardinality():
    load = frame()
    weather = pd.DataFrame({"temperature_c": [10.0]}, index=load.index[:1])
    joined = align_weather(load, weather)
    assert len(joined) == len(load) and joined.temperature_c.isna().sum() == 4
    with pytest.raises(ValidationError, match="unique"):
        align_weather(load, pd.concat([weather, weather]))


def test_explicit_dst_offsets_map_to_distinct_utc_hours():
    parsed = utc_timestamps(["2024-11-03T01:00:00-04:00", "2024-11-03T01:00:00-05:00"])
    assert parsed.is_unique
    assert parsed[1] - parsed[0] == pd.Timedelta(hours=1)
    assert parsed[0] == pd.Timestamp("2024-11-03T05:00Z")
