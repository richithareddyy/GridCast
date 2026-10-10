import pandas as pd
import pytest

from gridcast.data import decode_temperature, weather_station
from gridcast.validation import ValidationError

CONFIG = {
    "accepted_quality_codes": ["0", "1", "4", "5"],
    "accepted_report_types": ["FM-15", "FM-16"],
}


def sample():
    return pd.DataFrame(
        {
            "STATION": ["72509014739"] * 4,
            "NAME": ["BOSTON"] * 4,
            "DATE": [
                "2024-01-01T00:10",
                "2024-01-01T00:10",
                "2024-01-01T00:50",
                "2024-01-01T01:05",
            ],
            "TMP": ["+0100,1", "+0120,1", "+0140,1", "+0500,1"],
            "DEW": ["+0000,1"] * 4,
            "REPORT_TYPE": ["FM-15"] * 4,
        }
    )


def test_noaa_quality_codes_sentinel_and_tenths_celsius():
    vals, rejected = decode_temperature(
        pd.Series(
            [
                "+0105,1",
                "-0010,5",
                "+9999,1",
                "+0300,3",
                "+0200,2",
                "+0200,9",
            ]
        ),
        CONFIG["accepted_quality_codes"],
    )
    assert vals.iloc[:2].tolist() == [10.5, -1.0]
    assert vals.iloc[2:].isna().all() and rejected.sum() == 4
    with pytest.raises(ValidationError):
        decode_temperature(pd.Series(["+100"]), ["1"])


def test_hourly_weather_uses_own_interval_and_logs_duplicates():
    hours, qc, audit = weather_station(sample(), CONFIG)
    assert hours.temperature_c.iloc[0] == 12.5  # median(10,12) then mean(11,14)
    assert hours.temperature_c.iloc[1] == 50
    assert audit["simultaneous_accepted_reports_collapsed"] == 1
    assert len(qc) == 4  # Original rows and tokens retained.


def test_future_weather_cannot_change_earlier_hour():
    raw = sample()
    before, _, _ = weather_station(raw, CONFIG)
    raw.loc[3, "TMP"] = "-0300,1"
    after, _, _ = weather_station(raw, CONFIG)
    pd.testing.assert_frame_equal(before.iloc[:1], after.iloc[:1])


def test_report_filter_and_inconsistent_dewpoint_are_auditable():
    raw = sample()
    raw.loc[0, "REPORT_TYPE"] = "SOD"
    raw.loc[2, "DEW"] = "+0200,1"
    _, qc, report = weather_station(raw, CONFIG)
    assert report["excluded_type_rows"] == 1
    assert report["dewpoint_inconsistent_rows"] == 1
    assert qc.loc[2, "DEW"] == "+0200,1" and pd.isna(qc.loc[2, "dewpoint_c"])
