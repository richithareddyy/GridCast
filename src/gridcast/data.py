"""Fetch immutable public snapshots, validate load, and align observed weather."""

import argparse
import hashlib
import json
import logging
import os
import shutil
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from gridcast.features import calendar_features, partition
from gridcast.validation import (
    ValidationError,
    align_weather,
    normalize_hourly,
    require_columns,
    utc_timestamps,
    validate_periods,
)

LOG = logging.getLogger(__name__)


def load_config(root: Path, name: str) -> dict:
    with (root / "configs" / name).open() as f:
        return yaml.safe_load(f)


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def update_readme(root: Path, section: str, text: str) -> None:
    path = root / "README.md"
    contents = path.read_text()
    begin, end = f"<!-- BEGIN {section} -->", f"<!-- END {section} -->"
    if begin in contents:
        before, rest = contents.split(begin, 1)
        _, after = rest.split(end, 1)
        path.write_text(before + begin + "\n" + text + end + after)


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def fetch_snapshot(url: str, destination: Path, expected: dict | None = None) -> dict:
    """Never overwrite raw files; reject both local and remote checksum drift."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if expected is not None and expected["url"] != url:
        raise ValidationError(f"Source URL changed for {destination.name}")
    if destination.exists():
        if expected is None:
            raise ValidationError(f"Unregistered raw file: {destination.name}")
        if sha256(destination) != expected["sha256"]:
            raise ValidationError(f"Raw checksum mismatch: {destination.name}")
        return expected
    temporary = destination.with_suffix(destination.suffix + ".part")
    for attempt in range(3):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "GridCast/0.1 research"})
            with (
                urllib.request.urlopen(request, timeout=90) as response,
                temporary.open("wb") as out,
            ):
                headers = dict(response.headers)
                shutil.copyfileobj(response, out)
            digest = sha256(temporary)
            if expected is not None and digest != expected["sha256"]:
                raise ValidationError(
                    f"Remote snapshot changed: {destination.name}. Expected {expected['sha256']}; "
                    f"received {digest}. Restore the original snapshot "
                    "or explicitly version the study."
                )
            record = expected or {
                "url": url,
                "sha256": digest,
                "bytes": temporary.stat().st_size,
                "retrieved_at_utc": datetime.now(UTC).isoformat(),
                "last_modified": headers.get("Last-Modified"),
                "etag": headers.get("ETag"),
            }
            # Hard link fails if another writer created the destination: never replace raw bytes.
            os.link(temporary, destination)
            temporary.unlink()
            destination.chmod(0o444)
            LOG.info("Downloaded %s (%d bytes)", destination.name, record["bytes"])
            return record
        except ValidationError:
            temporary.unlink(missing_ok=True)
            raise
        except (OSError, TimeoutError):
            temporary.unlink(missing_ok=True)
            if attempt == 2:
                raise
            time.sleep(1 + attempt)
    raise RuntimeError("Unreachable download state")


def download(root: Path, cfg: dict) -> dict:
    lock_path = root / "configs/sources.lock.json"
    manifest = json.loads(lock_path.read_text()) if lock_path.exists() else {}
    sources = {cfg["raw_filename"]: cfg["source_url"]}
    weather = cfg["weather"]
    for station in weather["stations"]:
        for year in weather["years"]:
            name = f"noaa_{station['id']}_{year}.csv"
            sources[name] = weather["source_url_template"].format(year=year, station=station["id"])
    sources["iso_smd_template.xlsx"] = (
        "https://www.iso-ne.com/static-assets/documents/2016/08/smd_hourly_template_v0.xlsx"
    )

    def fetch_source(item):
        name, url = item
        return name, fetch_snapshot(url, root / "data/raw" / name, manifest.get(name))

    # Persist completed sources so failures are resumable without adopting unknown bytes.
    errors = []
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = [pool.submit(fetch_source, item) for item in sources.items()]
        for future in as_completed(futures):
            try:
                name, record = future.result()
                manifest[name] = record
                write_json(lock_path, manifest)
            except Exception as exc:
                errors.append(exc)
    if errors:
        raise errors[0]
    return manifest


def read_eia(root: Path, cfg: dict):
    path = root / "data/raw" / cfg["raw_filename"]
    required = [
        "BA",
        "UTC time",
        "Local time",
        "Time zone",
        "Demand",
        "Adjusted demand",
        "Imputed demand",
    ]
    notes = pd.read_excel(path, sheet_name="Notes", header=None)
    notes_map = {str(row.iloc[0]): str(row.iloc[1]) for _, row in notes.dropna(subset=[0]).iterrows()}
    if "MWh" not in notes_map.get("Demand", ""):
        raise ValidationError("Source workbook no longer documents Demand in MWh")
    if "end of the hour" not in notes_map.get("UTC time", ""):
        raise ValidationError("Source timestamp convention changed")
    source = pd.read_excel(path, sheet_name=cfg["sheet"], usecols=required)
    require_columns(source, required)
    if not source["BA"].eq(cfg["respondent"]).all():
        raise ValidationError("Unexpected balancing authority in workbook")
    if not source["Time zone"].eq("Eastern").all():
        raise ValidationError("Unexpected source calendar timezone")
    end = utc_timestamps(source["UTC time"], naive_is_utc=True)
    source.index = end - pd.Timedelta(hours=1)
    source.index.name = "timestamp_utc"
    source["source_excel_row"] = np.arange(len(source)) + 2
    full_source_rows = len(source)
    source = source.loc[
        (source.index >= pd.Timestamp(cfg["start"])) & (source.index < pd.Timestamp(cfg["end_exclusive"]))
    ].copy()
    observed_local_end = pd.DatetimeIndex(pd.to_datetime(source["Local time"]))
    calculated_local_end = (source.index + pd.Timedelta(hours=1)).tz_convert(cfg["calendar_timezone"]).tz_localize(None)
    if not observed_local_end.equals(calculated_local_end):
        raise ValidationError("UTC-to-local hour-end mapping disagrees with EIA local metadata")
    renamed = source.rename(
        columns={
            "Demand": "demand_mwh_reported",
            "Adjusted demand": "eia_adjusted_mwh",
            "Imputed demand": "eia_imputed_mwh",
        }
    )[["demand_mwh_reported", "eia_adjusted_mwh", "eia_imputed_mwh", "source_excel_row"]]
    normalized, audit = normalize_hourly(renamed, cfg["start"], cfg["end_exclusive"])
    audit.update(
        {
            "workbook_hourly_rows": full_source_rows,
            "out_of_study_rows_not_selected": full_source_rows - len(source),
            "eia_imputed_hours": int(normalized.eia_imputed_mwh.notna().sum()),
            "eia_adjusted_differs_from_reported": int(
                (~np.isclose(normalized.eia_adjusted_mwh, normalized.demand_mwh_reported, equal_nan=True)).sum()
            ),
            "source_units": "MWh",
            "target_units": "MW",
            "interval_hours": 1,
            "utc_local_end_mismatches": 0,
        }
    )
    write_json(
        root / "reports/source_definitions.json",
        {k: notes_map[k] for k in ["UTC time", "Local time", "Demand", "Imputed demand", "Adjusted demand"]},
    )
    normalized.to_parquet(root / "data/interim/demand.parquet")
    return normalized, audit


def decode_temperature(values: pd.Series, accepted: list[str]):
    split = values.astype("string").str.split(",", expand=True)
    if split.shape[1] != 2:
        raise ValidationError("NOAA temperature field must contain value,quality")
    numeric = pd.to_numeric(split[0], errors="raise")
    valid = split[1].isin(accepted) & numeric.ne(9999) & numeric.between(-932, 618)
    return (numeric / 10).where(valid).astype(float), ~valid.fillna(False)


def weather_station(frame: pd.DataFrame, cfg: dict):
    """Retain QC flags, collapse simultaneous reports, then average inside [hour,hour+1)."""
    require_columns(frame, ["DATE", "STATION", "NAME", "REPORT_TYPE", "TMP", "DEW"])
    out = frame.copy()
    out["observation_utc"] = utc_timestamps(out["DATE"], naive_is_utc=True)
    out["temperature_c"], out["temperature_rejected"] = decode_temperature(out["TMP"], cfg["accepted_quality_codes"])
    out["dewpoint_c"], out["dewpoint_rejected"] = decode_temperature(out["DEW"], cfg["accepted_quality_codes"])
    out["report_type_accepted"] = out.REPORT_TYPE.isin(cfg["accepted_report_types"])
    # Dew point above temperature beyond rounding tolerance is inconsistent; keep original tokens.
    inconsistent = out.dewpoint_c.gt(out.temperature_c + 0.5)
    out["dewpoint_inconsistent"] = inconsistent
    out.loc[inconsistent, "dewpoint_c"] = np.nan
    selected = out.loc[out.report_type_accepted]
    instantaneous = selected.groupby("observation_utc")[["temperature_c", "dewpoint_c"]].median()
    hours = instantaneous.groupby(instantaneous.index.floor("h")).mean()
    hours.index.name = "timestamp_utc"
    report = {
        "raw_rows": len(out),
        "accepted_type_rows": len(selected),
        "excluded_type_rows": int((~out.report_type_accepted).sum()),
        "accepted_types": cfg["accepted_report_types"],
        "simultaneous_accepted_reports_collapsed": int(selected.observation_utc.duplicated().sum()),
        "temperature_rejected_rows": int(out.temperature_rejected.sum()),
        "dewpoint_rejected_rows": int(out.dewpoint_rejected.sum()),
        "dewpoint_inconsistent_rows": int(inconsistent.sum()),
        "report_types": out.REPORT_TYPE.value_counts().to_dict(),
    }
    return hours, out, report


def read_weather(root: Path, cfg: dict, expected: pd.DatetimeIndex):
    weather_cfg = cfg["weather"]
    all_hours = pd.DataFrame(index=expected)
    reports = {}
    for station in weather_cfg["stations"]:
        chunks = []
        for year in weather_cfg["years"]:
            name = f"noaa_{station['id']}_{year}.csv"
            chunk = pd.read_csv(
                root / "data/raw" / name,
                dtype="string",
                usecols=[
                    "STATION",
                    "DATE",
                    "NAME",
                    "LATITUDE",
                    "LONGITUDE",
                    "REPORT_TYPE",
                    "TMP",
                    "DEW",
                ],
            )
            chunk["source_file"] = name
            chunk["source_csv_row"] = np.arange(len(chunk)) + 2
            if not chunk.STATION.eq(station["id"]).fillna(False).all():
                raise ValidationError(f"Station ID mismatch for {name}")
            name_ok = pd.Series(False, index=chunk.index)
            for alias in station["name_contains"]:
                name_ok |= chunk.NAME.str.contains(alias, regex=False)
            if not name_ok.fillna(False).all():
                raise ValidationError(f"Station name mismatch for {name}")
            for column in ["LATITUDE", "LONGITUDE"]:
                if not np.isclose(pd.to_numeric(chunk[column]), station[column.lower()], atol=0.02, rtol=0).all():
                    raise ValidationError(f"Station coordinates changed for {name}")
            chunks.append(chunk)
        source = pd.concat(chunks, ignore_index=True)
        hours, qc, audit = weather_station(source, weather_cfg)
        label = station["label"]
        qc.to_parquet(root / f"data/interim/weather_{label}_qc.parquet", index=False)
        hours = hours.reindex(expected)
        for col in ["temperature_c", "dewpoint_c"]:
            all_hours[f"{label}_{col}"] = hours[col]
        audit["station_id"] = station["id"]
        audit["station_names"] = sorted(source.NAME.unique().tolist())
        audit["latitude"] = float(source.LATITUDE.iloc[0])
        audit["longitude"] = float(source.LONGITUDE.iloc[0])
        audit["missing_temperature_hours"] = int(hours.temperature_c.isna().sum())
        audit["missing_dewpoint_hours"] = int(hours.dewpoint_c.isna().sum())
        reports[label] = audit
    for variable in ["temperature_c", "dewpoint_c"]:
        columns = [f"{s['label']}_{variable}" for s in weather_cfg["stations"]]
        all_hours[f"weather_{variable}"] = all_hours[columns].mean(axis=1, skipna=False)
        all_hours[f"{variable}_station_count"] = all_hours[columns].notna().sum(axis=1)
    all_hours["weather_missing"] = all_hours.weather_temperature_c.isna()
    all_hours["weather_available_at"] = all_hours.index + pd.Timedelta(hours=2)
    # End-of-hour plus one-hour assumed reporting delay, not an archived publication timestamp.
    all_hours["heating_degrees_c"] = (18 - all_hours.weather_temperature_c).clip(lower=0)
    all_hours["cooling_degrees_c"] = (all_hours.weather_temperature_c - 18).clip(lower=0)
    reports["alignment"] = {
        "expected_hours": len(expected),
        "joined_hours": len(all_hours),
        "missing_regional_temperature_hours": int(all_hours.weather_missing.sum()),
        "missing_regional_dewpoint_hours": int(all_hours.weather_dewpoint_c.isna().sum()),
        "station_weighting": "equal 1/3; all three required; proxy, not population/load weighted",
        "aggregation": "UTC [hour, hour+1); no time interpolation; simultaneous reports median",
        "availability": "assumed hour end + 1h; historical publication time unknown",
    }
    return all_hours, reports


def run(root: Path):
    for folder in ["data/raw", "data/interim", "data/processed", "reports"]:
        (root / folder).mkdir(parents=True, exist_ok=True)
    cfg = load_config(root, "data.yaml")
    evaluation = load_config(root, "evaluation.yaml")
    validate_periods(evaluation["periods"], cfg["start"], cfg["end_exclusive"])
    manifest = download(root, cfg)
    LOG.info("Normalizing EIA hour-ending demand")
    demand, demand_audit = read_eia(root, cfg)
    LOG.info("Aligning quality-checked NOAA observations")
    weather, weather_audit = read_weather(root, cfg, demand.index)
    merged = align_weather(demand, weather)
    merged = merged.join(calendar_features(merged.index, cfg["calendar_timezone"]))
    merged["demand_available_at"] = merged.index + pd.Timedelta(hours=2)
    split_report = {}
    for name, bounds in evaluation["periods"].items():
        part = partition(merged, bounds)
        part.to_parquet(root / f"data/processed/{name}.parquet")
        split_report[name] = {"rows": len(part), **bounds}
    report = {
        "demand": demand_audit,
        "weather": weather_audit,
        "partitions": split_report,
        "processed_columns": list(merged.columns),
        "processed_column_count": len(merged.columns),
        "data_config_sha256": sha256(root / "configs/data.yaml"),
        "evaluation_config_sha256": sha256(root / "configs/evaluation.yaml"),
        "source_hashes": {name: record["sha256"] for name, record in manifest.items()},
    }
    write_json(root / "reports/data_quality.json", report)
    log = {
        "run_at_utc": datetime.now(UTC).isoformat(),
        "stage": "data",
        "status": "ok",
        "transformations": [
            "selected ISNE reported Demand and UTC study window; kept source row lineage",
            "converted EIA hour-ending UTC to interval-start UTC; checked local metadata",
            "reindexed complete hourly grid, flagged missing/invalid values, no target imputation",
            "MWh / one hour -> average MW; source reported, adjusted and imputed columns retained",
            "decoded NOAA 0.1 C fields; QC flags and excluded report types preserved in interim",
            "median simultaneous reports; mean within UTC hour; all-three-station regional mean",
            "derived local calendar and federal holidays; retained leap days and both DST hours",
            "wrote isolated chronological partitions; test used for integrity checks only",
        ],
        "source_hashes": report["source_hashes"],
    }
    with (root / "reports/run_log.jsonl").open("a") as handle:
        handle.write(json.dumps(log) + "\n")
    write_json(root / "reports/transformations.json", {"transformations": log["transformations"]})
    (root / "reports/validation_failure.json").unlink(missing_ok=True)
    LOG.info("Validated %d hourly intervals; partitioned without imputation", len(merged))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    try:
        run(args.root.resolve())
    except Exception as exc:
        write_json(
            args.root / "reports/validation_failure.json",
            {
                "error_type": type(exc).__name__,
                "message": str(exc),
                "time_utc": datetime.now(UTC).isoformat(),
            },
        )
        raise


if __name__ == "__main__":
    main()
