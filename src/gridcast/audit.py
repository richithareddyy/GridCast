"""Rebuild the complete analysis offline in isolation and compare artifact hashes."""

import argparse
import importlib.metadata
import json
import platform
import re
import shutil
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from gridcast.backtesting import run as backtest_run
from gridcast.data import load_config, sha256, write_json
from gridcast.data import run as data_run
from gridcast.eda import run as eda_run
from gridcast.evaluation import score_groups
from gridcast.reliability import run as reliability_run
from gridcast.reporting import run as reporting_run
from gridcast.validation import ValidationError


def run(root: Path):
    files = [
        "reports/data_quality.json",
        "reports/source_definitions.json",
        "reports/transformations.json",
        "reports/eda_summary.json",
        "reports/frozen_thresholds.json",
        "reports/eda_findings.md",
        "reports/backtest_summary.json",
        "reports/model_selection.json",
        "reports/reliability_summary.json",
        "reports/reserved_fits.json",
        "reports/reliability_findings.md",
        "reports/statistical_fits.json",
        "reports/baseline_findings.md",
    ]
    for folder, pattern in [
        ("data/processed", "*.parquet"),
        ("data/interim", "*.parquet"),
        ("reports/figures", "*.png"),
        ("reports/tables", "*.csv"),
        ("reports/predictions", "*.parquet"),
        ("reports/models", "*.txt"),
        ("reports/models", "*.npz"),
        ("reports/reserved_models", "*.txt"),
        ("reports/results", "*.json"),
        ("reports/results", "*.md"),
    ]:
        files.extend(str(p.relative_to(root)) for p in sorted((root / folder).glob(pattern)))
    if not files or not (root / "reports/eda_summary.json").exists():
        raise RuntimeError("Run make analysis before make reproduce")
    (root / "work").mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="reproduce-", dir=root / "work") as tmp:
        clean = Path(tmp)
        shutil.copytree(root / "configs", clean / "configs")
        shutil.copy2(root / "README.md", clean / "README.md")
        (clean / "data/raw").mkdir(parents=True)
        for raw in (root / "data/raw").iterdir():
            if not raw.name.startswith("."):
                (clean / "data/raw" / raw.name).hardlink_to(raw)
        with patch("urllib.request.urlopen", side_effect=RuntimeError("Network forbidden in replay")):
            data_run(clean)
            eda_run(clean)
            backtest_run(clean)
            reliability_run(clean)
            reporting_run(clean)
        checks = {
            name: {
                "sha256": sha256(root / name),
                "match": sha256(root / name) == sha256(clean / name),
            }
            for name in files
        }
    report = {
        "all_match": all(item["match"] for item in checks.values()),
        "artifact_count": len(checks),
        "artifacts": checks,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "dependencies": {
            name: importlib.metadata.version(name)
            for name in [
                "lightgbm",
                "torch",
                "numpy",
                "pandas",
                "pyarrow",
                "matplotlib",
                "statsmodels",
                "holidays",
                "openpyxl",
                "PyYAML",
            ]
        },
        "method": "offline rebuild from checksum-verified raw snapshots in an isolated directory",
    }
    write_json(root / "reports/reproducibility.json", report)
    if not report["all_match"]:
        raise RuntimeError(f"Nondeterministic outputs: {[k for k, v in checks.items() if not v['match']]}")
    print(f"Reproducibility passed: {len(checks)} artifacts match byte-for-byte, network disabled.")


def verify_saved(root: Path):
    """Check committed evidence without raw data, downloads or model fitting."""
    evaluation = load_config(root, "evaluation.yaml")
    selection = json.loads((root / "reports/model_selection.json").read_text())
    validation = pd.read_parquet(root / "reports/predictions/validation.parquet")
    reserved = pd.read_parquet(root / "reports/predictions/reserved.parquet")
    intervals = pd.read_parquet(root / "reports/predictions/test_intervals.parquet")
    fits = json.loads((root / "reports/statistical_fits.json").read_text())
    reserved_fits = json.loads((root / "reports/reserved_fits.json").read_text())
    checks = {}

    def check(name, passed):
        if not passed:
            raise ValidationError(f"Saved-result audit failed: {name}")
        checks[name] = True

    horizon = evaluation["forecast"]["horizon_hours"]
    delay = pd.Timedelta(hours=evaluation["forecast"]["assumed_reporting_delay_hours"])
    for name, frame in [("validation", validation), ("reserved", reserved)]:
        check(f"{name}_unique_forecast_keys", not frame.duplicated(["origin_utc", "horizon", "model"]).any())
        check(
            f"{name}_target_alignment",
            frame.target_start_utc.eq(frame.origin_utc + pd.to_timedelta(frame.horizon - 1, unit="h")).all()
            and frame.target_end_utc.eq(frame.target_start_utc + pd.Timedelta(hours=1)).all(),
        )
        check(
            f"{name}_complete_paths",
            frame.groupby(["origin_utc", "model"])
            .horizon.apply(lambda values: set(values) == set(range(1, horizon + 1)))
            .all(),
        )
        check(f"{name}_shared_actuals", frame.groupby("target_start_utc").actual.nunique().eq(1).all())
    for partition in ["validation", "calibration", "test"]:
        frame = validation if partition == "validation" else reserved.loc[reserved.partition.eq(partition)]
        bounds = evaluation["periods"][partition]
        expected = pd.date_range(bounds["start"], bounds["end"], inclusive="left", freq="h")
        models = set(evaluation["baselines"]) & {"naive", "daily", "weekly"}
        models |= {"dhr_ar1", "lightgbm", "lstm"}
        if partition != "validation":
            models = {selection["selected_model"], "daily"}
        check(f"{partition}_models", set(frame.model) == models)
        check(
            f"{partition}_target_coverage",
            all(
                pd.DatetimeIndex(part.target_start_utc.sort_values()).equals(expected)
                for _, part in frame.groupby("model")
            ),
        )
    origins = pd.read_csv(root / "reports/tables/origin_audit.csv")
    issued = pd.to_datetime(origins.origin_utc, utc=True)
    available = pd.to_datetime(origins.latest_history_available_at, utc=True)
    history_start = pd.to_datetime(origins.latest_history_start_utc, utc=True)
    check(
        "origin_history_matured",
        (available <= issued).all() and available.eq(history_start + pd.Timedelta(hours=1) + delay).all(),
    )
    check(
        "validation_fits_matured",
        all(
            pd.Timestamp(fit["latest_label_available_at"]) <= pd.Timestamp(fit["fit_origin_utc"])
            and pd.Timestamp(fit["learning"]["latest_training_label_available_at"])
            <= pd.Timestamp(fit["fit_origin_utc"])
            and pd.Timestamp(fit["learning"]["inner_last_label_available_at"])
            <= pd.Timestamp(fit["learning"]["inner_stopping_start"])
            for fit in fits
        ),
    )
    check(
        "reserved_fits_matured",
        all(
            pd.Timestamp(fit["latest_training_label_available_at"]) <= pd.Timestamp(fit["fit_origin_utc"])
            and fit["boost_rounds"] == selection["frozen_boost_rounds"]
            for fit in reserved_fits
        ),
    )
    check(
        "future_weather_excluded",
        all(
            not any("weather" in column or "temperature" in column for column in fit["learning"]["feature_columns"])
            for fit in fits
        ),
    )
    check("interval_residuals_matured", (intervals.latest_residual_available_at <= intervals.origin_utc).all())
    check("interval_levels", set(intervals.level) == set(evaluation["uncertainty"]["levels"]))
    check(
        "interval_keys",
        not intervals.duplicated(["origin_utc", "horizon", "level"]).any()
        and len(intervals)
        == len(reserved.query("partition == 'test' and model == 'lightgbm'"))
        * len(evaluation["uncertainty"]["levels"]),
    )
    check(
        "interval_rank",
        intervals.quantile_rank.eq(np.ceil((intervals.calibration_n + 1) * intervals.level)).all()
        and intervals.quantile_rank.le(intervals.calibration_n).all(),
    )
    scores = score_groups(validation, ["model"])
    check(
        "validation_selection",
        scores.loc[scores.MAE.idxmin(), "model"] == selection["selected_model"]
        and not selection["reserved_outcomes_used_for_selection"]
        and selection["frozen_boost_rounds"] == int(np.median([fit["learning"]["tree_rounds"] for fit in fits])),
    )
    check(
        "selection_provenance",
        selection["evaluation_config_sha256"] == sha256(root / "configs/evaluation.yaml")
        and selection["validation_predictions_sha256"] == sha256(root / "reports/predictions/validation.parquet"),
    )
    for name, frame in [
        ("model_comparison", validation),
        ("test_model_comparison", reserved.loc[reserved.partition.eq("test")]),
    ]:
        saved = pd.read_csv(root / f"reports/tables/{name}.csv").set_index("model").sort_index()
        computed = score_groups(frame, ["model"]).set_index("model").sort_index()
        check(
            f"{name}_metrics",
            saved.index.equals(computed.index) and np.allclose(saved[computed.columns], computed, rtol=1e-9, atol=1e-8),
        )
    summary = json.loads((root / "reports/results/summary.json").read_text())
    check("report_provenance", all(sha256(root / name) == digest for name, digest in summary["source_sha256"].items()))
    readme = (root / "README.md").read_text()
    findings = (root / "reports/results/findings.md").read_text()
    block = readme.split("<!-- BEGIN RESULTS -->", 1)[1].split("<!-- END RESULTS -->", 1)[0]
    check("readme_findings", block.strip() == findings.strip())
    links = re.findall(r"\]\(([^)]+)\)", readme)
    check(
        "readme_local_links",
        all(
            (root / link.split("#")[0]).exists() for link in links if not link.startswith(("http://", "https://", "#"))
        ),
    )
    replay = json.loads((root / "reports/reproducibility.json").read_text())
    check("recorded_replay_passed", replay["all_match"] and all(item["match"] for item in replay["artifacts"].values()))
    committed = {name: item for name, item in replay["artifacts"].items() if name.startswith("reports/")}
    check(
        "replayed_report_hashes",
        all((root / name).is_file() and sha256(root / name) == item["sha256"] for name, item in committed.items()),
    )
    report = {
        "checks": checks,
        "checks_passed": len(checks),
        "report_artifacts_verified": len(committed),
        "recorded_replay_artifacts": replay["artifact_count"],
        "scope": "saved-result integrity and temporal boundaries; full offline replay is recorded separately",
    }
    write_json(root / "reports/final_audit.json", report)
    print(f"Saved-result audit passed: {len(checks)} checks; {len(committed)} report artifact hashes verified.")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--saved-only", action="store_true", help="Verify committed results without refitting")
    args = parser.parse_args()
    if args.saved_only:
        verify_saved(args.root.resolve())
    else:
        run(args.root.resolve())


if __name__ == "__main__":
    main()
