"""Rebuild the complete analysis offline in isolation and compare artifact hashes."""

import argparse
import importlib.metadata
import platform
import shutil
import tempfile
from pathlib import Path
from unittest.mock import patch

from gridcast.backtesting import run as backtest_run
from gridcast.data import run as data_run
from gridcast.data import sha256, write_json
from gridcast.eda import run as eda_run
from gridcast.reliability import run as reliability_run


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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    run(parser.parse_args().root.resolve())


if __name__ == "__main__":
    main()
