import shutil
from pathlib import Path

import pandas as pd
import pytest

from gridcast.audit import verify_saved
from gridcast.validation import ValidationError

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def saved_study(tmp_path):
    for folder in ["configs", "reports", "notebooks"]:
        shutil.copytree(ROOT / folder, tmp_path / folder)
    shutil.copy2(ROOT / "README.md", tmp_path / "README.md")
    return tmp_path


def test_saved_audit_needs_no_raw_data(saved_study):
    report = verify_saved(saved_study)
    assert report["checks_passed"] == len(report["checks"])
    assert all(report["checks"].values())
    assert not (saved_study / "data").exists()


@pytest.mark.parametrize(
    "mutation, expected",
    [
        ("future_residual", "interval_residuals_matured"),
        ("missing_forecast", "validation_complete_paths"),
        ("changed_metric", "model_comparison_metrics"),
        ("stale_readme", "readme_findings"),
        ("missing_figure", "replayed_report_hashes"),
    ],
)
def test_saved_audit_rejects_corrupt_evidence(saved_study, mutation, expected):
    if mutation == "future_residual":
        path = saved_study / "reports/predictions/test_intervals.parquet"
        frame = pd.read_parquet(path)
        frame.loc[0, "latest_residual_available_at"] = frame.loc[0, "origin_utc"] + pd.Timedelta(hours=1)
        frame.to_parquet(path, index=False)
    elif mutation == "missing_forecast":
        path = saved_study / "reports/predictions/validation.parquet"
        pd.read_parquet(path).iloc[1:].to_parquet(path, index=False)
    elif mutation == "changed_metric":
        path = saved_study / "reports/tables/model_comparison.csv"
        frame = pd.read_csv(path)
        frame.loc[0, "MAE"] += 100
        frame.to_csv(path, index=False)
    elif mutation == "stale_readme":
        path = saved_study / "README.md"
        path.write_text(path.read_text().replace("<!-- BEGIN RESULTS -->", "<!-- BEGIN RESULTS -->\nStale findings."))
    else:
        (saved_study / "reports/figures/03_month_hour_heatmap.png").unlink()
    with pytest.raises(ValidationError, match=expected):
        verify_saved(saved_study)
