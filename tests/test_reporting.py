import json
import shutil
from pathlib import Path

import pandas as pd
import pytest

from gridcast.data import sha256
from gridcast.reporting import run

ROOT = Path(__file__).resolve().parents[1]


def test_report_from_saved_results_without_raw_data(tmp_path, monkeypatch):
    for folder in ["configs", "reports/predictions"]:
        shutil.copytree(ROOT / folder, tmp_path / folder)
    for name in ["frozen_thresholds.json", "model_selection.json", "statistical_fits.json"]:
        shutil.copy2(ROOT / "reports" / name, tmp_path / "reports" / name)
    (tmp_path / "README.md").write_text("Study\n<!-- BEGIN RESULTS -->\n<!-- END RESULTS -->\nEnd\n")

    def forbid_network(*args, **kwargs):
        raise AssertionError("Reporting must not fetch data")

    monkeypatch.setattr("urllib.request.urlopen", forbid_network)
    summary = run(tmp_path)
    predictions = pd.read_parquet(tmp_path / "reports/predictions/reserved.parquet")
    selected = predictions.query("partition == 'test' and model == 'lightgbm'")
    expected_mae = (selected.actual - selected.prediction).abs().mean()
    metrics = pd.DataFrame(summary["test_metrics"]).set_index("model")
    assert metrics.loc["lightgbm", "MAE"] == pytest.approx(expected_mae)
    assert set(metrics.index) == {"daily", "lightgbm"}
    assert metrics.n.eq(8784).all()
    monthly = pd.read_csv(tmp_path / "reports/tables/anomaly_summary.csv")
    assert monthly.hours.sum() == len(selected)
    assert monthly.flagged_hours.sum() == summary["anomaly_hours"]
    intervals = pd.read_parquet(tmp_path / "reports/predictions/test_intervals.parquet").query("level == 0.95")
    assert summary["anomaly_hours"] == (~intervals.actual.between(intervals.lower, intervals.upper)).sum()
    for name, digest in summary["source_sha256"].items():
        assert sha256(tmp_path / name) == digest
    readme = (tmp_path / "README.md").read_text()
    assert readme.startswith("Study\n") and readme.endswith("End\n")
    assert f"{expected_mae:,.1f} MW" in readme
    assert not (tmp_path / "data").exists()
    files = [p for p in (tmp_path / "reports").rglob("*") if p.is_file()]
    before = {p: sha256(p) for p in files}
    run(tmp_path)
    assert before == {p: sha256(p) for p in files}
    assert readme == (tmp_path / "README.md").read_text()
    assert json.loads((tmp_path / "reports/results/summary.json").read_text()) == summary
