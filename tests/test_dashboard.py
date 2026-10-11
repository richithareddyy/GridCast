"""Exercise every research view and the choices that change evaluation scope."""

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "app/streamlit_app.py"


@pytest.mark.parametrize(
    "section",
    [
        "Overview",
        "Demand Explorer",
        "Forecast",
        "Model Comparison",
        "Extreme Events",
        "Anomalies",
        "Reliability",
        "Methodology",
    ],
)
def test_sections_render(section):
    app = AppTest.from_file(str(APP)).run()
    app.sidebar.radio[0].set_value(section).run()
    assert not app.exception
    assert not app.error


def test_forecast_scope_and_interactions():
    app = AppTest.from_file(str(APP)).run()
    app.sidebar.radio[0].set_value("Forecast").run()
    assert set(app.selectbox[1].options) == {"Daily seasonal", "LightGBM"}
    app.selectbox[1].set_value("lightgbm").run()
    app.select_slider[0].set_value(0.8).run()
    app.selectbox[2].set_value(app.selectbox[2].value).run()
    assert not app.exception
    app.selectbox[0].set_value("2023 validation").run()
    assert len(app.selectbox[1].options) == 6
    assert len(app.select_slider) == 0
    for model in ["naive", "daily", "weekly", "dhr_ar1", "lightgbm", "lstm"]:
        app.selectbox[1].set_value(model).run()
        assert not app.exception


def test_comparison_and_explorer_filters():
    app = AppTest.from_file(str(APP)).run()
    app.sidebar.radio[0].set_value("Model Comparison").run()
    app.selectbox[0].set_value("2024 final test").run()
    app.selectbox[1].set_value("MASE").run()
    app.selectbox[2].set_value("extreme_demand").run()
    assert len(app.dataframe[0].value) == 2
    assert not app.exception
    app.sidebar.radio[0].set_value("Demand Explorer").run()
    for partition in ["train", "validation", "calibration", "test"]:
        app.selectbox[0].set_value(partition).run()
        assert not app.exception


def test_history_snapshot_integrity():
    path = ROOT / "reports/app/history.parquet"
    manifest = json.loads((path.parent / "manifest.json").read_text())
    assert hashlib.sha256(path.read_bytes()).hexdigest() == manifest["history_sha256"]
    frame = pd.read_parquet(path)
    assert len(frame) == manifest["rows"] == 43848
    assert frame.index.is_unique
    assert str(frame.index.tz) == "UTC"
    assert frame.groupby("partition").size().to_dict() == {
        "train": 26304,
        "validation": 6552,
        "calibration": 2208,
        "test": 8784,
    }
    saved = pd.read_parquet(ROOT / "reports/predictions/reserved.parquet")
    actual = saved.loc[saved.model.eq("lightgbm")].set_index("target_start_utc").actual
    pd.testing.assert_series_equal(frame.loc[actual.index, "demand_mw"], actual, check_names=False)
