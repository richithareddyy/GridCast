import json
from pathlib import Path

import pytest

from gridcast.data import fetch_snapshot, sha256
from gridcast.validation import ValidationError


def test_cached_raw_is_verified_without_network(tmp_path, monkeypatch):
    path = tmp_path / "raw.csv"
    path.write_text("immutable")
    record = {"url": "https://example.invalid/data", "sha256": sha256(path)}
    monkeypatch.setattr("urllib.request.urlopen", lambda *a, **kw: pytest.fail("Unexpected network"))
    assert fetch_snapshot(record["url"], path, record) == record
    path.write_text("changed")
    with pytest.raises(ValidationError, match="checksum mismatch"):
        fetch_snapshot(record["url"], path, record)


def test_unknown_raw_file_is_never_adopted_or_overwritten(tmp_path):
    path = tmp_path / "raw.csv"
    path.write_text("unknown")
    with pytest.raises(ValidationError, match="Unregistered"):
        fetch_snapshot("https://example.invalid", path)
    assert path.read_text() == "unknown"


def test_remote_checksum_mismatch_leaves_no_raw_file(tmp_path):
    source = tmp_path / "source"
    source.write_text("new bytes")
    destination = tmp_path / "raw.csv"
    with pytest.raises(ValidationError, match="Remote snapshot changed"):
        fetch_snapshot(source.as_uri(), destination, {"url": source.as_uri(), "sha256": "0" * 64})
    assert not destination.exists() and not destination.with_suffix(".csv.part").exists()


def test_download_is_registered_readonly_and_reusable(tmp_path):
    source = tmp_path / "source"
    source.write_text("public snapshot")
    destination = tmp_path / "raw.csv"
    record = fetch_snapshot(source.as_uri(), destination)
    assert record["sha256"] == sha256(source)
    assert destination.stat().st_mode & 0o222 == 0
    assert fetch_snapshot(source.as_uri(), destination, record) == record


def test_notebook_is_valid_json_and_only_reads_training():
    path = Path(__file__).parents[1] / "notebooks/01_eda.ipynb"
    notebook = json.loads(path.read_text())
    assert notebook["nbformat"] == 4
    code = "\n".join("".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code")
    assert "test.parquet" not in code and "train.parquet" in code
