"""Small, deterministic dashboard data export; no model fitting at application runtime."""

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

COLUMNS = ["demand_mw", "weather_temperature_c", "hour", "weekday", "month", "holiday"]


def export_history(root: Path) -> None:
    """Export retrospective observations with explicit evaluation partitions."""
    frames, sources = [], {}
    for partition in ("train", "validation", "calibration", "test"):
        path = root / "data/processed" / f"{partition}.parquet"
        sources[str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
        frame = pd.read_parquet(path)[COLUMNS].copy()
        frame["partition"] = partition
        frame.index.name = "target_start_utc"
        frames.append(frame)
    history = pd.concat(frames).sort_index()
    if history.index.has_duplicates or history["demand_mw"].isna().any():
        raise ValueError("History must have unique timestamps and complete demand.")
    output = root / "reports/app"
    output.mkdir(parents=True, exist_ok=True)
    history.to_parquet(output / "history.parquet", compression="zstd")
    manifest = {
        "rows": len(history),
        "source_sha256": sources,
        "history_sha256": hashlib.sha256((output / "history.parquet").read_bytes()).hexdigest(),
        "purpose": "Retrospective exploration; not used for model selection or fitting.",
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    export_history(parser.parse_args().root)
