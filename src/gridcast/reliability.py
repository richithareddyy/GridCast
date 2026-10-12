"""Frozen-model calibration and prequential final-test reliability evaluation."""

import argparse
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from gridcast.anomalies import anomaly_events, rolling_reliability
from gridcast.data import load_config, sha256, update_readme, write_json
from gridcast.evaluation import condition_scores, mase_scale, score_groups
from gridcast.features import available_history, validate_history
from gridcast.learning import build_samples, origin_inputs, tree_data
from gridcast.uncertainty import interval_tables, residual_intervals
from gridcast.validation import ValidationError


def load_reserved_study(root, evaluation, selection):
    # The immutable selection evidence must exist before opening any reserved outcome partition.
    if selection["selected_model"] != "lightgbm":
        raise ValidationError("This frozen reliability protocol is for the selected LightGBM model")
    if sha256(root / "reports/predictions/validation.parquet") != selection["validation_predictions_sha256"]:
        raise ValidationError("Validation predictions no longer match model-selection evidence")
    if selection["frozen_boost_rounds"] < 1:
        raise ValidationError("Deployment tree count must be frozen before reserved outcomes are read")
    parts = []
    for name in ["train", "validation", "calibration", "test"]:
        frame = pd.read_parquet(root / f"data/processed/{name}.parquet")
        bounds = evaluation["periods"][name]
        expected = pd.date_range(bounds["start"], bounds["end"], inclusive="left", freq="h")
        if not frame.index.equals(expected):
            raise ValidationError(f"{name}: incorrect reserved-study time axis")
        parts.append(frame)
    result = pd.concat(parts)
    validate_history(result.demand_mw)
    return result


def forecast_reserved(study, evaluation, selection, artifact_dir=None):
    delay = selection["reporting_delay_hours"]
    lookback = selection["lookback_hours"]
    samples = build_samples(study.demand_mw, lookback, delay)
    begin = pd.Timestamp(evaluation["periods"]["calibration"]["start"])
    end = pd.Timestamp(evaluation["periods"]["test"]["end"])
    test_begin = pd.Timestamp(evaluation["periods"]["test"]["start"])
    params = {**selection["frozen_lightgbm_parameters"], "seed": selection["seed"]}
    records, audits = [], []
    for start in pd.date_range(begin, end, freq="MS", inclusive="left"):
        stop = min(start + pd.offsets.MonthBegin(1), end)
        train = samples.matured(start)
        booster = lgb.train(params, tree_data(train), num_boost_round=selection["frozen_boost_rounds"])
        history = available_history(study[["demand_mw"]], start, delay).demand_mw
        scale = mase_scale(history)
        fold = start.strftime("%Y-%m")
        audits.append(
            {
                "fold": fold,
                "fit_origin_utc": str(start),
                "training_origins": len(train.origins),
                "latest_training_origin": str(train.origins[-1]),
                "latest_training_label_available_at": str(train.origins[-1] + pd.Timedelta(hours=24 + delay)),
                "boost_rounds": selection["frozen_boost_rounds"],
                "mase_scale": scale,
            }
        )
        if artifact_dir is not None:
            artifact_dir.mkdir(parents=True, exist_ok=True)
            booster.save_model(str(artifact_dir / f"{fold}_lightgbm.txt"))
        print(
            f"Reserved fold {fold}: {len(train.origins)} matured paths, fixed {selection['frozen_boost_rounds']} trees",
            flush=True,
        )
        for origin in pd.date_range(start, stop, freq="24h", inclusive="left"):
            history = available_history(study[["demand_mw"]], origin, delay).demand_mw
            features, _, _, base = origin_inputs(history, origin, lookback, delay)
            predicted = base + booster.predict(features.to_numpy(dtype=np.float32), num_threads=1)
            # Outcomes and observed weather are attached only after both forecasts have been made.
            context = study.reindex(features.index)
            for model, values in [("lightgbm", predicted), ("daily", base)]:
                frame = pd.DataFrame(
                    {
                        "origin_utc": origin,
                        "target_start_utc": features.index,
                        "target_end_utc": features.index + pd.Timedelta(hours=1),
                        "horizon": np.arange(1, 25),
                        "fold": fold,
                        "partition": "test" if origin >= test_begin else "calibration",
                        "model": model,
                        "prediction": values,
                        "actual": context.demand_mw.to_numpy(),
                        "mase_scale": scale,
                    }
                )
                for col in ["weekend", "holiday", "peak_hour", "month", "weather_temperature_c"]:
                    frame[col] = context[col].to_numpy()
                records.append(frame)
    return pd.concat(records, ignore_index=True), audits


def evaluate_reliability(points, evaluation, thresholds):
    selected = points.loc[points.model.eq("lightgbm")].copy()
    test_points = points.loc[points.partition.eq("test")]
    test = selected.loc[selected.partition.eq("test")]
    spec = evaluation["uncertainty"]
    tables = []
    for _origin, frame in test.groupby("origin_utc"):
        tables.append(
            residual_intervals(
                frame,
                selected,
                levels=spec["levels"],
                window_days=spec["rolling_window_days"],
                min_samples=spec["minimum_samples_per_horizon"],
                delay=evaluation["forecast"]["assumed_reporting_delay_hours"],
            )
        )
    intervals = pd.concat(tables, ignore_index=True)
    keys = ["origin_utc", "target_start_utc", "horizon"]
    context = test.drop(columns=["prediction"])
    intervals = intervals.merge(context, on=keys, validate="many_to_one")
    events, episodes = anomaly_events(
        intervals, spec["anomaly_level"], evaluation["forecast"]["assumed_reporting_delay_hours"]
    )
    rolling = rolling_reliability(
        intervals,
        spec["monitor_window_days"],
        spec["monitor_minimum_hours"],
        evaluation["forecast"]["assumed_reporting_delay_hours"],
    )
    outputs = interval_tables(intervals, thresholds)
    outputs["test_model_comparison"] = score_groups(test_points, ["model"])
    outputs["test_condition_metrics"] = condition_scores(test_points, thresholds)
    outputs["test_month_metrics"] = score_groups(test_points, ["fold", "model"])
    outputs["forecast_anomalies"] = events
    outputs["anomaly_episodes"] = episodes
    outputs["rolling_reliability"] = rolling
    return intervals, outputs


def findings_text(outputs, selection, evaluation):
    overall = outputs["interval_conditions"].query("condition == 'overall'")
    extreme = outputs["interval_conditions"].query("condition == 'extreme_demand'").set_index("level")
    peak = outputs["interval_conditions"].query("condition == 'peak_hours'").set_index("level")
    point = outputs["test_model_comparison"].set_index("model")
    lines = [
        f"Final test: 2024, {int(point.loc['lightgbm', 'n']):,} hourly targets; "
        f"LightGBM with {selection['frozen_boost_rounds']} frozen trees and monthly expanding refits.",
        "",
        f"Point MAE: **{point.loc['lightgbm', 'MAE']:,.1f} MW** vs **{point.loc['daily', 'MAE']:,.1f} MW** "
        f"for daily seasonal naive; RMSE **{point.loc['lightgbm', 'RMSE']:,.1f} MW**, "
        f"WAPE **{point.loc['lightgbm', 'WAPE']:.2f}%**, MASE **{point.loc['lightgbm', 'MASE']:.3f}**.",
        "",
        "| Nominal | Coverage | Mean width (MW) | Interval score (MW) | Peak coverage | Extreme-demand coverage |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in overall.itertuples():
        lines.append(
            f"| {row.level:.0%} | {row.coverage:.2%} | {row.mean_width_mw:,.1f} | {row.interval_score_mw:,.1f} | "
            f"{peak.loc[row.level, 'coverage']:.2%} | {extreme.loc[row.level, 'coverage']:.2%} |"
        )
    covered95 = float(overall.loc[overall.level.eq(0.95), "coverage"].iloc[0])
    extreme95 = float(extreme.loc[0.95, "coverage"])
    lines += [
        "",
        f"At nominal 95%, coverage is **{covered95:.2%} overall** and **{extreme95:.2%} during extreme demand**. "
        "Aggregate calibration does not imply reliable coverage in the high-demand tail.",
    ]
    events = outputs["forecast_anomalies"]
    episodes = outputs["anomaly_episodes"]
    lines += [
        "",
        f"There are **{len(events):,}** hourly 95% interval violations in **{len(episodes):,}** "
        "contiguous UTC episodes. These are forecast anomalies, not verified outages or failures.",
        "",
        "The same horizon-wise calibration window is used for every condition; no post-test recalibration "
        "or interval-width tuning is performed. Condition groups overlap, and rare-event estimates have "
        "limited effective sample size. Coverage is empirical, not a distribution-free guarantee.",
    ]
    rolling = outputs["rolling_reliability"]
    if len(rolling):
        full_hours = (
            evaluation["uncertainty"]["monitor_window_days"] * 24
            - evaluation["forecast"]["assumed_reporting_delay_hours"]
        )
        full = rolling.loc[rolling.n.ge(full_hours)]
        if len(full):
            lines += [
                "",
                f"Full trailing-window MAE ranges from **{full.MAE.min():,.1f} to {full.MAE.max():,.1f} MW**; "
                f"95% coverage ranges from **{full.coverage_95.min():.1%} to {full.coverage_95.max():.1%}**. "
                "These descriptive changes are not a formal causal or structural-break diagnosis.",
            ]
    return "\n".join(lines) + "\n"


def run(root):
    evaluation = load_config(root, "evaluation.yaml")
    selection = json.loads((root / "reports/model_selection.json").read_text())
    thresholds = json.loads((root / "reports/frozen_thresholds.json").read_text())
    study = load_reserved_study(root, evaluation, selection)
    points, fits = forecast_reserved(study, evaluation, selection, root / "reports/reserved_models")
    return write_outputs(root, evaluation, selection, thresholds, points, fits)


def write_outputs(root, evaluation, selection, thresholds, points, fits):
    intervals, outputs = evaluate_reliability(points, evaluation, thresholds)
    points.to_parquet(root / "reports/predictions/reserved.parquet", index=False)
    intervals.to_parquet(root / "reports/predictions/test_intervals.parquet", index=False)
    for name, frame in outputs.items():
        frame.to_csv(root / f"reports/tables/{name}.csv", index=False, float_format="%.9f")
    write_json(root / "reports/reserved_fits.json", fits)
    write_json(
        root / "reports/reliability_summary.json",
        {
            "selection_sha256": sha256(root / "reports/model_selection.json"),
            "evaluation_config_sha256": sha256(root / "configs/evaluation.yaml"),
            "test_hours": int(points.loc[points.partition.eq("test") & points.model.eq("lightgbm")].shape[0]),
            "calibration_hours": int(
                points.loc[points.partition.eq("calibration") & points.model.eq("lightgbm")].shape[0]
            ),
            "interval_rows": len(intervals),
            "anomaly_hours": len(outputs["forecast_anomalies"]),
            "anomaly_episodes": len(outputs["anomaly_episodes"]),
            "residual_count_min": int(intervals.calibration_n.min()),
            "residual_count_max": int(intervals.calibration_n.max()),
            "all_residuals_matured": bool((intervals.latest_residual_available_at <= intervals.origin_utc).all()),
            "input_hashes": {name: sha256(root / f"data/processed/{name}.parquet") for name in evaluation["periods"]},
        },
    )
    from gridcast.visualization import reliability_figures

    reliability_figures(intervals, outputs, root)
    findings = findings_text(outputs, selection, evaluation)
    (root / "reports/reliability_findings.md").write_text(findings)
    update_readme(root, "RELIABILITY RESULTS", findings)
    print(
        "Reliability evaluation complete: frozen choices, calibration followed by prequential final test.", flush=True
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    run(parser.parse_args().root.resolve())


if __name__ == "__main__":
    main()
