"""Build the case study from saved forecasts without refitting models."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from gridcast.anomalies import anomaly_events, rolling_reliability
from gridcast.data import load_config, sha256, update_readme, write_json
from gridcast.eda import figure_style, save
from gridcast.evaluation import condition_scores, paired_comparisons, score_groups
from gridcast.uncertainty import interval_tables
from gridcast.visualization import COLORS, LABELS, backtest_figures, learning_figures, reliability_figures


def error_figures(hourly, seasonal, intervals, events, root):
    figure_style()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), layout="constrained")
    for model, part in hourly.groupby("model"):
        axes[0].plot(part.local_hour, part.MAE, label=LABELS[model], color=COLORS[model])
    axes[0].set(title="Validation error by local hour", xlabel="America/New_York hour", ylabel="MAE (MW)")
    axes[0].legend(ncol=2, fontsize=8)
    seasons = seasonal.pivot(index="condition", columns="model", values="MAE")
    seasons = seasons.reindex(["winter", "spring", "summer", "autumn"])[list(COLORS)]
    seasons.rename(columns=LABELS).plot.bar(ax=axes[1], color=list(COLORS.values()), rot=0, legend=False)
    axes[1].set(title="Validation error by season", xlabel="", ylabel="MAE (MW)")
    fig.supxlabel("Validation ends in September: autumn contains September only.", fontsize=9)
    save(fig, root, "21_hour_season_errors.png")

    largest = events.loc[events.residual_mw.abs().idxmax()]
    center = largest.target_start_utc.floor("D")
    window = intervals.loc[
        intervals.level.eq(0.95)
        & intervals.target_start_utc.ge(center - pd.Timedelta(days=1))
        & intervals.target_start_utc.lt(center + pd.Timedelta(days=2))
    ].sort_values("target_start_utc")
    flagged = events.loc[events.target_start_utc.isin(window.target_start_utc)]
    fig, ax = plt.subplots(figsize=(12, 4), layout="constrained")
    ax.fill_between(
        window.target_start_utc,
        window.lower / 1000,
        window.upper / 1000,
        color=COLORS["dhr_ar1"],
        alpha=0.15,
        label="95% interval",
    )
    ax.plot(window.target_start_utc, window.actual / 1000, color="black", label="Actual")
    ax.plot(window.target_start_utc, window.prediction / 1000, color=COLORS["lightgbm"], label="Forecast")
    ax.scatter(
        flagged.target_start_utc,
        flagged.actual / 1000,
        color=COLORS["weekly"],
        s=22,
        label="Interval violation",
        zorder=3,
    )
    ax.set(
        title="Largest absolute residual among flagged test hours", xlabel="UTC interval start", ylabel="Demand (GW)"
    )
    ax.legend(ncol=4, fontsize=8)
    fig.supxlabel("Selected retrospectively for diagnosis; no external event attribution.", fontsize=9)
    save(fig, root, "22_anomaly_case_study.png")
    return {
        "selection": "largest absolute residual among 95% interval violations; retrospective diagnostic example",
        "target_start_utc": str(largest.target_start_utc),
        "detected_at_utc": str(largest.detected_at_utc),
        "residual_mw": float(largest.residual_mw),
        "actual_mw": float(largest.actual),
        "prediction_mw": float(largest.prediction),
        "lower_mw": float(largest.lower),
        "upper_mw": float(largest.upper),
    }


def findings(summary):
    selected = summary["selected_model"]
    point = pd.DataFrame(summary["test_metrics"]).set_index("model")
    conditions = pd.DataFrame(summary["validation_conditions"])
    extreme = conditions.loc[conditions.condition.eq("extreme_demand")].sort_values("MAE").iloc[0]
    intervals = pd.DataFrame(summary["interval_coverage"])
    overall = intervals.loc[intervals.condition.eq("overall")]
    tail = intervals.loc[intervals.condition.eq("extreme_demand")].set_index("level")
    lines = [
        f"**{LABELS[selected]} had the lowest validation MAE.** On the final test, its MAE was "
        f"**{point.loc[selected, 'MAE']:,.1f} MW**, compared with **{point.loc['daily', 'MAE']:,.1f} MW** "
        f"for Daily Seasonal Naive ({summary['test_mae_improvement_pct']:.1f}% lower). "
        "Only the selected model and the predeclared daily baseline were evaluated on the final test.",
        "",
        "| Model | Validation MAE (MW) | RMSE (MW) | WAPE (%) | MASE |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in sorted(summary["validation_metrics"], key=lambda row: row["MAE"]):
        lines.append(
            f"| {LABELS[row['model']]} | {row['MAE']:,.1f} | {row['RMSE']:,.1f} | "
            f"{row['WAPE']:.2f} | {row['MASE']:.3f} |"
        )
    lines += [
        "",
        f"The ranking changes under stress: **{LABELS[extreme.model]}** had the lowest validation "
        f"extreme-demand MAE (**{extreme.MAE:,.1f} MW**). The threshold is the initial-training "
        f"95th percentile, **{summary['extreme_demand_threshold_mw']:,.1f} MW**, held fixed for all models.",
        "",
        "| Nominal interval | Test coverage | Mean width (MW) | Extreme-demand coverage |",
        "|---|---:|---:|---:|",
    ]
    for row in overall.itertuples():
        lines.append(
            f"| {row.level:.0%} | {row.coverage:.2%} | {row.mean_width_mw:,.1f} | "
            f"{tail.loc[row.level, 'coverage']:.2%} |"
        )
    case = summary["anomaly_case"]
    lines += [
        "",
        f"The test contains **{summary['anomaly_hours']:,} flagged hours** in "
        f"**{summary['anomaly_episodes']:,} contiguous UTC episodes**. The largest flagged absolute residual "
        f"occurred at **{case['target_start_utc']}**: actual minus forecast was **{case['residual_mw']:,.1f} MW**. "
        "These are unusual demand observations relative to the forecast, not verified outages or grid failures.",
        "",
        "Overall coverage conceals undercoverage during extreme demand. No test-based retuning or causal "
        "explanation of anomalies is used. Full metrics, condition slices, interval scores and the diagnostic "
        "case selection are in the linked reports.",
    ]
    return "\n".join(lines) + "\n"


def run(root: Path):
    inputs = [
        "configs/evaluation.yaml",
        "reports/frozen_thresholds.json",
        "reports/model_selection.json",
        "reports/statistical_fits.json",
        "reports/predictions/validation.parquet",
        "reports/predictions/reserved.parquet",
        "reports/predictions/test_intervals.parquet",
    ]
    evaluation = load_config(root, "evaluation.yaml")
    thresholds = json.loads((root / inputs[1]).read_text())
    selection = json.loads((root / inputs[2]).read_text())
    fits = json.loads((root / inputs[3]).read_text())
    validation = pd.read_parquet(root / inputs[4])
    test = pd.read_parquet(root / inputs[5]).query("partition == 'test'")
    intervals = pd.read_parquet(root / inputs[6])
    for folder in ["reports/tables", "reports/figures", "reports/results"]:
        (root / folder).mkdir(parents=True, exist_ok=True)

    validation_tables = {
        "model_comparison": score_groups(validation, ["model"]),
        "fold_metrics": score_groups(validation, ["fold", "model"]),
        "horizon_metrics": score_groups(validation, ["horizon", "model"]),
        "condition_metrics": condition_scores(validation, thresholds),
    }
    paired = paired_comparisons(validation, evaluation["comparison"]["block_days"])
    backtest_figures(validation, validation_tables, root)
    learning_figures(paired, fits, root)
    spec = evaluation["uncertainty"]
    delay = evaluation["forecast"]["assumed_reporting_delay_hours"]
    events, episodes = anomaly_events(intervals, spec["anomaly_level"], delay)
    reliability = interval_tables(intervals, thresholds)
    reliability.update(
        forecast_anomalies=events,
        anomaly_episodes=episodes,
        rolling_reliability=rolling_reliability(
            intervals, spec["monitor_window_days"], spec["monitor_minimum_hours"], delay
        ),
    )
    reliability_figures(intervals, reliability, root)

    conditions = pd.concat(
        [
            validation_tables["condition_metrics"].assign(partition="validation"),
            condition_scores(test, thresholds).assign(partition="test"),
        ],
        ignore_index=True,
    )
    local_hour = validation.target_start_utc.dt.tz_convert(evaluation["calendar_timezone"]).dt.hour
    hourly = score_groups(validation.assign(local_hour=local_hour), ["local_hour", "model"])
    seasonal = validation_tables["condition_metrics"].query("condition in ['winter', 'spring', 'summer', 'autumn']")
    case = error_figures(hourly, seasonal, intervals, events, root)
    point = intervals.loc[intervals.level.eq(spec["anomaly_level"])].copy()
    point["month_utc"] = point.target_start_utc.dt.strftime("%Y-%m")
    point["flagged"] = ~point.actual.between(point.lower, point.upper)
    monthly = point.groupby("month_utc").agg(hours=("flagged", "size"), flagged_hours=("flagged", "sum")).reset_index()
    monthly["flagged_fraction"] = monthly.flagged_hours / monthly.hours
    overall = pd.concat(
        [
            validation_tables["model_comparison"].assign(partition="validation"),
            score_groups(test, ["model"]).assign(partition="test"),
        ],
        ignore_index=True,
    )
    tables = {
        "overall_model_metrics": overall,
        "extreme_demand_metrics": conditions.query("condition == 'extreme_demand'"),
        "peak_hour_metrics": conditions.query("condition == 'peak_hours'"),
        "error_by_local_hour": hourly,
        "error_by_season": seasonal,
        "interval_coverage": reliability["interval_conditions"],
        "anomaly_summary": monthly,
    }
    for name, frame in tables.items():
        frame.to_csv(root / f"reports/tables/{name}.csv", index=False, float_format="%.9f")
    test_metrics = overall.loc[overall.partition.eq("test")].drop(columns="partition")
    scores = test_metrics.set_index("model")
    selected = selection["selected_model"]
    summary = {
        "selected_model": selected,
        "validation_metrics": validation_tables["model_comparison"].to_dict("records"),
        "validation_conditions": validation_tables["condition_metrics"].to_dict("records"),
        "test_metrics": test_metrics.to_dict("records"),
        "test_conditions": conditions.loc[conditions.partition.eq("test")].to_dict("records"),
        "test_mae_improvement_pct": float(100 * (1 - scores.loc[selected, "MAE"] / scores.loc["daily", "MAE"])),
        "extreme_demand_threshold_mw": thresholds["extreme_demand_threshold_mw"],
        "interval_coverage": reliability["interval_conditions"].to_dict("records"),
        "anomaly_hours": len(events),
        "anomaly_episodes": len(episodes),
        "anomaly_case": case,
        "forecast": evaluation["forecast"],
        "periods": evaluation["periods"],
        "source_sha256": {name: sha256(root / name) for name in inputs},
    }
    # Empty condition slices are JSON null, not the nonstandard NaN token.
    summary = json.loads(json.dumps(summary), parse_constant=lambda _: None)
    write_json(root / "reports/results/summary.json", summary)
    text = findings(summary)
    (root / "reports/results/findings.md").write_text(text)
    update_readme(root, "RESULTS", text)
    print(f"Reporting complete: {len(tables)} tables, 14 figures, results summary and README findings.")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    run(parser.parse_args().root.resolve())


if __name__ == "__main__":
    main()
