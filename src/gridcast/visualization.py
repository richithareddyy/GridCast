"""Question-led validation figures built from persisted backtest results."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

from gridcast.eda import figure_style, save

COLORS = {
    "naive": "#89929a",
    "daily": "#138d90",
    "weekly": "#c46b2a",
    "dhr_ar1": "#245880",
    "lightgbm": "#5c9040",
    "lstm": "#9a538b",
}
LABELS = {
    "naive": "Naive",
    "daily": "Daily",
    "weekly": "Weekly",
    "dhr_ar1": "Harmonic",
    "lightgbm": "LightGBM",
    "lstm": "LSTM",
}


def backtest_figures(predictions, tables, root):
    figure_style()
    comparison = tables["model_comparison"].set_index("model").reindex(COLORS)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), layout="constrained")
    for ax, metric in zip(axes, ["MAE", "RMSE"], strict=True):
        ax.bar(range(len(COLORS)), comparison[metric], color=list(COLORS.values()))
        ax.set(
            xticks=range(len(COLORS)),
            xticklabels=list(LABELS.values()),
            ylabel=f"{metric} (MW)",
            title=f"Which model has the lowest validation {metric}?",
        )
        ax.tick_params(axis="x", labelsize=8)
        ax.grid(axis="y")
    save(fig, root, "09_model_comparison.png")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), layout="constrained")
    for model, label in LABELS.items():
        fold = tables["fold_metrics"].loc[tables["fold_metrics"].model.eq(model)]
        horizon = tables["horizon_metrics"].loc[tables["horizon_metrics"].model.eq(model)]
        axes[0].plot(pd.to_datetime(fold.fold), fold.MAE, "o-", label=label, color=COLORS[model])
        axes[1].plot(horizon.horizon, horizon.MAE, label=label, color=COLORS[model])
    axes[0].set(title="Is model ranking stable across monthly folds?", ylabel="MAE (MW)", xlabel="2023 fold")
    axes[1].set(title="How does error change across the 24-hour path?", ylabel="MAE (MW)", xlabel="Horizon (hour)")
    axes[0].xaxis.set_major_locator(mdates.MonthLocator())
    axes[0].xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    axes[0].legend(fontsize=8)
    axes[1].axvline(24, ls="--", color="#89929a", lw=0.8)
    axes[1].set_xticks([1, 6, 12, 18, 24])
    save(fig, root, "10_fold_horizon_errors.png")

    first = predictions.origin_utc.min()
    subset = predictions.loc[predictions.target_start_utc < first + pd.Timedelta(days=7)]
    fig, ax = plt.subplots(figsize=(12, 4), layout="constrained")
    actual = subset.loc[subset.model.eq("daily")]
    ax.plot(actual.target_start_utc, actual.actual / 1000, color="black", lw=1.6, label="Actual")
    for model in ["daily", "lightgbm", "lstm"]:
        part = subset.loc[subset.model.eq(model)]
        ax.plot(
            part.target_start_utc, part.prediction / 1000, color=COLORS[model], lw=0.9, alpha=0.85, label=LABELS[model]
        )
    ax.set(
        title="How do forecasts behave in the first validation week?",
        ylabel="Load (GW)",
        xlabel="UTC interval start · fixed example window, daily origins",
    )
    ax.legend(ncol=4, fontsize=8)
    save(fig, root, "11_validation_forecasts.png")

    names = ["normal_weekday", "weekend", "holiday", "peak_hours", "extreme_demand", "extreme_temperature"]
    conditions = tables["condition_metrics"].pivot(index="condition", columns="model", values="MAE")
    conditions = conditions.reindex(names)[list(COLORS)]
    fig, ax = plt.subplots(figsize=(12, 4), layout="constrained")
    conditions.rename(columns=LABELS).plot.bar(ax=ax, color=list(COLORS.values()), rot=0)
    ax.set(title="When does each forecasting approach struggle?", ylabel="MAE (MW)", xlabel="")
    ax.set_xticklabels(["Weekday", "Weekend", "Holiday", "Peak hours", "Extreme load", "Extreme temp."])
    ax.legend(fontsize=8, ncol=3)
    fig.supxlabel("Validation only · overlapping groups · thresholds fitted on initial training only", fontsize=9)
    save(fig, root, "12_error_conditions.png")


def learning_figures(comparison, fits, root):
    import numpy as np

    comparison = comparison.sort_values("mean_mae_difference_mw")
    fig, ax = plt.subplots(figsize=(9, 4), layout="constrained")
    mean = comparison.mean_mae_difference_mw.to_numpy()
    ax.errorbar(
        mean,
        np.arange(len(mean)),
        xerr=np.vstack([mean - comparison.ci95_low, comparison.ci95_high - mean]),
        fmt="o",
        capsize=4,
    )
    ax.set_yticks(np.arange(len(mean)), [LABELS[k] for k in comparison.model])
    ax.axvline(0, color="black", ls="--", lw=1)
    ax.set(
        title="Are daily loss differences distinguishable from zero?",
        xlabel="MAE difference vs daily baseline (MW) · pointwise HAC 95% intervals",
    )
    fig.supxlabel("Negative favors model · seven daily lags · exploratory validation comparison", fontsize=9)
    save(fig, root, "13_paired_comparisons.png")

    importance = pd.DataFrame([fit["learning"]["tree_gain_importance"] for fit in fits])
    normalized = importance.div(importance.sum(axis=1), axis=0)
    top = normalized.mean().nlargest(12).sort_values()
    fig, ax = plt.subplots(figsize=(9, 5), layout="constrained")
    ax.barh(top.index, top.values * 100, color=COLORS["lightgbm"])
    ax.set(
        title="Which inputs does LightGBM use to reduce training loss?",
        xlabel="Mean share of split gain across nine fits (%)",
    )
    fig.supxlabel(
        "Descriptive model importance; correlated features share gain. This is not causal attribution.", fontsize=9
    )
    save(fig, root, "14_feature_importance.png")


def reliability_figures(intervals, outputs, root):

    figure_style()
    first = intervals.target_start_utc.min()

    def interval_plot(window, title, filename):
        fig, ax = plt.subplots(figsize=(12, 4), layout="constrained")
        for level, alpha in [(0.95, 0.15), (0.8, 0.22), (0.5, 0.35)]:
            part = window.loc[window.level.eq(level)].sort_values("target_start_utc")
            ax.fill_between(
                part.target_start_utc,
                part.lower.to_numpy() / 1000,
                part.upper.to_numpy() / 1000,
                color="#245880",
                alpha=alpha,
                label=f"{level:.0%} interval",
            )
        part = window.loc[window.level.eq(0.95)].sort_values("target_start_utc")
        ax.plot(part.target_start_utc, part.actual / 1000, color="black", lw=1.2, label="Actual")
        ax.plot(part.target_start_utc, part.prediction / 1000, color="#138d90", lw=1.2, label="LightGBM")
        ax.set(title=title, ylabel="Average load (GW)", xlabel="UTC interval start")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
        ax.legend(ncol=5, fontsize=8)
        save(fig, root, filename)

    interval_plot(
        intervals.loc[intervals.target_start_utc < first + pd.Timedelta(days=7)],
        "Do forecast intervals contain demand in the first test week?",
        "15_prediction_intervals.png",
    )
    metrics = outputs["interval_conditions"]
    fig, ax = plt.subplots(figsize=(7, 5), layout="constrained")
    for condition, label in [
        ("overall", "All test hours"),
        ("peak_hours", "Peak hours"),
        ("extreme_demand", "Extreme demand"),
        ("extreme_temperature", "Extreme temperature"),
    ]:
        part = metrics.loc[metrics.condition.eq(condition)].sort_values("level")
        ax.plot(part.level * 100, part.coverage * 100, "o-", label=f"{label} (n={int(part.n.iloc[0])})")
    ax.plot([40, 100], [40, 100], "--", color="black", label="Nominal = observed")
    ax.set(
        xlim=(40, 100),
        ylim=(0, 100),
        title="Are intervals calibrated across test conditions?",
        xlabel="Nominal coverage (%)",
        ylabel="Observed coverage (%)",
    )
    ax.legend(fontsize=8)
    save(fig, root, "16_interval_calibration.png")
    names = ["overall", "normal_weekday", "weekend", "holiday", "peak_hours", "extreme_demand", "extreme_temperature"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), layout="constrained")
    for ax, value, title in zip(
        axes,
        ["mean_width_mw", "interval_score_mw"],
        ["How wide are the intervals?", "What is the width-and-miss penalty?"],
        strict=True,
    ):
        part = metrics.pivot(index="condition", columns="level", values=value).reindex(names)
        part.rename(columns={0.5: "50%", 0.8: "80%", 0.95: "95%"}).plot.bar(
            ax=ax, rot=35, color=["#91b6cb", "#5085a4", "#245880"]
        )
        ax.set(title=title, ylabel="MW", xlabel="")
        ax.set_xticklabels(
            ["All", "Weekday", "Weekend", "Holiday", "Peak", "Extreme load", "Extreme temp."], ha="right"
        )
        ax.legend(title="Nominal", fontsize=8)
    save(fig, root, "17_interval_width_score.png")
    rolling = outputs["rolling_reliability"]
    fig, axes = plt.subplots(3, 2, figsize=(12, 8), layout="constrained")
    measures = [
        ("MAE", "MAE (MW)"),
        ("RMSE", "RMSE (MW)"),
        ("residual_mean_mw", "Residual mean (MW)"),
        ("residual_variance_mw2", "Residual variance (MW²)"),
        ("coverage_95", "95% interval coverage"),
        ("n", "Matured hours"),
    ]
    for ax, (column, label) in zip(axes.flat, measures, strict=True):
        ax.plot(rolling.as_of_utc, rolling[column], color="#245880")
        ax.set(ylabel=label, xlabel="As-of date (UTC)")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
        if column == "coverage_95":
            ax.axhline(0.95, ls="--", color="#c46b2a")
            ax.set_ylim(0, 1)
        if column == "residual_mean_mw":
            ax.axhline(0, ls="--", color="#c46b2a")
    fig.suptitle("How does trailing 30-day forecast reliability change?", fontsize=14)
    fig.supxlabel(
        "Past matured test labels only; first windows require 168 hours and grow to a full month.", fontsize=9
    )
    save(fig, root, "18_rolling_reliability.png")
    p = intervals.loc[intervals.level.eq(0.95)].sort_values("target_start_utc")
    events = outputs["forecast_anomalies"]
    fig, ax = plt.subplots(figsize=(12, 4), layout="constrained")
    radius = (p.upper - p.lower) / 2
    ax.fill_between(p.target_start_utc, -radius, radius, color="#245880", alpha=0.15, label="95% residual band")
    ax.plot(p.target_start_utc, p.actual - p.prediction, color="#245880", lw=0.4, label="Residual")
    ax.scatter(events.target_start_utc, events.residual_mw, s=5, color="#c46b2a", label="Interval violation", zorder=3)
    ax.axhline(0, color="black", lw=0.8)
    ax.set(
        title="When does demand fall outside its issued 95% interval?",
        ylabel="Actual − forecast (MW)",
        xlabel="UTC interval start",
    )
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    ax.legend(ncol=3, fontsize=8)
    save(fig, root, "19_forecast_anomalies.png")
    peak = p.loc[p.actual.idxmax(), "target_start_utc"].floor("D")
    interval_plot(
        intervals.loc[
            (intervals.target_start_utc >= peak - pd.Timedelta(days=2))
            & (intervals.target_start_utc < peak + pd.Timedelta(days=3))
        ],
        "How did issued intervals behave around maximum test demand?",
        "20_extreme_demand_intervals.png",
    )
