"""Question-led validation figures built from persisted backtest results."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

from gridcast.eda import figure_style, save

COLORS = {"naive": "#89929a", "daily": "#138d90", "weekly": "#c46b2a", "dhr_ar1": "#245880"}
LABELS = {"naive": "Naive", "daily": "Daily (available)", "weekly": "Weekly", "dhr_ar1": "Harmonic + AR(1)"}


def backtest_figures(predictions, tables, root):
    figure_style()
    comparison = tables["model_comparison"].set_index("model").reindex(COLORS)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), layout="constrained")
    for ax, metric in zip(axes, ["MAE", "RMSE"], strict=True):
        ax.bar(range(4), comparison[metric], color=list(COLORS.values()))
        ax.set(
            xticks=range(4),
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
    for model in ["daily", "weekly", "dhr_ar1"]:
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
    ax.legend(fontsize=8, ncol=2)
    fig.supxlabel("Validation only · overlapping groups · thresholds fitted on initial training only", fontsize=9)
    save(fig, root, "12_error_conditions.png")
