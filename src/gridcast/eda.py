"""Training-only EDA. No validation, calibration or test target is read here."""

import argparse
import calendar
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import acf

from gridcast.data import load_config, update_readme, write_json
from gridcast.features import partition
from gridcast.validation import ValidationError

BLUE, TEAL, ORANGE = "#245880", "#138d90", "#c46b2a"


def summarize_training(frame: pd.DataFrame, evaluation: dict) -> dict:
    train = partition(frame, evaluation["periods"]["train"])
    if train.empty or train.demand_mw.isna().any():
        raise ValidationError("EDA requires a complete training target; inspect quality report")
    y = train.demand_mw
    weather = train.weather_temperature_c
    hours = train.groupby("hour").demand_mw.mean()
    months = train.groupby("month").demand_mw.mean()
    quantile = evaluation["conditions"]["extremes"]["quantile"]
    demand_threshold = float(y.quantile(quantile))
    temp_spec = evaluation["conditions"]["extreme_temperature"]
    cold, hot = weather.quantile([temp_spec["lower_quantile"], temp_spec["upper_quantile"]])
    correlations = acf(y.to_numpy(), nlags=168, fft=True, missing="raise")
    annual = y.groupby(y.index.year).mean()
    conditional = (
        train.assign(
            temperature_bin=pd.cut(
                weather,
                [-np.inf, 0, 10, 20, 25, np.inf],
                labels=["below_0", "0_to_10", "10_to_20", "20_to_25", "above_25"],
            )
        )
        .groupby("temperature_bin", observed=False)
        .demand_mw.agg(["mean", "count"])
    )
    return {
        "scope": "initial_training_only",
        "train_rows": len(train),
        "start_utc": str(train.index.min()),
        "last_interval_start_utc": str(train.index.max()),
        "mean_mw": float(y.mean()),
        "min_mw": float(y.min()),
        "max_mw": float(y.max()),
        "max_timestamp_utc": str(y.idxmax()),
        "annual_mean_mw_utc_year": {str(k): float(v) for k, v in annual.items()},
        "last_vs_first_annual_mean_pct": float((annual.iloc[-1] / annual.iloc[0] - 1) * 100),
        "peak_average_local_hour": int(hours.idxmax()),
        "trough_average_local_hour": int(hours.idxmin()),
        "peak_average_local_month": int(months.idxmax()),
        "weekday_mean_mw": float(train.loc[~train.weekend, "demand_mw"].mean()),
        "weekend_mean_mw": float(train.loc[train.weekend, "demand_mw"].mean()),
        "holiday_mean_mw": float(train.loc[train.holiday, "demand_mw"].mean()),
        "nonholiday_mean_mw": float(train.loc[~train.holiday, "demand_mw"].mean()),
        "acf_24": float(correlations[24]),
        "acf_168": float(correlations[168]),
        "extreme_demand_threshold_mw": demand_threshold,
        "extreme_definition": f"demand >= training q{quantile}",
        "training_extreme_hours": int(y.ge(demand_threshold).sum()),
        "extreme_temperature_low_c": float(cold),
        "extreme_temperature_high_c": float(hot),
        "weather_complete_hours": int(weather.notna().sum()),
        "weather_missing_hours": int(weather.isna().sum()),
        "demand_by_temperature_bin": {
            str(k): {"mean_mw": float(v["mean"]) if v["count"] else None, "hours": int(v["count"])}
            for k, v in conditional.iterrows()
        },
    }


def figure_style():
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titleweight": "bold",
            "axes.titlepad": 14,
            "figure.facecolor": "white",
            "axes.facecolor": "#fafcfd",
            "grid.color": "#dfe5e8",
            "grid.alpha": 0.7,
        }
    )


def save(fig, root: Path, name: str):
    fig.savefig(
        root / "reports/figures" / name,
        dpi=160,
        bbox_inches="tight",
        metadata={"Software": "GridCast"},
    )
    plt.close(fig)


def plot_eda(train: pd.DataFrame, summary: dict, evaluation: dict, root: Path):
    figure_style()
    y = train.demand_mw
    fig, ax = plt.subplots(figsize=(12, 4.3), layout="constrained")
    ax.plot(y.index, y / 1000, color=BLUE, alpha=0.20, lw=0.35, label="Hourly")
    ax.plot(y.resample("D").mean() / 1000, color=TEAL, lw=0.8, label="Daily mean")
    ax.set(
        title="How does demand vary across the training years?",
        ylabel="Average load (GW)",
        xlabel="UTC interval start · 2020–2022 only",
    )
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=4))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.grid(axis="y")
    ax.legend(loc="upper right", ncol=2)
    save(fig, root, "01_demand_history.png")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), layout="constrained")
    for weekend, name, color in [(False, "Weekday", BLUE), (True, "Weekend", TEAL)]:
        profile = train.loc[train.weekend.eq(weekend)].groupby("hour").demand_mw.mean() / 1000
        axes[0].plot(profile.index, profile, label=name, color=color, lw=2)
    axes[0].set(
        title="When does demand rise within the day?",
        xlabel="Local interval-start hour",
        ylabel="Mean load (GW)",
        xticks=[0, 4, 8, 12, 16, 20, 23],
    )
    axes[0].legend()
    axes[0].grid(axis="y")
    week = train.groupby("weekday").demand_mw.mean() / 1000
    axes[1].bar(range(7), week, color=[BLUE] * 5 + [TEAL] * 2)
    axes[1].set(
        title="How different are weekdays and weekends?",
        ylabel="Mean load (GW)",
        xticks=range(7),
        xticklabels=list(calendar.day_abbr),
    )
    axes[1].grid(axis="y")
    save(fig, root, "02_daily_weekly_patterns.png")

    fig, ax = plt.subplots(figsize=(12, 4.5), layout="constrained")
    matrix = train.pivot_table(values="demand_mw", index="month", columns="hour", aggfunc="mean")
    im = ax.imshow(matrix / 1000, aspect="auto", cmap="YlGnBu", origin="lower")
    ax.set(
        title="How does the daily load shape change with season?",
        xlabel="Local interval-start hour",
        ylabel="Local month",
        yticks=range(12),
        yticklabels=list(calendar.month_abbr)[1:],
        xticks=range(0, 24, 2),
        xticklabels=range(0, 24, 2),
    )
    fig.colorbar(im, ax=ax, label="Mean load (GW)")
    save(fig, root, "03_month_hour_heatmap.png")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.3), layout="constrained")
    complete = train.dropna(subset=["weather_temperature_c"])
    hb = axes[0].hexbin(
        complete.weather_temperature_c,
        complete.demand_mw / 1000,
        gridsize=48,
        mincnt=1,
        bins="log",
        cmap="YlGnBu",
    )
    axes[0].set(
        title="Is demand related to realized temperature?",
        xlabel="Three-station mean (°C)",
        ylabel="Average load (GW)",
    )
    fig.colorbar(hb, ax=axes[0], label="Hourly observations (log scale)")
    bins = pd.cut(complete.weather_temperature_c, np.arange(-25, 46, 5))
    grouped = complete.groupby(bins, observed=True).demand_mw.agg(["mean", "count", "std"])
    axes[1].plot([i.mid for i in grouped.index], grouped["mean"] / 1000, "o-", color=TEAL)
    axes[1].set(
        title="What is the temperature–load shape?",
        xlabel="5°C bin midpoint",
        ylabel="Mean load (GW)",
    )
    axes[1].grid(axis="y")
    fig.supxlabel("Observed weather association; season and calendar confounding remain", fontsize=9)
    save(fig, root, "04_weather_relationship.png")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), layout="constrained")
    correlations = acf(y.to_numpy(), nlags=336, fft=True, missing="raise")
    axes[0].plot(range(337), correlations, color=BLUE)
    for lag in [24, 168, 336]:
        axes[0].axvline(lag, color=ORANGE, lw=0.8, ls="--")
    axes[0].set(
        title="Do daily and weekly persistence matter?",
        xlabel="Elapsed-hour lag",
        ylabel="Autocorrelation",
        xticks=[0, 24, 72, 168, 240, 336],
    )
    means = train.groupby("month").demand_mw.mean() / 1000
    axes[1].plot(means.index, means, "o-", color=TEAL)
    axes[1].set(
        title="Which months carry the highest average load?",
        xlabel="Local month",
        ylabel="Mean load (GW)",
        xticks=range(1, 13),
    )
    axes[1].grid(axis="y")
    save(fig, root, "05_autocorrelation_annual_cycle.png")

    daily = y.resample("D").mean()
    decomposition = STL(daily, period=7, robust=True).fit()
    fig, axes = plt.subplots(3, 1, figsize=(12, 6.3), sharex=True, layout="constrained")
    for ax, component, name in zip(
        axes,
        [decomposition.trend, decomposition.seasonal, decomposition.resid],
        ["Trend", "Weekly", "Remainder"],
        strict=True,
    ):
        ax.plot(component.index, component / 1000, color=BLUE, lw=0.8)
        ax.set_ylabel(f"{name} (GW)")
        ax.grid(axis="y")
    axes[0].set_title("What remains after separating the weekly cycle? (daily-mean STL)")
    axes[-1].set_xlabel("UTC date · two-sided decomposition for description only, never a feature")
    save(fig, root, "06_descriptive_decomposition.png")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), layout="constrained")
    conditions = [~train.weekend & ~train.holiday, train.weekend & ~train.holiday, train.holiday]
    axes[0].boxplot(
        [y.loc[m].to_numpy() / 1000 for m in conditions],
        tick_labels=["Weekday", "Weekend", "Holiday"],
        showfliers=False,
        patch_artist=True,
        boxprops={"facecolor": "#a9d6dc"},
        medianprops={"color": BLUE},
    )
    axes[0].set(title="How do calendar-day load distributions differ?", ylabel="Average load (GW)")
    peak = y.idxmax()
    window = y.loc[peak - pd.Timedelta(days=3) : peak + pd.Timedelta(days=3)]
    axes[1].plot(window.index, window / 1000, color=BLUE)
    axes[1].axhline(
        summary["extreme_demand_threshold_mw"] / 1000,
        color=ORANGE,
        ls="--",
        label="Fixed training 95th percentile",
    )
    axes[1].set(
        title="How persistent was the training maximum?",
        ylabel="Average load (GW)",
        xlabel="UTC date · training peak window",
    )
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    axes[1].legend(fontsize=8)
    axes[1].grid(axis="y")
    save(fig, root, "07_calendar_extremes.png")

    fig, ax = plt.subplots(figsize=(12, 2.8), layout="constrained")
    colors = [BLUE, TEAL, ORANGE, "#89929a"]
    for n, (name, bounds) in enumerate(evaluation["periods"].items()):
        begin, end = pd.Timestamp(bounds["start"]), pd.Timestamp(bounds["end"])
        ax.barh(0, (end - begin).days, left=mdates.date2num(begin), height=0.6, color=colors[n])
        ax.text(
            mdates.date2num(begin + (end - begin) / 2),
            0,
            "Cal." if name == "calibration" else name.title(),
            ha="center",
            va="center",
            color="white",
            fontsize=10,
        )
    ax.xaxis_date()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.set(
        yticks=[],
        title="Which periods may inform each experimental decision?",
        xlabel="UTC interval-start boundaries · final test values excluded from EDA",
    )
    save(fig, root, "08_chronological_protocol.png")


def report_text(summary: dict) -> str:
    s = summary
    weekday_gap = (1 - s["weekend_mean_mw"] / s["weekday_mean_mw"]) * 100
    annual = ", ".join(f"{year}: {value:,.1f} MW" for year, value in s["annual_mean_mw_utc_year"].items())
    return f"""Training-only findings ({s["train_rows"]:,} hours; 2020–2022):

- Mean demand is **{s["mean_mw"]:,.1f} MW**; annual means are {annual}.
  This short record includes pandemic years; the change is descriptive, not a structural trend claim.
- The average local daily profile peaks at **{s["peak_average_local_hour"]:02d}:00** and reaches its
  minimum at **{s["trough_average_local_hour"]:02d}:00**. **{calendar.month_name[s["peak_average_local_month"]]}**
  has the highest average load. Weekend mean load is **{weekday_gap:.1f}%** below weekdays.
- Autocorrelation is **{s["acf_24"]:.3f} at 24 hours** and **{s["acf_168"]:.3f} at 168 hours**.
  Daily/weekly baselines are therefore substantive competitors, not token comparisons.
- Extreme demand is fixed at the training 95th percentile: **{s["extreme_demand_threshold_mw"]:,.1f} MW**.
  The training maximum is **{s["max_mw"]:,.0f} MW** at **{s["max_timestamp_utc"]}** (interval start).
- The three-station temperature proxy is present for **{s["weather_complete_hours"]:,} training hours**;
  **{s["weather_missing_hours"]:,}** remain missing. Training temperature 5th/95th percentiles are
  **{s["extreme_temperature_low_c"]:.2f}°C / {s["extreme_temperature_high_c"]:.2f}°C**.
  Weather associations are observational and confounded by season and calendar.

Source: [eda_summary.json](reports/eda_summary.json). No validation, calibration or test outcomes
contribute to these findings or thresholds.
"""


def run(root: Path):
    evaluation = load_config(root, "evaluation.yaml")
    train_path = root / "data/processed/train.parquet"
    train = pd.read_parquet(train_path)
    bounded = partition(train, evaluation["periods"]["train"])
    if len(bounded) != len(train):
        raise ValidationError("Training artifact contains rows from another partition")
    summary = summarize_training(train, evaluation)
    for folder in ["reports/figures", "reports/tables"]:
        (root / folder).mkdir(parents=True, exist_ok=True)
    write_json(root / "reports/eda_summary.json", summary)
    write_json(
        root / "reports/frozen_thresholds.json",
        {
            k: summary[k]
            for k in [
                "scope",
                "start_utc",
                "last_interval_start_utc",
                "extreme_demand_threshold_mw",
                "extreme_temperature_low_c",
                "extreme_temperature_high_c",
            ]
        },
    )
    for by in ["hour", "weekday", "month", "holiday"]:
        train.groupby(by).demand_mw.agg(["mean", "median", "std", "min", "max", "count"]).to_csv(
            root / f"reports/tables/demand_by_{by}.csv", float_format="%.6f"
        )
    daily_extremes = train.demand_mw.resample("D").max().nlargest(15)
    daily_extremes.rename("daily_max_mw").to_csv(root / "reports/tables/extreme_training_days.csv")
    plot_eda(train, summary, evaluation, root)
    text = report_text(summary)
    (root / "reports/eda_findings.md").write_text(text.replace("reports/eda_summary", "eda_summary"))
    update_readme(root, "EDA FINDINGS", text)
    print(f"EDA complete: {len(train):,} training hours, 8 figures. Final test not read.")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    run(args.root.resolve())


if __name__ == "__main__":
    main()
