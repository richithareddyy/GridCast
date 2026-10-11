"""GridCast analytical console over frozen evaluation artifacts."""

from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
LABELS = {
    "naive": "Persistence",
    "daily": "Daily seasonal",
    "weekly": "Weekly seasonal",
    "dhr_ar1": "Harmonic + AR(1)",
    "lightgbm": "LightGBM",
    "lstm": "LSTM",
}
SECTIONS = [
    "Overview",
    "Demand Explorer",
    "Forecast",
    "Model Comparison",
    "Extreme Events",
    "Anomalies",
    "Reliability",
    "Methodology",
]
st.set_page_config(page_title="GridCast | Demand intelligence", page_icon="📈", layout="wide")


@st.cache_data(show_spinner=False)
def read_artifact(relative, modified):
    """The modification timestamp invalidates cached reads when artifacts change."""
    path = ROOT / relative
    return pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)


def data(relative):
    path = ROOT / relative
    if not path.exists():
        st.error(f"Required artifact is missing: {relative}. Restore the committed reports or run make analysis.")
        st.stop()
    return read_artifact(relative, path.stat().st_mtime_ns).copy()


def table(name):
    return data(f"reports/tables/{name}.csv")


def chart(fig):
    fig.update_layout(
        template="plotly_white",
        height=410,
        margin=dict(l=20, r=20, t=60, b=90),
        font=dict(color="#172B46"),
        paper_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", y=-0.25),
        title_font_size=16,
    )
    st.plotly_chart(fig, width="stretch")


def metrics(frame):
    row = frame.loc[frame.model.eq("lightgbm")].iloc[0]
    for column, label, value in zip(
        st.columns(3),
        ["Mean absolute error", "Root mean squared error", "Weighted error"],
        [f"{row.MAE:,.1f} MW", f"{row.RMSE:,.1f} MW", f"{row.WAPE:.2f}%"],
        strict=True,
    ):
        column.metric(label, value)


st.sidebar.title("GridCast")
st.sidebar.caption("NEW ENGLAND · ISNE")
section = st.sidebar.radio("Explore the research", SECTIONS)
st.sidebar.divider()
st.sidebar.markdown("**24-hour horizon**  \nDaily origins at 00:00 UTC  \nHourly demand in MW")
st.sidebar.caption(
    "Saved research results. Controls filter existing observations and forecasts; no models are retrained."
)
st.title(section)
st.caption("ELECTRICITY DEMAND FORECASTING UNDER UNCERTAINTY")

if section == "Overview":
    st.header("Better average forecasts. Uneven reliability at the extremes.")
    st.write(
        "Can nonlinear models improve next-day electricity demand forecasts while remaining reliable during "
        "high-demand periods? Six approaches are compared before evaluating the selected model on untouched 2024 data."
    )
    metrics(table("test_model_comparison"))
    left, right = st.columns([2, 1])
    with left:
        scores = table("test_model_comparison")
        chart(
            px.bar(
                scores,
                x="model",
                y="MAE",
                color="model",
                text_auto=".1f",
                title="2024 test · MAE (MW)",
                color_discrete_sequence=["#9AABC0", "#087F8C"],
            )
        )
    with right:
        st.subheader("What the evidence says")
        st.write("LightGBM won validation overall and beat the daily seasonal baseline on the final test.")
        coverage = table("interval_conditions").query("level == 0.95").set_index("condition")
        st.metric("95% interval · observed coverage", f"{coverage.loc['overall', 'coverage']:.2%}")
        st.metric("95% interval · extreme demand coverage", f"{coverage.loc['extreme_demand', 'coverage']:.2%}")
        st.warning("Nominal coverage is not guaranteed. Extreme-demand intervals miss more often.")
    st.info(
        "2020–2022 training → Jan–Sep 2023 validation → Oct–Dec 2023 calibration → 2024 final test. "
        "Only LightGBM and the predeclared daily baseline were evaluated on the final test."
    )

elif section == "Demand Explorer":
    history = data("reports/app/history.parquet").reset_index()
    partition = st.selectbox("Observation partition", ["train", "validation", "calibration", "test"])
    frame = history.loc[history.partition.eq(partition)]
    st.caption("Retrospective observations. Weather is an observed three-station proxy, not a future-weather forecast.")
    daily = frame.set_index("target_start_utc").demand_mw.resample("D").agg(["mean", "min", "max"]).reset_index()
    chart(
        px.line(
            daily,
            x="target_start_utc",
            y=["mean", "min", "max"],
            title="Daily demand · UTC days",
            labels={"value": "Demand (MW)", "target_start_utc": "UTC interval start"},
        )
    )
    left, right = st.columns(2)
    with left:
        chart(
            px.line(
                frame.groupby("hour", as_index=False).demand_mw.mean(),
                x="hour",
                y="demand_mw",
                title="Mean demand by local hour",
                labels={"hour": "America/New_York hour", "demand_mw": "MW"},
            )
        )
    with right:
        weather = frame.dropna(subset=["weather_temperature_c"]).copy()
        weather["temperature_bin_c"] = (weather.weather_temperature_c / 2).round() * 2
        relationship = weather.groupby("temperature_bin_c", as_index=False).agg(
            demand_mw=("demand_mw", "mean"), hours=("demand_mw", "size")
        )
        chart(
            px.scatter(
                relationship,
                x="temperature_bin_c",
                y="demand_mw",
                size="hours",
                title="Temperature and demand",
                labels={"demand_mw": "MW"},
            )
        )
    st.caption(
        f"{len(frame):,} hourly observations; {frame.weather_temperature_c.isna().sum():,} missing temperatures "
        "excluded from the weather plot. Local-hour groups preserve daylight-saving repetitions."
    )

elif section == "Forecast":
    period = st.selectbox("Evaluation period", ["2024 final test", "2023 validation"])
    test = period.startswith("2024")
    frame = data("reports/predictions/reserved.parquet" if test else "reports/predictions/validation.parquet")
    if test:
        frame = frame.loc[frame.partition.eq("test")]
    model = st.selectbox("Model", sorted(frame.model.unique()), format_func=lambda x: LABELS[x])
    frame = frame.loc[frame.model.eq(model)]
    origin = st.selectbox("Forecast origin (UTC)", sorted(frame.origin_utc.unique()))
    selected = frame.loc[frame.origin_utc.eq(origin)].sort_values("horizon")
    fig = go.Figure()
    if test and model == "lightgbm":
        level = st.select_slider(
            "Prediction interval", options=[0.5, 0.8, 0.95], value=0.95, format_func=lambda x: f"{x:.0%}"
        )
        bounds = data("reports/predictions/test_intervals.parquet")
        bounds = bounds.loc[bounds.origin_utc.eq(origin) & bounds.level.eq(level)].sort_values("horizon")
        fig.add_trace(go.Scatter(x=bounds.target_start_utc, y=bounds.upper, line=dict(width=0), name="Upper bound"))
        fig.add_trace(
            go.Scatter(
                x=bounds.target_start_utc,
                y=bounds.lower,
                fill="tonexty",
                fillcolor="rgba(8,127,140,.16)",
                line=dict(width=0),
                name=f"{level:.0%} interval",
            )
        )
    for field, label, color in [("actual", "Observed", "#172B46"), ("prediction", "Forecast", "#087F8C")]:
        fig.add_trace(go.Scatter(x=selected.target_start_utc, y=selected[field], name=label, line=dict(color=color)))
    fig.update_layout(yaxis_title="Demand (MW)", xaxis_title="UTC interval start")
    chart(fig)
    st.caption(
        "Observations are shown retrospectively. Forecasts use only information available at their origin, "
        "including a one-hour reporting delay. Intervals exist only for the selected final-test model."
    )
    st.download_button("Download this forecast", selected.to_csv(index=False), "forecast.csv", "text/csv")

elif section == "Model Comparison":
    period = st.selectbox("Evaluation period", ["2023 validation", "2024 final test"])
    prefix = "test_" if period.startswith("2024") else ""
    frame = table(prefix + "model_comparison")
    measure = st.selectbox("Error metric", ["MAE", "RMSE", "WAPE", "MASE"])
    chart(
        px.bar(
            frame.sort_values(measure),
            x="model",
            y=measure,
            color="model",
            text_auto=".2f",
            title=f"{period} · lower is better",
        )
    )
    st.dataframe(frame, hide_index=True, width="stretch")
    conditions = table(prefix + "condition_metrics")
    condition = st.selectbox("Operating condition", sorted(conditions.condition.unique()))
    st.dataframe(conditions.loc[conditions.condition.eq(condition)], hide_index=True, width="stretch")
    st.caption(
        "MAE/RMSE: MW; WAPE: percent; MASE: scaled error. Conditions overlap. Validation selected LightGBM; "
        "the final test compares only that frozen choice with the predeclared daily baseline."
    )

elif section == "Extreme Events":
    st.write("Extreme demand exceeds the training-only 95th percentile: 17,971.4 MW. The threshold stays fixed.")
    frame = table("test_condition_metrics")
    st.dataframe(
        frame.loc[frame.condition.isin(["extreme_demand", "nonextreme_demand"])], hide_index=True, width="stretch"
    )
    intervals = table("interval_conditions").query("condition == 'extreme_demand'")
    chart(
        px.bar(
            intervals,
            x="level",
            y="coverage",
            range_y=[0, 1],
            text_auto=".1%",
            title="Extreme-demand coverage",
        )
    )
    st.warning(
        "High-demand performance remains a limitation. Strong aggregate accuracy does not establish "
        "reliability during system stress; these results were not used to retune the model."
    )
    st.caption(
        "On validation, the harmonic statistical model had lower extreme-demand MAE than LightGBM. "
        "Model ranking depends on the operating condition."
    )

elif section == "Anomalies":
    frame = table("forecast_anomalies")
    frame["target_start_utc"] = pd.to_datetime(frame.target_start_utc, utc=True)
    month = st.selectbox("Month (UTC)", sorted(frame.target_start_utc.dt.strftime("%Y-%m").unique()))
    frame = frame.loc[frame.target_start_utc.dt.strftime("%Y-%m").eq(month)]
    flagged = frame.loc[frame.is_anomaly]
    fig = px.line(frame, x="target_start_utc", y="residual_mw", title="Forecast residuals (MW)")
    fig.add_trace(
        go.Scatter(
            x=flagged.target_start_utc,
            y=flagged.residual_mw,
            mode="markers",
            marker=dict(color="#C75B38", size=8),
            name="Outside 95% interval",
        )
    )
    chart(fig)
    st.metric("Flagged hours in selected month", len(flagged))
    st.caption(
        "Flags indicate forecast surprises, not proven faults or causal events. A flag becomes available "
        "only after its observed demand has been reported. Episodes join consecutive flagged UTC hours."
    )
    episodes = table("anomaly_episodes")
    start = pd.to_datetime(episodes.start_utc, utc=True)
    end = pd.to_datetime(episodes.end_utc, utc=True)
    st.dataframe(
        episodes.loc[(start <= frame.target_start_utc.max()) & (end >= frame.target_start_utc.min())],
        hide_index=True,
        width="stretch",
    )
    st.download_button("Download selected residuals", frame.to_csv(index=False), "anomaly_residuals.csv", "text/csv")

elif section == "Reliability":
    st.write(
        "Symmetric residual intervals use a rolling 90-day calibration window separately for each horizon. "
        "Only residuals whose labels have matured at the forecast origin enter calibration."
    )
    coverage = table("interval_conditions").query("condition == 'overall'")
    fig = px.line(coverage, x="level", y="coverage", markers=True, range_y=[0, 1], title="Nominal vs observed coverage")
    fig.add_trace(go.Scatter(x=[0.5, 0.95], y=[0.5, 0.95], name="Ideal coverage", line=dict(dash="dash")))
    chart(fig)
    rolling = table("rolling_reliability")
    measure = st.selectbox("Rolling diagnostic", ["MAE", "coverage_95", "residual_mean_mw", "residual_variance_mw2"])
    chart(px.line(rolling, x="as_of_utc", y=measure, title="Trailing 30-day diagnostic"))
    st.caption(
        "Early windows are partial (minimum 168 observations). Changing residual behavior is a drift "
        "diagnostic, not a formal alarm. Dependent time-series residuals do not provide an IID coverage guarantee."
    )
    st.dataframe(coverage, hide_index=True, width="stretch")

else:
    st.header("A chronological, auditable forecasting experiment")
    st.markdown("""
**Data.** EIA-930 hourly ISNE demand, 2020–2024; NOAA observations from Boston, Hartford and Providence.
Hourly energy divided by the one-hour interval is mean MW. Source interval ends are converted to UTC interval starts.
Local calendar features use America/New_York; missing weather is retained explicitly.

**Protocol.** Train on 2020–2022; select with nine expanding monthly folds in January–September 2023;
calibrate on October–December 2023; evaluate once on 2024. Every daily 00:00 UTC origin predicts 24 hours.
With the one-hour reporting delay, the newest usable demand interval starts two hours before the origin.

**Models.** Persistence, daily and weekly seasonal baselines; harmonic regression with AR(1) residuals;
LightGBM; and a compact LSTM. LightGBM was selected by validation MAE. The final tree count is fixed at 425.
The final test includes only LightGBM and the predeclared daily seasonal baseline.

**Leakage controls.** Fitting, scaling, stopping decisions and extreme-event thresholds precede final-test evaluation.
Future observed weather is never a predictor. Calibration residuals are admitted only after label availability.
The application reads saved artifacts and never fits models or alters research results.

**Uncertainty and anomalies.** Per-horizon absolute residual quantiles use a finite-sample rank and rolling 90-day
window. Intervals at 50%, 80% and 95% are symmetric. A 95% breach is a forecast anomaly, not proof of a grid fault.

**Limits.** One region, a station-averaged weather proxy, no operational weather forecasts, and a limited model search.
Extreme-demand and some monthly coverage fall below nominal levels.
Results are retrospective, not an operational dispatch system.

**Reproduce.** `make setup` installs pinned dependencies; `make analysis` rebuilds the research outputs;
`make app-data` exports observation history; `make app` starts this console. A clean checkout includes all application
artifacts. See the repository README and saved audit tables for the full protocol.
""")
    st.link_button("Read the project and results", "https://github.com/richithareddyy/GridCast")
