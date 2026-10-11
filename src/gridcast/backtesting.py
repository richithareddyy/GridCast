"""Expanding monthly fits and daily 24-hour validation paths. Holdouts stay sealed."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from gridcast.baselines import predict_baselines
from gridcast.data import load_config, sha256, write_json
from gridcast.evaluation import condition_scores, mase_scale, score_groups
from gridcast.features import available_history, forecast_features, validate_history
from gridcast.models import HarmonicAR1
from gridcast.validation import ValidationError

MODEL_LABELS = {
    "naive": "Naive",
    "daily": "Daily seasonal naive (available)",
    "weekly": "Weekly seasonal naive",
    "dhr_ar1": "Harmonic regression + AR(1)",
    "lightgbm": "LightGBM residual correction",
    "lstm": "LSTM residual correction",
}


def monthly_folds(evaluation: dict):
    bounds = evaluation["periods"]["validation"]
    start, end = pd.Timestamp(bounds["start"]), pd.Timestamp(bounds["end"])
    if start != start.normalize() or end != end.normalize() or start >= end:
        raise ValidationError("Validation bounds must be ordered UTC midnight boundaries")
    if str(start.tz) != "UTC" or str(end.tz) != "UTC":
        raise ValidationError("Validation bounds must use UTC")
    cursor = start
    while cursor < end:
        stop = min(cursor + pd.offsets.MonthBegin(1), end)
        yield cursor.strftime("%Y-%m"), cursor, stop
        cursor = stop


def load_development(root: Path, evaluation: dict):
    """Only two named development files can be loaded, never calibration/test."""
    parts = []
    for name in ["train", "validation"]:
        frame = pd.read_parquet(root / f"data/processed/{name}.parquet")
        bounds = evaluation["periods"][name]
        expected = pd.date_range(bounds["start"], bounds["end"], inclusive="left", freq="h")
        if not frame.index.equals(expected):
            raise ValidationError(f"{name} artifact has incorrect bounds, ordering or missing rows")
        parts.append(frame)
    joined = pd.concat(parts)
    validate_history(joined.demand_mw)
    return joined


def backtest(development: pd.DataFrame, evaluation: dict, artifact_dir=None):
    settings = evaluation["forecast"]
    horizon = settings["horizon_hours"]
    delay = settings["assumed_reporting_delay_hours"]
    if horizon != 24 or settings["origin_stride_hours"] != 24 or settings["origin_hour_utc"] != 0:
        raise ValidationError("This frozen protocol implements daily 00 UTC, 24-hour paths")
    y = development.demand_mw
    validate_history(y)
    records, feature_frames, audits, fits = [], [], [], []
    samples = None
    if "learning" in evaluation:
        from gridcast.learning import LearnedModels, build_samples

        if evaluation["learning"]["reporting_delay_hours"] != delay:
            raise ValidationError("Learner and evaluation reporting delays disagree")
        print("Building origin-aligned supervised paths (development partitions only)", flush=True)
        samples = build_samples(y, evaluation["learning"]["lookback_hours"], delay)

    for fold, start, stop in monthly_folds(evaluation):
        train = available_history(y.to_frame("y"), start, delay).y
        scale = mase_scale(train, evaluation["mase"]["seasonal_lag_hours"])
        model = HarmonicAR1(evaluation["statistical_model"]).fit(train)
        fits.append(
            {
                "fold": fold,
                "fit_origin_utc": str(start),
                "last_training_interval_start_utc": str(model.fit_end),
                "latest_label_available_at": str(model.fit_end + pd.Timedelta(hours=delay + 1)),
                "training_rows": model.fit_rows,
                "mase_scale": scale,
                "rho": model.rho,
                "iterations": model.iterations,
                "converged": True,
                "coefficients": model.coefficients.to_dict(),
            }
        )
        learners = None
        if samples is not None:
            print(f"Fold {fold}: fitting LightGBM and LSTM", flush=True)
            learners = LearnedModels(evaluation["learning"]).fit(samples, start, artifact_dir)
            fits[-1]["learning"] = learners.audit
            print(
                f"  Selected {learners.audit['tree_rounds']} trees, {learners.audit['lstm_epochs']} epochs", flush=True
            )
        print(f"Fold {fold}: {len(train):,} training hours, AR(1)={model.rho:.4f}", flush=True)
        for origin in pd.date_range(start, stop, freq="24h", inclusive="left"):
            history = available_history(y.to_frame("y"), origin, delay).y
            # Pass only known observations to all prediction routines, even though they also guard cutoffs.
            base = predict_baselines(history, origin, horizon, delay)
            model_predictions = base[["naive", "daily", "weekly"]].copy()
            model_predictions["dhr_ar1"] = model.predict(history, origin, horizon, delay)
            if learners is not None:
                for name, values in learners.predict(history, origin).items():
                    model_predictions[name] = values
            features = forecast_features(history, origin, horizon, delay)
            if learners is not None:
                from gridcast.learning import origin_inputs

                features = origin_inputs(history, origin, evaluation["learning"]["lookback_hours"], delay)[0]
            features["origin_utc"] = origin
            features["fold"] = fold
            features.index.name = "target_start_utc"
            feature_frames.append(features)
            audits.append(
                {
                    "origin_utc": origin,
                    "fold": fold,
                    "fit_end_utc": model.fit_end,
                    "latest_history_start_utc": history.index[-1],
                    "latest_history_available_at": history.index[-1] + pd.Timedelta(hours=delay + 1),
                    "daily_48h_fallback_count": int(
                        (base.index - base.daily_source_utc).eq(pd.Timedelta(hours=48)).sum()
                    ),
                    "unavailable_target_lag_24_count": int(features.lag_24.isna().sum()),
                }
            )
            # Actuals and observed future weather join only AFTER all forecasts and predictors exist.
            context = development.reindex(model_predictions.index)
            for model_name in model_predictions.columns:
                frame = pd.DataFrame(
                    {
                        "origin_utc": origin,
                        "target_start_utc": model_predictions.index,
                        "target_end_utc": model_predictions.index + pd.Timedelta(hours=1),
                        "horizon": np.arange(1, horizon + 1),
                        "fold": fold,
                        "model": model_name,
                        "prediction": model_predictions[model_name].to_numpy(),
                        "actual": context.demand_mw.to_numpy(),
                        "mase_scale": scale,
                        "latest_history_start_utc": history.index[-1],
                    }
                )
                for column in ["weekend", "holiday", "peak_hour", "month", "weather_temperature_c"]:
                    frame[column] = context[column].to_numpy()
                records.append(frame)
    predictions = pd.concat(records, ignore_index=True)
    if predictions.duplicated(["model", "origin_utc", "horizon"]).any():
        raise ValidationError("Repeated forecast key")
    return predictions, pd.concat(feature_frames), pd.DataFrame(audits), fits


def results_text(overall: pd.DataFrame, conditions: pd.DataFrame, predictions: pd.DataFrame):
    table = overall.set_index("model")
    peak = conditions.loc[conditions.condition.eq("peak_hours")].set_index("model").MAE
    extreme = conditions.loc[conditions.condition.eq("extreme_demand")].set_index("model").MAE
    lines = [
        "Validation only: January–September 2023; nine expanding monthly folds, "
        f"{predictions.origin_utc.nunique()} daily origins, 24 horizons per origin.",
        "",
        "| Model | MAE (MW) | RMSE (MW) | WAPE (%) | MASE | Peak MAE | Extreme MAE |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for model, label in MODEL_LABELS.items():
        if model not in table.index:
            continue
        row = table.loc[model]
        lines.append(
            f"| {label} | {row.MAE:,.1f} | {row.RMSE:,.1f} | {row.WAPE:.2f} | "
            f"{row.MASE:.3f} | {peak[model]:,.1f} | {extreme[model]:,.1f} |"
        )
    best = table.MAE.idxmin()
    seasonal_best = table.loc[["daily", "weekly"], "MAE"].idxmin()
    delta = (table.loc["dhr_ar1", "MAE"] / table.loc[seasonal_best, "MAE"] - 1) * 100
    direction = "higher" if delta >= 0 else "lower"
    lines += [
        "",
        f"**Lowest validation MAE: {MODEL_LABELS[best]}.** Harmonic regression + AR(1) has "
        f"{abs(delta):.1f}% {direction} MAE than the stronger seasonal baseline "
        f"({MODEL_LABELS[seasonal_best]}). This is a point-estimate comparison, "
        "not a claim of statistical significance or final-test performance.",
        "",
        "WAPE is expressed in percent. MASE uses each fold's available training-only "
        "168-hour scale. Peak means local hours 16–20; extreme means the fixed initial-training "
        "95th percentile. Condition groups overlap. All models are scored on identical labels.",
        "",
        "Source: `reports/tables/model_comparison.csv`; individual forecasts are stored in "
        "`reports/predictions/validation.parquet`. Calibration and final test do not contribute "
        "to these validation results.",
    ]
    for name in ["lightgbm", "lstm"]:
        if name in table.index:
            improvement = 100 * (1 - table.loc[name, "MAE"] / table.loc["daily", "MAE"])
            peak_change = 100 * (1 - peak[name] / peak["daily"])
            extreme_change = 100 * (1 - extreme[name] / extreme["daily"])
            lines.append(
                f"\n{MODEL_LABELS[name]}: MAE reduction vs daily seasonal = **{improvement:.1f}%** overall, "
                f"**{peak_change:.1f}%** at peak hours and **{extreme_change:.1f}%** during extreme demand. "
                "Negative reductions indicate worse performance."
            )
    fold_winners = (
        predictions.assign(loss=(predictions.actual - predictions.prediction).abs())
        .groupby(["fold", "model"])
        .loss.mean()
        .unstack()
        .idxmin(axis=1)
        .value_counts()
    )
    lines.append("\nMonthly MAE wins: " + "; ".join(f"{MODEL_LABELS[k]} {v}" for k, v in fold_winners.items()) + ".")
    return "\n".join(lines) + "\n"


def run(root: Path):
    evaluation = load_config(root, "evaluation.yaml")
    development = load_development(root, evaluation)
    predictions, features, origin_audit, fits = backtest(development, evaluation, root / "reports/models")
    return write_outputs(root, evaluation, predictions, features, origin_audit, fits)


def write_outputs(root, evaluation, predictions, features, origin_audit, fits):
    thresholds = json.loads((root / "reports/frozen_thresholds.json").read_text())
    for directory in ["reports/predictions", "reports/tables", "reports/figures"]:
        (root / directory).mkdir(parents=True, exist_ok=True)
    predictions.to_parquet(root / "reports/predictions/validation.parquet", index=False)
    features.to_parquet(root / "data/interim/validation_features.parquet")
    origin_audit.to_csv(root / "reports/tables/origin_audit.csv", index=False)
    overall = score_groups(predictions, ["model"])
    conditions = condition_scores(predictions, thresholds)
    outputs = {
        "model_comparison": overall,
        "fold_metrics": score_groups(predictions, ["fold", "model"]),
        "horizon_metrics": score_groups(predictions, ["horizon", "model"]),
        "condition_metrics": conditions,
    }
    from gridcast.evaluation import paired_comparisons

    outputs["paired_comparisons"] = paired_comparisons(predictions, evaluation["comparison"]["block_days"])
    for name, table in outputs.items():
        table.to_csv(
            root / f"reports/tables/{name}.csv",
            index=False,
            float_format="%.12g" if name == "paired_comparisons" else "%.9f",
        )
    write_json(root / "reports/statistical_fits.json", fits)
    write_json(
        root / "reports/backtest_summary.json",
        {
            "scope": "validation_only",
            "folds": len(fits),
            "origins": int(predictions.origin_utc.nunique()),
            "hours_per_model": int(predictions.groupby("model").size().iloc[0]),
            "prediction_rows": len(predictions),
            "feature_rows": len(features),
            "models": MODEL_LABELS,
            "best_validation_mae_model": str(overall.loc[overall.MAE.idxmin(), "model"]),
            "daily_latency_fallbacks": int(origin_audit.daily_48h_fallback_count.sum()),
            "all_fits_converged": all(fit["converged"] for fit in fits),
            "evaluation_config_sha256": sha256(root / "configs/evaluation.yaml"),
            "thresholds_sha256": sha256(root / "reports/frozen_thresholds.json"),
            "input_hashes": {name: sha256(root / f"data/processed/{name}.parquet") for name in ["train", "validation"]},
        },
    )
    best = str(overall.loc[overall.MAE.idxmin(), "model"])
    write_json(
        root / "reports/model_selection.json",
        {
            "selected_model": best,
            "frozen_lightgbm_parameters": evaluation["learning"]["lightgbm"]["parameters"],
            "frozen_boost_rounds": int(np.median([fit["learning"]["tree_rounds"] for fit in fits])),
            "deployment_round_rule": "median selected rounds across validation folds; fixed before calibration/test",
            "seed": evaluation["learning"]["seed"],
            "lookback_hours": evaluation["learning"]["lookback_hours"],
            "reporting_delay_hours": evaluation["forecast"]["assumed_reporting_delay_hours"],
            "rule": "lowest overall validation MAE among the six predefined model procedures",
            "selection_period": evaluation["periods"]["validation"],
            "reserved_outcomes_used_for_selection": False,
            "evaluation_config_sha256": sha256(root / "configs/evaluation.yaml"),
            "validation_predictions_sha256": sha256(root / "reports/predictions/validation.parquet"),
            "caveat": "validation-selected winner; final-test generalization is not yet established",
        },
    )
    from gridcast.visualization import backtest_figures, learning_figures

    backtest_figures(predictions, outputs, root)
    learning_figures(outputs["paired_comparisons"], fits, root)
    findings = results_text(overall, conditions, predictions)
    comparison = outputs["paired_comparisons"]
    findings += "\nPaired daily MAE differences vs the predeclared daily baseline (negative favors the model):\n\n"
    findings += "| Model | Difference (MW) | HAC 95% interval | Holm-adjusted p |\n|---|---:|---:|---:|\n"
    for row in comparison.itertuples():
        findings += (
            f"| {MODEL_LABELS[row.model]} | {row.mean_mae_difference_mw:.1f} | "
            f"[{row.ci95_low:.1f}, {row.ci95_high:.1f}] | {row.p_holm:.4g} |\n"
        )
    findings += (
        "\nHAC uses seven daily lags; intervals are pointwise asymptotic intervals, while p-values "
        "use Holm correction across all five comparisons. These are exploratory validation comparisons, "
        "not a correction for model selection or a final-test claim.\n"
    )
    (root / "reports/baseline_findings.md").write_text(findings)
    readme = root / "README.md"
    contents = readme.read_text()
    start, end = "<!-- BEGIN BACKTEST RESULTS -->", "<!-- END BACKTEST RESULTS -->"
    if start in contents:
        before, rest = contents.split(start, 1)
        _, after = rest.split(end, 1)
        readme.write_text(before + start + "\n" + findings + end + after)
    print(f"Backtest complete: {len(predictions):,} predictions; calibration and test not loaded.")
    return predictions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    run(parser.parse_args().root.resolve())


if __name__ == "__main__":
    main()
