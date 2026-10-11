import copy
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest
import torch

from gridcast.data import load_config
from gridcast.evaluation import paired_comparisons
from gridcast.learning import (
    LearnedModels,
    ResidualLSTM,
    build_samples,
    inner_split,
    neural_tensors,
    origin_inputs,
    scale_fit,
)
from gridcast.validation import ValidationError

ROOT = Path(__file__).parents[1]


@pytest.fixture(scope="module")
def study():
    index = pd.date_range("2021-01-01", periods=24 * 160, freq="h", tz="UTC")
    n = np.arange(len(index))
    y = pd.Series(12000 + 800 * np.sin(2 * np.pi * n / 24) + 300 * np.cos(2 * np.pi * n / 168) + n / 10, index=index)
    config = load_config(ROOT, "evaluation.yaml")["learning"]
    config["inner_stopping_days"] = 10
    config["lightgbm"]["max_rounds"] = 8
    config["lightgbm"]["patience"] = 3
    config["lstm"]["max_epochs"] = 2
    config["lstm"]["hidden_size"] = 4
    return y, build_samples(y), config


def test_supervised_alignment_and_maturity_purges_entire_path(study):
    y, samples, _ = study
    origin = samples.origins[-10]
    known = samples.matured(origin)
    assert (known.origins + pd.Timedelta(hours=25) <= origin).all()
    assert known.origins[-1] == origin - pd.Timedelta(days=2)
    train, stop = inner_split(known, 10)
    assert train.origins[-1] + pd.Timedelta(hours=25) <= stop.origins[0]
    np.testing.assert_array_equal(samples.actual[0], y.loc[samples.origins[0] :].iloc[:24])
    np.testing.assert_array_equal(
        samples.sequence[0, :, 0],
        y.loc[y.index < samples.origins[0] - pd.Timedelta(hours=1)].iloc[-168:].astype("float32"),
    )


def test_origin_inputs_ignore_unpublished_and_future_demand(study):
    y, samples, _ = study
    origin = samples.origins[100]
    before = origin_inputs(y, origin)
    changed = y.copy()
    changed.loc[changed.index >= origin - pd.Timedelta(hours=1)] = 1e9
    after = origin_inputs(changed, origin)
    pd.testing.assert_frame_equal(before[0], after[0])
    for a, b in zip(before[1:], after[1:], strict=True):
        np.testing.assert_array_equal(a, b)
    assert not any("weather" in c for c in before[0].columns)


def test_fold_selection_and_scaling_ignore_future_sample_values(study):
    _, samples, _ = study
    cutoff = samples.origins[-10]
    altered = copy.deepcopy(samples)
    future = altered.origins + pd.Timedelta(hours=25) > cutoff
    altered.x[future] = 1e9
    altered.sequence[future] = 1e9
    altered.actual[future] = 1e9
    original = samples.matured(cutoff)
    changed = altered.matured(cutoff)
    np.testing.assert_array_equal(original.actual, changed.actual)
    np.testing.assert_array_equal(original.x, changed.x)
    assert scale_fit(original) == scale_fit(changed)


def test_neural_scaling_and_decoder_exclude_unknown_target_lag(study):
    _, samples, _ = study
    subset = samples.subset(np.arange(100))
    mean, std = scale_fit(subset)
    sequence, context, target = neural_tensors(subset, mean, std)
    assert sequence.shape == (100, 168, 1)
    assert context.shape == (100, 24, 10)
    assert target.shape == (100, 24)
    assert abs(float(sequence.mean())) < 1e-5
    assert np.isnan(subset.x[:, -1, subset.columns.index("lag_24")]).all()
    assert torch.isfinite(context).all()


def test_learners_determinism_cutoff_and_saved_models(study, tmp_path):
    y, samples, config = study
    origin = samples.origins[-2]
    first = LearnedModels(config).fit(samples, origin, tmp_path)
    second = LearnedModels(config).fit(samples, origin)
    assert first.audit == second.audit
    prediction = first.predict(y, origin)
    for model, values in second.predict(y, origin).items():
        np.testing.assert_array_equal(prediction[model], values)
        assert len(values) == 24 and np.isfinite(values).all()
    changed = y.copy()
    changed.loc[changed.index >= origin - pd.Timedelta(hours=1)] = 1e9
    for model, values in first.predict(changed, origin).items():
        np.testing.assert_array_equal(prediction[model], values)
    with pytest.raises(ValidationError, match="later than"):
        first.predict(y, origin - pd.Timedelta(days=1))
    # Reload persisted parameters and scales, then verify predictions exactly.
    first.tree = lgb.Booster(model_file=str(next(tmp_path.glob("*.txt"))))
    weights = np.load(next(tmp_path.glob("*.npz")))
    restored = ResidualLSTM(config["lstm"]["hidden_size"])
    restored.load_state_dict({k: torch.tensor(weights[k]) for k in restored.state_dict()})
    restored.eval()
    first.neural = restored
    first.mean, first.std = float(weights["mean"]), float(weights["std"])
    for model, values in first.predict(y, origin).items():
        np.testing.assert_array_equal(prediction[model], values)


def test_hac_comparison_pairs_origins_and_adjusts_multiple_tests():
    origins = pd.date_range("2023-01-01", periods=40, tz="UTC")
    rows = []
    for i, origin in enumerate(origins):
        for model, error in [("daily", 10), ("better", 5 + np.sin(i)), ("worse", 15 + np.cos(i))]:
            rows.append({"origin_utc": origin, "model": model, "actual": 100.0, "prediction": 100 - error})
    frame = pd.DataFrame(rows)
    scores = paired_comparisons(frame).set_index("model")
    assert scores.loc["better", "ci95_high"] < 0
    assert scores.loc["worse", "ci95_low"] > 0
    assert (scores.p_holm >= scores.p_value).all()
    with pytest.raises(ValidationError, match="matched"):
        paired_comparisons(frame.iloc[1:])
