"""Origin-aligned supervised samples, boosted residuals and one compact LSTM."""

from dataclasses import dataclass
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import torch
from torch import nn

from gridcast.baselines import predict_baselines
from gridcast.features import available_history, forecast_features, validate_history
from gridcast.validation import ValidationError


@dataclass
class Samples:
    origins: pd.DatetimeIndex
    x: np.ndarray
    sequence: np.ndarray
    calendar: np.ndarray
    baseline: np.ndarray
    actual: np.ndarray
    columns: list[str]
    delay: int

    def subset(self, mask):
        return Samples(
            self.origins[mask],
            self.x[mask],
            self.sequence[mask],
            self.calendar[mask],
            self.baseline[mask],
            self.actual[mask],
            self.columns,
            self.delay,
        )

    def matured(self, origin):
        # A whole 24-hour training path must be published, including its last hour.
        return self.subset(self.origins + pd.Timedelta(hours=24 + self.delay) <= pd.Timestamp(origin))


def origin_inputs(y, origin, lookback=168, delay=1):
    features = forecast_features(y, origin, delay=delay)
    base = predict_baselines(y, origin, delay=delay).daily.to_numpy()
    features["daily_baseline"] = base
    # Cyclic target calendar is known at issue time. No observed weather enters either learner.
    for column, period in [("hour", 24), ("weekday", 7), ("month", 12)]:
        for name, function in [("sin", np.sin), ("cos", np.cos)]:
            features[f"{column}_{name}"] = function(2 * np.pi * features[column] / period)
    calendar_columns = [
        "hour_sin",
        "hour_cos",
        "weekday_sin",
        "weekday_cos",
        "month_sin",
        "month_cos",
        "weekend",
        "holiday",
    ]
    history = available_history(y.to_frame("y"), origin, delay).y.iloc[-lookback:]
    if len(history) != lookback or not np.isfinite(history).all():
        raise ValidationError("LSTM needs a complete available history window")
    # The neural decoder also sees known daily/weekly templates, supplied separately from calendar.
    decoder = features[calendar_columns].to_numpy(dtype=np.float32)
    sequence = history.to_numpy(dtype=np.float32)[:, None]
    return features, sequence, decoder, base


def build_samples(y: pd.Series, lookback=168, delay=1):
    validate_history(y)
    earliest = (y.index[0] + pd.Timedelta(hours=max(lookback, 168) + delay)).ceil("D")
    latest = (y.index[-1] - pd.Timedelta(hours=23)).floor("D")
    origins = pd.date_range(earliest, latest, freq="24h")
    if len(origins) == 0:
        raise ValidationError("Not enough history for supervised paths")
    rows, sequences, calendars, baselines, actuals = [], [], [], [], []
    for origin in origins:
        f, sequence, calendar, base = origin_inputs(y, origin, lookback, delay)
        rows.append(f.to_numpy(dtype=np.float32))
        sequences.append(sequence)
        calendars.append(calendar)
        baselines.append(base)
        actuals.append(y.reindex(f.index).to_numpy())
    actual = np.asarray(actuals)
    if not np.isfinite(actual).all():
        raise ValidationError("Missing supervised labels; no target imputation")
    return Samples(
        origins,
        np.asarray(rows),
        np.asarray(sequences),
        np.asarray(calendars),
        np.asarray(baselines),
        actual,
        list(f.columns),
        delay,
    )


def inner_split(samples, days):
    if len(samples.origins) < days + 90:
        raise ValidationError("At least 90 training origins plus the inner stopping window are required")
    start = samples.origins[-days]
    train = samples.matured(start)
    valid = samples.subset(samples.origins >= start)
    return train, valid


def tree_data(samples):
    return lgb.Dataset(
        samples.x.reshape(-1, samples.x.shape[-1]),
        label=(samples.actual - samples.baseline).reshape(-1),
        feature_name=samples.columns,
    )


class ResidualLSTM(nn.Module):
    """Encode one available week; decode the entire path without future actuals."""

    def __init__(self, hidden):
        super().__init__()
        self.encoder = nn.LSTM(1, hidden, batch_first=True)
        self.decoder = nn.Sequential(nn.Linear(hidden + 10, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, sequence, context):
        _, (state, _) = self.encoder(sequence)
        repeated = state[-1, :, None, :].expand(-1, 24, -1)
        return self.decoder(torch.cat([repeated, context], dim=-1)).squeeze(-1)


def scale_fit(samples):
    # Repeated windows weight observations by their use as training inputs; never use stopping/test inputs.
    mean = float(samples.sequence.mean(dtype=np.float64))
    std = float(samples.sequence.std(dtype=np.float64))
    if not np.isfinite(std) or std <= 0:
        raise ValidationError("Degenerate neural input scale")
    return mean, std


def neural_tensors(samples, mean, std):
    weekly = samples.x[:, :, samples.columns.index("lag_168")]
    context = np.concatenate(
        [samples.calendar, ((samples.baseline - mean) / std)[..., None], ((weekly - mean) / std)[..., None]], axis=-1
    )
    arrays = [(samples.sequence - mean) / std, context, (samples.actual - samples.baseline) / std]
    if not all(np.isfinite(a).all() for a in arrays):
        raise ValidationError("Nonfinite neural input; unavailable lag_24 is not a neural feature")
    return tuple(torch.tensor(a, dtype=torch.float32) for a in arrays)


def train_neural(train, spec, seed, valid=None, epochs=None):
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(seed)
    model = ResidualLSTM(spec["hidden_size"])
    mean, std = scale_fit(train)
    sequence, context, target = neural_tensors(train, mean, std)
    validation = neural_tensors(valid, mean, std) if valid is not None else None
    optimizer = torch.optim.Adam(model.parameters(), lr=spec["learning_rate"], weight_decay=spec["weight_decay"])
    best, best_epoch, stale, curve = float("inf"), 0, 0, []
    for epoch in range(1, (epochs or spec["max_epochs"]) + 1):
        model.train()
        # Chronological mini-batches; no shuffled split and no recurrent state between samples.
        for first in range(0, len(train.origins), spec["batch_size"]):
            sl = slice(first, first + spec["batch_size"])
            optimizer.zero_grad()
            loss = nn.functional.mse_loss(model(sequence[sl], context[sl]), target[sl])
            if not torch.isfinite(loss):
                raise ValidationError("Nonfinite LSTM training loss")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), spec["gradient_clip"])
            optimizer.step()
        if validation is not None:
            model.eval()
            with torch.no_grad():
                value = float((model(validation[0], validation[1]) - validation[2]).abs().mean()) * std
            curve.append({"epoch": epoch, "stopping_mae_mw": value})
            if value < best:
                best, best_epoch, stale = value, epoch, 0
            else:
                stale += 1
            if stale >= spec["patience"]:
                break
    model.eval()
    return model, mean, std, best_epoch if valid is not None else epochs, curve


class LearnedModels:
    def __init__(self, config):
        self.config = config

    def fit(self, samples, origin, artifact_dir: Path | None = None):
        train = samples.matured(origin)
        inner_train, inner_valid = inner_split(train, self.config["inner_stopping_days"])
        spec = self.config["lightgbm"]
        params = {**spec["parameters"], "seed": self.config["seed"]}
        stopping = lgb.train(
            params,
            tree_data(inner_train),
            num_boost_round=spec["max_rounds"],
            valid_sets=[tree_data(inner_valid)],
            callbacks=[lgb.early_stopping(spec["patience"], verbose=False)],
        )
        rounds = stopping.best_iteration
        self.tree = lgb.train(params, tree_data(train), num_boost_round=rounds)
        deep = self.config["lstm"]
        _, _, _, epochs, curve = train_neural(inner_train, deep, self.config["seed"], valid=inner_valid)
        self.neural, self.mean, self.std, _, _ = train_neural(train, deep, self.config["seed"], epochs=epochs)
        self.fit_origin = pd.Timestamp(origin)
        self.columns = samples.columns
        self.audit = {
            "fit_origin_utc": str(origin),
            "training_origins": len(train.origins),
            "latest_training_origin": str(train.origins[-1]),
            "latest_training_label_available_at": str(train.origins[-1] + pd.Timedelta(hours=24 + train.delay)),
            "inner_training_origins": len(inner_train.origins),
            "inner_last_label_available_at": str(inner_train.origins[-1] + pd.Timedelta(hours=24 + train.delay)),
            "inner_stopping_start": str(inner_valid.origins[0]),
            "inner_stopping_origins": len(inner_valid.origins),
            "tree_rounds": rounds,
            "lstm_epochs": epochs,
            "lstm_stopping_curve": curve,
            "lstm_input_mean": self.mean,
            "lstm_input_std": self.std,
            "lstm_parameters": sum(p.numel() for p in self.neural.parameters()),
            "feature_columns": self.columns,
            "tree_gain_importance": dict(zip(self.columns, self.tree.feature_importance("gain").tolist(), strict=True)),
        }
        if artifact_dir is not None:
            artifact_dir.mkdir(parents=True, exist_ok=True)
            fold = pd.Timestamp(origin).strftime("%Y-%m")
            self.tree.save_model(str(artifact_dir / f"{fold}_lightgbm.txt"))
            weights = {k: v.detach().numpy() for k, v in self.neural.state_dict().items()}
            np.savez(artifact_dir / f"{fold}_lstm.npz", mean=self.mean, std=self.std, **weights)
        return self

    def predict(self, y, origin):
        if pd.Timestamp(origin) < self.fit_origin:
            raise ValidationError("Learned model fit is later than prediction origin")
        spec = self.config
        f, sequence, calendar, base = origin_inputs(y, origin, spec["lookback_hours"], spec["reporting_delay_hours"])
        if list(f.columns) != self.columns:
            raise ValidationError("Learned model feature schema changed")
        sample = Samples(
            pd.DatetimeIndex([origin]),
            f.to_numpy(dtype=np.float32)[None],
            sequence[None],
            calendar[None],
            base[None],
            base[None],
            self.columns,
            spec["reporting_delay_hours"],
        )
        tensors = neural_tensors(sample, self.mean, self.std)
        with torch.no_grad():
            deep = self.neural(tensors[0], tensors[1]).numpy()[0] * self.std + base
        tree = self.tree.predict(f.to_numpy(dtype=np.float32), num_threads=1) + base
        if not np.isfinite(tree).all() or not np.isfinite(deep).all():
            raise ValidationError("Nonfinite learned forecast")
        return {"lightgbm": tree, "lstm": deep}


def runtime_smoke_test():
    """Exercise both native runtimes, catching OpenMP conflicts during setup."""
    torch.set_num_threads(1)
    x = np.arange(400, dtype=float).reshape(100, 4)
    tree = lgb.train(
        {"objective": "regression", "num_threads": 1, "verbosity": -1}, lgb.Dataset(x, label=x[:, 0]), num_boost_round=2
    )
    assert np.isfinite(tree.predict(x, num_threads=1)).all()
    model = ResidualLSTM(4)
    output = model(torch.ones(2, 168, 1), torch.ones(2, 24, 10))
    output.sum().backward()
    assert torch.isfinite(output).all()
    print("LightGBM and PyTorch CPU runtime smoke test passed.")


if __name__ == "__main__":
    runtime_smoke_test()
