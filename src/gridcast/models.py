"""Dynamic harmonic regression with feasible-GLS AR(1) errors."""

import numpy as np
import pandas as pd
import statsmodels.api as sm

from gridcast.features import available_history, calendar_features, validate_history
from gridcast.validation import ValidationError


def harmonic_design(index: pd.DatetimeIndex, specification: dict):
    calendar = calendar_features(index)
    local = index.tz_convert("America/New_York")
    daily_phase = local.hour.to_numpy() / 24
    weekly_phase = (local.dayofweek.to_numpy() + daily_phase) / 7
    annual_phase = (local.dayofyear.to_numpy() - 1 + daily_phase) / np.where(local.is_leap_year, 366, 365)
    x = pd.DataFrame({"intercept": np.ones(len(index))}, index=index)
    reference = pd.Timestamp(specification["linear_trend_reference"])
    x["trend_years"] = (index - reference).total_seconds() / (365.2425 * 86400)
    x["weekend"] = calendar.weekend.astype(float)
    if specification["holiday"]:
        x["holiday"] = calendar.holiday.astype(float)
    for name, phase, order in [
        ("daily", daily_phase, specification["daily_fourier_order"]),
        ("weekly", weekly_phase, specification["weekly_fourier_order"]),
        ("annual", annual_phase, specification["annual_fourier_order"]),
    ]:
        for k in range(1, order + 1):
            for wave, function in [("sin", np.sin), ("cos", np.cos)]:
                col = f"{name}_{wave}_{k}"
                x[col] = function(2 * np.pi * k * phase)
                if name == "daily" and specification["weekend_daily_interactions"]:
                    x[f"{col}_weekend"] = x[col] * x.weekend
    return x


class HarmonicAR1:
    """Fit coefficients monthly; condition forecasts on the last *available* residual."""

    def __init__(self, specification: dict):
        self.specification = specification

    def fit(self, history: pd.Series):
        validate_history(history)
        if len(history) < 336 or not np.isfinite(history).all():
            raise ValidationError("Statistical model needs complete finite training history")
        x = harmonic_design(history.index, self.specification)
        if np.linalg.matrix_rank(x.to_numpy()) < x.shape[1]:
            raise ValidationError("Rank-deficient harmonic design")
        model = sm.GLSAR(history.to_numpy(), x.to_numpy(), rho=1)
        result = model.iterative_fit(
            maxiter=self.specification["max_iterations"], rtol=self.specification["relative_tolerance"]
        )
        if not result.converged:
            raise ValidationError("GLSAR failed to converge; no failed fit is scored")
        self.rho = float(model.rho[0])
        if not np.isfinite(self.rho) or abs(self.rho) >= 1:
            raise ValidationError("Nonstationary AR(1) errors; inspect fit")
        self.coefficients = pd.Series(result.params, index=x.columns)
        self.fit_end = history.index[-1]
        self.fit_rows = len(history)
        self.iterations = int(result.iter)
        return self

    def predict(self, series: pd.Series, origin, horizon=24, delay=1):
        validate_history(series)
        origin = pd.Timestamp(origin)
        history = available_history(series.to_frame("y"), origin, delay).y
        if history.empty or self.fit_end > history.index[-1]:
            raise ValidationError("Fit used observations unavailable at prediction origin")
        last = history.iloc[-1]
        if not np.isfinite(last):
            raise ValidationError("Missing latest state observation")
        targets = pd.date_range(origin, periods=horizon, freq="h")
        last_x = harmonic_design(history.index[-1:], self.specification)
        last_residual = float(last - (last_x.to_numpy() @ self.coefficients.to_numpy())[0])
        x = harmonic_design(targets, self.specification)
        steps = (targets - history.index[-1]).total_seconds() / 3600
        # The first target is two elapsed steps from the last observation when delay=1.
        prediction = x.to_numpy() @ self.coefficients.to_numpy() + self.rho**steps * last_residual
        if not np.isfinite(prediction).all():
            raise ValidationError("Nonfinite statistical forecast")
        return pd.Series(prediction, index=targets, name="dhr_ar1")
