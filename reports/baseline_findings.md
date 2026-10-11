Validation only: January–September 2023; nine expanding monthly folds, 273 daily origins, 24 horizons per origin.

| Model | MAE (MW) | RMSE (MW) | WAPE (%) | MASE | Peak MAE | Extreme MAE |
|---|---:|---:|---:|---:|---:|---:|
| Naive | 2,515.5 | 3,003.1 | 19.46 | 1.876 | 957.8 | 1,663.7 |
| Daily seasonal naive (available) | 924.3 | 1,288.9 | 7.15 | 0.689 | 990.7 | 1,761.2 |
| Weekly seasonal naive | 1,438.3 | 1,987.1 | 11.13 | 1.072 | 1,576.3 | 3,202.4 |
| Harmonic regression + AR(1) | 1,008.5 | 1,283.8 | 7.80 | 0.752 | 873.6 | 1,415.5 |
| LightGBM residual correction | 723.7 | 1,019.2 | 5.60 | 0.539 | 823.2 | 1,662.8 |
| LSTM residual correction | 751.9 | 1,043.4 | 5.82 | 0.560 | 838.4 | 1,618.8 |

**Lowest validation MAE: LightGBM residual correction.** Harmonic regression + AR(1) has 9.1% higher MAE than the stronger seasonal baseline (Daily seasonal naive (available)). This is a point-estimate comparison, not a claim of statistical significance or final-test performance.

WAPE is expressed in percent. MASE uses each fold's available training-only 168-hour scale. Peak means local hours 16–20; extreme means the fixed initial-training 95th percentile. Condition groups overlap. All models are scored on identical labels.

Source: `reports/tables/model_comparison.csv`; individual forecasts are stored in `reports/predictions/validation.parquet`. Calibration and final test do not contribute to these validation results.

LightGBM residual correction: MAE reduction vs daily seasonal = **21.7%** overall, **16.9%** at peak hours and **5.6%** during extreme demand. Negative reductions indicate worse performance.

LSTM residual correction: MAE reduction vs daily seasonal = **18.7%** overall, **15.4%** at peak hours and **8.1%** during extreme demand. Negative reductions indicate worse performance.

Monthly MAE wins: LightGBM residual correction 8; LSTM residual correction 1.

Paired daily MAE differences vs the predeclared daily baseline (negative favors the model):

| Model | Difference (MW) | HAC 95% interval | Holm-adjusted p |
|---|---:|---:|---:|
| Harmonic regression + AR(1) | 84.2 | [5.4, 163.0] | 0.03616 |
| LightGBM residual correction | -200.6 | [-242.5, -158.6] | 3.12e-20 |
| LSTM residual correction | -172.4 | [-217.4, -127.3] | 1.912e-13 |
| Naive | 1591.3 | [1432.5, 1750.1] | 3.58e-85 |
| Weekly seasonal naive | 514.0 | [303.0, 725.0] | 3.609e-06 |

HAC uses seven daily lags; intervals are pointwise asymptotic intervals, while p-values use Holm correction across all five comparisons. These are exploratory validation comparisons, not a correction for model selection or a final-test claim.
