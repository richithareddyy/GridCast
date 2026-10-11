Validation only: January–September 2023; nine expanding monthly folds, 273 daily origins, 24 horizons per origin.

| Model | MAE (MW) | RMSE (MW) | WAPE (%) | MASE | Peak MAE | Extreme MAE |
|---|---:|---:|---:|---:|---:|---:|
| Naive | 2,515.5 | 3,003.1 | 19.46 | 1.876 | 957.8 | 1,663.7 |
| Daily seasonal naive (available) | 924.3 | 1,288.9 | 7.15 | 0.689 | 990.7 | 1,761.2 |
| Weekly seasonal naive | 1,438.3 | 1,987.1 | 11.13 | 1.072 | 1,576.3 | 3,202.4 |
| Harmonic regression + AR(1) | 1,008.5 | 1,283.8 | 7.80 | 0.752 | 873.6 | 1,415.5 |

**Lowest validation MAE: Daily seasonal naive (available).** Harmonic regression + AR(1) has 9.1% higher MAE than the stronger seasonal baseline (Daily seasonal naive (available)). This is a point-estimate comparison, not a claim of statistical significance or final-test performance.

WAPE is expressed in percent. MASE uses each fold's available training-only 168-hour scale. Peak means local hours 16–20; extreme means the fixed initial-training 95th percentile. Condition groups overlap. All models are scored on identical labels.

Source: `reports/tables/model_comparison.csv`; individual forecasts are stored in `reports/predictions/validation.parquet`. Calibration and final test remain unscored.
