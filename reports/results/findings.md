**LightGBM had the lowest validation MAE.** On the final test, its MAE was **705.2 MW**, compared with **907.8 MW** for Daily Seasonal Naive (22.3% lower). Only the selected model and the predeclared daily baseline were evaluated on the final test.

| Model | Validation MAE (MW) | RMSE (MW) | WAPE (%) | MASE |
|---|---:|---:|---:|---:|
| LightGBM | 723.7 | 1,019.2 | 5.60 | 0.539 |
| LSTM | 751.9 | 1,043.4 | 5.82 | 0.560 |
| Daily | 924.3 | 1,288.9 | 7.15 | 0.689 |
| Harmonic | 1,008.5 | 1,283.8 | 7.80 | 0.752 |
| Weekly | 1,438.3 | 1,987.1 | 11.13 | 1.072 |
| Naive | 2,515.5 | 3,003.1 | 19.46 | 1.876 |

The ranking changes under stress: **Harmonic** had the lowest validation extreme-demand MAE (**1,415.5 MW**). The threshold is the initial-training 95th percentile, **17,971.4 MW**, held fixed for all models.

| Nominal interval | Test coverage | Mean width (MW) | Extreme-demand coverage |
|---|---:|---:|---:|
| 50% | 51.25% | 1,152.2 | 25.15% |
| 80% | 78.85% | 2,268.5 | 58.19% |
| 95% | 93.84% | 3,939.9 | 84.50% |

The test contains **541 flagged hours** in **175 contiguous UTC episodes**. The largest flagged absolute residual occurred at **2024-06-18 23:00:00+00:00**: actual minus forecast was **6,410.3 MW**. These are unusual demand observations relative to the forecast, not verified outages or grid failures.

Overall coverage conceals undercoverage during extreme demand. No test-based retuning or causal explanation of anomalies is used. Full metrics, condition slices, interval scores and the diagnostic case selection are in the linked reports.
