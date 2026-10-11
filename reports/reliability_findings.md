Final test: 2024, 8,784 hourly targets; LightGBM with 425 frozen trees and monthly expanding refits.

Point MAE: **705.2 MW** vs **907.8 MW** for daily seasonal naive; RMSE **1,008.0 MW**, WAPE **5.41%**, MASE **0.524**.

| Nominal | Coverage | Mean width (MW) | Interval score (MW) | Peak coverage | Extreme-demand coverage |
|---|---:|---:|---:|---:|---:|
| 50% | 51.25% | 1,152.2 | 2,313.7 | 50.55% | 25.15% |
| 80% | 78.85% | 2,268.5 | 3,447.9 | 77.87% | 58.19% |
| 95% | 93.84% | 3,939.9 | 4,997.4 | 93.77% | 84.50% |

At nominal 95%, coverage is **93.84% overall** and **84.50% during extreme demand**. Aggregate calibration does not imply reliable coverage in the high-demand tail.

There are **541** hourly 95% interval violations in **175** contiguous UTC episodes. These are forecast anomalies, not verified outages or failures.

The same horizon-wise calibration window is used for every condition; no post-test recalibration or interval-width tuning is performed. Condition groups overlap, and rare-event estimates have limited effective sample size. Coverage is empirical, not a distribution-free guarantee.

Full trailing-window MAE ranges from **437.4 to 1,292.7 MW**; 95% coverage ranges from **77.2% to 100.0%**. These descriptive changes are not a formal causal or structural-break diagnosis.
