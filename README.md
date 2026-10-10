# GridCast
### Electricity Demand Forecasting Under Uncertainty

## Question

**How accurately can short-term electricity demand be forecast, how uncertain are those forecasts,
and when do sophisticated models actually outperform strong seasonal baselines?**
A secondary question asks whether residuals identify unusual demand behavior.

**Milestone 1: data pipeline and EDA complete, awaiting review.** There are no forecasting scores
or trained models yet. This is a data science project focused on temporal evaluation, uncertainty
and honest comparison.

## Dataset

Selected: **EIA-930, New England balancing authority (ISNE), reported hourly Demand, 2020–2024**.
The original operator reports to the federal statistical provider; no third-party mirror is used.
EIA offers a credential-free workbook with explicit UTC timestamps, reported/adjusted/imputed fields
and a usable five-year history. [Download](https://www.eia.gov/electricity/gridmonitor/knownissues/xls/ISNE.xlsx),
[Grid Monitor](https://www.eia.gov/electricity/gridmonitor/dashboard/electric_overview/US48/US48).

Three candidates were investigated:

| Candidate | Verified scope, target and clock | Access, missingness and weather | Decision |
|---|---|---|---|
| EIA-930 / ISNE | New England; hourly demand; workbook units MWh; hour-ending UTC plus Eastern local metadata. Original EIA series begin July 2015; this study uses 2020–2024. | Workbook downloaded without credentials. Study has no missing/duplicate UTC intervals, missing demand, nonpositive demand or imputed demand. NOAA airports can be aligned in UTC. | **Selected:** explicit time axis and reproducible public access. |
| ISO New England SMD archive | New England/control area and eight zones; hourly `System_Load` and `RT_Demand` in MW; local hour-ending labels. Archive listing includes 2016–2026, with current year partial. | Annual public workbooks, with observed dry-bulb/dew-point weather in °F. Template notes explicitly synthesize the spring hour and average repeated autumn hours. Other missingness not measured because rejected before full ingestion. | Useful provider data, but DST preprocessing loses real-hour information needed here. |
| PJM Data Miner, metered load | PJM load areas; hourly MW; both interval-beginning UTC and Eastern Prevailing Time. Public feed inspected; historical earliest bound not independently verified. | Public UI; automated queries require an account. Provider restricts redistribution without membership. Missingness not measured. NOAA alignment is feasible, but the large footprint needs spatial weighting. | Not selected for an account-free, publicly reproducible portfolio. |

Sources: [EIA history/access](https://www.eia.gov/todayinenergy/detail.php?id=40993),
[ISO-NE archive](https://www.iso-ne.com/isoexpress/web/reports/pricing/-/tree/zone-info),
[ISO-NE template, Notes sheet](https://www.iso-ne.com/static-assets/documents/2016/08/smd_hourly_template_v0.xlsx),
[PJM feed](https://dataminer2.pjm.com/feed/hrl_load_metered),
[PJM access/redistribution terms](https://www.pjm.com/markets-and-operations/etools/data-miner-2).
Unverified candidates are not described as complete or gap-free.

**Study dimensions:** 43,848 hourly rows, 34 processed columns, plus the UTC index.
Interval starts span `2020-01-01 00:00Z` through `2024-12-31 23:00Z`.
The full downloaded workbook has 98,856 hourly rows; selection of the study window is logged.
These counts come from [data_quality.json](reports/data_quality.json).

**Target and provenance.** The workbook's Notes sheet labels `Demand` as MWh.
For each one-hour interval, energy divided by one hour gives average demand in **MW**;
`demand_mwh_reported` preserves the source, and `demand_mw` is the target.
Adjusted/imputed columns are retained for auditing, never silently substituted.
Workbook definitions are extracted into [source_definitions.json](reports/source_definitions.json).
All raw files are read-only, unmodified snapshots. [sources.lock.json](configs/sources.lock.json)
records URL, retrieval timestamp, size, SHA-256 and available HTTP metadata.

**Time handling.** EIA's UTC hour ending is shifted back exactly one elapsed hour to the interval
start; local calendar features use `America/New_York`. The local hour-end metadata is cross-checked.
There are zero unexpected UTC duplicates/gaps. All five 23-hour spring days and five 25-hour autumn
days are preserved, along with 48 UTC leap-day hours. The five repeated local hours are distinct
UTC observations. UTC year boundaries leave partial local days at the two study edges; these are
reported, not padded. No timezone-naive joins, DST averaging, invented hours or dropped leap days.

**Weather.** NOAA NCEI ISD observations from Boston Logan (`72509014739`), Hartford Bradley
(`72508014740`) and Providence/T. F. Green (`72507014765`) are downloaded from the
[NOAA NODD archive](https://noaa-global-hourly-pds.s3.amazonaws.com/index.html).
NOAA [documents the archive transition](https://www.nesdis.noaa.gov/news/service-location-change-integrated-surface-data-global-hourly);
ISD is suitable for this historical 2020–2024 study, but is no longer updated beyond August 2025.

We decode temperature/dew point from tenths °C, accept QC codes 0/1/4/5 and instantaneous
FM-15/FM-16 reports, and retain exclusions/flags with source row lineage in interim Parquet files.
Simultaneous reports are reduced by median, then observations are averaged within each UTC hour.
The equal-weight three-station mean requires all three stations: **149 temperature hours and
226 dew-point hours remain missing**, with no interpolation. This southern New England proxy is
not a population-weighted regional temperature. Station IDs/coordinates and documented names
are checked; metadata renames do not silently change stations. [NOAA format/QC definitions](https://www.ncei.noaa.gov/pub/data/noaa/isd-format-document.pdf).

## Forecasting Setup

**Primary task: forecast the next 24 hourly values**, issued daily at 00:00 UTC.
Horizon `h=1..24` targets average demand during `[origin+h−1h, origin+h)`.
This is a 24-hour path, not 24 separate one-step forecasts with actuals revealed inside the path.
The final horizon is 24 hours ahead.

We assume one hour of reporting delay: usable demand must have `interval_end <= origin−1h`.
Weather hour means have the same assumed delay. Publication timestamps are not observed;
this is an explicit availability approximation. EIA's current reporting instructions allow
submission within 60 minutes of hour end. [EIA-930 instructions](https://www.eia.gov/survey/form/eia_930/instructions.pdf).

## Preventing Leakage

- Training, validation, calibration and test are separate Parquet files. EDA reads **only training**.
  The final test receives schema/time/QC checks, not distributional exploration or model selection.
- The availability helper excludes the latest unpublished hour. Lag features must be evaluated
  relative to the forecast origin, not just shifted against the target. Future calendar features
  are known; future demand, weather, market prices and adjusted load are not feature inputs.
- NOAA weather is **observed**, not an archived forecast. The primary experiment excludes future
  observed weather. A future experiment using it must be labeled **conditional on realized weather**.
- Initial-training percentiles define extreme demand/weather. Scalers, imputers and model fitting
  must use each fold's available training history. No target imputation for scoring.
- STL uses two-sided smoothing solely for descriptive plots; its output must never become a feature.
- Latest-revised EIA and NOAA snapshots are not historical publication vintages. Even with a delay,
  this remains a retrospective study, not a verified operational simulation.

Tests mutate future demand/weather to confirm unchanged training summaries, thresholds and
available history; test weather cannot alter an earlier aggregation interval.

## Demand Patterns

<!-- BEGIN EDA FINDINGS -->
Training-only findings (26,304 hours; 2020–2022):

- Mean demand is **13,292.8 MW**; annual means are 2020: 13,160.7 MW, 2021: 13,370.1 MW, 2022: 13,347.9 MW.
  This short record includes pandemic years; the change is descriptive, not a structural trend claim.
- The average local daily profile peaks at **18:00** and reaches its
  minimum at **03:00**. **August**
  has the highest average load. Weekend mean load is **7.2%** below weekdays.
- Autocorrelation is **0.874 at 24 hours** and **0.718 at 168 hours**.
  Daily/weekly baselines are therefore substantive competitors, not token comparisons.
- Extreme demand is fixed at the training 95th percentile: **17,971.4 MW**.
  The training maximum is **25,101 MW** at **2021-06-29 21:00:00+00:00** (interval start).
- The three-station temperature proxy is present for **26,234 training hours**;
  **70** remain missing. Training temperature 5th/95th percentiles are
  **-3.70°C / 27.07°C**.
  Weather associations are observational and confounded by season and calendar.

Source: [eda_summary.json](reports/eda_summary.json). No validation, calibration or test outcomes
contribute to these findings or thresholds.
<!-- END EDA FINDINGS -->

![Daily and weekly load profiles](reports/figures/02_daily_weekly_patterns.png)
![Demand and observed temperature](reports/figures/04_weather_relationship.png)

The temperature relationship is U-shaped in the training data: load is lowest in mild conditions
and higher during cold and especially hot conditions. This motivates nonlinear weather covariates,
but does not establish a causal effect. The remaining plots cover the full training history,
month/hour structure, annual cycle, autocorrelation, descriptive weekly decomposition, holiday
comparisons, training extreme episodes and the evaluation timeline. Each title states its question.

## Models

Planned after review: persistence, daily and weekly seasonal naive; dynamic harmonic regression
with autoregressive errors; LightGBM; and one compact LSTM, only after the other models work.
Multiple periodicities in EDA motivate dynamic regression rather than a large single-season SARIMA.
No advanced model is presumed better. Parameters will be selected only within validation.

Under the one-hour delay, daily seasonal naive uses `target_end−24h` for horizons 1–23 and
`target_end−48h` for horizon 24; the previous-day value for that final hour is unpublished.
This availability-adjusted rule will be clearly labeled. Weekly naive uses elapsed 168-hour lags;
UTC seasonal lags do not always match local clock hours across DST.

## Backtesting

All bounds below are UTC **interval starts**, with exclusive end dates.

| Role | Start | End (exclusive) | Hours | Permitted use |
|---|---|---|---:|---|
| Initial training | 2020-01-01 | 2023-01-01 | 26,304 | EDA, initial fit, fixed extreme thresholds |
| Validation | 2023-01-01 | 2023-10-01 | 6,552 | Nine expanding monthly folds; model selection |
| Calibration | 2023-10-01 | 2024-01-01 | 2,208 | Reserved out-of-sample residuals, no model selection |
| Final test | 2024-01-01 | 2025-01-01 | 8,784 | Sealed until model choices are finalized |

Refit monthly and issue daily paths. All training labels must have matured by the origin's
reporting cutoff, including at split boundaries. Save every origin/horizon/fold prediction.
Test refits may use matured past test labels with fixed hyperparameters; no test-based tuning.
The calibration quarter sacrifices autumn/winter model-selection coverage to separate model
selection from initial interval estimation. Report that limitation rather than reusing the test.

## Results

**No model results yet.** Future outputs will report MAE, RMSE, WAPE and training-scaled weekly
MASE, by fold and condition. Paired daily losses will use dependence-aware uncertainty (HAC or
block bootstrap), with multiple-comparison correction. Small differences need statistical and
practical interpretation; no scores are fabricated in this milestone.

## Error Analysis

The frozen taxonomy covers weekdays, weekends, federal actual/observed holidays, local 16:00–20:00
peak hours, meteorological seasons, extreme demand and extreme temperature. Report counts and
missing-weather membership; do not treat unknown temperature as normal. Holiday comparisons are
unadjusted and federal holidays do not capture all regional or market-calendar effects.

## Extreme Demand

Use `demand >=` the initial-training 95th percentile, fixed across all models and future periods.
Thresholds are saved in [frozen_thresholds.json](reports/frozen_thresholds.json), not recomputed
on test data. A large demand value is not by itself a data error or forecast anomaly.

## Forecast Uncertainty

Planned 50%, 80% and 95% intervals use past out-of-sample residuals by horizon, initially from the
calibration quarter and subsequently a rolling 90-day window. Only matured residuals may update
intervals. Coverage, width and interval score must be assessed overall and by condition.
Dependence and drift invalidate an unqualified IID conformal guarantee. Calibration is empirical.

## Anomalies

Deferred until genuine out-of-sample forecasts exist. Interval violations will be called
**forecast anomalies**, with observed demand, forecast, residual, interval, calendar and weather
context. They are not evidence of outages or emergencies without external ground truth.
Rolling error/bias/variance and coverage analysis will follow in Milestone 4.

## Limitations

Three training years include pandemic disruption. Latest-revised load and QC'd weather can differ
from values available historically. The availability delay is assumed, not reconstructed from
publication logs. Three airports imperfectly cover a six-state region; weather gaps remain.
Demand excludes some behind-the-meter activity and follows EIA's balancing-authority definition,
not household consumption. The one-region study cannot establish cross-grid generalization.

## Reproducing the Analysis

Requires **CPython 3.12**, tested on macOS arm64. Run from this repository:

```bash
make setup       # Isolated .venv; pinned direct/transitive dependencies
make data        # Download or verify immutable sources, validate, align, partition
make analysis    # Data pipeline + training-only EDA; no notebook execution required
make test        # Unit and temporal-leakage tests, offline
make lint        # Static code checks
make reproduce   # Isolated offline rebuild; compare generated artifact SHA-256 hashes
```

The 17 source snapshots occupy about 200 MB. Retain `data/raw/` for exact future replay:
provider download URLs are mutable. A changed remote or cached hash fails closed; never silently
refresh a frozen experiment. [requirements.lock](requirements.lock) pins the dependency closure;
[reproducibility.json](reports/reproducibility.json) records the tested runtime and output hashes.
Bitwise figure reproducibility is verified on this runtime, not promised across operating systems.
Notebooks are optional views of the same outputs; a Jupyter frontend is not needed by the pipeline.
Streamlit and `make app` arrive in Milestone 5, after real forecast outputs exist.

## Repository Structure

```text
configs/       data.yaml, evaluation.yaml, sources.lock.json
src/gridcast/  data.py, validation.py, features.py, eda.py, audit.py
notebooks/     01_eda.ipynb
reports/       quality/provenance/EDA/reproducibility JSON, findings, tables, eight figures
tests/         time contracts, source immutability, NOAA QC, future-mutation leakage tests
data/          raw (immutable), interim (lineage/QC), processed (four isolated partitions)
```

Raw/intermediate/processed files, the virtual environment and scratch work are ignored by Git.
`reports/transformations.json` documents each transformation; `reports/run_log.jsonl` logs local
runs. On failure, a diagnostic JSON is written and processing stops. No observations are silently
removed or filled. Forecasting modules and notebooks will be added only when implemented.

Six milestones: **1 Data (complete)** → 2 Baselines + Backtesting → 3 Machine Learning →
4 Uncertainty + Anomalies → 5 Application → 6 Final Audit. Work pauses here for review;
push/remote setup belongs to the final milestone.
