# Viva preparation guide

Short answers to the questions each team member should be able to handle. Numbers refer to the
final run (see `results/benchmark.csv` and the README results section).

## Problem & data

**What exactly do you predict?** PM2.5 at station *s* and hour *t + 24*, given everything known
at hour *t*. Classification: whether that value exceeds 90 µg/m³ (CPCB "Poor" or worse).

**Why 24 h and hourly?** A day of lead time is enough to issue an advisory; hourly resolution
shows *which hours* (typically night/early morning in winter) are unhealthy.

**Why not AQI as the target?** AQI is a max over sub-indices of 24 h averages of several
pollutants – it mixes targets and leaks easily. PM2.5 is the dominant driver; the AQI category
for PM2.5 is reported from the forecast.

**Dataset size?** 2.59 M raw station-hours, 107 stations with PM2.5, 26 cities, 2015 – mid-2020;
1.91 M forecastable rows after cleaning (target observed and PM2.5 history available). 141 features.
The whole dataset is used – no sampling (except a stratified 150 k subsample for the two
single-threaded sklearn boosters, which preserves station × year × month proportions).

**Extra sources and alignment?** ERA5 weather (nearest 0.25° cell to the city centre, IST hours),
NASA FIRMS fires (distance & bearing from the city, previous day only), festival calendar
(target date). See `docs/data_sources.md`.

## Preprocessing

**Faulty readings?** Negative values; unit errors (CO > 50 mg/m³ at GJ001 – values look like
µg/m³); implausible O3 > 400; PM2.5 > PM10 (physically impossible – PM10 dropped); flat-lined
sensors (12+ identical hours); isolated spikes (> 4× trailing 24 h median **and** > 4× the
city's other stations at the same hour). City-wide spikes (Diwali) are kept – that is how we
avoid deleting genuine episodes. All rules are causal.

**Missing values?** Missingness differs by station (some never report PM2.5), year (stations
commissioned 2018–19), pollutant (xylene 80 % missing, NH3/PM10 ~45 %). It is mildly
informative – sensors are missing slightly more often in cleaner periods. We add missing
indicators, fraction-missing-in-24 h and hours-since-last-reading. Imputation: compared forward
fill, seasonal (same hour yesterday) and linear interpolation; chose forward fill (≤ 24 h).
Interpolation reconstructs best but uses the *next* reading → leakage (our test catches it).

**Why is the target not imputed?** Imputed labels would teach the model to predict our own
imputation, and inflate scores.

## Leakage

**Which columns would leak?** AQI and AQI_Bucket (computed from same-time pollutants), any
same-time target value, observed weather at t+24, centred rolling windows, interpolation, scalers
or PCA fitted on all data, random splits, thresholds tuned on test.

**How do you prove there is none?** `tests/test_leakage.py` corrupts all data after a cut-off and
checks no feature at earlier times changes; target alignment; no feature with |corr| ≥ 0.95;
split embargo.

**Is using PM2.5 at time t leakage?** No – the target is PM2.5 at t+24. Value at t is known when
the forecast is issued (that is exactly the persistence baseline).

**Weather: nowcast or forecast?** Only weather up to *t* (ERA5 analysis of current/past weather).
We show separately how much a perfect weather forecast at t+24 would add ("oracle" ablation) –
the gap motivates connecting an NWP feed (e.g. IMD/GFS) operationally.

## Validation

**Splits?** Chronological: train ≤ 2018, validation H1 2019, test Jul 2019 – Jun 2020; 24 h
embargo. Tune on validation; refit on train+validation; test once.

**Why not random split?** Neighbouring hours are almost identical and stations in a city are
correlated – random splits leak future and neighbour information and overstate performance.

**Spatial dependence?** Main task = known stations. Robustness: leave-one-city-out and
unseen-station tests quantify generalisation to new locations.

**Baselines?** Persistence (t+24 = t, same hour – "tomorrow = today"), seasonal naive (7-day
same-hour mean), climatology (station × month × hour mean). All models must beat persistence.

## Models (be ready to explain each)

* **Decision tree** – recursive binary splits minimising squared error; depth and min leaf size
  tuned on validation. Interpretable, high variance.
* **Random forest** – bagging of deep trees on bootstrap samples with random feature subsets;
  averages away variance. Parallel.
* **AdaBoost (R2)** – sequentially re-weights samples with large errors; each tree focuses on
  hard cases; weighted median of trees.
* **Gradient boosting** – each tree fits the negative gradient (residuals) of the loss;
  shrinkage (learning rate) + subsampling.
* **HistGradientBoosting / LightGBM** – gradient boosting on histogram-binned features (fast);
  LightGBM grows trees leaf-wise, uses GOSS/EFB; native missing-value handling.
* **XGBoost** – regularised (L1/L2 on leaf weights) second-order gradient boosting; hist
  algorithm; sparsity-aware split finding.
* **CatBoost** – symmetric (oblivious) trees, ordered boosting to reduce target-leakage bias in
  gradient estimates.
* **PCA** – orthogonal projection onto directions of maximal variance; fitted on the training
  period after median imputation and standardisation; keeps 95 % variance. It hurts tree models
  because trees split on single interpretable features (e.g. a lag) and PCA mixes them; it also
  discards low-variance but predictive directions.
* **Why √ target?** PM2.5 is right-skewed; √ stabilises variance so errors at 30 and 300 µg/m³
  are weighted more evenly. Chosen on validation over log and raw.
* **Early stopping** – boosters stop when validation loss stops improving (150 rounds patience);
  final model uses 1.1 × best rounds on the larger train+val set.

## Metrics

MAE (average absolute error, robust), RMSE (penalises large errors), R² (variance explained),
MAE on the top 5 % / 1 % (extremes), bias on extremes; for warnings precision (how many alerts
were right), recall (how many Poor+ hours we caught), F1, PR-AUC (threshold-free, suited to
imbalance) – never accuracy alone (≈ 80 % of test hours are not Poor+). Skill = 1 − MAE/MAE_persistence.

## Typical follow-up questions

* *Why does the model under-predict extremes?* Regression to the mean, √ transform, and
  extremes are rare/unpredictable 24 h ahead (sudden fireworks, dust storms). See bias-by-band.
* *Why is the lockdown period over-predicted?* An abrupt emission drop never seen in training –
  a distribution shift; the model only "knows" emissions through recent pollution levels and
  catches up after a day.
* *What would you improve?* NWP forecast weather, FIRMS VIIRS (more fires), traffic data,
  station-level coordinates, quantile/probabilistic forecasts, online re-training.
* *Limitations?* City-level weather, ERA5 not real-time (in operations use the IMD/GFS analysis),
  stations with short history, COVID shift, PM sensor caps at 1000.
