# Predicting Air Quality – 24 h-ahead PM2.5 early warning for India

Analytics for a regional pollution-control board: forecast **PM2.5 24 hours ahead** at every
CPCB monitoring station and flag hours that will be **"Poor or worse" (> 90 µg/m³)** so that
advisories can be issued a day in advance.

## Headline results (held-out test year: Jul 2019 – Jun 2020, 763 k station-hours, 107 stations)

**Regression – PM2.5 at t+24 h** (µg/m³; skill = 1 − MAE / MAE_persistence)

| Model | MAE | RMSE | R² | Skill | MAE top 5 % | Train time |
|---|---|---|---|---|---|---|
| **XGBoost** | **22.38** | **42.98** | **0.683** | **14.3 %** | 108.5 | 93 s |
| Ensemble (LightGBM+XGBoost+CatBoost) | 22.43 | 43.30 | 0.679 | 14.1 % | 109.7 | 372 s |
| Random Forest | 22.50 | 43.08 | 0.682 | 13.8 % | 109.0 | 252 s |
| LightGBM | 22.58 | 43.58 | 0.674 | 13.5 % | 110.3 | 96 s |
| HistGradientBoosting | 22.64 | 43.56 | 0.675 | 13.3 % | 110.0 | 173 s |
| CatBoost | 22.67 | 43.90 | 0.670 | 13.2 % | 111.4 | 183 s |
| Gradient Boosting (sklearn)* | 22.76 | 44.03 | 0.668 | 12.8 % | 110.9 | 493 s |
| AdaBoost* | 23.40 | 43.92 | 0.669 | 10.4 % | 109.7 | 479 s |
| Decision Tree | 24.03 | 45.99 | 0.637 | 8.0 % | 113.3 | 236 s |
| *Persistence (t+24 = t)* | 26.12 | 50.88 | 0.556 | – | 111.0 | – |
| *Seasonal naive (7-day same hour)* | 27.73 | 52.27 | 0.531 | – | 124.4 | – |
| *Climatology* | 40.26 | 62.45 | 0.331 | – | 119.4 | – |

\* trained on a 150 k-row stratified sample (single-threaded sklearn boosting).
Every prescribed model beats all three baselines. **PCA** (72 of 141 components = 95 %
variance, fit on the training period) **hurts every model** by 0.6–2.0 µg/m³ MAE (e.g.
XGBoost 22.38 → 23.68): trees exploit individual, physically meaningful features (current
PM2.5, same-hour lags, BLH), which PCA mixes, and the low-variance directions it drops are
still predictive. Full table: `results/benchmark.csv`.

**Early warning – will PM2.5 at t+24 exceed 90 µg/m³ ("Poor or worse")?** (20.6 % of test hours are positive)

| Approach | Best model | Precision | Recall | F1 | PR-AUC |
|---|---|---|---|---|---|
| Direct classifier (threshold tuned on validation) | XGBoost | 0.752 | 0.822 | **0.786** | 0.867 |
| Direct classifier – highest recall / PR-AUC | CatBoost | 0.726 | **0.847** | 0.782 | **0.869** |
| Regression forecast, threshold tuned on validation (≈ 83 µg/m³) | XGBoost | 0.753 | 0.820 | 0.785 | 0.861 |
| Regression forecast at the nominal 90 µg/m³ | XGBoost | 0.790 | 0.773 | 0.782 | 0.861 |
| Persistence at 90 µg/m³ | – | 0.744 | 0.748 | 0.746 | 0.811 |

Direct classifiers and regression-derived warnings perform almost identically once the
regression threshold is tuned. The tuned threshold sits *below* 90 because the regression
forecasts are biased low at high concentrations; lowering it trades a little precision for
~5 points of recall, which matters for public-health warnings. Full table:
`results/classification_benchmark.csv`.

---

## 1. Problem definition

| Item | Choice | Justification |
|---|---|---|
| Unit of prediction | station × hour | Warnings are issued per locality; hourly resolution identifies *which hours* are unhealthy. |
| Target (regression) | PM2.5 (µg/m³) at **t + 24 h** | PM2.5 is the dominant AQI driver in India; 24 h gives the board one day of lead time. |
| Target (classification) | PM2.5(t+24) > 90 µg/m³ (CPCB "Poor" band starts at 91) | Directly actionable early-warning label. Both framings are compared: classes derived from the regression forecast vs a direct classifier. |
| Forecast issue time | every hour *t* | Every feature uses only data available at *t* (see leakage audit). |
| Weather set-up | **observed ERA5 weather up to t** (no weather at t+24) | Honest forecast set-up. ERA5 at t+24 is used only in a labelled "oracle/perfect forecast" ablation. |
| Spatial setting | **known stations** (chronological split over all stations) + leave-one-city-out and unseen-station tests | The board forecasts for its existing network; new-station/new-city generalisation is reported separately. |

## 2. Data (≥ 100 k rows, ≥ 50 features, genuine sources)

* **CPCB station-hour data** (Kaggle *Air Quality Data in India 2015–2020*): 2.59 M station-hours,
  110 stations (107 with PM2.5), 26 cities, 12 pollutants, Jan 2015 – Jul 2020. **The full
  dataset is used** (no sampling) – 1.91 M forecastable station-hours after cleaning.
* **ERA5 hourly reanalysis** (ECMWF/Copernicus, via Open-Meteo): temperature, humidity, dew point,
  pressure, rain, wind, cloud, radiation and **boundary-layer height** for each city's grid cell.
* **NASA FIRMS MODIS** active fires (559 k detections, 2014–2020): counts/FRP around each city and
  **upwind** fires (crop-residue burning).
* **Festival/holiday calendar** (Diwali, Dussehra, Holi, Lohri, national holidays).

**141 engineered features** in 8 groups – PM2.5 history (33), other pollutants (30), missingness
(14), neighbouring stations (6), meteorology (31), fires (8), calendar (15), location (2).
Full provenance, alignment and licences: [docs/data_sources.md](docs/data_sources.md).

## 3. Preprocessing

* **Sensor faults** (all rules causal – they only look at *t* and earlier or at other stations at
  the same hour, so they could run live): negative values; physically implausible magnitudes
  (CO > 50 mg/m³ – a unit error at GJ001; O3 > 400 µg/m³ – a fault period at DL028); PM2.5 > PM10;
  flat-lined sensors (≥ 12 identical hours); isolated PM spikes (> 4× the trailing 24 h median
  **and** > 4× the other stations in the city at the same hour). Genuine episodes (Diwali, smog)
  raise the whole city and are therefore kept. Counts: `results/fault_log.csv`.
* **Missingness** is explored by pollutant, station, season, year and pollution level
  (`results/figures/eda_missingness_*`). Missing indicators, fraction-missing-in-24 h and
  hours-since-last-reading are features. **Three imputation strategies were compared**
  (`src/imputation_study.py`): forward fill (causal), seasonal same-hour-yesterday fill (causal),
  linear interpolation (non-causal). Forward fill (≤ 24 h) was chosen – best causal
  reconstruction of hidden readings and equal downstream accuracy; interpolation looks better
  only because it uses future readings (our leakage test detects it).
* **Target** = cleaned, *un-imputed* PM2.5 at t+24 (imputed values never become labels).
* **Splits**: chronological on issue time – train ≤ 2018, validation H1 2019 (tuning,
  early stopping, thresholds), test Jul 2019 – Jun 2020 (one full annual cycle incl. the
  stubble/Diwali season and the COVID lockdown). 24 h embargo before each boundary. Final
  models are refit on train+validation and scored once on test. Rolling-origin validation and
  leave-one-city-out are in the robustness analysis.
* **Scaling / imputation / PCA** live inside sklearn pipelines fitted on the training period.
* **Target transform**: models learn √PM2.5 (chosen on validation over log1p and raw); metrics
  are in µg/m³.

### Leakage audit

Feature-group table and preprocessing checks: [docs/leakage_audit.md](docs/leakage_audit.md).
`tests/test_leakage.py` automatically verifies that corrupting the future never changes any
feature at or before *t* (and that interpolation, as a negative control, does).

## 4. Models

Baselines – **Persistence** (PM2.5 at t+24 = PM2.5 at t, i.e. "tomorrow = today" at the same
hour), **Seasonal naive** (mean of the same hour over the last 7 days), **Climatology**
(station × month × hour mean).
Prescribed models – **Decision Tree**, **Random Forest**, boosting: **AdaBoost**, **Gradient
Boosting** (sklearn), **HistGradientBoosting**, **XGBoost**, **LightGBM**, **CatBoost** – each
trained **with and without PCA** (95 % variance, fit on the training period), plus a mean
ensemble of the three boosting libraries. The same splits, target and features are used for
all. AdaBoost and sklearn GradientBoosting (single-threaded, exact splits) are trained on a
150 k-row station × year × month-stratified sample of the same training period for compute
reasons; all other models use the full 1.15 M training rows.

## 5. Additional analyses (all six done)

| | Analysis | Script | Outputs |
|---|---|---|---|
| A | Feature-group ablation, top-k feature selection, PCA variance levels, oracle weather | `src/analysis_ablation.py` | `results/ablation_*.csv`, `figures/abl_*` |
| B | Seasonal/diurnal/spatial patterns, class separation, predicted vs actual incl. episodes | `src/eda.py`, `src/analysis_visuals.py` | `figures/eda_*`, `figures/viz_*` |
| C | Training time, inference time, model size vs accuracy | `src/train.py`, `src/analysis_visuals.py` | `results/efficiency.csv`, `figures/eff_*` |
| D | Sensor noise, extra missingness, fewer years/stations, unseen stations, leave-one-city-out, walk-forward, lockdown shift | `src/analysis_robustness.py` | `results/robust_*.csv`, `figures/rob_*` |
| E | Errors by season, hour, city, station, pollution band, spikes, sudden onsets, gaps; error correlation | `src/analysis_errors.py` | `results/errors_*.csv`, `figures/err_*` |
| F | SHAP (5 tree models), partial dependence in µg/m³, group permutation importance, physical-plausibility check | `src/analysis_explain.py` | `results/explain_*.csv`, `figures/xai_*` |

## 6. User interface

`streamlit run app.py` – integrated with the trained pipeline:

* **Forecast** – choose city, station, day and models → hourly 24 h-ahead forecasts for that day,
  AQI category badges, number of Poor+ hours, peak time, direct-classifier probability of
  Poor+, automatic early-warning banner, history + forecast chart with AQI colour bands,
  per-model error against actuals.
* **Early-warning map** – every station for a chosen day on an interactive map coloured by
  forecast category, ranked risk table, CSV download of the warning list.
* **Explain & what-if** – SHAP contributions for a single forecast (by feature and by group)
  and a scenario simulator (wind, mixing height, rain, humidity, upwind fires, emission cuts)
  showing every model's response.
* **Benchmark / Analyses / Data & EDA** – all result tables and figures, leakage audit,
  imputation comparison, station time-series explorer.

## 7. Reproduce

```bash
python -m venv .venv && .venv\Scripts\activate      # Windows (source .venv/bin/activate on Linux/Mac)
pip install -r requirements.txt
python run_all.py            # fetch -> prep -> leakage tests -> EDA -> imputation -> train -> classifiers -> analyses
streamlit run app.py
```

Individual steps: `python src/fetch_external.py`, `python src/data_prep.py`,
`python -m pytest tests -q`, `python src/eda.py`, `python src/train.py`,
`python src/train_classifiers.py`, `python src/analysis_{errors,explain,ablation,robustness,visuals,plots}.py`.
Everything is downloaded automatically (no Kaggle credentials needed).

## 8. Repository layout

```
app.py                     Streamlit UI
run_all.py                 end-to-end pipeline runner
src/config.py              paths, horizon, thresholds, splits, city coordinates, festivals
src/fetch_external.py      CPCB (Kaggle), ERA5 (Open-Meteo), FIRMS downloads
src/data_prep.py           cleaning, imputation, merges, 141 features, targets, splits
src/eda.py                 exploratory analysis figures
src/imputation_study.py    ffill vs seasonal vs interpolation
src/models.py              model factory, metrics, helpers
src/train.py               regression benchmark (+/- PCA, baselines, ensemble)
src/train_classifiers.py   direct classifiers vs regression-derived warnings
src/analysis_*.py          analyses A-F and their plots
tests/test_leakage.py      automated leakage audit
docs/                      data sources, leakage audit
results/                   metrics (csv), predictions (parquet), figures
models/                    trained pipelines (joblib, git-ignored)
```
