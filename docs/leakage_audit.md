# Leakage audit

**Task:** at issue time *t*, forecast PM2.5 at *t + 24 h* (regression) and whether it exceeds
90 µg/m³ – CPCB "Poor or worse" (classification). Every input must be known at *t*.

| Feature group (prefix) | Examples | Safe? | Why |
|---|---|---|---|
| **AQI, AQI_Bucket** | – | ❌ **Excluded** | Deterministic function of same-time pollutant sub-indices; AQI bucket is a label derived from it. Dropped at load time (`load_raw`). |
| Same-time target (`PM2.5` at *t+24*) | – | ❌ Excluded | It *is* the answer. The target is built by shifting the cleaned, **un-imputed** observations by −24 h. |
| PM2.5 history (`pm25_`) | lags 0–168 h, trailing means 3 h–14 d, std, min/max, EWMA, diffs, same-hour-of-day mean | ✅ | Lags ≥ 0 relative to *t*; all rolling windows are trailing (`rolling(w)` ends at *t*, never `center=True`). PM2.5 at *t* is known when the forecast for *t+24* is issued. |
| Other pollutants (`pol_`) | PM10, NOx, CO, O3 … current, lags, 24 h/7 d means | ✅ | Same-time readings at *t* are legitimate for a *t+24* target (they are inputs to AQI at *t*, not at *t+24*). |
| Missingness (`miss_`) | missing flags at *t*, fraction missing in last 24 h, hours since last PM2.5 reading | ✅ | Derived from the observation mask up to *t*. |
| Neighbour stations (`nbr_`) | leave-one-out city mean at *t*, city 24 h mean, city max, inverse-distance regional mean (other cities ≤ 400 km) | ✅ | Other stations' readings at *t* – available in real time from the CAAQMS network. The station's own value is excluded from the city mean. |
| Meteorology (`met_`) | ERA5 at *t*, trailing 24/72/168 h aggregates, 24 h tendencies, ventilation coefficient | ✅ (nowcast-of-current-weather) | Only weather up to *t*. Observed weather at *t+24* would be leakage, so it is **not** used. |
| Future weather (`fut_`) | ERA5 at *t+24* | ⚠️ **Excluded from all models**; only in one labelled ablation | Proxy for a perfect numerical weather forecast. Reported separately as an upper bound of what an NWP feed could add. |
| Fire hotspots (`fire_`) | FIRMS counts/FRP within 100–500 km, upwind fires, 1/3/7-day sums | ✅ | Use detections from the previous day(s) only (≥ 1 day lag > FIRMS NRT latency). |
| Calendar (`cal_`) | hour/day-of-week/month/day-of-year of the **target time**, holidays, days from Diwali, stubble season | ✅ | Calendar of *t+24* is known in advance. |
| Location (`loc_`) | city latitude/longitude | ✅ | Static metadata. |

## Preprocessing steps that could leak – and how they are handled

| Step | Risk | Handling |
|---|---|---|
| Imputation | Linear interpolation fills a gap using the **next** reading after it (future). | Forward fill (≤ 24 h) is used; interpolation is run only as a negative control in the imputation study, and the automated test confirms it leaks. |
| Rolling windows | `center=True` or windows ending at *t+1* include future values. | All windows trailing; verified by the future-perturbation test. |
| Fault cleaning | Spike detection with a centred median uses future values. | Rules are causal: trailing 24 h median and other stations at the same hour; flat-line rule flags from the 12th identical hour onward. |
| Scaling / PCA / median imputer | Fitting on the full data leaks test statistics. | All transformers are inside sklearn `Pipeline`s fitted on the training period only. |
| Train/val/test split | Random splits leak future and neighbour information; a training row at 23:00 on the last training day has a target in the next period. | Chronological split on issue time with a **24 h embargo** before each boundary. Hyper-parameters chosen on validation; test touched once. |
| Climatology baseline | Computing climatology on all data. | Computed from the training period only. |
| Threshold tuning (classification) | Tuning on test. | Thresholds tuned for F1 on the validation period. |

## Automated checks (`tests/test_leakage.py`)

1. **Future-perturbation test** – corrupt every pollutant reading after *t₀* for one city (and
   delete 30 % of future PM2.5), rebuild all 141 features, and assert no feature at issue times
   ≤ *t₀* changes. ✅ passes.
2. **Negative control** – the same test with linear interpolation **must** detect leakage. ✅
3. **Target alignment / forbidden columns** – `target(t) == PM2.5_obs(t+24)`; AQI, labels and
   `fut_` columns are not features; no feature has |corr| ≥ 0.95 with the target. ✅
4. **Split embargo** – no training/validation target time falls in a later period. ✅
