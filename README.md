# Air Quality Prediction

Forecast PM2.5 **24 hours ahead** per station using CPCB monitoring data, and flag
"Poor or worse" air quality (PM2.5 > 90 µg/m³) for early warnings.

## Data
- Kaggle: *Air Quality Data in India (2015–2020)* — put `station_hour.csv` and `stations.csv` in `data/`
- TODO: merge meteorology (ERA5 / NASA POWER), NASA FIRMS fire counts, festival calendar

## Pipeline
| Step | File |
|---|---|
| Cleaning, imputation, lag/rolling/calendar features, chronological split | `src/data_prep.py` |
| Baselines, Decision Tree, Random Forest, AdaBoost, GBM, XGBoost, LightGBM, ± PCA | `src/train.py` |
| UI (station/date → forecast + AQI category, model comparison) | `app.py` |

## Leakage notes
- `AQI` / `AQI_Bucket` dropped (computed from same-time pollutants)
- Only lagged values and rolling windows ending at forecast time are used
- Chronological split: train ≤ 2018, val = H1 2019, test = H2 2019 onward

## Run
```bash
pip install -r requirements.txt
python src/data_prep.py
python src/train.py
streamlit run app.py
```

## TODO
- EDA notebook (missingness by station/season, diurnal/seasonal patterns)
- Compare ffill vs interpolation imputation
- Leave-one-station-out validation
- Additional analyses (pick 3): ablation, visualization, efficiency, robustness, error analysis, SHAP
