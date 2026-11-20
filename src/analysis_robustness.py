"""
Analysis D - Robustness & sensitivity.

1. Sensor noise      : multiplicative Gaussian noise on all pollutant-derived
                       inputs (pm25_/pol_/nbr_), sigma = 5-40 %  (final models)
2. More missing data : randomly blank 10-60 % of pollutant-derived inputs
                       (final models; sklearn models fall back to median
                       imputation, boosters use native NaN handling)
3. Less training data: fewer training years / fewer training stations (refits)
4. Distribution shift: (a) leave-one-city-out (new-city generalisation),
                       (b) unseen stations, (c) COVID-lockdown period,
                       (d) rolling-origin (walk-forward) validation
Outputs: results/robust_*.csv
"""
import os
import sys

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import HORIZON, MODEL_DIR, RESULTS_DIR, SEED  # noqa: E402
from models import (MODEL_ORDER, fast_fit_predict, load_frame,  # noqa: E402
                    regression_metrics, stratified_subsample)

REFIT_MODELS = ["DecisionTree", "RandomForest", "HistGradientBoosting", "XGBoost",
                "LightGBM", "CatBoost"]
N_SUB = 250_000


def _save(rows, name):
    pd.DataFrame(rows).to_csv(os.path.join(RESULTS_DIR, f"robust_{name}.csv"), index=False)


def perturbation_tests(test, feats):
    rng = np.random.default_rng(SEED)
    smp = test.sample(min(150_000, len(test)), random_state=SEED)
    pol = [f for f in feats if f.startswith(("pm25_", "pol_", "nbr_"))]
    noise_rows, miss_rows = [], []
    loaded = {}
    for name in MODEL_ORDER:
        path = os.path.join(MODEL_DIR, f"{name}.joblib")
        if not os.path.exists(path):
            continue
        loaded[name] = joblib.load(path)
    from train import inv
    for name, b in loaded.items():
        pipe = b["pipeline"]

        def predict(X):
            return inv(pipe.predict(X[b["features"]]), b["transform"])
        for sigma in [0.0, 0.05, 0.10, 0.20, 0.40]:
            X = smp[feats].copy()
            if sigma:
                X[pol] = X[pol] * rng.normal(1, sigma, size=(len(X), len(pol))).astype(np.float32)
            noise_rows.append({"model": name, "noise_sigma": sigma,
                               **regression_metrics(smp["target"], predict(X))})
        for rate in [0.0, 0.1, 0.3, 0.6]:
            X = smp[feats].copy()
            if rate:
                mask = rng.random((len(X), len(pol))) < rate
                X[pol] = X[pol].mask(mask)
            miss_rows.append({"model": name, "missing_rate": rate,
                              **regression_metrics(smp["target"], predict(X))})
        print("perturbation", name, "done", flush=True)
        _save(noise_rows, "noise")
        _save(miss_rows, "missing")


def training_size_tests(fin, test, feats):
    rows = []
    for label, start in [("1.5 yr (2018-2019H1)", "2018-01-01"),
                         ("2.5 yr (2017-2019H1)", "2017-01-01"),
                         ("3.5 yr (2016-2019H1)", "2016-01-01"),
                         ("4.5 yr (all)", "2015-01-01")]:
        sub = stratified_subsample(fin[fin["Datetime"] >= start], N_SUB)
        for name in REFIT_MODELS:
            pred = fast_fit_predict(name, sub[feats], sub["target"], test[feats])
            rows.append({"experiment": "fewer years", "setting": label, "model": name,
                         "train_rows_available": int((fin["Datetime"] >= start).sum()),
                         **regression_metrics(test["target"], pred)})
            print(label, name, round(rows[-1]["MAE"], 2), flush=True)
        _save(rows, "training_size")
    stations = np.array(sorted(fin["StationId"].unique()))
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(stations)
    for frac in [0.25, 0.5, 0.75, 1.0]:
        keep = set(perm[:int(round(frac * len(perm)))])
        sub = stratified_subsample(fin[fin["StationId"].isin(keep)], N_SUB)
        seen = test["StationId"].isin(keep)
        for name in REFIT_MODELS:
            pred = fast_fit_predict(name, sub[feats], sub["target"], test[feats])
            r = {"experiment": "fewer stations", "setting": f"{int(frac * 100)}% stations",
                 "model": name, **regression_metrics(test["target"], pred)}
            if frac < 1:
                r["MAE_seen_stations"] = float(np.mean(np.abs(pred[seen] - test["target"][seen])))
                r["MAE_unseen_stations"] = float(np.mean(np.abs(pred[~seen] - test["target"][~seen])))
            rows.append(r)
            print(frac, name, round(r["MAE"], 2), flush=True)
        _save(rows, "training_size")


def loco_tests(fin, test, feats):
    cities = ["Delhi", "Bengaluru", "Hyderabad", "Chennai", "Lucknow", "Kolkata", "Mumbai",
              "Patna", "Jaipur", "Gurugram"]
    base_sub = stratified_subsample(fin, N_SUB)
    rows = []
    for name in ["DecisionTree", "RandomForest", "XGBoost", "LightGBM", "CatBoost"]:
        base_pred = pd.Series(fast_fit_predict(name, base_sub[feats], base_sub["target"],
                                               test[feats]), index=test.index)
        for city in cities:
            te = test[test["City"] == city]
            if len(te) < 1000:
                continue
            sub = stratified_subsample(fin[fin["City"] != city], N_SUB)
            pred = fast_fit_predict(name, sub[feats], sub["target"], te[feats])
            rows.append({"model": name, "city": city, "n_test": len(te),
                         "MAE_city_held_out": float(np.mean(np.abs(pred - te["target"]))),
                         "MAE_city_in_training": float(np.mean(np.abs(base_pred[te.index] - te["target"]))),
                         "MAE_persistence": float(np.mean(np.abs(te["pm25_lag0"] - te["target"]))),
                         "R2_city_held_out": regression_metrics(te["target"], pred)["R2"]})
            print("LOCO", name, city, round(rows[-1]["MAE_city_held_out"], 2),
                  round(rows[-1]["MAE_city_in_training"], 2), flush=True)
        _save(rows, "loco")


def walk_forward(df, feats):
    origins = ["2017-07-01", "2018-01-01", "2018-07-01", "2019-01-01", "2019-07-01", "2020-01-01"]
    rows = []
    for o in origins:
        o = pd.Timestamp(o)
        end = o + pd.DateOffset(months=6)
        tr = df[df["Datetime"] <= o - pd.Timedelta(hours=HORIZON)]
        te = df[(df["Datetime"] > o) & (df["Datetime"] <= min(end, df["Datetime"].max()))]
        sub = stratified_subsample(tr, N_SUB)
        rows.append({"origin": o.date(), "model": "Persistence", "n_train": len(tr),
                     "n_test": len(te), **regression_metrics(te["target"], te["pm25_lag0"])})
        for name in ["DecisionTree", "RandomForest", "XGBoost", "LightGBM"]:
            pred = fast_fit_predict(name, sub[feats], sub["target"], te[feats])
            rows.append({"origin": o.date(), "model": name, "n_train": len(tr),
                         "n_test": len(te), **regression_metrics(te["target"], pred)})
            print("walk-forward", o.date(), name, round(rows[-1]["MAE"], 2), flush=True)
        _save(rows, "walk_forward")


def period_shift():
    """Final-model error by period (pre-lockdown vs lockdown) from saved predictions."""
    pt = pd.read_parquet(os.path.join(RESULTS_DIR, "predictions_test.parquet"))
    T = pt["Datetime"] + pd.Timedelta(hours=HORIZON)
    period = np.select([T < "2019-10-01", T < "2020-03-01", T < "2020-03-25", T <= "2020-05-31"],
                       ["Jul-Sep 2019 (monsoon)", "Oct 2019-Feb 2020 (winter)",
                        "Mar 2020 (pre-lockdown)", "25 Mar-May 2020 (lockdown)"],
                       "Jun 2020 (unlock)")
    rows = []
    models = [c for c in pt.columns if c not in ("StationId", "City", "Datetime", "target",
                                                 "target_poor") and "+PCA" not in c]
    for p in np.unique(period):
        m = period == p
        for c in models:
            r = regression_metrics(pt["target"][m], pt[c][m].fillna(pt[c].median()))
            rows.append({"period": p, "model": c, "n": int(m.sum()),
                         "mean_target": float(pt["target"][m].mean()),
                         **{k: r[k] for k in ["MAE", "RMSE", "R2", "Bias_top5pct"]},
                         "Bias": float((pt[c][m] - pt["target"][m]).mean())})
    _save(rows, "period_shift")


def main(parts=("perturb", "size", "loco", "walk", "period")):
    df, feats, S = load_frame()
    final, test = S["final"], S["test"]
    if "period" in parts:
        period_shift()
    if "perturb" in parts:
        perturbation_tests(test, feats)
    if "size" in parts:
        training_size_tests(final, test, feats)
    if "loco" in parts:
        loco_tests(final, test, feats)
    if "walk" in parts:
        walk_forward(df, feats)


if __name__ == "__main__":
    main(tuple(sys.argv[1:]) or ("perturb", "size", "loco", "walk", "period"))
