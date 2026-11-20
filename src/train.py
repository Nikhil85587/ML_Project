"""
Regression benchmark for the 24 h-ahead PM2.5 forecast.

Protocol (identical for every model)
  1. tune  : fit on TRAIN (<= 2018), evaluate on VAL (H1 2019) -> early-stopping
             rounds for boosters, depth for the decision tree, val predictions
             (used later to tune warning thresholds)
  2. final : refit on TRAIN+VAL with the tuned settings, evaluate once on TEST
             (H2 2019 - H1 2020, never touched during development)
Every model is trained with and without PCA (PCA fit on the training period
only). Baselines: persistence (t+24 = t), seasonal naive (mean of the same hour
over the last 7 days) and a station x month x hour climatology.

Outputs: results/benchmark.csv, results/predictions_{val,test}.parquet,
results/pca_explained_variance.csv, models/<name>.joblib
"""
import argparse
import gc
import json
import os
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import MODEL_DIR, RESULTS_DIR, SEED, TARGET_TRANSFORM  # noqa: E402
from models import (BOOSTERS, MODEL_ORDER, NATIVE_NAN, SLOW_MODELS, SUBSAMPLE_N, TUNE_N,  # noqa: E402
                    Timer, early_stopping_fit, frame, load_matrix, make_pipeline,
                    make_regressor, model_size_mb, regression_metrics, staged_at,
                    stratified_positions)

ID_COLS = ["StationId", "City", "Datetime", "target", "target_poor"]


# --------------------------------------------------------------- transforms
def fwd(y, how=TARGET_TRANSFORM):
    return {"log1p": np.log1p, "sqrt": np.sqrt, "none": np.asarray}[how](y)


def inv(p, how=TARGET_TRANSFORM):
    p = np.asarray(p, dtype=float)
    out = {"log1p": np.expm1, "sqrt": lambda x: np.clip(x, 0, None) ** 2,
           "none": lambda x: x}[how](p)
    return np.clip(out, 0, None)


# --------------------------------------------------------------- baselines
def climatology(train, other):
    """Mean target per station x month x hour of the forecast target time,
    falling back to city x month x hour and month x hour."""
    T = train["Datetime"] + pd.Timedelta(hours=24)
    tr = train.assign(m=T.dt.month, h=T.dt.hour)
    To = other["Datetime"] + pd.Timedelta(hours=24)
    ot = other.assign(m=To.dt.month, h=To.dt.hour)
    pred = pd.Series(np.nan, index=other.index)
    for keys in (["StationId", "m", "h"], ["City", "m", "h"], ["m", "h"]):
        tab = tr.groupby(keys)["target"].mean().rename("clim")
        fill = ot[keys].join(tab, on=keys)["clim"]
        pred = pred.fillna(fill)
    return pred.to_numpy()


def baselines(train_meta, other_meta, other_X):
    return {"Persistence": other_X["pm25_lag0"].to_numpy(),
            "SeasonalNaive7d": other_X["pm25_samehour7"].to_numpy(),
            "Climatology": climatology(train_meta, other_meta)}


# --------------------------------------------------------------- one model
def tune_and_fit(name, use_pca, Xtr, ytr, Xva, yva, Xfin, yfin, Xte, log):
    """Returns (final_pipeline, val_pred, test_pred, info)."""
    info = {}
    native = name in NATIVE_NAN
    # --- stage 1: tuning on validation ------------------------------
    if name in BOOSTERS:
        prep = make_pipeline("passthrough", use_pca=use_pca, native_nan=native)
        if len(prep.steps) > 1:
            prep = Pipeline(prep.steps[:-1])
            Ztr = prep.fit_transform(Xtr)
            Zva = prep.transform(Xva)
        else:
            Ztr, Zva = Xtr, Xva
        m, best = early_stopping_fit(name, Ztr, fwd(ytr), Zva, fwd(yva), "reg")
        if name == "HistGradientBoosting":
            val_pred = inv(staged_at(m, Zva, best))
        else:
            val_pred = inv(m.predict(Zva))
        n_final = int(best * 1.1) + 10                     # slightly more data -> more rounds
        info["best_iter"] = best
        del m, Ztr, Zva
        model_final = make_regressor(name, n_final)
    elif name == "DecisionTree" and not use_pca:
        # grid on a random 250k-row subset of TRAIN (compute), scored on full VAL
        best, best_mae = None, np.inf
        rs = np.random.default_rng(SEED).choice(len(Xtr), min(TUNE_N, len(Xtr)), replace=False)
        Xg, yg = Xtr.iloc[rs], ytr[rs]
        for depth in (10, 14, 18):
            for leaf in (50, 200, 500):
                p = make_pipeline(make_regressor(name), native_nan=False)
                p.set_params(model__max_depth=depth, model__min_samples_leaf=leaf)
                p.fit(Xg, fwd(yg))
                mae = np.mean(np.abs(inv(p.predict(Xva)) - yva))
                if mae < best_mae:
                    best, best_mae = (depth, leaf), mae
        p = make_pipeline(make_regressor(name), native_nan=False)
        p.set_params(model__max_depth=best[0], model__min_samples_leaf=best[1])
        p.fit(Xtr, fwd(ytr))
        val_pred = inv(p.predict(Xva))
        info["best_params"] = {"max_depth": best[0], "min_samples_leaf": best[1]}
        log["DecisionTree_params"] = info["best_params"]
        model_final = make_regressor(name)
        model_final.set_params(max_depth=best[0], min_samples_leaf=best[1])
    else:
        model_final = make_regressor(name)
        if name == "DecisionTree" and "DecisionTree_params" in log:
            model_final.set_params(**log["DecisionTree_params"])
        val_pred = None
        if not use_pca:                                    # val preds for threshold tuning
            p = make_pipeline(make_regressor(name), native_nan=native)
            if name == "DecisionTree" and "DecisionTree_params" in log:
                p.set_params(**{f"model__{k}": v for k, v in log["DecisionTree_params"].items()})
            p.fit(Xtr, fwd(ytr))          # Xtr is already subsampled for slow models
            val_pred = inv(p.predict(Xva))

    # --- stage 2: refit on train+val, evaluate on test ---------------
    pipe = make_pipeline(model_final, use_pca=use_pca, native_nan=native)
    with Timer() as t_fit:
        pipe.fit(Xfin, fwd(yfin))
    with Timer() as t_pred:
        test_pred = inv(pipe.predict(Xte))
    info.update(train_s=t_fit.s, infer_ms_per_1k=1e6 * t_pred.s / len(Xte),
                n_train=len(Xfin), size_mb=model_size_mb(pipe))
    if use_pca:
        info["pca_components"] = int(pipe.named_steps["pca"].n_components_)
    return pipe, val_pred, test_pred, info


# --------------------------------------------------------------- main
def main(models=None, pca_modes=(False, True)):
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    meta, X, feats, B = load_matrix()
    gc.collect()
    M = {k: meta.iloc[a:b] for k, (a, b) in B.items()}            # meta slices
    F = {k: frame(X, feats, r) for k, r in B.items()}              # zero-copy feature views
    y = {k: M[k]["target"].to_numpy() for k in B}
    train, val, test, final = M["train"], M["val"], M["test"], M["final"]
    print(f"features={len(feats)} train={len(train):,} val={len(val):,} test={len(test):,} "
          f"final={len(final):,}")

    # subsamples for the single-threaded sklearn boosters (same periods)
    fin_sub_idx = stratified_positions(meta, B["final"], SUBSAMPLE_N)
    tr_sub_idx = stratified_positions(meta, B["train"], SUBSAMPLE_N)
    results, log = {}, {}
    pv_out = val[ID_COLS].copy()
    pt_out = test[ID_COLS].copy()
    for k, v in baselines(train, val, F["val"]).items():
        pv_out[k] = v
    for k, v in baselines(final, test, F["test"]).items():
        pt_out[k] = v
        results[k] = {**regression_metrics(y["test"], v), "train_s": 0.0,
                      "infer_ms_per_1k": 0.0, "size_mb": 0.0}

    for use_pca in pca_modes:
        for name in models or MODEL_ORDER:
            key = name + ("+PCA" if use_pca else "")
            if name in SLOW_MODELS:
                Xtr, ytr = frame(X, feats, idx=tr_sub_idx), meta["target"].to_numpy()[tr_sub_idx]
                Xfin, yfin = frame(X, feats, idx=fin_sub_idx), meta["target"].to_numpy()[fin_sub_idx]
            else:
                Xtr, ytr, Xfin, yfin = F["train"], y["train"], F["final"], y["final"]
            print(f"\n=== {key} (final n={len(Xfin):,})", flush=True)
            pipe, val_pred, test_pred, info = tune_and_fit(
                name, use_pca, Xtr, ytr, F["val"], y["val"], Xfin, yfin, F["test"], log)
            del Xtr, Xfin
            met = regression_metrics(y["test"], test_pred)
            results[key] = {**met, **{k: v for k, v in info.items()
                                      if not isinstance(v, dict)}}
            print({k: round(v, 3) if isinstance(v, float) else v for k, v in results[key].items()},
                  flush=True)
            pt_out[key] = test_pred.astype(np.float32)
            if val_pred is not None:
                pv_out[key] = np.asarray(val_pred, dtype=np.float32)
            joblib.dump({"pipeline": pipe, "features": feats, "transform": TARGET_TRANSFORM,
                         "name": key, "info": info},
                        os.path.join(MODEL_DIR, f"{key}.joblib"), compress=3)
            if use_pca and name == (models or MODEL_ORDER)[0]:
                pca = pipe.named_steps["pca"]
                ev = pd.DataFrame({"component": np.arange(1, len(pca.explained_variance_ratio_) + 1),
                                   "explained_variance_ratio": pca.explained_variance_ratio_,
                                   "cumulative": np.cumsum(pca.explained_variance_ratio_)})
                ev.to_csv(os.path.join(RESULTS_DIR, "pca_explained_variance.csv"), index=False)
                # full spectrum (all components) for the scree plot
                from sklearn.decomposition import PCA
                rs = np.random.default_rng(SEED).choice(len(final), min(200_000, len(final)),
                                                        replace=False)
                Z = pipe.named_steps["scale"].transform(pipe.named_steps["impute"].transform(
                    frame(X, feats, idx=np.sort(rs))))
                full = PCA(random_state=SEED).fit(Z)
                pd.DataFrame({"component": np.arange(1, len(full.explained_variance_ratio_) + 1),
                              "explained_variance_ratio": full.explained_variance_ratio_,
                              "cumulative": np.cumsum(full.explained_variance_ratio_)}).to_csv(
                    os.path.join(RESULTS_DIR, "pca_full_spectrum.csv"), index=False)
            gc.collect()
            # write progressively so partial runs are still usable
            out = pd.DataFrame(results).T
            out.index.name = "model"
            bench_path = os.path.join(RESULTS_DIR, "benchmark.csv")
            if models and os.path.exists(bench_path):            # merge partial reruns
                old = pd.read_csv(bench_path, index_col=0)
                out = pd.concat([old.drop(index=out.index, errors="ignore"), out])
            out.to_csv(bench_path)
            pt_out.to_parquet(os.path.join(RESULTS_DIR, "predictions_test.parquet"), index=False)
            pv_out.to_parquet(os.path.join(RESULTS_DIR, "predictions_val.parquet"), index=False)
    # ---- simple ensemble of the gradient-boosting libraries (mean forecast)
    ens = [m for m in ["LightGBM", "XGBoost", "CatBoost"] if m in pt_out and m in results]
    if len(ens) >= 2:
        key = "Ensemble(LGBM+XGB+Cat)" if len(ens) == 3 else "Ensemble(" + "+".join(ens) + ")"
        pt_out[key] = pt_out[ens].mean(axis=1).astype(np.float32)
        if all(e in pv_out for e in ens):
            pv_out[key] = pv_out[ens].mean(axis=1).astype(np.float32)
        results[key] = {**regression_metrics(y["test"], pt_out[key]),
                        "train_s": sum(results[e]["train_s"] for e in ens),
                        "infer_ms_per_1k": sum(results[e]["infer_ms_per_1k"] for e in ens),
                        "size_mb": sum(results[e]["size_mb"] for e in ens), "n_train": len(final)}
        bench_path = os.path.join(RESULTS_DIR, "benchmark.csv")
        out = pd.DataFrame(results).T
        out.index.name = "model"
        if models and os.path.exists(bench_path):
            old = pd.read_csv(bench_path, index_col=0)
            out = pd.concat([old.drop(index=out.index, errors="ignore"), out])
        out.to_csv(bench_path)
        pt_out.to_parquet(os.path.join(RESULTS_DIR, "predictions_test.parquet"), index=False)
        pv_out.to_parquet(os.path.join(RESULTS_DIR, "predictions_val.parquet"), index=False)
    with open(os.path.join(RESULTS_DIR, "train_log.json"), "w") as fh:
        json.dump({**log, "features": feats}, fh, indent=1, default=str)
    print(pd.DataFrame(results).T[["MAE", "RMSE", "R2", "MAE_top5pct", "F1", "PR_AUC",
                                   "train_s"]].round(3).to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=None)
    ap.add_argument("--no-pca", action="store_true")
    a = ap.parse_args()
    main(a.models, (False,) if a.no_pca else (False, True))
