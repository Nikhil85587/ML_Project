"""
Early-warning classification: will PM2.5 at t+24 be "Poor or worse" (> 90)?

Two approaches, compared on the same splits as the regression benchmark:
  * regression-derived : threshold the regression forecast (at the nominal 90
                         ug/m3 and at a threshold tuned for F1 on VAL)
  * direct classifier  : the same model families trained on the binary label,
                         decision threshold tuned for F1 on VAL
Final classifiers are refit on TRAIN+VAL and scored once on TEST.
Outputs: results/classification_benchmark.csv, results/cls_scores_test.parquet,
         models/cls_<name>.joblib
"""
import argparse
import gc
import os
import sys

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import MODEL_DIR, POOR_THRESHOLD, RESULTS_DIR  # noqa: E402
from models import (BOOSTERS, MODEL_ORDER, NATIVE_NAN, SLOW_MODELS, SUBSAMPLE_N,  # noqa: E402
                    Timer, best_f1_threshold, classification_metrics, early_stopping_fit,
                    frame, load_matrix, make_classifier, make_pipeline, model_size_mb,
                    staged_at, stratified_positions)


def regression_derived():
    """Classification from the saved regression forecasts."""
    pv = pd.read_parquet(os.path.join(RESULTS_DIR, "predictions_val.parquet"))
    pt = pd.read_parquet(os.path.join(RESULTS_DIR, "predictions_test.parquet"))
    rows = {}
    yv = (pv["target"] > POOR_THRESHOLD).astype(int)
    yt = (pt["target"] > POOR_THRESHOLD).astype(int)
    for col in pt.columns:
        if col in ("StationId", "City", "Datetime", "target", "target_poor") or "+PCA" in col:
            continue
        s = pt[col].fillna(pt[col].median())
        rows[f"{col} [reg@90]"] = {**classification_metrics(yt, s, POOR_THRESHOLD),
                                   "approach": "regression-derived (90)", "family": col}
        if col in pv:
            thr = best_f1_threshold(yv, pv[col].fillna(pv[col].median()))
            rows[f"{col} [reg@tuned]"] = {**classification_metrics(yt, s, thr),
                                          "approach": "regression-derived (tuned)", "family": col}
    return rows


def main(models=None):
    meta, X, feats, B = load_matrix()
    gc.collect()
    lab = meta["target_poor"].to_numpy().astype(int)
    F = {k: frame(X, feats, r) for k, r in B.items()}
    yl = {k: lab[a:b] for k, (a, b) in B.items()}
    yva, yte = yl["val"], yl["test"]
    test = meta.iloc[B["test"][0]:B["test"][1]]
    print(f"positive rate train={yl['train'].mean():.3f} val={yva.mean():.3f} test={yte.mean():.3f}")
    rows = regression_derived()
    scores = test[["StationId", "City", "Datetime", "target", "target_poor"]].copy()
    fin_sub_idx = stratified_positions(meta, B["final"], SUBSAMPLE_N)
    tr_sub_idx = stratified_positions(meta, B["train"], SUBSAMPLE_N)

    for name in models or MODEL_ORDER:
        print(f"\n=== classifier {name}", flush=True)
        native = name in NATIVE_NAN
        if name in SLOW_MODELS:
            Xtr, ytr = frame(X, feats, idx=tr_sub_idx), lab[tr_sub_idx]
            Xfin, yfin = frame(X, feats, idx=fin_sub_idx), lab[fin_sub_idx]
        else:
            Xtr, ytr, Xfin, yfin = F["train"], yl["train"], F["final"], yl["final"]
        Xva = F["val"]
        # ---- stage 1: tune on validation
        if name in BOOSTERS:
            m, best = early_stopping_fit(name, Xtr, ytr, Xva, yva, "cls")
            if name == "HistGradientBoosting":
                sv = staged_at(m, Xva, best, proba=True)[:, 1]
            else:
                sv = m.predict_proba(Xva)[:, 1]
            clf = make_classifier(name, int(best * 1.1) + 10)
        else:
            p = make_pipeline(make_classifier(name), native_nan=native)
            p.fit(Xtr, ytr)
            sv = p.predict_proba(Xva)[:, 1]
            del p
            clf = make_classifier(name)
        thr = best_f1_threshold(yva, sv)
        # ---- stage 2: refit on train+val, score test
        pipe = make_pipeline(clf, native_nan=native)
        with Timer() as tf:
            pipe.fit(Xfin, yfin)
        with Timer() as tp:
            st = pipe.predict_proba(F["test"])[:, 1]
        del Xtr, Xfin
        scores[name] = st.astype(np.float32)
        rows[f"{name} [direct]"] = {**classification_metrics(yte, st, thr),
                                    "approach": "direct classifier", "family": name,
                                    "train_s": tf.s, "infer_ms_per_1k": 1e6 * tp.s / len(test),
                                    "size_mb": model_size_mb(pipe)}
        print({k: round(v, 3) if isinstance(v, float) else v
               for k, v in rows[f"{name} [direct]"].items()}, flush=True)
        joblib.dump({"pipeline": pipe, "features": feats, "threshold": thr, "name": name},
                    os.path.join(MODEL_DIR, f"cls_{name}.joblib"), compress=3)
        out = pd.DataFrame(rows).T
        out.index.name = "model"
        out.to_csv(os.path.join(RESULTS_DIR, "classification_benchmark.csv"))
        scores.to_parquet(os.path.join(RESULTS_DIR, "cls_scores_test.parquet"), index=False)
        gc.collect()
    out = pd.DataFrame(rows).T
    print(out[["approach", "Precision", "Recall", "F1", "PR_AUC", "ROC_AUC"]].to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=None)
    main(ap.parse_args().models)
