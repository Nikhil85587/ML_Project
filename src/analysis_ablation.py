"""
Analysis A - Feature analysis & ablation.

For every model family: train on (a) all features, (b) all features minus one
group, (c) only one group (+ calendar), (d) all + oracle future weather at t+24
(perfect-forecast proxy - shows the value a real NWP forecast could add).
Plus feature selection: LightGBM/XGBoost/RandomForest/DecisionTree on the top-k
features ranked by LightGBM gain (computed on TRAIN only), and PCA with
different retained-variance levels.

Repeated fits use a stratified 250k-row subsample of TRAIN+VAL (all stations,
years and months represented) and are evaluated on the full TEST period.
Outputs: results/ablation_groups.csv, results/ablation_topk.csv,
         results/ablation_pca.csv
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import RESULTS_DIR  # noqa: E402
from data_prep import GROUP_NAMES, feature_group  # noqa: E402
from models import (MODEL_ORDER, Timer, fast_fit_predict, load_frame,  # noqa: E402
                    regression_metrics, stratified_subsample)

N_SUB = 250_000


def main():
    full, all_feats, S = load_frame(include_future=True)
    feats = [c for c in all_feats if not c.startswith("fut_")]
    fut = [c for c in all_feats if c.startswith("fut_")]
    train, test = S["train"], S["test"]
    fin = stratified_subsample(S["final"], N_SUB)
    groups = sorted({feature_group(f) for f in feats})
    configs = {"All features": feats}
    for g in groups:
        configs[f"- {GROUP_NAMES[g]}"] = [f for f in feats if feature_group(f) != g]
    configs["Only PM2.5 history"] = [f for f in feats if f.startswith("pm25_")]
    configs["PM2.5 history + calendar"] = [f for f in feats if f.startswith(("pm25_", "cal_"))]
    configs["No pollutant inputs (met+fire+cal+loc)"] = [
        f for f in feats if f.startswith(("met_", "fire_", "cal_", "loc_"))]
    if fut:
        configs["All + oracle weather at t+24"] = feats + fut

    rows = []
    out_path = os.path.join(RESULTS_DIR, "ablation_groups.csv")
    for cname, cols in configs.items():
        for name in MODEL_ORDER:
            with Timer() as t:
                pred = fast_fit_predict(name, fin[cols], fin["target"], test[cols])
            met = regression_metrics(test["target"], pred)
            rows.append({"config": cname, "model": name, "n_features": len(cols),
                         "fit_s": t.s, **met})
            print(cname, name, f"MAE={met['MAE']:.2f} R2={met['R2']:.3f} ({t.s:.0f}s)", flush=True)
        pd.DataFrame(rows).to_csv(out_path, index=False)

    # ---------- feature selection: top-k by LightGBM gain on TRAIN ----
    import lightgbm as lgb
    from train import fwd
    ranker = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.05, num_leaves=63,
                               importance_type="gain", verbose=-1)
    tr_sub = stratified_subsample(train, N_SUB)
    ranker.fit(tr_sub[feats], fwd(tr_sub["target"]))
    order = pd.Series(ranker.feature_importances_, index=feats).sort_values(ascending=False)
    order.to_csv(os.path.join(RESULTS_DIR, "feature_ranking_gain.csv"), header=["gain"])
    rows = []
    for k in [5, 10, 20, 40, 60, 80, len(feats)]:
        cols = list(order.index[:k])
        for name in ["DecisionTree", "RandomForest", "XGBoost", "LightGBM", "CatBoost"]:
            pred = fast_fit_predict(name, fin[cols], fin["target"], test[cols])
            rows.append({"k": k, "model": name, **regression_metrics(test["target"], pred)})
            print("top", k, name, f"MAE={rows[-1]['MAE']:.2f}", flush=True)
        pd.DataFrame(rows).to_csv(os.path.join(RESULTS_DIR, "ablation_topk.csv"), index=False)

    # ---------- PCA retained variance ---------------------------------
    rows = []
    for var in [0.80, 0.90, 0.95, 0.99]:
        for name in ["DecisionTree", "RandomForest", "LightGBM", "XGBoost"]:
            pred, ncomp = fast_fit_predict(name, fin[feats], fin["target"], test[feats],
                                           pca_var=var, return_ncomp=True)
            rows.append({"pca_var": var, "n_components": ncomp, "model": name,
                         **regression_metrics(test["target"], pred)})
            print("PCA", var, ncomp, name, f"MAE={rows[-1]['MAE']:.2f}", flush=True)
        pd.DataFrame(rows).to_csv(os.path.join(RESULTS_DIR, "ablation_pca.csv"), index=False)


if __name__ == "__main__":
    main()
