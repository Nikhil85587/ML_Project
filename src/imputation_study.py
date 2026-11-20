"""
Compare imputation strategies two ways.

1. Reconstruction test: hide known PM2.5/PM10/NO2 readings (random single
   hours + contiguous 3-12 h gaps, mimicking real outages) and measure how well
   each method recovers them.
2. Downstream test: rebuild the full feature set with each strategy and train
   the same LightGBM forecaster (train -> val/test).

Strategies: ffill (causal), seasonal (causal, same hour yesterday / last week),
interp (linear interpolation - NON-causal, uses the reading after the gap).
Output: results/imputation_comparison.csv, results/imputation_reconstruction.csv
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import RESULTS_DIR, SEED  # noqa: E402
from data_prep import build_dataset, chrono_split, clean, feature_columns, impute, load_raw  # noqa: E402
from models import make_regressor, regression_metrics  # noqa: E402
from train import fwd, inv  # noqa: E402

METHODS = ["ffill", "seasonal", "interp"]


def reconstruction_test(cols=("PM2.5", "PM10", "NO2")):
    rng = np.random.default_rng(SEED)
    raw = clean(load_raw())
    rows = []
    for col in cols:
        known = np.flatnonzero(raw[col].notna().to_numpy())
        # 50% isolated hours, 50% inside synthetic 3-12 h gaps
        single = rng.choice(known, 20_000, replace=False)
        starts = rng.choice(known, 2_000, replace=False)
        gaps = np.concatenate([np.arange(s, s + rng.integers(3, 13)) for s in starts])
        gaps = gaps[gaps < len(raw)]
        for kind, idx in [("single hour", single), ("3-12 h gap", gaps)]:
            idx = np.intersect1d(idx, known)
            truth = raw[col].to_numpy()[idx]
            for m in METHODS:
                d = raw[["StationId", col]].copy()
                d.loc[d.index[idx], col] = np.nan
                d = impute(d, m, cols=[col])
                est = d[col].to_numpy()[idx]
                ok = ~np.isnan(est)
                rows.append({"pollutant": col, "gap_type": kind, "method": m,
                             "MAE": float(np.mean(np.abs(est[ok] - truth[ok]))),
                             "coverage": float(ok.mean())})
                print(rows[-1], flush=True)
    return pd.DataFrame(rows)


def downstream_test():
    rows = []
    for m in METHODS:
        df, _ = build_dataset(m, verbose=True)
        feats = feature_columns(df)
        tr, va, te = chrono_split(df)
        model = make_regressor("LightGBM", 400)
        model.fit(tr[feats], fwd(tr["target"]))
        for split, d in [("val", va), ("test", te)]:
            met = regression_metrics(d["target"], inv(model.predict(d[feats])))
            rows.append({"method": m, "split": split, **met})
            print(rows[-1], flush=True)
        del df, tr, va, te
    return pd.DataFrame(rows)


if __name__ == "__main__":
    rec = reconstruction_test()
    rec.to_csv(os.path.join(RESULTS_DIR, "imputation_reconstruction.csv"), index=False)
    down = downstream_test()
    down.to_csv(os.path.join(RESULTS_DIR, "imputation_comparison.csv"), index=False)
    print(rec.pivot_table(index=["pollutant", "gap_type"], columns="method", values="MAE").round(2))
    print(down.pivot_table(index="method", columns="split", values=["MAE", "RMSE", "R2"]).round(3))
