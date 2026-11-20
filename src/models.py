"""Model factory, metrics and shared helpers for training and analysis scripts."""
import io
import os
import sys
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import (AdaBoostClassifier, AdaBoostRegressor,
                              GradientBoostingClassifier, GradientBoostingRegressor,
                              HistGradientBoostingClassifier, HistGradientBoostingRegressor,
                              RandomForestClassifier, RandomForestRegressor)
from sklearn.impute import SimpleImputer
from sklearn.metrics import (average_precision_score, f1_score, mean_absolute_error,
                             mean_squared_error, precision_score, r2_score, recall_score,
                             roc_auc_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import POOR_THRESHOLD, PROCESSED, SEED  # noqa: E402

N_JOBS = max(1, (os.cpu_count() or 4) - 2)

# Models whose training cost grows badly with n (single-threaded sklearn
# boosting) are trained on a stratified subsample of the same training period.
SUBSAMPLE_N = 150_000          # rows for AdaBoost / sklearn GradientBoosting
TUNE_N = 250_000               # rows for the decision-tree hyper-parameter grid
SLOW_MODELS = {"AdaBoost", "GradientBoosting"}
MODEL_ORDER = ["DecisionTree", "RandomForest", "AdaBoost", "GradientBoosting",
               "HistGradientBoosting", "XGBoost", "LightGBM", "CatBoost"]
BOOSTERS = {"HistGradientBoosting", "XGBoost", "LightGBM", "CatBoost"}


# --------------------------------------------------------------- data helpers
def load_processed(columns=None):
    return pd.read_parquet(PROCESSED, columns=columns)


META_COLS = ["StationId", "City", "Datetime", "target", "target_poor"]


def load_matrix(include_future=False, extra=()):
    """Memory-lean loader. Returns (meta, X, feats, bounds):
    meta   - id/target columns, rows sorted by forecast-issue time
    X      - float32 feature matrix aligned with meta (read column-chunk-wise)
    bounds - row ranges of the chronological splits. Because rows are sorted by
             time, every split is a contiguous slice (views, no copies):
             train = issue time <= TRAIN_END - 24 h; val = (TRAIN_END, VAL_END - 24 h];
             test = > VAL_END; final (refit set) = issue time <= VAL_END - 24 h.
    """
    import pyarrow.parquet as pq
    from config import HORIZON, TRAIN_END, VAL_END
    from data_prep import FEATURE_PREFIXES
    names = pq.read_schema(PROCESSED).names
    pref = FEATURE_PREFIXES + (("fut_",) if include_future else ())
    feats = [c for c in names if c.startswith(pref)]
    meta = pd.read_parquet(PROCESSED, columns=META_COLS + [c for c in extra if c not in META_COLS])
    order = np.argsort(meta["Datetime"].to_numpy(), kind="stable")
    meta = meta.iloc[order].reset_index(drop=True)
    X = np.empty((len(meta), len(feats)), dtype=np.float32)
    for i in range(0, len(feats), 16):
        cols = feats[i:i + 16]
        X[:, i:i + len(cols)] = pd.read_parquet(PROCESSED, columns=cols).to_numpy(np.float32)[order]
    t = meta["Datetime"].to_numpy()
    gap = np.timedelta64(HORIZON, "h")
    te, ve = np.datetime64(pd.Timestamp(TRAIN_END)), np.datetime64(pd.Timestamp(VAL_END))
    ss = lambda v: int(np.searchsorted(t, v, side="right"))  # noqa: E731
    bounds = {"train": (0, ss(te - gap)), "val": (ss(te), ss(ve - gap)),
              "test": (ss(ve), len(t)), "final": (0, ss(ve - gap))}
    return meta, X, feats, bounds


def frame(X, feats, rng=None, idx=None):
    """Zero-copy DataFrame over a row range of X (or a copy for index arrays)."""
    if idx is not None:
        return pd.DataFrame(X[idx], columns=feats, copy=False)
    a, b = rng if rng is not None else (0, len(X))
    return pd.DataFrame(X[a:b], columns=feats, copy=False)


def load_frame(include_future=False):
    """Convenience for the analysis scripts: one DataFrame (features + meta),
    sorted by time, and split slices returned as iloc views."""
    meta, X, feats, B = load_matrix(include_future)
    full = pd.DataFrame(X, columns=feats, copy=False)
    for c in META_COLS:
        full[c] = meta[c].to_numpy()
    splits = {k: full.iloc[a:b] for k, (a, b) in B.items()}
    return full, feats, splits


def stratified_positions(meta, rng, n, seed=SEED):
    """Row positions (absolute) of a station x year x month-stratified sample of
    n rows from the slice `rng` of meta."""
    a, b = rng
    sub = meta.iloc[a:b]
    if len(sub) <= n:
        return np.arange(a, b)
    key = [sub["StationId"], sub["Datetime"].dt.year, sub["Datetime"].dt.month]
    pos = sub.groupby(key, group_keys=False).sample(frac=n / len(sub), random_state=seed).index
    return np.sort(pos.to_numpy())


def stratified_subsample(df, n, seed=SEED):
    """Sample n rows preserving the station x year x month composition, so all
    stations, seasons and years stay represented in proportion."""
    if len(df) <= n:
        return df
    frac = n / len(df)
    key = [df["StationId"], df["Datetime"].dt.year, df["Datetime"].dt.month]
    return df.groupby(key, group_keys=False).sample(frac=frac, random_state=seed)


# --------------------------------------------------------------- factories
def make_regressor(name, n_estimators=None, fast=False):
    """Return an unfitted regressor. `fast=True` gives lighter settings used in
    the repeated-fit analyses (ablation, robustness)."""
    from catboost import CatBoostRegressor
    from lightgbm import LGBMRegressor
    from xgboost import XGBRegressor
    if name == "DecisionTree":
        return DecisionTreeRegressor(max_depth=14, min_samples_leaf=100, random_state=SEED)
    if name == "RandomForest":
        return RandomForestRegressor(n_estimators=40 if fast else 120, max_depth=22,
                                     min_samples_leaf=20, max_features=0.33,
                                     max_samples=0.1 if fast else 0.2,
                                     n_jobs=N_JOBS, random_state=SEED)
    if name == "AdaBoost":
        return AdaBoostRegressor(DecisionTreeRegressor(max_depth=8, min_samples_leaf=100,
                                                       max_features=0.5),
                                 n_estimators=20 if fast else 50, learning_rate=0.1,
                                 loss="linear", random_state=SEED)
    if name == "GradientBoosting":
        return GradientBoostingRegressor(n_estimators=80 if fast else 200, max_depth=6,
                                         learning_rate=0.1, subsample=0.5,
                                         max_features=0.3, min_samples_leaf=100,
                                         random_state=SEED)
    if name == "HistGradientBoosting":
        return HistGradientBoostingRegressor(max_iter=n_estimators or 1500, learning_rate=0.05,
                                             max_leaf_nodes=63, min_samples_leaf=300,
                                             l2_regularization=10.0, max_features=0.5,
                                             early_stopping=False, random_state=SEED)
    if name == "XGBoost":
        return XGBRegressor(n_estimators=n_estimators or 4000, learning_rate=0.03,
                            max_depth=8, min_child_weight=50, subsample=0.8,
                            colsample_bytree=0.5, reg_lambda=10.0, tree_method="hist",
                            max_bin=256, n_jobs=N_JOBS, random_state=SEED)
    if name == "LightGBM":
        return LGBMRegressor(n_estimators=n_estimators or 4000, learning_rate=0.03,
                             num_leaves=63, min_child_samples=300, subsample=0.8,
                             subsample_freq=1, colsample_bytree=0.5, reg_lambda=10.0,
                             n_jobs=N_JOBS, random_state=SEED, verbose=-1)
    if name == "CatBoost":
        return CatBoostRegressor(iterations=n_estimators or 4000, learning_rate=0.06,
                                 depth=8, l2_leaf_reg=5, rsm=0.5, loss_function="RMSE",
                                 thread_count=N_JOBS, random_seed=SEED, verbose=0)
    raise ValueError(name)


def make_classifier(name, n_estimators=None, pos_weight=1.0):
    from catboost import CatBoostClassifier
    from lightgbm import LGBMClassifier
    from xgboost import XGBClassifier
    if name == "DecisionTree":
        return DecisionTreeClassifier(max_depth=14, min_samples_leaf=100, random_state=SEED)
    if name == "RandomForest":
        return RandomForestClassifier(n_estimators=120, max_depth=22, min_samples_leaf=20,
                                      max_features=0.33, max_samples=0.2,
                                      n_jobs=N_JOBS, random_state=SEED)
    if name == "AdaBoost":
        return AdaBoostClassifier(DecisionTreeClassifier(max_depth=6, min_samples_leaf=100,
                                                         max_features=0.5),
                                  n_estimators=60, learning_rate=0.2, random_state=SEED)
    if name == "GradientBoosting":
        return GradientBoostingClassifier(n_estimators=200, max_depth=6, learning_rate=0.1,
                                          subsample=0.5, max_features=0.3,
                                          min_samples_leaf=100, random_state=SEED)
    if name == "HistGradientBoosting":
        return HistGradientBoostingClassifier(max_iter=n_estimators or 1500, learning_rate=0.05,
                                              max_leaf_nodes=63, min_samples_leaf=300,
                                              l2_regularization=10.0, max_features=0.5,
                                              early_stopping=False, random_state=SEED)
    if name == "XGBoost":
        return XGBClassifier(n_estimators=n_estimators or 4000, learning_rate=0.03,
                             max_depth=8, min_child_weight=50, subsample=0.8,
                             colsample_bytree=0.5, reg_lambda=10.0, tree_method="hist",
                             eval_metric="aucpr", n_jobs=N_JOBS, random_state=SEED)
    if name == "LightGBM":
        return LGBMClassifier(n_estimators=n_estimators or 4000, learning_rate=0.03,
                              num_leaves=63, min_child_samples=300, subsample=0.8,
                              subsample_freq=1, colsample_bytree=0.5, reg_lambda=10.0,
                              n_jobs=N_JOBS, random_state=SEED, verbose=-1)
    if name == "CatBoost":
        return CatBoostClassifier(iterations=n_estimators or 4000, learning_rate=0.06,
                                  depth=8, l2_leaf_reg=5, rsm=0.5, thread_count=N_JOBS,
                                  random_seed=SEED, verbose=0)
    raise ValueError(name)


def make_pipeline(model, use_pca=False, native_nan=False, pca_var=0.95):
    """sklearn models get median imputation (fit on the training period); the
    PCA variants also standardise and project onto the components explaining
    `pca_var` of the training variance."""
    steps = []
    if use_pca or not native_nan:
        steps.append(("impute", SimpleImputer(strategy="median")))
    if use_pca:
        steps += [("scale", StandardScaler()), ("pca", PCA(n_components=pca_var, random_state=SEED))]
    steps.append(("model", model))
    return Pipeline(steps)


NATIVE_NAN = {"HistGradientBoosting", "XGBoost", "LightGBM", "CatBoost"}


def early_stopping_fit(name, X_tr, y_tr, X_va, y_va, task="reg", max_iter=4000, patience=150):
    """Fit a booster on train with early stopping on validation; return the
    best number of iterations. Inputs are numpy arrays (already transformed)."""
    if name == "LightGBM":
        import lightgbm as lgb
        m = (make_regressor if task == "reg" else make_classifier)(name, max_iter)
        m.fit(X_tr, y_tr, eval_set=[(X_va, y_va)],
              callbacks=[lgb.early_stopping(patience, verbose=False)])
        return m, m.best_iteration_ or max_iter
    if name == "XGBoost":
        m = (make_regressor if task == "reg" else make_classifier)(name, max_iter)
        m.set_params(early_stopping_rounds=patience)
        m.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
        return m, (m.best_iteration or max_iter) + 1
    if name == "CatBoost":
        m = (make_regressor if task == "reg" else make_classifier)(name, max_iter)
        m.fit(X_tr, y_tr, eval_set=(X_va, y_va), early_stopping_rounds=patience)
        return m, m.get_best_iteration() + 1
    if name == "HistGradientBoosting":
        # sklearn HGB has no external eval set: track validation loss over a
        # staged prediction instead
        m = (make_regressor if task == "reg" else make_classifier)(name, 1200)
        m.fit(X_tr, y_tr)
        if task == "reg":
            losses = [mean_squared_error(y_va, p) for p in m.staged_predict(X_va)]
        else:
            from sklearn.metrics import log_loss
            losses = [log_loss(y_va, p[:, 1]) for p in m.staged_predict_proba(X_va)]
        return m, int(np.argmin(losses)) + 1
    raise ValueError(name)


def staged_at(model, X, it, proba=False):
    """Prediction of a fitted HistGradientBoosting model after `it` iterations
    (streams the stages instead of materialising all of them)."""
    import itertools
    gen = model.staged_predict_proba(X) if proba else model.staged_predict(X)
    return next(itertools.islice(gen, it - 1, None))


# fixed boosting rounds for the repeated-fit analyses (no early stopping, so
# that ablation / robustness runs never look at the test period)
FAST_ROUNDS = {"HistGradientBoosting": 400, "XGBoost": 700, "LightGBM": 700, "CatBoost": 1000}


def fast_fit_predict(name, X, y, X_test, pca_var=None, return_ncomp=False, transform=None):
    """Fit a lighter version of `name` on (X, y) and predict X_test (ug/m3).
    Used by the ablation, robustness and efficiency analyses."""
    from train import fwd, inv
    native = name in NATIVE_NAN
    model = make_regressor(name, FAST_ROUNDS.get(name), fast=True)
    pipe = make_pipeline(model, use_pca=pca_var is not None, native_nan=native,
                         pca_var=pca_var or 0.95)
    pipe.fit(X, fwd(np.asarray(y, dtype=float)))
    pred = inv(pipe.predict(X_test))
    if return_ncomp:
        return pred, int(pipe.named_steps["pca"].n_components_) if pca_var else X.shape[1]
    return pred


# --------------------------------------------------------------- metrics
def regression_metrics(y_true, y_pred, threshold=POOR_THRESHOLD):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    p95, p99 = np.percentile(y_true, [95, 99])
    top5, top1 = y_true >= p95, y_true >= p99
    ct, cp = y_true > threshold, y_pred > threshold
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": mean_squared_error(y_true, y_pred) ** 0.5,
        "R2": r2_score(y_true, y_pred),
        "MAE_top5pct": mean_absolute_error(y_true[top5], y_pred[top5]),
        "MAE_top1pct": mean_absolute_error(y_true[top1], y_pred[top1]),
        "Bias_top5pct": float(np.mean(y_pred[top5] - y_true[top5])),
        "Precision": precision_score(ct, cp, zero_division=0),
        "Recall": recall_score(ct, cp, zero_division=0),
        "F1": f1_score(ct, cp, zero_division=0),
        "PR_AUC": average_precision_score(ct, y_pred),
    }


def classification_metrics(y_true, score, threshold):
    y_true = np.asarray(y_true).astype(int)
    pred = np.asarray(score) >= threshold
    return {
        "Precision": precision_score(y_true, pred, zero_division=0),
        "Recall": recall_score(y_true, pred, zero_division=0),
        "F1": f1_score(y_true, pred, zero_division=0),
        "PR_AUC": average_precision_score(y_true, score),
        "ROC_AUC": roc_auc_score(y_true, score),
        "Accuracy": float((pred == y_true).mean()),
        "Threshold": threshold,
    }


def best_f1_threshold(y_true, score):
    from sklearn.metrics import precision_recall_curve
    p, r, t = precision_recall_curve(np.asarray(y_true).astype(int), score)
    f1 = 2 * p * r / np.clip(p + r, 1e-12, None)
    i = int(np.nanargmax(f1[:-1]))
    return float(t[i])


def model_size_mb(obj):
    buf = io.BytesIO()
    joblib.dump(obj, buf, compress=3)
    return buf.tell() / 1e6


class Timer:
    def __enter__(self):
        self.t = time.perf_counter()
        return self

    def __exit__(self, *a):
        self.s = time.perf_counter() - self.t
