"""
Benchmark: persistence + seasonal baselines, Decision Tree, Random Forest,
boosting (AdaBoost, Gradient Boosting, XGBoost, LightGBM), with and without PCA.
"""
import time
import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import (AdaBoostRegressor, GradientBoostingRegressor,
                              RandomForestRegressor)
from sklearn.impute import SimpleImputer
from sklearn.metrics import (f1_score, mean_absolute_error, mean_squared_error,
                             precision_score, r2_score, recall_score,
                             average_precision_score)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

from data_prep import POOR_THRESHOLD, chrono_split, feature_columns

MODELS = {
    "DecisionTree": DecisionTreeRegressor(max_depth=12),
    "RandomForest": RandomForestRegressor(n_estimators=100, max_depth=15, n_jobs=-1),
    "AdaBoost": AdaBoostRegressor(n_estimators=100),
    "GradientBoosting": GradientBoostingRegressor(n_estimators=200),
    "XGBoost": XGBRegressor(n_estimators=300, learning_rate=0.05),
    "LightGBM": LGBMRegressor(n_estimators=300, learning_rate=0.05, verbose=-1),
}


def evaluate(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    top = y_true >= np.percentile(y_true, 95)
    cls_true, cls_pred = y_true > POOR_THRESHOLD, y_pred > POOR_THRESHOLD
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": mean_squared_error(y_true, y_pred) ** 0.5,
        "R2": r2_score(y_true, y_pred),
        "MAE_top5pct": mean_absolute_error(y_true[top], y_pred[top]),
        "Precision": precision_score(cls_true, cls_pred, zero_division=0),
        "Recall": recall_score(cls_true, cls_pred, zero_division=0),
        "F1": f1_score(cls_true, cls_pred, zero_division=0),
        "PR_AUC": average_precision_score(cls_true, y_pred),
    }


def main():
    df = pd.read_parquet("data/processed.parquet")
    feats = feature_columns(df)
    train, val, test = chrono_split(df)
    X_tr, y_tr = train[feats], train["target"]
    X_te, y_te = test[feats], test["target"]

    results = {}
    # baselines
    results["Persistence"] = evaluate(y_te, test["PM2.5_lag0"])           # t+24 = t
    results["Seasonal(24h mean)"] = evaluate(y_te, test["PM2.5_roll24_mean"])

    for use_pca in [False, True]:
        for name, model in MODELS.items():
            steps = [SimpleImputer(strategy="median"), StandardScaler()]
            if use_pca:
                steps.append(PCA(n_components=0.95))
            pipe = make_pipeline(*steps, model)

            t0 = time.time()
            pipe.fit(X_tr, y_tr)
            train_time = time.time() - t0
            t0 = time.time()
            pred = pipe.predict(X_te)
            infer_time = time.time() - t0

            key = f"{name}{'+PCA' if use_pca else ''}"
            results[key] = {**evaluate(y_te, pred),
                            "train_s": train_time, "infer_s": infer_time}
            joblib.dump(pipe, f"models/{key}.joblib")
            print(key, results[key])

            if use_pca and name == "DecisionTree":
                pca = pipe.named_steps["pca"]
                print("PCA components:", pca.n_components_,
                      "explained var:", pca.explained_variance_ratio_.sum())

    out = pd.DataFrame(results).T
    out.to_csv("results/benchmark.csv")
    print(out.round(3))


if __name__ == "__main__":
    main()
