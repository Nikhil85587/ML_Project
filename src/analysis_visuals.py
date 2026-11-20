"""
Analysis B - Data & prediction visualisation, and
Analysis C - Model efficiency (training time, inference time, model size vs accuracy).

Also draws the benchmark / PCA / classification comparison charts.
Outputs: results/figures/viz_*.png, eff_*.png, bench_*.png, results/efficiency.csv
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import CITY_COORDS, FIG_DIR, HORIZON, POOR_THRESHOLD, PROCESSED, RESULTS_DIR, SEED  # noqa: E402

sns.set_theme(style="whitegrid")
MAIN = ["DecisionTree", "RandomForest", "AdaBoost", "GradientBoosting", "HistGradientBoosting",
        "XGBoost", "LightGBM", "CatBoost"]
BASE = ["Persistence", "SeasonalNaive7d", "Climatology"]


def save(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, f"{name}.png"), dpi=110)
    plt.close(fig)


def benchmark_charts(bench):
    order = [m for m in BASE + MAIN if m in bench.index]
    fig, axes = plt.subplots(1, 3, figsize=(20, 5.5))
    for ax, met in zip(axes, ["MAE", "RMSE", "R2"]):
        d = pd.DataFrame({"no PCA": bench.reindex(order)[met],
                          "with PCA": bench.reindex([m + "+PCA" if m in MAIN else m
                                                     for m in order])[met].to_numpy()},
                         index=order)
        d.loc[[m for m in order if m in BASE], "with PCA"] = np.nan
        d.plot.bar(ax=ax, width=0.8, color=["#3366aa", "#ee9944"])
        if "Persistence" in bench.index:
            ax.axhline(bench.loc["Persistence", met], color="red", ls="--", lw=1, label="Persistence")
        ax.set(title=f"Test {met} (24 h ahead)", xlabel="")
        ax.tick_params(axis="x", rotation=45)
        if met == "R2":
            ax.set_ylim(0, 1)
    axes[0].legend()
    save(fig, "bench_regression")
    fig, ax = plt.subplots(figsize=(10, 5))
    d = bench.reindex(order)[["MAE_top5pct", "MAE_top1pct"]]
    d.plot.bar(ax=ax, width=0.8)
    ax.set(title="Error on extremes: MAE on the top 5 % / 1 % of actual PM2.5", xlabel="")
    ax.tick_params(axis="x", rotation=45)
    save(fig, "bench_extremes")
    p = os.path.join(RESULTS_DIR, "pca_full_spectrum.csv")
    if os.path.exists(p):
        ev = pd.read_csv(p)
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.bar(ev["component"], ev["explained_variance_ratio"], color="#88aacc", label="per component")
        ax2 = ax.twinx()
        ax2.plot(ev["component"], ev["cumulative"], color="k", label="cumulative")
        ax2.axhline(0.95, color="red", ls="--", lw=1)
        k95 = int((ev["cumulative"] < 0.95).sum() + 1)
        ax2.axvline(k95, color="red", ls=":", lw=1)
        ax.set(title=f"PCA on the training period: {k95} of {len(ev)} components explain 95 % variance",
               xlabel="component", ylabel="explained variance ratio")
        ax2.set_ylabel("cumulative")
        save(fig, "bench_pca_scree")


def classification_charts():
    p = os.path.join(RESULTS_DIR, "classification_benchmark.csv")
    if not os.path.exists(p):
        return
    cb = pd.read_csv(p, index_col=0)
    fig, axes = plt.subplots(1, 2, figsize=(18, 5.5))
    piv = cb.pivot_table(index="family", columns="approach", values="F1")
    piv = piv.reindex([m for m in BASE + MAIN if m in piv.index])
    piv.plot.bar(ax=axes[0], width=0.8)
    axes[0].set(title="F1 for 'Poor or worse' at t+24: direct vs regression-derived", xlabel="")
    axes[0].tick_params(axis="x", rotation=45)
    piv = cb.pivot_table(index="family", columns="approach", values="PR_AUC")
    piv = piv.reindex([m for m in BASE + MAIN if m in piv.index])
    piv.plot.bar(ax=axes[1], width=0.8)
    axes[1].set(title="PR-AUC (threshold-free)", xlabel="")
    axes[1].tick_params(axis="x", rotation=45)
    save(fig, "bench_classification")
    sp = os.path.join(RESULTS_DIR, "cls_scores_test.parquet")
    if os.path.exists(sp):
        from sklearn.metrics import precision_recall_curve
        sc = pd.read_parquet(sp)
        pt = pd.read_parquet(os.path.join(RESULTS_DIR, "predictions_test.parquet"))
        y = (sc["target"] > POOR_THRESHOLD).astype(int)
        fig, ax = plt.subplots(figsize=(8, 6.5))
        for m in ["LightGBM", "XGBoost", "CatBoost", "RandomForest", "DecisionTree"]:
            if m in sc:
                p_, r_, _ = precision_recall_curve(y, sc[m])
                ax.plot(r_, p_, label=f"{m} (direct)")
        for m in ["LightGBM", "Persistence"]:
            if m in pt:
                p_, r_, _ = precision_recall_curve(y, pt[m].fillna(0))
                ax.plot(r_, p_, ls="--", label=f"{m} (regression score)")
        ax.axhline(y.mean(), color="grey", ls=":", label="no-skill")
        ax.set(xlabel="recall", ylabel="precision", title="Precision-recall: Poor-or-worse warnings (test)")
        ax.legend(fontsize=8)
        save(fig, "bench_pr_curves")


def efficiency(bench):
    rows = bench.reindex([m for m in MAIN + [x + "+PCA" for x in MAIN] if m in bench.index])
    eff = rows[["MAE", "RMSE", "R2", "train_s", "infer_ms_per_1k", "size_mb", "n_train"]].copy()
    eff.to_csv(os.path.join(RESULTS_DIR, "efficiency.csv"))
    base = eff.loc[[m for m in MAIN if m in eff.index]]
    fig, axes = plt.subplots(1, 3, figsize=(20, 5))
    for ax, col, lab in zip(axes, ["train_s", "infer_ms_per_1k", "size_mb"],
                            ["training time (s)", "inference time (ms / 1000 rows)", "model size (MB, compressed)"]):
        d = pd.DataFrame({"no PCA": base[col],
                          "with PCA": eff.reindex([m + "+PCA" for m in base.index])[col].to_numpy()},
                         index=base.index)
        d.plot.bar(ax=ax, width=0.8, logy=True, color=["#3366aa", "#ee9944"])
        ax.set(title=lab, xlabel="")
        ax.tick_params(axis="x", rotation=45)
    save(fig, "eff_costs")
    fig, axes = plt.subplots(1, 3, figsize=(20, 5.5))
    for ax, col, lab in zip(axes, ["train_s", "infer_ms_per_1k", "size_mb"],
                            ["training time (s)", "inference ms / 1000 rows", "model size (MB)"]):
        for idx, r in eff.iterrows():
            pca = "+PCA" in idx
            ax.scatter(r[col], r["MAE"], s=80, marker="s" if pca else "o",
                       color="#ee9944" if pca else "#3366aa")
            ax.annotate(idx.replace("GradientBoosting", "GB"), (r[col], r["MAE"]), fontsize=7,
                        xytext=(3, 3), textcoords="offset points")
        if "Persistence" in bench.index:
            ax.axhline(bench.loc["Persistence", "MAE"], color="red", ls="--", lw=1)
        ax.set_xscale("log")
        ax.set(xlabel=lab + " (log)", ylabel="test MAE", title=f"Accuracy vs {lab}")
    save(fig, "eff_tradeoff")


def prediction_viz(pt):
    """Predicted vs actual: scatter, episodes, seasonal/diurnal means, spatial errors."""
    models = [m for m in ["Persistence", "DecisionTree", "RandomForest", "XGBoost", "LightGBM",
                          "CatBoost"] if m in pt]
    smp = pt.sample(min(200_000, len(pt)), random_state=SEED)
    fig, axes = plt.subplots(2, 3, figsize=(17, 10))
    for ax, m in zip(axes.ravel(), models):
        hb = ax.hexbin(smp["target"].clip(upper=500), smp[m].clip(upper=500), gridsize=70,
                       bins="log", cmap="viridis")
        ax.plot([0, 500], [0, 500], "r--", lw=1)
        ax.axhline(POOR_THRESHOLD, color="orange", lw=0.7)
        ax.axvline(POOR_THRESHOLD, color="orange", lw=0.7)
        ax.set(title=m, xlabel="actual PM2.5 (t+24)", ylabel="forecast")
    fig.colorbar(hb, ax=axes, shrink=0.6, label="log count")
    fig.suptitle("Forecast vs actual (test); orange lines = Poor threshold")
    fig.savefig(os.path.join(FIG_DIR, "viz_pred_vs_actual.png"), dpi=110)
    plt.close(fig)

    T = pt["Datetime"] + pd.Timedelta(hours=HORIZON)
    episodes = [("DL031", "2019-10-15", "2019-11-20", "Delhi R K Puram - Diwali & stubble smog 2019"),
                ("UP012", "2019-12-01", "2020-01-15", "Lucknow Central School - winter episodes"),
                ("KA003", "2020-03-01", "2020-05-15", "Bengaluru BWSSB - COVID lockdown drop"),
                ("TN001", "2019-11-01", "2019-12-15", "Chennai Alandur - NE monsoon / smog transport")]
    fig, axes = plt.subplots(len(episodes), 1, figsize=(16, 4 * len(episodes)))
    for ax, (st, a, b, title) in zip(axes, episodes):
        m = (pt["StationId"] == st) & (T >= a) & (T <= b)
        if not m.any():
            continue
        d = pt[m].assign(T=T[m]).set_index("T")
        # reindex to a full hourly grid so lines break at data gaps
        d = d[~d.index.duplicated()].reindex(pd.date_range(d.index.min(), d.index.max(), freq="h"))
        ax.plot(d.index, d["target"], color="k", lw=1.6, label="actual")
        for mm, c in zip(["Persistence", "RandomForest", "XGBoost", "LightGBM", "CatBoost"],
                         ["#bbbbbb", "#55aa55", "#aa55aa", "#3366aa", "#ee7733"]):
            if mm in d:
                ax.plot(d.index, d[mm], color=c, lw=1, label=mm, alpha=0.9)
        ax.axhline(POOR_THRESHOLD, color="red", ls="--", lw=0.8)
        ax.set(title=f"{title}: 24 h-ahead forecasts vs actual", ylabel="PM2.5")
        ax.legend(ncol=6, fontsize=8, loc="upper right")
    save(fig, "viz_episodes")

    fig, axes = plt.subplots(1, 2, figsize=(16, 4.5))
    mon = pt.groupby(T.dt.to_period("M"))[["target"] + models].mean()
    mon.index = mon.index.to_timestamp()
    mon.plot(ax=axes[0], style=["k-"] + ["-"] * len(models), lw=1.4)
    axes[0].set(title="Monthly mean: actual vs forecasts (test)", xlabel="", ylabel="PM2.5")
    hr = pt.groupby(T.dt.hour)[["target"] + models].mean()
    hr.plot(ax=axes[1], style=["k-"] + ["-"] * len(models), lw=1.4)
    axes[1].set(title="Diurnal mean: actual vs forecasts (test)", xlabel="hour", ylabel="PM2.5")
    save(fig, "viz_seasonal_diurnal_fit")

    best = "LightGBM" if "LightGBM" in pt else models[-1]
    city = pt.groupby("City").apply(lambda d: pd.Series({
        "MAE": np.mean(np.abs(d[best] - d["target"])),
        "skill": 1 - np.mean(np.abs(d[best] - d["target"])) / np.mean(np.abs(d["Persistence"] - d["target"])),
        "n": len(d)}), include_groups=False)
    city["lat"] = city.index.map(lambda c: CITY_COORDS[c][0])
    city["lon"] = city.index.map(lambda c: CITY_COORDS[c][1])
    fig, ax = plt.subplots(figsize=(8, 8.5))
    sc = ax.scatter(city["lon"], city["lat"], c=city["skill"], s=40 + city["n"] / 1500,
                    cmap="RdYlGn", vmin=-0.1, vmax=0.4, edgecolor="k")
    for c, r in city.iterrows():
        ax.annotate(f"{c}\n{r['MAE']:.0f}", (r["lon"], r["lat"]), fontsize=7, xytext=(4, 2),
                    textcoords="offset points")
    plt.colorbar(sc, label=f"skill vs persistence (1 - MAE_{best}/MAE_persist)")
    ax.set(title=f"Spatial view of forecast skill ({best}); label = MAE", xlabel="lon", ylabel="lat")
    save(fig, "viz_spatial_skill")


def class_separation():
    """2-D PCA projection of the engineered features, coloured by the label."""
    from sklearn.decomposition import PCA
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import pyarrow.parquet as pq
    from data_prep import FEATURE_PREFIXES
    feats = [c for c in pq.read_schema(PROCESSED).names if c.startswith(FEATURE_PREFIXES)]
    df = pd.read_parquet(PROCESSED, columns=feats + ["target"])
    smp = df.sample(min(30_000, len(df)), random_state=SEED)
    del df
    Z = PCA(2, random_state=SEED).fit_transform(
        StandardScaler().fit_transform(SimpleImputer(strategy="median").fit_transform(smp[feats])))
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    y = smp["target"] > POOR_THRESHOLD
    axes[0].scatter(Z[~y, 0], Z[~y, 1], s=2, alpha=0.3, c="#55a84f", label="below Poor at t+24")
    axes[0].scatter(Z[y, 0], Z[y, 1], s=2, alpha=0.3, c="#e93f33", label="Poor or worse at t+24")
    axes[0].legend(markerscale=5)
    axes[0].set(title="Class separation in the first two principal components", xlabel="PC1", ylabel="PC2")
    sc = axes[1].scatter(smp["pm25_mean24"].clip(upper=400), smp["met_blh_max24"], s=2, alpha=0.4,
                         c=np.log1p(smp["target"]), cmap="inferno")
    plt.colorbar(sc, ax=axes[1], label="log(1 + PM2.5 at t+24)")
    axes[1].set(xlabel="PM2.5 mean of last 24 h", ylabel="max boundary-layer height, last 24 h (m)",
                title="Persistence x ventilation drive tomorrow's PM2.5")
    save(fig, "viz_class_separation")


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    bench = pd.read_csv(os.path.join(RESULTS_DIR, "benchmark.csv"), index_col=0)
    benchmark_charts(bench)
    efficiency(bench)
    classification_charts()
    pt = pd.read_parquet(os.path.join(RESULTS_DIR, "predictions_test.parquet"))
    prediction_viz(pt)
    class_separation()
    print("figures written to", FIG_DIR)


if __name__ == "__main__":
    main()
