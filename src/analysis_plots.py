"""Figures for the ablation (A) and robustness (D) result tables."""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import FIG_DIR, RESULTS_DIR  # noqa: E402

sns.set_theme(style="whitegrid")
ORDER = ["DecisionTree", "RandomForest", "AdaBoost", "GradientBoosting", "HistGradientBoosting",
         "XGBoost", "LightGBM", "CatBoost"]


def r(name):
    p = os.path.join(RESULTS_DIR, name)
    return pd.read_csv(p) if os.path.exists(p) else None


def save(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, f"{name}.png"), dpi=110)
    plt.close(fig)


def ablation():
    ab = r("ablation_groups.csv")
    if ab is not None:
        piv = ab.pivot_table(index="config", columns="model", values="MAE")
        piv = piv[[m for m in ORDER if m in piv]]
        delta = piv - piv.loc["All features"]
        fig, axes = plt.subplots(1, 2, figsize=(22, 7))
        sns.heatmap(piv, annot=True, fmt=".2f", cmap="RdYlGn_r", ax=axes[0])
        axes[0].set_title("Test MAE (ug/m3) per feature configuration")
        sns.heatmap(delta.drop("All features"), annot=True, fmt="+.2f", cmap="RdBu_r", center=0,
                    vmin=-4, vmax=4, ax=axes[1])
        axes[1].set_title("Change in MAE vs all features (+ = worse without / with only this group)")
        for ax in axes:
            ax.set_ylabel("")
        save(fig, "abl_groups")
    tk, pc = r("ablation_topk.csv"), r("ablation_pca.csv")
    if tk is not None or pc is not None:
        fig, axes = plt.subplots(1, 2, figsize=(16, 5))
        if tk is not None:
            for m, d in tk.groupby("model"):
                axes[0].plot(d["k"], d["MAE"], marker="o", label=m)
            axes[0].set(xscale="log", xlabel="number of top features (LightGBM gain ranking)",
                        ylabel="test MAE", title="Feature selection: top-k features")
            axes[0].legend()
        if pc is not None:
            for m, d in pc.groupby("model"):
                axes[1].plot(d["n_components"], d["MAE"], marker="s", label=m)
                for _, row in d.iterrows():
                    axes[1].annotate(f"{row['pca_var']:.0%}", (row["n_components"], row["MAE"]), fontsize=7)
            axes[1].set(xlabel="PCA components (label = retained variance)", ylabel="test MAE",
                        title="PCA: retained variance vs accuracy")
            axes[1].legend()
        save(fig, "abl_topk_pca")


def robustness():
    nz, ms = r("robust_noise.csv"), r("robust_missing.csv")
    if nz is not None and ms is not None:
        fig, axes = plt.subplots(1, 2, figsize=(16, 5))
        for m, d in nz.groupby("model"):
            axes[0].plot(d["noise_sigma"] * 100, d["MAE"], marker="o", label=m)
        axes[0].set(xlabel="multiplicative sensor noise on pollutant inputs (sigma, %)",
                    ylabel="MAE", title="Robustness to sensor noise")
        for m, d in ms.groupby("model"):
            axes[1].plot(d["missing_rate"] * 100, d["MAE"], marker="o", label=m)
        axes[1].set(xlabel="extra missing pollutant inputs (%)", ylabel="MAE",
                    title="Robustness to missing data")
        axes[1].legend(fontsize=8)
        save(fig, "rob_noise_missing")
    ts = r("robust_training_size.csv")
    if ts is not None:
        fig, axes = plt.subplots(1, 2, figsize=(17, 5))
        for ax, (exp, d) in zip(axes, ts.groupby("experiment")):
            piv = d.pivot_table(index="setting", columns="model", values="MAE", sort=False)
            piv.plot(ax=ax, marker="o")
            ax.set(title=f"Less training data: {exp}", ylabel="test MAE", xlabel="")
        save(fig, "rob_training_size")
        st = ts[ts["experiment"] == "fewer stations"].dropna(subset=["MAE_unseen_stations"])
        if len(st):
            fig, ax = plt.subplots(figsize=(11, 4.5))
            m = st.melt(id_vars=["setting", "model"], value_vars=["MAE_seen_stations", "MAE_unseen_stations"])
            sns.barplot(data=m, x="model", y="value", hue="variable", errorbar=None, ax=ax)
            ax.set(title="Known vs new (unseen) stations, averaged over station fractions",
                   ylabel="MAE", xlabel="")
            save(fig, "rob_unseen_stations")
    lo = r("robust_loco.csv")
    if lo is not None:
        fig, ax = plt.subplots(figsize=(14, 5))
        d = lo.melt(id_vars=["model", "city"], value_vars=["MAE_city_in_training", "MAE_city_held_out",
                                                          "MAE_persistence"])
        best = lo.groupby("model")["MAE_city_held_out"].mean().idxmin()
        sns.barplot(data=d[d["model"] == best], x="city", y="value", hue="variable", ax=ax)
        ax.set(title=f"Leave-one-city-out ({best}): city in training vs held out vs persistence",
               ylabel="MAE", xlabel="")
        save(fig, "rob_loco")
    wf = r("robust_walk_forward.csv")
    if wf is not None:
        fig, ax = plt.subplots(figsize=(11, 4.5))
        wf.pivot_table(index="origin", columns="model", values="MAE").plot(ax=ax, marker="o")
        ax.set(title="Rolling-origin (walk-forward) validation: 6-month test windows",
               ylabel="MAE", xlabel="forecast origin")
        save(fig, "rob_walk_forward")
    ps = r("robust_period_shift.csv")
    if ps is not None:
        sel = ["Persistence", "Climatology", "RandomForest", "XGBoost", "LightGBM", "CatBoost"]
        d = ps[ps["model"].isin(sel)]
        fig, axes = plt.subplots(1, 2, figsize=(17, 5))
        sns.barplot(data=d, x="period", y="MAE", hue="model", ax=axes[0])
        axes[0].tick_params(axis="x", rotation=15)
        axes[0].set_title("MAE by test sub-period (distribution shift)")
        sns.barplot(data=d, x="period", y="Bias", hue="model", ax=axes[1])
        axes[1].tick_params(axis="x", rotation=15)
        axes[1].axhline(0, color="k", lw=0.8)
        axes[1].set_title("Mean bias by period (lockdown -> over-prediction)")
        save(fig, "rob_period")


if __name__ == "__main__":
    os.makedirs(FIG_DIR, exist_ok=True)
    ablation()
    robustness()
    print("plots done")
