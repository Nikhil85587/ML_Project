"""
Analysis F - Interpretability & explainability.

* SHAP (TreeExplainer) for DecisionTree, RandomForest, XGBoost, LightGBM,
  CatBoost -> per-feature and per-group mean |SHAP|, beeswarm, dependence plots
* Partial dependence (in ug/m3) on wind speed, boundary-layer height,
  humidity, temperature, rain, upwind fires, days-from-Diwali for all models
* Permutation importance by feature group for every model (drop in MAE)
* Physical-plausibility check of the partial-dependence slopes
Outputs: results/explain_*.csv, results/figures/xai_*.png
"""
import os
import sys

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import FIG_DIR, MODEL_DIR, RESULTS_DIR, SEED  # noqa: E402
from data_prep import GROUP_NAMES, feature_group  # noqa: E402
from models import MODEL_ORDER, load_frame  # noqa: E402
from train import inv  # noqa: E402

sns.set_theme(style="whitegrid")
PDP_FEATURES = {"met_ws": ("wind speed at t (m/s)", "-"),
                "met_blh_max24": ("max boundary-layer height, last 24 h (m)", "-"),
                "met_vent_min24": ("min ventilation coefficient, last 24 h (m2/s)", "-"),
                "met_rh": ("relative humidity at t (%)", "?"),
                "met_temp": ("temperature at t (C)", "-"),
                "met_precip_sum24": ("rain, last 24 h (mm)", "-"),
                "fire_up500_d1": ("upwind fires within 500 km, previous day", "+"),
                "cal_days_from_diwali": ("days from Diwali (target date)", "peak")}
SHAP_MODELS = ["DecisionTree", "RandomForest", "XGBoost", "LightGBM", "CatBoost"]


def save(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, f"xai_{name}.png"), dpi=110)
    plt.close(fig)


def load_models():
    out = {}
    for name in MODEL_ORDER:
        p = os.path.join(MODEL_DIR, f"{name}.joblib")
        if os.path.exists(p):
            out[name] = joblib.load(p)
    return out


def shap_values(name, bundle, X):
    """Return SHAP values in model (transformed-target) space."""
    import shap
    pipe = bundle["pipeline"]
    model = pipe.named_steps["model"]
    Z = X
    if "impute" in pipe.named_steps:
        Z = pd.DataFrame(pipe.named_steps["impute"].transform(X), columns=X.columns, index=X.index)
    if name == "CatBoost":
        from catboost import Pool
        vals = model.get_feature_importance(Pool(Z), type="ShapValues")
        return vals[:, :-1], Z
    expl = shap.TreeExplainer(model)
    return expl.shap_values(Z, check_additivity=False), Z


def main():
    import shap
    os.makedirs(FIG_DIR, exist_ok=True)
    _, _, S = load_frame()
    test = S["test"]
    models = load_models()
    feats = next(iter(models.values()))["features"]
    rng = np.random.default_rng(SEED)
    smp = test.sample(min(4000, len(test)), random_state=SEED)
    X = smp[feats]

    # ---------------- SHAP --------------------------------------------
    imp_rows, group_rows = {}, {}
    sv_store = {}
    for name in SHAP_MODELS:
        if name not in models:
            continue
        n = 600 if name == "RandomForest" else len(X)
        sv, Z = shap_values(name, models[name], X.iloc[:n])
        sv_store[name] = (sv, Z)
        imp = pd.Series(np.abs(sv).mean(axis=0), index=feats)
        imp_rows[name] = imp / imp.sum()
        group_rows[name] = imp_rows[name].groupby([feature_group(f) for f in feats]).sum()
        print("SHAP", name, "top:", list(imp.sort_values(ascending=False).index[:5]), flush=True)
    imp = pd.DataFrame(imp_rows)
    imp.to_csv(os.path.join(RESULTS_DIR, "explain_shap_importance.csv"))
    grp = pd.DataFrame(group_rows)
    grp.index = [GROUP_NAMES.get(g, g) for g in grp.index]
    grp.to_csv(os.path.join(RESULTS_DIR, "explain_shap_groups.csv"))

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    top = imp.mean(axis=1).sort_values(ascending=False).index[:20]
    imp.loc[top[::-1]].plot.barh(ax=axes[0], width=0.8)
    axes[0].set(title="Normalised mean |SHAP| - top 20 features, by model", xlabel="share of total")
    grp.sort_values(grp.columns[-1]).plot.barh(ax=axes[1], width=0.8)
    axes[1].set(title="SHAP importance by feature group", xlabel="share of total")
    save(fig, "shap_importance_models")

    best = "LightGBM" if "LightGBM" in sv_store else next(iter(sv_store))
    sv, Z = sv_store[best]
    plt.figure()
    shap.summary_plot(sv, Z, max_display=22, show=False)
    plt.title(f"SHAP beeswarm - {best} (log-space contributions)")
    plt.gcf().set_size_inches(10, 9)
    save(plt.gcf(), f"shap_beeswarm_{best}")
    fig, axes = plt.subplots(2, 4, figsize=(20, 9))
    for ax, f in zip(axes.ravel(), PDP_FEATURES):
        if f in Z:
            i = feats.index(f)
            ax.scatter(Z[f], sv[:, i], s=4, alpha=0.4, c=Z["pm25_lag0"].clip(upper=300), cmap="viridis")
            ax.axhline(0, color="k", lw=0.6)
            ax.set(xlabel=PDP_FEATURES[f][0], ylabel="SHAP value", title=f)
    fig.suptitle(f"SHAP dependence ({best}); colour = current PM2.5")
    save(fig, f"shap_dependence_{best}")

    # ---------------- Partial dependence in ug/m3 -------------------
    pdp_rows = []
    pdp_X = test.sample(min(1500, len(test)), random_state=SEED + 1)[feats]
    fig, axes = plt.subplots(2, 4, figsize=(20, 9))
    for ax, (f, (lab, expect)) in zip(axes.ravel(), PDP_FEATURES.items()):
        if f not in feats:
            continue
        grid = np.unique(np.nanquantile(pdp_X[f], np.linspace(0.02, 0.98, 15)))
        for name, b in models.items():
            curve = []
            for v in grid:
                Xg = pdp_X.copy()
                Xg[f] = v
                curve.append(inv(b["pipeline"].predict(Xg), b["transform"]).mean())
            ax.plot(grid, curve, label=name, lw=2 if name in ("LightGBM", "XGBoost") else 1)
            slope = np.polyfit(grid, curve, 1)[0] if len(grid) > 2 else np.nan
            pdp_rows.append({"feature": f, "model": name, "expected_sign": expect,
                             "slope_ug_per_unit": slope, "pd_min": min(curve), "pd_max": max(curve),
                             "pd_range": max(curve) - min(curve)})
        ax.set(xlabel=lab, ylabel="mean forecast PM2.5", title=f"PDP: {f} (expected {expect})")
    axes[0, 0].legend(fontsize=7)
    save(fig, "pdp_models")
    pdp = pd.DataFrame(pdp_rows)

    def plausible(r):
        if r["expected_sign"] == "-":
            return r["slope_ug_per_unit"] < 0
        if r["expected_sign"] == "+":
            return r["slope_ug_per_unit"] > 0
        return np.nan
    pdp["physically_plausible"] = pdp.apply(plausible, axis=1)
    pdp.to_csv(os.path.join(RESULTS_DIR, "explain_pdp_physical_check.csv"), index=False)

    # ---------------- permutation importance by group ----------------
    perm_X = test.sample(min(40_000, len(test)), random_state=SEED + 2)
    y = perm_X["target"].to_numpy()
    groups = sorted({feature_group(f) for f in feats})
    rows = []
    for name, b in models.items():
        base = np.mean(np.abs(inv(b["pipeline"].predict(perm_X[feats]), b["transform"]) - y))
        for g in groups:
            cols = [f for f in feats if feature_group(f) == g]
            Xp = perm_X[feats].copy()
            idx = rng.permutation(len(Xp))
            Xp[cols] = Xp[cols].to_numpy()[idx]           # shuffle the group jointly
            mae = np.mean(np.abs(inv(b["pipeline"].predict(Xp), b["transform"]) - y))
            rows.append({"model": name, "group": GROUP_NAMES[g], "base_MAE": base,
                         "permuted_MAE": mae, "MAE_increase": mae - base})
        print("permutation", name, flush=True)
    perm = pd.DataFrame(rows)
    perm.to_csv(os.path.join(RESULTS_DIR, "explain_permutation_groups.csv"), index=False)
    piv = perm.pivot(index="group", columns="model", values="MAE_increase")
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.heatmap(piv[[m for m in MODEL_ORDER if m in piv]], annot=True, fmt=".1f", cmap="Reds", ax=ax)
    ax.set_title("Permutation importance: MAE increase when a feature group is shuffled (ug/m3)")
    save(fig, "permutation_groups")
    print(pdp.groupby("feature")["physically_plausible"].mean())


if __name__ == "__main__":
    main()
