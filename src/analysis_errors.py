"""
Analysis E - Error analysis across all models (from the final TEST predictions).

Where and why do models fail? Error by season, hour of day, city, station,
pollution level (CPCB band of the actual value), severe spikes (top 1 %),
sudden onsets (large 24 h increases), data gaps; plus missed warnings and the
similarity of error patterns between models.
Outputs: results/errors_*.csv, results/figures/err_*.png
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
from config import FIG_DIR, HORIZON, POOR_THRESHOLD, PROCESSED, RESULTS_DIR  # noqa: E402

sns.set_theme(style="whitegrid")
SEASON = {12: "Winter", 1: "Winter", 2: "Winter", 3: "Pre-monsoon", 4: "Pre-monsoon",
          5: "Pre-monsoon", 6: "Monsoon", 7: "Monsoon", 8: "Monsoon", 9: "Monsoon",
          10: "Post-monsoon", 11: "Post-monsoon"}
BANDS = [0, 30, 60, 90, 120, 250, 2000]
BAND_LABELS = ["Good", "Satisfactory", "Moderate", "Poor", "Very Poor", "Severe"]


def save(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, f"err_{name}.png"), dpi=110)
    plt.close(fig)


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    pt = pd.read_parquet(os.path.join(RESULTS_DIR, "predictions_test.parquet"))
    extra = pd.read_parquet(PROCESSED, columns=["StationId", "Datetime", "pm25_lag0",
                                                "miss_pm25_hours_since", "nbr_city_n"])
    pt = pt.merge(extra, on=["StationId", "Datetime"], how="left")
    models = [c for c in pt.columns if c not in ("StationId", "City", "Datetime", "target",
                                                 "target_poor", "pm25_lag0",
                                                 "miss_pm25_hours_since", "nbr_city_n")
              and "+PCA" not in c]
    T = pt["Datetime"] + pd.Timedelta(hours=HORIZON)
    pt["season"] = T.dt.month.map(SEASON)
    pt["hour"] = T.dt.hour
    pt["band"] = pd.cut(pt["target"], BANDS, labels=BAND_LABELS, include_lowest=True)
    pt["jump"] = pt["target"] - pt["pm25_lag0"]
    for m in models:
        pt[m] = pt[m].fillna(pt[m].median())
    err = pd.DataFrame({m: pt[m] - pt["target"] for m in models})
    ae = err.abs()

    def by(key):
        return ae.groupby(pt[key], observed=True).mean()

    tables = {
        "season": by("season").T, "hour": by("hour").T, "band": by("band").T,
        "city": by("City").T,
    }
    bias_band = err.groupby(pt["band"], observed=True).mean().T
    tables["bias_band"] = bias_band
    for k, t in tables.items():
        t.to_csv(os.path.join(RESULTS_DIR, f"errors_by_{k}.csv"))

    main_models = [m for m in ["Persistence", "SeasonalNaive7d", "Climatology", "DecisionTree",
                               "RandomForest", "AdaBoost", "GradientBoosting",
                               "HistGradientBoosting", "XGBoost", "LightGBM", "CatBoost"]
                   if m in models]

    # --- heatmaps: model x season / band (MAE) and bias by band
    fig, axes = plt.subplots(1, 3, figsize=(22, 6))
    sns.heatmap(tables["season"].loc[main_models, ["Winter", "Pre-monsoon", "Monsoon", "Post-monsoon"]],
                annot=True, fmt=".1f", cmap="Reds", ax=axes[0])
    axes[0].set_title("MAE by season (target time)")
    sns.heatmap(tables["band"].loc[main_models], annot=True, fmt=".0f", cmap="Reds", ax=axes[1])
    axes[1].set_title("MAE by actual pollution band")
    sns.heatmap(bias_band.loc[main_models], annot=True, fmt=".0f", cmap="coolwarm", center=0, ax=axes[2])
    axes[2].set_title("Mean bias (pred - actual) by band: under-prediction of extremes")
    save(fig, "heatmaps_season_band")

    # --- hour of day
    fig, ax = plt.subplots(figsize=(11, 4.5))
    for m in main_models:
        lw = 2.5 if m in ("LightGBM", "CatBoost", "XGBoost", "Persistence") else 1
        tables["hour"].loc[m].plot(ax=ax, label=m, lw=lw)
    ax.set(title="MAE by hour of day of the target time", xlabel="hour (IST)", ylabel="MAE (ug/m3)")
    ax.legend(ncol=3, fontsize=8)
    save(fig, "by_hour")

    # --- city
    best = min([m for m in main_models if m not in ("Persistence", "SeasonalNaive7d", "Climatology")],
               key=lambda m: ae[m].mean())
    city = pd.DataFrame({"MAE_best": ae[best].groupby(pt["City"]).mean(),
                         "MAE_persistence": ae["Persistence"].groupby(pt["City"]).mean(),
                         "mean_PM25": pt.groupby("City")["target"].mean(),
                         "n": pt.groupby("City").size()})
    city["skill_vs_persistence"] = 1 - city["MAE_best"] / city["MAE_persistence"]
    city["relative_MAE"] = city["MAE_best"] / city["mean_PM25"]
    city.sort_values("MAE_best").to_csv(os.path.join(RESULTS_DIR, "errors_city_detail.csv"))
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    c2 = city.sort_values("MAE_best")
    axes[0].barh(c2.index, c2["MAE_persistence"], color="#bbbbbb", label="Persistence")
    axes[0].barh(c2.index, c2["MAE_best"], color="#3366aa", label=best)
    axes[0].set(title=f"MAE by city ({best} vs persistence)", xlabel="MAE (ug/m3)")
    axes[0].legend()
    axes[1].scatter(city["mean_PM25"], city["MAE_best"], s=city["n"] / 800, alpha=0.7)
    for c, r in city.iterrows():
        axes[1].annotate(c, (r["mean_PM25"], r["MAE_best"]), fontsize=7)
    axes[1].set(xlabel="mean PM2.5 in test period", ylabel="MAE",
                title="Absolute error scales with pollution level")
    save(fig, "by_city")

    # --- station (worst 15)
    st = ae[[best, "Persistence"]].groupby(pt["StationId"]).mean()
    st["n"] = pt.groupby("StationId").size()
    st["mean_PM25"] = pt.groupby("StationId")["target"].mean()
    st["hours_since_obs_mean"] = pt.groupby("StationId")["miss_pm25_hours_since"].mean()
    st.sort_values(best, ascending=False).to_csv(os.path.join(RESULTS_DIR, "errors_by_station.csv"))

    # --- spikes, sudden onsets, gaps
    p99 = pt["target"].quantile(0.99)
    cases = {
        "All": np.ones(len(pt), bool),
        "Top 1% (severe spikes)": pt["target"] >= p99,
        "Sudden rise (>+100 in 24h)": pt["jump"] > 100,
        "Sudden drop (<-100 in 24h)": pt["jump"] < -100,
        "PM2.5 imputed at issue time": pt["miss_pm25_hours_since"] > 0,
        "Single-station city": pt["nbr_city_n"].fillna(0) == 0,
        "Diwali window (+-3 d)": T.dt.normalize().isin(pd.to_datetime(
            ["2019-10-24", "2019-10-25", "2019-10-26", "2019-10-27", "2019-10-28",
             "2019-10-29", "2019-10-30"])),
    }
    rows = []
    for k, msk in cases.items():
        r = {"case": k, "n": int(np.sum(msk))}
        for m in main_models:
            r[m] = float(ae[m][msk].mean())
        rows.append(r)
    cs = pd.DataFrame(rows).set_index("case")
    cs.to_csv(os.path.join(RESULTS_DIR, "errors_by_case.csv"))
    fig, ax = plt.subplots(figsize=(13, 5))
    sel = [m for m in ["Persistence", "Climatology", "DecisionTree", "RandomForest", "XGBoost",
                       "LightGBM", "CatBoost"] if m in cs.columns]
    cs[sel].plot.bar(ax=ax, width=0.8)
    ax.set(title="MAE in difficult situations", ylabel="MAE (ug/m3)", xlabel="")
    ax.tick_params(axis="x", rotation=20)
    save(fig, "difficult_cases")

    # --- residual distribution & predicted vs actual band confusion for best model
    fig, axes = plt.subplots(1, 2, figsize=(15, 4.8))
    for m in ["Persistence", "RandomForest", best]:
        if m in err:
            axes[0].hist(err[m].clip(-200, 200), bins=160, histtype="step", density=True, label=m, lw=1.5)
    axes[0].set(title="Residual distribution (pred - actual)", xlabel="ug/m3")
    axes[0].legend()
    pb = pd.cut(pt[best], BANDS, labels=BAND_LABELS, include_lowest=True)
    cm = pd.crosstab(pt["band"], pb, normalize="index")
    sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues", ax=axes[1])
    axes[1].set(title=f"AQI-band confusion ({best}), rows = actual", xlabel="predicted", ylabel="actual")
    save(fig, "residuals_band_confusion")

    # --- missed warnings (false negatives) by month and city
    poor = pt["target"] > POOR_THRESHOLD
    fn = poor & (pt[best] <= POOR_THRESHOLD)
    miss = pd.DataFrame({"missed_warnings": fn.groupby(pt["City"]).sum(),
                         "poor_events": poor.groupby(pt["City"]).sum()})
    miss["miss_rate"] = miss["missed_warnings"] / miss["poor_events"].replace(0, np.nan)
    miss.sort_values("poor_events", ascending=False).to_csv(
        os.path.join(RESULTS_DIR, "errors_missed_warnings_city.csv"))

    # --- error correlation between models
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(err[main_models].sample(min(200_000, len(err[main_models])), random_state=0).corr(), annot=True, fmt=".2f",
                cmap="viridis", ax=ax)
    ax.set_title("Correlation of residuals between models (similar failure modes)")
    save(fig, "error_correlation")

    # --- worst individual forecasts
    worst = pt.assign(abs_err=ae[best]).nlargest(25, "abs_err")[
        ["StationId", "City", "Datetime", "pm25_lag0", "target", best, "abs_err"]]
    worst.to_csv(os.path.join(RESULTS_DIR, "errors_worst_forecasts.csv"), index=False)
    print("best model:", best)
    print(cs.round(1).to_string())


if __name__ == "__main__":
    main()
