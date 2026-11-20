"""
Exploratory data analysis -> results/figures/eda_*.png + results/eda_summary.json

Covers: stations / cities / time span / sampling, coverage over time, the
missingness pattern (by pollutant, station, season, year and pollution level),
seasonal, diurnal and spatial patterns, target distribution & class balance,
sensor faults, and relationships with meteorology, fires and festivals.
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (CATEGORY_COLORS, CITY_COORDS, FESTIVALS, FIG_DIR, POLLUTANTS,  # noqa: E402
                    POOR_THRESHOLD, RESULTS_DIR, pm25_category)
from data_prep import load_raw, load_weather  # noqa: E402

sns.set_theme(style="whitegrid", context="notebook")
SEASON = {12: "Winter", 1: "Winter", 2: "Winter", 3: "Pre-monsoon", 4: "Pre-monsoon",
          5: "Pre-monsoon", 6: "Monsoon", 7: "Monsoon", 8: "Monsoon", 9: "Monsoon",
          10: "Post-monsoon", 11: "Post-monsoon"}
SEASON_ORDER = ["Winter", "Pre-monsoon", "Monsoon", "Post-monsoon"]
NORTH = {"Delhi", "Gurugram", "Lucknow", "Patna", "Jaipur", "Amritsar", "Chandigarh"}


def save(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, f"eda_{name}.png"), dpi=110)
    plt.close(fig)


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    raw = load_raw()
    raw["year"] = raw["Datetime"].dt.year
    raw["month"] = raw["Datetime"].dt.month
    raw["hour"] = raw["Datetime"].dt.hour
    raw["season"] = raw["month"].map(SEASON)
    raw["region"] = np.where(raw["City"].isin(NORTH), "North (IGP)", "Rest of India")
    S = {}
    S["rows"] = int(len(raw))
    S["stations"] = int(raw["StationId"].nunique())
    S["cities"] = int(raw["City"].nunique())
    S["states"] = int(raw["State"].nunique())
    S["time_span"] = [str(raw["Datetime"].min()), str(raw["Datetime"].max())]
    S["sampling"] = "hourly (regular hourly grid per station, gaps stored as NaN)"
    S["units"] = {p: ("mg/m3" if p == "CO" else "ug/m3") for p in POLLUTANTS}
    S["missing_frac"] = raw[POLLUTANTS].isna().mean().round(4).to_dict()
    S["pm25_describe"] = raw["PM2.5"].describe().round(2).to_dict()
    S["stations_per_city"] = raw.groupby("City")["StationId"].nunique().sort_values(
        ascending=False).to_dict()

    # 1 --- coverage: stations reporting PM2.5 per month
    cov = (raw.dropna(subset=["PM2.5"]).groupby(raw["Datetime"].dt.to_period("M"))["StationId"]
           .nunique())
    fig, ax = plt.subplots(figsize=(11, 3.5))
    cov.plot(ax=ax, color="#3366aa")
    ax.set(title="Number of stations reporting PM2.5 per month", xlabel="", ylabel="stations")
    save(fig, "coverage_over_time")

    # 2 --- missingness by pollutant x year and by pollutant x season
    miss_year = raw.groupby("year")[POLLUTANTS].apply(lambda d: d.isna().mean())
    miss_season = raw.groupby("season")[POLLUTANTS].apply(lambda d: d.isna().mean()).loc[SEASON_ORDER]
    fig, axes = plt.subplots(1, 2, figsize=(15, 4.5))
    sns.heatmap(miss_year.T, annot=True, fmt=".2f", cmap="Reds", ax=axes[0], cbar=False)
    axes[0].set_title("Missing fraction by pollutant and year")
    sns.heatmap(miss_season.T, annot=True, fmt=".2f", cmap="Reds", ax=axes[1], cbar=False)
    axes[1].set_title("Missing fraction by pollutant and season")
    save(fig, "missingness_pollutant_year_season")

    # 3 --- missingness by station (PM2.5), sorted
    ms = raw.groupby("StationId")["PM2.5"].apply(lambda s: s.isna().mean()).sort_values()
    fig, ax = plt.subplots(figsize=(16, 4))
    ms.plot.bar(ax=ax, color="#cc5544", width=0.85)
    ax.set(title="PM2.5 missing fraction by station (within each station's active period)",
           ylabel="missing fraction", xlabel="")
    ax.tick_params(axis="x", labelsize=6)
    save(fig, "missingness_by_station")

    # 4 --- station x month availability heatmap (PM2.5)
    avail = (raw.assign(ok=raw["PM2.5"].notna()).groupby(
        ["StationId", raw["Datetime"].dt.to_period("Q")])["ok"].mean().unstack())
    fig, ax = plt.subplots(figsize=(14, 14))
    sns.heatmap(avail, cmap="viridis", ax=ax, cbar_kws={"label": "fraction of hours with PM2.5"})
    ax.set(title="PM2.5 availability by station and quarter (white = station not yet installed)",
           xlabel="", ylabel="")
    ax.tick_params(axis="y", labelsize=6)
    save(fig, "availability_station_quarter")

    # 5 --- is missingness informative? PM2.5 just before gaps vs overall,
    #        and gap rate by previous-day pollution level
    g = raw.groupby("StationId", sort=False)["PM2.5"]
    prev24 = g.transform(lambda s: s.shift(1).rolling(24, min_periods=6).mean())
    is_miss = raw["PM2.5"].isna()
    lvl = pd.cut(prev24, [0, 30, 60, 90, 120, 250, 2000],
                 labels=["Good", "Satisf.", "Moderate", "Poor", "Very Poor", "Severe"])
    rate = is_miss.groupby(lvl, observed=True).mean()
    fig, ax = plt.subplots(figsize=(7, 4))
    rate.plot.bar(ax=ax, color="#aa4466")
    ax.set(title="P(PM2.5 missing) vs pollution level of previous 24 h",
           ylabel="missing probability", xlabel="previous-24 h PM2.5 band")
    save(fig, "missingness_vs_pollution")
    S["missing_rate_by_prev_level"] = rate.round(4).to_dict()

    # 6 --- seasonal & diurnal patterns
    fig, axes = plt.subplots(1, 2, figsize=(15, 4.5))
    sns.boxplot(data=raw.dropna(subset=["PM2.5"]).sample(400_000, random_state=0), x="month",
                y="PM2.5", hue="region", showfliers=False, ax=axes[0])
    axes[0].axhline(POOR_THRESHOLD, color="red", ls="--", lw=1, label="Poor threshold")
    axes[0].set_title("Monthly PM2.5 distribution (seasonality)")
    d = raw.groupby(["season", "hour"])["PM2.5"].mean().unstack(0)[SEASON_ORDER]
    d.plot(ax=axes[1])
    axes[1].set(title="Mean diurnal PM2.5 cycle by season", xlabel="hour of day (IST)",
                ylabel="PM2.5 (ug/m3)")
    save(fig, "seasonal_diurnal")

    # weekly pattern & yearly trend
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    raw.groupby(raw["Datetime"].dt.dayofweek)["PM2.5"].mean().plot.bar(ax=axes[0], color="#557799")
    axes[0].set(title="Mean PM2.5 by day of week (0=Mon)", xlabel="", ylabel="PM2.5")
    monthly = raw.set_index("Datetime").groupby("region")["PM2.5"].resample("MS").mean().unstack(0)
    monthly.plot(ax=axes[1])
    axes[1].axvspan(pd.Timestamp("2020-03-25"), pd.Timestamp("2020-05-31"), color="grey",
                    alpha=0.3, label="COVID lockdown")
    axes[1].legend()
    axes[1].set(title="Monthly mean PM2.5 by region (trend + lockdown dip)", xlabel="")
    save(fig, "weekly_trend")

    # 7 --- spatial pattern: city means on a lat/lon map + city correlations
    city = raw.groupby("City").agg(pm=("PM2.5", "mean"),
                                   poor=("PM2.5", lambda s: (s > POOR_THRESHOLD).mean()),
                                   n=("StationId", "nunique"))
    city["lat"] = city.index.map(lambda c: CITY_COORDS[c][0])
    city["lon"] = city.index.map(lambda c: CITY_COORDS[c][1])
    fig, ax = plt.subplots(figsize=(8, 8.5))
    sc = ax.scatter(city["lon"], city["lat"], s=60 + city["n"] * 25, c=city["pm"], cmap="YlOrRd",
                    edgecolor="k", vmin=20, vmax=130)
    for c, r in city.iterrows():
        ax.annotate(c, (r["lon"], r["lat"]), fontsize=7, xytext=(4, 4), textcoords="offset points")
    plt.colorbar(sc, ax=ax, label="mean PM2.5 (ug/m3)")
    ax.set(title="Spatial pattern: city mean PM2.5 (size = #stations)", xlabel="longitude",
           ylabel="latitude")
    save(fig, "spatial_city_map")
    S["city_mean_pm25"] = city["pm"].round(1).sort_values(ascending=False).to_dict()

    ch = raw.groupby(["Datetime", "City"])["PM2.5"].mean().unstack()
    top = city.sort_values("n", ascending=False).index[:12]
    fig, ax = plt.subplots(figsize=(9, 7.5))
    sns.heatmap(ch[top].corr(), annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, ax=ax)
    ax.set_title("Correlation of hourly city-mean PM2.5 (spatial dependence)")
    save(fig, "city_correlation")
    # within-city station correlation (Delhi)
    dl = raw[raw["City"] == "Delhi"].pivot_table(index="Datetime", columns="StationId",
                                                 values="PM2.5")
    c = dl.corr().where(~np.eye(dl.shape[1], dtype=bool)).stack()
    S["delhi_station_corr_median"] = float(c.median())

    # 8 --- target distribution & class balance
    pm = raw["PM2.5"].dropna()
    cats = pm.map(pm25_category).value_counts().reindex(list(CATEGORY_COLORS))
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.2))
    axes[0].hist(pm.clip(upper=600), bins=120, color="#4477aa")
    axes[0].set_yscale("log")
    axes[0].axvline(POOR_THRESHOLD, color="red", ls="--")
    axes[0].set(title="PM2.5 distribution (log count; heavy right tail)", xlabel="PM2.5 (ug/m3)")
    axes[1].bar(cats.index, cats.values, color=[CATEGORY_COLORS[k] for k in cats.index],
                edgecolor="k")
    axes[1].set(title="CPCB PM2.5 category counts (hourly)")
    for i, v in enumerate(cats.values):
        axes[1].text(i, v, f"{v / cats.sum():.1%}", ha="center", va="bottom", fontsize=9)
    save(fig, "target_distribution_classes")
    S["category_share"] = (cats / cats.sum()).round(4).to_dict()
    S["poor_or_worse_share"] = float((pm > POOR_THRESHOLD).mean())
    by_month = raw.groupby("month")["PM2.5"].apply(lambda s: (s.dropna() > POOR_THRESHOLD).mean())
    S["poor_share_by_month"] = by_month.round(3).to_dict()

    # 9 --- sensor faults (from data_prep.clean)
    fl = os.path.join(RESULTS_DIR, "fault_log.csv")
    if os.path.exists(fl):
        f = pd.read_csv(fl)
        f = f[f["values_removed"] > 0]
        piv = f.pivot_table(index="pollutant", columns="rule", values="values_removed", aggfunc="sum")
        fig, ax = plt.subplots(figsize=(11, 4.5))
        piv.plot.barh(stacked=True, ax=ax, logx=True)
        ax.set(title="Readings removed by sensor-fault rule (log scale)", xlabel="values removed")
        save(fig, "sensor_faults")
        S["faults_removed"] = f.groupby("rule")["values_removed"].sum().to_dict()
    # outlier boxplots per pollutant
    fig, ax = plt.subplots(figsize=(13, 4))
    sns.boxplot(data=raw[POLLUTANTS].sample(min(300_000, len(raw[POLLUTANTS])), random_state=0), ax=ax, fliersize=1)
    ax.set_yscale("symlog")
    ax.set_title("Raw pollutant distributions (symlog) - caps at 1000/500 are instrument limits")
    save(fig, "pollutant_boxplots")

    # 10 --- meteorology vs PM2.5 (class separation)
    w = load_weather()
    if w is not None:
        m = raw[["City", "Datetime", "PM2.5", "season"]].merge(
            w[["City", "Datetime", "met_blh", "met_ws", "met_rh", "met_temp", "met_precip_sum24"]],
            on=["City", "Datetime"]).dropna(subset=["PM2.5"])
        m["Poor+"] = np.where(m["PM2.5"] > POOR_THRESHOLD, "Poor or worse", "Below Poor")
        smp = m.sample(min(200_000, len(m)), random_state=0)
        fig, axes = plt.subplots(1, 4, figsize=(18, 4))
        for ax, col, lab in zip(axes, ["met_blh", "met_ws", "met_rh", "met_temp"],
                                ["boundary-layer height (m)", "wind speed (m/s)",
                                 "relative humidity (%)", "temperature (C)"]):
            sns.violinplot(data=smp, x="Poor+", y=col, ax=ax, cut=0, inner="quartile",
                           hue="Poor+", palette={"Poor or worse": "#e93f33", "Below Poor": "#55a84f"})
            ax.set(title=lab, xlabel="", ylabel="")
        fig.suptitle("Class separation: meteorology at the time of Poor+ vs other hours")
        save(fig, "class_separation_meteorology")
        fig, axes = plt.subplots(1, 3, figsize=(16, 4))
        for ax, col, bins, lab in [(axes[0], "met_blh", np.arange(0, 3001, 150), "BLH (m)"),
                                   (axes[1], "met_ws", np.arange(0, 10.5, 0.5), "wind speed (m/s)"),
                                   (axes[2], "met_rh", np.arange(0, 101, 5), "RH (%)")]:
            b = pd.cut(m[col], bins)
            gm = m.groupby([b, "season"], observed=True)["PM2.5"].mean().unstack()[SEASON_ORDER]
            gm.index = [i.mid for i in gm.index]
            gm.plot(ax=ax, marker=".")
            ax.set(xlabel=lab, ylabel="mean PM2.5", title=f"PM2.5 vs {lab}")
        save(fig, "pm25_vs_meteorology")
        S["corr_pm25_met"] = m[["PM2.5", "met_blh", "met_ws", "met_rh", "met_temp",
                                "met_precip_sum24"]].corr(method="spearman")["PM2.5"].round(3).to_dict()

    # 11 --- fires & festivals (Delhi)
    proc = os.path.join(os.path.dirname(RESULTS_DIR), "data", "processed.parquet")
    if os.path.exists(proc):
        p = pd.read_parquet(proc, columns=["City", "Datetime", "PM2.5_obs", "fire_up500_d1",
                                           "fire_n500_d1"])
        dl = p[p["City"] == "Delhi"]
        daily = dl.groupby(dl["Datetime"].dt.normalize()).agg(
            pm=("PM2.5_obs", "mean"), up=("fire_up500_d1", "first"), n=("fire_n500_d1", "first"))
        fig, axes = plt.subplots(1, 2, figsize=(16, 4.2))
        for yr, ax in zip([2018, 2019], axes):
            s = daily[(daily.index >= f"{yr}-09-15") & (daily.index <= f"{yr}-12-15")]
            ax.plot(s.index, s["pm"], color="k", label="Delhi PM2.5 (daily mean)")
            ax2 = ax.twinx()
            ax2.bar(s.index, s["up"], color="orange", alpha=0.5, label="upwind fires (prev. day)")
            dw = pd.Timestamp([d for d in FESTIVALS["Diwali"] if d.startswith(str(yr))][0])
            ax.axvline(dw, color="purple", ls="--", label="Diwali")
            ax.set_title(f"Stubble-burning season {yr}: Delhi PM2.5 vs upwind FIRMS fires")
            ax.legend(loc="upper left", fontsize=8)
            ax2.legend(loc="upper right", fontsize=8)
        save(fig, "fires_delhi")
        S["delhi_corr_pm_upwind_fires_oct_nov"] = float(
            daily[daily.index.month.isin([10, 11])][["pm", "up"]].corr(method="spearman").iloc[0, 1])

    # Diwali composite: national mean PM2.5 around Diwali (+-10 days)
    rows = []
    hourly = raw.groupby("Datetime")["PM2.5"].mean()
    for d in FESTIVALS["Diwali"][:-1]:
        d = pd.Timestamp(d)
        s = hourly[(hourly.index >= d - pd.Timedelta(days=10)) & (hourly.index < d + pd.Timedelta(days=10))]
        rows.append(pd.Series(s.values, index=(s.index - d) / pd.Timedelta(days=1), name=d.year))
    fig, ax = plt.subplots(figsize=(10, 4))
    for r in rows:
        ax.plot(r.index, r.values, label=str(r.name), lw=1)
    ax.axvline(0, color="k", ls="--")
    ax.set(title="Network-mean PM2.5 around Diwali (day 0) - festival spikes",
           xlabel="days from Diwali", ylabel="PM2.5")
    ax.legend()
    save(fig, "diwali_effect")

    with open(os.path.join(RESULTS_DIR, "eda_summary.json"), "w") as fh:
        json.dump(S, fh, indent=1, default=str)
    print(json.dumps({k: S[k] for k in ["rows", "stations", "cities", "time_span",
                                        "poor_or_worse_share"]}, indent=1))


if __name__ == "__main__":
    main()
