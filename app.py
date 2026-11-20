"""
Air-quality early-warning dashboard (Streamlit), integrated with the ML pipeline.

    streamlit run app.py

Pages
  Forecast          - pick station + date: hourly 24 h-ahead PM2.5 forecasts for
                      the next day from every model, AQI category, warning hours,
                      direct-classifier probability, comparison with actuals
  Early-warning map - all stations for one day: forecast peak category on a map
  Explain & what-if - SHAP explanation of a single forecast and a weather
                      scenario simulator (change wind / mixing height / rain ...)
  Benchmark         - regression & classification results, PCA, efficiency
  Analyses          - ablation, visualisation, efficiency, robustness, errors, XAI
  Data & EDA        - dataset, sources, missingness, leakage audit
"""
import glob
import json
import os
import sys

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import pyarrow.parquet as pq
import streamlit as st

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "src"))
from config import (CATEGORY_COLORS, CITY_COORDS, FIG_DIR, HORIZON, MODEL_DIR,  # noqa: E402
                    POOR_THRESHOLD, PROCESSED, RESULTS_DIR, pm25_category)
from data_prep import GROUP_NAMES, feature_group  # noqa: E402

st.set_page_config(page_title="Air Quality Early Warning", page_icon="🌫️", layout="wide")
MODEL_ORDER = ["LightGBM", "CatBoost", "XGBoost", "HistGradientBoosting", "RandomForest",
               "GradientBoosting", "AdaBoost", "DecisionTree"]
st.markdown("""
<style>
.cat-badge {padding:4px 10px;border-radius:12px;font-weight:600;color:#111;display:inline-block}
.big-num {font-size:2.1rem;font-weight:700;line-height:1.1}
.small {color:#888;font-size:0.85rem}
</style>""", unsafe_allow_html=True)


# ============================================================ loaders
@st.cache_data(show_spinner=False)
def station_index():
    t = pq.read_table(PROCESSED, columns=["StationId", "StationName", "City", "Datetime"])
    d = t.to_pandas()
    g = d.groupby(["StationId", "StationName", "City"])["Datetime"].agg(["min", "max", "size"])
    return g.reset_index()


@st.cache_data(show_spinner="Loading station data ...", max_entries=8)
def station_data(station):
    t = pq.read_table(PROCESSED, filters=[("StationId", "=", station)])
    return t.to_pandas().sort_values("Datetime").reset_index(drop=True)


@st.cache_resource(show_spinner="Loading models ...")
def load_models(kind="reg"):
    out = {}
    pref = "cls_" if kind == "cls" else ""
    for name in MODEL_ORDER:
        p = os.path.join(MODEL_DIR, f"{pref}{name}.joblib")
        if os.path.exists(p):
            out[name] = joblib.load(p)
    return out


@st.cache_data(show_spinner=False)
def read_csv(name, **kw):
    p = os.path.join(RESULTS_DIR, name)
    return pd.read_csv(p, **kw) if os.path.exists(p) else None


@st.cache_data(show_spinner="Loading test-period forecasts ...")
def test_predictions():
    p = os.path.join(RESULTS_DIR, "predictions_test.parquet")
    if not os.path.exists(p):
        return None
    pt = pd.read_parquet(p)
    sp = os.path.join(RESULTS_DIR, "cls_scores_test.parquet")
    if os.path.exists(sp):
        sc = pd.read_parquet(sp)
        if "LightGBM" in sc:
            pt["P(Poor+) LightGBM"] = sc["LightGBM"].to_numpy()
    return pt


def inv(p, how):
    p = np.asarray(p, dtype=float)
    if how == "log1p":
        return np.clip(np.expm1(p), 0, None)
    if how == "sqrt":
        return np.clip(p, 0, None) ** 2
    return np.clip(p, 0, None)


def predict(bundle, X):
    return inv(bundle["pipeline"].predict(X[bundle["features"]]), bundle["transform"])


def badge(cat):
    return f"<span class='cat-badge' style='background:{CATEGORY_COLORS[cat]}'>{cat}</span>"


def fig_path(name):
    p = os.path.join(FIG_DIR, name)
    return p if os.path.exists(p) else None


def show_fig(name, caption=None):
    p = fig_path(name)
    if p:
        st.image(p, caption=caption, use_container_width=True)
    else:
        st.info(f"Figure `{name}` not generated yet - run the analysis scripts.")


# ============================================================ sidebar
st.sidebar.title("🌫️ AQ Early Warning")
st.sidebar.caption("24 h-ahead PM2.5 forecasts for CPCB stations (India, 2015-2020)")
page = st.sidebar.radio("Page", ["🔮 Forecast", "🗺️ Early-warning map", "🧠 Explain & what-if",
                                 "📊 Benchmark", "🔬 Analyses", "🗂️ Data & EDA", "ℹ️ About"])
if not os.path.exists(PROCESSED):
    st.error("data/processed.parquet not found. Run `python src/data_prep.py` first.")
    st.stop()
models = load_models("reg")
if not models and page in ("🔮 Forecast", "🧠 Explain & what-if"):
    st.error("No trained models in models/. Run `python src/train.py` first.")
    st.stop()
idx = station_index()


def station_picker(key):
    cities = sorted(idx["City"].unique())
    c1, c2 = st.columns([1, 2])
    default_city = cities.index("Delhi") if "Delhi" in cities else 0
    city = c1.selectbox("City", cities, index=default_city, key=f"city_{key}")
    sub = idx[idx["City"] == city]
    labels = {r.StationId: f"{r.StationName} ({r.StationId})" for r in sub.itertuples()}
    station = c2.selectbox("Station", list(labels), format_func=labels.get, key=f"st_{key}")
    return city, station, sub.set_index("StationId").loc[station]


# ============================================================ FORECAST
if page == "🔮 Forecast":
    st.title("🔮 Day-ahead PM2.5 forecast")
    st.caption("A forecast issued at hour *t* predicts PM2.5 at *t + 24 h* using only data available "
               "at *t* (pollutant history, neighbouring stations, ERA5 weather up to *t*, previous-day "
               "fire hotspots, calendar). Pick the day you want a forecast **for**.")
    city, station, meta = station_picker("fc")
    sd = station_data(station)
    first = (meta["min"] + pd.Timedelta(hours=HORIZON)).date()
    last = (meta["max"] + pd.Timedelta(hours=HORIZON)).date()
    default = min(max(pd.Timestamp("2019-11-03").date(), first), last)
    c1, c2, c3 = st.columns([1, 2, 1])
    day = c1.date_input("Forecast day", default, min_value=first, max_value=last)
    chosen = c2.multiselect("Models", list(models), default=[m for m in ["LightGBM", "CatBoost",
                                                                         "XGBoost", "RandomForest"]
                                                             if m in models])
    show_actual = c3.toggle("Show actuals", True)
    if not chosen:
        st.warning("Select at least one model.")
        st.stop()
    target_times = pd.date_range(pd.Timestamp(day), periods=24, freq="h")
    issue = sd[sd["Datetime"].isin(target_times - pd.Timedelta(hours=HORIZON))].copy()
    if issue.empty:
        st.warning("No forecast can be issued for this day (station offline the day before).")
        st.stop()
    issue["target_time"] = issue["Datetime"] + pd.Timedelta(hours=HORIZON)
    preds = pd.DataFrame({"target_time": issue["target_time"].to_numpy()})
    for m in chosen:
        preds[m] = predict(models[m], issue)
    preds["Ensemble (mean)"] = preds[chosen].mean(axis=1)
    preds["actual"] = issue["target"].to_numpy()
    cls = load_models("cls")
    if "LightGBM" in cls:
        b = cls["LightGBM"]
        preds["P(Poor+)"] = b["pipeline"].predict_proba(issue[b["features"]])[:, 1]
        cls_thr = b["threshold"]
    lead = "Ensemble (mean)" if len(chosen) > 1 else chosen[0]

    # ---- headline cards
    peak_i = preds[lead].idxmax()
    n_poor = int((preds[lead] > POOR_THRESHOLD).sum())
    daily_mean = preds[lead].mean()
    k1, k2, k3, k4 = st.columns(4)
    k1.markdown(f"<div class='small'>Forecast daily mean</div><div class='big-num'>{daily_mean:.0f}"
                f" <span class='small'>µg/m³</span></div>{badge(pm25_category(daily_mean))}",
                unsafe_allow_html=True)
    k2.markdown(f"<div class='small'>Forecast peak</div><div class='big-num'>{preds[lead].max():.0f}"
                f"</div><span class='small'>at {preds.loc[peak_i, 'target_time']:%H:%M}</span> "
                f"{badge(pm25_category(preds[lead].max()))}", unsafe_allow_html=True)
    k3.markdown(f"<div class='small'>Hours forecast Poor or worse</div><div class='big-num'>"
                f"{n_poor} / {len(preds)}</div>", unsafe_allow_html=True)
    if "P(Poor+)" in preds:
        k4.markdown(f"<div class='small'>Max P(Poor+) - direct classifier</div><div class='big-num'>"
                    f"{preds['P(Poor+)'].max():.0%}</div><span class='small'>warning if ≥ "
                    f"{cls_thr:.0%}</span>", unsafe_allow_html=True)
    if n_poor >= 6 or ("P(Poor+)" in preds and (preds["P(Poor+)"] >= cls_thr).sum() >= 6):
        st.error(f"⚠️ EARLY WARNING: {station} is expected to reach 'Poor or worse' PM2.5 for "
                 f"{n_poor} hours on {day}. Consider issuing a public advisory.")
    elif n_poor > 0:
        st.warning(f"Caution: {n_poor} hour(s) forecast above {POOR_THRESHOLD} µg/m³.")
    else:
        st.success("No unhealthy hours forecast.")

    # ---- chart: history + forecasts
    hist = sd[(sd["Datetime"] >= pd.Timestamp(day) - pd.Timedelta(days=6)) &
              (sd["Datetime"] < pd.Timestamp(day) + pd.Timedelta(days=1))]
    fig = go.Figure()
    for cat, (lo, hi) in {"Good": (0, 30), "Satisfactory": (30, 60), "Moderate": (60, 90),
                          "Poor": (90, 120), "Very Poor": (120, 250), "Severe": (250, 1000)}.items():
        fig.add_hrect(y0=lo, y1=hi, fillcolor=CATEGORY_COLORS[cat], opacity=0.10, line_width=0)
    fig.add_trace(go.Scatter(x=hist["Datetime"], y=hist["PM2.5_obs"], name="observed",
                             line=dict(color="black", width=1.5)))
    for m in chosen + (["Ensemble (mean)"] if len(chosen) > 1 else []):
        fig.add_trace(go.Scatter(x=preds["target_time"], y=preds[m], name=m, mode="lines+markers",
                                 line=dict(width=3 if m == lead else 1.5)))
    fig.add_vline(x=pd.Timestamp(day), line_dash="dot")
    ymax = float(np.nanmax([hist["PM2.5_obs"].max(), preds[chosen].to_numpy().max(), 100]))
    fig.update_layout(height=430, yaxis_title="PM2.5 (µg/m³)", hovermode="x unified",
                      yaxis_range=[0, ymax * 1.1], legend=dict(orientation="h", y=1.08),
                      margin=dict(t=30, b=10))
    st.plotly_chart(fig, use_container_width=True)

    # ---- table
    tbl = preds.copy()
    tbl["hour"] = tbl["target_time"].dt.strftime("%H:%M")
    tbl["category"] = tbl[lead].map(pm25_category)
    cols = ["hour"] + chosen + (["Ensemble (mean)"] if len(chosen) > 1 else []) + \
        (["P(Poor+)"] if "P(Poor+)" in tbl else []) + (["actual"] if show_actual else []) + ["category"]
    fmt = {c: "{:.0f}" for c in chosen + ["Ensemble (mean)", "actual"]}
    fmt["P(Poor+)"] = "{:.0%}"

    def colour(v):
        return f"background-color: {CATEGORY_COLORS.get(v, '')}" if v in CATEGORY_COLORS else ""
    with st.expander("Hourly forecast table", expanded=False):
        st.dataframe(tbl[cols].style.format(fmt, na_rep="–").map(colour, subset=["category"]),
                     use_container_width=True, height=400)
    if show_actual and preds["actual"].notna().any():
        ok = preds["actual"].notna()
        rows = [{"model": m, "MAE (this day)": np.mean(np.abs(preds.loc[ok, m] - preds.loc[ok, "actual"])),
                 "forecast mean": preds[m].mean(), "actual mean": preds.loc[ok, "actual"].mean()}
                for m in chosen + (["Ensemble (mean)"] if len(chosen) > 1 else [])]
        st.subheader("Model comparison for this day")
        st.dataframe(pd.DataFrame(rows).set_index("model").style.format("{:.1f}"),
                     use_container_width=True)

# ============================================================ MAP
elif page == "🗺️ Early-warning map":
    st.title("🗺️ Network early-warning board")
    pt = test_predictions()
    if pt is None:
        st.error("Run `python src/train.py` to generate test-period forecasts.")
        st.stop()
    st.caption("Forecasts produced by the pipeline for the held-out test period (Jul 2019 – Jun 2020). "
               "Each station's peak and mean forecast for the chosen day.")
    avail = [m for m in MODEL_ORDER if m in pt.columns]
    c1, c2 = st.columns([1, 1])
    pt["target_time"] = pt["Datetime"] + pd.Timedelta(hours=HORIZON)
    day = c1.date_input("Day", pd.Timestamp("2019-11-03").date(),
                        min_value=pt["target_time"].min().date(), max_value=pt["target_time"].max().date())
    model = c2.selectbox("Model", avail)
    d = pt[pt["target_time"].dt.date == day]
    if d.empty:
        st.warning("No forecasts for this day.")
        st.stop()
    agg = d.groupby(["StationId", "City"]).agg(peak=(model, "max"), mean=(model, "mean"),
                                               actual_mean=("target", "mean"),
                                               poor_hours=(model, lambda s: int((s > POOR_THRESHOLD).sum())))
    if "P(Poor+) LightGBM" in d:
        agg["max_P_poor"] = d.groupby(["StationId", "City"])["P(Poor+) LightGBM"].max()
    agg = agg.reset_index().merge(idx[["StationId", "StationName"]], on="StationId")
    agg["category"] = agg["mean"].map(pm25_category)
    rng = np.random.default_rng(0)
    agg["lat"] = agg["City"].map(lambda c: CITY_COORDS[c][0]) + rng.normal(0, 0.06, len(agg))
    agg["lon"] = agg["City"].map(lambda c: CITY_COORDS[c][1]) + rng.normal(0, 0.06, len(agg))
    k1, k2, k3 = st.columns(3)
    k1.metric("Stations with a forecast", len(agg))
    k2.metric("Stations with ≥ 6 Poor+ hours", int((agg["poor_hours"] >= 6).sum()))
    k3.metric("Network mean forecast", f"{agg['mean'].mean():.0f} µg/m³")
    fig = px.scatter_map(agg, lat="lat", lon="lon", color="category", size="peak",
                         color_discrete_map=CATEGORY_COLORS, hover_name="StationName",
                         hover_data={"City": True, "mean": ":.0f", "peak": ":.0f",
                                     "poor_hours": True, "lat": False, "lon": False},
                         zoom=3.6, center={"lat": 22.5, "lon": 80}, height=620,
                         category_orders={"category": list(CATEGORY_COLORS)})
    fig.update_layout(map_style="open-street-map", margin=dict(l=0, r=0, t=0, b=0))
    st.plotly_chart(fig, use_container_width=True)
    st.subheader("Stations ranked by risk")
    show = agg.sort_values("mean", ascending=False)[
        ["StationName", "City", "mean", "peak", "poor_hours"] +
        (["max_P_poor"] if "max_P_poor" in agg else []) + ["actual_mean", "category"]]
    st.dataframe(show.style.format({"mean": "{:.0f}", "peak": "{:.0f}", "actual_mean": "{:.0f}",
                                    "max_P_poor": "{:.0%}"}, na_rep="–"),
                 use_container_width=True, height=420)
    st.download_button("Download warning list (CSV)", show.to_csv(index=False),
                       file_name=f"aq_warnings_{day}.csv")

# ============================================================ EXPLAIN
elif page == "🧠 Explain & what-if":
    st.title("🧠 Why this forecast? & weather what-if")
    city, station, meta = station_picker("xai")
    sd = station_data(station)
    c1, c2, c3 = st.columns(3)
    first = (meta["min"] + pd.Timedelta(hours=HORIZON)).date()
    last = (meta["max"] + pd.Timedelta(hours=HORIZON)).date()
    day = c1.date_input("Target day", min(max(pd.Timestamp("2019-11-03").date(), first), last),
                        min_value=first, max_value=last)
    hour = c2.slider("Target hour", 0, 23, 8)
    xmodel = c3.selectbox("Model to explain", [m for m in ["LightGBM", "XGBoost", "CatBoost",
                                                           "RandomForest", "DecisionTree"] if m in models])
    tt = pd.Timestamp(day) + pd.Timedelta(hours=hour)
    row = sd[sd["Datetime"] == tt - pd.Timedelta(hours=HORIZON)]
    if row.empty:
        st.warning("No forecast for this target time.")
        st.stop()
    b = models[xmodel]
    feats = b["features"]
    base_pred = float(predict(b, row)[0])
    st.markdown(f"**Forecast for {tt:%d %b %Y %H:%M}** ({xmodel}): "
                f"<span class='big-num'>{base_pred:.0f}</span> µg/m³ {badge(pm25_category(base_pred))}"
                f" &nbsp; actual: **{row['target'].iloc[0]:.0f}**", unsafe_allow_html=True)
    tab1, tab2 = st.tabs(["Feature contributions (SHAP)", "What-if scenario"])
    with tab1:
        import shap
        pipe = b["pipeline"]
        X = row[feats]
        Z = X
        if "impute" in pipe.named_steps:
            Z = pd.DataFrame(pipe.named_steps["impute"].transform(X), columns=feats)
        model = pipe.named_steps["model"]
        if xmodel == "CatBoost":
            from catboost import Pool
            sv_all = model.get_feature_importance(Pool(Z), type="ShapValues")[0]
            sv, base = sv_all[:-1], sv_all[-1]
        else:
            ex = shap.TreeExplainer(model)
            sv = np.asarray(ex.shap_values(Z, check_additivity=False))[0]
            base = float(np.ravel(ex.expected_value)[0])
        contrib = pd.DataFrame({"feature": feats, "value": X.iloc[0].to_numpy(), "shap": sv})
        contrib["group"] = contrib["feature"].map(lambda f: GROUP_NAMES.get(feature_group(f), f))
        top = contrib.reindex(contrib["shap"].abs().sort_values(ascending=False).index).head(15)
        st.caption("Contributions are in log(1+PM2.5) space: positive = pushes the forecast up. "
                   f"Baseline (average training forecast) = {np.expm1(base):.0f} µg/m³.")
        fig = go.Figure(go.Bar(x=top["shap"][::-1], y=[f"{f} = {v:.3g}" for f, v in
                                                       zip(top["feature"][::-1], top["value"][::-1])],
                               orientation="h",
                               marker_color=["#e93f33" if s > 0 else "#3366aa" for s in top["shap"][::-1]]))
        fig.update_layout(height=520, xaxis_title="SHAP contribution (log space)",
                          margin=dict(t=10, l=10))
        st.plotly_chart(fig, use_container_width=True)
        grp = contrib.groupby("group")["shap"].sum().sort_values()
        st.plotly_chart(px.bar(grp, orientation="h", title="Net contribution by feature group",
                               labels={"value": "sum of SHAP", "group": ""}).update_layout(
            showlegend=False, height=330), use_container_width=True)
    with tab2:
        st.caption("Change the meteorological / emission inputs and see how every model's forecast "
                   "responds (other features fixed). Useful for intervention planning and for checking "
                   "that the models behave physically.")
        r0 = row[feats].iloc[0]
        c = st.columns(3)
        ws = c[0].slider("Wind speed now (m/s)", 0.0, 12.0, float(np.nan_to_num(r0.get("met_ws", 2))), 0.1)
        blh = c[1].slider("Max mixing height, last 24 h (m)", 100.0, 4000.0,
                          float(np.nan_to_num(r0.get("met_blh_max24", 1000))), 50.0)
        rain = c[2].slider("Rain, last 24 h (mm)", 0.0, 50.0, float(np.nan_to_num(r0.get("met_precip_sum24", 0))), 0.5)
        c = st.columns(3)
        rh = c[0].slider("Relative humidity now (%)", 5.0, 100.0, float(np.nan_to_num(r0.get("met_rh", 60))), 1.0)
        fires = c[1].slider("Upwind fires yesterday (500 km)", 0.0, 3000.0,
                            float(np.nan_to_num(r0.get("fire_up500_d1", 0))), 10.0)
        pm_scale = c[2].slider("Scale current pollution (emission cut %)", -80, 50, 0, 5)
        Xw = row.copy()
        f_ws = ws / max(float(np.nan_to_num(r0.get("met_ws", 1))), 0.1)
        Xw["met_ws"], Xw["met_ws_mean24"] = ws, Xw["met_ws_mean24"] * f_ws
        Xw["met_blh_max24"] = blh
        Xw["met_vent"] = Xw["met_blh"] * ws
        Xw["met_vent_min24"] = Xw["met_vent_min24"] * f_ws
        Xw["met_precip_sum24"] = rain
        Xw["met_precip_sum72"] = max(rain, float(np.nan_to_num(r0.get("met_precip_sum72", 0))))
        Xw["met_rh"] = rh
        Xw["fire_up500_d1"] = fires
        if pm_scale:
            pcols = [f for f in feats if f.startswith(("pm25_", "pol_", "nbr_")) and
                     not f.endswith(("_ratio", "_n")) and "diff" not in f]
            Xw[pcols] = Xw[pcols] * (1 + pm_scale / 100)
        rows = []
        for m, bb in models.items():
            p0, p1 = float(predict(bb, row)[0]), float(predict(bb, Xw)[0])
            rows.append({"model": m, "baseline": p0, "scenario": p1, "change": p1 - p0,
                         "scenario category": pm25_category(p1)})
        res = pd.DataFrame(rows).set_index("model")
        st.dataframe(res.style.format({"baseline": "{:.0f}", "scenario": "{:.0f}", "change": "{:+.0f}"}),
                     use_container_width=True)
        st.plotly_chart(px.bar(res.reset_index(), x="model", y=["baseline", "scenario"], barmode="group",
                               labels={"value": "PM2.5 forecast (µg/m³)"}).update_layout(height=360),
                        use_container_width=True)

# ============================================================ BENCHMARK
elif page == "📊 Benchmark":
    st.title("📊 Model benchmark (test period: Jul 2019 – Jun 2020)")
    bench = read_csv("benchmark.csv", index_col=0)
    if bench is None:
        st.error("Run `python src/train.py`.")
        st.stop()
    tab1, tab2, tab3, tab4 = st.tabs(["Regression", "Classification (Poor+)", "PCA", "Efficiency"])
    with tab1:
        cols = ["MAE", "RMSE", "R2", "MAE_top5pct", "MAE_top1pct", "Bias_top5pct", "Precision",
                "Recall", "F1", "PR_AUC"]
        b = bench[[c for c in cols if c in bench]].copy()
        if "Persistence" in b.index:
            b.insert(1, "Skill vs persistence", 1 - b["MAE"] / b.loc["Persistence", "MAE"])
        st.dataframe(b.sort_values("MAE").style.format("{:.3f}").background_gradient(
            subset=["MAE", "RMSE"], cmap="RdYlGn_r").background_gradient(subset=["R2"], cmap="RdYlGn"),
            use_container_width=True, height=min(800, 40 + 35 * len(b)))
        st.caption("MAE/RMSE in µg/m³ for PM2.5 at t+24. Precision/Recall/F1 use the regression forecast "
                   "thresholded at 90 µg/m³. Skill = 1 − MAE/MAE_persistence.")
        met = st.selectbox("Metric", ["MAE", "RMSE", "R2", "MAE_top5pct", "F1", "PR_AUC"])
        bb = b.reset_index().rename(columns={"index": "model"})
        bb["variant"] = np.where(bb["model"].str.contains("PCA"), "with PCA",
                                 np.where(bb["model"].isin(["Persistence", "SeasonalNaive7d", "Climatology"]),
                                          "baseline", "no PCA"))
        bb["family"] = bb["model"].str.replace("+PCA", "", regex=False)
        st.plotly_chart(px.bar(bb, x="family", y=met, color="variant", barmode="group",
                               color_discrete_map={"no PCA": "#3366aa", "with PCA": "#ee9944",
                                                   "baseline": "#999999"}).update_layout(height=420),
                        use_container_width=True)
        show_fig("bench_extremes.png")
    with tab2:
        cb = read_csv("classification_benchmark.csv", index_col=0)
        if cb is not None:
            cols = ["approach", "family", "Precision", "Recall", "F1", "PR_AUC", "ROC_AUC", "Threshold"]
            st.dataframe(cb[[c for c in cols if c in cb]].sort_values("F1", ascending=False).style.format(
                {c: "{:.3f}" for c in cols[2:]}), use_container_width=True, height=600)
            show_fig("bench_classification.png")
            show_fig("bench_pr_curves.png")
        else:
            st.info("Run `python src/train_classifiers.py`.")
    with tab3:
        show_fig("bench_pca_scree.png")
        if all(m in bench.index for m in ["LightGBM", "LightGBM+PCA"]):
            fams = [m for m in MODEL_ORDER if m in bench.index and m + "+PCA" in bench.index]
            comp = pd.DataFrame({"MAE (all features)": bench.loc[fams, "MAE"].to_numpy(),
                                 "MAE (PCA)": bench.loc[[f + "+PCA" for f in fams], "MAE"].to_numpy(),
                                 "components": bench.loc[[f + "+PCA" for f in fams]].get(
                                     "pca_components", pd.Series(np.nan, index=fams)).to_numpy()},
                                index=fams)
            comp["Δ MAE"] = comp["MAE (PCA)"] - comp["MAE (all features)"]
            st.dataframe(comp.style.format("{:.2f}"), use_container_width=True)
    with tab4:
        eff = read_csv("efficiency.csv", index_col=0)
        if eff is not None:
            st.dataframe(eff.style.format("{:.3f}"), use_container_width=True)
        show_fig("eff_tradeoff.png")
        show_fig("eff_costs.png")

# ============================================================ ANALYSES
elif page == "🔬 Analyses":
    st.title("🔬 Additional analyses")
    tabs = st.tabs(["A · Ablation", "B · Visualisation", "C · Efficiency", "D · Robustness",
                    "E · Error analysis", "F · Explainability"])
    with tabs[0]:
        ab = read_csv("ablation_groups.csv")
        if ab is not None:
            piv = ab.pivot_table(index="config", columns="model", values="MAE")
            base = piv.loc["All features"] if "All features" in piv.index else None
            st.subheader("Test MAE when a feature group is removed / kept")
            st.dataframe(piv.style.format("{:.2f}").background_gradient(cmap="RdYlGn_r", axis=None),
                         use_container_width=True)
            if base is not None:
                st.caption("Δ MAE vs all features (positive = the group helps)")
                st.dataframe((piv - base).style.format("{:+.2f}").background_gradient(
                    cmap="RdBu_r", axis=None, vmin=-3, vmax=3), use_container_width=True)
        show_fig("abl_groups.png")
        show_fig("abl_topk_pca.png")
    with tabs[1]:
        for f, c in [("viz_episodes.png", "Forecasts during pollution episodes"),
                     ("viz_pred_vs_actual.png", None), ("viz_seasonal_diurnal_fit.png", None),
                     ("viz_spatial_skill.png", None), ("viz_class_separation.png", None),
                     ("eda_seasonal_diurnal.png", None), ("eda_spatial_city_map.png", None)]:
            show_fig(f, c)
    with tabs[2]:
        show_fig("eff_tradeoff.png")
        show_fig("eff_costs.png")
    with tabs[3]:
        for f in ["rob_noise_missing.png", "rob_training_size.png", "rob_loco.png",
                  "rob_walk_forward.png", "rob_period.png"]:
            show_fig(f)
        lo = read_csv("robust_loco.csv")
        if lo is not None:
            st.subheader("Leave-one-city-out")
            st.dataframe(lo.round(2), use_container_width=True)
    with tabs[4]:
        for f in ["err_heatmaps_season_band.png", "err_by_hour.png", "err_by_city.png",
                  "err_difficult_cases.png", "err_residuals_band_confusion.png", "err_error_correlation.png"]:
            show_fig(f)
        w = read_csv("errors_worst_forecasts.csv")
        if w is not None:
            st.subheader("Worst individual forecasts (best model)")
            st.dataframe(w.round(1), use_container_width=True)
    with tabs[5]:
        for f in glob.glob(os.path.join(FIG_DIR, "xai_*.png")):
            st.image(f, use_container_width=True)
        ph = read_csv("explain_pdp_physical_check.csv")
        if ph is not None:
            st.subheader("Physical-plausibility check of partial dependence slopes")
            st.dataframe(ph.round(3), use_container_width=True)

# ============================================================ DATA
elif page == "🗂️ Data & EDA":
    st.title("🗂️ Data, preprocessing & EDA")
    summ = os.path.join(RESULTS_DIR, "eda_summary.json")
    if os.path.exists(summ):
        S = json.load(open(summ))
        k = st.columns(5)
        k[0].metric("Station-hours", f"{S['rows']:,}")
        k[1].metric("Stations", S["stations"])
        k[2].metric("Cities", S["cities"])
        k[3].metric("Span", f"{S['time_span'][0][:10]} → {S['time_span'][1][:10]}")
        k[4].metric("Poor+ share", f"{S['poor_or_worse_share']:.1%}")
    leak = os.path.join(ROOT, "docs", "leakage_audit.md")
    tabs = st.tabs(["Sources", "Missingness", "Patterns", "Faults & outliers", "Leakage audit",
                    "Imputation", "Station explorer"])
    with tabs[0]:
        src = os.path.join(ROOT, "docs", "data_sources.md")
        st.markdown(open(src, encoding="utf-8").read() if os.path.exists(src) else "See README.")
    with tabs[1]:
        for f in ["eda_missingness_pollutant_year_season.png", "eda_missingness_by_station.png",
                  "eda_missingness_vs_pollution.png", "eda_availability_station_quarter.png",
                  "eda_coverage_over_time.png"]:
            show_fig(f)
    with tabs[2]:
        for f in ["eda_seasonal_diurnal.png", "eda_weekly_trend.png", "eda_spatial_city_map.png",
                  "eda_city_correlation.png", "eda_target_distribution_classes.png",
                  "eda_class_separation_meteorology.png", "eda_pm25_vs_meteorology.png",
                  "eda_fires_delhi.png", "eda_diwali_effect.png"]:
            show_fig(f)
    with tabs[3]:
        show_fig("eda_sensor_faults.png")
        show_fig("eda_pollutant_boxplots.png")
        fl = read_csv("fault_log.csv")
        if fl is not None:
            st.dataframe(fl[fl["values_removed"] > 0], use_container_width=True)
    with tabs[4]:
        st.markdown(open(leak, encoding="utf-8").read() if os.path.exists(leak) else "")
    with tabs[5]:
        r = read_csv("imputation_reconstruction.csv")
        d = read_csv("imputation_comparison.csv")
        if r is not None:
            st.subheader("Reconstruction of artificially hidden readings (MAE)")
            st.dataframe(r.pivot_table(index=["pollutant", "gap_type"], columns="method",
                                       values="MAE").round(2), use_container_width=True)
        if d is not None:
            st.subheader("Downstream forecast quality with each imputation strategy (LightGBM)")
            st.dataframe(d.pivot_table(index="method", columns="split",
                                       values=["MAE", "RMSE", "R2"]).round(3), use_container_width=True)
    with tabs[6]:
        city, station, meta = station_picker("eda")
        sd = station_data(station)
        var = st.multiselect("Series", ["PM2.5_obs", "pol_PM10_lag0", "pol_NO2_lag0", "met_blh",
                                        "met_ws", "met_temp", "met_rh", "fire_up500_d1"],
                             default=["PM2.5_obs"])
        rng = st.slider("Period", meta["min"].to_pydatetime(), meta["max"].to_pydatetime(),
                        ((meta["max"] - pd.Timedelta(days=60)).to_pydatetime(), meta["max"].to_pydatetime()))
        sub = sd[(sd["Datetime"] >= rng[0]) & (sd["Datetime"] <= rng[1])]
        st.plotly_chart(px.line(sub, x="Datetime", y=var).update_layout(height=420),
                        use_container_width=True)
        st.caption(f"Missing PM2.5 in this window: {sub['PM2.5_obs'].isna().mean():.1%}")

# ============================================================ ABOUT
else:
    st.title("ℹ️ About this project")
    readme = os.path.join(ROOT, "README.md")
    st.markdown(open(readme, encoding="utf-8").read())
