"""Streamlit UI: pick station + date + model, see 24h-ahead PM2.5 forecast and AQI category."""
import glob
import os

import joblib
import pandas as pd
import streamlit as st

import sys
sys.path.append("src")
from data_prep import feature_columns

st.title("Air Quality Forecast (PM2.5, 24h ahead)")


@st.cache_data
def load_data():
    return pd.read_parquet("data/processed.parquet")


def pm25_category(v):
    bands = [(30, "Good"), (60, "Satisfactory"), (90, "Moderate"),
             (120, "Poor"), (250, "Very Poor")]
    for limit, label in bands:
        if v <= limit:
            return label
    return "Severe"


df = load_data()
feats = feature_columns(df)
models = {os.path.basename(p)[:-7]: p for p in glob.glob("models/*.joblib")}

station = st.selectbox("Station", sorted(df["StationId"].unique()))
sdf = df[df["StationId"] == station]
date = st.date_input("Date", sdf["Datetime"].max().date())
hour = st.slider("Hour", 0, 23, 12)
chosen = st.multiselect("Models", list(models), default=list(models)[:3])

row = sdf[sdf["Datetime"] == pd.Timestamp(date) + pd.Timedelta(hours=hour)]
if row.empty:
    st.warning("No data for this station/time.")
else:
    rows = []
    for m in chosen:
        pred = joblib.load(models[m]).predict(row[feats])[0]
        rows.append({"Model": m, "PM2.5 forecast": round(pred, 1),
                     "Category": pm25_category(pred)})
    st.table(pd.DataFrame(rows))
    st.write("Actual PM2.5 24h later:", row["target"].iloc[0])

st.subheader("Recent PM2.5 history")
st.line_chart(sdf.set_index("Datetime")["PM2.5_lag0"].tail(24 * 14))

if os.path.exists("results/benchmark.csv"):
    st.subheader("Model benchmark")
    st.dataframe(pd.read_csv("results/benchmark.csv", index_col=0).round(3))
