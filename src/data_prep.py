"""
Data loading, cleaning and feature engineering.

Target: PM2.5 24 hours ahead (regression) + "Poor or worse" flag (classification).
Only past information is used for features (lags / shifted rolling windows).
"""
import numpy as np
import pandas as pd

DATA_DIR = "data"
HORIZON = 24  # hours ahead
POLLUTANTS = ["PM2.5", "PM10", "NO", "NO2", "NOx", "NH3", "CO", "SO2", "O3",
              "Benzene", "Toluene", "Xylene"]
POOR_THRESHOLD = 90  # PM2.5 ug/m3 -> CPCB "Poor" band starts at 91


def load_raw():
    """Kaggle 'Air Quality Data in India' -> station_hour.csv + stations.csv"""
    df = pd.read_csv(f"{DATA_DIR}/station_hour.csv", parse_dates=["Datetime"])
    stations = pd.read_csv(f"{DATA_DIR}/stations.csv")
    df = df.merge(stations[["StationId", "City", "State"]], on="StationId", how="left")
    # AQI and AQI_Bucket are computed from same-time pollutants -> leakage, drop them
    df = df.drop(columns=["AQI", "AQI_Bucket"], errors="ignore")
    return df.sort_values(["StationId", "Datetime"]).reset_index(drop=True)


def clean(df):
    """Basic sensor-fault handling. TODO: refine thresholds after EDA."""
    for col in POLLUTANTS:
        if col not in df:
            continue
        df.loc[df[col] < 0, col] = np.nan                    # impossible negatives
        # flat-line sensor: same value for 12+ hours in a row
        same = df.groupby("StationId")[col].diff().eq(0)
        run = same.groupby((~same).cumsum()).cumsum()
        df.loc[run >= 12, col] = np.nan
    return df


def impute(df, method="ffill"):
    """Two strategies to compare: forward fill vs linear interpolation (per station)."""
    for col in POLLUTANTS:
        if col not in df:
            continue
        df[f"{col}_missing"] = df[col].isna().astype(int)    # missing indicator
        g = df.groupby("StationId")[col]
        if method == "ffill":
            df[col] = g.ffill(limit=6)
        else:
            df[col] = g.transform(lambda s: s.interpolate(limit=6))
    return df


def add_features(df):
    g = df.groupby("StationId")
    # calendar
    df["hour"] = df["Datetime"].dt.hour
    df["dow"] = df["Datetime"].dt.dayofweek
    df["month"] = df["Datetime"].dt.month
    df["is_weekend"] = (df["dow"] >= 5).astype(int)
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    # lags (current value at time t is known when forecasting t+24)
    for col in ["PM2.5", "PM10", "NO2", "CO", "O3", "SO2"]:
        for lag in [0, 1, 3, 6, 24]:
            df[f"{col}_lag{lag}"] = g[col].shift(lag)
        # rolling stats on past data only (window ends at t)
        for w in [6, 24, 168]:
            df[f"{col}_roll{w}_mean"] = g[col].transform(lambda s: s.rolling(w, min_periods=1).mean())
        df[f"{col}_roll24_std"] = g[col].transform(lambda s: s.rolling(24, min_periods=2).std())

    # TODO: merge meteorology (ERA5 / NASA POWER), FIRMS fire counts, festival calendar

    # targets
    df["target"] = g["PM2.5"].shift(-HORIZON)
    df["target_poor"] = (df["target"] > POOR_THRESHOLD).astype(int)
    return df


def feature_columns(df):
    drop = set(POLLUTANTS) | {"StationId", "Datetime", "City", "State",
                              "target", "target_poor"}
    return [c for c in df.columns if c not in drop]


def chrono_split(df, train_end="2018-12-31", val_end="2019-06-30"):
    """Chronological split: train <= 2018, val = H1 2019, test = rest."""
    train = df[df["Datetime"] <= train_end]
    val = df[(df["Datetime"] > train_end) & (df["Datetime"] <= val_end)]
    test = df[df["Datetime"] > val_end]
    return train, val, test


def build_dataset(impute_method="ffill"):
    df = load_raw()
    df = clean(df)
    df = impute(df, impute_method)
    df = add_features(df)
    df = df.dropna(subset=["target", "PM2.5_lag0"])
    return df


if __name__ == "__main__":
    data = build_dataset()
    print(data.shape)
    print(len(feature_columns(data)), "features")
    data.to_parquet(f"{DATA_DIR}/processed.parquet")
