"""
Data loading, cleaning, imputation, external-data merge and feature engineering.

Forecast task
-------------
At issue time t (every hour, every station) forecast the PM2.5 concentration at
t + 24 h (regression) and whether it will be "Poor or worse" (> 90 ug/m3,
classification). Every feature is computed from information available at t:

* pollutant lags / rolling windows end at t (windows are trailing, never centred)
* same-station / same-city / regional readings are taken at t or earlier
* meteorology is ERA5 reanalysis at t or earlier (no observed weather at t+24)
* fire hotspots come from the previous complete day(s)
* calendar features describe the *target* time t+24, which is known in advance

The target is built from the *cleaned but un-imputed* PM2.5 series, so imputed
values never become labels.

Feature-name prefixes define the feature groups used throughout the project:
pm25_ (target lags), pol_ (other pollutants), miss_ (missingness), nbr_
(neighbouring stations), met_ (meteorology), fire_ (fire hotspots), cal_
(calendar), loc_ (location) and fut_ (future weather, excluded by default).
"""
import glob
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (CITY_COORDS, DATA_DIR, EXT_DIR, FESTIVALS, HORIZON,  # noqa: E402
                    NATIONAL_HOLIDAYS, POLLUTANTS, POOR_THRESHOLD, PROCESSED,
                    RESULTS_DIR, TRAIN_END, VAL_END)

FEATURE_PREFIXES = ("pm25_", "pol_", "miss_", "nbr_", "met_", "fire_", "cal_", "loc_")
GROUP_NAMES = {"pm25_": "PM2.5 lags & rolling", "pol_": "Other pollutants",
               "miss_": "Missingness indicators", "nbr_": "Neighbour stations",
               "met_": "Meteorology (ERA5)", "fire_": "Fire hotspots (FIRMS)",
               "cal_": "Calendar & festivals", "loc_": "Location",
               "fut_": "Future weather (oracle)"}
F32 = np.float32


# =============================================================== loading
def load_raw():
    """CPCB station-hour file + station metadata. AQI columns are dropped
    (deterministic function of same-time pollutant sub-indices -> leakage)."""
    df = pd.read_csv(os.path.join(DATA_DIR, "station_hour.csv"),
                     usecols=["StationId", "Datetime"] + POLLUTANTS,
                     parse_dates=["Datetime"], dtype={p: F32 for p in POLLUTANTS})
    st = pd.read_csv(os.path.join(DATA_DIR, "stations.csv"))
    df["Datetime"] = df["Datetime"].astype("datetime64[ns]")
    df = df.merge(st[["StationId", "StationName", "City", "State"]], on="StationId", how="left")
    # stations that never report PM2.5 cannot provide a target
    has_pm = df.groupby("StationId")["PM2.5"].transform("count") > 0
    df = df[has_pm]
    return df.sort_values(["StationId", "Datetime"]).reset_index(drop=True)


# =============================================================== cleaning
def _run_length(df, col):
    """Length of the current run of identical consecutive values (causal)."""
    same = df.groupby("StationId", sort=False)[col].diff().eq(0)
    grp = (~same).cumsum()
    return same.groupby(grp).cumsum()


def clean(df, log=None):
    """Sensor-fault handling. All rules are causal (look only at t and earlier,
    or at other stations at the same hour), so the same logic could run live.

    Returns the cleaned frame; counts of removed values go into `log`.
    """
    log = {} if log is None else log

    def flag(mask, col, rule):
        n = int(mask.sum())
        if n:
            df.loc[mask, col] = np.nan
        log[(col, rule)] = log.get((col, rule), 0) + n

    for col in POLLUTANTS:
        flag(df[col] < 0, col, "negative")
    # unit inconsistency / physically implausible magnitudes
    flag(df["CO"] > 50, "CO", "implausible (>50 mg/m3, unit error)")       # GJ001 reports ug-scale CO
    flag(df["O3"] > 400, "O3", "implausible (>400 ug/m3)")                 # DL028 fault period
    # PM2.5 is a subset of PM10: PM2.5 > PM10 means one of them is wrong; the
    # target pollutant is kept, PM10 is discarded
    bad = df["PM10"].notna() & df["PM2.5"].notna() & (df["PM2.5"] > 1.05 * df["PM10"] + 5)
    flag(bad, "PM10", "PM2.5 > PM10")
    # flat-lined sensors: identical value for >= 12 consecutive hours (zeros in
    # NOx/CO/VOCs are below-detection values, but a 12 h zero run is still stuck)
    for col in POLLUTANTS:
        flag(_run_length(df, col) >= 11, col, "flat-line >= 12 h")
    # isolated spikes in PM2.5/PM10: a reading > 4x its trailing 24 h median AND
    # > 4x the median of the other stations in the same city at the same hour
    # (or > 6x trailing median for single-station cities). Genuine episodes
    # (Diwali, stubble smoke) lift the whole city, so they are kept.
    for col in ["PM2.5", "PM10"]:
        trail = (df.groupby("StationId", sort=False)[col]
                 .transform(lambda s: s.shift(1).rolling(24, min_periods=6).median()))
        city_med = df.groupby(["City", "Datetime"])[col].transform("median")
        city_n = df.groupby(["City", "Datetime"])[col].transform("count")
        x = df[col]
        spike_multi = (city_n >= 3) & (x > 4 * trail) & (x > 4 * city_med) & (x > 250)
        spike_single = (city_n < 3) & (x > 6 * trail) & (x > 400)
        flag(spike_multi | spike_single, col, "isolated spike")
    return df


# =============================================================== imputation
def impute(df, method="ffill", cols=None):
    """Impute pollutant inputs per station. The target is NOT taken from here.

    ffill    : carry last valid reading forward (<= 24 h). Causal.
    seasonal : fill with the station's value 24 h earlier (same hour yesterday),
               then 168 h earlier, then ffill. Causal; keeps the diurnal cycle.
    interp   : linear interpolation (<= 24 h). NON-causal - uses the next valid
               reading after a gap - included only to show how it inflates scores.
    """
    g = df.groupby("StationId", sort=False)
    for col in cols or POLLUTANTS:
        if method == "ffill":
            df[col] = g[col].ffill(limit=24)
        elif method == "seasonal":
            s = df[col]
            for lag in (24, 48, 168):
                s = s.fillna(g[col].shift(lag))
            df[col] = s
            df[col] = df.groupby("StationId", sort=False)[col].ffill(limit=24)
        elif method == "interp":
            df[col] = g[col].transform(lambda s: s.interpolate(limit=24, limit_area="inside"))
        else:
            raise ValueError(method)
    return df


# =============================================================== external data
def load_weather():
    files = glob.glob(os.path.join(EXT_DIR, "era5", "*.parquet"))
    if not files:
        print("WARNING: no ERA5 files found - run src/fetch_external.py")
        return None
    w = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    w = w.rename(columns={"time": "Datetime"}).sort_values(["City", "Datetime"])
    w["Datetime"] = w["Datetime"].astype("datetime64[ns]")
    ws = w["wind_speed_10m"] / 3.6                       # km/h -> m/s
    wd = np.deg2rad(w["wind_direction_10m"])
    out = pd.DataFrame({"City": w["City"], "Datetime": w["Datetime"]})
    out["met_temp"] = w["temperature_2m"]
    out["met_rh"] = w["relative_humidity_2m"]
    out["met_dewpt"] = w["dew_point_2m"]
    out["met_pres"] = w["surface_pressure"]
    out["met_precip"] = w["precipitation"]
    out["met_ws"] = ws
    out["met_u"] = -ws * np.sin(wd)                      # eastward component
    out["met_v"] = -ws * np.cos(wd)                      # northward component
    out["met_cloud"] = w["cloud_cover"]
    out["met_rad"] = w["shortwave_radiation"]
    out["met_blh"] = w["boundary_layer_height"]
    out["met_dpd"] = out["met_temp"] - out["met_dewpt"]  # dew-point depression
    out["met_vent"] = out["met_blh"] * out["met_ws"]     # ventilation coefficient
    g = out.groupby("City", sort=False)

    def roll(col, w_, fn="mean"):
        return g[col].transform(lambda s: getattr(s.rolling(w_, min_periods=1), fn)())

    for col in ["met_temp", "met_rh", "met_ws", "met_blh", "met_vent", "met_u", "met_v"]:
        out[f"{col}_mean24"] = roll(col, 24)
    out["met_blh_max24"] = roll("met_blh", 24, "max")    # daytime mixing depth
    out["met_vent_min24"] = roll("met_vent", 24, "min")  # worst night-time ventilation
    out["met_precip_sum24"] = roll("met_precip", 24, "sum")
    out["met_precip_sum72"] = roll("met_precip", 72, "sum")
    out["met_pres_tend24"] = out["met_pres"] - g["met_pres"].shift(24)
    out["met_temp_tend24"] = out["met_temp"] - g["met_temp"].shift(24)
    out["met_rh_mean168"] = roll("met_rh", 168)
    # "future" weather at t+24 - ERA5 used as a perfect-forecast proxy. NOT a
    # legitimate input for a real forecast; only used in a labelled ablation.
    for col in ["met_temp", "met_rh", "met_ws", "met_u", "met_v", "met_blh",
                "met_vent", "met_precip", "met_dpd"]:
        out["fut_" + col[4:]] = g[col].shift(-HORIZON)
    num = out.columns.drop(["City", "Datetime"])
    out[num] = out[num].astype(F32)
    return out


def _haversine(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = (np.sin((lat2 - lat1) / 2) ** 2 +
         np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2)
    return 6371 * 2 * np.arcsin(np.sqrt(a))


def _bearing(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    x = np.sin(lon2 - lon1) * np.cos(lat2)
    y = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(lon2 - lon1)
    return (np.degrees(np.arctan2(x, y)) + 360) % 360


def load_fires(weather):
    """Daily FIRMS MODIS counts around each city, keyed by the IST date on which
    they become usable (day D detections -> used from D+1, i.e. lag 1 day)."""
    files = sorted(glob.glob(os.path.join(EXT_DIR, "firms", "*.csv")))
    if not files:
        print("WARNING: no FIRMS files found - run src/fetch_external.py")
        return None
    f = pd.concat([pd.read_csv(p, usecols=["latitude", "longitude", "acq_date", "acq_time",
                                           "confidence", "frp", "type"]) for p in files])
    f = f[(f["confidence"] >= 30) & (f["type"].isin([0]))]   # vegetation fires, nominal+ conf.
    t = (pd.to_datetime(f["acq_date"]) + pd.to_timedelta(f["acq_time"] // 100, unit="h")
         + pd.to_timedelta(f["acq_time"] % 100, unit="m") + pd.Timedelta(hours=5, minutes=30))
    f = f.assign(date=t.dt.normalize().astype("datetime64[ns]"))
    # daily vector-mean wind direction per city (direction the wind blows FROM)
    if weather is not None:
        wd = weather.assign(date=weather["Datetime"].dt.normalize()).groupby(
            ["City", "date"])[["met_u", "met_v"]].mean()
        wd["wfrom"] = (np.degrees(np.arctan2(-wd["met_u"], -wd["met_v"])) + 360) % 360
        wd = wd["wfrom"]
    rows = []
    lat_f, lon_f = f["latitude"].to_numpy(), f["longitude"].to_numpy()
    for city, (lat, lon) in CITY_COORDS.items():
        d = _haversine(lat, lon, lat_f, lon_f)
        near = d <= 500
        sub = f[near].assign(dist=d[near], bearing=_bearing(lat, lon, lat_f[near], lon_f[near]))
        sub["City"] = city
        if weather is not None:
            sub = sub.join(wd.rename("wfrom"), on=["City", "date"])
            diff = np.abs((sub["bearing"] - sub["wfrom"] + 180) % 360 - 180)
            sub["upwind"] = (diff <= 45).astype(int)
        else:
            sub["upwind"] = 0
        rows.append(sub)
    f = pd.concat(rows)
    agg = f.groupby(["City", "date"]).agg(
        n100=("dist", lambda s: (s <= 100).sum()), n250=("dist", lambda s: (s <= 250).sum()),
        n500=("dist", "size"), frp500=("frp", "sum"), up500=("upwind", "sum"))
    # complete calendar per city
    days = pd.date_range("2014-12-01", "2020-07-03", freq="D").astype("datetime64[ns]")
    idx = pd.MultiIndex.from_product([list(CITY_COORDS), days], names=["City", "date"])
    agg = agg.reindex(idx, fill_value=0).astype(F32)
    g = agg.groupby(level="City")
    out = pd.DataFrame(index=agg.index)
    # usable on day D: detections from D-1 (and earlier)
    for c in ["n100", "n250", "n500", "frp500", "up500"]:
        out[f"fire_{c}_d1"] = g[c].shift(1)
    out["fire_n500_3d"] = g["n500"].transform(lambda s: s.shift(1).rolling(3, min_periods=1).sum())
    out["fire_up500_3d"] = g["up500"].transform(lambda s: s.shift(1).rolling(3, min_periods=1).sum())
    out["fire_n500_7d"] = g["n500"].transform(lambda s: s.shift(1).rolling(7, min_periods=1).sum())
    return out.astype(F32).reset_index()


# =============================================================== features
def _roll(g, col, w, fn="mean", min_periods=1):
    r = g[col].rolling(w, min_periods=min_periods)
    return getattr(r, fn)().reset_index(level=0, drop=True)


def add_features(df, obs_pm25, weather=None, fires=None):
    """df holds imputed pollutants; obs_pm25 the cleaned, un-imputed PM2.5."""
    t0 = time.time()
    F = {}
    g = df.groupby("StationId", sort=False)
    pm = "PM2.5"

    # ---------------- PM2.5 history (target lags) ----------------------
    for lag in [0, 1, 2, 3, 6, 9, 12, 18, 24, 48, 72, 168]:
        F[f"pm25_lag{lag}"] = g[pm].shift(lag)
    for w in [3, 6, 12, 24, 48, 72, 168, 336]:
        F[f"pm25_mean{w}"] = _roll(g, pm, w)
    F["pm25_std24"] = _roll(g, pm, 24, "std", 2)
    F["pm25_std168"] = _roll(g, pm, 168, "std", 2)
    F["pm25_min24"] = _roll(g, pm, 24, "min")
    F["pm25_max24"] = _roll(g, pm, 24, "max")
    F["pm25_ewm6"] = g[pm].transform(lambda s: s.ewm(halflife=6, ignore_na=True).mean())
    F["pm25_ewm24"] = g[pm].transform(lambda s: s.ewm(halflife=24, ignore_na=True).mean())
    F["pm25_diff1"] = F["pm25_lag0"] - F["pm25_lag1"]
    F["pm25_diff3"] = F["pm25_lag0"] - F["pm25_lag3"]
    F["pm25_diff24"] = F["pm25_lag0"] - F["pm25_lag24"]
    same_hour = pd.concat([g[pm].shift(24 * k) for k in range(7)], axis=1)
    F["pm25_samehour7"] = same_hour.mean(axis=1)            # seasonal-naive forecast
    F["pm25_samehour_ratio"] = F["pm25_samehour7"] / (F["pm25_mean168"] + 1)
    # value 23 h ago = hour *after* the target hour, yesterday (diurnal shape)
    F["pm25_lag23"] = g[pm].shift(23)

    # ---------------- other pollutants ---------------------------------
    for col in POLLUTANTS[1:]:
        key = col.replace(".", "")
        F[f"pol_{key}_lag0"] = df[col]
        F[f"pol_{key}_mean24"] = _roll(g, col, 24)
    for col in ["PM10", "NO2", "CO", "O3", "SO2"]:
        F[f"pol_{col}_lag3"] = g[col].shift(3)
        F[f"pol_{col}_lag24"] = g[col].shift(24)
        F[f"pol_{col}_mean168"] = _roll(g, col, 168)
    F["pol_pm_ratio"] = df[pm] / (df["PM10"] + 1)
    F["pol_PM10_diff24"] = df["PM10"] - g["PM10"].shift(24)

    # ---------------- missingness ------------------------------------
    obs = obs_pm25.notna()
    for col in POLLUTANTS:
        F[f"miss_{col.replace('.', '')}"] = df[f"_raw_missing_{col}"].astype(F32)
    F["miss_pm25_frac24"] = (1 - obs.astype(F32).groupby(df["StationId"], sort=False)
                             .rolling(24, min_periods=1).mean().reset_index(level=0, drop=True))
    last_obs = df["Datetime"].where(obs).groupby(df["StationId"], sort=False).ffill()
    F["miss_pm25_hours_since"] = ((df["Datetime"] - last_obs).dt.total_seconds() / 3600).clip(upper=48)

    feats = pd.DataFrame(F, index=df.index).astype(F32)
    feats[~np.isfinite(feats)] = np.nan
    print(f"  pollutant features: {feats.shape[1]} cols ({time.time() - t0:.0f}s)")

    # ---------------- neighbour stations -----------------------------
    key = [df["City"], df["Datetime"]]
    o = obs_pm25
    s_sum = o.groupby(key).transform("sum")
    s_cnt = o.groupby(key).transform("count")
    loo_cnt = s_cnt - obs
    feats["nbr_city_pm25_lag0"] = ((s_sum - o.fillna(0)) / loo_cnt.replace(0, np.nan)).astype(F32)
    feats["nbr_city_n"] = loo_cnt.astype(F32)
    r24 = feats["pm25_mean24"]
    r_sum = r24.groupby(key).transform("sum")
    r_cnt = r24.groupby(key).transform("count")
    feats["nbr_city_pm25_mean24"] = ((r_sum - r24.fillna(0)) /
                                     (r_cnt - r24.notna()).replace(0, np.nan)).astype(F32)
    city_max = o.groupby(key).transform("max")
    feats["nbr_city_pm25_max"] = city_max.where(loo_cnt > 0).astype(F32)
    # regional: inverse-distance mean of other cities within 400 km
    city_hour = o.groupby([df["Datetime"], df["City"]]).mean().unstack("City")
    full_idx = pd.date_range(city_hour.index.min(), city_hour.index.max(), freq="h")
    city_hour = city_hour.reindex(full_idx)
    city_hour_ff = city_hour.ffill(limit=3)
    reg_now, reg_24 = {}, {}
    city_roll24 = city_hour.rolling(24, min_periods=6).mean()
    for c in city_hour.columns:
        lat, lon = CITY_COORDS.get(c, (np.nan, np.nan))
        others = [(o_, _haversine(lat, lon, *CITY_COORDS[o_])) for o_ in city_hour.columns
                  if o_ != c and o_ in CITY_COORDS]
        others = [(o_, d) for o_, d in others if d <= 400]
        if not others:
            continue
        names = [o_ for o_, _ in others]
        wts = np.array([1 / d for _, d in others])
        for src, dst in [(city_hour_ff, reg_now), (city_roll24, reg_24)]:
            vals = src[names].to_numpy()
            m = ~np.isnan(vals)
            num = np.nansum(vals * wts, axis=1)
            den = (m * wts).sum(axis=1)
            dst[c] = pd.Series(np.where(den > 0, num / np.where(den > 0, den, 1), np.nan),
                               index=src.index)
    reg_now = pd.DataFrame(reg_now).stack().rename("nbr_region_pm25_lag0")
    reg_24 = pd.DataFrame(reg_24).stack().rename("nbr_region_pm25_mean24")
    reg = pd.concat([reg_now, reg_24], axis=1)
    reg.index.names = ["Datetime", "City"]
    tmp = df[["Datetime", "City"]].join(reg, on=["Datetime", "City"])
    feats["nbr_region_pm25_lag0"] = tmp["nbr_region_pm25_lag0"].astype(F32).to_numpy()
    feats["nbr_region_pm25_mean24"] = tmp["nbr_region_pm25_mean24"].astype(F32).to_numpy()

    # ---------------- calendar of the TARGET time (known in advance) ---
    T = df["Datetime"] + pd.Timedelta(hours=HORIZON)
    cal = pd.DataFrame(index=df.index)
    cal["cal_hour_sin"] = np.sin(2 * np.pi * T.dt.hour / 24)
    cal["cal_hour_cos"] = np.cos(2 * np.pi * T.dt.hour / 24)
    cal["cal_dow"] = T.dt.dayofweek
    cal["cal_weekend"] = (T.dt.dayofweek >= 5).astype(int)
    cal["cal_month_sin"] = np.sin(2 * np.pi * T.dt.month / 12)
    cal["cal_month_cos"] = np.cos(2 * np.pi * T.dt.month / 12)
    cal["cal_doy_sin"] = np.sin(2 * np.pi * T.dt.dayofyear / 365.25)
    cal["cal_doy_cos"] = np.cos(2 * np.pi * T.dt.dayofyear / 365.25)
    Td = T.dt.normalize()
    hol = pd.to_datetime(NATIONAL_HOLIDAYS + sum(FESTIVALS.values(), []))
    cal["cal_holiday"] = Td.isin(hol).astype(int)
    diwali = pd.to_datetime(FESTIVALS["Diwali"]).to_numpy()
    dd = (Td.to_numpy()[:, None] - diwali[None, :]) / np.timedelta64(1, "D")
    nearest = dd[np.arange(len(dd)), np.abs(dd).argmin(axis=1)]
    cal["cal_days_from_diwali"] = np.clip(nearest, -30, 30)
    cal["cal_diwali_window"] = ((nearest >= -1) & (nearest <= 3)).astype(int)
    doy = T.dt.dayofyear
    cal["cal_stubble_season"] = ((doy >= 283) & (doy <= 334)).astype(int)   # ~10 Oct - 30 Nov
    cal["cal_winter"] = T.dt.month.isin([11, 12, 1, 2]).astype(int)
    feats = pd.concat([feats, cal.astype(F32)], axis=1)

    # ---------------- location --------------------------------------
    feats["loc_lat"] = df["City"].map({c: v[0] for c, v in CITY_COORDS.items()}).astype(F32)
    feats["loc_lon"] = df["City"].map({c: v[1] for c, v in CITY_COORDS.items()}).astype(F32)

    out = pd.concat([df[["StationId", "StationName", "City", "State", "Datetime"]], feats], axis=1)
    out["PM2.5_obs"] = obs_pm25.astype(F32)

    # ---------------- external merges -------------------------------
    if weather is not None:
        out = out.merge(weather, on=["City", "Datetime"], how="left")
    if fires is not None:
        out["date"] = out["Datetime"].dt.normalize().astype("datetime64[ns]")
        out = out.merge(fires, on=["City", "date"], how="left").drop(columns="date")

    # ---------------- targets (from un-imputed observations) ---------
    out["target"] = out.groupby("StationId", sort=False)["PM2.5_obs"].shift(-HORIZON)
    out["target_poor"] = (out["target"] > POOR_THRESHOLD).astype(F32)
    out.loc[out["target"].isna(), "target_poor"] = np.nan
    print(f"  all features built ({time.time() - t0:.0f}s)")
    return out


def feature_columns(df, include_future=False):
    pref = FEATURE_PREFIXES + (("fut_",) if include_future else ())
    return [c for c in df.columns if c.startswith(pref)]


def feature_group(col):
    for p in GROUP_NAMES:
        if col.startswith(p):
            return p
    return "other"


def chrono_split(df, train_end=TRAIN_END, val_end=VAL_END, embargo=HORIZON):
    """Chronological split on forecast-issue time with an embargo of HORIZON
    hours before each boundary so no training target falls in a later period."""
    t = df["Datetime"]
    te, ve = pd.Timestamp(train_end), pd.Timestamp(val_end)
    gap = pd.Timedelta(hours=embargo)
    train = df[t <= te - gap]
    val = df[(t > te) & (t <= ve - gap)]
    test = df[t > ve]
    return train, val, test


def build_dataset(impute_method="ffill", with_external=True, verbose=True):
    t0 = time.time()
    df = load_raw()
    log = {}
    df = clean(df, log)
    for col in POLLUTANTS:
        df[f"_raw_missing_{col}"] = df[col].isna()
    obs_pm25 = df["PM2.5"].copy()
    df = impute(df, impute_method)
    weather = load_weather() if with_external else None
    fires = load_fires(weather) if with_external else None
    out = add_features(df, obs_pm25, weather, fires)
    # keep rows where a forecast can be issued (some PM2.5 history at t) and a
    # label exists
    out = out[out["target"].notna() & out["pm25_lag0"].notna()].reset_index(drop=True)
    if verbose:
        print(f"built dataset [{impute_method}]: {out.shape}, "
              f"{len(feature_columns(out))} features, {time.time() - t0:.0f}s")
    return out, log


if __name__ == "__main__":
    data, fault_log = build_dataset("ffill")
    pd.Series(fault_log).rename("values_removed").rename_axis(["pollutant", "rule"]).to_csv(
        os.path.join(RESULTS_DIR, "fault_log.csv"))
    data.to_parquet(PROCESSED, index=False)
    print(data.shape, len(feature_columns(data)), "features")
    print("rows by split:", [len(x) for x in chrono_split(data)])
