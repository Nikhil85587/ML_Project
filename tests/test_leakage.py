"""
Automated leakage audit.

Run:  python -m pytest tests -q      (or: python tests/test_leakage.py)

1. Future-perturbation test: corrupt every pollutant reading after a cut-off
   time t0 for one city, rebuild all features, and check that no feature value
   at issue times <= t0 changes. Any window that peeks into the future
   (centred rolling, non-causal imputation, mis-aligned merge) would fail.
2. Target-alignment test: target(t) equals the observed PM2.5 at t+24.
3. Forbidden-columns test: AQI / AQI_Bucket / same-time target never appear as
   features, and no feature is (near-)identical to the target.
4. Split test: no training target time falls inside the validation/test period.
"""
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from config import HORIZON, POLLUTANTS, PROCESSED, TRAIN_END, VAL_END  # noqa: E402
from data_prep import (add_features, chrono_split, clean, feature_columns,  # noqa: E402
                       impute, load_fires, load_raw, load_weather)

CITY = "Lucknow"
T0 = pd.Timestamp("2019-03-01 12:00")


def _build(raw, weather, fires, method="ffill"):
    df = clean(raw.copy())
    for col in POLLUTANTS:
        df[f"_raw_missing_{col}"] = df[col].isna()
    obs = df["PM2.5"].copy()
    df = impute(df, method)
    return add_features(df, obs, weather, fires)


def test_future_perturbation():
    raw = load_raw()
    raw = raw[raw["City"] == CITY].reset_index(drop=True)
    weather, fires = load_weather(), None
    if weather is not None:
        fires = load_fires(weather)
    base = _build(raw, weather, fires)
    pert = raw.copy()
    fut = pert["Datetime"] > T0
    rng = np.random.default_rng(0)
    for col in POLLUTANTS:
        pert.loc[fut, col] = (pert.loc[fut, col] * rng.uniform(0.2, 5, fut.sum()) + 50).astype(np.float32)
    # also blank out part of the future to change missingness patterns
    pert.loc[fut & (rng.random(len(pert)) < 0.3), "PM2.5"] = np.nan
    new = _build(pert, weather, fires)
    feats = feature_columns(base)
    past = base["Datetime"] <= T0
    a = base.loc[past, feats].to_numpy(dtype=float)
    b = new.loc[past, feats].to_numpy(dtype=float)
    same = np.isclose(a, b, equal_nan=True, rtol=1e-5, atol=1e-4)
    bad = [f for f, ok in zip(feats, same.all(axis=0)) if not ok]
    assert not bad, f"features at t<=t0 changed when only the future changed: {bad}"
    # sanity: the perturbation does change the targets of the last 24 h before t0
    tgt_changed = ~np.isclose(base.loc[past, "target"], new.loc[past, "target"], equal_nan=True)
    assert tgt_changed.any()


def test_interpolation_is_leaky():
    """Documented negative control: linear interpolation must FAIL the test."""
    raw = load_raw()
    raw = raw[raw["City"] == CITY].reset_index(drop=True)
    base = _build(raw, None, None, "interp")
    pert = raw.copy()
    fut = pert["Datetime"] > T0
    pert.loc[fut, "PM2.5"] = (pert.loc[fut, "PM2.5"] + 300).astype(np.float32)
    new = _build(pert, None, None, "interp")
    past = base["Datetime"] <= T0
    assert not np.allclose(base.loc[past, "pm25_lag0"], new.loc[past, "pm25_lag0"],
                           equal_nan=True), "interpolation should leak future values"


def _processed():
    if not os.path.exists(PROCESSED):
        import pytest
        pytest.skip("processed.parquet not built")
    return pd.read_parquet(PROCESSED, columns=None)


def test_target_alignment_and_forbidden_columns():
    df = _processed()
    feats = feature_columns(df)
    for col in ["AQI", "AQI_Bucket", "target", "target_poor", "PM2.5_obs"]:
        assert col not in feats
    assert not any(f.startswith("fut_") for f in feats)
    smp = df.sample(200_000, random_state=0)
    corr = smp[feats].corrwith(smp["target"]).abs()
    assert corr.max() < 0.95, f"suspiciously high correlation: {corr.idxmax()}={corr.max():.3f}"
    st = df[df["StationId"] == df["StationId"].iloc[0]].set_index("Datetime")
    shifted = st["PM2.5_obs"].reindex(st.index + pd.Timedelta(hours=HORIZON)).to_numpy()
    ok = ~np.isnan(shifted)
    assert np.allclose(st["target"].to_numpy()[ok], shifted[ok])


def test_split_embargo():
    df = _processed()[["Datetime", "target"]]
    tr, va, te = chrono_split(df)
    assert (tr["Datetime"] + pd.Timedelta(hours=HORIZON)).max() <= pd.Timestamp(TRAIN_END)
    assert (va["Datetime"] + pd.Timedelta(hours=HORIZON)).max() <= pd.Timestamp(VAL_END)
    assert va["Datetime"].min() > pd.Timestamp(TRAIN_END)
    assert te["Datetime"].min() > pd.Timestamp(VAL_END)


if __name__ == "__main__":
    for fn in [test_future_perturbation, test_interpolation_is_leaky,
               test_target_alignment_and_forbidden_columns, test_split_embargo]:
        fn()
        print("PASS", fn.__name__)
