"""
Download the external data sources merged with the CPCB monitoring data.

1. CPCB station data  - Kaggle "Air Quality Data in India (2015-2020)" (rohanrao),
   compiled from the Central Pollution Control Board CAAQMS portal.
2. Meteorology        - ECMWF ERA5 hourly reanalysis (Copernicus Climate Change
   Service), served by the Open-Meteo historical-weather API, one 0.25 deg grid
   cell per city (nearest cell to the city centre). Licence: CC-BY 4.0.
3. Fire hotspots      - NASA FIRMS MODIS Collection 6.1 active-fire detections,
   yearly country files for India (crop-residue / biomass burning). Public domain
   (NASA data policy; please cite FIRMS).

Usage:  python src/fetch_external.py
"""
import os
import shutil
import sys
import time

import pandas as pd
import requests

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import CITY_COORDS, DATA_DIR, EXT_DIR  # noqa: E402

ERA5_VARS = ["temperature_2m", "relative_humidity_2m", "dew_point_2m",
             "surface_pressure", "precipitation", "wind_speed_10m",
             "wind_direction_10m", "cloud_cover", "shortwave_radiation",
             "boundary_layer_height"]
START, END = "2015-01-01", "2020-07-03"
FIRMS_YEARS = range(2014, 2021)  # 2014 so the first days of 2015 have fire history


def fetch_cpcb():
    """Copy the Kaggle CPCB files into data/ (anonymous kagglehub download)."""
    need = ["station_hour.csv", "stations.csv"]
    if all(os.path.exists(os.path.join(DATA_DIR, f)) for f in need):
        print("CPCB files already present")
        return
    import kagglehub
    path = kagglehub.dataset_download("rohanrao/air-quality-data-in-india")
    for f in need + ["station_day.csv"]:
        shutil.copy(os.path.join(path, f), DATA_DIR)
    print("CPCB files copied from", path)


def fetch_era5():
    out_dir = os.path.join(EXT_DIR, "era5")
    os.makedirs(out_dir, exist_ok=True)
    for city, (lat, lon) in CITY_COORDS.items():
        out = os.path.join(out_dir, f"{city}.parquet")
        if os.path.exists(out):
            continue
        params = {"latitude": lat, "longitude": lon, "start_date": START,
                  "end_date": END, "hourly": ",".join(ERA5_VARS),
                  "timezone": "Asia/Kolkata", "models": "era5"}
        js = None
        for attempt in range(10):
            try:
                r = requests.get("https://archive-api.open-meteo.com/v1/archive",
                                 params=params, timeout=300)
                if r.status_code == 200:
                    js = r.json()
                    break
                msg = f"HTTP {r.status_code}"
            except (requests.RequestException, ValueError) as e:
                msg = e.__class__.__name__
            wait = 30 * (attempt + 1)
            print(f"  {city}: {msg}, retrying in {wait}s", flush=True)
            time.sleep(wait)
        if js is None:
            raise RuntimeError(f"ERA5 download failed for {city}")
        df = pd.DataFrame(js["hourly"])
        df["time"] = pd.to_datetime(df["time"])
        df.insert(0, "City", city)
        df["grid_lat"], df["grid_lon"] = js["latitude"], js["longitude"]
        df.to_parquet(out, index=False)
        print(f"ERA5 {city}: {len(df)} rows (grid {js['latitude']},{js['longitude']})")
        time.sleep(15)  # stay well inside the free-tier rate limit


def fetch_firms():
    out_dir = os.path.join(EXT_DIR, "firms")
    os.makedirs(out_dir, exist_ok=True)
    for y in FIRMS_YEARS:
        out = os.path.join(out_dir, f"modis_{y}_India.csv")
        if os.path.exists(out):
            continue
        url = f"https://firms.modaps.eosdis.nasa.gov/data/country/modis/{y}/modis_{y}_India.csv"
        for attempt in range(6):
            try:
                r = requests.get(url, timeout=300)
                r.raise_for_status()
                break
            except requests.RequestException as e:
                print(f"  FIRMS {y}: {e.__class__.__name__}, retry {attempt + 1}")
                time.sleep(10 * (attempt + 1))
        else:
            raise RuntimeError(f"could not download {url}")
        with open(out + ".tmp", "wb") as fh:
            fh.write(r.content)
        os.replace(out + ".tmp", out)
        print(f"FIRMS {y}: {len(r.content) / 1e6:.1f} MB")


if __name__ == "__main__":
    os.makedirs(EXT_DIR, exist_ok=True)
    fetch_cpcb()
    fetch_firms()
    fetch_era5()
    print("done")
