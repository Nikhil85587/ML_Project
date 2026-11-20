# Data sources, alignment and caveats

| # | Source | Provider / programme | What we use | Resolution | Licence / terms |
|---|---|---|---|---|---|
| 1 | **Air Quality Data in India (2015–2020)**, `station_hour.csv`, `stations.csv` (Kaggle, *rohanrao*) | Compiled from the **Central Pollution Control Board (CPCB)** CAAQMS portal | 12 pollutants (PM2.5, PM10, NO, NO2, NOx, NH3, CO, SO2, O3, benzene, toluene, xylene) at 110 stations / 26 cities (+ AQI columns, which we **drop**) | Station, hourly, Jan 2015 – Jul 2020 | Kaggle dataset licence (CC0/public CPCB data); cite CPCB |
| 2 | **ERA5 hourly reanalysis** via the Open-Meteo historical-weather API (`models=era5`) | **ECMWF / Copernicus Climate Change Service (C3S)** | 2 m temperature, relative humidity, dew point, surface pressure, precipitation, 10 m wind speed & direction, cloud cover, shortwave radiation, **boundary-layer height** | 0.25° grid (~28 km), hourly, IST | CC-BY 4.0 (Copernicus licence; Open-Meteo CC-BY 4.0) |
| 3 | **NASA FIRMS MODIS Collection 6.1** active-fire detections, yearly country files for India (2014–2020) | **NASA LANCE / FIRMS** (Terra & Aqua MODIS) | Fire location, acquisition time, confidence, fire radiative power (FRP) | 1 km pixel, ~4 overpasses/day | NASA open-data policy (free use, please cite FIRMS) |
| 4 | Festival & holiday calendar (Diwali, Dussehra, Holi, Lohri, Christmas, New Year, Republic/Independence/Gandhi Jayanti) | Government of India public holiday calendars (hand-compiled, `src/config.py`) | Holiday flags, days-from-Diwali | Daily | Public information |

All downloads are scripted in `src/fetch_external.py` (no credentials required).

## Why these sources

* **CPCB** is the national regulator's monitoring network – a genuine government source with
  2.59 M station-hours (far above the 100 k requirement) and spatial coverage from
  Amritsar to Thiruvananthapuram.
* **Meteorology controls dispersion.** Low boundary-layer height (BLH), calm wind and high
  humidity trap pollutants (winter inversions); rain removes particles. ERA5 is the
  reference global reanalysis and is the only freely available hourly product with BLH for
  all 26 cities and the whole 2015–2020 period.
* **Crop-residue burning** in Punjab/Haryana (Oct–Nov) and forest fires (spring, NE India)
  are major PM2.5 sources for downwind cities. FIRMS is the standard satellite fire record.
* **Festivals** (Diwali fireworks) cause the sharpest spikes in the record.

## Temporal and spatial alignment

| Source | Spatial alignment | Temporal alignment | Availability at forecast time *t* |
|---|---|---|---|
| CPCB | – | Native hourly timestamps (IST) | Readings up to *t* |
| ERA5 | Nearest 0.25° grid cell to each **city centre** (`CITY_COORDS`); every station in a city shares it – ERA5's 28 km grid is coarser than intra-city station spacing, so this loses little | Requested directly in `Asia/Kolkata`, so hours match CPCB | Values up to *t* only (current value, trailing 24/72/168 h aggregates, 24 h tendencies). Weather at *t+24* is **not** used except in one clearly labelled "oracle" ablation |
| FIRMS | Great-circle distance from the city centre: counts within 100 / 250 / 500 km, FRP within 500 km, and **upwind** fires (bearing within ±45° of the day's vector-mean wind-from direction) | UTC acquisition time → IST date | Detections of day *D* are used from day *D+1* (lag ≥ 1 day; NRT latency is ~3 h, so this is conservative) |
| Calendar | – | Date of the **target** time *t+24* | Known in advance |

## Quality caveats

* **CPCB**: sensor outages, stations commissioned at different dates (most Delhi DPCC stations
  start 2018, Mumbai/Kolkata mid-2019), instrument caps at 1000 µg/m³ (PM) / 500 (gases), a
  CO unit inconsistency at GJ001 (Ahmedabad), an O3 fault period at DL028, flat-lined sensors.
  Two stations (DL011, KA005, KA010 for PM2.5) never report PM2.5 and cannot supply a target.
* **ERA5**: reanalysis, not station observations – it smooths local extremes (urban heat
  island, sea breeze) and BLH is a model diagnostic. Using it at *t* is equivalent to having a
  good analysis of current weather, which operational centres do have.
* **FIRMS MODIS**: misses fires under cloud/smoke and small fires; only ~4 overpasses per day.
  Fire counts are a proxy for emissions, not a transport model.
* **Calendar**: lunar festival dates were checked against official holiday lists; regional
  festivals are not included.
* **COVID-19 lockdown (25 Mar – May 2020)** is inside the test period – an abrupt emission drop
  no training example resembles (see robustness analysis).
