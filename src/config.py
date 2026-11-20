"""Project-wide constants: paths, forecast set-up, splits, external metadata."""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")
EXT_DIR = os.path.join(DATA_DIR, "external")
# env overrides allow a quick smoke-test run on a sample without touching outputs
MODEL_DIR = os.environ.get("AQ_MODEL_DIR", os.path.join(ROOT, "models"))
RESULTS_DIR = os.environ.get("AQ_RESULTS_DIR", os.path.join(ROOT, "results"))
FIG_DIR = os.path.join(RESULTS_DIR, "figures")
PROCESSED = os.environ.get("AQ_PROCESSED", os.path.join(DATA_DIR, "processed.parquet"))

# ---------------------------------------------------------------- forecast task
HORIZON = 24            # hours ahead: forecast issued at t predicts PM2.5 at t+24
TARGET_POLLUTANT = "PM2.5"
POOR_THRESHOLD = 90     # ug/m3; CPCB PM2.5 sub-index "Poor" band starts at 91
SEED = 42
# Models are fitted on a transformed target (variance-stabilising for the
# heavy right tail of PM2.5); all metrics are computed back in ug/m3.
# Chosen on the validation period: sqrt matched log1p on MAE but gave lower
# RMSE, higher Poor+ F1 and smaller errors on extremes; raw was worst on MAE.
TARGET_TRANSFORM = "sqrt"

# chronological split (forecast-issue time). Test covers a full annual cycle,
# including the Oct-Dec 2019 stubble/Diwali season and the 2020 COVID lockdown.
TRAIN_END = "2018-12-31 23:00"
VAL_END = "2019-06-30 23:00"

POLLUTANTS = ["PM2.5", "PM10", "NO", "NO2", "NOx", "NH3", "CO", "SO2", "O3",
              "Benzene", "Toluene", "Xylene"]

# CPCB PM2.5 (24-h) sub-index breakpoints -> AQI category
PM25_BANDS = [(30, "Good"), (60, "Satisfactory"), (90, "Moderate"),
              (120, "Poor"), (250, "Very Poor"), (float("inf"), "Severe")]
CATEGORY_COLORS = {"Good": "#55a84f", "Satisfactory": "#a3c853",
                   "Moderate": "#fff833", "Poor": "#f29c33",
                   "Very Poor": "#e93f33", "Severe": "#af2d24"}


def pm25_category(v):
    for limit, label in PM25_BANDS:
        if v <= limit:
            return label
    return "Severe"


# City-centre coordinates (deg). ERA5 is on a 0.25 deg (~28 km) grid, so a
# city-level point is the natural resolution for aligning stations to weather.
CITY_COORDS = {
    "Ahmedabad": (23.02, 72.57), "Aizawl": (23.73, 92.72),
    "Amaravati": (16.51, 80.52), "Amritsar": (31.63, 74.87),
    "Bengaluru": (12.97, 77.59), "Bhopal": (23.26, 77.41),
    "Brajrajnagar": (21.82, 83.92), "Chandigarh": (30.73, 76.78),
    "Chennai": (13.08, 80.27), "Coimbatore": (11.02, 76.96),
    "Delhi": (28.61, 77.21), "Ernakulam": (9.98, 76.28),
    "Gurugram": (28.46, 77.03), "Guwahati": (26.14, 91.74),
    "Hyderabad": (17.39, 78.49), "Jaipur": (26.91, 75.79),
    "Jorapokhar": (23.70, 86.41), "Kochi": (9.97, 76.30),
    "Kolkata": (22.57, 88.36), "Lucknow": (26.85, 80.95),
    "Mumbai": (19.08, 72.88), "Patna": (25.59, 85.14),
    "Shillong": (25.58, 91.89), "Talcher": (20.95, 85.23),
    "Thiruvananthapuram": (8.52, 76.94), "Visakhapatnam": (17.69, 83.22),
}

# Major festivals linked to emission spikes (firecrackers, bonfires, traffic).
# Hand-compiled from public Government of India holiday calendars.
FESTIVALS = {
    "Diwali": ["2015-11-11", "2016-10-30", "2017-10-19", "2018-11-07",
               "2019-10-27", "2020-11-14"],
    "Dussehra": ["2015-10-22", "2016-10-11", "2017-09-30", "2018-10-19",
                 "2019-10-08", "2020-10-25"],
    "Holi": ["2015-03-06", "2016-03-24", "2017-03-13", "2018-03-02",
             "2019-03-21", "2020-03-10"],
    "Christmas": [f"{y}-12-25" for y in range(2015, 2021)],
    "NewYear": [f"{y}-01-01" for y in range(2015, 2021)],
    "Lohri": [f"{y}-01-13" for y in range(2015, 2021)],
}
NATIONAL_HOLIDAYS = ([f"{y}-01-26" for y in range(2015, 2021)] +
                     [f"{y}-08-15" for y in range(2015, 2021)] +
                     [f"{y}-10-02" for y in range(2015, 2021)])
