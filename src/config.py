"""Project configuration for the heat-stress stream-mining assignment."""

from pathlib import Path

import pandas as pd

# ======================================================================
# 1. PATHS
# ======================================================================
# This file lives in src/, so the project root is two levels up.
BASE_DIR = Path(__file__).resolve().parent.parent

RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
PREPROCESSED_DIR = BASE_DIR / "data" / "preprocessed"
RESULTS_DIR = BASE_DIR / "results"

for directory in (RAW_DIR, PROCESSED_DIR, PREPROCESSED_DIR, RESULTS_DIR):
    directory.mkdir(parents=True, exist_ok=True)

# ======================================================================
# 2. DATA SOURCE
# ======================================================================
START_DATE = "2013-01-01"
END_DATE = "2023-12-31"
TIMEZONE = "Europe/Lisbon"

CITIES = {
    "Aveiro": {"latitude": 40.6405, "longitude": -8.6538},
    "Viseu": {"latitude": 40.6572, "longitude": -7.9142},
}

# ======================================================================
# 3. SCHEMA AND DATA QUALITY
# ======================================================================
WEATHER_COLUMNS = [
    "temperature",
    "relative_humidity",
    "precipitation",
    "wind_speed",
    "night_min_temperature",
]

REQUIRED_COLUMNS = ["time", *WEATHER_COLUMNS, "city"]

# Physically plausible ranges, used by the audit
PHYSICAL_BOUNDS = {
    "temperature": (-30, 50),
    "relative_humidity": (0, 100),
    "precipitation": (0, 200),
    "wind_speed": (0, 150),
    "night_min_temperature": (-30, 50),
}

# ======================================================================
# 4. NIGHT-MINIMUM FEATURE
# ======================================================================
# Night window: 21:00 -> 06:00 local time
NIGHT_START_HOUR = 21
NIGHT_END_HOUR = 6
NIGHT_HOURS = 24 - NIGHT_START_HOUR + NIGHT_END_HOUR  # 9 hours

# Extra days downloaded before START_DATE so the first rows have a complete
# preceding night. Trimmed away after the feature is computed.
NIGHT_BUFFER_DAYS = 2

# ======================================================================
# 5. STREAM SETUP: WARM-UP AND LABELLING
# ======================================================================
WARMUP_START = pd.Timestamp("2013-01-01")
WARMUP_END = pd.Timestamp("2015-12-31")
WARMUP_PERCENTAGE = 0.10

HOT_PERCENTILE = 0.80           # fraction in [0, 1], not a percentage
MIN_CONSECUTIVE_HOT_DAYS = 3


# ======================================================================
# 6. MODELLING
# ======================================================================
WINDOW_SIZE = 24 * 30           # 30 days of hourly observations
FEATURE_WINDOWS = (6, 24)       # hours
RANDOM_SEED = 42

# ======================================================================
# 7. PLOTTING: VARIABLES
# ======================================================================
VAR_STYLE = {
    "temperature":           dict(label="Temperature",           unit="°C",   color="#d1495b", kind="line"),
    "relative_humidity":     dict(label="Relative Humidity",     unit="%",    color="#2a9d8f", kind="line"),
    "precipitation":         dict(label="Precipitation",         unit="mm",   color="#1d4e89", kind="bar"),
    "wind_speed":            dict(label="Wind speed",            unit="km/h", color="#c98b27", kind="line"),
    "night_min_temperature": dict(label="Night min temperature", unit="°C",   color="#6d597a", kind="line"),
}

# Short labels and units used in compact plots
VAR_LABEL = {
    "temperature": "Temperature",
    "relative_humidity": "Humidity",
    "precipitation": "Precipitation",
    "wind_speed": "Wind speed",
    "night_min_temperature": "Night min temp",
}

VAR_UNIT = {
    "temperature": "°C",
    "relative_humidity": "%",
    "precipitation": "mm/h",
    "wind_speed": "km/h",
    "night_min_temperature": "°C",
}

# ======================================================================
# 8. PLOTTING: COLOURS AND AXES
# ======================================================================
C_NON = "#5b7fa6"   # non-event
C_EVT = "#d62728"   # heat-stress event
C_TAU = "#7b2cbf"   # threshold line
FALLBACK_COLORS = ["#5b7fa6", "#8d6bb8", "#6a994e", "#e76f51"]

PRECIP_TICKS = [0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50]

# ======================================================================
# 9. PLOTTING: SEASONS (meteorological)
# ======================================================================
SEASONS = ["Winter", "Spring", "Summer", "Autumn"]

SEASON_MONTHS = {
    "Winter": "Dec–Feb",
    "Spring": "Mar–May",
    "Summer": "Jun–Aug",
    "Autumn": "Sep–Nov",
}

MONTH_TO_SEASON = {
    12: "Winter", 1: "Winter", 2: "Winter",
    3: "Spring", 4: "Spring", 5: "Spring",
    6: "Summer", 7: "Summer", 8: "Summer",
    9: "Autumn", 10: "Autumn", 11: "Autumn",
}


# ======================================================================
# 10. FEATURE ENGINEERING
# ======================================================================
FEATURE_COLS_A = [
    #"temperature_lag1",
    "relative_humidity_lag1",
    "precipitation_lag1",
    "wind_speed_lag1",
]

FEATURE_COLS_B = FEATURE_COLS_A + [
    #"delta_temp",
    "rolling_mean_temp_6h",
    "rolling_std_temp_6h",
    "rolling_mean_temp_24h",
    "heat_events_last_24h",
]