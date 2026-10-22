
from pathlib import Path

# Project configuration
START_DATE = "2013-01-01"
END_DATE = "2023-12-31"
TIMEZONE = "Europe/Lisbon"

WARMUP_END = "2020-09-13"
WARMUP_PERCENTAGE = 0.33  # Currently unused

HOT_PERCENTILE = 0.80
MIN_CONSECUTIVE_HOT_DAYS = 3

WINDOW_SIZE = 24 * 30  # 30 days of hourly observations
FEATURE_WINDOWS = (6, 24)
RANDOM_SEED = 42

# Paths relative to this file's directory
BASE_DIR = Path(__file__).resolve().parent

RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
RESULTS_DIR = BASE_DIR / "results"

# Create directories if they do not exist
for directory in (RAW_DIR, PROCESSED_DIR, RESULTS_DIR):
    directory.mkdir(parents=True, exist_ok=True)

# City coordinates
CITIES = {
    "Aveiro": {
        "latitude": 40.6405,
        "longitude": -8.6538,
    },
    "Viseu": {
        "latitude": 40.6572,
        "longitude": -7.9142,
    },
}

WEATHER_COLUMNS = [
    "temperature",
    "humidity",
    "precipitation",
    "wind_speed",
]

PHYSICAL_BOUNDS = {
    "temperature": (-10.0, 50.0),
    "humidity": (0.0, 100.0),
    "precipitation": (0.0, 150.0),
    "wind_speed": (0.0, 150.0),
}