"""Weekend Challenge 01 — Instructor Solution / Worked Example.

Data Stream Mining — NOVA IMS

Challenge idea
--------------
Choose real-world historical data, save a reproducible raw snapshot, transform
it into timestamped observations, replay those observations one at a time, and
maintain useful statistics without recalculating them from the full dataset.

This solution uses Open-Meteo historical weather data for Lisbon by default.
It is intentionally verbose: the goal is not only to run the code, but to make
its structure easy to read, modify, and reuse.

Recommended notebooks to consult
--------------------------------
1. Week01_Extra_From_APIs_to_Real_World_Data_Streams.ipynb
   - Section 4: From API Data to a Replay Stream
   - Section 6: Open-Meteo
   - Section 11: Live Mode vs Replay Mode

2. Week02_Foundations_of_Data_Stream_Mining_Lab.ipynb
   - Section 4: Count, Sum, and Running Mean
   - Section 5: Variance Without Storing the Data (Welford)
   - Section 6: The Same Idea with River

How to run
----------
    python weekend_challenge_01_solution.py

Packages
--------
    python -m pip install requests pandas matplotlib

Important streaming rule
------------------------
Once the replay loop starts, the online algorithm must NOT use df.mean(),
df.var(), df.describe(), or other whole-table calculations to maintain its
state. Batch calculations are used only at the end to validate the online
implementation.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Callable, Iterable, Iterator

import pandas as pd
import requests


# ============================================================================
# 1. STUDENT CUSTOMIZATION ZONE — CHANGE HERE
# ============================================================================
# These are the main values students are encouraged to change.
#
# Try, for example:
# - Porto:   41.1579, -8.6291
# - Faro:    37.0194, -7.9304
# - Funchal: 32.6669, -16.9241
#
# Open-Meteo's Historical Weather API lets us request a fixed historical
# period, which is excellent for reproducible replay experiments.

CITY_NAME = "Lisbon"
LATITUDE = 38.7223
LONGITUDE = -9.1393

# Pick a historical period. Keeping fixed dates makes the example reproducible.
START_DATE = "2025-09-01"
END_DATE = "2025-09-07"

# We download several variables so the raw snapshot can support experiments.
# If you add a new STREAM_VARIABLE, also add it to HOURLY_VARIABLES.
HOURLY_VARIABLES = [
    "temperature_2m",
    "relative_humidity_2m",
    "precipitation",
    "wind_speed_10m",
]

# Main numeric variable used by the replay stream.
# Other easy choices: "relative_humidity_2m", "precipitation",
# "wind_speed_10m".
STREAM_VARIABLE = "temperature_2m"

TIMEZONE = "Europe/Lisbon"

# False = reuse an existing raw JSON snapshot if it is already present.
# True  = contact the API again and overwrite the snapshot.
FORCE_DOWNLOAD = False

# Output folders are created next to this script.
BASE_DIR = Path(__file__).resolve().parent
RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
FIGURES_DIR = BASE_DIR / "figures"

# Official Open-Meteo Historical Weather endpoint.
OPEN_METEO_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"


# ============================================================================
# 2. SMALL HELPERS
# ============================================================================

def slugify(text: str) -> str:
    """Turn a label into a simple filename-friendly string."""
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def build_raw_filename(city_name: str, start_date: str, end_date: str) -> str:
    """Create a stable name for the saved raw API response."""
    return f"open_meteo_{slugify(city_name)}_{start_date}_{end_date}.json"


def build_processed_filename(
    city_name: str,
    start_date: str,
    end_date: str,
    variable: str,
) -> str:
    """Create a stable name for the cleaned table used by the replay."""
    return (
        f"open_meteo_{slugify(city_name)}_{start_date}_{end_date}_"
        f"{slugify(variable)}.csv"
    )


def build_open_meteo_params(
    latitude: float,
    longitude: float,
    start_date: str,
    end_date: str,
    hourly_variables: list[str],
    timezone: str,
) -> dict:
    """Build the Open-Meteo query parameters in one testable place."""
    return {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": ",".join(hourly_variables),
        "timezone": timezone,
    }


def ensure_output_directories() -> None:
    """Create the folders used by the example."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================================
# 3. DATA ACQUISITION — REAL-WORLD API
# ============================================================================
# Consult Week01_Extra_From_APIs_to_Real_World_Data_Streams.ipynb,
# especially the Open-Meteo section, if this part is unfamiliar.

def download_open_meteo(
    latitude: float = LATITUDE,
    longitude: float = LONGITUDE,
    start_date: str = START_DATE,
    end_date: str = END_DATE,
    hourly_variables: list[str] = HOURLY_VARIABLES,
    timezone: str = TIMEZONE,
) -> dict:
    """Download a historical period from Open-Meteo and return parsed JSON."""
    params = build_open_meteo_params(
        latitude=latitude,
        longitude=longitude,
        start_date=start_date,
        end_date=end_date,
        hourly_variables=hourly_variables,
        timezone=timezone,
    )

    print("\n[1/7] Contacting Open-Meteo Historical Weather API...")
    print(f"      Period: {start_date} -> {end_date}")
    print(f"      Coordinates: ({latitude}, {longitude})")

    try:
        response = requests.get(
            OPEN_METEO_ARCHIVE_URL,
            params=params,
            timeout=30,
        )
        response.raise_for_status()
        payload = response.json()
    except requests.RequestException as exc:
        raise RuntimeError(
            "Could not download Open-Meteo data. Check the internet connection, "
            "or reuse a previously saved raw JSON snapshot with FORCE_DOWNLOAD=False."
        ) from exc
    except ValueError as exc:
        raise RuntimeError("Open-Meteo returned a response that was not valid JSON.") from exc

    if isinstance(payload, dict) and payload.get("error"):
        raise RuntimeError(
            "Open-Meteo reported an API error: "
            + str(payload.get("reason", "unknown reason"))
        )

    return payload


def load_or_download_raw(
    raw_path: Path,
    downloader: Callable[[], dict],
    force_download: bool = False,
) -> dict:
    """Reuse a saved raw response when possible; otherwise download and save it.

    This implements the course rule:

        COLLECT ONCE -> SAVE RAW DATA -> REPLAY MANY TIMES
    """
    raw_path = Path(raw_path)

    if raw_path.exists() and not force_download:
        print(f"\n[1/7] Reusing raw snapshot: {raw_path}")
        with raw_path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    payload = downloader()

    raw_path.parent.mkdir(parents=True, exist_ok=True)
    with raw_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)

    print(f"      Raw response saved to: {raw_path}")
    return payload


# ============================================================================
# 4. PREPARE THE DATA
# ============================================================================
# Here pandas is allowed: we are turning a raw snapshot into a clean,
# timestamped source table. The streaming restriction starts later, inside the
# replay loop.

def prepare_dataframe(payload: dict, variable: str) -> pd.DataFrame:
    """Create a clean chronological DataFrame for one numeric variable."""
    if "hourly" not in payload:
        raise ValueError("The API payload does not contain an 'hourly' section.")

    hourly = payload["hourly"]

    if "time" not in hourly:
        raise ValueError("The hourly payload does not contain timestamps.")

    if variable not in hourly:
        available = ", ".join(k for k in hourly.keys() if k != "time")
        raise ValueError(
            f"Variable {variable!r} is not available. Available hourly variables: "
            f"{available}"
        )

    df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(hourly["time"], errors="coerce"),
            variable: pd.to_numeric(hourly[variable], errors="coerce"),
        }
    )

    before = len(df)
    df = (
        df.dropna(subset=["timestamp", variable])
        .sort_values("timestamp")
        .reset_index(drop=True)
    )
    removed = before - len(df)

    print("\n[2/7] Preparing the data...")
    print(f"      Valid observations: {len(df):,}")
    print(f"      Missing/invalid rows removed: {removed:,}")

    if df.empty:
        raise RuntimeError("No valid observations remain after preprocessing.")

    return df


def save_processed_data(df: pd.DataFrame, output_path: Path) -> Path:
    """Save the cleaned table that will act as the replay source."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return output_path


# ============================================================================
# 5. REPLAY — TURN A TABLE INTO A STREAM
# ============================================================================
# Consult Week01 Extra, Section 4: From API Data to a Replay Stream.

def replay_stream(df: pd.DataFrame, value_col: str) -> Iterator[dict]:
    """Yield one observation at a time, in chronological order.

    Each yielded dictionary is one event. The consumer never receives the
    entire DataFrame at once.
    """
    ordered = df.sort_values("timestamp")

    for _, row in ordered.iterrows():
        yield {
            "timestamp": row["timestamp"],
            "value": float(row[value_col]),
        }


# ============================================================================
# 6. ONLINE STATISTICS — THE STATE OF THE STREAM PROCESSOR
# ============================================================================
# Consult Week02_Foundations_of_Data_Stream_Mining_Lab.ipynb,
# Sections 4 and 5.
class RunningStats:
    """Constant-size online state for basic statistics.

    State kept:
        n        -> number of observations seen
        total    -> running sum
        minimum  -> smallest value seen
        maximum  -> largest value seen
        mean     -> running mean
        M2       -> Welford's sum of squared deviations

    Notice what is NOT stored: the raw history of observations.
    """

    def __init__(self) -> None:
        self.n = 0
        self.total = 0.0
        self.minimum: float | None = None
        self.maximum: float | None = None
        self.mean = 0.0
        self.M2 = 0.0

    def update(self, x: float) -> None:
        """Consume one new observation and update the persistent state."""
        x = float(x)

        # Count and sum.
        self.n += 1
        self.total += x

        # Running min/max.
        if self.minimum is None or x < self.minimum:
            self.minimum = x
        if self.maximum is None or x > self.maximum:
            self.maximum = x

        # Welford's numerically stable online mean/variance update.
        delta = x - self.mean
        self.mean += delta / self.n
        delta2 = x - self.mean
        self.M2 += delta * delta2

    @property
    def variance(self) -> float:
        """Sample variance (ddof=1), or NaN before two observations exist."""
        if self.n < 2:
            return float("nan")
        return self.M2 / (self.n - 1)

    @property
    def std(self) -> float:
        """Sample standard deviation, or NaN before two observations exist."""
        if self.n < 2:
            return float("nan")
        return math.sqrt(self.variance)


# ============================================================================
# 7. THE STREAMING LOOP — THIS IS THE HEART OF THE CHALLENGE
# ============================================================================
def process_stream(events: Iterable[dict]) -> tuple[RunningStats, pd.DataFrame]:
    """Process observations sequentially and record state evolution for plots.

    IMPORTANT
    ---------
    The RunningStats object is the algorithmic state.

    The `history_rows` list below is kept ONLY because the assignment asks us
    to draw how the running statistics evolved. A production streaming system
    could send these points to a dashboard or monitoring store instead.
    """
    stats = RunningStats()
    history_rows: list[dict] = []

    print("\n[4/7] Replaying observations one at a time...")

    for event in events:
        # ------------------------------------------------------------
        # THIS IS THE STREAMING PART:
        # old state + current observation -> new state
        # ------------------------------------------------------------
        stats.update(event["value"])

        # Teaching/visualization history — not required by RunningStats.
        history_rows.append(
            {
                "timestamp": event["timestamp"],
                "value": event["value"],
                "count": stats.n,
                "mean": stats.mean,
                "variance": stats.variance,
                "std": stats.std,
            }
        )

    history = pd.DataFrame(history_rows)

    print(f"      Observations processed: {stats.n:,}")
    print("      Raw observations retained by RunningStats: 0")
    print("      Persistent statistical state: n, sum, min, max, mean, M2")

    return stats, history


# ============================================================================
# 8. VALIDATE ONLINE RESULTS AGAINST BATCH CALCULATIONS
# ============================================================================
def validate_against_batch(
    df: pd.DataFrame,
    stats: RunningStats,
    value_col: str,
) -> dict:
    """Compare the final online state with trusted batch calculations.

    This happens AFTER replay. Batch statistics are validation, not the
    streaming algorithm.
    """
    series = pd.to_numeric(df[value_col], errors="coerce").dropna()

    batch_mean = float(series.mean())
    batch_variance = float(series.var(ddof=1))
    batch_std = float(series.std(ddof=1))

    return {
        "batch_mean": batch_mean,
        "online_mean": stats.mean,
        "mean_difference": stats.mean - batch_mean,
        "batch_variance": batch_variance,
        "online_variance": stats.variance,
        "variance_difference": stats.variance - batch_variance,
        "batch_std": batch_std,
        "online_std": stats.std,
        "std_difference": stats.std - batch_std,
    }


# ============================================================================
# 9. VISUALIZATION
# ============================================================================
def plot_results(
    history: pd.DataFrame,
    figures_dir: Path,
    variable_name: str,
    unit: str,
    city_name: str,
) -> list[Path]:
    """Create the three plots requested by the challenge."""
    import matplotlib.pyplot as plt

    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    label = variable_name.replace("_", " ").title()
    unit_suffix = f" ({unit})" if unit else ""

    # Plot 1 — Original time series.
    raw_path = figures_dir / "01_raw_series.png"
    plt.figure(figsize=(10, 4))
    plt.plot(history["timestamp"], history["value"])
    plt.xlabel("Time")
    plt.ylabel(f"{label}{unit_suffix}")
    plt.title(f"{city_name} — Original time series")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(raw_path, dpi=150)
    plt.close()

    # Plot 2 — Raw observations plus running mean.
    mean_path = figures_dir / "02_running_mean.png"
    plt.figure(figsize=(10, 4))
    plt.plot(
        history["timestamp"],
        history["value"],
        alpha=0.45,
        label="Observation",
    )
    plt.plot(
        history["timestamp"],
        history["mean"],
        label="Running mean",
    )
    plt.xlabel("Time")
    plt.ylabel(f"{label}{unit_suffix}")
    plt.title(f"{city_name} — Raw stream and running mean")
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(mean_path, dpi=150)
    plt.close()

    # Plot 3 — Running standard deviation.
    std_path = figures_dir / "03_running_std.png"
    plt.figure(figsize=(10, 4))
    plt.plot(history["timestamp"], history["std"])
    plt.xlabel("Time")
    plt.ylabel(f"Running standard deviation{unit_suffix}")
    plt.title(f"{city_name} — Running standard deviation")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(std_path, dpi=150)
    plt.close()

    return [raw_path, mean_path, std_path]


# ============================================================================
# 10. SCIENTIFIC SUMMARY
# ============================================================================
def print_scientific_summary(
    stats: RunningStats,
    validation: dict,
    variable: str,
    unit: str,
    df: pd.DataFrame,
    processed_path: Path,
    figure_paths: list[Path],
) -> None:
    """Print a compact interpretation of what the script just did."""
    suffix = f" {unit}" if unit else ""

    print("\n" + "=" * 72)
    print("WEEKEND CHALLENGE 01 — FINAL SUMMARY")
    print("=" * 72)
    print(f"Variable:               {variable}")
    print(f"First timestamp:        {df['timestamp'].iloc[0]}")
    print(f"Last timestamp:         {df['timestamp'].iloc[-1]}")
    print(f"Observations processed: {stats.n:,}")
    print()
    print(f"Online sum:             {stats.total:.6f}{suffix}")
    print(f"Online minimum:         {stats.minimum:.6f}{suffix}")
    print(f"Online maximum:         {stats.maximum:.6f}{suffix}")
    print(f"Online mean:            {stats.mean:.6f}{suffix}")
    print(f"Online variance:        {stats.variance:.6f}")
    print(f"Online std:             {stats.std:.6f}{suffix}")
    print()
    print("Validation against batch calculations")
    print(f"Mean difference:        {validation['mean_difference']:.12g}")
    print(f"Variance difference:    {validation['variance_difference']:.12g}")
    print(f"Std difference:         {validation['std_difference']:.12g}")
    print()
    print("Streaming interpretation")
    print("- RunningStats stored no raw observations.")
    print("- Its statistical state remained constant in size as the stream grew.")
    print("- The DataFrame was used as the replay source, not as the statistics engine.")
    print("- Batch calculations were used only after replay to validate the result.")
    print()
    print(f"Processed CSV: {processed_path}")
    print("Figures:")
    for path in figure_paths:
        print(f"  - {path}")
    print("=" * 72)


# ============================================================================
# 11. MAIN PROGRAM
# ============================================================================
def main() -> None:
    """Run the complete worked example from acquisition to validation."""
    print("=" * 72)
    print("WEEKEND CHALLENGE 01 — REAL-WORLD REPLAY STREAM")
    print("Instructor solution / worked example")
    print("=" * 72)

    if STREAM_VARIABLE not in HOURLY_VARIABLES:
        raise ValueError(
            f"STREAM_VARIABLE={STREAM_VARIABLE!r} is not in HOURLY_VARIABLES. "
            "Add it to the list before running the script."
        )

    ensure_output_directories()

    raw_path = RAW_DIR / build_raw_filename(CITY_NAME, START_DATE, END_DATE)

    payload = load_or_download_raw(
        raw_path=raw_path,
        downloader=lambda: download_open_meteo(
            latitude=LATITUDE,
            longitude=LONGITUDE,
            start_date=START_DATE,
            end_date=END_DATE,
            hourly_variables=HOURLY_VARIABLES,
            timezone=TIMEZONE,
        ),
        force_download=FORCE_DOWNLOAD,
    )

    df = prepare_dataframe(payload, STREAM_VARIABLE)

    processed_path = PROCESSED_DIR / build_processed_filename(
        CITY_NAME,
        START_DATE,
        END_DATE,
        STREAM_VARIABLE,
    )
    save_processed_data(df, processed_path)
    print(f"\n[3/7] Clean replay source saved to: {processed_path}")

    print("\n      First observations:")
    print(df.head().to_string(index=False))

    # --------------------------------------------------------------------
    # STREAM STARTS HERE.
    # From this point onward, do not use df.mean(), df.var(), or df.describe()
    # to maintain the statistics. Pretend each past event is gone after it is
    # processed.
    # --------------------------------------------------------------------
    events = replay_stream(df, value_col=STREAM_VARIABLE)
    stats, history = process_stream(events)

    print("\n[5/7] Validating online results against batch calculations...")
    validation = validate_against_batch(df, stats, STREAM_VARIABLE)

    unit = payload.get("hourly_units", {}).get(STREAM_VARIABLE, "")

    print("\n[6/7] Creating plots...")
    figure_paths = plot_results(
        history=history,
        figures_dir=FIGURES_DIR,
        variable_name=STREAM_VARIABLE,
        unit=unit,
        city_name=CITY_NAME,
    )

    print("\n[7/7] Printing scientific summary...")
    print_scientific_summary(
        stats=stats,
        validation=validation,
        variable=STREAM_VARIABLE,
        unit=unit,
        df=df,
        processed_path=processed_path,
        figure_paths=figure_paths,
    )


# ============================================================================
# 12. IDEAS TO TRY AFTER THE OFFICIAL SOLUTION
# ============================================================================
# 1. Change STREAM_VARIABLE to precipitation or wind_speed_10m.
# 2. Change Lisbon to Porto, Faro, Funchal, or another location.
# 3. Change the historical period.
# 4. Replace Open-Meteo acquisition with IPMA, REN, or ThingSpeak while
#    keeping prepare -> replay -> state -> plots -> validation conceptually the
#    same.
# 5. In Week 3, replace global statistics with window-based statistics and ask:
#       "Should observations from long ago still have the same importance?"
#
# Remember:
#       THE STREAM DISAPPEARS. THE STATE REMAINS.


if __name__ == "__main__":
    main()
