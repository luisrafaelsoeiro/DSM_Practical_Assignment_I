
import json

import numpy as np
import pandas as pd
import requests

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from river import stats

from config import (
    PHYSICAL_BOUNDS,
    PROCESSED_DIR,
    RAW_DIR,
    TIMEZONE,
    WEATHER_COLUMNS,
    HOT_PERCENTILE,
    WARMUP_PERCENTAGE,
    MIN_CONSECUTIVE_HOT_DAYS
)



def fetch_open_meteo_data(
    latitude: float,
    longitude: float,
    start_date: str,
    end_date: str,
    city_name: str,
    force_download: bool = False,
) -> pd.DataFrame:
    """Load cached weather data or download and save it if unavailable."""

    raw_path = RAW_DIR / f"{city_name.lower()}.json"
    processed_path = PROCESSED_DIR / f"{city_name.lower()}.csv"

    # Reuse existing processed data to avoid unnecessary downloads
    if processed_path.exists() and not force_download:
        print(f"{city_name}: loading cached data from {processed_path}")
        df = pd.read_csv(processed_path, parse_dates=["time"])
        return df

    # Download data only if the cache is missing or a refresh is requested
    url = "https://archive-api.open-meteo.com/v1/archive"

    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": [
            "temperature_2m",
            "relative_humidity_2m",
            "precipitation",
            "wind_speed_10m",
        ],
        "timezone": TIMEZONE,
    }

    response = requests.get(url, params=params, timeout=60)
    response.raise_for_status()
    data = response.json()

    if "hourly" not in data or "time" not in data["hourly"]:
        raise ValueError(
            f"Invalid hourly API response for {city_name}."
        )

    # Save the raw API response
    with raw_path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)

    # Create and standardize the DataFrame
    df = pd.DataFrame(data["hourly"])

    df = df.rename(
        columns={
            "temperature_2m": "temperature",
            "relative_humidity_2m": "humidity",
            "wind_speed_10m": "wind_speed",
        }
    )

    df["time"] = pd.to_datetime(df["time"])
    df["city"] = city_name

    df = df.sort_values("time").reset_index(drop=True)

    # Save processed data
    df.to_csv(processed_path, index=False)

    print(
        f"{city_name}: downloaded {len(df):,} observations "
        f"and saved to {processed_path}"
    )

    return df


def audit_weather_data(
    df: pd.DataFrame,
    city_name: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Audit weather data for quality issues.

    Checks duplicate timestamps, physically implausible values,
    missing weather values, and missing hourly timestamps.

    Returns:
        result: Data with duplicate timestamps removed and invalid
                physical values replaced with NaN.
        audit: A DataFrame summarizing detected issues.
    """

    result = df.copy()
    audit_records = []

    # Ensure timestamps are datetime objects
    result["time"] = pd.to_datetime(result["time"])

    # 1. Detect and remove duplicate timestamps
    duplicate_count = int(result["time"].duplicated().sum())

    if duplicate_count:
        audit_records.append({
            "city": city_name,
            "issue": "Duplicate timestamps",
            "variable": "time",
            "count": duplicate_count,
        })

        result = result.drop_duplicates(
            subset="time",
            keep="first",
        ).reset_index(drop=True)

    # 2. Detect physically implausible values
    for column, (lower, upper) in PHYSICAL_BOUNDS.items():
        invalid = (
            (result[column] < lower)
            | (result[column] > upper)
        ).fillna(False)

        invalid_count = int(invalid.sum())

        if invalid_count:
            audit_records.append({
                "city": city_name,
                "issue": (
                    f"Outside physical bounds [{lower}, {upper}]"
                ),
                "variable": column,
                "count": invalid_count,
            })

            # Mark invalid observations as missing
            result.loc[invalid, column] = np.nan

    # 3. Count missing values in each weather variable
    for column in WEATHER_COLUMNS:
        missing_count = int(result[column].isna().sum())

        if missing_count:
            audit_records.append({
                "city": city_name,
                "issue": "Missing values",
                "variable": column,
                "count": missing_count,
            })

    # 4. Detect missing hourly timestamps
    if not result.empty:
        expected_times = pd.date_range(
            start=result["time"].min(),
            end=result["time"].max(),
            freq="h",
        )

        actual_times = pd.DatetimeIndex(result["time"])

        missing_times = expected_times.difference(actual_times)
        missing_time_count = len(missing_times)
    else:
        missing_time_count = 0

    audit_records.append({
        "city": city_name,
        "issue": "Missing hourly timestamps",
        "variable": "time",
        "count": missing_time_count,
    })

    audit = pd.DataFrame(
        audit_records,
        columns=["city", "issue", "variable", "count"],
    )

    return result, audit


# Dictionary for storing downloaded weather DataFrames
weather_data = {}


def get_model_ready_data(
    df: pd.DataFrame,
    city_name: str = "Unknown",
) -> pd.DataFrame:
    """
    Prepare weather data for stream mining.

    Checks timestamps and imputes missing weather values using
    past observations only.

    Raises:
        ValueError: If timestamps are missing, duplicated,
                    or not strictly increasing.
        ValueError: If missing weather values cannot be imputed
                    without using future observations.
    """

    result = df.copy()

    # --------------------------------------------------
    # 1. Validate timestamps
    # --------------------------------------------------

    # Timestamps must exist and be valid.
    if result["time"].isna().any():
        count = int(result["time"].isna().sum())
        raise ValueError(
            f"{city_name}: Found {count} missing timestamps."
        )

    # Duplicate timestamps make the stream ambiguous.
    duplicate_mask = result["time"].duplicated(keep=False)

    if duplicate_mask.any():
        examples = result.loc[duplicate_mask, "time"].head(5).tolist()

        raise ValueError(
            f"{city_name}: Found duplicated timestamps. "
            f"Examples: {examples}"
        )

    # Do not sort: the input must already be chronological.
    # A non-increasing timestamp indicates an invalid sequence.
    time_diff = result["time"].diff()

    out_of_order = time_diff <= pd.Timedelta(0)

    if out_of_order.any():
        positions = result.index[out_of_order].tolist()[:5]

        raise ValueError(
            f"{city_name}: Timestamps are not strictly increasing. "
            f"Problematic row indices: {positions}"
        )

    # --------------------------------------------------
    # 2. Impute missing weather values using past data
    # --------------------------------------------------

    missing_before = result[WEATHER_COLUMNS].isna().sum()

    # Forward-fill: use the last previously observed value.
    result[WEATHER_COLUMNS] = result[WEATHER_COLUMNS].ffill()

    missing_after = result[WEATHER_COLUMNS].isna().sum()

    # Forward-fill cannot impute missing values before the
    # first valid observation of a variable.
    remaining = missing_after[missing_after > 0]

    if not remaining.empty:
        raise ValueError(
            f"{city_name}: Cannot impute all missing values "
            f"without future data. Remaining missing counts: "
            f"{remaining.to_dict()}"
        )

    # --------------------------------------------------
    # 3. Report the changes
    # --------------------------------------------------

    total_imputed = int(
        (missing_before - missing_after).sum()
    )

    print(f"\n{city_name}: Model-ready data")
    print(f"Rows retained: {len(result):,}")
    print(f"Missing values imputed: {total_imputed}")

    for column in WEATHER_COLUMNS:
        count = int(missing_before[column] - missing_after[column])

        if count > 0:
            print(f"  {column}: {count} value(s) imputed")

    return result



def iter_daily_maxima(df: pd.DataFrame):
    """
    Yield one daily maximum temperature at a time.

    Missing temperatures are ignored. Days without any valid
    temperature observations are not yielded.
    """

    if df.empty:
        return

    current_date = None
    current_max = None

    for row in df.sort_values("time").itertuples(index=False):
        timestamp = pd.Timestamp(row.time)
        day = timestamp.normalize()

        temperature = row.temperature

        # A new day means the previous day's observations have ended.
        if current_date is not None and day != current_date:
            if current_max is not None:
                yield current_date, current_max

            current_date = day
            current_max = None

        elif current_date is None:
            current_date = day

        # Ignore missing or non-finite temperatures.
        if pd.isna(temperature) or not np.isfinite(temperature):
            continue

        temperature = float(temperature)

        if current_max is None:
            current_max = temperature
        else:
            current_max = max(current_max, temperature)

    # Yield the final day's maximum if it has valid observations.
    # This assumes the final day's observation period is complete.
    if current_date is not None and current_max is not None:
        yield current_date, current_max


def get_warmup_end(df: pd.DataFrame) -> pd.Timestamp:
    """
    Find the end date of the initial warm-up portion.

    The warm-up consists of WARMUP_PERCENTAGE of the distinct
    calendar dates present in the DataFrame.
    """

    if df.empty:
        raise ValueError("Cannot determine the warm-up period from empty data.")

    timestamps = pd.to_datetime(df["time"])

    dates = pd.DatetimeIndex(
        timestamps.dt.normalize().drop_duplicates()
    ).sort_values()

    if len(dates) == 0:
        raise ValueError("No valid timestamps are available.")

    if not 0 < WARMUP_PERCENTAGE <= 1:
        raise ValueError("WARMUP_PERCENTAGE must be in (0, 1].")

    warmup_days = max(
        1,
        int(np.ceil(len(dates) * WARMUP_PERCENTAGE)),
    )

    return dates[warmup_days - 1]


def estimate_threshold_incrementally(
    city_data: dict[str, pd.DataFrame],
    warmup_ends: dict[str, pd.Timestamp],
) -> tuple[float, int]:
    """
    Estimate a pooled hot-day threshold using an online quantile.

    Each valid daily maximum from each city's warm-up period
    updates the same quantile estimator.

    Returns:
        threshold: Estimated HOT_PERCENTILE quantile in °C.
        observations_seen: Number of daily maxima processed.
    """

    quantile = stats.Quantile(q=HOT_PERCENTILE)
    observations_seen = 0

    for city, df in city_data.items():
        if city not in warmup_ends:
            raise ValueError(f"Missing warm-up end date for {city}.")

        if df.empty:
            continue

        timestamps = pd.to_datetime(df["time"])
        warmup_end = warmup_ends[city]

        # Only observations within this city's warm-up period.
        warmup_df = df.loc[
            timestamps.dt.normalize() <= warmup_end
        ]

        # Update the online quantile once per daily maximum.
        for _, daily_max in iter_daily_maxima(warmup_df):
            quantile.update(daily_max)
            observations_seen += 1

    if observations_seen == 0:
        raise ValueError(
            "The warm-up period contains no valid daily maxima."
        )

    return float(quantile.get()), observations_seen


def estimate_threshold_daily_max(
    city_data: dict[str, pd.DataFrame],
    warmup_ends: dict[str, pd.Timestamp],
) -> tuple[float, int]:
    """
    Estimate the pooled warm-up percentile of daily maxima.

    Uses River's online quantile estimator.
    """

    quantile = stats.Quantile(q=HOT_PERCENTILE)
    observations_seen = 0

    for city, df in city_data.items():
        if city not in warmup_ends:
            raise ValueError(f"Missing warm-up end date for {city}.")

        if df.empty:
            continue

        timestamps = pd.to_datetime(df["time"])
        warmup_end = warmup_ends[city]

        warmup_df = df.loc[
            timestamps.dt.normalize() <= warmup_end
        ]

        for _, daily_max in iter_daily_maxima(warmup_df):
            quantile.update(daily_max)
            observations_seen += 1

    if observations_seen == 0:
        raise ValueError(
            "The warm-up period contains no valid daily maxima."
        )

    return float(quantile.get()), observations_seen


def estimate_threshold_hourly(
    city_data: dict[str, pd.DataFrame],
    warmup_ends: dict[str, pd.Timestamp],
) -> tuple[float, int]:
    """
    Estimate the pooled warm-up percentile of hourly temperatures.

    Every valid hourly temperature updates River's online estimator.
    """

    quantile = stats.Quantile(q=HOT_PERCENTILE)
    observations_seen = 0

    for city, df in city_data.items():
        if city not in warmup_ends:
            raise ValueError(f"Missing warm-up end date for {city}.")

        if df.empty:
            continue

        timestamps = pd.to_datetime(df["time"])
        warmup_end = warmup_ends[city]

        warmup_df = df.loc[
            timestamps.dt.normalize() <= warmup_end
        ].sort_values("time")

        # Ignore missing and non-finite temperatures.
        temperatures = pd.to_numeric(
            warmup_df["temperature"],
            errors="coerce",
        )

        temperatures = temperatures[
            np.isfinite(temperatures)
        ]

        for temperature in temperatures:
            quantile.update(float(temperature))
            observations_seen += 1

    if observations_seen == 0:
        raise ValueError(
            "The warm-up period contains no valid hourly temperatures."
        )

    return float(quantile.get()), observations_seen


def get_pooled_hourly_warmup(
    city_data: dict[str, pd.DataFrame],
    warmup_ends: dict[str, pd.Timestamp],
) -> np.ndarray:
    """
    Collect all valid hourly warm-up temperatures across cities.

    Used to calculate an exact NumPy quantile for comparison
    with River's online estimate.
    """

    temperature_arrays = []

    for city, df in city_data.items():
        if city not in warmup_ends:
            raise ValueError(f"Missing warm-up end date for {city}.")

        if df.empty:
            continue

        timestamps = pd.to_datetime(df["time"])

        warmup_df = df.loc[
            timestamps.dt.normalize() <= warmup_ends[city]
        ]

        temperatures = pd.to_numeric(
            warmup_df["temperature"],
            errors="coerce",
        ).to_numpy(dtype=float)

        temperatures = temperatures[np.isfinite(temperatures)]

        if len(temperatures) > 0:
            temperature_arrays.append(temperatures)

    if not temperature_arrays:
        raise ValueError(
            "No valid hourly warm-up temperatures are available."
        )

    return np.concatenate(temperature_arrays)

class OnlineHeatwaveDetector:
    """Stateful detector for consecutive hot days in an hourly stream."""

    def __init__(
        self,
        threshold: float,
        min_consecutive_days: int = 3,
    ):
        self.threshold = threshold
        self.min_consecutive_days = min_consecutive_days

        self.current_date = None
        self.current_day_max = None
        self.previous_completed_date = None
        self.consecutive_hot_days = 0
        self.heatwave_active = False

    def _finish_day(self):
        """Finalize the current day when the next day begins."""

        completed_date = self.current_date
        completed_max = self.current_day_max

        if self.previous_completed_date is not None:
            day_gap = (
                completed_date - self.previous_completed_date
            ).days

            if day_gap != 1:
                # a missing calendar day breaks the consecutive run
                self.consecutive_hot_days = 0
                self.heatwave_active = False

        hot_day = completed_max >= self.threshold

        if hot_day:
            self.consecutive_hot_days += 1
        else:
            self.consecutive_hot_days = 0
            self.heatwave_active = False

        if self.consecutive_hot_days >= self.min_consecutive_days:
            self.heatwave_active = True

        self.previous_completed_date = completed_date

        return completed_date, completed_max, hot_day

    def update(self, timestamp, temperature):
        """Consume one observation and return the current detector state."""

        timestamp = pd.Timestamp(timestamp)
        day = timestamp.normalize()
        temperature = float(temperature)

        completed_day = None

        if self.current_date is None:
            self.current_date = day
            self.current_day_max = temperature

        elif day == self.current_date:
            self.current_day_max = max(
                self.current_day_max,
                temperature,
            )

        else:
            completed_day = self._finish_day()

            self.current_date = day
            self.current_day_max = temperature

        return {
            "time": timestamp,
            "daily_max_so_far": self.current_day_max,
            "completed_day": (
                completed_day[0] if completed_day else pd.NaT
            ),
            "completed_day_max": (
                completed_day[1] if completed_day else np.nan
            ),
            "hot_day_confirmed": (
                completed_day[2] if completed_day else None
            ),
            "heatwave_active": int(self.heatwave_active),
        }


def label_stream_online(
    df: pd.DataFrame,
    threshold: float,
    warmup_end: pd.Timestamp,
) -> pd.DataFrame:
    """
    Replay observations after warm-up and generate causal alert labels.

    The detector begins with no prior hot-day history. It does not
    backdate an alert to days before the event was confirmed.
    """

    stream = df.loc[
        df["time"].dt.normalize() > warmup_end
    ].copy()

    stream = stream.sort_values("time").reset_index(drop=True)

    detector = OnlineHeatwaveDetector(
        threshold=threshold,
        min_consecutive_days=MIN_CONSECUTIVE_HOT_DAYS,
    )

    records = []

    for row in stream.itertuples(index=False):
        records.append(
            detector.update(row.time, row.temperature)
        )

    labels = pd.DataFrame(records)

    result = stream.merge(
        labels,
        on="time",
        how="left",
        validate="one_to_one",
    )

    result["target"] = result["heatwave_active"].astype(int)

    return result

def label_stream_hourly(
    df: pd.DataFrame,
    threshold: float,
    warmup_end: pd.Timestamp,
) -> pd.DataFrame:
    """
    Label each hourly observation according to whether its
    temperature reaches the fixed hot-temperature threshold.

    Observations from the warm-up period are excluded.
    target = 1 if temperature >= threshold, otherwise 0.
    """

    # Keep only observations after the warm-up period.
    stream = df.loc[
        df["time"].dt.normalize() > warmup_end
    ].copy()

    # Process observations in chronological order.
    stream = stream.sort_values("time").reset_index(drop=True)

    # Assign the hourly heat-stress label.
    # Missing temperatures remain unlabelled rather than being
    # incorrectly classified as non-hot observations.
    stream["target"] = np.where(
        stream["temperature"].isna(),
        np.nan,
        (stream["temperature"] >= threshold).astype(int),
    )

    return stream


def _episodes(target: pd.Series) -> pd.Series:
    """Duration (in days) of each run of consecutive positive labels."""
    a = target.astype(int)
    run_id = (a != a.shift()).cumsum()
    lengths = a.groupby(run_id).agg(["first", "size"])
    return lengths.loc[lengths["first"] == 1, "size"] / 24.0   # hourly data


TARGET_PALETTE = ["#2a9d8f", "#e76f51", "#264653", "#e9c46a", "#8d6bb8", "#5b7fa6"]
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
          "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

def plot_target_distribution(
    labeled_data: dict,
    warmup_end=None,
    target_col: str = "target",
    smooth_days: int = 30,
    show: bool = True,
):
    """
    Target class distribution for every city in one figure.

      1. Class balance (share of hours, counts on the bars)
      2. Positive rate by month of the year  -> seasonality of the class prior
      3. Positive rate over time (smoothed)  -> class-prior drift in the stream
      4. Duration of positive episodes (days)
      5. Positive rate per year

    Returns the figure and a summary DataFrame.
    """
    fig = make_subplots(
        rows=3, cols=2,
        specs=[[{}, {}],
               [{"colspan": 2}, None],
               [{}, {}]],
        row_heights=[0.30, 0.38, 0.32],
        vertical_spacing=0.12,
        horizontal_spacing=0.09,
        subplot_titles=(
            "Class balance",
            "Positive rate by month of the year",
            f"Positive rate over time ({smooth_days}-day moving average of daily share)",
            "Duration of positive episodes",
            "Positive rate per year",
        ),
    )

    summary = []
    prepared = {}

    for k, (city, df) in enumerate(labeled_data.items()):
        d = df[["time", target_col]].copy()
        d["time"] = pd.to_datetime(d["time"])
        d = d.dropna(subset=[target_col]).sort_values("time")
        d[target_col] = d[target_col].astype(int)
        prepared[city] = d

        color = TARGET_PALETTE[k % len(TARGET_PALETTE)]
        s = d.set_index("time")[target_col]
        n = len(s)
        n_pos = int(s.sum())
        n_neg = n - n_pos
        pct_pos = 100 * n_pos / n

        # ---- 1. class balance --------------------------------------
        fig.add_trace(
            go.Bar(
                x=["Negative (0)", "Positive (1)"],
                y=[100 * n_neg / n, pct_pos],
                text=[f"{n_neg:,}<br>({100 * n_neg / n:.1f}%)",
                      f"{n_pos:,}<br>({pct_pos:.1f}%)"],
                textposition="outside",
                marker_color=color,
                name=city, legendgroup=city,
                hovertemplate="%{x}: %{y:.1f}% of hours<extra>" + city + "</extra>",
            ),
            row=1, col=1,
        )

        # ---- 2. positive rate by month of the year -----------------
        by_month = s.groupby(s.index.month).mean() * 100
        by_month = by_month.reindex(range(1, 13))
        fig.add_trace(
            go.Bar(
                x=MONTHS, y=by_month.values,
                marker_color=color,
                name=city, legendgroup=city, showlegend=False,
                hovertemplate="%{x}: %{y:.1f}% positive<extra>" + city + "</extra>",
            ),
            row=1, col=2,
        )

        # ---- 3. positive rate over time ----------------------------
        daily = s.resample("D").mean() * 100
        smooth = daily.rolling(smooth_days, min_periods=max(1, smooth_days // 3),
                               center=True).mean()
        fig.add_trace(
            go.Scatter(
                x=daily.index, y=smooth.values, mode="lines",
                line=dict(color=color, width=2.4),
                name=city, legendgroup=city, showlegend=False,
                hovertemplate="%{y:.1f}% positive<extra>" + city + "</extra>",
            ),
            row=2, col=1,
        )

        # ---- 4. episode durations ----------------------------------
        ep = _episodes(s)
        if len(ep):
            fig.add_trace(
                go.Histogram(
                    x=ep.values, marker_color=color, opacity=0.65,
                    name=city, legendgroup=city, showlegend=False,
                    xbins=dict(size=1),
                    hovertemplate="%{x:.0f}-day episodes: %{y}<extra>" + city + "</extra>",
                ),
                row=3, col=1,
            )

        # ---- 5. positive rate per year -----------------------------
        by_year = s.groupby(s.index.year).mean() * 100
        fig.add_trace(
            go.Bar(
                x=[str(y) for y in by_year.index], y=by_year.values,
                text=[f"{v:.1f}%" for v in by_year.values],
                textposition="outside",
                marker_color=color,
                name=city, legendgroup=city, showlegend=False,
                hovertemplate="%{x}: %{y:.1f}% positive<extra>" + city + "</extra>",
            ),
            row=3, col=2,
        )

        summary.append({
            "city": city,
            "hours": n,
            "negative": n_neg,
            "positive": n_pos,
            "positive_%": round(pct_pos, 2),
            "imbalance (neg:pos)": round(n_neg / n_pos, 1) if n_pos else np.inf,
            "episodes": len(ep),
            "median_episode_days": round(float(ep.median()), 1) if len(ep) else np.nan,
            "max_episode_days": round(float(ep.max()), 1) if len(ep) else np.nan,
        })

    summary = pd.DataFrame(summary).set_index("city")

    # ------------------------------------------------------------------
    # Warm-up shading on the time panel
    # ------------------------------------------------------------------
    if warmup_end is not None:
        first = min(d["time"].iloc[0] for d in prepared.values())
        w = pd.Timestamp(warmup_end)
        tz = prepared[next(iter(prepared))]["time"].dt.tz
        if tz is not None and w.tzinfo is None:
            w = w.tz_localize(tz)
        fig.add_vrect(x0=first, x1=w, fillcolor="rgba(120,120,120,0.14)",
                      line_width=0, layer="below", row=2, col=1)
        xax = fig.get_subplot(2, 1).xaxis.plotly_name.replace("xaxis", "x")
        yax = fig.get_subplot(2, 1).yaxis.plotly_name.replace("yaxis", "y")
        fig.add_annotation(
            x=first + (w - first) / 2, y=1, xref=xax, yref=f"{yax} domain",
            text="Warm-up", showarrow=False, yanchor="top",
            font=dict(size=11, color="grey"),
        )

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------
    ratios = " · ".join(
        f"{c}: 1 positive per {r:.1f} negatives"
        for c, r in summary["imbalance (neg:pos)"].items()
    )
    fig.update_layout(
        template="plotly_white",
        width=1250, height=1000,
        barmode="group",
        title=dict(
            text=f"<b>Target Class Distribution</b><br><sup>{ratios}</sup>",
            x=0.02,
        ),
        legend=dict(orientation="h", x=1, xanchor="right", y=1.04,
                    yanchor="bottom"),
        font=dict(family="Inter, Arial, sans-serif", size=12),
        margin=dict(l=70, r=40, t=120, b=50),
        hovermode="closest",
    )
    fig.update_yaxes(title_text="% of hours", range=[0, 112], row=1, col=1)
    fig.update_yaxes(title_text="% positive", row=1, col=2)
    fig.update_yaxes(title_text="% positive", row=2, col=1)
    fig.update_yaxes(title_text="Episodes", row=3, col=1)
    fig.update_yaxes(title_text="% positive", row=3, col=2)
    fig.update_xaxes(title_text="Episode length (days)", row=3, col=1)

    # a little headroom for the text on the per-year bars
    ymax = max(
        (np.nanmax(t.y) for t in fig.data
         if isinstance(t, go.Bar) and t.xaxis == "x5" and len(t.y)),
        default=None,
    )
    if ymax:
        fig.update_yaxes(range=[0, ymax * 1.25], row=3, col=2)

    for ann in fig.layout.annotations:           # subplot titles, left-aligned
        if ann.text and ann.text != "Warm-up":
            ann.update(x=ann.x, xanchor="center", font=dict(size=13))

    if show:
        fig.show()
        print(summary.to_string())

    return fig, summary


# Colour palette
C_TEMP = "#5b7fa6"        # hourly temperature
C_DAILY = "#1f3b5c"       # daily maximum (hidden by default, click in legend)
C_TAU = "#7b2cbf"         # frozen threshold  (purple, distinct from the rest)
C_ALERT = "#d62728"       # heat-wave alert active
C_EXCESS = "rgba(255, 159, 28, 0.35)"   # area above the threshold
C_WARMUP = "rgba(120, 120, 120, 0.12)"


def plot_heatwave_detection(
    df: pd.DataFrame,
    city: str,
    threshold: float = None,
    warmup_end=None,
    temperature_col: str = "temperature",
    target_col: str = "target",
    show: bool = True,
):
    """
    Online heat-wave detection overview for one city.

    Top panel
      - hourly temperature (+ optional daily maximum, toggle in the legend)
      - frozen threshold as a purple dashed line
      - orange area where the temperature exceeds the threshold
      - red line + light red bands wherever the heat-wave alert is active
      - grey band for the warm-up period (threshold learning)
    Bottom panel
      - share of hours per month with the alert active, i.e. where the
        cooling effort concentrates
    """
    if threshold is None:
        threshold = TAU_T_HOT  # global defined earlier in the notebook

    data = df[["time", temperature_col, target_col]].copy()
    data["time"] = pd.to_datetime(data["time"])
    data = data.sort_values("time").reset_index(drop=True)

    time = data["time"]
    temp = data[temperature_col]
    alert = data[target_col].fillna(0).astype(int).eq(1)

    # ------------------------------------------------------------------
    # Summary statistics (shown in the subtitle)
    # ------------------------------------------------------------------
    a = alert.astype(int)
    starts = (a.diff().fillna(a) == 1)
    ends = (a.diff(-1).fillna(a) == 1)
    n_episodes = int(starts.sum())
    pct_alert = 100 * alert.mean()
    pct_above = 100 * (temp >= threshold).mean()

    # ------------------------------------------------------------------
    # Figure: main panel + monthly alert share
    # ------------------------------------------------------------------
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.76, 0.24],
        vertical_spacing=0.06,
    )

    # Area above the threshold (two traces: baseline + clipped temperature)
    fig.add_trace(
        go.Scatter(
            x=time,
            y=np.full(len(time), threshold),
            mode="lines",
            line=dict(width=0),
            hoverinfo="skip",
            showlegend=False,
        ),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=time,
            y=np.maximum(temp.values, threshold),
            mode="lines",
            line=dict(width=0),
            fill="tonexty",
            fillcolor=C_EXCESS,
            name="Above threshold",
            hoverinfo="skip",
        ),
        row=1, col=1,
    )

    # Hourly temperature
    fig.add_trace(
        go.Scatter(
            x=time,
            y=temp,
            mode="lines",
            line=dict(color=C_TEMP, width=1),
            name="Hourly temperature",
            hovertemplate="%{y:.1f} °C",
        ),
        row=1, col=1,
    )

    # Daily maximum (hidden by default)
    daily_max = data.set_index("time")[temperature_col].resample("D").max()
    fig.add_trace(
        go.Scatter(
            x=daily_max.index + pd.Timedelta(hours=12),
            y=daily_max.values,
            mode="lines",
            line=dict(color=C_DAILY, width=1.6),
            name="Daily maximum",
            visible="legendonly",
            hovertemplate="%{y:.1f} °C",
        ),
        row=1, col=1,
    )

    # Alert segments on top of the temperature line
    fig.add_trace(
        go.Scatter(
            x=time,
            y=temp.where(alert),
            mode="lines",
            line=dict(color=C_ALERT, width=2),
            name="Heat-wave alert active",
            connectgaps=False,
            hovertemplate="%{y:.1f} °C (alert)",
        ),
        row=1, col=1,
    )

    # Frozen threshold: purple dashed line, clearly different from the rest
    fig.add_trace(
        go.Scatter(
            x=[time.iloc[0], time.iloc[-1]],
            y=[threshold, threshold],
            mode="lines",
            line=dict(color=C_TAU, width=2.2, dash="dash"),
            name=f"Frozen threshold ({threshold:.1f} °C)",
            hovertemplate="τ = %{y:.1f} °C",
        ),
        row=1, col=1,
    )

    # ------------------------------------------------------------------
    # Background bands: alert episodes and warm-up
    # ------------------------------------------------------------------
    shapes = []
    ep_start = time[starts].tolist()
    ep_end = time[ends].tolist()
    if len(ep_start) <= 200:
        for x0, x1 in zip(ep_start, ep_end):
            shapes.append(dict(
                type="rect", xref="x", yref="y domain",
                x0=x0, x1=x1 + pd.Timedelta(hours=1), y0=0, y1=1,
                fillcolor="rgba(214, 39, 40, 0.10)",
                line=dict(width=0), layer="below",
            ))

    if warmup_end is not None:
        warmup_end = pd.Timestamp(warmup_end)
        if time.dt.tz is not None and warmup_end.tzinfo is None:
            warmup_end = warmup_end.tz_localize(time.dt.tz)
        shapes.append(dict(
            type="rect", xref="x", yref="y domain",
            x0=time.iloc[0], x1=warmup_end, y0=0, y1=1,
            fillcolor=C_WARMUP, line=dict(width=0), layer="below",
        ))
        shapes.append(dict(
            type="line", xref="x", yref="y domain",
            x0=warmup_end, x1=warmup_end, y0=0, y1=1,
            line=dict(color="grey", width=1, dash="dot"),
        ))
        fig.add_annotation(
            x=time.iloc[0] + (warmup_end - time.iloc[0]) / 2,
            y=1, xref="x", yref="y domain",
            text="Warm-up<br>(threshold learning)",
            showarrow=False, yanchor="top",
            font=dict(size=11, color="grey"),
        )

    fig.update_layout(shapes=shapes)

    # Threshold label at the right edge
    fig.add_annotation(
        x=time.iloc[-1], y=threshold, xref="x", yref="y",
        text=f"<b>τ = {threshold:.1f} °C</b>",
        showarrow=False, xanchor="right", yanchor="bottom", yshift=3,
        font=dict(size=12, color=C_TAU),
    )

    # ------------------------------------------------------------------
    # Bottom panel: monthly share of hours with the alert active
    # ------------------------------------------------------------------
    monthly = (
        data.assign(alert=alert.astype(float))
        .set_index("time")["alert"]
        .resample("MS")
        .mean()
        * 100
    )
    fig.add_trace(
        go.Bar(
            x=monthly.index,
            y=monthly.values,
            xperiod="M1",
            xperiodalignment="middle",
            marker=dict(color=C_ALERT, opacity=0.85),
            name="Alert hours per month",
            hovertemplate="%{y:.0f}% of hours in alert",
            showlegend=False,
        ),
        row=2, col=1,
    )

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------
    fig.update_layout(
        template="plotly_white",
        width=1250,
        height=640,
        hovermode="x unified",
        title=dict(
            text=(
                f"<b>Online Heat-Wave Detection — {city}</b>"
                f"<br><sup>{n_episodes} alert episode(s) · "
                f"{pct_alert:.1f}% of hours in alert · "
                f"{pct_above:.1f}% of hours above the threshold</sup>"
            ),
            x=0.02,
        ),
        margin=dict(l=70, r=30, t=100, b=50),
        legend=dict(
            orientation="h", x=0, y=1.0, yanchor="bottom",
            bgcolor="rgba(0,0,0,0)",
        ),
        font=dict(family="Inter, Arial, sans-serif", size=12),
    )

    fig.update_xaxes(
        showgrid=True, gridcolor="rgba(0,0,0,0.06)",
        rangeselector=dict(
            buttons=[
                dict(count=1, label="1M", step="month", stepmode="backward"),
                dict(count=3, label="3M", step="month", stepmode="backward"),
                dict(count=6, label="6M", step="month", stepmode="backward"),
                dict(count=1, label="1Y", step="year", stepmode="backward"),
                dict(step="all", label="All"),
            ],
            x=1, xanchor="right", y=1.0, yanchor="bottom",
        ),
        row=1, col=1,
    )
    fig.update_yaxes(
        title_text="Temperature (°C)",
        showgrid=True, gridcolor="rgba(0,0,0,0.06)",
        zeroline=False,
        row=1, col=1,
    )
    fig.update_yaxes(
        title_text="Hours in alert (%)",
        range=[0, 100],
        showgrid=True, gridcolor="rgba(0,0,0,0.06)",
        row=2, col=1,
    )
    fig.update_xaxes(title_text="Time", row=2, col=1)

    if show:
        fig.show()

    return fig

def plot_calendar_heatmap(
    df: pd.DataFrame,
    city: str,
    threshold: float,
    years=(2021, 2022, 2023),
    temperature_col: str = "temperature",
    bin_edges=(0, 10, 20, 30),
    show: bool = True,
):
    """
    Stacked calendar heatmap (one row per year, all in the same figure).
    - X-axis: weeks of the year (weeks start on Sunday)
    - Y-axis: weekdays, Sunday to Saturday
    - Colour: discrete temperature classes (daily maximum)
    - Months: separated by black outlines (no numbers in the cells)
    - Hot days (daily max >= threshold): last, red class
    """

    # ------------------------------------------------------------------
    # Daily maximum temperature
    # ------------------------------------------------------------------
    data = df[["time", temperature_col]].copy()
    data["time"] = pd.to_datetime(data["time"])
    daily_all = data.set_index("time")[temperature_col].resample("D").max()

    # ------------------------------------------------------------------
    # Discrete temperature classes (the hot threshold is always the last edge)
    # ------------------------------------------------------------------
    edges = [e for e in bin_edges if e < threshold] + [threshold]
    n_cls = len(edges) + 1          # class 0 .. n_cls-1
    NO_DATA = n_cls                 # extra class for missing days

    labels = [f"< {edges[0]:g} °C"]
    for lo, hi in zip(edges[:-1], edges[1:]):
        labels.append(f"{lo:g} – {hi:g} °C")
    labels.append(f"≥ {threshold:.2f} °C (hot day)")

    base = ["#a64dff", "#b3dcf7", "#4db84d", "#ffc20e", "#ff8c1a"]
    idx = np.linspace(0, len(base) - 1, n_cls - 1).round().astype(int)
    colors = [base[i] for i in idx] + ["#e60000"]
    no_data_color = "#d0d0d0"

    # Discrete colour scale: integer class k -> colors[k]
    all_colors = colors + [no_data_color]
    n_total = len(all_colors)
    colorscale = []
    for k, c in enumerate(all_colors):
        colorscale += [[k / n_total, c], [(k + 1) / n_total, c]]

    # ------------------------------------------------------------------
    # Figure with one row per year
    # ------------------------------------------------------------------
    years = list(years)
    n_years = len(years)
    fig = make_subplots(
        rows=n_years, cols=1, vertical_spacing=0.07
    )

    for row, year in enumerate(years, start=1):
        dates = pd.date_range(f"{year}-01-01", f"{year}-12-31", freq="D")
        daily = daily_all.reindex(dates)

        # Grid starts on the Sunday on/before 1 January
        first_sunday = dates[0] - pd.Timedelta(days=(dates[0].weekday() + 1) % 7)
        week_idx = ((dates - first_sunday).days // 7).astype(int)
        wday_idx = np.array([(d.weekday() + 1) % 7 for d in dates])  # Sun=0
        n_weeks = int(week_idx.max()) + 1

        z = np.full((7, n_weeks), np.nan)
        month_grid = np.zeros((7, n_weeks), dtype=int)   # 0 = outside the year
        date_text = np.full((7, n_weeks), "", dtype=object)
        temp_text = np.full((7, n_weeks), "", dtype=object)

        for i, date in enumerate(dates):
            r, c = wday_idx[i], week_idx[i]
            value = daily.iloc[i]
            month_grid[r, c] = date.month
            date_text[r, c] = date.strftime("%a, %Y-%m-%d")

            if pd.isna(value):
                z[r, c] = NO_DATA
                temp_text[r, c] = "Unavailable"
            else:
                z[r, c] = int(np.searchsorted(edges, value, side="right"))
                temp_text[r, c] = f"{value:.1f} °C"

        fig.add_trace(
            go.Heatmap(
                x=np.arange(n_weeks),
                y=np.arange(7),
                z=z,
                zmin=-0.5,
                zmax=n_total - 0.5,
                colorscale=colorscale,
                showscale=False,
                xgap=1,
                ygap=1,
                customdata=np.stack([date_text, temp_text], axis=-1),
                hovertemplate=(
                    "%{customdata[0]}<br>"
                    "Daily maximum: %{customdata[1]}"
                    "<extra></extra>"
                ),
                connectgaps=False,
            ),
            row=row,
            col=1,
        )

        # --------------------------------------------------------------
        # Black outlines around each month
        # --------------------------------------------------------------
        xs, ys = [], []

        def seg(x0, y0, x1, y1):
            xs.extend([x0, x1, None])
            ys.extend([y0, y1, None])

        for r in range(7):
            for c in range(n_weeks):
                m = month_grid[r, c]
                if m == 0:
                    continue

                # right / bottom edges: draw when neighbour is another month
                # (or outside); left / top: only when neighbour is outside,
                # to avoid drawing shared edges twice.
                right = month_grid[r, c + 1] if c + 1 < n_weeks else 0
                left = month_grid[r, c - 1] if c - 1 >= 0 else 0
                down = month_grid[r + 1, c] if r + 1 < 7 else 0
                up = month_grid[r - 1, c] if r - 1 >= 0 else 0

                if right != m:
                    seg(c + 0.5, r - 0.5, c + 0.5, r + 0.5)
                if down != m:
                    seg(c - 0.5, r + 0.5, c + 0.5, r + 0.5)
                if left == 0:
                    seg(c - 0.5, r - 0.5, c - 0.5, r + 0.5)
                if up == 0:
                    seg(c - 0.5, r - 0.5, c + 0.5, r - 0.5)

        fig.add_trace(
            go.Scatter(
                x=xs,
                y=ys,
                mode="lines",
                line=dict(color="black", width=1.6),
                hoverinfo="skip",
                showlegend=False,
            ),
            row=row,
            col=1,
        )

        # --------------------------------------------------------------
        # Axes
        # --------------------------------------------------------------
        month_pos, month_lab = [], []
        for m in range(1, 13):
            cols = np.where((month_grid == m).any(axis=0))[0]
            month_pos.append(cols.mean())
            month_lab.append(pd.Timestamp(year=year, month=m, day=1).strftime("%b").upper())

        fig.update_xaxes(
            range=[-0.5, n_weeks - 0.5],
            tickmode="array",
            tickvals=month_pos if row == 1 else [],
            ticktext=month_lab if row == 1 else [],
            side="top",
            showgrid=False,
            zeroline=False,
            ticks="",
            tickfont=dict(size=11),
            row=row,
            col=1,
        )
        fig.update_yaxes(
            range=[6.5, -0.5],            # Sunday at the top
            tickmode="array",
            tickvals=list(range(7)),
            ticktext=["Sunday", "Monday", "Tuesday", "Wednesday",
                      "Thursday", "Friday", "Saturday"],
            tickfont=dict(size=9),
            showgrid=False,
            zeroline=False,
            ticks="",
            row=row,
            col=1,
        )

        # Year label on the left of each block
        fig.add_annotation(
            text=f"<b>{year}</b>",
            xref="paper",
            yref="y domain" if row == 1 else f"y{row} domain",
            x=-0.075,
            y=0.95,
            showarrow=False,
            xanchor="right",
            font=dict(size=13),
        )

    # ------------------------------------------------------------------
    # Legend (dummy traces, since heatmaps have no discrete legend)
    # ------------------------------------------------------------------
    for lab, col in zip(labels, colors):
        fig.add_trace(
            go.Scatter(
                x=[None], y=[None], mode="markers",
                marker=dict(symbol="square", size=12, color=col,
                            line=dict(color="#888", width=0.5)),
                name=lab,
                hoverinfo="skip",
            ),
            row=1, col=1,
        )

    fig.update_layout(
        title=dict(
            text=(
                f"{city} — Daily Maximum Temperature, {years[0]}–{years[-1]}"
                f"<br><sup>Hot day: daily maximum ≥ {threshold:.2f} °C</sup>"
            ),
            x=0.5,
        ),
        template="plotly_white",
        width=1200,
        height=140 + 175 * n_years,
        margin=dict(l=120, r=40, t=110, b=60),
        legend=dict(orientation="h", x=0.5, xanchor="center",
                    y=-0.02 if n_years > 1 else -0.15, yanchor="top"),
    )

    if show:
        fig.show()

    return fig
