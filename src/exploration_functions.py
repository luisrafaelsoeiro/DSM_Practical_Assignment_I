import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from scipy.stats import ks_2samp

import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly.colors import qualitative

from config import (
    PROCESSED_DIR,
    RAW_DIR,
    TIMEZONE,
    C_TAU,
    VAR_STYLE,
    FALLBACK_COLORS,
    NIGHT_HOURS,
    NIGHT_START_HOUR,
    NIGHT_END_HOUR,
    REQUIRED_COLUMNS,
    NIGHT_BUFFER_DAYS
)

# Folder for the full-resolution (heavy) figures. It is git-ignored, so
# nothing in here ever reaches the repo. Better: define FIGURES_DIR in config.py
# and import it above instead of this fallback.
try:
    from config import FIGURES_DIR
except ImportError:
    FIGURES_DIR = Path("figures")


def export_figure(fig: go.Figure, name: str) -> Path:
    """
    Save a full-resolution figure to FIGURES_DIR/<name>.html.

    Plotly.js is loaded from the CDN (`include_plotlyjs="cdn"`), so each file
    only holds the data, not the ~4 MB library. Open it in a browser to zoom
    into the raw hourly data. Nothing is embedded in the notebook.
    """
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / f"{name}.html"
    fig.write_html(path, include_plotlyjs="cdn")
    print(f"Full-resolution figure saved to {path}")
    return path



def add_night_min_temperature(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add `night_min_temperature`: the minimum temperature of the most recent
    *completed* night (NIGHT_START_HOUR -> NIGHT_END_HOUR, local time).

    A night is labelled with the date on which it ends. For a timestamp t,
    the preceding night is the one that ended at 06:00 on t's date if
    t >= 06:00, otherwise the one that ended on the previous day. Only past
    data is used, so there is no look-ahead leakage in the stream.
    Nights with too few observations (e.g. the first, partial night) -> NaN.
    """
    result = df.sort_values("time").reset_index(drop=True).copy()

    time = pd.to_datetime(result["time"])
    hour = time.dt.hour
    day = time.dt.normalize()

    is_night = (hour >= NIGHT_START_HOUR) | (hour < NIGHT_END_HOUR)

    # Label each night with the date on which it ends
    night_label = day.where(hour < NIGHT_END_HOUR, day + pd.Timedelta(days=1))

    night = pd.DataFrame({
        "night_label": night_label[is_night],
        "temperature": result.loc[is_night, "temperature"],
    })
    grouped = night.groupby("night_label")["temperature"]

    # Tolerate one missing hour (e.g. DST), but not partial nights
    night_min = grouped.min().where(grouped.size() >= NIGHT_HOURS - 1)

    # Most recent night that has already finished at each timestamp
    last_night = day.where(hour >= NIGHT_END_HOUR, day - pd.Timedelta(days=1))
    result["night_min_temperature"] = last_night.map(night_min)

    return result



def build_weather_dataframe(data: dict, city_name: str, start_date: str) -> pd.DataFrame:
    df = pd.DataFrame(data["hourly"]).rename(
        columns={
            "temperature_2m": "temperature",
            "relative_humidity_2m": "relative_humidity",
            "wind_speed_10m": "wind_speed",
        }
    )
    df["time"] = pd.to_datetime(df["time"])
    df = add_night_min_temperature(df)
    df["city"] = city_name

    # Drop the buffer days now that the feature has been computed
    df = df[df["time"] >= pd.Timestamp(start_date)]

    return df[REQUIRED_COLUMNS].sort_values("time").reset_index(drop=True)



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

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    if processed_path.exists() and not force_download:
        df = pd.read_csv(processed_path, parse_dates=["time"])

        if set(REQUIRED_COLUMNS).issubset(df.columns):
            print(f"{city_name}: loading cached data from {processed_path}")
            return df

        # Cache has the old schema: rebuild from the raw JSON if we have it
        if raw_path.exists():
            print(f"{city_name}: cache outdated, rebuilding from {raw_path}")
            with raw_path.open("r", encoding="utf-8") as file:
                data = json.load(file)
            df = build_weather_dataframe(data, city_name)
            df.to_csv(processed_path, index=False)
            return df
        # otherwise fall through and download again

    url = "https://archive-api.open-meteo.com/v1/archive"

    buffered_start = (
        pd.Timestamp(start_date) - pd.Timedelta(days=NIGHT_BUFFER_DAYS)
    ).strftime("%Y-%m-%d")


    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": buffered_start,
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
        raise ValueError(f"Invalid hourly API response for {city_name}.")

    with raw_path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)

    df = build_weather_dataframe(data, city_name, start_date)
    df.to_csv(processed_path, index=False)

    print(
        f"{city_name}: downloaded {len(df):,} observations "
        f"and saved to {processed_path}"
    )

    return df



def _ts(value, tz):
    """Timestamp aligned with the data's timezone."""
    ts = pd.Timestamp(value)
    if tz is not None and ts.tzinfo is None:
        ts = ts.tz_localize(tz)
    return ts



def plot_multivariate_timeseries(
    df: pd.DataFrame,
    city: str,
    variables,
    threshold: float = None,
    warmup_end=None,
    start=None,
    end=None,
    smooth: str = "24h",
    units: dict = None,
    show: bool = True,
    plot_freq: str | None = "D",
    export_name: str | None = None,
):
    """
    Aligned multivariate time-series panels (one panel per variable, shared
    time axis) for one city.

    Size control (this is what made the notebook huge):
      plot_freq   : the figure embedded in the notebook is resampled to this
                    frequency ("D" -> 24x fewer points, "12h" -> 12x, "6h" -> 4x).
                    Use None for the full hourly data.
      export_name : if set, the FULL hourly figure is also written to
                    FIGURES_DIR/<export_name>.html (git-ignored).

      - light line : raw (hourly) observations
      - dark line  : centred moving average (default 24 h)
      - precipitation : daily totals as bars
      - temperature panel : optional frozen threshold (purple dashed line)
      - grey band : warm-up period
      - star      : peak of each variable (value and date in the hover/label)
    """
    data = df[["time"] + list(variables)].copy()
    data["time"] = pd.to_datetime(data["time"])
    data = data.sort_values("time").drop_duplicates("time").set_index("time")
    tz = data.index.tz

    if start is not None:
        data = data.loc[_ts(start, tz):]
    if end is not None:
        data = data.loc[:_ts(end, tz)]

    # Heavy version: full hourly data, saved to disk instead of the notebook
    if export_name is not None and plot_freq is not None:
        full = plot_multivariate_timeseries(
            df, city, variables, threshold=threshold, warmup_end=warmup_end,
            start=start, end=end, smooth=smooth, units=units, show=False,
            plot_freq=None, export_name=None,
        )
        export_figure(full, export_name)

    n_rows = len(variables)
    fig = make_subplots(
        rows=n_rows, cols=1, shared_xaxes=True,
        vertical_spacing=0.035,
    )

    for r, var in enumerate(variables, start=1):
        style = VAR_STYLE.get(var, dict(
            label=var.replace("_", " ").capitalize(), unit="",
            color=FALLBACK_COLORS[(r - 1) % len(FALLBACK_COLORS)], kind="line"))
        if units and var in units:
            style = {**style, "unit": units[var]}
        label, unit, color = style["label"], style["unit"], style["color"]
        s = data[var]

        if style["kind"] == "bar":
            # daily totals are far easier to read than 26k hourly bars
            daily = s.resample("D").sum(min_count=1)
            fig.add_trace(
                go.Bar(
                    x=daily.index + pd.Timedelta(hours=12), y=daily.values,
                    marker=dict(color=color, line=dict(width=0)),
                    width=24 * 3600 * 1000,
                    name=f"{label} (daily total)",
                    hovertemplate=f"%{{y:.1f}} {unit}/day",
                ),
                row=r, col=1,
            )
            peak_x, peak_y = daily.idxmax() + pd.Timedelta(hours=12), daily.max()
        else:
            # Moving average on the full data, THEN downsample for plotting
            ma = s.rolling(smooth, center=True, min_periods=1).mean()
            s_plot = s
            if plot_freq is not None:
                s_plot = s.resample(plot_freq).mean()
                ma = ma.resample(plot_freq).mean()
            fig.add_trace(
                go.Scatter(
                    x=s_plot.index, y=s_plot.values, mode="lines",
                    line=dict(color=color, width=0.8), opacity=0.35,
                    name=f"{label} ({plot_freq or 'hourly'})",
                    hovertemplate=f"%{{y:.1f}} {unit}",
                ),
                row=r, col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=ma.index, y=ma.values, mode="lines",
                    line=dict(color=color, width=1.8),
                    name=f"{label} ({smooth} avg)",
                    hovertemplate=f"%{{y:.1f}} {unit}",
                ),
                row=r, col=1,
            )
            peak_x, peak_y = s.idxmax(), s.max()

        # Peak marker
        if pd.notna(peak_y):
            fig.add_trace(
                go.Scatter(
                    x=[peak_x], y=[peak_y], mode="markers+text",
                    marker=dict(symbol="star", size=11, color=color,
                                line=dict(color="white", width=1)),
                    text=[f"max {peak_y:.1f} {unit}"],
                    textposition="top center",
                    textfont=dict(size=11, color=color),
                    cliponaxis=False,
                    hoverinfo="skip", showlegend=False,
                ),
                row=r, col=1,
            )

        # Axis styling
        yaxis_kw = dict(
            title_text=f"{label} ({unit})" if unit else label,
            showgrid=True, gridcolor="rgba(0,0,0,0.06)", zeroline=False,
        )
        if var == "humidity":
            yaxis_kw["range"] = [0, 105]
        elif style["kind"] == "bar" or var == "wind_speed":
            yaxis_kw["rangemode"] = "tozero"
        fig.update_yaxes(row=r, col=1, **yaxis_kw)

        # Frozen threshold on the temperature panel
        if var == "temperature" and threshold is not None:
            fig.add_trace(
                go.Scatter(
                    x=[s.index[0], s.index[-1]], y=[threshold, threshold],
                    mode="lines", line=dict(color=C_TAU, width=2, dash="dash"),
                    name=f"Frozen threshold ({threshold:.1f} °C)",
                    hovertemplate="τ = %{y:.1f} °C",
                ),
                row=r, col=1,
            )
            fig.add_annotation(
                x=s.index[-1], y=threshold, xref="x", yref=f"y{'' if r == 1 else r}",
                text=f"<b>τ = {threshold:.1f} °C</b>", showarrow=False,
                xanchor="right", yanchor="bottom", yshift=3,
                font=dict(size=12, color=C_TAU),
            )

    # ------------------------------------------------------------------
    # Warm-up band across all panels
    # ------------------------------------------------------------------
    shapes = []
    if warmup_end is not None:
        w = _ts(warmup_end, tz)
        shapes.append(dict(
            type="rect", xref="x", yref="paper",
            x0=data.index[0], x1=w, y0=0, y1=1,
            fillcolor="rgba(120,120,120,0.12)", line=dict(width=0),
            layer="below",
        ))
        shapes.append(dict(
            type="line", xref="x", yref="paper",
            x0=w, x1=w, y0=0, y1=1,
            line=dict(color="grey", width=1, dash="dot"),
        ))
        fig.add_annotation(
            x=data.index[0] + (w - data.index[0]) / 2, y=1.0,
            xref="x", yref="paper", text="Warm-up", showarrow=False,
            yanchor="bottom", font=dict(size=11, color="grey"),
        )
    fig.update_layout(shapes=shapes)

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------
    missing = 100 * data[list(variables)].isna().mean().mean()
    fig.update_layout(
        template="plotly_white",
        width=1250,
        height=190 * n_rows + 140,
        showlegend=False,
        hovermode="x unified",
        title=dict(
            text=(
                f"<b>Weather Variables — {city}</b>"
                f"<br><sup>{data.index[0]:%Y-%m-%d} → {data.index[-1]:%Y-%m-%d} · "
                f"{len(data):,} observations · {missing:.1f}% missing · "
                f"light line = raw ({plot_freq or 'hourly'}), "
                f"dark line = {smooth} moving average</sup>"
            ),
            x=0.02,
        ),
        margin=dict(l=80, r=30, t=110, b=50),
        font=dict(family="Inter, Arial, sans-serif", size=12),
    )
    fig.update_xaxes(
        showgrid=True, gridcolor="rgba(0,0,0,0.06)",
        showspikes=True, spikemode="across", spikethickness=1,
        spikecolor="grey", spikedash="dot",
    )
    fig.update_xaxes(
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
    fig.update_xaxes(title_text="Time", row=n_rows, col=1)

    if show:
        fig.show()
    return fig


def count_iqr_outliers(series: pd.Series) -> int:
    """Count values outside the 1.5 * IQR whiskers (same rule as the box plot)."""
    series = series.dropna()
    q1, q3 = series.quantile([0.25, 0.75])
    iqr = q3 - q1
    return int(((series < q1 - 1.5 * iqr) | (series > q3 + 1.5 * iqr)).sum())


def _box_stats(values: pd.Series) -> tuple[dict, pd.Series]:
    """
    Box statistics computed in pandas (same rule Plotly uses: quartiles with
    linear interpolation, whiskers = most extreme points within 1.5 * IQR).
    Returns (stats, mask_of_outliers).
    """
    q1, median, q3 = values.quantile([0.25, 0.5, 0.75])
    iqr = q3 - q1
    inside = (values >= q1 - 1.5 * iqr) & (values <= q3 + 1.5 * iqr)
    stats = dict(
        q1=q1, median=median, q3=q3,
        lowerfence=values[inside].min(), upperfence=values[inside].max(),
        mean=values.mean(), sd=values.std(),
    )
    return stats, ~inside


def _sample_outliers(out: pd.DataFrame, col: str, max_points: int | None, seed: int = 0):
    """
    Keep at most `max_points` outliers: always the most extreme ones at both
    ends (the interesting ones for heat stress), plus a random sample of the rest.
    """
    if max_points is None or len(out) <= max_points:
        return out
    n_ext = max_points // 4
    extreme = pd.concat([out.nsmallest(n_ext, col), out.nlargest(n_ext, col)])
    extreme = extreme[~extreme.index.duplicated()]
    rest = out.drop(extreme.index)
    keep = rest.sample(min(len(rest), max_points - len(extreme)), random_state=seed)
    return pd.concat([extreme, keep])


def plot_outlier_boxplots(
    weather_data: dict[str, pd.DataFrame],
    variables: list[str],
    n_cols: int = 3,
    district_colors: dict[str, str] | None = None,
    height_per_row: int = 425,
    title: str = "Outlier Analysis of Weather Variables by District",
    max_outliers: int | None = 400,
) -> go.Figure:
    """
    Plot one box plot panel per variable, with one box per city/district.

    Each panel title shows the number of IQR outliers per city, using the
    same 1.5 * IQR rule as the box whiskers. The grid grows with the number
    of variables; unused cells in the last row are left empty.

    Size control: boxes are drawn from pre-computed statistics (q1, median,
    q3, fences, mean), so the raw ~26k points per box are NOT stored in the
    figure. Only the outliers are sent, capped at `max_outliers` per
    city/variable (extremes always kept). `max_outliers=None` keeps them all
    (use that one for the exported version).
    """
    cities = list(weather_data.keys())

    # Default colours: fall back to Plotly's qualitative palette
    palette = qualitative.Plotly
    district_colors = {
        city: (district_colors or {}).get(city, palette[i % len(palette)])
        for i, city in enumerate(cities)
    }

    n_vars = len(variables)
    n_cols = min(n_cols, n_vars)
    n_rows = math.ceil(n_vars / n_cols)

    # Subplot titles with outlier counts per city
    subplot_titles = []
    for variable in variables:
        counts = " | ".join(
            f"{city}: {count_iqr_outliers(weather_data[city][variable])}"
            for city in cities
        )
        subplot_titles.append(
            f"<b>{variable.replace('_', ' ').title()}</b>"
            f"<br><sup>Outliers - {counts}</sup>"
        )

    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        subplot_titles=subplot_titles,
        vertical_spacing=0.14 if n_rows <= 2 else 0.10,
        horizontal_spacing=0.08,
    )

    for i, variable in enumerate(variables):
        row, col = i // n_cols + 1, i % n_cols + 1
        label = variable.replace("_", " ").title()

        for city in cities:
            city_df = weather_data[city][["time", variable]].dropna().reset_index(drop=True)
            color = district_colors[city]

            stats, is_outlier = _box_stats(city_df[variable])

            # Box from pre-computed statistics (no raw data stored)
            fig.add_trace(
                go.Box(
                    x=[city],
                    q1=[stats["q1"]], median=[stats["median"]], q3=[stats["q3"]],
                    lowerfence=[stats["lowerfence"]], upperfence=[stats["upperfence"]],
                    mean=[stats["mean"]], sd=[stats["sd"]],
                    name=city,
                    legendgroup=city,
                    showlegend=(i == 0),  # one legend entry per city
                    boxmean=True,
                    marker=dict(color=color),
                    line=dict(width=1.5),
                ),
                row=row,
                col=col,
            )

            # Outliers (only these are stored, optionally capped)
            out = _sample_outliers(city_df[is_outlier], variable, max_outliers)
            fig.add_trace(
                go.Scatter(
                    x=[city] * len(out),
                    y=out[variable],
                    mode="markers",
                    legendgroup=city,
                    showlegend=False,
                    marker=dict(color=color, size=4, opacity=0.6),
                    customdata=out["time"].dt.strftime("%Y-%m-%d %H:%M"),
                    hovertemplate=(
                        f"<b>{city}</b><br>{label}: %{{y:.2f}}"
                        "<br>Time: %{customdata}<extra></extra>"
                    ),
                ),
                row=row,
                col=col,
            )

        fig.update_yaxes(title_text=label, row=row, col=col)

    fig.update_layout(
        title=dict(text=title, x=0.5),
        height=height_per_row * n_rows,
        template="plotly_white",
        legend=dict(
            title="District", orientation="h", y=1.08, x=0.5, xanchor="center"
        ),
        margin=dict(t=140),
    )

    return fig



# ----------------------------------------------------------------------
# Shared helpers
# ----------------------------------------------------------------------
def _resolve_colors(cities, district_colors=None):
    palette = qualitative.Plotly
    district_colors = district_colors or {}
    return {
        city: district_colors.get(city, palette[i % len(palette)])
        for i, city in enumerate(cities)
    }


def _make_grid(n_items, n_cols, titles, vertical_spacing=0.14):
    n_cols = min(n_cols, n_items)
    n_rows = int(np.ceil(n_items / n_cols))
    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        subplot_titles=titles,
        vertical_spacing=vertical_spacing,
        horizontal_spacing=0.07,
    )
    return fig, n_rows, n_cols


def _get_values(df: pd.DataFrame, variable: str) -> pd.Series:
    """
    Values used for distribution plots.

    `night_min_temperature` is constant for 24 h (it changes only at the end
    of the night), so using every hourly row would count each night ~24 times.
    We keep one value per day, taken at NIGHT_END_HOUR.
    """
    if variable == "night_min_temperature":
        return df.loc[df["time"].dt.hour == NIGHT_END_HOUR, variable].dropna()
    return df[variable].dropna()


def _label(variable: str) -> str:
    return variable.replace("_", " ").title()


def _binned(values, lo: float, hi: float, nbins: int, density: bool = True):
    """
    Histogram computed in numpy so the figure stores `nbins` bars instead of
    every raw value (go.Histogram embeds all the data in the notebook).
    Returns (bin_centres, heights, bin_width).
    density=True  -> probability density; density=False -> share of values.
    """
    if hi <= lo:
        lo, hi = lo - 0.5, lo + 0.5
    edges = np.linspace(lo, hi, nbins + 1)
    counts, _ = np.histogram(values, bins=edges)
    width = edges[1] - edges[0]
    heights = counts / (counts.sum() * width) if density else counts / max(len(values), 1)
    return (edges[:-1] + edges[1:]) / 2, heights, width


# ----------------------------------------------------------------------
# 1. Overlaid histograms (both cities in the same panel)
# ----------------------------------------------------------------------
def plot_distribution_overlays(
    weather_data: dict[str, pd.DataFrame],
    variables: list[str],
    n_cols: int = 3,
    nbins: int = 50,
    district_colors: dict[str, str] | None = None,
    height_per_row: int = 380,
) -> go.Figure:
    """
    One panel per variable, with the cities overlaid as density histograms.
    Both cities share the same bins so shapes are directly comparable.
    Dashed vertical lines mark each city's median.
    """
    cities = list(weather_data)
    colors = _resolve_colors(cities, district_colors)

    titles = [f"<b>{_label(v)}</b>" for v in variables]
    fig, n_rows, n_cols = _make_grid(len(variables), n_cols, titles)

    for i, variable in enumerate(variables):
        row, col = i // n_cols + 1, i % n_cols + 1
        values = {c: _get_values(weather_data[c], variable) for c in cities}

        pooled = pd.concat(values.values())
        lo, hi = pooled.min(), pooled.max()

        for city in cities:
            centres, density, width = _binned(values[city], lo, hi, nbins)
            fig.add_trace(
                go.Bar(
                    x=centres,
                    y=density,
                    width=width,
                    name=city,
                    legendgroup=city,
                    showlegend=(i == 0),
                    opacity=0.55,
                    marker=dict(color=colors[city], line=dict(width=0)),
                    hovertemplate=(
                        f"<b>{city}</b><br>{_label(variable)}: %{{x}}"
                        "<br>Density: %{y:.4f}<extra></extra>"
                    ),
                ),
                row=row,
                col=col,
            )
            fig.add_vline(
                x=values[city].median(),
                line=dict(color=colors[city], dash="dash", width=1.5),
                row=row,
                col=col,
            )

        fig.update_xaxes(title_text=_label(variable), row=row, col=col)
        fig.update_yaxes(title_text="Density", row=row, col=col)

    fig.update_layout(
        title=dict(text="Distribution of Weather Variables by District", x=0.5),
        barmode="overlay",
        bargap=0,
        height=height_per_row * n_rows,
        template="plotly_white",
        legend=dict(title="District", orientation="h", y=1.08, x=0.5, xanchor="center"),
        margin=dict(t=130),
    )
    return fig


# ----------------------------------------------------------------------
# 2. ECDF comparison (tails visible, KS statistic in the title)
# ----------------------------------------------------------------------
def plot_ecdf_comparison(
    weather_data: dict[str, pd.DataFrame],
    variables: list[str],
    n_cols: int = 3,
    n_points: int = 500,
    district_colors: dict[str, str] | None = None,
    height_per_row: int = 380,
) -> go.Figure:
    """
    Empirical CDFs of each variable, cities overlaid.

    The ECDF is drawn on a grid of `n_points` quantiles (keeps the figure
    light for 8,760 points). With exactly two cities, the two-sample
    Kolmogorov-Smirnov statistic is shown in each panel title: it is the
    maximum vertical gap between the two curves.
    """
    cities = list(weather_data)
    colors = _resolve_colors(cities, district_colors)
    probs = np.linspace(0, 1, n_points)

    titles = []
    values_by_var = {}
    for v in variables:
        values_by_var[v] = {c: _get_values(weather_data[c], v) for c in cities}
        title = f"<b>{_label(v)}</b>"
        if len(cities) == 2:
            a, b = (values_by_var[v][c] for c in cities)
            ks = ks_2samp(a, b)
            title += f"<br><sup>KS = {ks.statistic:.3f} (p = {ks.pvalue:.2g})</sup>"
        titles.append(title)

    fig, n_rows, n_cols = _make_grid(len(variables), n_cols, titles)

    for i, variable in enumerate(variables):
        row, col = i // n_cols + 1, i % n_cols + 1

        for city in cities:
            x = np.quantile(values_by_var[variable][city], probs)
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=probs,
                    mode="lines",
                    name=city,
                    legendgroup=city,
                    showlegend=(i == 0),
                    line=dict(color=colors[city], width=2),
                    hovertemplate=(
                        f"<b>{city}</b><br>{_label(variable)}: %{{x:.2f}}"
                        "<br>Cumulative share: %{y:.1%}<extra></extra>"
                    ),
                ),
                row=row,
                col=col,
            )

        fig.update_xaxes(title_text=_label(variable), row=row, col=col)
        fig.update_yaxes(title_text="Cumulative share", tickformat=".0%", row=row, col=col)

    fig.update_layout(
        title=dict(text="ECDF of Weather Variables by District", x=0.5),
        height=height_per_row * n_rows,
        template="plotly_white",
        legend=dict(title="District", orientation="h", y=1.08, x=0.5, xanchor="center"),
        margin=dict(t=130),
    )
    return fig


# ----------------------------------------------------------------------
# 3. Q-Q plot between two cities
# ----------------------------------------------------------------------
def plot_qq_comparison(
    weather_data: dict[str, pd.DataFrame],
    city_x: str,
    city_y: str,
    variables: list[str],
    n_cols: int = 3,
    tail_start: float = 0.95,
    height_per_row: int = 400,
) -> go.Figure:
    """
    Two-sample Q-Q plot: quantiles of `city_x` (x axis) against `city_y`.

    Points on the dashed 45° line mean identical distributions. Points above
    it mean `city_y` has higher values at that quantile. Quantiles at or
    above `tail_start` are highlighted in red, since the upper tail is what
    matters for heat stress.
    """
    probs = np.concatenate([np.linspace(0.01, 0.99, 99), [0.995, 0.999]])
    titles = [f"<b>{_label(v)}</b>" for v in variables]
    fig, n_rows, n_cols = _make_grid(len(variables), n_cols, titles)

    for i, variable in enumerate(variables):
        row, col = i // n_cols + 1, i % n_cols + 1

        qx = np.quantile(_get_values(weather_data[city_x], variable), probs)
        qy = np.quantile(_get_values(weather_data[city_y], variable), probs)
        is_tail = probs >= tail_start

        for mask, name, color in [
            (~is_tail, "Body", "#7f7f7f"),
            (is_tail, f"Upper tail (≥ p{int(tail_start * 100)})", "#d62728"),
        ]:
            fig.add_trace(
                go.Scatter(
                    x=qx[mask],
                    y=qy[mask],
                    mode="markers",
                    name=name,
                    legendgroup=name,
                    showlegend=(i == 0),
                    marker=dict(color=color, size=6, opacity=0.8),
                    customdata=probs[mask],
                    hovertemplate=(
                        f"{city_x}: %{{x:.2f}}<br>{city_y}: %{{y:.2f}}"
                        "<br>Percentile: %{customdata:.1%}<extra></extra>"
                    ),
                ),
                row=row,
                col=col,
            )

        lo = float(min(qx.min(), qy.min()))
        hi = float(max(qx.max(), qy.max()))
        fig.add_trace(
            go.Scatter(
                x=[lo, hi],
                y=[lo, hi],
                mode="lines",
                line=dict(color="black", dash="dash", width=1),
                showlegend=False,
                hoverinfo="skip",
            ),
            row=row,
            col=col,
        )

        fig.update_xaxes(title_text=f"{city_x}", row=row, col=col)
        fig.update_yaxes(title_text=f"{city_y}", row=row, col=col)

    fig.update_layout(
        title=dict(text=f"Q-Q Plot: {city_y} vs {city_x}", x=0.5),
        height=height_per_row * n_rows,
        template="plotly_white",
        legend=dict(orientation="h", y=1.08, x=0.5, xanchor="center"),
        margin=dict(t=130),
    )
    return fig


# ----------------------------------------------------------------------
# 4. Precipitation (wet hours) and wind (windy hours) in detail
# ----------------------------------------------------------------------
def plot_precipitation_wind_detail(
    weather_data: dict[str, pd.DataFrame],
    precip_col: str = "precipitation",
    wind_col: str = "wind_speed",
    wet_threshold: float = 0.1,
    windy_quantile: float = 0.90,
    nbins: int = 40,
    n_points: int = 300,
    district_colors: dict[str, str] | None = None,
    height: int = 800,
) -> go.Figure:
    """
    Precipitation and wind are skewed, and precipitation is zero-inflated,
    so the full distribution hides what matters. This figure separates:

      (1,1) share of wet hours (precipitation >= `wet_threshold`) per city
      (1,2) intensity distribution of wet hours only (log y axis)
      (2,1) share of windy hours by hour of day
      (2,2) ECDF of wind speed within windy hours only

    "Windy" means wind speed at or above the `windy_quantile` of the
    pooled wind speed across all cities, so both cities use the same cut-off.
    """
    cities = list(weather_data)
    colors = _resolve_colors(cities, district_colors)

    pooled_wind = pd.concat([df[wind_col].dropna() for df in weather_data.values()])
    wind_thr = float(pooled_wind.quantile(windy_quantile))

    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=[
            f"<b>Share of wet hours</b><br><sup>precipitation ≥ {wet_threshold} mm</sup>",
            "<b>Wet-hour intensity</b><br><sup>wet hours only, log scale</sup>",
            f"<b>Windy hours by hour of day</b><br><sup>wind ≥ {wind_thr:.1f} (p{int(windy_quantile * 100)} pooled)</sup>",
            "<b>Wind speed in windy hours</b><br><sup>ECDF of the upper tail</sup>",
        ],
        vertical_spacing=0.2,
        horizontal_spacing=0.1,
    )

    # Shared bins for wet-hour intensity
    wet_pooled = pd.concat(
        [df[precip_col].dropna().loc[lambda s: s >= wet_threshold] for df in weather_data.values()]
    )
    wet_lo, wet_hi = wet_threshold, float(wet_pooled.max())

    probs = np.linspace(0, 1, n_points)

    for city in cities:
        df = weather_data[city]
        color = colors[city]

        # (1,1) share of wet hours
        precip = df[precip_col].dropna()
        wet_share = (precip >= wet_threshold).mean() * 100
        fig.add_trace(
            go.Bar(
                x=[city],
                y=[wet_share],
                name=city,
                legendgroup=city,
                marker_color=color,
                text=[f"{wet_share:.1f}%"],
                textposition="outside",
                hovertemplate=f"<b>{city}</b><br>Wet hours: %{{y:.2f}}%<extra></extra>",
            ),
            row=1,
            col=1,
        )

        # (1,2) wet-hour intensity
        wet_values = precip[precip >= wet_threshold]
        centres, share, wet_width = _binned(
            wet_values, wet_lo, wet_hi, nbins, density=False
        )
        fig.add_trace(
            go.Bar(
                x=centres,
                y=share,
                width=wet_width,
                name=city,
                legendgroup=city,
                showlegend=False,
                opacity=0.55,
                marker=dict(color=color, line=dict(width=0)),
                hovertemplate=(
                    f"<b>{city}</b><br>Intensity: %{{x}} mm"
                    "<br>Share of wet hours: %{y:.3f}<extra></extra>"
                ),
            ),
            row=1,
            col=2,
        )

        # (2,1) share of windy hours by hour of day
        wind_df = df[["time", wind_col]].dropna()
        windy_by_hour = (
            (wind_df[wind_col] >= wind_thr)
            .groupby(wind_df["time"].dt.hour)
            .mean()
            * 100
        )
        fig.add_trace(
            go.Scatter(
                x=windy_by_hour.index,
                y=windy_by_hour.values,
                mode="lines+markers",
                name=city,
                legendgroup=city,
                showlegend=False,
                line=dict(color=color, width=2),
                hovertemplate=(
                    f"<b>{city}</b><br>Hour: %{{x}}:00"
                    "<br>Windy hours: %{y:.1f}%<extra></extra>"
                ),
            ),
            row=2,
            col=1,
        )

        # (2,2) ECDF of wind speed within windy hours
        windy_values = wind_df.loc[wind_df[wind_col] >= wind_thr, wind_col]
        fig.add_trace(
            go.Scatter(
                x=np.quantile(windy_values, probs),
                y=probs,
                mode="lines",
                name=city,
                legendgroup=city,
                showlegend=False,
                line=dict(color=color, width=2),
                hovertemplate=(
                    f"<b>{city}</b><br>Wind speed: %{{x:.1f}}"
                    "<br>Cumulative share: %{y:.1%}<extra></extra>"
                ),
            ),
            row=2,
            col=2,
        )

    fig.update_yaxes(title_text="% of hours", row=1, col=1)
    fig.update_xaxes(title_text="Precipitation (mm/h)", row=1, col=2)
    fig.update_yaxes(title_text="Share of wet hours", type="log", row=1, col=2)
    fig.update_xaxes(title_text="Hour of day", dtick=3, row=2, col=1)
    fig.update_yaxes(title_text="% of hours windy", row=2, col=1)
    fig.update_xaxes(title_text="Wind speed", row=2, col=2)
    fig.update_yaxes(title_text="Cumulative share", tickformat=".0%", row=2, col=2)

    fig.update_layout(
        title=dict(text="Precipitation and Wind: Wet and Windy Hours", x=0.5),
        barmode="overlay",
        bargap=0,
        height=height,
        template="plotly_white",
        legend=dict(title="District", orientation="h", y=1.1, x=0.5, xanchor="center"),
        margin=dict(t=140),
    )
    return fig



def plot_correlation_heatmaps(
    weather_data: dict[str, pd.DataFrame],
    variables: list[str],
    colorscale: str = "RdBu_r",
    height: int = 520,
    width_per_city: int = 560,
) -> go.Figure:
    """
    Plot one annotated correlation heatmap per city, side by side,
    sharing a single colour scale (-1 to 1, centred on 0).
    """
    cities = list(weather_data.keys())
    n = len(cities)

    fig = make_subplots(
        rows=1,
        cols=n,
        subplot_titles=[f"{city} — Correlation Matrix" for city in cities],
        horizontal_spacing=0.14,
    )

    for i, city in enumerate(cities, start=1):
        corr = weather_data[city][variables].corr()

        fig.add_trace(
            go.Heatmap(
                z=corr.values,
                x=corr.columns,
                y=corr.index,
                zmin=-1,
                zmax=1,
                zmid=0,
                colorscale=colorscale,
                texttemplate="%{z:.2f}",
                hovertemplate=(
                    "%{y} vs %{x}<br>correlation = %{z:.3f}<extra></extra>"
                ),
                showscale=(i == n),  # single shared colourbar
                colorbar=dict(title="r", thickness=15) if i == n else None,
            ),
            row=1,
            col=i,
        )

        # Diagonal from top-left to bottom-right, like seaborn
        fig.update_yaxes(autorange="reversed", row=1, col=i)
        fig.update_xaxes(tickangle=-45, row=1, col=i)

    fig.update_layout(
        height=height,
        width=width_per_city * n,
        template="plotly_white",
        margin=dict(l=40, r=40, t=80, b=40),
    )

    return fig