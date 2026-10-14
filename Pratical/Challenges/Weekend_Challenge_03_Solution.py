"""Weekend Challenge 03 — Instructor Solution / Worked Example.

Data Stream Mining — NOVA IMS
Week 4 — Online Learning & Stream Evaluation

Challenge idea
--------------
Extend the real-world Open-Meteo replay project from Weeks 1–3 into a
predictive stream. We use two cities and ask a simple classification
question:

    Can meteorological observations identify which city generated them?

The scientific focus is EVALUATION, not sophisticated models. The script
compares familiar batch learners with two deliberately simple online
learners under the same chronological evaluation protocol.

Four strategies
---------------
1. Online Majority                       -> ONLINE baseline
2. Frozen Logistic Regression            -> OFFLINE / batch, trained once
3. Online Logistic Regression            -> ONLINE / incremental
4. Online Gaussian Naive Bayes           -> ONLINE / incremental

The main teaching comparison is deliberately close to apples-to-apples:
a familiar batch Logistic Regression and an incremental logistic learner
share the same linear/logistic modelling idea but use different training
regimes. The online version is implemented with SGDClassifier(loss="log_loss")
and partial_fit(), so it can update one observation at a time.

The central streaming rule
--------------------------
    PREDICT -> EVALUATE -> LEARN (when the learner is online)

The current label is never allowed to influence the prediction being
scored for that same observation.

    THE FUTURE MUST NEVER TRAIN THE PAST.

How to run
----------
    python Weekend_Challenge_03_Solution.py

Recommended packages
--------------------
    python -m pip install requests pandas numpy matplotlib scikit-learn river

The online Logistic Regression uses scikit-learn's SGDClassifier with
``loss="log_loss"`` and ``partial_fit``. River is preferred for the
incremental Gaussian Naive Bayes learner. If River is not installed, this
teaching script automatically uses a small built-in incremental GaussianNB
fallback so the file remains portable and testable.

Important interpretation note
-----------------------------
Timing and Python-level memory measurements depend on the machine and
software environment. Use them to compare strategies within the same run,
not as universal performance benchmarks.
"""

from __future__ import annotations

import json
import math
import re
import time
import tracemalloc
from collections import Counter, deque
from copy import deepcopy
from pathlib import Path
from typing import Callable, Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


# =============================================================================
# 1. STUDENT CUSTOMIZATION ZONE — CHANGE HERE
# =============================================================================
# Start by changing ONLY these values. Get the official example working first.

CITY_A = {"name": "Lisbon", "latitude": 38.7223, "longitude": -9.1393}
CITY_B = {"name": "Porto", "latitude": 41.1579, "longitude": -8.6291}

START_DATE = "2025-09-01"
END_DATE = "2025-09-07"
TIMEZONE = "Europe/Lisbon"

FEATURES = [
    "temperature_2m",
    "relative_humidity_2m",
    "precipitation",
    "wind_speed_10m",
]

# Earliest fraction of TIMESTAMPS reserved for initial batch-model training.
INITIAL_TRAIN_FRACTION = 0.25

# The online Logistic Regression is an SGD-based incremental learner.
# "optimal" lets scikit-learn choose a stable learning-rate schedule.
ONLINE_LOGISTIC_ALPHA = 0.0001

# Recent predictive performance is measured over the latest predictions.
EVALUATION_WINDOW_SIZE = 100

# Optional teaching experiment. It modifies a COPY of the evaluation stream.
ENABLE_EVALUATION_STRESS_TEST = False
STRESS_TEST_START_FRACTION = 0.55
STRESS_TEST_END_FRACTION = 0.70

RANDOM_SEED = 42
FORCE_DOWNLOAD = False

# Output folders are created next to this script.
BASE_DIR = Path(__file__).resolve().parent
RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
FIGURES_DIR = BASE_DIR / "figures_week04"

OPEN_METEO_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"


# =============================================================================
# 2. SMALL FILE / API HELPERS
# =============================================================================

def slugify(text: str) -> str:
    """Turn a label into a filename-friendly string."""
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def build_raw_filename(city_name: str, start_date: str, end_date: str) -> str:
    return f"open_meteo_{slugify(city_name)}_{start_date}_{end_date}.json"


def build_processed_filename(city_name: str, start_date: str, end_date: str) -> str:
    return f"open_meteo_{slugify(city_name)}_{start_date}_{end_date}_week04.csv"


def build_open_meteo_params(
    latitude: float,
    longitude: float,
    start_date: str,
    end_date: str,
    hourly_variables: Sequence[str],
    timezone: str,
) -> dict:
    return {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": ",".join(hourly_variables),
        "timezone": timezone,
    }


def ensure_output_directories(
    raw_dir: Path = RAW_DIR,
    processed_dir: Path = PROCESSED_DIR,
    figures_dir: Path = FIGURES_DIR,
) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)


def download_open_meteo(city: dict) -> dict:
    """Download one fixed historical period from Open-Meteo."""
    params = build_open_meteo_params(
        latitude=float(city["latitude"]),
        longitude=float(city["longitude"]),
        start_date=START_DATE,
        end_date=END_DATE,
        hourly_variables=FEATURES,
        timezone=TIMEZONE,
    )
    print(f"      Downloading {city['name']} ({START_DATE} -> {END_DATE})...")
    try:
        response = requests.get(OPEN_METEO_ARCHIVE_URL, params=params, timeout=30)
        response.raise_for_status()
        payload = response.json()
    except requests.RequestException as exc:
        raise RuntimeError(
            "Could not download Open-Meteo data. Check the connection or reuse "
            "a saved raw snapshot with FORCE_DOWNLOAD=False."
        ) from exc
    except ValueError as exc:
        raise RuntimeError("Open-Meteo returned invalid JSON.") from exc

    if isinstance(payload, dict) and payload.get("error"):
        raise RuntimeError(f"Open-Meteo API error: {payload.get('reason', 'unknown reason')}")
    return payload


def load_or_download_raw(
    raw_path: Path,
    downloader: Callable[[], dict],
    force_download: bool = False,
) -> dict:
    """COLLECT ONCE -> SAVE RAW DATA -> REPLAY MANY TIMES."""
    raw_path = Path(raw_path)
    if raw_path.exists() and not force_download:
        print(f"      Reusing raw snapshot: {raw_path.name}")
        return json.loads(raw_path.read_text(encoding="utf-8"))

    payload = downloader()
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"      Raw response saved to: {raw_path}")
    return payload


def load_city_payload(city: dict, raw_dir: Path = RAW_DIR) -> tuple[Path, dict]:
    raw_path = raw_dir / build_raw_filename(city["name"], START_DATE, END_DATE)
    payload = load_or_download_raw(
        raw_path,
        downloader=lambda: download_open_meteo(city),
        force_download=FORCE_DOWNLOAD,
    )
    return raw_path, payload


# =============================================================================
# 3. PREPARE A TWO-CITY MULTIVARIATE STREAM
# =============================================================================

def prepare_city_frame(payload: dict, city_name: str, features: Sequence[str]) -> pd.DataFrame:
    """Validate and clean one Open-Meteo city payload."""
    hourly = payload.get("hourly", {})
    if "time" not in hourly:
        raise ValueError("Open-Meteo payload is missing hourly.time")

    missing = [feature for feature in features if feature not in hourly]
    if missing:
        raise ValueError(f"Open-Meteo payload is missing required features: {missing}")

    data = {"timestamp": pd.to_datetime(hourly["time"], errors="coerce")}
    for feature in features:
        data[feature] = pd.to_numeric(pd.Series(hourly[feature]), errors="coerce")

    df = pd.DataFrame(data)
    df["city"] = city_name
    before = len(df)
    df = df.dropna(subset=["timestamp", *features]).copy()
    dropped = before - len(df)
    if dropped:
        print(f"      {city_name}: dropped {dropped} row(s) with missing required values.")

    return df[["timestamp", *features, "city"]].reset_index(drop=True)


def build_two_city_events(
    city_frames: dict[str, pd.DataFrame],
    features: Sequence[str],
    *,
    interleave: bool = True,
) -> list[dict]:
    """Convert clean city frames into one deterministic chronological stream."""
    frames = []
    for city_name, frame in city_frames.items():
        local = frame.copy()
        local["city"] = city_name
        frames.append(local[["timestamp", *features, "city"]])

    if not frames:
        return []

    combined = pd.concat(frames, ignore_index=True)
    if interleave:
        # Same timestamps are tied by city name so repeated runs are identical.
        combined = combined.sort_values(["timestamp", "city"], kind="mergesort")
    else:
        combined = combined.sort_values(["city", "timestamp"], kind="mergesort")

    events: list[dict] = []
    for row in combined.itertuples(index=False):
        events.append(
            {
                "timestamp": row.timestamp,
                "x": {feature: float(getattr(row, feature)) for feature in features},
                "y": str(row.city),
            }
        )
    return events


def split_initial_training(
    events: Sequence[dict],
    train_fraction: float,
) -> tuple[list[dict], list[dict]]:
    """Split by complete timestamp groups, never through a tied city pair."""
    if not (0.0 < train_fraction < 1.0):
        raise ValueError("train_fraction must be strictly between 0 and 1")
    if not events:
        raise ValueError("events must not be empty")

    timestamps = sorted({event["timestamp"] for event in events})
    if len(timestamps) < 2:
        raise ValueError("At least two distinct timestamps are required")

    n_train_timestamps = int(len(timestamps) * train_fraction)
    n_train_timestamps = max(1, min(len(timestamps) - 1, n_train_timestamps))
    train_timestamp_set = set(timestamps[:n_train_timestamps])

    train = [event for event in events if event["timestamp"] in train_timestamp_set]
    evaluation = [event for event in events if event["timestamp"] not in train_timestamp_set]

    if len({event["y"] for event in train}) < 2:
        raise ValueError("Initial training period must contain both city classes")
    return train, evaluation


# =============================================================================
# 4. INCREMENTAL EVALUATION STATE
# =============================================================================

class OnlineConfusionMatrix:
    """Exact cumulative or removable-window confusion matrix."""

    def __init__(self, labels: Sequence[str]):
        self.labels = list(labels)
        if not self.labels:
            raise ValueError("labels must not be empty")
        self._index = {label: i for i, label in enumerate(self.labels)}
        self.matrix = np.zeros((len(self.labels), len(self.labels)), dtype=int)
        self.total = 0
        self.correct = 0

    def update(self, y_true: str, y_pred: str) -> None:
        if y_true not in self._index or y_pred not in self._index:
            raise ValueError("Unknown label in confusion matrix update")
        i = self._index[y_true]
        j = self._index[y_pred]
        self.matrix[i, j] += 1
        self.total += 1
        if y_true == y_pred:
            self.correct += 1

    def remove(self, y_true: str, y_pred: str) -> None:
        i = self._index[y_true]
        j = self._index[y_pred]
        if self.matrix[i, j] <= 0:
            raise ValueError("Cannot remove a confusion-matrix pair that is not present")
        self.matrix[i, j] -= 1
        self.total -= 1
        if y_true == y_pred:
            self.correct -= 1

    def accuracy(self) -> float:
        return self.correct / self.total if self.total else 0.0

    def precision(self, label: str) -> float:
        i = self._index[label]
        tp = float(self.matrix[i, i])
        predicted = float(self.matrix[:, i].sum())
        return tp / predicted if predicted else 0.0

    def recall(self, label: str) -> float:
        i = self._index[label]
        tp = float(self.matrix[i, i])
        actual = float(self.matrix[i, :].sum())
        return tp / actual if actual else 0.0

    def f1(self, label: str) -> float:
        p = self.precision(label)
        r = self.recall(label)
        return 2.0 * p * r / (p + r) if (p + r) else 0.0

    def macro_precision(self) -> float:
        return float(np.mean([self.precision(label) for label in self.labels]))

    def macro_recall(self) -> float:
        return float(np.mean([self.recall(label) for label in self.labels]))

    def macro_f1(self) -> float:
        return float(np.mean([self.f1(label) for label in self.labels]))

    def as_array(self) -> np.ndarray:
        return self.matrix.copy()


class StreamingEvaluator:
    """Maintain cumulative AND recent predictive state incrementally."""

    def __init__(self, labels: Sequence[str], window_size: int):
        if window_size <= 0:
            raise ValueError("window_size must be positive")
        self.labels = list(labels)
        self.window_size = int(window_size)
        self.cumulative = OnlineConfusionMatrix(self.labels)
        self.window = OnlineConfusionMatrix(self.labels)
        self.recent_pairs: deque[tuple[str, str]] = deque(maxlen=self.window_size)
        self.observations_seen = 0
        self.predictions_evaluated = 0
        self.cold_starts = 0

    def observe(self, y_true: str, y_pred: str | None) -> dict:
        self.observations_seen += 1

        if y_pred is None:
            self.cold_starts += 1
            return self.snapshot()

        # deque(maxlen=...) silently discards the oldest item. We remove its
        # contribution FIRST so the windowed confusion matrix remains exact.
        if len(self.recent_pairs) == self.window_size:
            old_true, old_pred = self.recent_pairs[0]
            self.window.remove(old_true, old_pred)

        self.recent_pairs.append((y_true, y_pred))
        self.cumulative.update(y_true, y_pred)
        self.window.update(y_true, y_pred)
        self.predictions_evaluated += 1
        return self.snapshot()

    def snapshot(self) -> dict:
        return {
            "observations_seen": self.observations_seen,
            "evaluated": self.predictions_evaluated,
            "cold_starts": self.cold_starts,
            "cumulative_accuracy": self.cumulative.accuracy(),
            "window_accuracy": self.window.accuracy(),
            "cumulative_macro_precision": self.cumulative.macro_precision(),
            "window_macro_precision": self.window.macro_precision(),
            "cumulative_macro_recall": self.cumulative.macro_recall(),
            "window_macro_recall": self.window.macro_recall(),
            "cumulative_macro_f1": self.cumulative.macro_f1(),
            "window_macro_f1": self.window.macro_f1(),
        }


# =============================================================================
# 5. FOUR MODEL STRATEGIES: OFFLINE VS ONLINE
# =============================================================================

class OnlineMajorityClassifier:
    """Minimal online baseline: state is only the class-frequency table."""

    def __init__(self):
        self.counts: Counter[str] = Counter()
        self.learn_count = 0

    def predict_one(self, x: dict[str, float]) -> str | None:
        if not self.counts:
            return None
        max_count = max(self.counts.values())
        # Deterministic tie rule for reproducible classroom output.
        return min(label for label, count in self.counts.items() if count == max_count)

    def learn_one(self, x: dict[str, float], y: str) -> "OnlineMajorityClassifier":
        self.counts[y] += 1
        self.learn_count += 1
        return self


def _rows_to_xy(
    rows: Sequence[tuple[dict[str, float], str]],
    feature_names: Sequence[str] | None = None,
) -> tuple[list[list[float]], list[str], list[str]]:
    if not rows:
        raise ValueError("At least one training row is required")
    if feature_names is None:
        feature_names = list(rows[0][0].keys())
    names = list(feature_names)
    X = [[float(x[name]) for name in names] for x, _ in rows]
    y = [str(label) for _, label in rows]
    return X, y, names


class FrozenBatchClassifier:
    """Traditional OFFLINE classifier: fit once, then remain frozen."""

    def __init__(self, estimator):
        self.estimator = estimator
        self.feature_names: list[str] | None = None
        self.fit_count = 0

    def fit_initial(self, rows: Sequence[tuple[dict[str, float], str]]) -> None:
        X, y, self.feature_names = _rows_to_xy(rows)
        self.estimator.fit(X, y)
        self.fit_count += 1

    def predict_one(self, x: dict[str, float]) -> str:
        if self.feature_names is None:
            raise RuntimeError("fit_initial must be called before prediction")
        values = [[float(x[name]) for name in self.feature_names]]
        return str(self.estimator.predict(values)[0])

    def observe_after_evaluation(self, x: dict[str, float], y: str) -> None:
        # Explicitly do nothing. This is what makes the model frozen.
        return None


class OnlineLogisticRegressionAdapter:
    """Incremental Logistic Regression using scaler.partial_fit + SGD.partial_fit.

    This is intentionally simple and uses only scikit-learn so students can
    connect a familiar linear classifier with a true one-observation-at-a-time
    update mechanism. Prediction always uses state from the past; scaling and
    classifier updates happen only inside ``learn_one``.
    """

    def __init__(
        self,
        classes: Sequence[str],
        random_state: int = RANDOM_SEED,
        alpha: float = ONLINE_LOGISTIC_ALPHA,
    ):
        labels = [str(label) for label in classes]
        if len(labels) < 2:
            raise ValueError("Online logistic regression requires at least two classes")
        self.classes = np.array(sorted(labels), dtype=object)
        self.scaler = StandardScaler()
        self.model = SGDClassifier(
            loss="log_loss",
            learning_rate="optimal",
            alpha=float(alpha),
            random_state=int(random_state),
        )
        self.feature_names: list[str] | None = None
        self.initialized = False
        self.learn_count = 0
        self.backend = 'sklearn SGDClassifier(loss="log_loss", partial_fit)'

    def _vector(self, x: dict[str, float]) -> np.ndarray:
        if self.feature_names is None:
            self.feature_names = list(x.keys())
        return np.asarray([[float(x[name]) for name in self.feature_names]], dtype=float)

    def predict_one(self, x: dict[str, float]) -> str | None:
        if not self.initialized:
            return None
        values = self._vector(x)
        scaled = self.scaler.transform(values)
        return str(self.model.predict(scaled)[0])

    def learn_one(self, x: dict[str, float], y: str) -> "OnlineLogisticRegressionAdapter":
        values = self._vector(x)
        # The scaler is also incremental. It is updated only after the prediction
        # for this observation has already been evaluated by the outer loop.
        self.scaler.partial_fit(values)
        scaled = self.scaler.transform(values)
        if not self.initialized:
            self.model.partial_fit(scaled, [str(y)], classes=self.classes)
            self.initialized = True
        else:
            self.model.partial_fit(scaled, [str(y)])
        self.learn_count += 1
        return self


class _FallbackIncrementalGaussianNB:
    """Dependency-free online GaussianNB used only when River is unavailable."""

    def __init__(self, variance_floor: float = 1e-6):
        self.variance_floor = float(variance_floor)
        self.class_counts: Counter[str] = Counter()
        # class -> feature -> [n, mean, M2]
        self.feature_stats: dict[str, dict[str, list[float]]] = {}
        self.total_count = 0

    def learn_one(self, x: dict[str, float], y: str):
        self.total_count += 1
        self.class_counts[y] += 1
        n = self.class_counts[y]
        stats = self.feature_stats.setdefault(y, {})

        for feature, value in x.items():
            value = float(value)
            if feature not in stats:
                stats[feature] = [1.0, value, 0.0]
                continue

            _, mean, m2 = stats[feature]
            delta = value - mean
            mean += delta / n
            delta2 = value - mean
            m2 += delta * delta2
            stats[feature] = [float(n), mean, m2]
        return self

    def _variance(self, label: str, feature: str) -> float:
        n, _, m2 = self.feature_stats[label][feature]
        if n <= 1:
            return self.variance_floor
        return max(m2 / (n - 1.0), self.variance_floor)

    def predict_one(self, x: dict[str, float]) -> str | None:
        if not self.class_counts:
            return None

        labels = sorted(self.class_counts)
        if len(labels) == 1:
            return labels[0]

        best_label: str | None = None
        best_score = -math.inf

        for label in labels:
            prior = self.class_counts[label] / self.total_count
            score = math.log(max(prior, 1e-15))
            stats = self.feature_stats[label]

            for feature, value in x.items():
                if feature not in stats:
                    continue
                _, mean, _ = stats[feature]
                variance = self._variance(label, feature)
                diff = float(value) - mean
                score += -0.5 * (
                    math.log(2.0 * math.pi * variance) + (diff * diff) / variance
                )

            if score > best_score or (
                score == best_score and (best_label is None or label < best_label)
            ):
                best_score = score
                best_label = label

        return best_label


class RiverGaussianNBAdapter:
    """True online learner: River GaussianNB, with a portable fallback."""

    def __init__(self):
        try:
            from river import naive_bayes  # type: ignore
        except ImportError:
            self.model = _FallbackIncrementalGaussianNB()
            self.backend = "built-in incremental GaussianNB fallback"
        else:
            self.model = naive_bayes.GaussianNB()
            self.backend = "river.naive_bayes.GaussianNB"
        self.learn_count = 0

    def predict_one(self, x: dict[str, float]) -> str | None:
        prediction = self.model.predict_one(x)
        return None if prediction is None else str(prediction)

    def learn_one(self, x: dict[str, float], y: str) -> "RiverGaussianNBAdapter":
        self.model.learn_one(x, y)
        self.learn_count += 1
        return self


def model_strategy_names() -> list[str]:
    return [
        "Online Majority",
        "Frozen Logistic Regression (OFFLINE)",
        "Online Logistic Regression",
        "Online GaussianNB",
    ]


# =============================================================================
# 6. COMMON PREQUENTIAL / TEST-THEN-TRAIN EXPERIMENT
# =============================================================================

def run_model_experiment(
    name: str,
    events: Sequence[dict],
    model,
    evaluator: StreamingEvaluator,
    update_callback: Callable[[dict[str, float], str], object] | None,
) -> dict:
    """Run one model under a common chronological evaluation protocol."""
    history_rows: list[dict] = []
    prediction_seconds = 0.0
    update_seconds = 0.0
    peak_bytes = 0

    total_start = time.perf_counter()
    tracemalloc.start()
    try:
        # =====================================================================
        # THIS IS THE STREAMING PART
        # =====================================================================
        for event in events:
            # 1) PREDICT using state built only from the past.
            t0 = time.perf_counter()
            y_pred = model.predict_one(event["x"])
            prediction_seconds += time.perf_counter() - t0

            # 2) TEST / EVALUATE before the current label can train anything.
            snapshot = evaluator.observe(event["y"], y_pred)

            # 3) LEARN / UPDATE only AFTER evaluation (if this model can learn online).
            t1 = time.perf_counter()
            if update_callback is not None:
                update_callback(event["x"], event["y"])
            update_seconds += time.perf_counter() - t1

            _, current_peak = tracemalloc.get_traced_memory()
            peak_bytes = max(peak_bytes, current_peak)

            # This history exists for plotting. The evaluator itself does not
            # need the complete past prediction history.
            history_rows.append(
                {
                    "timestamp": event["timestamp"],
                    "model": name,
                    **snapshot,
                }
            )
    finally:
        _, current_peak = tracemalloc.get_traced_memory()
        peak_bytes = max(peak_bytes, current_peak)
        tracemalloc.stop()

    total_seconds = max(time.perf_counter() - total_start, 1e-12)
    n_events = len(events)

    return {
        "summary": evaluator.snapshot(),
        "history": pd.DataFrame(history_rows),
        "confusion_matrix": evaluator.cumulative.as_array(),
        "timing": {
            "total_seconds": total_seconds,
            "prediction_seconds": prediction_seconds,
            "update_seconds": update_seconds,
            "events_per_second": n_events / total_seconds if n_events else 0.0,
            "mean_ms_per_event": 1000.0 * total_seconds / n_events if n_events else 0.0,
            "prediction_ms_per_event": 1000.0 * prediction_seconds / n_events if n_events else 0.0,
            "update_ms_per_event": 1000.0 * update_seconds / n_events if n_events else 0.0,
            "peak_traced_memory_mb": peak_bytes / (1024.0 * 1024.0),
        },
    }


def apply_evaluation_stress_test(
    events: Sequence[dict],
    start_fraction: float = STRESS_TEST_START_FRACTION,
    end_fraction: float = STRESS_TEST_END_FRACTION,
) -> list[dict]:
    """Temporarily swap the two class labels in a copied middle segment."""
    if not (0.0 <= start_fraction < end_fraction <= 1.0):
        raise ValueError("stress-test fractions must satisfy 0 <= start < end <= 1")

    copied = deepcopy(list(events))
    labels = sorted({event["y"] for event in copied})
    if len(labels) != 2:
        raise ValueError("The teaching stress test requires exactly two classes")

    swap = {labels[0]: labels[1], labels[1]: labels[0]}
    lo = int(len(copied) * start_fraction)
    hi = int(len(copied) * end_fraction)

    for event in copied[lo:hi]:
        event["y"] = swap[event["y"]]
    return copied


def run_week04_comparison(
    initial_training_events: Sequence[dict],
    evaluation_events: Sequence[dict],
    *,
    labels: Sequence[str],
    evaluation_window_size: int = EVALUATION_WINDOW_SIZE,
) -> tuple[dict[str, dict], dict[str, object]]:
    """Run the same future stream through four comparable strategies.

    Every model receives the same chronological warm-up. The frozen logistic
    model consumes it in one batch fit; the online learners consume the exact
    same observations one by one without scoring them. Evaluation then starts
    at the same timestamp for all strategies.
    """
    initial_rows = [(event["x"], event["y"]) for event in initial_training_events]

    online_majority = OnlineMajorityClassifier()
    for x, y in initial_rows:
        online_majority.learn_one(x, y)

    # Familiar OFFLINE model: preprocessing + logistic regression fitted once.
    frozen_logistic = FrozenBatchClassifier(
        make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=1000, random_state=RANDOM_SEED),
        )
    )
    frozen_logistic.fit_initial(initial_rows)

    # Same modelling idea, different learning regime: update after every event.
    online_logistic = OnlineLogisticRegressionAdapter(
        classes=labels,
        random_state=RANDOM_SEED,
        alpha=ONLINE_LOGISTIC_ALPHA,
    )
    for x, y in initial_rows:
        online_logistic.learn_one(x, y)

    gaussian_nb = RiverGaussianNBAdapter()
    for x, y in initial_rows:
        gaussian_nb.learn_one(x, y)

    models = {
        "Online Majority": (online_majority, online_majority.learn_one),
        "Frozen Logistic Regression (OFFLINE)": (
            frozen_logistic,
            frozen_logistic.observe_after_evaluation,
        ),
        "Online Logistic Regression": (
            online_logistic,
            online_logistic.learn_one,
        ),
        "Online GaussianNB": (gaussian_nb, gaussian_nb.learn_one),
    }

    results: dict[str, dict] = {}
    for name in model_strategy_names():
        model, updater = models[name]
        evaluator = StreamingEvaluator(labels, window_size=evaluation_window_size)
        results[name] = run_model_experiment(
            name,
            evaluation_events,
            model,
            evaluator,
            update_callback=updater,
        )

    metadata = {
        "online_warmup_events": len(initial_rows),
        "online_majority_learn_count": online_majority.learn_count,
        "online_logistic_learn_count": online_logistic.learn_count,
        "gaussian_learn_count": gaussian_nb.learn_count,
        "online_logistic_backend": online_logistic.backend,
        "gaussian_backend": gaussian_nb.backend,
    }
    return results, metadata


# =============================================================================
# 7. FIGURES — THE EVALUATION BECOMES A TIME SERIES
# =============================================================================

def _save_figure(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()
    return path


def plot_two_city_weather_profiles(
    city_frames: dict[str, pd.DataFrame],
    figures_dir: Path,
) -> Path:
    fig, axes = plt.subplots(len(FEATURES), 1, figsize=(12, 10), sharex=True)
    if len(FEATURES) == 1:
        axes = [axes]

    for ax, feature in zip(axes, FEATURES):
        for city_name in sorted(city_frames):
            frame = city_frames[city_name]
            ax.plot(frame["timestamp"], frame[feature], label=city_name, linewidth=1.2)
        ax.set_ylabel(feature)
        ax.grid(alpha=0.2)

    axes[0].legend(loc="best")
    axes[0].set_title("Two-city weather profiles — same period, multiple dimensions")
    axes[-1].set_xlabel("Time")
    fig.autofmt_xdate()
    return _save_figure(figures_dir / "08_two_city_weather_profiles.png")


def plot_cumulative_accuracy(results: dict[str, dict], figures_dir: Path) -> Path:
    plt.figure(figsize=(12, 5.5))
    for name in model_strategy_names():
        history = results[name]["history"]
        plt.plot(history["timestamp"], history["cumulative_accuracy"], label=name)
    plt.ylim(0.0, 1.02)
    plt.xlabel("Time")
    plt.ylabel("Cumulative accuracy")
    plt.title("Cumulative predictive performance over the stream")
    plt.legend(loc="best")
    plt.grid(alpha=0.2)
    plt.gcf().autofmt_xdate()
    return _save_figure(figures_dir / "09_cumulative_accuracy.png")


def plot_cumulative_vs_window_accuracy(results: dict[str, dict], figures_dir: Path) -> Path:
    history = results["Online Logistic Regression"]["history"]
    plt.figure(figsize=(12, 5.5))
    plt.plot(history["timestamp"], history["cumulative_accuracy"], label="Cumulative accuracy")
    plt.plot(history["timestamp"], history["window_accuracy"], label=f"Last-{EVALUATION_WINDOW_SIZE} accuracy")
    plt.ylim(0.0, 1.02)
    plt.xlabel("Time")
    plt.ylabel("Accuracy")
    plt.title("Online Logistic Regression — cumulative vs recent performance")
    plt.legend(loc="best")
    plt.grid(alpha=0.2)
    plt.gcf().autofmt_xdate()
    return _save_figure(figures_dir / "10_cumulative_vs_window_accuracy.png")


def plot_cumulative_vs_window_f1(results: dict[str, dict], figures_dir: Path) -> Path:
    history = results["Online Logistic Regression"]["history"]
    plt.figure(figsize=(12, 5.5))
    plt.plot(history["timestamp"], history["cumulative_macro_f1"], label="Cumulative macro-F1")
    plt.plot(history["timestamp"], history["window_macro_f1"], label=f"Last-{EVALUATION_WINDOW_SIZE} macro-F1")
    plt.ylim(0.0, 1.02)
    plt.xlabel("Time")
    plt.ylabel("Macro-F1")
    plt.title("Online Logistic Regression — cumulative vs recent macro-F1")
    plt.legend(loc="best")
    plt.grid(alpha=0.2)
    plt.gcf().autofmt_xdate()
    return _save_figure(figures_dir / "11_cumulative_vs_window_f1.png")


def plot_online_confusion_matrix(
    results: dict[str, dict],
    labels: Sequence[str],
    figures_dir: Path,
) -> Path:
    matrix = results["Online Logistic Regression"]["confusion_matrix"]
    plt.figure(figsize=(6.5, 5.5))
    plt.imshow(matrix, interpolation="nearest")
    plt.title("Online Logistic Regression — cumulative confusion matrix")
    plt.xlabel("Predicted class")
    plt.ylabel("True class")
    plt.xticks(range(len(labels)), labels)
    plt.yticks(range(len(labels)), labels)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            plt.text(j, i, str(int(matrix[i, j])), ha="center", va="center")
    plt.colorbar(label="Count")
    return _save_figure(figures_dir / "12_online_confusion_matrix.png")


def plot_model_comparison(results: dict[str, dict], figures_dir: Path) -> Path:
    names = model_strategy_names()
    accuracy = [results[name]["summary"]["cumulative_accuracy"] for name in names]
    f1 = [results[name]["summary"]["cumulative_macro_f1"] for name in names]

    x = np.arange(len(names), dtype=float)
    width = 0.36
    plt.figure(figsize=(12, 6))
    plt.bar(x - width / 2, accuracy, width, label="Accuracy")
    plt.bar(x + width / 2, f1, width, label="Macro-F1")
    plt.xticks(x, names, rotation=18, ha="right")
    plt.ylim(0.0, 1.02)
    plt.ylabel("Score")
    plt.title("Final predictive comparison under one common protocol")
    plt.legend(loc="best")
    plt.grid(axis="y", alpha=0.2)
    return _save_figure(figures_dir / "13_model_comparison.png")


def plot_processing_performance(results: dict[str, dict], figures_dir: Path) -> Path:
    names = model_strategy_names()
    prediction_ms = [results[name]["timing"]["prediction_ms_per_event"] for name in names]
    update_ms = [results[name]["timing"]["update_ms_per_event"] for name in names]

    x = np.arange(len(names), dtype=float)
    width = 0.36
    plt.figure(figsize=(12, 6))
    plt.bar(x - width / 2, prediction_ms, width, label="Prediction ms/event")
    plt.bar(x + width / 2, update_ms, width, label="Update/refit ms/event")
    plt.xticks(x, names, rotation=18, ha="right")
    plt.ylabel("Milliseconds per event")
    plt.title("Computational cost — environment dependent")
    plt.legend(loc="best")
    plt.grid(axis="y", alpha=0.2)
    return _save_figure(figures_dir / "14_processing_performance.png")


# =============================================================================
# 8. ORCHESTRATION + SCIENTIFIC SUMMARY
# =============================================================================

def _load_default_payloads(raw_dir: Path) -> dict[str, dict]:
    payloads = {}
    for city in (CITY_A, CITY_B):
        _, payload = load_city_payload(city, raw_dir=raw_dir)
        payloads[city["name"]] = payload
    return payloads


def print_scientific_summary(
    results: dict[str, dict],
    metadata: dict[str, object],
    *,
    labels: Sequence[str],
    evaluation_events: Sequence[dict],
) -> None:
    print("\n" + "=" * 78)
    print("WEEK 4 — ONLINE LEARNING & STREAM EVALUATION SUMMARY")
    print("=" * 78)
    print("Task:       identify the city from meteorological observations")
    print(f"Cities:     {' vs '.join(labels)}")
    print(f"Features:   {', '.join(FEATURES)}")
    print("Protocol:   chronological test-then-train / prequential evaluation")
    print(f"Warm-up:    {metadata['online_warmup_events']:,} observations shared by ALL models")
    print(f"Events:     {len(evaluation_events):,} evaluation observations")
    print(f"Window:     last {EVALUATION_WINDOW_SIZE} evaluated predictions")
    print(f"Online LR:  {metadata['online_logistic_backend']}")
    print(f"GaussianNB: {metadata['gaussian_backend']}")

    print("\nMODEL PERFORMANCE")
    print("-" * 78)
    for name in model_strategy_names():
        s = results[name]["summary"]
        print(f"\n{name}")
        print(f"  observations seen       : {s['observations_seen']:,}")
        print(f"  predictions evaluated   : {s['evaluated']:,}")
        print(f"  cold starts             : {s['cold_starts']:,}")
        print(f"  cumulative accuracy     : {s['cumulative_accuracy']:.4f}")
        print(f"  cumulative macro-P      : {s['cumulative_macro_precision']:.4f}")
        print(f"  cumulative macro-R      : {s['cumulative_macro_recall']:.4f}")
        print(f"  cumulative macro-F1     : {s['cumulative_macro_f1']:.4f}")
        print(f"  recent accuracy         : {s['window_accuracy']:.4f}")
        print(f"  recent macro-F1         : {s['window_macro_f1']:.4f}")

    print("\nSAME PREDICTIVE IDEA — DIFFERENT LEARNING REGIME")
    print("-" * 78)
    print("  Frozen Logistic Regression : fitted once on the shared warm-up")
    print("  Online Logistic Regression : updated after every evaluated event")
    print("  Both use a linear/logistic decision idea; their training procedures differ.")

    print("\nCOMPUTATIONAL PERFORMANCE")
    print("-" * 78)
    for name in model_strategy_names():
        t = results[name]["timing"]
        print(f"\n{name}")
        print(f"  runtime                 : {t['total_seconds']:.4f} s")
        print(f"  throughput              : {t['events_per_second']:.1f} events/s")
        print(f"  prediction              : {t['prediction_ms_per_event']:.5f} ms/event")
        print(f"  update/refit            : {t['update_ms_per_event']:.5f} ms/event")
        print(f"  peak traced memory      : {t['peak_traced_memory_mb']:.3f} MB")

    print("\nSTREAMING INTERPRETATION")
    print("-" * 78)
    print("- Every scored prediction was produced before the current label was used.")
    print("- Cumulative metrics summarize the complete predictive history.")
    print("- Window metrics describe recent predictive behaviour.")
    print("- A frozen batch model can predict one event at a time without learning online.")
    print("- Online learning is continuous updating, not a guarantee of higher accuracy.")
    print("- The frozen logistic model never changes after warm-up; the online logistic model does.")
    print("- Evaluation itself is a streaming algorithm: its state is updated incrementally.")
    print("\nTHE FUTURE MUST NEVER TRAIN THE PAST.")
    print("=" * 78)


def run_all_experiments(
    *,
    city_payloads: dict[str, dict] | None = None,
    raw_dir: Path = RAW_DIR,
    processed_dir: Path = PROCESSED_DIR,
    figures_dir: Path = FIGURES_DIR,
    print_summary: bool = True,
) -> list[Path]:
    """Run the complete Week 4 worked example and return generated figure paths."""
    ensure_output_directories(raw_dir=raw_dir, processed_dir=processed_dir, figures_dir=figures_dir)

    print("\n[1/7] Loading two-city Open-Meteo data...")
    payloads = city_payloads if city_payloads is not None else _load_default_payloads(raw_dir)

    expected_cities = [CITY_A["name"], CITY_B["name"]]
    missing = [city for city in expected_cities if city not in payloads]
    if missing:
        raise ValueError(f"Missing city payload(s): {missing}")

    print("[2/7] Preparing clean multivariate city frames...")
    city_frames: dict[str, pd.DataFrame] = {}
    for city_name in expected_cities:
        frame = prepare_city_frame(payloads[city_name], city_name, FEATURES)
        city_frames[city_name] = frame
        out_csv = processed_dir / build_processed_filename(city_name, START_DATE, END_DATE)
        frame.to_csv(out_csv, index=False)
        print(f"      {city_name}: {len(frame):,} clean rows -> {out_csv.name}")

    print("[3/7] Building chronological replay stream...")
    events = build_two_city_events(city_frames, FEATURES)
    initial_train, evaluation_events = split_initial_training(events, INITIAL_TRAIN_FRACTION)
    print(f"      Shared initial warm-up: {len(initial_train):,} events")
    print(f"      Streaming evaluation : {len(evaluation_events):,} events")

    if ENABLE_EVALUATION_STRESS_TEST:
        print("      Applying controlled PERFORMANCE-CHANGE teaching experiment to a COPY.")
        evaluation_events = apply_evaluation_stress_test(
            evaluation_events,
            start_fraction=STRESS_TEST_START_FRACTION,
            end_fraction=STRESS_TEST_END_FRACTION,
        )

    labels = sorted(expected_cities)

    print("[4/7] Running four strategies under one common evaluation protocol...")
    results, metadata = run_week04_comparison(
        initial_train,
        evaluation_events,
        labels=labels,
        evaluation_window_size=EVALUATION_WINDOW_SIZE,
    )

    print("[5/7] Generating Week 4 figures...")
    paths = [
        plot_two_city_weather_profiles(city_frames, figures_dir),
        plot_cumulative_accuracy(results, figures_dir),
        plot_cumulative_vs_window_accuracy(results, figures_dir),
        plot_cumulative_vs_window_f1(results, figures_dir),
        plot_online_confusion_matrix(results, labels, figures_dir),
        plot_model_comparison(results, figures_dir),
        plot_processing_performance(results, figures_dir),
    ]

    print("[6/7] Figures created:")
    for path in paths:
        print(f"      {path.name}")

    if print_summary:
        print("[7/7] Scientific summary")
        print_scientific_summary(
            results,
            metadata,
            labels=labels,
            evaluation_events=evaluation_events,
        )

    return paths


def main() -> None:
    np.random.seed(RANDOM_SEED)
    run_all_experiments()


if __name__ == "__main__":
    main()
