"""Weekend Challenge 04 — Instructor Solution / Worked Example.

Data Stream Mining — NOVA IMS
Week 5 — Incremental Classification Algorithms

Challenge idea
--------------
Extend Weekend Challenge 03 instead of starting over. Keep the same real-world
Open-Meteo replay stream and the same chronological prequential protocol, but
replace the simple Week 4 comparison with learners that evolve in different
ways:

1. GaussianNB                         -> statistical parameters evolve
2. Online Logistic Regression        -> numerical parameters evolve
3. Hoeffding Tree                     -> parameters + tree structure evolve
4. Public ENFS-Uni0 implementation    -> parameters + fuzzy-rule structure evolve

The scientific question is not "which model wins?". It is:

    WHAT CHANGES INSIDE EACH LEARNER WHILE IT LEARNS?

Core streaming rule
-------------------
    PREDICT -> EVALUATE -> LEARN

The current label never influences the prediction being scored for that same
observation.

Important ENFS note
-------------------
When the ``evolvingfuzzysystems`` package is installed, this script uses its
public ``ENFS_Uni0`` class. Treat that result as the behavior of the public
Python implementation used here; do not assume it is mathematically identical
to every detail of the original published ENFS-Uni0 formulation.

If River or evolvingfuzzysystems are unavailable, transparent educational
fallbacks are used so the file can still run. The script prints the active
backend at startup. For the intended course environment install:

    python -m pip install river evolvingfuzzysystems

Run
---
    python Weekend_Challenge_04_Solution.py

The script saves figures 15–22, continuing the numbering from Challenge 03.
"""

from __future__ import annotations

import importlib.metadata as importlib_metadata
import importlib.util
import json
import math
import re
import time
import warnings
from collections import Counter, deque
from pathlib import Path
from typing import Callable, Sequence

import matplotlib

# This script saves figures to disk and does not need an interactive GUI.
# Force a non-interactive backend before importing pyplot so Windows does not
# allocate Tk image buffers for every figure (which can fail with TclError).
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import GaussianNB as SkGaussianNB
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier


# =============================================================================
# 1. STUDENT CUSTOMIZATION ZONE — CHANGE HERE
# =============================================================================

CITY_A = {"name": "Lisbon", "latitude": 38.7223, "longitude": -9.1393}
CITY_B = {"name": "Porto", "latitude": 41.1579, "longitude": -8.6291}
LABEL_NAMES = {0: CITY_A["name"], 1: CITY_B["name"]}
CITY_TO_LABEL = {name: label for label, name in LABEL_NAMES.items()}

# A slightly longer period than Challenge 03 gives structural learners more
# opportunities to accumulate evidence and evolve.
START_DATE = "2025-06-01"
END_DATE = "2025-09-07"
TIMEZONE = "Europe/Lisbon"

FEATURES = [
    "temperature_2m",
    "relative_humidity_2m",
    "precipitation",
    "wind_speed_10m",
]
SMALL_FEATURES = ["temperature_2m", "relative_humidity_2m"]

INITIAL_TRAIN_FRACTION = 0.20
EVALUATION_WINDOW_SIZE = 200
RANDOM_SEED = 42
FORCE_DOWNLOAD = False

# Structural-model controls. These are teaching defaults, not universal optima.
HOEFFDING_GRACE_PERIOD = 50
HOEFFDING_DELTA = 1e-3
HOEFFDING_TAU = 0.10
ENFS_LAMBDA_FF = 0.99
ENFS_SIM_THRESHOLD = 0.90
ENFS_MAX_RULES = 60

BASE_DIR = Path(__file__).resolve().parent
RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
FIGURES_DIR = BASE_DIR / "figures_week05"
OPEN_METEO_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"


# =============================================================================
# 2. DEPENDENCY DISCOVERY
# =============================================================================

RIVER_AVAILABLE = importlib.util.find_spec("river") is not None
EFS_AVAILABLE = importlib.util.find_spec("evolvingfuzzysystems") is not None

if RIVER_AVAILABLE:
    from river import linear_model, naive_bayes, optim, tree  # type: ignore

if EFS_AVAILABLE:
    from evolvingfuzzysystems.classification import ENFS_Uni0  # type: ignore


def package_version(name: str) -> str:
    try:
        return importlib_metadata.version(name)
    except importlib_metadata.PackageNotFoundError:
        return "not installed"


# =============================================================================
# 3. OPEN-METEO ACQUISITION AND REPLAY
# =============================================================================

def slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def build_raw_filename(city_name: str, start_date: str, end_date: str) -> str:
    return f"open_meteo_{slugify(city_name)}_{start_date}_{end_date}.json"


def build_processed_filename(city_name: str, start_date: str, end_date: str) -> str:
    return f"open_meteo_{slugify(city_name)}_{start_date}_{end_date}_week05.csv"


def ensure_output_directories(raw_dir=RAW_DIR, processed_dir=PROCESSED_DIR, figures_dir=FIGURES_DIR):
    Path(raw_dir).mkdir(parents=True, exist_ok=True)
    Path(processed_dir).mkdir(parents=True, exist_ok=True)
    Path(figures_dir).mkdir(parents=True, exist_ok=True)


def build_open_meteo_params(city: dict) -> dict:
    return {
        "latitude": float(city["latitude"]),
        "longitude": float(city["longitude"]),
        "start_date": START_DATE,
        "end_date": END_DATE,
        "hourly": ",".join(FEATURES),
        "timezone": TIMEZONE,
    }


def download_open_meteo(city: dict) -> dict:
    print(f"      Downloading {city['name']} ({START_DATE} -> {END_DATE})...")
    try:
        response = requests.get(OPEN_METEO_ARCHIVE_URL, params=build_open_meteo_params(city), timeout=45)
        response.raise_for_status()
        payload = response.json()
    except requests.RequestException as exc:
        raise RuntimeError(
            "Could not download Open-Meteo data. Check the connection or reuse a saved raw snapshot "
            "with FORCE_DOWNLOAD=False."
        ) from exc
    if isinstance(payload, dict) and payload.get("error"):
        raise RuntimeError(f"Open-Meteo API error: {payload.get('reason', 'unknown reason')}")
    return payload


def load_or_download_raw(raw_path: Path, downloader: Callable[[], dict], force_download=False) -> dict:
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
    path = Path(raw_dir) / build_raw_filename(city["name"], START_DATE, END_DATE)
    payload = load_or_download_raw(path, lambda: download_open_meteo(city), FORCE_DOWNLOAD)
    return path, payload


def prepare_city_frame(payload: dict, city_name: str, features: Sequence[str]) -> pd.DataFrame:
    hourly = payload.get("hourly", {})
    if "time" not in hourly:
        raise ValueError("Open-Meteo payload is missing hourly.time")
    missing = [f for f in features if f not in hourly]
    if missing:
        raise ValueError(f"Open-Meteo payload is missing required features: {missing}")

    data = {"timestamp": pd.to_datetime(hourly["time"], errors="coerce")}
    for feature in features:
        data[feature] = pd.to_numeric(pd.Series(hourly[feature]), errors="coerce")
    df = pd.DataFrame(data)
    df["city"] = city_name
    df = df.dropna(subset=["timestamp", *features]).copy()
    if city_name not in CITY_TO_LABEL:
        raise ValueError(f"Unknown city label: {city_name}")
    df["y"] = CITY_TO_LABEL[city_name]
    return df[["timestamp", *features, "city", "y"]].reset_index(drop=True)


def build_two_city_events(city_frames: dict[str, pd.DataFrame], features: Sequence[str]) -> list[dict]:
    frames = []
    for city_name, frame in city_frames.items():
        local = frame.copy()
        local["city"] = city_name
        local["y"] = CITY_TO_LABEL[city_name]
        frames.append(local[["timestamp", *features, "city", "y"]])
    combined = pd.concat(frames, ignore_index=True).sort_values(["timestamp", "city"], kind="mergesort")
    return [
        {
            "timestamp": row.timestamp,
            "x": {f: float(getattr(row, f)) for f in features},
            "y": int(row.y),
            "city": str(row.city),
        }
        for row in combined.itertuples(index=False)
    ]


def split_initial_training(events: Sequence[dict], train_fraction: float) -> tuple[list[dict], list[dict]]:
    if not (0 < train_fraction < 1):
        raise ValueError("train_fraction must be between 0 and 1")
    timestamps = sorted({e["timestamp"] for e in events})
    if len(timestamps) < 2:
        raise ValueError("At least two timestamps are required")
    cut = max(1, min(len(timestamps)-1, int(len(timestamps)*train_fraction)))
    warm_ts = set(timestamps[:cut])
    warm = [e for e in events if e["timestamp"] in warm_ts]
    evaluation = [e for e in events if e["timestamp"] not in warm_ts]
    if {e["y"] for e in warm} != {0, 1}:
        raise ValueError("Warm-up must contain both city classes")
    return warm, evaluation


def fit_shared_scaler(warm_events: Sequence[dict], feature_names: Sequence[str]) -> MinMaxScaler:
    X = np.asarray([[e["x"][f] for f in feature_names] for e in warm_events], dtype=float)
    return MinMaxScaler(feature_range=(0.0, 1.0), clip=True).fit(X)


def transform_events(events: Sequence[dict], scaler: MinMaxScaler, feature_names: Sequence[str]) -> list[dict]:
    names = list(feature_names)
    X = np.asarray([[e["x"][f] for f in names] for e in events], dtype=float)
    Xt = scaler.transform(X)
    out = []
    for e, row in zip(events, Xt):
        out.append({
            "timestamp": e["timestamp"],
            "x": {f"x{i}": float(row[i]) for i in range(len(names))},
            "y": int(e["y"]),
            "city": e["city"],
        })
    return out


# =============================================================================
# 4. INCREMENTAL METRICS
# =============================================================================

def _metrics_from_counts(tp, tn, fp, fn):
    total = tp + tn + fp + fn
    accuracy = (tp + tn) / total if total else 0.0
    p1 = tp / (tp + fp) if (tp + fp) else 0.0
    r1 = tp / (tp + fn) if (tp + fn) else 0.0
    f1_1 = 2*p1*r1/(p1+r1) if (p1+r1) else 0.0
    p0 = tn / (tn + fn) if (tn + fn) else 0.0
    r0 = tn / (tn + fp) if (tn + fp) else 0.0
    f1_0 = 2*p0*r0/(p0+r0) if (p0+r0) else 0.0
    return float(accuracy), float((f1_0+f1_1)/2)


class StreamingBinaryEvaluator:
    def __init__(self, window_size=EVALUATION_WINDOW_SIZE):
        self.tp = self.tn = self.fp = self.fn = 0
        self.cold_starts = 0
        self.recent = deque(maxlen=int(window_size))

    def update(self, y_true: int, y_pred: int | None):
        if y_pred is None:
            self.cold_starts += 1
            return self.snapshot()
        y_true, y_pred = int(y_true), int(y_pred)
        if y_true == 1 and y_pred == 1: self.tp += 1
        elif y_true == 0 and y_pred == 0: self.tn += 1
        elif y_true == 0 and y_pred == 1: self.fp += 1
        elif y_true == 1 and y_pred == 0: self.fn += 1
        self.recent.append((y_true, y_pred))
        return self.snapshot()

    def snapshot(self):
        accuracy, macro_f1 = _metrics_from_counts(self.tp, self.tn, self.fp, self.fn)
        rtp = rtn = rfp = rfn = 0
        for yt, yp in self.recent:
            if yt == 1 and yp == 1: rtp += 1
            elif yt == 0 and yp == 0: rtn += 1
            elif yt == 0 and yp == 1: rfp += 1
            elif yt == 1 and yp == 0: rfn += 1
        wacc, wf1 = _metrics_from_counts(rtp, rtn, rfp, rfn)
        return {
            "accuracy": accuracy,
            "macro_f1": macro_f1,
            "window_accuracy": wacc,
            "window_macro_f1": wf1,
            "evaluated": self.tp+self.tn+self.fp+self.fn,
            "cold_starts": self.cold_starts,
        }


# =============================================================================
# 5. MODEL ADAPTERS
# =============================================================================

class SkIncrementalAdapter:
    def __init__(self, estimator, n_features):
        self.estimator = estimator
        self.n_features = int(n_features)
        self.initialized = False
        self.observed_classes = set()

    def _row(self, x):
        return np.array([[x[f"x{i}"] for i in range(self.n_features)]], dtype=float)

    def predict_one(self, x):
        if not self.initialized:
            return None
        if len(self.observed_classes) == 1:
            return int(next(iter(self.observed_classes)))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return int(self.estimator.predict(self._row(x))[0])

    def learn_one(self, x, y):
        X, yy = self._row(x), np.array([int(y)])
        self.observed_classes.add(int(y))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if not self.initialized:
                self.estimator.partial_fit(X, yy, classes=np.array([0, 1]))
                self.initialized = True
            else:
                self.estimator.partial_fit(X, yy)
        return self

    @property
    def weights(self):
        if self.initialized and hasattr(self.estimator, "coef_"):
            row = np.asarray(self.estimator.coef_)[0]
            return {f"x{i}": float(row[i]) for i in range(self.n_features)}
        return {}

    @property
    def intercept(self):
        if self.initialized and hasattr(self.estimator, "intercept_"):
            return float(np.asarray(self.estimator.intercept_).ravel()[0])
        return 0.0


class FallbackGrowingTree:
    """Educational structural-growth fallback; NOT a Hoeffding/VFDT implementation."""
    def __init__(self, n_features, grace_period=50, seed=42, max_buffer=1200):
        self.n_features = int(n_features)
        self.grace_period = int(grace_period)
        self.seed = int(seed)
        self.max_buffer = int(max_buffer)
        self.X, self.y = [], []
        self.seen = 0
        self.fitted = False
        self.model = DecisionTreeClassifier(max_depth=1, random_state=seed)
        self.n_nodes = 1
        self.n_leaves = 1
        self.height = 0

    def _vec(self, x): return [x[f"x{i}"] for i in range(self.n_features)]

    def predict_one(self, x):
        if not self.fitted:
            return None
        return int(self.model.predict(np.array([self._vec(x)]))[0])

    def learn_one(self, x, y):
        self.X.append(self._vec(x)); self.y.append(int(y)); self.seen += 1
        if len(self.X) > self.max_buffer:
            self.X = self.X[-self.max_buffer:]; self.y = self.y[-self.max_buffer:]
        if self.seen >= self.grace_period and self.seen % self.grace_period == 0 and len(set(self.y)) > 1:
            depth = min(7, 1 + self.seen // (4*self.grace_period))
            self.model = DecisionTreeClassifier(max_depth=depth, min_samples_leaf=8, random_state=self.seed)
            self.model.fit(np.array(self.X), np.array(self.y))
            self.fitted = True
            tr = self.model.tree_
            self.n_nodes = int(tr.node_count)
            self.n_leaves = int(np.sum(tr.children_left == -1))
            self.height = int(self.model.get_depth())
        return self


class ENFSAdapter:
    def __init__(self, model, n_features):
        self.model = model; self.n_features = int(n_features)
    def _vec(self, x): return np.array([x[f"x{i}"] for i in range(self.n_features)], dtype=float)
    def predict_one(self, x):
        pred = self.model.predict_one(self._vec(x))
        return None if pred is None else int(pred)
    def learn_one(self, x, y):
        self.model.learn_one(self._vec(x), int(y)); return self
    def __getattr__(self, name): return getattr(self.model, name)


class FallbackENFS:
    """Educational evolving-rule fallback; NOT the ENFS-Uni0 algorithm."""
    def __init__(self, n_features, n_classes=2, lambda_ff=0.99, sim_threshold=0.90,
                 max_rules=30, random_state=42):
        self.n_features = int(n_features); self.n_classes = int(n_classes)
        self.lambda_ff = float(lambda_ff); self.sim_threshold = float(sim_threshold)
        self.max_rules = int(max_rules); self.rng = np.random.default_rng(random_state)
        self.rules = []; self.history_n_rules = []

    @property
    def n_rules_(self): return len(self.rules)

    def _vec(self, x):
        if isinstance(x, dict): return np.array([x[f"x{i}"] for i in range(self.n_features)], dtype=float)
        return np.asarray(x, dtype=float).ravel()

    def predict_one(self, x):
        if not self.rules: return None
        v = self._vec(x)
        j = int(np.argmin([np.linalg.norm(v-r["center"]) for r in self.rules]))
        return int(np.argmax(self.rules[j]["class_counts"]))

    def learn_one(self, x, y):
        v, y = self._vec(x), int(y)
        if not self.rules:
            self.rules.append({"center": v.copy(), "support": 1,
                               "class_counts": np.bincount([y], minlength=self.n_classes).astype(float)})
        else:
            d = np.array([np.linalg.norm(v-r["center"]) for r in self.rules])
            j = int(np.argmin(d))
            threshold = 0.16 + 0.12*(1-self.sim_threshold)
            if d[j] > threshold and len(self.rules) < self.max_rules:
                self.rules.append({"center": v.copy(), "support": 1,
                                   "class_counts": np.bincount([y], minlength=self.n_classes).astype(float)})
            else:
                r = self.rules[j]; r["support"] += 1
                eta = max(1-self.lambda_ff, 1/r["support"])
                r["center"] = (1-eta)*r["center"] + eta*v
                r["class_counts"][y] += 1
        self.history_n_rules.append(self.n_rules_)
        return self


def make_gaussian_nb(n_features):
    if RIVER_AVAILABLE:
        model = naive_bayes.GaussianNB(); model.backend = "river.naive_bayes.GaussianNB"; return model
    model = SkIncrementalAdapter(SkGaussianNB(), n_features); model.backend = "sklearn GaussianNB.partial_fit fallback"; return model


def make_logistic(n_features):
    if RIVER_AVAILABLE:
        model = linear_model.LogisticRegression(optimizer=optim.SGD(0.05)); model.backend = "river.linear_model.LogisticRegression"; return model
    model = SkIncrementalAdapter(
        SGDClassifier(loss="log_loss", learning_rate="constant", eta0=0.05,
                      random_state=RANDOM_SEED, tol=None), n_features)
    model.backend = "sklearn SGDClassifier(log_loss) fallback"; return model


def make_tree(n_features):
    if RIVER_AVAILABLE:
        model = tree.HoeffdingTreeClassifier(
            grace_period=HOEFFDING_GRACE_PERIOD,
            delta=HOEFFDING_DELTA,
            tau=HOEFFDING_TAU,
            leaf_prediction="mc",
        )
        model.backend = "river.tree.HoeffdingTreeClassifier(mc leaves)"; return model
    model = FallbackGrowingTree(n_features, HOEFFDING_GRACE_PERIOD, RANDOM_SEED)
    model.backend = "educational growing-tree fallback (NOT VFDT)"; return model


def make_enfs(n_features):
    if EFS_AVAILABLE:
        model = ENFS_Uni0(
            n_features=n_features, n_classes=2, lambda_ff=ENFS_LAMBDA_FF,
            sim_threshold=ENFS_SIM_THRESHOLD, max_rules=ENFS_MAX_RULES,
            random_state=RANDOM_SEED,
        )
        adapter = ENFSAdapter(model, n_features)
        adapter.backend = f"evolvingfuzzysystems {package_version('evolvingfuzzysystems')} public ENFS_Uni0"
        return adapter
    model = FallbackENFS(n_features=n_features, n_classes=2, lambda_ff=ENFS_LAMBDA_FF,
                         sim_threshold=ENFS_SIM_THRESHOLD, max_rules=ENFS_MAX_RULES,
                         random_state=RANDOM_SEED)
    model.backend = "educational evolving-rule fallback (NOT ENFS-Uni0)"; return model


def model_strategy_names():
    return ["GaussianNB", "Online Logistic Regression", "Hoeffding Tree", "Public ENFS-Uni0 implementation"]


def required_figure_names():
    return [
        "15_incremental_learning_curves.png",
        "16_macro_f1_learning_curves.png",
        "17_gaussiannb_state_evolution.png",
        "18_logistic_parameter_evolution.png",
        "19_hoeffding_structure_evolution.png",
        "20_fuzzy_rule_evolution.png",
        "21_feature_representation_comparison.png",
        "22_performance_cost_complexity_summary.png",
    ]


def tree_complexity(model):
    out = {}
    for name in ("n_nodes", "n_leaves", "height"):
        value = getattr(model, name, 0)
        try:
            value = value() if callable(value) else value
        except Exception:
            value = 0
        out[name] = int(value or 0)
    return out


def rule_count(model):
    for name in ("n_rules_", "n_rules", "num_rules"):
        if hasattr(model, name):
            value = getattr(model, name)
            try: value = value() if callable(value) else value
            except Exception: continue
            return int(value)
    history = getattr(model, "history_n_rules", [])
    return int(history[-1]) if history else 0


def logistic_weights(model, n_features):
    weights = getattr(model, "weights", {})
    if isinstance(weights, dict):
        return np.array([float(weights.get(f"x{i}", 0.0)) for i in range(n_features)]), float(getattr(model, "intercept", 0.0))
    return np.zeros(n_features), 0.0


class RunningClassGaussianState:
    def __init__(self, n_features):
        self.n_features = int(n_features)
        self.count = {0: 0, 1: 0}
        self.mean = {0: np.zeros(n_features), 1: np.zeros(n_features)}
        self.M2 = {0: np.zeros(n_features), 1: np.zeros(n_features)}
    def update(self, x, y):
        v = np.array([x[f"x{i}"] for i in range(self.n_features)]); y = int(y)
        self.count[y] += 1; n = self.count[y]
        delta = v-self.mean[y]; self.mean[y] += delta/n; self.M2[y] += delta*(v-self.mean[y])
    def snapshot(self):
        return {y: {"count": self.count[y], "mean": self.mean[y].copy(),
                    "variance": self.M2[y]/self.count[y] if self.count[y] else np.zeros(self.n_features)} for y in (0,1)}


# =============================================================================
# 6. PREQUENTIAL EXPERIMENTS
# =============================================================================

def run_prequential(events, model, labels=(0,1), track_state=None):
    evaluator = StreamingBinaryEvaluator(EVALUATION_WINDOW_SIZE)
    history = []; predictions = []; pred_s = upd_s = 0.0
    for index, event in enumerate(events, start=1):
        t0 = time.perf_counter(); y_pred = model.predict_one(event["x"]); pred_s += time.perf_counter()-t0
        predictions.append(None if y_pred is None else int(y_pred))
        snapshot = evaluator.update(event["y"], y_pred)  # EVALUATE BEFORE LEARN
        t1 = time.perf_counter(); model.learn_one(event["x"], event["y"]); upd_s += time.perf_counter()-t1
        state = track_state(index, model, event) if track_state else None
        history.append({"i": index, "timestamp": event["timestamp"], **snapshot})
    return {
        "history": pd.DataFrame(history),
        "summary": evaluator.snapshot(),
        "predictions": predictions,
        "prediction_seconds": pred_s,
        "update_seconds": upd_s,
        "model": model,
    }


def warm_up_model(model, events):
    for event in events:
        model.learn_one(event["x"], event["y"])
    return model


def run_model_suite(warm, evaluation, n_features):
    models = {
        "GaussianNB": make_gaussian_nb(n_features),
        "Online Logistic Regression": make_logistic(n_features),
        "Hoeffding Tree": make_tree(n_features),
        "Public ENFS-Uni0 implementation": make_enfs(n_features),
    }
    results = {}
    for name in model_strategy_names():
        model = warm_up_model(models[name], warm)
        results[name] = run_prequential(evaluation, model)
    return results


# =============================================================================
# 7. FIGURE HELPERS
# =============================================================================

def _save(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout(); plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    return path


def plot_learning_curves(results, figures_dir):
    path = Path(figures_dir)/"15_incremental_learning_curves.png"
    plt.figure(figsize=(11,5))
    for name, result in results.items():
        h=result["history"]; plt.plot(h["timestamp"], h["accuracy"], label=name)
    plt.xlabel("Time"); plt.ylabel("Cumulative accuracy"); plt.ylim(0,1.02)
    plt.title("Incremental learners under one common prequential protocol"); plt.legend(fontsize=8)
    return _save(path)


def plot_f1_curves(results, figures_dir):
    path=Path(figures_dir)/"16_macro_f1_learning_curves.png"
    plt.figure(figsize=(11,5))
    for name,result in results.items():
        h=result["history"]; plt.plot(h["timestamp"],h["macro_f1"],label=name)
    plt.xlabel("Time"); plt.ylabel("Cumulative Macro-F1"); plt.ylim(0,1.02)
    plt.title("Macro-F1 learning curves"); plt.legend(fontsize=8)
    return _save(path)


def plot_gaussian_state(warm, evaluation, figures_dir, n_features):
    path=Path(figures_dir)/"17_gaussiannb_state_evolution.png"
    state=RunningClassGaussianState(n_features)
    for e in warm: state.update(e["x"],e["y"])
    checkpoints=np.unique(np.linspace(1,max(2,len(evaluation)),6,dtype=int)).tolist()
    rows=[]
    for k,e in enumerate(evaluation,start=1):
        state.update(e["x"],e["y"])
        if k in checkpoints:
            snap=state.snapshot()
            for cls in (0,1): rows.append({"n":k,"class":LABEL_NAMES[cls],"mean":snap[cls]["mean"][0],"variance":snap[cls]["variance"][0]})
    df=pd.DataFrame(rows)
    plt.figure(figsize=(9,5))
    for cls,part in df.groupby("class"):
        plt.plot(part["n"],part["mean"],marker="o",label=f"{cls}: mean(feature 1)")
        plt.plot(part["n"],part["variance"],marker="s",linestyle="--",label=f"{cls}: variance(feature 1)")
    plt.xlabel("Evaluation observations processed"); plt.ylabel("Scaled statistic")
    plt.title("GaussianNB: sufficient-statistics evolution"); plt.legend(fontsize=8)
    return _save(path)


def plot_logistic_state(warm,evaluation,figures_dir,n_features):
    path=Path(figures_dir)/"18_logistic_parameter_evolution.png"
    model=warm_up_model(make_logistic(n_features),warm)
    checkpoints=set(np.unique(np.linspace(1,max(2,len(evaluation)),8,dtype=int)).tolist())
    rows=[]
    for k,e in enumerate(evaluation,start=1):
        _=model.predict_one(e["x"]); model.learn_one(e["x"],e["y"])
        if k in checkpoints:
            w,b=logistic_weights(model,n_features)
            row={"n":k,"bias":b}; row.update({f"w{i}":w[i] for i in range(n_features)}); rows.append(row)
    df=pd.DataFrame(rows)
    plt.figure(figsize=(9,5))
    for i in range(min(n_features,4)): plt.plot(df["n"],df[f"w{i}"],marker="o",label=f"weight x{i}")
    plt.plot(df["n"],df["bias"],linestyle="--",label="bias")
    plt.xlabel("Evaluation observations processed"); plt.ylabel("Parameter value")
    plt.title("Online Logistic Regression: parameter evolution"); plt.legend(fontsize=8)
    return _save(path)


def plot_tree_growth(warm,evaluation,figures_dir,n_features):
    path=Path(figures_dir)/"19_hoeffding_structure_evolution.png"
    all_events = list(warm) + list(evaluation)
    model=make_tree(n_features)
    checkpoints=set(np.unique(np.linspace(1,max(2,len(all_events)),10,dtype=int)).tolist())
    rows=[]
    for k,e in enumerate(all_events,start=1):
        _=model.predict_one(e["x"]); model.learn_one(e["x"],e["y"])
        if k in checkpoints: rows.append({"n":k,**tree_complexity(model)})
    df=pd.DataFrame(rows)
    plt.figure(figsize=(9,5))
    for col in ("n_nodes","n_leaves","height"):
        plt.plot(df["n"],df[col],marker="o",label=col.replace("n_",""))
    plt.xlabel("Chronological observations processed"); plt.ylabel("Structural complexity")
    plt.title("Hoeffding Tree: architecture evolution"); plt.legend()
    return _save(path)


def plot_fuzzy_growth(warm,evaluation,figures_dir,n_features):
    path=Path(figures_dir)/"20_fuzzy_rule_evolution.png"
    all_events = list(warm) + list(evaluation)
    model=make_enfs(n_features)
    rows=[]
    for k,e in enumerate(all_events,start=1):
        _=model.predict_one(e["x"]); model.learn_one(e["x"],e["y"])
        rows.append({"n":k,"rules":rule_count(model)})
    df=pd.DataFrame(rows)
    plt.figure(figsize=(9,5)); plt.plot(df["n"],df["rules"])
    plt.xlabel("Chronological observations processed"); plt.ylabel("Number of fuzzy rules")
    plt.title("Public ENFS-Uni0 implementation: rule-base evolution")
    return _save(path)


def _summary_table(results, n_features):
    rows=[]
    for name,result in results.items():
        model=result["model"]; s=result["summary"]
        if name=="GaussianNB": complexity=4*n_features+2; complexity_label=f"{complexity} statistics"
        elif name=="Online Logistic Regression": complexity=n_features+1; complexity_label=f"{complexity} coefficients"
        elif name=="Hoeffding Tree": complexity=tree_complexity(model)["n_nodes"]; complexity_label=f"{complexity} nodes"
        else: complexity=rule_count(model); complexity_label=f"{complexity} rules"
        rows.append({
            "model":name,"accuracy":s["accuracy"],"macro_f1":s["macro_f1"],
            "prediction_ms/event":1000*result["prediction_seconds"]/max(1,s["evaluated"]),
            "update_ms/event":1000*result["update_seconds"]/max(1,s["evaluated"]),
            "complexity":complexity_label,
            "backend":getattr(model,"backend",model.__class__.__name__),
        })
    return pd.DataFrame(rows)


def plot_representation_comparison(payloads,figures_dir,processed_dir):
    rows=[]
    for representation,features in [("small: temp + humidity",SMALL_FEATURES),("full: 4 weather features",FEATURES)]:
        frames={city:prepare_city_frame(payload,city,FEATURES) for city,payload in payloads.items()}
        events=build_two_city_events(frames,features)
        warm,ev=split_initial_training(events,INITIAL_TRAIN_FRACTION)
        scaler=fit_shared_scaler(warm,features); warm_t=transform_events(warm,scaler,features); ev_t=transform_events(ev,scaler,features)
        results=run_model_suite(warm_t,ev_t,len(features))
        for name,result in results.items(): rows.append({"representation":representation,"model":name,"accuracy":result["summary"]["accuracy"]})
    df=pd.DataFrame(rows); Path(processed_dir).mkdir(parents=True,exist_ok=True); df.to_csv(Path(processed_dir)/"week05_feature_representation_summary.csv",index=False)
    path=Path(figures_dir)/"21_feature_representation_comparison.png"
    models=model_strategy_names(); x=np.arange(len(models)); width=0.36
    small=df[df.representation.str.startswith("small")].set_index("model").loc[models,"accuracy"]
    full=df[df.representation.str.startswith("full")].set_index("model").loc[models,"accuracy"]
    plt.figure(figsize=(11,5)); plt.bar(x-width/2,small,width,label="2 features"); plt.bar(x+width/2,full,width,label="4 features")
    plt.xticks(x,models,rotation=15,ha="right"); plt.ylabel("Final prequential accuracy"); plt.ylim(0,1)
    plt.title("Feature representation: more inputs do not guarantee the same effect for every learner"); plt.legend()
    return _save(path),df


def plot_summary_table(summary_df,figures_dir):
    path=Path(figures_dir)/"22_performance_cost_complexity_summary.png"
    display_df=summary_df[["model","accuracy","macro_f1","prediction_ms/event","update_ms/event","complexity"]].copy()
    display_df["accuracy"]=display_df["accuracy"].map(lambda v:f"{v:.3f}")
    display_df["macro_f1"]=display_df["macro_f1"].map(lambda v:f"{v:.3f}")
    display_df["prediction_ms/event"]=display_df["prediction_ms/event"].map(lambda v:f"{v:.3f}")
    display_df["update_ms/event"]=display_df["update_ms/event"].map(lambda v:f"{v:.3f}")
    fig,ax=plt.subplots(figsize=(12,3.8)); ax.axis("off")
    table=ax.table(cellText=display_df.values,colLabels=["Model","Accuracy","Macro-F1","Pred ms/event","Update ms/event","Complexity"],loc="center",cellLoc="center")
    table.auto_set_font_size(False); table.set_fontsize(8); table.scale(1,1.5)
    ax.set_title("Performance + computational cost + model complexity",pad=12)
    return _save(path)


# =============================================================================
# 8. ORCHESTRATION
# =============================================================================

def _load_default_payloads(raw_dir):
    payloads={}
    for city in (CITY_A,CITY_B):
        _,payload=load_city_payload(city,Path(raw_dir)); payloads[city["name"]]=payload
    return payloads


def print_scientific_summary(results,summary_df,warm,evaluation,feature_names):
    print("\n"+"="*84)
    print("WEEKEND CHALLENGE 04 — EVOLVE THE LEARNER")
    print("="*84)
    print(f"Period: {START_DATE} -> {END_DATE}")
    print(f"Features: {', '.join(feature_names)}")
    print(f"Shared warm-up: {len(warm):,} observations")
    print(f"Scored evaluation stream: {len(evaluation):,} observations")
    print(f"River: {package_version('river')}")
    print(f"evolvingfuzzysystems: {package_version('evolvingfuzzysystems')}")
    print("\nBACKENDS")
    for row in summary_df.itertuples(index=False): print(f"  {row.model}: {row.backend}")
    print("\nFINAL RESULTS")
    print(summary_df[["model","accuracy","macro_f1","prediction_ms/event","update_ms/event","complexity"]].to_string(index=False))
    print("\nINTERPRETATION")
    print("- Do not ask only which model won.")
    print("- Ask what changed inside each learner: statistics, parameters, nodes, or rules.")
    print("- Compare prediction quality, update cost, and architectural complexity together.")
    print("- Public ENFS-Uni0 results describe the public Python implementation used here.")
    print("\nTHE STREAM CHANGES THE MODEL — ONE OBSERVATION AT A TIME.")


def run_all_experiments(*,city_payloads=None,figures_dir=FIGURES_DIR,processed_dir=PROCESSED_DIR,raw_dir=RAW_DIR,print_summary=True):
    figures_dir=Path(figures_dir); processed_dir=Path(processed_dir); raw_dir=Path(raw_dir)
    ensure_output_directories(raw_dir,processed_dir,figures_dir)
    payloads=city_payloads if city_payloads is not None else _load_default_payloads(raw_dir)

    frames={city:prepare_city_frame(payload,city,FEATURES) for city,payload in payloads.items()}
    for city,frame in frames.items(): frame.to_csv(processed_dir/build_processed_filename(city,START_DATE,END_DATE),index=False)

    events=build_two_city_events(frames,FEATURES)
    warm,evaluation=split_initial_training(events,INITIAL_TRAIN_FRACTION)
    scaler=fit_shared_scaler(warm,FEATURES)
    warm_t=transform_events(warm,scaler,FEATURES); evaluation_t=transform_events(evaluation,scaler,FEATURES)
    results=run_model_suite(warm_t,evaluation_t,len(FEATURES))
    summary_df=_summary_table(results,len(FEATURES)); summary_df.to_csv(processed_dir/"week05_model_summary.csv",index=False)

    paths=[]
    paths.append(plot_learning_curves(results,figures_dir))
    paths.append(plot_f1_curves(results,figures_dir))
    paths.append(plot_gaussian_state(warm_t,evaluation_t,figures_dir,len(FEATURES)))
    paths.append(plot_logistic_state(warm_t,evaluation_t,figures_dir,len(FEATURES)))
    paths.append(plot_tree_growth(warm_t,evaluation_t,figures_dir,len(FEATURES)))
    paths.append(plot_fuzzy_growth(warm_t,evaluation_t,figures_dir,len(FEATURES)))
    p,_=plot_representation_comparison(payloads,figures_dir,processed_dir); paths.append(p)
    paths.append(plot_summary_table(summary_df,figures_dir))

    if print_summary: print_scientific_summary(results,summary_df,warm_t,evaluation_t,FEATURES)
    return paths


def main():
    print("Weekend Challenge 04 — Evolve the Learner")
    print(f"River available: {RIVER_AVAILABLE} ({package_version('river')})")
    print(f"evolvingfuzzysystems available: {EFS_AVAILABLE} ({package_version('evolvingfuzzysystems')})")
    if not RIVER_AVAILABLE:
        print("WARNING: River is missing; educational fallbacks are active.")
    if not EFS_AVAILABLE:
        print("WARNING: evolvingfuzzysystems is missing; the fuzzy fallback is NOT ENFS-Uni0.")
    paths=run_all_experiments()
    print("\nGenerated figures:")
    for path in paths: print("  -",path)


if __name__ == "__main__":
    main()
