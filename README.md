# PA08 – Heat Stress Event in Aveiro and Viseu

**Data Stream Mining · NOVA IMS · Practical Assignment I – Business-Oriented Data Stream Mining**

| | |
|---|---|
| **Assignment code** | PA08 |
| **Group** | Group PA08 |
| **Students** | Luís Soeiro (20211536) – City A (Aveiro) · Maria Inês Santos (20231630) – City B (Viseu) |
| **City A / City B** | Aveiro / Viseu |
| **Business scenario** | Chocolate Distribution Centre |
| **Business question** | Which location may require more cooling effort during relatively hot periods? |
| **Official target** | T2 – Heat Stress Event |
| **Historical period** | 2013-01-01 → 2023-12-31 (hourly) |
| **Threshold warm-up** | 2013-01-01 → 2015-12-31 (common to both cities) |
| **Python** | 3.12.10 |
| **Random seed** | 42 |

---

## Contents

1. [Project overview](#1-project-overview)
2. [Repository structure](#2-repository-structure)
3. [Environment and installation](#3-environment-and-installation)
4. [How to reproduce the experiment](#4-how-to-reproduce-the-experiment)
5. [Data](#5-data)
6. [Target construction (T2)](#6-target-construction-t2)
7. [Feature representations](#7-feature-representations)
8. [Models and evaluation protocol](#8-models-and-evaluation-protocol)
9. [Leakage and fairness controls](#9-leakage-and-fairness-controls)
10. [Outputs and where they come from](#10-outputs-and-where-they-come-from)
11. [Configuration reference](#11-configuration-reference)
12. [Original analyses](#12-original-analyses)
13. [Known limitations](#13-known-limitations)
14. [AI assistance](#14-ai-assistance)

---

## 1. Project overview

A chocolate distribution centre is choosing between Aveiro (coastal) and Viseu (inland). Chocolate is temperature-sensitive, so the deciding weather factor is how often, and for how long, each site goes through relatively hot periods that would increase cooling demand.

The project has two parts:

1. **Descriptive / business:** how often, how persistently, and in which temporal patterns heat-stress hours occur in each city.
2. **Predictive / technical:** how well and how cheaply incremental classifiers can predict the heat-stress state one hour ahead, using only information available before that hour.

The report gives two separate recommendations: a **business recommendation** (which site looks more suitable from the cooling perspective) and a **technical recommendation** (which streaming approach suits continuous monitoring).

---

## 2. Repository structure

```
DSM_Practical_Assignment_I/
├── README.md
├── requirements.txt
├── src/
│   ├── config.py                    # all paths, dates, thresholds, feature lists, seed
│   ├── exploration_functions.py     # Open-Meteo download, cleaning helpers, EDA plots
│   ├── preprocessing_functions.py   # audit, imputation, threshold, labelling, features
│   ├── model_functions.py           # (imports only – modelling code lives in 03)
│   ├── 01_exploration.ipynb         # business framing, data collection, stream profile
│   ├── 02_preprocessing.ipynb       # cleaning, frozen threshold, target, past-only features
│   ├── 03_modelling.ipynb           # prequential experiments, tuning, stability analysis
│   └── figures/                     # full-resolution EDA figures exported by 01
├── data/
│   ├── raw/                         # original API responses (aveiro.json, viseu.json)
│   ├── processed/                   # hourly tables per city (aveiro.csv, viseu.csv)
│   └── preprocessed/                # model-ready streams + preproc_meta.json
└── results/
    ├── metrics.csv                  # first run, default hyperparameters, full evaluation stream
    ├── metrics_default.csv          # final protocol, default hyperparameters
    ├── metrics_tuned.csv            # final protocol, tuned hyperparameters (headline results)
    ├── configurations.csv           # versions, seed, threshold, periods, parameters, features
    ├── best_hyperparameters.csv
    ├── tuning_results.csv
    ├── tuning_effect.csv
    ├── confusion_matrices_tuned.csv
    ├── feature_relevance_logistic.csv
    ├── figures/                     # report figures (HTML interactive, PNG for confusion matrices)
    ├── stability/                   # model-stability analysis tables + findings
    └── seasonality/                 # seasonality-improvement analysis tables + findings
```

> `Pratical/` (course lab notebooks and weekend-challenge solutions) and the handbook PDF are reference material only. They are not part of the experiment and are left out of the audit ZIP.

---

## 3. Environment and installation

Tested with **Python 3.12.10** on Windows.

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements-lock.txt

pip install -r requirements.txt
```
`requirements-lock.txt` is the full `pip freeze` of the environment that produced the results in `results/` (Python 3.12.10, Windows). `requirements.txt` lists only the main packages.
```

| Package | Version |
|---|---|
| river | 0.26.1 |
| numpy | 2.2.6 |
| pandas | 2.3.3 |
| scikit-learn | 1.7.2 |
| scipy | 1.15.3 |
| plotly | 5.24.1 |
| matplotlib | 3.10.6 |
| seaborn | 0.12.3 |
| requests | 2.32.5 |

`kaleido` is optional. If it's installed, Plotly figures are also exported as PNG next to the HTML.

The exact versions used in each run are written automatically to `results/configurations.csv`.

---

## 4. How to reproduce the experiment

Run the notebooks **in order, from inside `src/`** (they import `config.py` and the helper modules from that folder):

```bash
cd src
jupyter nbconvert --to notebook --execute --inplace 01_exploration.ipynb
jupyter nbconvert --to notebook --execute --inplace 02_preprocessing.ipynb
jupyter nbconvert --to notebook --execute --inplace 03_modelling.ipynb
```

You can also open them in Jupyter / VS Code and choose *Run All*.

| Step | Notebook | Reads | Writes |
|---|---|---|---|
| 1 | `01_exploration.ipynb` | Open-Meteo API, or the cache in `data/` | `data/raw/*.json`, `data/processed/*.csv`, `src/figures/*` |
| 2 | `02_preprocessing.ipynb` | `data/processed/*.csv` | `data/preprocessed/preproc_*.csv`, `stream_*.csv`, `preproc_meta.json` |
| 3 | `03_modelling.ipynb` | `data/preprocessed/*` | everything in `results/` |

**Data cache.** `fetch_open_meteo_data` reuses `data/processed/<city>.csv` when it exists, so the reported results come from the stored snapshots and not from a new download. To re-download, call it with `force_download=True`. Open-Meteo reanalysis values can be revised over time, so a fresh download may differ slightly from the stored snapshot.

**Runtime.** Section 7 of `03_modelling.ipynb` (stability: five seeds and three window lengths) is the slowest part. It caches its replays in `results/stability/stability_replay_cache.pkl`. Delete that file to force a full recomputation.

---

## 5. Data

| Item | Value |
|---|---|
| Source | Open-Meteo Historical Weather API – `https://archive-api.open-meteo.com/v1/archive` |
| Resolution | Hourly |
| Time zone | `Europe/Lisbon` |
| Aveiro | lat 40.6405, lon −8.6538 |
| Viseu | lat 40.6572, lon −7.9142 |
| API variables | `temperature_2m`, `relative_humidity_2m`, `precipitation`, `wind_speed_10m` |
| Renamed to | `temperature` (°C), `relative_humidity` (%), `precipitation` (mm), `wind_speed` (km/h) |
| Derived variable | `night_min_temperature`: minimum temperature of the previous 21:00–06:00 night (2 buffer days are downloaded before the start date and then trimmed) |
| Size | 96,408 hourly observations per city |

### Data quality procedure (`02_preprocessing.ipynb`, section 3)

All interventions are logged in an audit table (`audit_weather_data`):

1. **Duplicate timestamps** are removed, keeping the first observation in chronological order.
2. **Physically implausible values** (outside the bounds in `PHYSICAL_BOUNDS`) are set to NaN.
3. **Missing values** are imputed by forward fill, which uses past values only.
4. **Missing timestamps** stop the pipeline with an error (fail loudly) rather than being silently filled.

Extreme but plausible values are **kept**. A very hot hour is the event the business needs to detect.

---

## 6. Target construction (T2)

**Definition (handbook, chapter 6):**

$$\tau_{T,hot} = Q_{0.80}(T)\ \text{over the pooled hourly warm-up temperatures of both cities}, \qquad y_t = 1 \iff T_t \ge \tau_{T,hot}$$

| Item | Value |
|---|---|
| Warm-up for the threshold | 2013-01-01 → 2015-12-31, Aveiro + Viseu pooled |
| Estimator | River `stats.Quantile(0.80)`, updated one hourly observation at a time |
| **Frozen threshold** | **τ = 20.21 °C** (exact NumPy check on the same data: 20.10 °C) |
| Target column | `target_hot` |
| Labelled period | 2016-01-01 → 2023-12-31 (no labels during the warm-up) |

The threshold is estimated once, saved to `data/preprocessed/preproc_meta.json`, and **never recomputed** during evaluation.

> The notebook also explores a daily-maximum threshold (25.82 °C) and a "≥ 3 consecutive hot days" heatwave detector, **for comparison only**. The models predict the official hourly T2 label `target_hot`.

---

## 7. Feature representations

All features use observations **strictly before** the target hour *t*. The current temperature `temperature(t)` defines the target, so it is never an input.

**Representation A – previous state** (`FEATURE_COLS_A`)

| Feature | Definition |
|---|---|
| `temperature_lag1` | T(t−1) |
| `relative_humidity_lag1` | H(t−1) |
| `precipitation_lag1` | P(t−1) |
| `wind_speed_lag1` | W(t−1) |

**Representation B – stream-aware temporal context** (`FEATURE_COLS_B` = A plus the following)

| Feature | Definition |
|---|---|
| `delta_temp` | T(t−1) − T(t−2) |
| `rolling_mean_temp_6h` | mean of T(t−6 … t−1) |
| `rolling_std_temp_6h` | std of T(t−6 … t−1) |
| `rolling_mean_temp_24h` | mean of T(t−24 … t−1) |
| `heat_events_last_24h` | number of hours in t−24 … t−1 with T ≥ τ |

The first 24 evaluation hours of each city are dropped, so every row has a full 24 h of history. Both representations are therefore scored on exactly the same rows.

---

## 8. Models and evaluation protocol

### Models (River)

| Model | Family | Implementation |
|---|---|---|
| Gaussian Naive Bayes | single learner | `naive_bayes.GaussianNB` |
| Online Logistic Regression | single learner | `StandardScaler` (online) → `linear_model.LogisticRegression(SGD)` |
| Hoeffding Tree | single learner | `tree.HoeffdingTreeClassifier` |
| Online Bagging | ensemble | `ensemble.BaggingClassifier` (Hoeffding Tree base) |
| Adaptive Random Forest | ensemble | `forest.ARFClassifier` |
| Majority class | reference baseline | `dummy.PriorClassifier` |
| No-change (persistence) | reference baseline | `dummy.NoChangeClassifier` |

The two baselines are not required by the handbook. They are included because the heat-stress state is highly persistent, so "repeat the previous label" is a strong reference point (κ-t is measured against it).

### Prequential protocol

For every event, in chronological order: **PREDICT → EVALUATE → LEARN**.

- Fresh model instances for every city × representation × configuration.
- Every model sees exactly the same chronological events.
- Prediction time and update time are measured separately for each model and event.
- Cumulative scores plus a 30-day sliding window (`WINDOW_SIZE = 720` h); one history row is stored every 24 events.

### Hyperparameter tuning (`03_modelling.ipynb`, cells T0–T3)

| Item | Value |
|---|---|
| Tuning data | first 25 % of the evaluation stream (2016-01 → 2017-12) |
| Scored stream | 2018-01-01 06:00 → 2023-12-31 23:00, **52,578 events per city** |
| Inner protocol | first 30 % of the tuning data used for training only, the remaining 70 % scored prequentially |
| Selection metric | Macro-F1; a candidate must beat the default by ≥ 0.002, otherwise the default is kept |
| Scope | pooled: one configuration per model and representation, shared by both cities |
| Search | random search (Logistic Regression 15, Hoeffding Tree 30, Online Bagging 20, ARF 20 candidates; the default is always a candidate) |
| Final run | models learn unscored on the tuning period, then are scored on the remaining stream. Default and tuned configurations are scored on the **same** events. |

Search spaces, defaults and the selected values are recorded in `results/configurations.csv` and `results/best_hyperparameters.csv`.

### Metrics

- **Predictive:** Accuracy, Macro-F1, Precision / Recall / F1 for the event class, confusion matrix, κ-t (relative to No-change), rolling Macro-F1.
- **Computational:** prediction ms/event, update ms/event, model complexity (statistics, coefficients, tree nodes or ensemble size).

---

## 9. Leakage and fairness controls

| Handbook rule | How it is enforced |
|---|---|
| Thresholds from the warm-up only, then frozen | τ estimated on 2013–2015 and stored in `preproc_meta.json`; no labels or scores are produced during the warm-up |
| No current target-defining value as input | `temperature(t)` excluded; `03_modelling.ipynb` asserts that no representation contains it |
| Lags and windows end at t−1 | every feature is built from `shift(1)` before rolling (`prepare_stream_features`) |
| No future-based preprocessing | forward-fill imputation; online `StandardScaler` |
| Chronological replay, no random split | streams asserted monotonic and unique in time |
| Predict before learn | enforced inside `track_online_performance` |
| Same history for every model | one shared event iterator per city × representation |
| Tuning never sees scored data | assertion: max tuning time < min scored time |
| Fresh instances | `make_models()` builds new objects on every call |
| Same protocol for both cities | common τ, warm-up, features, parameters and metrics |

---

## 10. Outputs and where they come from

### Tables (`results/`)

| File | Produced by | Content |
|---|---|---|
| `metrics.csv` | 03 – cell "Run experiments" | first run: default parameters, full 2016–2023 stream, no model warm-up |
| `metrics_default.csv` | 03 – T3 | final protocol, default parameters |
| `metrics_tuned.csv` | 03 – T3 | **final protocol, tuned parameters (headline results)** |
| `configurations.csv` | 03 – T3 (overwrites the first run) | versions, seed, τ, periods, search spaces, selected parameters, feature lists |
| `tuning_results.csv`, `best_hyperparameters.csv` | 03 – T2 | every candidate's tuning score; the selected configuration |
| `tuning_effect.csv` | 03 – T3 | tuned − default, per city × representation × model |
| `confusion_matrices_tuned.csv` | 03 – M1 | TP / TN / FP / FN of the tuned models |
| `feature_relevance_logistic.csv` | 03 – F1 | standardised coefficients, drop-one and single-feature ablations |
| `stability/*` | 03 – section 7 (S0–S9) | yearly / seasonal / rolling stability, event-level onsets, ADWIN drifts, seeds, window sensitivity, `stability_findings.md` |
| `seasonality/*` | seasonality-improvement analysis | seasonal experts vs global model, per-period thresholds, `seasonality_findings.md` |

### Main figures (`results/figures/`)

| Figure | Supports |
|---|---|
| `event_rate_over_time.html`, `event_profiles.html` | comparative data analysis: how often and when heat stress happens in each city |
| `incremental_mean_temperature.html` | stream characterisation: running mean ± std per city |
| `confusion_*_{A,B}.png`, `confusion_tuned_*.html` | required confusion matrices |
| `representation_comparison.html`, `representation_delta.html` | A vs B comparison |
| `single_vs_ensembles.html` | single Hoeffding Tree vs ensembles |
| `precision_recall_map.html` | precision / recall operating points with iso-F1 curves |
| `cost_performance.html` | Macro-F1 vs ms/event, with the Pareto frontier |
| `curve_rolling_macro_f1.html`, `curve_kappa_t.html`, `curve_recall.html` | performance over time |
| `tuning_landscape.html`, `tuning_effect.html`, `confusion_error_shift.html` | effect of hyperparameter tuning |
| `logistic_coefficients.html`, `logistic_weight_paths.html`, `feature_alone.html`, `feature_drop_one.html` | feature relevance |
| `stability_*.html` | model stability over time |
| `seasonality_*.html` | seasonality-improvement analysis |

EDA figures from `01_exploration.ipynb` (full hourly time series, outlier boxplots) are in `src/figures/`.

---

## 11. Configuration reference

Everything lives in `src/config.py`. Change values there only. The notebooks import them.

| Constant | Value | Meaning |
|---|---|---|
| `START_DATE`, `END_DATE` | 2013-01-01, 2023-12-31 | requested historical period |
| `TIMEZONE` | Europe/Lisbon | API time zone |
| `WARMUP_START`, `WARMUP_END` | 2013-01-01, 2015-12-31 | threshold warm-up |
| `HOT_PERCENTILE` | 0.80 | T2 quantile |
| `NIGHT_START_HOUR`, `NIGHT_END_HOUR` | 21, 6 | night window for `night_min_temperature` |
| `NIGHT_BUFFER_DAYS` | 2 | extra days downloaded before the start date |
| `PHYSICAL_BOUNDS` | see file | plausibility limits used by the audit |
| `WINDOW_SIZE` | 720 | sliding evaluation window (30 days) |
| `FEATURE_WINDOWS` | (6, 24) | rolling-feature windows (hours) |
| `FEATURE_COLS_A`, `FEATURE_COLS_B` | see section 7 | feature representations |
| `RANDOM_SEED` | 42 | seed for ensembles and the hyperparameter sampler |
| `MIN_CONSECUTIVE_HOT_DAYS` | 3 | heatwave detector (comparison only, not the target) |

Stability analysis (`results/stability/stability_configuration.csv`): 720 h windows with a 24 h stride, seeds {42, 7, 123, 2024, 31415}, ADWIN δ = 0.002, window grid {168, 720, 2160} h, hot season May–October.

---

## 12. Original analyses

Beyond the course starter code (handbook 3.2):

- **Event-level evaluation:** share of heat episodes whose first hour is caught, alarm delay, missed episodes, and spurious alarm episodes per year. This is closer to how a cooling manager experiences the system than hourly accuracy.
- **Model stability over time:** yearly trends, seasonal breakdown, ADWIN error-drift detection and recovery, seed and window-length sensitivity.
- **Cost-performance Pareto frontier** across learners.
- **Feature relevance** of the online logistic regression: coefficient paths, plus drop-one and single-feature ablations.
- **Seasonality improvement:** seasonal expert models vs one global model in spring/autumn transition periods.
- **Calendar heatmaps and event profiles** comparing hot periods in the two cities.

---

## 13. Known limitations

- **Weather only.** The business recommendation covers the cooling dimension only. Rent, wages, transport access and other location factors are outside this dataset.
- **Reanalysis data.** Open-Meteo values are modelled (gridded reanalysis), not station measurements, and the coordinates are city centres rather than candidate sites.
- **Pair-relative threshold.** τ = 20.21 °C is "hot relative to Aveiro + Viseu in 2013–2015", not an absolute cooling set-point for chocolate storage.
- **Persistent target.** Hourly heat stress is strongly autocorrelated, so the No-change baseline is very hard to beat on hourly metrics. Event-level and κ-t evidence is reported for this reason.
- **Tuning data.** Hyperparameters were tuned on the first 25 % of the evaluation stream, which is therefore excluded from the scored results.

---

## 14. AI assistance

Generative AI was used as permitted in handbook chapter 16. The mandatory AI Assistance Statement is at the end of the report (`PA08_GroupYY_Report.pdf`).
