### Stability findings (generated from the tables above)

**Aveiro, Representation A**
- Most stable learner: **Online Logistic Regression** (IQR 0.090, worst 5 % window 0.719, yearly range 0.031). Least stable: Online Bagging (IQR 0.222, p05 0.377).
- No-change: IQR 0.019, p05 0.864. Learners beat it in 0%–1% of informative windows; median windowed κ-t of the best learner: -0.63.
- **Degrading over the years** (OLS slope < 0, p < 0.10): Hoeffding Tree -0.019/yr (p=0.02, 2020→2023); Online Bagging -0.023/yr (p=0.02, 2019→2023).
- Worst season for most learners: JJA; off-season false alarms range 0.0–126.7 h per 30 days.
- Event onsets (No-change catches 0 % by construction): Gaussian NB 28% of first hours, median delay 1 h, 26% episodes missed, 173.3 spurious alarm episodes/yr; Adaptive Random Forest 26% of first hours, median delay 1 h, 27% episodes missed, 39.0 spurious alarm episodes/yr; Online Logistic Regression 19% of first hours, median delay 1 h, 22% episodes missed, 31.7 spurious alarm episodes/yr; Hoeffding Tree 8% of first hours, median delay 2 h, 68% episodes missed, 91.7 spurious alarm episodes/yr; Online Bagging 6% of first hours, median delay 2 h, 80% episodes missed, 28.0 spurious alarm episodes/yr. Read onset hits together with spurious alarms: firing often buys onsets.
- Error drifts (ADWIN, learners): 242 (146 worsening); 60% in season-transition months vs 50% expected by chance (binomial p = 0.001) → drift is mainly **seasonal**. Worsening drifts: 13 not recovered within 90 days; the rest recovered after a median 9 days.
- Adaptive Random Forest replaced 614 trees during the stream (its own ADWIN detectors) — evidence of active adaptation.
- Seed sensitivity: std of overall macro-F1 across 5 seeds ≤ 0.0020 — negligible next to the differences between models.

**Aveiro, Representation B**
- Most stable learner: **Online Logistic Regression** (IQR 0.045, worst 5 % window 0.801, yearly range 0.022). Least stable: Gaussian NB (IQR 0.067, p05 0.687).
- No-change: IQR 0.019, p05 0.864. Learners beat it in 0%–13% of informative windows; median windowed κ-t of the best learner: -0.26.
- No learner shows a significant downward trend across years (p < 0.10).
- Worst season for most learners: MAM; off-season false alarms range 0.0–1.2 h per 30 days.
- Event onsets (No-change catches 0 % by construction): Gaussian NB 45% of first hours, median delay 0 h, 14% episodes missed, 21.7 spurious alarm episodes/yr; Hoeffding Tree 32% of first hours, median delay 1 h, 16% episodes missed, 33.7 spurious alarm episodes/yr; Online Logistic Regression 31% of first hours, median delay 1 h, 13% episodes missed, 36.8 spurious alarm episodes/yr; Online Bagging 27% of first hours, median delay 1 h, 17% episodes missed, 14.5 spurious alarm episodes/yr; Adaptive Random Forest 25% of first hours, median delay 1 h, 24% episodes missed, 16.3 spurious alarm episodes/yr. Read onset hits together with spurious alarms: firing often buys onsets.
- Error drifts (ADWIN, learners): 139 (88 worsening); 76% in season-transition months vs 50% expected by chance (binomial p = 0.000) → drift is mainly **seasonal**. Worsening drifts: 8 not recovered within 90 days; the rest recovered after a median 9 days.
- Adaptive Random Forest replaced 618 trees during the stream (its own ADWIN detectors) — evidence of active adaptation.
- Seed sensitivity: std of overall macro-F1 across 5 seeds ≤ 0.0009 — negligible next to the differences between models.

**Viseu, Representation A**
- Most stable learner: **Online Logistic Regression** (IQR 0.039, worst 5 % window 0.815, yearly range 0.026). Least stable: Gaussian NB (IQR 0.102, p05 0.634).
- No-change: IQR 0.011, p05 0.889. Learners beat it in 0%–1% of informative windows; median windowed κ-t of the best learner: -0.50.
- No learner shows a significant downward trend across years (p < 0.10).
- Worst season for most learners: MAM; off-season false alarms range 0.1–96.9 h per 30 days.
- Event onsets (No-change catches 0 % by construction): Gaussian NB 46% of first hours, median delay 0 h, 8% episodes missed, 127.3 spurious alarm episodes/yr; Adaptive Random Forest 37% of first hours, median delay 1 h, 13% episodes missed, 33.8 spurious alarm episodes/yr; Online Logistic Regression 26% of first hours, median delay 1 h, 10% episodes missed, 25.3 spurious alarm episodes/yr; Hoeffding Tree 20% of first hours, median delay 2 h, 16% episodes missed, 106.2 spurious alarm episodes/yr; Online Bagging 19% of first hours, median delay 2 h, 20% episodes missed, 55.3 spurious alarm episodes/yr. Read onset hits together with spurious alarms: firing often buys onsets.
- Error drifts (ADWIN, learners): 228 (129 worsening); 61% in season-transition months vs 50% expected by chance (binomial p = 0.000) → drift is mainly **seasonal**. Worsening drifts: 4 not recovered within 90 days; the rest recovered after a median 10 days.
- Adaptive Random Forest replaced 497 trees during the stream (its own ADWIN detectors) — evidence of active adaptation.
- Seed sensitivity: std of overall macro-F1 across 5 seeds ≤ 0.0022 — negligible next to the differences between models.

**Viseu, Representation B**
- Most stable learner: **Online Bagging** (IQR 0.051, worst 5 % window 0.810, yearly range 0.029). Least stable: Gaussian NB (IQR 0.084, p05 0.606).
- No-change: IQR 0.011, p05 0.889. Learners beat it in 0%–19% of informative windows; median windowed κ-t of the best learner: -0.28.
- No learner shows a significant downward trend across years (p < 0.10).
- Worst season for most learners: SON; off-season false alarms range 0.0–0.2 h per 30 days.
- Event onsets (No-change catches 0 % by construction): Gaussian NB 62% of first hours, median delay 0 h, 11% episodes missed, 9.8 spurious alarm episodes/yr; Adaptive Random Forest 33% of first hours, median delay 1 h, 14% episodes missed, 15.5 spurious alarm episodes/yr; Online Logistic Regression 32% of first hours, median delay 1 h, 8% episodes missed, 36.8 spurious alarm episodes/yr; Online Bagging 30% of first hours, median delay 1 h, 12% episodes missed, 16.2 spurious alarm episodes/yr; Hoeffding Tree 25% of first hours, median delay 1 h, 11% episodes missed, 33.0 spurious alarm episodes/yr. Read onset hits together with spurious alarms: firing often buys onsets.
- Error drifts (ADWIN, learners): 120 (73 worsening); 87% in season-transition months vs 50% expected by chance (binomial p = 0.000) → drift is mainly **seasonal**. Worsening drifts: 7 not recovered within 90 days; the rest recovered after a median 10 days.
- Adaptive Random Forest replaced 492 trees during the stream (its own ADWIN detectors) — evidence of active adaptation.
- Seed sensitivity: std of overall macro-F1 across 5 seeds ≤ 0.0008 — negligible next to the differences between models.

**Data drift**
- Aveiro: yearly event rate 12.0% (2021) – 18.8% (2023); a 6.8 pp swing with τ_hot frozen = **prior-probability shift**.
- Viseu: yearly event rate 17.4% (2021) – 23.0% (2022); a 5.6 pp swing with τ_hot frozen = **prior-probability shift**.
- Largest input shifts (hot season, KS): Aveiro heat_events_last_24h 2021 (0.21); Aveiro rolling_mean_temp_24h 2021 (0.19); Aveiro heat_events_last_24h 2019 (0.19).
- Window length: the IQR ranking of learners agrees across 7, 30, 90-day windows with mean Kendall τ = 0.61.