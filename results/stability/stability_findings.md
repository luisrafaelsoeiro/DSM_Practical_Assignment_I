### Stability findings (generated from the tables above)

**Aveiro, Representation A**
- Most stable learner: **Online Bagging** (IQR 0.022, worst 5 % window 0.859, yearly range 0.017). Least stable: Adaptive Random Forest (IQR 0.031, p05 0.833).
- No-change: IQR 0.019, p05 0.864. Learners beat it in 2%–67% of informative windows; median windowed κ-t of the best learner: +0.06.
- No learner shows a significant downward trend across years (p < 0.10).
- Worst season for most learners: MAM; off-season false alarms range 0.0–7.3 h per 30 days.
- Event onsets (No-change catches 0 % by construction): Gaussian NB 39% of first hours, median delay 1 h, 7% episodes missed, 68.7 spurious alarm episodes/yr; Online Logistic Regression 15% of first hours, median delay 1 h, 13% episodes missed, 12.7 spurious alarm episodes/yr; Adaptive Random Forest 12% of first hours, median delay 1 h, 16% episodes missed, 16.2 spurious alarm episodes/yr; Hoeffding Tree 7% of first hours, median delay 1 h, 13% episodes missed, 20.7 spurious alarm episodes/yr; Online Bagging 2% of first hours, median delay 1 h, 14% episodes missed, 11.3 spurious alarm episodes/yr. Read onset hits together with spurious alarms: firing often buys onsets.
- Error drifts (ADWIN, learners): 92 (55 worsening); 67% in season-transition months vs 50% expected by chance (binomial p = 0.001) → drift is mainly **seasonal**. Worsening drifts: 15 not recovered within 90 days; the rest recovered after a median 10 days.
- Adaptive Random Forest replaced 178 trees during the stream (its own ADWIN detectors) — evidence of active adaptation.
- Seed sensitivity: std of overall macro-F1 across 5 seeds ≤ 0.0013 — negligible next to the differences between models.

**Aveiro, Representation B**
- Most stable learner: **Online Bagging** (IQR 0.024, worst 5 % window 0.904, yearly range 0.015). Least stable: Gaussian NB (IQR 0.050, p05 0.760).
- No-change: IQR 0.019, p05 0.864. Learners beat it in 0%–99% of informative windows; median windowed κ-t of the best learner: +0.44.
- No learner shows a significant downward trend across years (p < 0.10).
- Worst season for most learners: MAM; off-season false alarms range 0.0–1.5 h per 30 days.
- Event onsets (No-change catches 0 % by construction): Gaussian NB 68% of first hours, median delay 0 h, 8% episodes missed, 29.3 spurious alarm episodes/yr; Online Bagging 56% of first hours, median delay 0 h, 7% episodes missed, 26.7 spurious alarm episodes/yr; Hoeffding Tree 56% of first hours, median delay 0 h, 8% episodes missed, 40.0 spurious alarm episodes/yr; Online Logistic Regression 50% of first hours, median delay 0 h, 11% episodes missed, 25.8 spurious alarm episodes/yr; Adaptive Random Forest 36% of first hours, median delay 1 h, 15% episodes missed, 16.8 spurious alarm episodes/yr. Read onset hits together with spurious alarms: firing often buys onsets.
- Error drifts (ADWIN, learners): 94 (55 worsening); 72% in season-transition months vs 50% expected by chance (binomial p = 0.000) → drift is mainly **seasonal**. Worsening drifts: 6 not recovered within 90 days; the rest recovered after a median 10 days.
- Adaptive Random Forest replaced 419 trees during the stream (its own ADWIN detectors) — evidence of active adaptation.
- Seed sensitivity: std of overall macro-F1 across 5 seeds ≤ 0.0008 — negligible next to the differences between models.

**Viseu, Representation A**
- Most stable learner: **Hoeffding Tree** (IQR 0.017, worst 5 % window 0.885, yearly range 0.015). Least stable: Adaptive Random Forest (IQR 0.022, p05 0.845).
- No-change: IQR 0.011, p05 0.889. Learners beat it in 6%–68% of informative windows; median windowed κ-t of the best learner: +0.02.
- No learner shows a significant downward trend across years (p < 0.10).
- Worst season for most learners: MAM; off-season false alarms range 0.0–10.0 h per 30 days.
- Event onsets (No-change catches 0 % by construction): Gaussian NB 53% of first hours, median delay 0 h, 4% episodes missed, 49.5 spurious alarm episodes/yr; Adaptive Random Forest 16% of first hours, median delay 1 h, 8% episodes missed, 11.2 spurious alarm episodes/yr; Online Logistic Regression 13% of first hours, median delay 1 h, 8% episodes missed, 8.0 spurious alarm episodes/yr; Hoeffding Tree 5% of first hours, median delay 1 h, 6% episodes missed, 11.3 spurious alarm episodes/yr; Online Bagging 0% of first hours, median delay 1 h, 7% episodes missed, 7.7 spurious alarm episodes/yr. Read onset hits together with spurious alarms: firing often buys onsets.
- Error drifts (ADWIN, learners): 79 (44 worsening); 82% in season-transition months vs 50% expected by chance (binomial p = 0.000) → drift is mainly **seasonal**. Worsening drifts: 5 not recovered within 90 days; the rest recovered after a median 12 days.
- Adaptive Random Forest replaced 140 trees during the stream (its own ADWIN detectors) — evidence of active adaptation.
- Seed sensitivity: std of overall macro-F1 across 5 seeds ≤ 0.0009 — negligible next to the differences between models.

**Viseu, Representation B**
- Most stable learner: **Online Bagging** (IQR 0.015, worst 5 % window 0.940, yearly range 0.010). Least stable: Adaptive Random Forest (IQR 0.026, p05 0.840).
- No-change: IQR 0.011, p05 0.889. Learners beat it in 1%–100% of informative windows; median windowed κ-t of the best learner: +0.56.
- **Degrading over the years** (OLS slope < 0, p < 0.10): Online Bagging -0.002/yr (p=0.02, 2018→2023); Adaptive Random Forest -0.002/yr (p=0.06, 2018→2023).
- Worst season for most learners: MAM; off-season false alarms range 0.0–0.4 h per 30 days.
- Event onsets (No-change catches 0 % by construction): Gaussian NB 77% of first hours, median delay 0 h, 7% episodes missed, 10.8 spurious alarm episodes/yr; Hoeffding Tree 67% of first hours, median delay 0 h, 5% episodes missed, 29.3 spurious alarm episodes/yr; Online Bagging 66% of first hours, median delay 0 h, 4% episodes missed, 17.0 spurious alarm episodes/yr; Online Logistic Regression 61% of first hours, median delay 0 h, 6% episodes missed, 22.3 spurious alarm episodes/yr; Adaptive Random Forest 51% of first hours, median delay 0 h, 8% episodes missed, 13.2 spurious alarm episodes/yr. Read onset hits together with spurious alarms: firing often buys onsets.
- Error drifts (ADWIN, learners): 76 (43 worsening); 82% in season-transition months vs 50% expected by chance (binomial p = 0.000) → drift is mainly **seasonal**. Worsening drifts: 5 not recovered within 90 days; the rest recovered after a median 10 days.
- Adaptive Random Forest replaced 334 trees during the stream (its own ADWIN detectors) — evidence of active adaptation.
- Seed sensitivity: std of overall macro-F1 across 5 seeds ≤ 0.0004 — negligible next to the differences between models.

**Data drift**
- Aveiro: yearly event rate 12.0% (2021) – 18.8% (2023); a 6.8 pp swing with τ_hot frozen = **prior-probability shift**.
- Viseu: yearly event rate 17.4% (2021) – 23.0% (2022); a 5.6 pp swing with τ_hot frozen = **prior-probability shift**.
- Largest input shifts (hot season, KS): Aveiro heat_events_last_24h 2021 (0.21); Aveiro rolling_mean_temp_24h 2021 (0.19); Aveiro heat_events_last_24h 2019 (0.19).
- Window length: the IQR ranking of learners agrees across 7, 30, 90-day windows with mean Kendall τ = 0.68.