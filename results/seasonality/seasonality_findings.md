### Seasonality improvement: findings

**Aveiro**
- Overall macro-F1: Global LR B 0.944 → seasonal experts (blended) 0.943 (Δ -0.002); best learned variant: **Global LR B** (0.944); No-change 0.926.
- Mean Δ macro-F1 vs Global LR B in the transition zones: Global LR B + season features -0.002; Global LR B + per-period threshold -0.001; Seasonal experts (hard routing) -0.001; Seasonal experts (blended) -0.000; Seasonal experts (blended) + per-period threshold -0.002.
- Mean Δ in the specialist periods (early, core, late): Global LR B + season features -0.003; Global LR B + per-period threshold -0.000; Seasonal experts (hard routing) -0.003; Seasonal experts (blended) -0.002; Seasonal experts (blended) + per-period threshold -0.003.
- Blending vs hard routing in the zones: +0.001 (positive = blending helps at the boundaries).
- Stability (blended vs global): rolling IQR 0.029 vs 0.026; worst 5 % month 0.880 vs 0.883; error drifts 13 vs 14; beats No-change in 86% vs 90% of months.
- Event level (blended vs global): onsets caught in early/late season 44% vs 42%; spurious alarm episodes per year 31.3 vs 25.8; alarm episodes per true episode 1.20 vs 1.14.

**Viseu**
- Overall macro-F1: Global LR B 0.964 → seasonal experts (blended) 0.965 (Δ +0.000); best learned variant: **Seasonal experts (blended)** (0.965); No-change 0.945.
- Mean Δ macro-F1 vs Global LR B in the transition zones: Global LR B + season features -0.000; Global LR B + per-period threshold -0.001; Seasonal experts (hard routing) +0.002; Seasonal experts (blended) +0.002; Seasonal experts (blended) + per-period threshold -0.001.
- Mean Δ in the specialist periods (early, core, late): Global LR B + season features -0.001; Global LR B + per-period threshold -0.000; Seasonal experts (hard routing) +0.001; Seasonal experts (blended) +0.001; Seasonal experts (blended) + per-period threshold +0.001.
- Blending vs hard routing in the zones: +0.000 (positive = blending helps at the boundaries).
- Stability (blended vs global): rolling IQR 0.023 vs 0.019; worst 5 % month 0.911 vs 0.911; error drifts 12 vs 13; beats No-change in 99% vs 100% of months.
- Event level (blended vs global): onsets caught in early/late season 62% vs 58%; spurious alarm episodes per year 23.0 vs 22.3; alarm episodes per true episode 1.15 vs 1.14.

**Consistency across years (both cities pooled)**
- Global LR B + season features: specialist periods — better in 8/36 cells, median Δ -0.001, Wilcoxon p = 0.000; transition zones — better in 7/29 cells, median Δ -0.001, Wilcoxon p = 0.019.
- Global LR B + per-period threshold: specialist periods — better in 14/36 cells, median Δ -0.000, Wilcoxon p = 0.334; transition zones — better in 8/29 cells, median Δ -0.001, Wilcoxon p = 0.090.
- Seasonal experts (hard routing): specialist periods — better in 14/36 cells, median Δ -0.001, Wilcoxon p = 0.279; transition zones — better in 15/29 cells, median Δ +0.001, Wilcoxon p = 0.696.
- Seasonal experts (blended): specialist periods — better in 16/36 cells, median Δ -0.000, Wilcoxon p = 0.550; transition zones — better in 17/29 cells, median Δ +0.001, Wilcoxon p = 0.594.
- Seasonal experts (blended) + per-period threshold: specialist periods — better in 13/36 cells, median Δ -0.001, Wilcoxon p = 0.273; transition zones — better in 13/29 cells, median Δ -0.002, Wilcoxon p = 0.190.

**Verdict**
- The hypothesis is **not supported** at the required consistency (≥ 2/3 of year-zone cells better and Wilcoxon p < 0.05). The global model's seasonal re-fitting already captures most of the transition behaviour; any remaining difficulty in the transitions is largely in the data (short, sporadic episodes).