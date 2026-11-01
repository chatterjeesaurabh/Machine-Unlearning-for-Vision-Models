# M4 - Gradient based Weight Saliency (add-on to every method)

Last updated: 2026-10-28. Chronology: `records/JOURNAL.md`. Per-method tables: `records/results/runs_<method>.md` (column WS = yes).

## 1. Method and implementation
**Code:** `src/mu/saliency.py` (`compute_saliency_mask`, `random_mask`, `MaskedUpdate`) (reference:
`sample_repos/Unlearn-Saliency-master/Classification/generate_mask.py` + mask handling in its `unlearn/*.py`).
**Mask:** g = sum over the forget set of grad_theta CE(theta_0; x, y) (eval mode, no augmentation, Original weights);
mask = 1 for the top `ratio` fraction of |g|, global threshold over ALL parameters (conv, BN affine, fc).
`ratio` = fraction of weights UPDATED (the Weight Saliency paper's "sparsity" = 1 - ratio).
**Application:** before each optimizer step grad *= mask; after the step masked-out weights are reset to theta_0 and their momentum
buffer zeroed (otherwise weight decay and momentum would still move them). BN running statistics are buffers and are not masked.
Cost: mask computation 1.7-2.4 s; peak GPU memory 1.1 -> 3.8 GB (theta_0 copy + mask).

## 2. Hyperparameter sources
Weight Saliency paper: default ratio 0.5; search 0.1-0.9; larger forget sets prefer fewer updated weights. In the paper the mask is
combined with Random Relabeling (their proposed method) and, in its appendix, with fine-tuning, Random Relabeling, GA and IU.
Ours: the mask is added to all five methods; in the pilot the lr of every masked variant was re-tuned on the same grid.

## 3. Main-grid effect (3 seeds, avg_gap to Retrain, lower is better; best of the pair in bold)
| scenario | GA | GA+WS | GA+Desc. | GA+Desc.+WS | Teacher-Student | TS+WS | Random Relab. | RR+WS | Boundary Relab. | BR+WS |
|---|---|---|---|---|---|---|---|---|---|---|
| S1 class 0 | 4.58 | **4.22** | 1.34 | **0.81** | **0.60** | 1.02 | 1.19 | **1.17** | 6.92 | **4.03** |
| S2-10 | 7.20 | **7.12** | **3.52** | 3.59 | 4.32 | **4.30** | **3.91** | 4.36 | **11.22** | 11.31 |
| S2-30 | 5.55 | **5.45** | **4.15** | 4.38 | 4.17 | **3.97** | 3.25 | **3.10** | **9.92** | 10.12 |
| S2-50 | 8.78 | **8.63** | **3.80** | 4.04 | 58.20 | **57.92** | 5.43 | **2.58** | 8.98 | **8.76** |
Improved configurations (3 seeds; see the method pages): Boundary C1 1.73 vs +WS 1.69; GA G1 4.69 vs +WS 3.74; Teacher-Student T4-best
S1 1.00 vs +WS 0.76, S2-30 3.34 vs 3.47, S2-50 3.62 vs 3.43.

## 4. Ablations (seeds 0-1 for ratios, 0-2 for the random mask; hyperparameters = each method's WS variant), avg_gap
| scenario | method | ratio 0.1 | 0.3 | **0.5 (main)** | 0.7 | 0.9 | random mask 0.5 | no mask |
|---|---|---|---|---|---|---|---|---|
| S1 | GA | 4.35 | **4.00** | 4.22 | 4.27 | 4.45 | 4.91 | 4.58 |
| S1 | GA + Descent | 0.78 | **0.69** | 0.81 | 1.15 | 1.44 | 0.96 | 1.34 |
| S1 | Teacher-Student | 5.16 | 2.71 | 1.02 | 0.58 | **0.53** | 8.47 | 0.60 |
| S1 | Random Relabeling | 1.82 | 1.00 | 1.17 | **0.99** | 1.07 | 2.88 | 1.19 |
| S1 | Boundary Relabeling | 3.76 | **3.61** | 4.03 | 4.56 | 6.34 | 6.98 | 6.92 |
| S2-50 | GA | **8.25** | 8.52 | 8.63 | 8.71 | 8.76 | 8.16 | 8.78 |
| S2-50 | GA + Descent | 4.64 | 4.28 | 4.04 | 3.86 | **3.81** | 4.53 | 3.80 |
| S2-50 | Teacher-Student | 4.60 | 49.26 | 57.92 | 61.33 | 62.56 | 34.36 | 58.20 |
| S2-50 | Random Relabeling | 4.81 | 3.64 | **2.58** | 3.25 | 5.39 | 4.20 | 5.43 |
| S2-50 | Boundary Relabeling | 9.16 | 8.83 | 8.76 | 8.87 | 8.98 | 8.78 | 8.98 |
(The S2-50 Teacher-Student column used the main-grid TS setting, which collapses on S2-50; ratio 0.1 avoids the collapse because
only 10% of the weights can move.)

## 5. Observations
- **Saliency, not just sparsity, matters:** at the same density (0.5) the saliency mask beats a random mask for every method in S1
  (GA 4.22 vs 4.91, GA+Descent 0.81 vs 0.96, TS 1.02 vs 8.47, Random Relabeling 1.17 vs 2.88, Boundary 4.03 vs 6.98) and for Random
  Relabeling / GA+Descent in S2-50.
- **Largest gains** where the base method damages the retain set: Boundary Relabeling S1 (6.92 -> 4.03), GA+Descent S1 (1.34 -> 0.81),
  Random Relabeling S2-50 (5.43 -> 2.58; the paper's own method = Random Relabeling + mask).
- **No gain / slight loss** where the base method is already close to Retrain (Teacher-Student S1) or barely changes the model (S2 GA,
  S2 Boundary).
- **Best ratio is method-dependent:** 0.3 for GA / GA+Descent / Boundary, >= 0.7 for Teacher-Student and Random Relabeling in S1;
  in S2-50 GA+Descent prefers 0.9, Random Relabeling 0.5. Consistent with the paper's remark that the optimum depends on the setting.
- **Limitation found:** the mask does not cover BN running statistics, so it does not prevent the BN-statistics artifact of the
  forget-only methods (GA, Boundary Relabeling; see their pages).

## 6. Evaluation results with and without WS (2026-10-30; details `records/results/evaluation.md`)
Each row compares the same method and configuration without and with the WS mask (3 seeds, mean). Best configurations (G3, D1, T1, R1,
C1-mix in S1; G3-S2, D1-S2, T4-best S2, R1-S2, C0 in S2) except the perturbation rows (main-grid models).
MIA acc and probe/NCC acc in %; avg_gap = mean |metric - Retrain| (lower = better); perturbed forget acc in %; share of weights = fraction
of weights whose INT4 value equals the Original's.
| evaluation (scenario) | model | without WS | with WS |
|---|---|---|---|
| MIA acc, best config (S1) | M1 Gradient Ascent | 51.78 | 52.13 |
| MIA acc, best config (S1) | M2 GA + Descent | 49.25 | 50.67 |
| MIA acc, best config (S1) | M3a Teacher-Student | 52.58 | 52.18 |
| MIA acc, best config (S1) | M3b-R Random Relabeling | 55.47 | 55.48 |
| MIA acc, best config (S1) | M3b-B Boundary Relabeling | 51.03 | 51.90 |
| probe acc on class 0 (S1) | M1 Gradient Ascent | 86.20 | 85.60 |
| probe acc on class 0 (S1) | M2 GA + Descent | 81.80 | 82.87 |
| probe acc on class 0 (S1) | M3a Teacher-Student | 88.50 | 88.20 |
| probe acc on class 0 (S1) | M3b-R Random Relabeling | 94.30 | 94.17 |
| probe acc on class 0 (S1) | M3b-B Boundary Relabeling | 94.33 | 94.27 |
| NCC acc on class 0 (S1) | M1 Gradient Ascent | 30.33 | 34.80 |
| NCC acc on class 0 (S1) | M2 GA + Descent | 32.47 | 30.40 |
| NCC acc on class 0 (S1) | M3a Teacher-Student | 45.60 | 46.50 |
| NCC acc on class 0 (S1) | M3b-R Random Relabeling | 82.10 | 80.33 |
| NCC acc on class 0 (S1) | M3b-B Boundary Relabeling | 74.77 | 75.03 |
| Exp 1 avg_gap, low partition (S1c0) | M1 Gradient Ascent | 1.50 | 1.38 |
| Exp 1 avg_gap, low partition (S1c0) | M2 GA + Descent | 1.39 | 1.17 |
| Exp 1 avg_gap, low partition (S1c0) | M3a Teacher-Student | 0.56 | 0.59 |
| Exp 1 avg_gap, low partition (S1c0) | M3b-R Random Relabeling | 2.06 | 6.75 |
| Exp 1 avg_gap, low partition (S1c0) | M3b-B Boundary Relabeling | 0.74 | 1.06 |
| Exp 1 avg_gap, medium partition (S1c0) | M1 Gradient Ascent | 1.30 | 1.40 |
| Exp 1 avg_gap, medium partition (S1c0) | M2 GA + Descent | 2.56 | 2.07 |
| Exp 1 avg_gap, medium partition (S1c0) | M3a Teacher-Student | 1.39 | 1.82 |
| Exp 1 avg_gap, medium partition (S1c0) | M3b-R Random Relabeling | 0.53 | 1.35 |
| Exp 1 avg_gap, medium partition (S1c0) | M3b-B Boundary Relabeling | 0.48 | 0.12 |
| Exp 1 avg_gap, high partition (S1c0) | M1 Gradient Ascent | 0.84 | 0.91 |
| Exp 1 avg_gap, high partition (S1c0) | M2 GA + Descent | 3.46 | 2.44 |
| Exp 1 avg_gap, high partition (S1c0) | M3a Teacher-Student | 0.84 | 0.63 |
| Exp 1 avg_gap, high partition (S1c0) | M3b-R Random Relabeling | 1.36 | 1.40 |
| Exp 1 avg_gap, high partition (S1c0) | M3b-B Boundary Relabeling | 0.90 | 0.62 |
| Exp 1 avg_gap, low partition (S2k359p10) | M1 Gradient Ascent | 5.42 | 5.63 |
| Exp 1 avg_gap, low partition (S2k359p10) | M2 GA + Descent | 3.86 | 4.27 |
| Exp 1 avg_gap, low partition (S2k359p10) | M3a Teacher-Student | 3.25 | 4.38 |
| Exp 1 avg_gap, low partition (S2k359p10) | M3b-R Random Relabeling | 4.26 | 6.17 |
| Exp 1 avg_gap, low partition (S2k359p10) | M3b-B Boundary Relabeling | 6.19 | 6.34 |
| Exp 1 avg_gap, medium partition (S2k359p10) | M1 Gradient Ascent | 4.67 | 4.66 |
| Exp 1 avg_gap, medium partition (S2k359p10) | M2 GA + Descent | 3.73 | 3.47 |
| Exp 1 avg_gap, medium partition (S2k359p10) | M3a Teacher-Student | 1.13 | 1.39 |
| Exp 1 avg_gap, medium partition (S2k359p10) | M3b-R Random Relabeling | 4.63 | 3.35 |
| Exp 1 avg_gap, medium partition (S2k359p10) | M3b-B Boundary Relabeling | 7.45 | 7.54 |
| Exp 1 avg_gap, high partition (S2k359p10) | M1 Gradient Ascent | 1.28 | 1.35 |
| Exp 1 avg_gap, high partition (S2k359p10) | M2 GA + Descent | 3.14 | 2.94 |
| Exp 1 avg_gap, high partition (S2k359p10) | M3a Teacher-Student | 0.70 | 0.70 |
| Exp 1 avg_gap, high partition (S2k359p10) | M3b-R Random Relabeling | 7.78 | 4.81 |
| Exp 1 avg_gap, high partition (S2k359p10) | M3b-B Boundary Relabeling | 8.61 | 8.61 |
| Exp 1 avg_gap, low partition (S2k359p30) | M1 Gradient Ascent | 3.37 | 3.73 |
| Exp 1 avg_gap, low partition (S2k359p30) | M2 GA + Descent | 3.12 | 3.55 |
| Exp 1 avg_gap, low partition (S2k359p30) | M3a Teacher-Student | 7.62 | 3.79 |
| Exp 1 avg_gap, low partition (S2k359p30) | M3b-R Random Relabeling | 5.49 | 5.39 |
| Exp 1 avg_gap, low partition (S2k359p30) | M3b-B Boundary Relabeling | 11.78 | 11.85 |
| Exp 1 avg_gap, medium partition (S2k359p30) | M1 Gradient Ascent | 4.42 | 4.39 |
| Exp 1 avg_gap, medium partition (S2k359p30) | M2 GA + Descent | 3.58 | 3.91 |
| Exp 1 avg_gap, medium partition (S2k359p30) | M3a Teacher-Student | 3.73 | 2.93 |
| Exp 1 avg_gap, medium partition (S2k359p30) | M3b-R Random Relabeling | 3.26 | 3.03 |
| Exp 1 avg_gap, medium partition (S2k359p30) | M3b-B Boundary Relabeling | 12.17 | 12.33 |
| Exp 1 avg_gap, high partition (S2k359p30) | M1 Gradient Ascent | 1.91 | 1.90 |
| Exp 1 avg_gap, high partition (S2k359p30) | M2 GA + Descent | 3.88 | 3.64 |
| Exp 1 avg_gap, high partition (S2k359p30) | M3a Teacher-Student | 1.96 | 1.12 |
| Exp 1 avg_gap, high partition (S2k359p30) | M3b-R Random Relabeling | 3.33 | 2.79 |
| Exp 1 avg_gap, high partition (S2k359p30) | M3b-B Boundary Relabeling | 14.65 | 14.44 |
| Exp 1 avg_gap, low partition (S2k359p50) | M1 Gradient Ascent | 3.07 | 3.06 |
| Exp 1 avg_gap, low partition (S2k359p50) | M2 GA + Descent | 3.29 | 3.68 |
| Exp 1 avg_gap, low partition (S2k359p50) | M3a Teacher-Student | 4.64 | 4.72 |
| Exp 1 avg_gap, low partition (S2k359p50) | M3b-R Random Relabeling | 4.95 | 5.09 |
| Exp 1 avg_gap, low partition (S2k359p50) | M3b-B Boundary Relabeling | 10.83 | 10.97 |
| Exp 1 avg_gap, medium partition (S2k359p50) | M1 Gradient Ascent | 4.44 | 4.56 |
| Exp 1 avg_gap, medium partition (S2k359p50) | M2 GA + Descent | 4.12 | 4.45 |
| Exp 1 avg_gap, medium partition (S2k359p50) | M3a Teacher-Student | 3.52 | 3.65 |
| Exp 1 avg_gap, medium partition (S2k359p50) | M3b-R Random Relabeling | 3.29 | 3.23 |
| Exp 1 avg_gap, medium partition (S2k359p50) | M3b-B Boundary Relabeling | 12.02 | 12.16 |
| Exp 1 avg_gap, high partition (S2k359p50) | M1 Gradient Ascent | 1.76 | 1.78 |
| Exp 1 avg_gap, high partition (S2k359p50) | M2 GA + Descent | 3.85 | 3.72 |
| Exp 1 avg_gap, high partition (S2k359p50) | M3a Teacher-Student | 1.07 | 0.83 |
| Exp 1 avg_gap, high partition (S2k359p50) | M3b-R Random Relabeling | 2.49 | 2.24 |
| Exp 1 avg_gap, high partition (S2k359p50) | M3b-B Boundary Relabeling | 14.27 | 14.10 |
| Exp 2 avg_gap, one-shot (S1c0) | M1 Gradient Ascent | 1.07 | 1.18 |
| Exp 2 avg_gap, one-shot (S1c0) | M2 GA + Descent | 1.34 | 0.81 |
| Exp 2 avg_gap, one-shot (S1c0) | M3a Teacher-Student | 0.60 | 1.02 |
| Exp 2 avg_gap, one-shot (S1c0) | M3b-R Random Relabeling | 1.19 | 1.17 |
| Exp 2 avg_gap, one-shot (S1c0) | M3b-B Boundary Relabeling | 0.16 | 0.28 |
| Exp 2 avg_gap, sequential (S1c0) | M1 Gradient Ascent | 1.34 | 1.53 |
| Exp 2 avg_gap, sequential (S1c0) | M2 GA + Descent | 3.04 | 3.56 |
| Exp 2 avg_gap, sequential (S1c0) | M3a Teacher-Student | 30.43 | 0.98 |
| Exp 2 avg_gap, sequential (S1c0) | M3b-R Random Relabeling | 28.65 | 28.11 |
| Exp 2 avg_gap, sequential (S1c0) | M3b-B Boundary Relabeling | 27.91 | 28.19 |
| Exp 2 avg_gap, one-shot (S2k359p10) | M1 Gradient Ascent | 4.80 | 5.01 |
| Exp 2 avg_gap, one-shot (S2k359p10) | M2 GA + Descent | 3.52 | 3.59 |
| Exp 2 avg_gap, one-shot (S2k359p10) | M3a Teacher-Student | 3.81 | 3.87 |
| Exp 2 avg_gap, one-shot (S2k359p10) | M3b-R Random Relabeling | 3.91 | 4.36 |
| Exp 2 avg_gap, one-shot (S2k359p10) | M3b-B Boundary Relabeling | 11.22 | 11.31 |
| Exp 2 avg_gap, sequential (S2k359p10) | M1 Gradient Ascent | 5.07 | 4.70 |
| Exp 2 avg_gap, sequential (S2k359p10) | M2 GA + Descent | 4.89 | 4.21 |
| Exp 2 avg_gap, sequential (S2k359p10) | M3a Teacher-Student | 2.78 | 2.85 |
| Exp 2 avg_gap, sequential (S2k359p10) | M3b-R Random Relabeling | 2.98 | 2.81 |
| Exp 2 avg_gap, sequential (S2k359p10) | M3b-B Boundary Relabeling | 13.51 | 13.48 |
| Exp 2 avg_gap, one-shot (S2k359p30) | M1 Gradient Ascent | 4.40 | 4.54 |
| Exp 2 avg_gap, one-shot (S2k359p30) | M2 GA + Descent | 4.15 | 4.38 |
| Exp 2 avg_gap, one-shot (S2k359p30) | M3a Teacher-Student | 3.34 | 3.47 |
| Exp 2 avg_gap, one-shot (S2k359p30) | M3b-R Random Relabeling | 3.25 | 3.10 |
| Exp 2 avg_gap, one-shot (S2k359p30) | M3b-B Boundary Relabeling | 9.92 | 10.12 |
| Exp 2 avg_gap, sequential (S2k359p30) | M1 Gradient Ascent | 7.37 | 6.98 |
| Exp 2 avg_gap, sequential (S2k359p30) | M2 GA + Descent | 6.83 | 4.89 |
| Exp 2 avg_gap, sequential (S2k359p30) | M3a Teacher-Student | 63.87 | 62.38 |
| Exp 2 avg_gap, sequential (S2k359p30) | M3b-R Random Relabeling | 2.67 | 2.69 |
| Exp 2 avg_gap, sequential (S2k359p30) | M3b-B Boundary Relabeling | 14.95 | 14.80 |
| Exp 2 avg_gap, one-shot (S2k359p50) | M1 Gradient Ascent | 4.18 | 4.18 |
| Exp 2 avg_gap, one-shot (S2k359p50) | M2 GA + Descent | 3.80 | 4.04 |
| Exp 2 avg_gap, one-shot (S2k359p50) | M3a Teacher-Student | 3.62 | 3.43 |
| Exp 2 avg_gap, one-shot (S2k359p50) | M3b-R Random Relabeling | 5.43 | 2.58 |
| Exp 2 avg_gap, one-shot (S2k359p50) | M3b-B Boundary Relabeling | 8.98 | 8.76 |
| Exp 2 avg_gap, sequential (S2k359p50) | M1 Gradient Ascent | 6.53 | 6.44 |
| Exp 2 avg_gap, sequential (S2k359p50) | M2 GA + Descent | 9.27 | 6.26 |
| Exp 2 avg_gap, sequential (S2k359p50) | M3a Teacher-Student | 61.64 | 60.48 |
| Exp 2 avg_gap, sequential (S2k359p50) | M3b-R Random Relabeling | 2.55 | 2.55 |
| Exp 2 avg_gap, sequential (S2k359p50) | M3b-B Boundary Relabeling | 13.33 | 13.05 |
| perturbed forget acc, tau 0.03, main grid (S1c0) | M1 Gradient Ascent | 0.02 | 0.03 |
| perturbed forget acc, tau 0.03, main grid (S1c0) | M2 GA + Descent | 0.00 | 0.00 |
| perturbed forget acc, tau 0.03, main grid (S1c0) | M3a Teacher-Student | 0.00 | 0.00 |
| perturbed forget acc, tau 0.03, main grid (S1c0) | M3b-R Random Relabeling | 0.00 | 0.00 |
| perturbed forget acc, tau 0.03, main grid (S1c0) | M3b-B Boundary Relabeling | 0.78 | 0.15 |
| perturbed forget acc, tau 0.03, main grid (S2k359p30) | M1 Gradient Ascent | 45.47 | 45.70 |
| perturbed forget acc, tau 0.03, main grid (S2k359p30) | M2 GA + Descent | 56.08 | 56.43 |
| perturbed forget acc, tau 0.03, main grid (S2k359p30) | M3a Teacher-Student | 87.48 | 83.37 |
| perturbed forget acc, tau 0.03, main grid (S2k359p30) | M3b-R Random Relabeling | 71.04 | 78.41 |
| perturbed forget acc, tau 0.03, main grid (S2k359p30) | M3b-B Boundary Relabeling | 33.03 | 32.93 |
| share of weights with Q(theta_u) = Q(theta_o) at INT4, best config (S1c0) | M1 Gradient Ascent | 0.31 | 0.50 |
| share of weights with Q(theta_u) = Q(theta_o) at INT4, best config (S1c0) | M2 GA + Descent | 0.29 | 0.46 |
| share of weights with Q(theta_u) = Q(theta_o) at INT4, best config (S1c0) | M3a Teacher-Student | 0.29 | 0.46 |
| share of weights with Q(theta_u) = Q(theta_o) at INT4, best config (S1c0) | M3b-R Random Relabeling | 0.24 | 0.36 |
| share of weights with Q(theta_u) = Q(theta_o) at INT4, best config (S1c0) | M3b-B Boundary Relabeling | 0.29 | 0.47 |
| share of weights with Q(theta_u) = Q(theta_o) at INT4, best config (S2k359p10) | M1 Gradient Ascent | 0.31 | 0.49 |
| share of weights with Q(theta_u) = Q(theta_o) at INT4, best config (S2k359p10) | M2 GA + Descent | 0.30 | 0.48 |
| share of weights with Q(theta_u) = Q(theta_o) at INT4, best config (S2k359p10) | M3a Teacher-Student | 0.28 | 0.44 |
| share of weights with Q(theta_u) = Q(theta_o) at INT4, best config (S2k359p10) | M3b-R Random Relabeling | 0.18 | 0.26 |
| share of weights with Q(theta_u) = Q(theta_o) at INT4, best config (S2k359p10) | M3b-B Boundary Relabeling | 0.31 | 0.49 |
| share of weights with Q(theta_u) = Q(theta_o) at INT4, best config (S2k359p30) | M1 Gradient Ascent | 0.31 | 0.48 |
| share of weights with Q(theta_u) = Q(theta_o) at INT4, best config (S2k359p30) | M2 GA + Descent | 0.30 | 0.47 |
| share of weights with Q(theta_u) = Q(theta_o) at INT4, best config (S2k359p30) | M3a Teacher-Student | 0.27 | 0.41 |
| share of weights with Q(theta_u) = Q(theta_o) at INT4, best config (S2k359p30) | M3b-R Random Relabeling | 0.20 | 0.27 |
| share of weights with Q(theta_u) = Q(theta_o) at INT4, best config (S2k359p30) | M3b-B Boundary Relabeling | 0.31 | 0.49 |
| share of weights with Q(theta_u) = Q(theta_o) at INT4, best config (S2k359p50) | M1 Gradient Ascent | 0.31 | 0.48 |
| share of weights with Q(theta_u) = Q(theta_o) at INT4, best config (S2k359p50) | M2 GA + Descent | 0.30 | 0.47 |
| share of weights with Q(theta_u) = Q(theta_o) at INT4, best config (S2k359p50) | M3a Teacher-Student | 0.28 | 0.42 |
| share of weights with Q(theta_u) = Q(theta_o) at INT4, best config (S2k359p50) | M3b-R Random Relabeling | 0.21 | 0.28 |
| share of weights with Q(theta_u) = Q(theta_o) at INT4, best config (S2k359p50) | M3b-B Boundary Relabeling | 0.31 | 0.48 |
| forget acc change INT4 - fp32, best config (S1c0) | M1 Gradient Ascent | -0.07 | -0.04 |
| forget acc change INT4 - fp32, best config (S1c0) | M2 GA + Descent | +0.00 | +0.00 |
| forget acc change INT4 - fp32, best config (S1c0) | M3a Teacher-Student | -0.06 | -0.19 |
| forget acc change INT4 - fp32, best config (S1c0) | M3b-R Random Relabeling | +0.01 | +0.00 |
| forget acc change INT4 - fp32, best config (S1c0) | M3b-B Boundary Relabeling | +0.00 | +0.00 |
| forget acc change INT4 - fp32, best config (S2k359p10) | M1 Gradient Ascent | -3.95 | -3.31 |
| forget acc change INT4 - fp32, best config (S2k359p10) | M2 GA + Descent | -4.32 | -3.83 |
| forget acc change INT4 - fp32, best config (S2k359p10) | M3a Teacher-Student | -1.43 | -1.19 |
| forget acc change INT4 - fp32, best config (S2k359p10) | M3b-R Random Relabeling | -1.38 | -0.12 |
| forget acc change INT4 - fp32, best config (S2k359p10) | M3b-B Boundary Relabeling | -8.02 | -7.93 |
| forget acc change INT4 - fp32, best config (S2k359p30) | M1 Gradient Ascent | -5.29 | -5.60 |
| forget acc change INT4 - fp32, best config (S2k359p30) | M2 GA + Descent | -4.57 | -4.33 |
| forget acc change INT4 - fp32, best config (S2k359p30) | M3a Teacher-Student | -2.07 | -2.16 |
| forget acc change INT4 - fp32, best config (S2k359p30) | M3b-R Random Relabeling | -0.92 | -0.43 |
| forget acc change INT4 - fp32, best config (S2k359p30) | M3b-B Boundary Relabeling | -7.91 | -8.12 |
| forget acc change INT4 - fp32, best config (S2k359p50) | M1 Gradient Ascent | -6.31 | -6.07 |
| forget acc change INT4 - fp32, best config (S2k359p50) | M2 GA + Descent | -4.80 | -3.64 |
| forget acc change INT4 - fp32, best config (S2k359p50) | M3a Teacher-Student | -2.50 | -3.23 |
| forget acc change INT4 - fp32, best config (S2k359p50) | M3b-R Random Relabeling | -1.20 | -0.89 |
| forget acc change INT4 - fp32, best config (S2k359p50) | M3b-B Boundary Relabeling | -7.26 | -7.19 |

Conclusion: WS mainly matters for stability (sequential unlearning: Teacher-Student S1 30.43 -> 0.98, GA+Descent S2-30 6.83 -> 4.89 and
S2-50 9.27 -> 6.26; large forget sets: Random Relabeling S2-50 main grid 5.43 -> 2.58) and for methods that damage the retain set; it does
not change what is encoded in the features (probe/NCC unchanged). It raises the share of weights that keep the Original's INT4 value
(best configurations: 0.26-0.50 with WS vs 0.18-0.31 without; half of the weights are never updated), but the INT4 change of the forget
accuracy is about the same with and without WS (differences <= 1.3 points; no relapse in either case).
