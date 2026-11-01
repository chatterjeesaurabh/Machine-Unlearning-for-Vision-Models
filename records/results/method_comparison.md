# Method comparison: main grid vs improved configurations (all 3 seeds, mean)

Last updated: 2026-10-28. avg_gap = mean |metric - Retrain| over forget acc, retain acc, test acc and MIA acc (lower = closer to
Retrain). Configuration names are defined on each method page (`records/methods/`). WS = Gradient based Weight Saliency mask 0.5.
Best per scenario in **bold**. Source: `records/results/all_runs.csv`; per-method full tables `records/results/runs_<method>.md`.

## S1 - class 0 (Retrain: forget 0.00, retain 100, test other-cls 94.82, MIA 51.0; Original avg_gap 29.05)
| method | main grid config | forget | retain | test other-cls | MIA | avg_gap | best improved config (3 seeds) | forget | retain | test other-cls | MIA | avg_gap |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| M1 Gradient Ascent | G0 | 5.33 | 93.91 | 86.99 | 51.17 | 4.58 | G3 (mixed BN, lr 3e-4, iter stop) | 0.45 | 99.17 | 92.34 | 51.78 | 1.07 |
| M2 GA + Descent | D1 | 0.00 | 99.09 | 91.87 | 49.25 | 1.34 | D1 + WS | 0.00 | 99.43 | 92.54 | 50.67 | 0.81 |
| M3a Teacher-Student | T1 | 0.33 | 99.99 | 94.30 | 52.58 | 0.60 | T4-best (Illusion sched., alpha 0.5, gamma 1) | 0.01 | 99.84 | 94.10 | 54.18 | 1.00 |
| M3b-R Random Relabeling | R1 | 0.00 | 100.00 | 94.54 | 55.47 | 1.19 | R1 + WS | 0.00 | 100.00 | 94.64 | 55.48 | 1.17 |
| M3b-B Boundary Relabeling | C0 (paper) | 23.59 | 99.48 | 92.31 | 54.28 | 6.92 | **C1-mix (logit2, lr 1e-3, retain pass, mixed BN)** | **0.00** | **100.00** | **94.56** | **51.03** | **0.16** |

### S1 - with and without Gradient based Weight Saliency (WS, mask ratio 0.5; 3 seeds, mean)
| method | main grid config | forget | retain | test other-cls | MIA | avg_gap | best improved config (3 seeds) | forget | retain | test other-cls | MIA | avg_gap |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| M1 Gradient Ascent | G0 | 5.33 | 93.91 | 86.99 | 51.17 | 4.58 | G3 (mixed BN, lr 3e-4, iter stop) | 0.45 | 99.17 | 92.34 | 51.78 | 1.07 |
| M1 Gradient Ascent + WS | G0 + WS | 6.30 | 95.47 | 88.37 | 51.75 | 4.22 | G3 + WS | 0.62 | 99.15 | 92.40 | 52.13 | 1.18 |
| M2 GA + Descent | D1 | 0.00 | 99.09 | 91.87 | 49.25 | 1.34 | D1 + WS | 0.00 | 99.43 | 92.54 | 50.67 | 0.81 |
| M2 GA + Descent + WS | D1 + WS | 0.00 | 99.43 | 92.54 | 50.67 | 0.81 | D1 + WS (improved config already uses WS) | 0.00 | 99.43 | 92.54 | 50.67 | 0.81 |
| M3a Teacher-Student | T1 | 0.33 | 99.99 | 94.30 | 52.58 | 0.60 | T4-best (Illusion sched., alpha 0.5, gamma 1) | 0.01 | 99.84 | 94.10 | 54.18 | 1.00 |
| M3a Teacher-Student + WS | T1 + WS | 2.56 | 99.99 | 94.26 | 52.18 | 1.02 | T4-best + WS | 0.55 | 100.00 | 94.75 | 53.33 | 0.76 |
| M3b-R Random Relabeling | R1 | 0.00 | 100.00 | 94.54 | 55.47 | 1.19 | R1 + WS | 0.00 | 100.00 | 94.64 | 55.48 | 1.17 |
| M3b-R Random Relabeling + WS | R1 + WS | 0.00 | 100.00 | 94.64 | 55.48 | 1.17 | R1 + WS (improved config already uses WS) | 0.00 | 100.00 | 94.64 | 55.48 | 1.17 |
| M3b-B Boundary Relabeling | C0 (paper) | 23.59 | 99.48 | 92.31 | 54.28 | 6.92 | **C1-mix (logit2, lr 1e-3, retain pass, mixed BN)** | **0.00** | **100.00** | **94.56** | **51.03** | **0.16** |
| M3b-B Boundary Relabeling + WS | C0-lr4 + WS (WS variant's selected lr 1e-4) | 3.22 | 97.48 | 89.19 | 56.50 | 4.03 | C1-mix + WS | 0.00 | 100.00 | 94.60 | 51.90 | 0.28 |

## S2 - random % of classes 3, 5, 9 (avg_gap to Retrain, lower = better)
Main grid = paper-range settings; best config = improved configuration (names on the method pages). Original = no unlearning.
| scenario | model | avg_gap main grid, no WS | avg_gap main grid, WS | avg_gap best config, no WS | avg_gap best config, WS | best config |
|---|---|---|---|---|---|---|
| S2k359p10 | Original (no unlearning) | 4.47 | | | | |
| S2k359p10 | M1 Gradient Ascent | 7.20 | 7.12 | 4.80 | 5.01 | G3-S2 |
| S2k359p10 | M2 GA + Descent | 3.52 | 3.59 | 3.52 | 3.59 | D1-S2 |
| S2k359p10 | M3a Teacher-Student | 4.32 | 4.30 | 3.81 | 3.87 | T4-best S2 |
| S2k359p10 | M3b-R Random Relabeling | 3.91 | 4.36 | 3.91 | 4.36 | R1-S2 |
| S2k359p10 | M3b-B Boundary Relabeling | 11.22 | 11.31 | 11.22 | 11.31 | C0 |
| S2k359p30 | Original (no unlearning) | 4.75 | | | | |
| S2k359p30 | M1 Gradient Ascent | 5.55 | 5.45 | 4.40 | 4.54 | G3-S2 |
| S2k359p30 | M2 GA + Descent | 4.15 | 4.38 | 4.15 | 4.38 | D1-S2 |
| S2k359p30 | M3a Teacher-Student | 4.17 | 3.97 | 3.34 | 3.47 | T4-best S2 |
| S2k359p30 | M3b-R Random Relabeling | 3.25 | 3.10 | 3.25 | 3.10 | R1-S2 |
| S2k359p30 | M3b-B Boundary Relabeling | 9.92 | 10.12 | 9.92 | 10.12 | C0 |
| S2k359p50 | Original (no unlearning) | 5.26 | | | | |
| S2k359p50 | M1 Gradient Ascent | 8.78 | 8.63 | 4.18 | 4.18 | G3-S2 |
| S2k359p50 | M2 GA + Descent | 3.80 | 4.04 | 3.80 | 4.04 | D1-S2 |
| S2k359p50 | M3a Teacher-Student | 58.20 | 57.92 | 3.62 | 3.43 | T4-best S2 |
| S2k359p50 | M3b-R Random Relabeling | 5.43 | 2.58 | 5.43 | 2.58 | R1-S2 |
| S2k359p50 | M3b-B Boundary Relabeling | 8.98 | 8.76 | 8.98 | 8.76 | C0 |
In S2 the Retrain forget accuracy is ~89-92% (remaining same-class samples keep the classes learnable); no method reaches a low
forget accuracy without dropping same-class retain/test accuracy, and the MIA of every method stays at 53-60% (Retrain ~50%).

## Problems found and fixes (details on the method pages)
| problem | affected | fix | effect (S1) |
|---|---|---|---|
| BN running statistics drift on forget-only batches ("forgetting" lives in BN stats; relapses after BN recalibration) | GA, Boundary (also GA+Descent's separate forward) | mixed BN batches (`bn_mix_retain` / `joint_forward`), or frozen BN (+ gradient clipping for GA) | GA 4.58 -> 1.07; Boundary C1 1.73 -> 0.16 |
| one-step FGSM (eps 0.1) assigns far-away classes (bird/frog instead of ship) | Boundary | pgd (first class crossed), logit2, or FGSM eps 0.03 | avg_gap 6.92 (C0, 3 seeds) -> 1.73 (C1, 3 seeds) -> 0.16 (C1-mix, 3 seeds) |
| teacher soft labels (T=4) re-teach the forgotten class; too few max-steps with forget batch 512 | Teacher-Student | small KL weight / Illusion schedule (batch 64) | avg_gap 28.57 (paper Table 3, seed 0) -> 0.60 (T1, 3 seeds); 1.00 (T4-best S1, 3 seeds).0 |
| max-step count grows with forget-set size -> collapse | Teacher-Student S2 | scale forget batch with forget-set size | S2-50 58.2 -> 3.62 |
| GA overshoots within an epoch | GA | iteration-level early stop | part of G1/G3 |
| frozen-BN GA explodes (no BN rescaling) | GA | gradient clipping | retain 22-76% -> 92.5% |
| lr chosen on S2-30 too aggressive on S2-50 | Random Relabeling | Weight Saliency mask (or lr 0.01) | S2-50 5.43 -> 2.58 |
| NaN logits reported as 100% forget accuracy | all | `diverged` flag, NaN-safe MIA | - |
| features of the forgotten class remain linearly decodable (probe 88-95%) | all methods incl. Retrain (87.7%) | open (would need a feature-level loss) | - |

## Evaluation summary (best configurations; details `records/results/evaluation.md`)

**How to read these tables (every cell is a single value, mean over 3 seeds):**
- **config**: the configuration of the method used (names defined on the method pages; best configuration per method from the tables above).
- **MIA acc (%)**: accuracy of the loss-based membership-inference attacker at telling forget images from unseen test images of the same
  class(es). 50% = cannot tell them apart; the target is the Retrain model's value.
- **probe acc on class 0 (%)** (S1 only): a new linear classifier is trained on the frozen features of the unlearned model (using all
  training images, including class 0) and tested on the class-0 test images. High = class 0 can still be read out of the features.
- **NCC acc on class 0 (%)** (S1 only): class-0 test images classified by the nearest class-mean feature vector (no training). High = class 0
  still forms its own group in feature space.
- **10-NN purity of class 0 (%)** (S1 only): for each class-0 test image, the share of its 10 nearest test images (in feature space) that are
  also class 0. High = class-0 images still cluster together.
- **Exp 1 avg_gap (low / medium / high partition)**: entanglement experiment 1. The forget set is split into three equal parts by how close
  each forget image is to the retain images in feature space (low = far from retain images, high = close). Each part is unlearned on its
  own, starting from the Original model. **S1:** forget = the part, retain = the normal S1 retain set (no airplane); the other two parts
  are not used; reference = the normal S1 Retrain evaluated on the part (forget acc 0%). **S2:** forget = the part, retain = all other
  training data (the other two parts included); reference = a model retrained without only that part. avg_gap = mean absolute
  difference to the reference over forget acc, retain acc, test acc and MIA acc (lower = closer to retraining). Per-partition forget
  accuracies: `records/results/evaluation.md` section 3.1.
  For the **Original** row, avg_gap is the same difference for the untouched Original model, i.e. how far the model is from the retrained
  model before any unlearning - the gap a method has to close. 0 for Retrain by definition.
- **Exp 2 avg_gap (one-shot / sequential)**: entanglement experiment 2. *One-shot* = the whole forget set unlearned in one run (the normal
  setting). *Sequential* = the three parts unlearned one after another (low, then medium, then high), each run starting from the previous
  run's model. Both final models are compared with the normal Retrain model of the scenario on the whole forget set. For the Original row
  the value is the gap with no unlearning at all.

### S1 - class 0 (single-class forgetting)
| model | WS | config | MIA acc (%) | probe acc on class 0 (%) | NCC acc on class 0 (%) | 10-NN purity of class 0 (%) | Exp 1 avg_gap: low partition | Exp 1 avg_gap: medium | Exp 1 avg_gap: high | Exp 2 avg_gap: one-shot | Exp 2 avg_gap: sequential |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Retrain (reference) | | | 51.02 | 87.73 | 41.43 | 52.58 | 0 | 0 | 0 | 0 | 0 |
| Original (no unlearning) | | | 57.78 | 95.50 | 95.40 | 93.00 | 28.59 | 29.52 | 29.27 | 29.05 | - |
| M1 Gradient Ascent | no | G3 | 51.78 | 86.20 | 30.33 | 32.63 | 1.50 | 1.30 | 0.84 | 1.07 | 1.34 |
| M1 Gradient Ascent | yes | G3 | 52.13 | 85.60 | 34.80 | 29.19 | 1.38 | 1.40 | 0.91 | 1.18 | 1.53 |
| M2 GA + Descent | no | D1 | 49.25 | 81.80 | 32.47 | 32.38 | 1.39 | 2.56 | 3.46 | 1.34 | 3.04 |
| M2 GA + Descent | yes | D1 | 50.67 | 82.87 | 30.40 | 32.73 | 1.17 | 2.07 | 2.44 | 0.81 | 3.56 |
| M3a Teacher-Student | no | T1 | 52.58 | 88.50 | 45.60 | 56.07 | 0.56 | 1.39 | 0.84 | 0.60 | 30.43 |
| M3a Teacher-Student | yes | T1 | 52.18 | 88.20 | 46.50 | 57.77 | 0.59 | 1.82 | 0.63 | 1.02 | 0.98 |
| M3b-R Random Relabeling | no | R1 | 55.47 | 94.30 | 82.10 | 89.24 | 2.06 | 0.53 | 1.36 | 1.19 | 28.65 |
| M3b-R Random Relabeling | yes | R1 | 55.48 | 94.17 | 80.33 | 89.41 | 6.75 | 1.35 | 1.40 | 1.17 | 28.11 |
| M3b-B Boundary Relabeling | no | C1-mix | 51.03 | 94.33 | 74.77 | 82.16 | 0.74 | 0.48 | 0.90 | 0.16 | 27.91 |
| M3b-B Boundary Relabeling | yes | C1-mix | 51.90 | 94.27 | 75.03 | 82.94 | 1.06 | 0.12 | 0.62 | 0.28 | 28.19 |

### S2 - random 10% / 30% / 50% of classes 3, 5, 9 (subset forgetting; linear-probe metrics are defined only for class forgetting)
| scenario | model | WS | config | MIA acc (%) | Exp 1 avg_gap: low partition | Exp 1 avg_gap: medium | Exp 1 avg_gap: high | Exp 2 avg_gap: one-shot | Exp 2 avg_gap: sequential |
|---|---|---|---|---|---|---|---|---|---|
| S2k359p10 | Retrain (reference) | | | 49.64 | 0 | 0 | 0 | 0 | 0 |
| S2k359p10 | Original (no unlearning) | | | 58.95 | 6.24 | 3.32 | 0.67 | 4.47 | - |
| S2k359p10 | M1 Gradient Ascent | no | G3-S2 | 60.10 | 5.42 | 4.67 | 1.28 | 4.80 | 5.07 |
| S2k359p10 | M1 Gradient Ascent | yes | G3-S2 | 60.40 | 5.63 | 4.66 | 1.35 | 5.01 | 4.70 |
| S2k359p10 | M2 GA + Descent | no | D1-S2 | 57.99 | 3.86 | 3.73 | 3.14 | 3.52 | 4.89 |
| S2k359p10 | M2 GA + Descent | yes | D1-S2 | 58.20 | 4.27 | 3.47 | 2.94 | 3.59 | 4.21 |
| S2k359p10 | M3a Teacher-Student | no | T4-best S2 | 57.02 | 3.25 | 1.13 | 0.70 | 3.81 | 2.78 |
| S2k359p10 | M3a Teacher-Student | yes | T4-best S2 | 57.09 | 4.38 | 1.39 | 0.70 | 3.87 | 2.85 |
| S2k359p10 | M3b-R Random Relabeling | no | R1-S2 | 55.62 | 4.26 | 4.63 | 7.78 | 3.91 | 2.98 |
| S2k359p10 | M3b-R Random Relabeling | yes | R1-S2 | 57.07 | 6.17 | 3.35 | 4.81 | 4.36 | 2.81 |
| S2k359p10 | M3b-B Boundary Relabeling | no | C0 | 59.25 | 6.19 | 7.45 | 8.61 | 11.22 | 13.51 |
| S2k359p10 | M3b-B Boundary Relabeling | yes | C0 | 59.21 | 6.34 | 7.54 | 8.61 | 11.31 | 13.48 |
| S2k359p30 | Retrain (reference) | | | 49.95 | 0 | 0 | 0 | 0 | 0 |
| S2k359p30 | Original (no unlearning) | | | 59.46 | 5.85 | 3.06 | 0.75 | 4.75 | - |
| S2k359p30 | M1 Gradient Ascent | no | G3-S2 | 59.62 | 3.37 | 4.42 | 1.91 | 4.40 | 7.37 |
| S2k359p30 | M1 Gradient Ascent | yes | G3-S2 | 59.64 | 3.73 | 4.39 | 1.90 | 4.54 | 6.98 |
| S2k359p30 | M2 GA + Descent | no | D1-S2 | 59.12 | 3.12 | 3.58 | 3.88 | 4.15 | 6.83 |
| S2k359p30 | M2 GA + Descent | yes | D1-S2 | 59.44 | 3.55 | 3.91 | 3.64 | 4.38 | 4.89 |
| S2k359p30 | M3a Teacher-Student | no | T4-best S2 | 56.13 | 7.62 | 3.73 | 1.96 | 3.34 | 63.87 |
| S2k359p30 | M3a Teacher-Student | yes | T4-best S2 | 56.64 | 3.79 | 2.93 | 1.12 | 3.47 | 62.38 |
| S2k359p30 | M3b-R Random Relabeling | no | R1-S2 | 53.16 | 5.49 | 3.26 | 3.33 | 3.25 | 2.67 |
| S2k359p30 | M3b-R Random Relabeling | yes | R1-S2 | 55.04 | 5.39 | 3.03 | 2.79 | 3.10 | 2.69 |
| S2k359p30 | M3b-B Boundary Relabeling | no | C0 | 59.52 | 11.78 | 12.17 | 14.65 | 9.92 | 14.95 |
| S2k359p30 | M3b-B Boundary Relabeling | yes | C0 | 59.42 | 11.85 | 12.33 | 14.44 | 10.12 | 14.80 |
| S2k359p50 | Retrain (reference) | | | 50.75 | 0 | 0 | 0 | 0 | 0 |
| S2k359p50 | Original (no unlearning) | | | 59.47 | 6.43 | 3.34 | 0.76 | 5.26 | - |
| S2k359p50 | M1 Gradient Ascent | no | G3-S2 | 59.46 | 3.07 | 4.44 | 1.76 | 4.18 | 6.53 |
| S2k359p50 | M1 Gradient Ascent | yes | G3-S2 | 59.51 | 3.06 | 4.56 | 1.78 | 4.18 | 6.44 |
| S2k359p50 | M2 GA + Descent | no | D1-S2 | 59.07 | 3.29 | 4.12 | 3.85 | 3.80 | 9.27 |
| S2k359p50 | M2 GA + Descent | yes | D1-S2 | 59.43 | 3.68 | 4.45 | 3.72 | 4.04 | 6.26 |
| S2k359p50 | M3a Teacher-Student | no | T4-best S2 | 57.04 | 4.64 | 3.52 | 1.07 | 3.62 | 61.64 |
| S2k359p50 | M3a Teacher-Student | yes | T4-best S2 | 57.42 | 4.72 | 3.65 | 0.83 | 3.43 | 60.48 |
| S2k359p50 | M3b-R Random Relabeling | no | R1-S2 | 51.78 | 4.95 | 3.29 | 2.49 | 5.43 | 2.55 |
| S2k359p50 | M3b-R Random Relabeling | yes | R1-S2 | 53.74 | 5.09 | 3.23 | 2.24 | 2.58 | 2.55 |
| S2k359p50 | M3b-B Boundary Relabeling | no | C0 | 59.92 | 10.83 | 12.02 | 14.27 | 8.98 | 13.33 |
| S2k359p50 | M3b-B Boundary Relabeling | yes | C0 | 59.68 | 10.97 | 12.16 | 14.10 | 8.76 | 13.05 |

## Post-unlearning quantization (best configurations; details `records/results/evaluation.md` section 6, `records/results/quant_best.md`)
Simulated weight-only, per-output-channel symmetric quantization of all Conv/Linear weights (BatchNorm in fp32); 3 seeds, mean;
avg_gap against the fp32 Retrain of the scenario. Figures: `records/figures/fig_quant_best_*.png`, `fig_quant_margin_diag.png`.

### S1 - class 0 (single-class forgetting)
| model | WS | config | forget acc fp32 | forget acc INT8 | forget acc INT6 | forget acc INT4 | change forget (INT4 - fp32) | change retain | change test | change MIA | avg_gap fp32 | avg_gap INT4 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Retrain (reference) |  |  | 0.00 | 0.00 | 0.00 | 0.00 | +0.00 | -0.01 | -0.34 | -0.07 | 0.00 | 0.10 |
| Original (no unlearning) |  |  | 100.00 | 100.00 | 100.00 | 100.00 | +0.00 | +0.00 | -0.33 | -0.60 | 29.05 | 28.82 |
| M1 Gradient Ascent |  | G3 | 0.46 | 0.44 | 0.41 | 0.39 | -0.07 | -0.42 | -0.71 | +0.23 | 1.06 | 1.39 |
| M1 Gradient Ascent | yes | G3 | 0.63 | 0.63 | 0.62 | 0.59 | -0.04 | -0.42 | -0.82 | +0.12 | 1.17 | 1.50 |
| M2 GA + Descent |  | D1 | 0.00 | 0.00 | 0.00 | 0.00 | +0.00 | -0.33 | -0.43 | -0.37 | 1.34 | 1.62 |
| M2 GA + Descent | yes | D1 | 0.00 | 0.00 | 0.00 | 0.00 | +0.00 | -0.31 | -0.49 | +0.37 | 0.75 | 0.87 |
| M3a Teacher-Student |  | T1 | 0.33 | 0.32 | 0.30 | 0.27 | -0.06 | -0.03 | -0.30 | +0.35 | 0.60 | 0.75 |
| M3a Teacher-Student | yes | T1 | 2.56 | 2.56 | 2.51 | 2.37 | -0.19 | -0.02 | -0.20 | -0.68 | 1.02 | 0.86 |
| M3b-R Random Relabeling |  | R1 | 0.00 | 0.00 | 0.00 | 0.01 | +0.01 | -0.01 | -0.30 | -0.02 | 1.19 | 1.26 |
| M3b-R Random Relabeling | yes | R1 | 0.00 | 0.00 | 0.00 | 0.00 | +0.00 | -0.00 | -0.27 | +0.50 | 1.17 | 1.36 |
| M3b-B Boundary Relabeling |  | C1-mix | 0.00 | 0.00 | 0.00 | 0.00 | +0.00 | -0.01 | -0.37 | -0.43 | 0.07 | 0.26 |
| M3b-B Boundary Relabeling | yes | C1-mix | 0.00 | 0.00 | 0.00 | 0.00 | +0.00 | -0.01 | -0.37 | -0.87 | 0.27 | 0.16 |

### S2 - random 10% / 30% / 50% of classes 3, 5, 9 (subset forgetting)
| scenario | model | WS | config | forget acc fp32 | forget acc INT8 | forget acc INT6 | forget acc INT4 | change forget (INT4 - fp32) | change retain | change test | change MIA | avg_gap fp32 | avg_gap INT4 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| S2k359p10 | Retrain (reference) |  |  | 91.63 | 91.63 | 91.46 | 90.59 | -1.04 | +0.00 | -0.28 | -0.16 | 0.00 | 0.37 |
| S2k359p10 | Original (no unlearning) |  |  | 100.00 | 100.00 | 100.00 | 100.00 | +0.00 | +0.00 | -0.33 | +0.72 | 4.47 | 4.63 |
| S2k359p10 | M1 Gradient Ascent |  | G3-S2 | 95.38 | 95.43 | 94.81 | 91.43 | -3.95 | -1.00 | -0.98 | -0.98 | 4.79 | 4.15 |
| S2k359p10 | M1 Gradient Ascent | yes | G3-S2 | 96.74 | 96.79 | 96.42 | 93.43 | -3.31 | -0.87 | -1.02 | -1.05 | 5.01 | 4.40 |
| S2k359p10 | M2 GA + Descent |  | D1-S2 | 91.41 | 91.28 | 91.26 | 87.09 | -4.32 | -0.84 | -0.87 | -1.37 | 3.46 | 4.63 |
| S2k359p10 | M2 GA + Descent | yes | D1-S2 | 92.32 | 92.44 | 92.27 | 88.49 | -3.83 | -0.75 | -0.85 | -0.88 | 3.59 | 4.38 |
| S2k359p10 | M3a Teacher-Student |  | T4-best S2 | 97.53 | 97.53 | 97.48 | 96.10 | -1.43 | -0.27 | -0.55 | -0.52 | 3.82 | 3.54 |
| S2k359p10 | M3a Teacher-Student | yes | T4-best S2 | 97.85 | 97.88 | 97.88 | 96.67 | -1.19 | -0.21 | -0.46 | -0.12 | 3.87 | 3.71 |
| S2k359p10 | M3b-R Random Relabeling |  | R1-S2 | 95.16 | 95.26 | 95.14 | 93.78 | -1.38 | -0.77 | -0.72 | -0.53 | 3.91 | 3.80 |
| S2k359p10 | M3b-R Random Relabeling | yes | R1-S2 | 97.23 | 97.26 | 97.14 | 97.11 | -0.12 | -0.34 | -0.50 | -0.47 | 4.36 | 4.42 |
| S2k359p10 | M3b-B Boundary Relabeling |  | C0 | 73.56 | 73.98 | 72.10 | 65.53 | -8.02 | -2.13 | -1.67 | -0.68 | 11.22 | 14.01 |
| S2k359p10 | M3b-B Boundary Relabeling | yes | C0 | 73.23 | 73.58 | 71.78 | 65.31 | -7.93 | -2.14 | -1.75 | -0.86 | 11.31 | 14.05 |
| S2k359p30 | Retrain (reference) |  |  | 90.86 | 90.92 | 90.94 | 90.63 | -0.23 | -0.00 | -0.33 | +0.15 | 0.00 | 0.18 |
| S2k359p30 | Original (no unlearning) |  |  | 100.00 | 100.00 | 100.00 | 100.00 | +0.00 | +0.00 | -0.33 | +0.72 | 4.75 | 4.85 |
| S2k359p30 | M1 Gradient Ascent |  | G3-S2 | 90.55 | 90.55 | 89.51 | 85.26 | -5.29 | -1.17 | -1.28 | -0.81 | 4.32 | 6.05 |
| S2k359p30 | M1 Gradient Ascent | yes | G3-S2 | 90.46 | 90.41 | 89.55 | 84.86 | -5.60 | -1.24 | -1.39 | -0.86 | 4.36 | 6.20 |
| S2k359p30 | M2 GA + Descent |  | D1-S2 | 92.92 | 92.91 | 92.94 | 88.35 | -4.57 | -0.97 | -1.19 | -0.75 | 4.15 | 4.61 |
| S2k359p30 | M2 GA + Descent | yes | D1-S2 | 93.53 | 93.65 | 93.51 | 89.20 | -4.33 | -0.87 | -1.01 | -0.76 | 4.38 | 4.41 |
| S2k359p30 | M3a Teacher-Student |  | T4-best S2 | 94.60 | 94.59 | 94.46 | 92.53 | -2.07 | -0.59 | -0.81 | -0.36 | 3.33 | 3.07 |
| S2k359p30 | M3a Teacher-Student | yes | T4-best S2 | 94.50 | 94.47 | 94.53 | 92.35 | -2.16 | -0.44 | -0.65 | -0.42 | 3.47 | 3.10 |
| S2k359p30 | M3b-R Random Relabeling |  | R1-S2 | 88.67 | 88.77 | 88.60 | 87.74 | -0.92 | -0.63 | -0.67 | -0.23 | 3.25 | 3.75 |
| S2k359p30 | M3b-R Random Relabeling | yes | R1-S2 | 94.35 | 94.47 | 94.40 | 93.92 | -0.43 | -0.41 | -0.40 | -0.37 | 3.10 | 3.10 |
| S2k359p30 | M3b-B Boundary Relabeling |  | C0 | 75.54 | 75.79 | 74.34 | 67.63 | -7.91 | -1.84 | -1.67 | -1.13 | 9.92 | 12.49 |
| S2k359p30 | M3b-B Boundary Relabeling | yes | C0 | 74.99 | 75.25 | 73.92 | 66.86 | -8.12 | -1.82 | -1.63 | -1.12 | 10.12 | 12.73 |
| S2k359p50 | Retrain (reference) |  |  | 88.74 | 88.75 | 88.69 | 87.99 | -0.75 | +0.00 | -0.33 | +0.03 | 0.00 | 0.28 |
| S2k359p50 | Original (no unlearning) |  |  | 100.00 | 100.00 | 100.00 | 100.00 | +0.00 | +0.00 | -0.33 | +0.70 | 5.26 | 5.36 |
| S2k359p50 | M1 Gradient Ascent |  | G3-S2 | 88.31 | 88.35 | 86.96 | 82.00 | -6.31 | -1.10 | -1.70 | -0.88 | 4.06 | 6.11 |
| S2k359p50 | M1 Gradient Ascent | yes | G3-S2 | 88.05 | 88.07 | 86.81 | 81.98 | -6.07 | -1.06 | -1.55 | -0.81 | 4.19 | 6.15 |
| S2k359p50 | M2 GA + Descent |  | D1-S2 | 89.26 | 89.20 | 88.92 | 84.45 | -4.80 | -0.85 | -1.13 | -0.76 | 3.73 | 4.97 |
| S2k359p50 | M2 GA + Descent | yes | D1-S2 | 90.67 | 90.58 | 90.47 | 87.03 | -3.64 | -0.64 | -1.03 | -0.51 | 4.04 | 4.27 |
| S2k359p50 | M3a Teacher-Student |  | T4-best S2 | 94.10 | 94.04 | 93.79 | 91.60 | -2.50 | -0.43 | -0.88 | -0.44 | 3.62 | 3.21 |
| S2k359p50 | M3a Teacher-Student | yes | T4-best S2 | 91.67 | 91.65 | 91.94 | 88.44 | -3.23 | -0.53 | -1.04 | -0.52 | 3.42 | 3.02 |
| S2k359p50 | M3b-R Random Relabeling |  | R1-S2 | 81.07 | 81.08 | 81.14 | 79.86 | -1.20 | -1.00 | -0.99 | -0.25 | 5.43 | 6.17 |
| S2k359p50 | M3b-R Random Relabeling | yes | R1-S2 | 88.96 | 88.99 | 89.16 | 88.07 | -0.89 | -0.78 | -0.57 | -0.49 | 2.21 | 2.54 |
| S2k359p50 | M3b-B Boundary Relabeling |  | C0 | 75.11 | 75.37 | 73.81 | 67.84 | -7.26 | -1.23 | -1.58 | -0.95 | 8.98 | 11.26 |
| S2k359p50 | M3b-B Boundary Relabeling | yes | C0 | 75.54 | 75.77 | 74.36 | 68.35 | -7.19 | -1.22 | -1.57 | -0.89 | 8.76 | 11.03 |

**Observations and conclusion:** no method regains forgotten knowledge at any bit width (largest forget-acc increase +0.42, INT8); INT8 and
INT6 change at most 1.46 points. In S1 the forgetting is stable at INT4 for every method. In S2, INT4 lowers forget accuracy (Gradient
Ascent -3.3 to -6.3, GA + Descent -3.6 to -4.8, Teacher-Student -1.2 to -3.2, Random Relabeling -0.1 to -1.4, Boundary Relabeling -7.2 to
-8.1; Retrain -0.2 to -1.0) but also retain (-0.2 to -2.1) and test accuracy (-0.4 to -1.8): it is non-selective degradation of forget-class
samples that unlearning left close to the decision boundary (margin diagnostic), not extra unlearning. avg_gap improves only where the
fp32 model under-forgets (Teacher-Student in all S2 scenarios, Gradient Ascent on S2-10). Random Relabeling is the most stable method,
Boundary Relabeling (C0) and Gradient Ascent the least.
