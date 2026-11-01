# M1 - Gradient Ascent

Last updated: 2026-10-28. Full run list (every run, best row in bold): `records/results/runs_GA.md`. Chronology: `records/JOURNAL.md`.

## 1. Method and implementation
**Code:** `src/mu/unlearn/methods.py::gradient_ascent`, `forward_forget` (reference: `sample_repos/RUM-main/RUM-main/unlearn/GA.py`). Config key: `GA`.
Starting from the Original model, for every forget batch: `loss = -CE(f(x), y)`; SGD step. Forget set only (no retain data).
Our additions: epoch-level early stop (`stop_forget_acc`), iteration-level early stop (`stop_every_iters`: every k steps check
forget acc on a fixed 1000-sample forget subset), gradient clipping (`grad_clip`), BatchNorm options (`bn_mode=frozen`,
`bn_mix_retain`, `bn_recalib`).

## 2. Hyperparameter sources
- No dedicated paper; described as "NegGrad" in the Teacher-Student paper (qualitatively) and as a baseline in:
  Weight Saliency paper: 5 epochs, lr search [1e-5, 1e-3]; RUM paper: 10 epochs, lr [1e-4, 0.1].
- Common to all our runs: batch 256, SGD momentum 0.9, weight decay 5e-4, random crop + flip, bf16. Early-stop target: S1 forget
  acc <= 0 (epoch level) or <= 1% (iteration level); S2 forget acc <= Retrain's forget acc of the same seed.

## 3. Named configurations
| name | lr | epochs | early stop | BatchNorm | clip | where |
|---|---|---|---|---|---|---|
| **G0** main grid | S1 1e-4, S2 1e-5 | 5 | epoch level | train mode (repo) | - | pilot, main grid |
| G0 + recal | as G0 | 5 | epoch | train, then BN re-estimated on 40 retain batches (no weight change) | - | imp |
| **G1** | S1 1e-3, S2 1e-4 | 10 (RUM) | every 2 iterations | train | - | imp, imp2 (3 seeds) |
| G-frozen | 3e-4 or 1e-3 | 10 | every 2 it. | frozen (eval mode) | - | pilot, bnfix |
| G-clip | 3e-4 or 1e-3 | 10 | every 2 it. | train | 1.0 | bnfix |
| **G2** frozen+clip | 1e-3 (also 3e-4) | 10 | every 2 it. | frozen | 1.0 | imp |
| **G3** mix | S1 1e-4 / **3e-4** / 1e-3, S2 1e-4 | 10 | every 2 it. | mixed: retain batch (256) concatenated to every forget forward pass | - | bnfix, imp5 (3 seeds) |
| G3 + clip | 1e-3 | 10 | every 2 it. | mixed | 1.0 | bnfix |

## 4. Results - S1 (class 0), Retrain: forget 0.00, retain 100, test other-cls 94.82, MIA 51.0
Seed 0 unless "3 seeds". sel_gap (selection; no test data) and avg_gap (report): mean |metric - Retrain|. Best in **bold**.
| config | seeds | forget | retain | test other-cls | MIA | sel_gap | avg_gap |
|---|---|---|---|---|---|---|---|
| Original | 3 | 100.00 | 100.00 | 94.76 | 57.78 | 28.51 | 29.05 |
| G0 (main grid) | 3 | 5.33 | 93.91 | 86.99 | 51.17 | 5.55 | 4.58 |
| G0 + WS (main grid) | 3 | 6.30 | 95.47 | 88.37 | 51.75 | 4.97 | 4.22 |
| G0 + recal | 1 | 92.00 | 98.22 | 91.63 | 57.55 | 25.74 | 26.15 |
| G0 with BN frozen (lr 1e-4, 5 ep) | 1 | 99.98 | 100.00 | 94.96 | 61.10 | 29.26 | 29.61 |
| G0 with BN frozen, lr 1e-3 (5 ep, no early stop) | 1 | diverged (NaN) | | | | | |
| G1 (iter stop, lr 1e-3) | 1 | 0.29 | 92.23 | 85.76 | 52.30 | 4.21 | 4.19 |
| G1 | 3 | 0.61 | 91.58 | 84.98 | 51.92 | 5.46 | 4.69 |
| G1 + WS | 3 | 0.50 | 93.44 | 86.67 | 51.45 | 4.27 | 3.74 |
| G1 + recal | 1 | 91.00 | 97.97 | 91.29 | 56.8 | 25.54 | 25.70 |
| G-frozen lr 3e-4 | 1 | 0.00 | 22.23 | 21.42 | 50.5 | 36.35 | 36.24 |
| G-frozen lr 1e-3 | 1 | 0.00 | 75.83 | 71.66 | 51.0 | 12.38 | 11.44 |
| G-clip lr 3e-4 | 1 | 12.80 | 97.18 | 89.93 | 52.7 | 5.40 | 5.00 |
| G-clip lr 1e-3 | 1 | 3.20 | 89.75 | 83.31 | 50.0 | 6.09 | 6.29 |
| G2 frozen+clip lr 1e-3 | 1 | 0.51 | 92.45 | 85.73 | 50.10 | 4.72 | 4.45 |
| G2 frozen+clip lr 3e-4 | 1 | 0.64 | 89.75 | 83.47 | 52.35 | 5.67 | 5.43 |
| G2 + recal | 1 | 99.93 | 99.09 | 92.40 | 54.8 | 27.58 | 27.84 |
| G3 mix lr 1e-4 | 1 | 0.73 | 99.27 | 92.34 | 52.4 | 1.60 | 1.07 |
| G3 mix lr 3e-4 | 1 | 0.44 | 99.13 | 92.34 | 51.6 | 1.55 | 0.88 |
| G3 mix lr 1e-3 | 1 | 0.22 | 98.65 | 91.71 | 50.8 | 2.07 | 1.31 |
| G3 mix + clip lr 1e-3 | 1 | 0.89 | 97.72 | 90.20 | 51.2 | 2.85 | 1.92 |
| G3 mix lr 1e-3 + recal | 1 | 40.76 | 98.94 | 92.83 | 53.1 | 11.36 | 11.28 |
| **G3 mix lr 3e-4** | **3** | **0.45 ± 0.08** | **99.17 ± 0.05** | **92.34** | **51.78** | **1.40 ± 0.15** | **1.07 ± 0.17** |
| G3 mix lr 3e-4 + WS | 3 | 0.62 | 99.15 | 92.40 | 52.13 | 1.21 | 1.18 |

Paper references (other models / training; orientation only):
| source | forget | retain | test |
|---|---|---|---|
| Weight Saliency paper, GA class-wise (ResNet-18, CIFAR-10) | 0.09 | 38.92 | 38.18 |
| Boundary paper, Negative Gradient (All-CNN, CIFAR-10 class) | 7.84 | 97.16 | 80.42 (remaining classes) |
| ours G0 (paper-range settings) | 5.33 | 93.91 | 86.99 |
| ours best (G3 mix, 3 seeds) | 0.45 | 99.17 | 92.34 |

## 5. Results - S2 (random 30% of classes 3, 5, 9), Retrain: forget 90.86, retain 100, test 94.46 (classes 3,5,9: 91.03), MIA 49.95
| config | seeds | forget | retain | test | test classes 3,5,9 | MIA | sel_gap |
|---|---|---|---|---|---|---|---|
| Original | 3 | 100.00 | 100.00 | 94.82 | - | 59.46 | 4.51 |
| G0 (main grid, lr 1e-5) | 3 | 87.47 | 97.16 | 88.31 | 68.33 | 59.76 | 5.12 |
| **G1 (iter stop, lr 1e-4)** | 1 | 91.58 | 98.05 | 89.23 | 71.27 | 59.73 | **4.01** |
| G1 | 3 | 86.17 | 96.92 | 88.12 | 67.53 | 59.34 | 5.51 |
| G2 frozen+clip lr 3e-4 | 1 | 90.99 | 97.95 | 89.10 | 72.33 | 59.07 | 4.16 |
| G3 mix lr 1e-4 | 1 | 91.85 | 98.14 | 89.46 | 72.10 | 59.45 | 4.03 |
| G3 mix lr 1e-4 | 3 | 90.53 | 97.93 | 89.20 | - | 59.62 | 4.01 |
| G3 mix lr 1e-4, on S2-10 | 3 | 95.41 | 99.00 | 90.67 | - | 60.10 | 4.11 |
| G3 mix lr 1e-4, on S2-50 | 3 | 88.32 | 98.01 | 88.64 | - | 59.46 | 4.06 |
| G-frozen lr 1e-3 (no clip) | 1 | 8.77 | 64.22 | 55.55 | 6.67 | 52.68 | 40.26 |
| G1 + recal | 1 | 99.31 | 99.14 | 93.01 | 90.73 | 57.40 | 4.06 |
In S2 no GA variant beats "doing nothing" by much (Original 4.51): GA lowers the accuracy of the remaining same-class TEST
images (91 -> ~70%) as much as that of the forget images, and the MIA stays at ~59% (Retrain 50%).

## 6. Issues found and how they were resolved
**(a) BatchNorm-statistics artifact.** Forget-only batches (S1: all airplanes) overwrite the BN running statistics within ~30 steps
(running = 0.9 running + 0.1 batch); at test time all images are normalised with airplane statistics, which suppresses
airplane-specific activations. Evidence (S1, G0): relative weight change **2.0%** vs BN running mean **23.9%** / var 14.5%;
re-estimating only the BN statistics on retain data restores forget acc from 5.3% to **92%**; with BN frozen the same lr does
nothing (forget 99.98%). So the main-grid GA "forgetting" was mostly BN drift. Not mentioned in any of the papers; the
reference repos use BN train mode. In the papers' random-forgetting setting forget batches contain all classes, so BN barely
drifts - which also explains why our S2 GA forgets more than the Weight Saliency paper's GA (forget 99.3% at 10% random).

**(b) Frozen BN alone is unstable; it needs gradient clipping.** Same lr / epochs / early stop, one factor changed:
frozen lr 3e-4 -> retain 22%; frozen lr 1e-3 -> retain 76%; frozen + clip lr 1e-3 -> retain 92.5% (G2). The GA loss is unbounded;
in BN train mode the batch normalisation rescales the growing activations, in eval mode nothing does, so activations and
gradients explode once the forget logits start to fall; clipping bounds the step. (The earlier comparison "pilot frozen vs imp
frozen+clip" also differed in epochs and early stopping; the controlled runs above isolate clipping.)

**(c) Fix that works: mixed BN batches (G3).** A retain batch is concatenated to every forget forward pass, so BN normalises with
and updates from representative statistics; the loss is still only -CE on the forget part (no retain loss, i.e. still pure GA).
S1: forget 0.44, retain 99.13, test other 92.34, sel_gap 1.55 (G0 5.55, G1 4.21, G2 4.72). Clipping is not needed with mixed BN
(G3 + clip is worse). Caveat: robustness check G3 + recal -> forget 40.8% (partial relapse; G0/G1/G2 relapse to 91-100%), so
part of the effect still depends on the BN statistics; methods with a retain loss (GA + Descent, Random Relabeling, Boundary C1)
are the robust ones.

**(d) Overshooting within an epoch.** With epoch-level early stopping GA at lr >= 3e-4 passes forget acc 0 long before the epoch ends
and keeps damaging the retain set. Resolution: iteration-level early stop (G1/G3): stops after 16-50 iterations in S1.

**(e) Divergence.** BN-frozen lr 1e-3 without clipping (pilot) produced NaN logits; NaN logits made argmax return class 0 and
reported a false 100% forget accuracy. Resolution: `diverged` flag in metrics + NaN-safe MIA; such runs are excluded from selection.

## 7. Comparison with the other methods (S1, 3-seed main grid; `records/results/main_table.md`)
| method | forget | retain | test other-cls | MIA | avg_gap |
|---|---|---|---|---|---|
| M1 Gradient Ascent (G0) | 5.33 | 93.91 | 86.99 | 51.17 | 4.58 |
| M2 GA + Descent | 0.00 | 99.09 | 91.87 | 49.25 | 1.34 |
| M3a Teacher-Student | 0.33 | 99.99 | 94.30 | 52.58 | 0.60 |
| M3b-R Random Relabeling | 0.00 | 100.00 | 94.54 | 55.47 | 1.19 |
| M3b-B Boundary Relabeling (main) | 23.59 | 99.48 | 92.31 | 54.28 | 6.92 |
| **M1 improved G3 mix (3 seeds)** | 0.45 | 99.17 | 92.34 | 51.78 | **1.07** |

## 8. Conclusion / recommended setting
- **S1: G3** (mixed BN batches, lr 3e-4, 10 epochs, iteration-level early stop): 3 seeds forget 0.45, retain 99.17, test other-cls
  92.34, avg_gap 1.07 (main grid G0: 4.58). Still not fully robust (partial relapse after BN recalibration) and still ~2.5 points
  below Retrain on the other classes - GA has no retain loss.
- S2: GA cannot do better than ~4.0 (doing nothing: 4.5); it damages same-class test images as much as forget images.

## 9. Evaluation results (updated 2026-10-30; full details `records/results/evaluation.md`)
Best configuration used: S1 G3 (mixed BN, lr 3e-4, 10 ep, iteration-level early stop); S2 G3-S2 (mixed BN, lr 1e-4); main-grid models (G0) for perturbation.
3 seeds, mean. avg_gap = mean |metric - Retrain| (forget, retain, test acc, MIA).

### 9.1 Entanglement Exp 1 - each partition unlearned separately
S1: corrected setup (other two partitions excluded from forget and retain; reference = S1 Retrain on the partition). S2: RUM protocol
(other partitions in the retain set; Retrain per partition). Per-partition accuracies and MIA:
| scenario | model | WS | forget acc: low | forget acc: medium | forget acc: high | S1 only: acc on the 2 excluded partitions, low run | S1 only: excluded acc, medium run | S1 only: excluded acc, high run | retain acc (mean of 3 runs) | test acc (mean of 3 runs) | MIA: low | MIA: medium | MIA: high |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| S1c0 | Retrain (reference) |  | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 100.0 | 85.4 | 53.1 | 49.2 | 49.7 |
| S1c0 | Original (no unlearning) |  | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 94.8 | 58.0 | 57.9 | 57.3 |
| S1c0 | M1 Gradient Ascent |  | 0.4 | 0.4 | 0.7 | 0.2 | 0.4 | 5.4 | 99.1 | 83.2 | 51.4 | 50.5 | 49.6 |
| S1c0 | M1 Gradient Ascent | yes | 0.2 | 0.1 | 0.4 | 0.1 | 0.2 | 3.2 | 98.8 | 82.9 | 53.0 | 49.8 | 49.3 |
| S2k359p10 | Retrain (reference) |  | 77.6 | 94.6 | 99.6 | - | - | - | 100.0 | 93.5 | 57.3 | 51.3 | 56.7 |
| S2k359p10 | Original (no unlearning) |  | 100.0 | 100.0 | 100.0 | - | - | - | 100.0 | 94.8 | 57.7 | 57.7 | 57.7 |
| S2k359p10 | M1 Gradient Ascent |  | 94.4 | 100.0 | 99.3 | - | - | - | 99.5 | 92.2 | 54.3 | 63.1 | 59.7 |
| S2k359p10 | M1 Gradient Ascent | yes | 95.6 | 100.0 | 99.1 | - | - | - | 99.5 | 92.3 | 54.6 | 63.1 | 59.7 |
| S2k359p30 | Retrain (reference) |  | 79.0 | 94.7 | 99.6 | - | - | - | 100.0 | 94.8 | 56.7 | 52.1 | 56.7 |
| S2k359p30 | Original (no unlearning) |  | 100.0 | 100.0 | 100.0 | - | - | - | 100.0 | 94.8 | 58.9 | 59.0 | 59.0 |
| S2k359p30 | M1 Gradient Ascent |  | 77.7 | 93.2 | 99.2 | - | - | - | 98.4 | 90.8 | 50.5 | 60.7 | 60.5 |
| S2k359p30 | M1 Gradient Ascent | yes | 76.6 | 96.5 | 99.4 | - | - | - | 98.6 | 91.0 | 50.3 | 61.6 | 60.8 |
| S2k359p50 | Retrain (reference) |  | 77.8 | 94.3 | 99.4 | - | - | - | 100.0 | 94.5 | 56.3 | 51.8 | 57.2 |
| S2k359p50 | Original (no unlearning) |  | 100.0 | 100.0 | 100.0 | - | - | - | 100.0 | 94.8 | 59.1 | 59.3 | 59.4 |
| S2k359p50 | M1 Gradient Ascent |  | 76.5 | 94.2 | 99.0 | - | - | - | 98.4 | 90.7 | 50.6 | 61.4 | 60.5 |
| S2k359p50 | M1 Gradient Ascent | yes | 76.6 | 94.0 | 99.1 | - | - | - | 98.4 | 90.6 | 50.6 | 61.4 | 60.7 |

avg_gap to the partition's Retrain:
| scenario | model | WS | low | medium | high | high - low |
|---|---|---|---|---|---|---|
| S1c0 | Original (no unlearning) | | 28.59 | 29.52 | 29.27 | +0.67 |
| S1c0 | (Retrain forget acc on partition, %) | | 0.0 | 0.0 | 0.0 | |
| S1c0 | M1 Gradient Ascent |  | 1.50 | 1.30 | 0.84 | -0.66 |
| S1c0 | M1 Gradient Ascent | yes | 1.38 | 1.40 | 0.91 | -0.47 |
| S2k359p10 | Original (no unlearning) | | 6.24 | 3.32 | 0.67 | -5.58 |
| S2k359p10 | (Retrain forget acc on partition, %) | | 77.6 | 94.6 | 99.6 | |
| S2k359p10 | M1 Gradient Ascent |  | 5.42 | 4.67 | 1.28 | -4.13 |
| S2k359p10 | M1 Gradient Ascent | yes | 5.63 | 4.66 | 1.35 | -4.28 |
| S2k359p30 | Original (no unlearning) | | 5.85 | 3.06 | 0.75 | -5.10 |
| S2k359p30 | (Retrain forget acc on partition, %) | | 79.0 | 94.7 | 99.6 | |
| S2k359p30 | M1 Gradient Ascent |  | 3.37 | 4.42 | 1.91 | -1.46 |
| S2k359p30 | M1 Gradient Ascent | yes | 3.73 | 4.39 | 1.90 | -1.83 |
| S2k359p50 | Original (no unlearning) | | 6.43 | 3.34 | 0.76 | -5.67 |
| S2k359p50 | (Retrain forget acc on partition, %) | | 77.8 | 94.3 | 99.4 | |
| S2k359p50 | M1 Gradient Ascent |  | 3.07 | 4.44 | 1.76 | -1.31 |
| S2k359p50 | M1 Gradient Ascent | yes | 3.06 | 4.56 | 1.78 | -1.28 |

### 9.2 Entanglement Exp 2 - sequential low -> medium -> high vs one-shot (final model, whole forget set)
| scenario | WS | mode | forget | retain | test | MIA | avg_gap |
|---|---|---|---|---|---|---|---|
| S1c0 |  | one-shot | 0.45 | 99.17 | 83.17 | 51.78 | 1.07 |
| S1c0 |  | sequential | 0.53 | 98.71 | 82.41 | 50.50 | 1.34 |
| S1c0 | yes | one-shot | 0.62 | 99.15 | 83.24 | 52.13 | 1.18 |
| S1c0 | yes | sequential | 0.53 | 98.61 | 82.61 | 52.43 | 1.53 |
| S2k359p10 |  | one-shot | 95.41 | 99.00 | 90.67 | 60.10 | 4.80 |
| S2k359p10 |  | sequential | 88.69 | 96.93 | 89.20 | 58.49 | 5.07 |
| S2k359p10 | yes | one-shot | 96.74 | 99.32 | 91.13 | 60.40 | 5.01 |
| S2k359p10 | yes | sequential | 89.88 | 97.25 | 89.56 | 58.56 | 4.70 |
| S2k359p30 |  | one-shot | 90.53 | 97.93 | 89.20 | 59.62 | 4.40 |
| S2k359p30 |  | sequential | 81.41 | 95.82 | 87.19 | 58.51 | 7.37 |
| S2k359p30 | yes | one-shot | 90.47 | 97.92 | 89.22 | 59.64 | 4.54 |
| S2k359p30 | yes | sequential | 82.51 | 96.06 | 87.43 | 58.56 | 6.98 |
| S2k359p50 |  | one-shot | 88.32 | 98.01 | 88.64 | 59.46 | 4.18 |
| S2k359p50 |  | sequential | 80.74 | 96.67 | 86.93 | 58.71 | 6.53 |
| S2k359p50 | yes | one-shot | 88.07 | 97.96 | 88.51 | 59.51 | 4.18 |
| S2k359p50 | yes | sequential | 81.01 | 96.70 | 86.94 | 58.67 | 6.44 |

### 9.3 Linear probe / embedding separability (S1; figure `records/figures/fig_embed_best_S1c0.png`)
| model | output forget-cls | probe forget-cls | NCC forget-cls | probe other-cls | silhouette class 0 | 10-NN purity class 0 |
|---|---|---|---|---|---|---|
| Original | 95.40 ± 0.00 | 95.50 ± 0.17 | 95.40 ± 0.00 | 94.69 ± 0.05 | 0.127 ± 0.000 | 93.00 ± 0.00 |
| Retrain | 0.00 ± 0.00 | 87.73 ± 0.87 | 41.43 ± 2.97 | 91.19 ± 0.08 | -0.040 ± 0.005 | 52.58 ± 1.06 |
| GA (G3) | 0.60 ± 0.00 | 86.20 ± 0.44 | 30.33 ± 1.12 | 92.14 ± 0.16 | 0.025 ± 0.003 | 32.63 ± 0.77 |
| GA (G3) + WS | 0.80 ± 0.61 | 85.60 ± 0.95 | 34.80 ± 1.82 | 92.43 ± 0.17 | -0.022 ± 0.017 | 29.19 ± 2.20 |

### 9.4 Perturbed forget samples (main-grid models; Gaussian tau, 20 perturbations per image)
| scenario | model | WS | clean forget acc | perturbed forget acc tau 0.03 | recovery tau 0.03 | perturbed tau 0.1 | recovery tau 0.1 |
|---|---|---|---|---|---|---|---|
| S1c0 | Original |  | 100.0 | 80.6 | 0.0 | 8.7 | 0.0 |
| S1c0 | Retrain |  | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| S1c0 | M1 Gradient Ascent |  | 5.3 | 0.0 | 0.0 | 0.0 | 0.0 |
| S1c0 | M1 Gradient Ascent | yes | 6.3 | 0.0 | 0.0 | 0.0 | 0.0 |
| S2k359p30 | Original |  | 100.0 | 91.6 | 0.0 | 8.2 | 0.0 |
| S2k359p30 | Retrain |  | 90.9 | 78.4 | 44.5 | 7.1 | 8.6 |
| S2k359p30 | M1 Gradient Ascent |  | 87.5 | 45.5 | 21.7 | 0.2 | 0.0 |
| S2k359p30 | M1 Gradient Ascent | yes | 87.7 | 45.7 | 21.2 | 0.2 | 0.0 |

### 9.5 Post-unlearning quantization (best configurations; weight-only per-channel, BN in fp32; details `records/results/quant_best.md`)
| scenario | model | WS | config | forget fp32 | forget INT8 | forget INT6 | forget INT4 | change forget (INT4 - fp32) | change retain | change test | change MIA | avg_gap fp32 | avg_gap INT4 | share Q(theta_u)=Q(theta_o), INT4 | mean abs(delta)/step, INT4 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| S1c0 | Original |  |  | 100.00 | 100.00 | 100.00 | 100.00 | +0.00 | +0.00 | -0.33 | -0.60 | 29.05 | 28.82 | 1.000 | 0.000 |
| S1c0 | Retrain |  |  | 0.00 | 0.00 | 0.00 | 0.00 | +0.00 | -0.01 | -0.34 | -0.07 | 0.00 | 0.10 | 0.102 | 15.126 |
| S1c0 | M1 Gradient Ascent |  | G3 | 0.46 | 0.44 | 0.41 | 0.39 | -0.07 | -0.42 | -0.71 | +0.23 | 1.06 | 1.39 | 0.310 | 0.012 |
| S1c0 | M1 Gradient Ascent | yes | G3 | 0.63 | 0.63 | 0.62 | 0.59 | -0.04 | -0.42 | -0.82 | +0.12 | 1.17 | 1.50 | 0.495 | 0.010 |
| S2k359p10 | Original |  |  | 100.00 | 100.00 | 100.00 | 100.00 | +0.00 | +0.00 | -0.33 | +0.72 | 4.47 | 4.63 | 1.000 | 0.000 |
| S2k359p10 | Retrain |  |  | 91.63 | 91.63 | 91.46 | 90.59 | -1.04 | +0.00 | -0.28 | -0.16 | 0.00 | 0.37 | 0.104 | 17.137 |
| S2k359p10 | M1 Gradient Ascent |  | G3-S2 | 95.38 | 95.43 | 94.81 | 91.43 | -3.95 | -1.00 | -0.98 | -0.98 | 4.79 | 4.15 | 0.313 | 0.003 |
| S2k359p10 | M1 Gradient Ascent | yes | G3-S2 | 96.74 | 96.79 | 96.42 | 93.43 | -3.31 | -0.87 | -1.02 | -1.05 | 5.01 | 4.40 | 0.489 | 0.002 |
| S2k359p30 | Original |  |  | 100.00 | 100.00 | 100.00 | 100.00 | +0.00 | +0.00 | -0.33 | +0.72 | 4.75 | 4.85 | 1.000 | 0.000 |
| S2k359p30 | Retrain |  |  | 90.86 | 90.92 | 90.94 | 90.63 | -0.23 | -0.00 | -0.33 | +0.15 | 0.00 | 0.18 | 0.102 | 14.628 |
| S2k359p30 | M1 Gradient Ascent |  | G3-S2 | 90.55 | 90.55 | 89.51 | 85.26 | -5.29 | -1.17 | -1.28 | -0.81 | 4.32 | 6.05 | 0.313 | 0.004 |
| S2k359p30 | M1 Gradient Ascent | yes | G3-S2 | 90.46 | 90.41 | 89.55 | 84.86 | -5.60 | -1.24 | -1.39 | -0.86 | 4.36 | 6.20 | 0.485 | 0.003 |
| S2k359p50 | Original |  |  | 100.00 | 100.00 | 100.00 | 100.00 | +0.00 | +0.00 | -0.33 | +0.70 | 5.26 | 5.36 | 1.000 | 0.000 |
| S2k359p50 | Retrain |  |  | 88.74 | 88.75 | 88.69 | 87.99 | -0.75 | +0.00 | -0.33 | +0.03 | 0.00 | 0.28 | 0.101 | 14.567 |
| S2k359p50 | M1 Gradient Ascent |  | G3-S2 | 88.31 | 88.35 | 86.96 | 82.00 | -6.31 | -1.10 | -1.70 | -0.88 | 4.06 | 6.11 | 0.313 | 0.005 |
| S2k359p50 | M1 Gradient Ascent | yes | G3-S2 | 88.05 | 88.07 | 86.81 | 81.98 | -6.07 | -1.06 | -1.55 | -0.81 | 4.19 | 6.15 | 0.482 | 0.004 |

### 9.6 Observations
- MIA: best configuration 51.78% in S1 (Retrain 51.0); in S2 ~59.6% (Retrain ~50) - GA does not make S2 forget samples look unseen.
- Exp 1, S1 (corrected setup: other partitions excluded): forget acc 0.4 / 0.4 / 0.7% (low / medium / high; WS 0.2 / 0.1 / 0.4%), MIA
  51.4 / 50.5 / 49.6%, avg_gap 1.50 / 1.30 / 0.84 - no harder on the high partition. The forgetting spreads to the 3000 unused airplanes
  (0.2% after the low run) but less after the high run (5.4%). S2: GA closes most of the gap on the low-entangled (memorised) partition (S2-30 low 3.37 vs Original
  5.85; S2-50 3.07 vs 6.43) and adds little damage on the high partition (1.3-1.9 vs 0.7).
- Exp 2: sequential low -> high is worse than one-shot in every scenario except GA + WS on S2-10 (e.g. S2-30 7.37 vs 4.40): repeated
  ascent steps keep moving the earlier partitions.
- Linear probe (G3): output 0.6%, probe 86.2% (Retrain 87.7%), NCC 30.3% (Retrain 41.4%), 10-NN purity 32.6% (Retrain 52.6%): the airplane
  cluster is dispersed even more than in Retrain.
- Perturbation (main-grid G0): S1 0% on perturbed forget images; S2-30 45.5% vs Retrain 78.4% -> over-unlearning.
- Quantization (best configurations G3 / G3-S2): S1 no relapse (INT4 forget 0.46 -> 0.39%, test -0.71). S2: INT4 lowers forget acc by
  3.3-6.3 points (S2-30 90.55 -> 85.26%, below Retrain 90.86%) together with retain (-1.0 to -1.2) and test acc (-1.0 to -1.7). The update
  is tiny (0.002-0.005 INT4 steps) and leaves 16.5% of the correctly classified forget images low-margin (Retrain 5.3%); INT4 flips 5.5%
  of the forget and 4.0% of the same-class test images (S2-30 seed 0). avg_gap S2-10 4.79 -> 4.15, S2-30 4.32 -> 6.05, S2-50 4.06 -> 6.11.
  (The main-grid G0 model stored its forgetting in the BN running statistics; BN recalibration restored 92% forget acc, section 6.)
