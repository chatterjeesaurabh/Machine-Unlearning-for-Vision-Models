# M3b-R - Random Relabeling

Last updated: 2026-10-28. Chronology: `records/JOURNAL.md`. Pilot `records/results/pilot_all.csv`, Stage 5 `records/results/improve_table.md`.

## 1. Method and implementation
**Code:** `src/mu/unlearn/methods.py::random_relabel` (reference: `sample_repos/Unlearn-Saliency-master/Classification/unlearn/RL.py`). Config key: `RLrand`.
Each epoch: one pass over the forget set with random WRONG labels `y' = (y + U{1..9}) mod 10` (paper: y' != y; the repo's
`randint` can return y), then one pass over the retain set with true labels (as in the repo; the paper's eq. 6 weights the retain
term by alpha, the repo does not). Random Relabeling + Weight Saliency mask = the Weight Saliency paper's proposed method.

## 2. Hyperparameter sources
Weight Saliency paper: 10 epochs, lr search [1e-3, 1e-1]; repo README lr 0.013 for 10% random forgetting (with mask). RUM: 10 ep,
lr [0.01, 0.1]. Ours: 10 epochs, batch 256, SGD momentum 0.9, wd 5e-4, augmentation; pilot lr {0.005, 0.01, 0.02, 0.05}.

## 3. Configurations
| name | lr | where |
|---|---|---|
| **R1** main grid S1 | 0.005 | pilot, main grid |
| **R1-S2** main grid S2 | 0.02 (selected on S2-30) | main grid |
| R2 S2-50 | 0.005 or 0.01 (0.01 better) | imp |
| R1 + recal | 0.005 + BN re-estimated on retain (robustness check) | bnfix |

## 4. Results (avg_gap = mean |metric - Retrain|; best in bold)
| scenario | config | seeds | forget | retain | test | MIA | avg_gap |
|---|---|---|---|---|---|---|---|
| S1 | Retrain (reference) | 3 | 0.00 | 100.00 | 85.38 | 51.02 | 0 |
| S1 | R1 | 3 | 0.00 | 100.00 | 85.09 | 55.47 | 1.19 |
| S1 | R1 + WS | 3 | 0.00 | 100.00 | 85.18 | 55.48 | **1.17** |
| S1 | R1 + recal | 1 | 0.00 | 99.34 | 84.02 | 53.35 | 0.91 |
| S2-10 (Retrain 91.63) | R1-S2 | 3 | 95.16 | 97.80 | 90.69 | 55.62 | **3.91** |
| S2-30 (Retrain 90.86) | R1-S2 | 3 | 88.67 | 96.51 | 90.34 | 53.16 | 3.25 |
| S2-30 | R1-S2 + WS | 3 | 94.35 | 98.81 | 91.83 | 55.04 | **3.10** |
| S2-50 (Retrain 88.74) | R1-S2 (lr 0.02) | 3 | 81.07 ± 5.60 | 93.31 | 87.42 | 51.78 | 5.43 |
| S2-50 | R1-S2 + WS | 3 | 88.96 | 97.67 | 90.46 | 53.74 | **2.58** |
| S2-50 | R2 lr 0.01 | 1 | 98.83 | 99.85 | 93.39 | 57.28 | 4.17 |

Paper references (Weight Saliency paper, ResNet-18 CIFAR-10): class-wise RL forget 10.67 / retain 99.92 / test 94.52;
RL + mask (their method) forget 0.09 / retain 99.93 / test 94.56; random 10% (across all classes) RL forget 92.39 / retain 99.67.
Ours S1: forget 0.00 / retain 100 (better than their RL, equal to their RL + mask).

## 5. Issues and observations
- Most reliable method overall: matches Retrain in S1 and is the best or second-best in every S2 ratio.
- MIA in S1 is 55.5% (Retrain 51%): random wrong labels make forget samples low-confidence / high-loss in a recognisable way
  (the "Streisand effect" discussed in the slides); Boundary C1-mix reaches 51.0%.
- lr transfers badly across forget-set sizes: lr 0.02 (selected on S2-30) is too aggressive for S2-50 (forget 81 ± 5.6, high
  variance); the Weight Saliency mask stabilises it (2.58) - consistent with the paper's argument for the mask.
- BN: the last pass of every epoch is a retain pass, so BN statistics stay calibrated; re-estimating BN on retain keeps forget
  acc at 0.00 (robust).
- Representations: output forget 0.03% but probe forget-class accuracy 94.3% and NCC 82% (Retrain 87.7 / 41.4): the class is
  still fully decodable from the features (`records/results/repr_table.md`), as reported by the Illusion paper for SalUn/RL.

## 6. Comparison (S1, 3 seeds, avg_gap): GA 4.58 (improved 1.07) | GA+Descent 1.34 | Teacher-Student 0.60 |
**Random Relabeling 1.19 (+WS 1.17)** | Boundary 6.92 (improved C1-mix 0.16).

## 9. Evaluation results (updated 2026-10-30; full details `records/results/evaluation.md`)
Best configuration used: S1 R1 (lr 0.005); S2 R1-S2 (lr 0.02); main-grid models (R1 / R1-S2) for perturbation.
3 seeds, mean. avg_gap = mean |metric - Retrain| (forget, retain, test acc, MIA).

### 9.1 Entanglement Exp 1 - each partition unlearned separately
S1: corrected setup (other two partitions excluded from forget and retain; reference = S1 Retrain on the partition). S2: RUM protocol
(other partitions in the retain set; Retrain per partition). Per-partition accuracies and MIA:
| scenario | model | WS | forget acc: low | forget acc: medium | forget acc: high | S1 only: acc on the 2 excluded partitions, low run | S1 only: excluded acc, medium run | S1 only: excluded acc, high run | retain acc (mean of 3 runs) | test acc (mean of 3 runs) | MIA: low | MIA: medium | MIA: high |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| S1c0 | Retrain (reference) |  | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 100.0 | 85.4 | 53.1 | 49.2 | 49.7 |
| S1c0 | Original (no unlearning) |  | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 94.8 | 58.0 | 57.9 | 57.3 |
| S1c0 | M3b-R Random Relabeling |  | 2.8 | 0.6 | 0.5 | 0.2 | 1.4 | 2.2 | 100.0 | 85.4 | 58.3 | 50.6 | 54.4 |
| S1c0 | M3b-R Random Relabeling | yes | 17.4 | 1.9 | 0.8 | 5.1 | 4.0 | 3.2 | 100.0 | 85.7 | 62.0 | 52.6 | 54.2 |
| S2k359p10 | Retrain (reference) |  | 77.6 | 94.6 | 99.6 | - | - | - | 100.0 | 93.5 | 57.3 | 51.3 | 56.7 |
| S2k359p10 | Original (no unlearning) |  | 100.0 | 100.0 | 100.0 | - | - | - | 100.0 | 94.8 | 57.7 | 57.7 | 57.7 |
| S2k359p10 | M3b-R Random Relabeling |  | 82.9 | 93.6 | 95.3 | - | - | - | 92.8 | 86.6 | 52.3 | 54.8 | 60.5 |
| S2k359p10 | M3b-R Random Relabeling | yes | 88.7 | 96.3 | 94.7 | - | - | - | 96.2 | 89.2 | 49.2 | 56.1 | 59.3 |
| S2k359p30 | Retrain (reference) |  | 79.0 | 94.7 | 99.6 | - | - | - | 100.0 | 94.8 | 56.7 | 52.1 | 56.7 |
| S2k359p30 | Original (no unlearning) |  | 100.0 | 100.0 | 100.0 | - | - | - | 100.0 | 94.8 | 58.9 | 59.0 | 59.0 |
| S2k359p30 | M3b-R Random Relabeling |  | 89.5 | 96.9 | 98.9 | - | - | - | 97.8 | 90.9 | 49.4 | 56.9 | 61.3 |
| S2k359p30 | M3b-R Random Relabeling | yes | 92.2 | 98.6 | 99.7 | - | - | - | 99.0 | 91.9 | 51.5 | 57.4 | 62.0 |
| S2k359p50 | Retrain (reference) |  | 77.8 | 94.3 | 99.4 | - | - | - | 100.0 | 94.5 | 56.3 | 51.8 | 57.2 |
| S2k359p50 | Original (no unlearning) |  | 100.0 | 100.0 | 100.0 | - | - | - | 100.0 | 94.8 | 59.1 | 59.3 | 59.4 |
| S2k359p50 | M3b-R Random Relabeling |  | 86.0 | 97.1 | 99.3 | - | - | - | 98.2 | 91.1 | 48.8 | 56.5 | 61.0 |
| S2k359p50 | M3b-R Random Relabeling | yes | 89.9 | 97.9 | 99.9 | - | - | - | 99.1 | 92.0 | 50.8 | 57.5 | 61.8 |

avg_gap to the partition's Retrain:
| scenario | model | WS | low | medium | high | high - low |
|---|---|---|---|---|---|---|
| S1c0 | Original (no unlearning) | | 28.59 | 29.52 | 29.27 | +0.67 |
| S1c0 | (Retrain forget acc on partition, %) | | 0.0 | 0.0 | 0.0 | |
| S1c0 | M3b-R Random Relabeling |  | 2.06 | 0.53 | 1.36 | -0.69 |
| S1c0 | M3b-R Random Relabeling | yes | 6.75 | 1.35 | 1.40 | -5.34 |
| S2k359p10 | Original (no unlearning) | | 6.24 | 3.32 | 0.67 | -5.58 |
| S2k359p10 | (Retrain forget acc on partition, %) | | 77.6 | 94.6 | 99.6 | |
| S2k359p10 | M3b-R Random Relabeling |  | 4.26 | 4.63 | 7.78 | +3.52 |
| S2k359p10 | M3b-R Random Relabeling | yes | 6.17 | 3.35 | 4.81 | -1.36 |
| S2k359p30 | Original (no unlearning) | | 5.85 | 3.06 | 0.75 | -5.10 |
| S2k359p30 | (Retrain forget acc on partition, %) | | 79.0 | 94.7 | 99.6 | |
| S2k359p30 | M3b-R Random Relabeling |  | 5.49 | 3.26 | 3.33 | -2.16 |
| S2k359p30 | M3b-R Random Relabeling | yes | 5.39 | 3.03 | 2.79 | -2.60 |
| S2k359p50 | Original (no unlearning) | | 6.43 | 3.34 | 0.76 | -5.67 |
| S2k359p50 | (Retrain forget acc on partition, %) | | 77.8 | 94.3 | 99.4 | |
| S2k359p50 | M3b-R Random Relabeling |  | 4.95 | 3.29 | 2.49 | -2.46 |
| S2k359p50 | M3b-R Random Relabeling | yes | 5.09 | 3.23 | 2.24 | -2.85 |

### 9.2 Entanglement Exp 2 - sequential low -> medium -> high vs one-shot (final model, whole forget set)
| scenario | WS | mode | forget | retain | test | MIA | avg_gap |
|---|---|---|---|---|---|---|---|
| S1c0 |  | one-shot | 0.00 | 100.00 | 85.09 | 55.47 | 1.19 |
| S1c0 |  | sequential | 97.94 | 100.00 | 93.55 | 59.50 | 28.65 |
| S1c0 | yes | one-shot | 0.00 | 100.00 | 85.18 | 55.48 | 1.17 |
| S1c0 | yes | sequential | 96.44 | 100.00 | 93.42 | 58.98 | 28.11 |
| S2k359p10 |  | one-shot | 95.16 | 97.80 | 90.69 | 55.62 | 3.91 |
| S2k359p10 |  | sequential | 93.38 | 98.25 | 90.81 | 54.23 | 2.98 |
| S2k359p10 | yes | one-shot | 97.23 | 98.78 | 91.47 | 57.07 | 4.36 |
| S2k359p10 | yes | sequential | 94.67 | 99.01 | 91.84 | 54.05 | 2.81 |
| S2k359p30 |  | one-shot | 88.67 | 96.51 | 90.34 | 53.16 | 3.25 |
| S2k359p30 |  | sequential | 93.80 | 98.60 | 91.32 | 53.15 | 2.67 |
| S2k359p30 | yes | one-shot | 94.35 | 98.81 | 91.83 | 55.04 | 3.10 |
| S2k359p30 | yes | sequential | 93.14 | 98.90 | 91.32 | 54.20 | 2.69 |
| S2k359p50 |  | one-shot | 81.07 | 93.31 | 87.42 | 51.78 | 5.43 |
| S2k359p50 |  | sequential | 92.84 | 98.90 | 91.57 | 53.56 | 2.55 |
| S2k359p50 | yes | one-shot | 88.96 | 97.67 | 90.46 | 53.74 | 2.58 |
| S2k359p50 | yes | sequential | 91.52 | 98.74 | 90.95 | 54.10 | 2.55 |

### 9.3 Linear probe / embedding separability (S1; figure `records/figures/fig_embed_best_S1c0.png`)
| model | output forget-cls | probe forget-cls | NCC forget-cls | probe other-cls | silhouette class 0 | 10-NN purity class 0 |
|---|---|---|---|---|---|---|
| Original | 95.40 ± 0.00 | 95.50 ± 0.17 | 95.40 ± 0.00 | 94.69 ± 0.05 | 0.127 ± 0.000 | 93.00 ± 0.00 |
| Retrain | 0.00 ± 0.00 | 87.73 ± 0.87 | 41.43 ± 2.97 | 91.19 ± 0.08 | -0.040 ± 0.005 | 52.58 ± 1.06 |
| Random Relabel (R1) | 0.03 ± 0.06 | 94.30 ± 0.69 | 82.10 ± 0.95 | 94.01 ± 0.08 | 0.175 ± 0.015 | 89.24 ± 0.64 |
| Random Relabel (R1) + WS | 0.03 ± 0.06 | 94.17 ± 0.12 | 80.33 ± 1.22 | 94.00 ± 0.08 | 0.198 ± 0.007 | 89.41 ± 0.64 |

### 9.4 Perturbed forget samples (main-grid models; Gaussian tau, 20 perturbations per image)
| scenario | model | WS | clean forget acc | perturbed forget acc tau 0.03 | recovery tau 0.03 | perturbed tau 0.1 | recovery tau 0.1 |
|---|---|---|---|---|---|---|---|
| S1c0 | Original |  | 100.0 | 80.6 | 0.0 | 8.7 | 0.0 |
| S1c0 | Retrain |  | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| S1c0 | M3b-R Random Relabeling |  | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| S1c0 | M3b-R Random Relabeling | yes | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| S2k359p30 | Original |  | 100.0 | 91.6 | 0.0 | 8.2 | 0.0 |
| S2k359p30 | Retrain |  | 90.9 | 78.4 | 44.5 | 7.1 | 8.6 |
| S2k359p30 | M3b-R Random Relabeling |  | 88.7 | 71.0 | 41.4 | 7.2 | 9.3 |
| S2k359p30 | M3b-R Random Relabeling | yes | 94.3 | 78.4 | 42.2 | 9.1 | 9.7 |

### 9.5 Post-unlearning quantization (best configurations; weight-only per-channel, BN in fp32; details `records/results/quant_best.md`)
| scenario | model | WS | config | forget fp32 | forget INT8 | forget INT6 | forget INT4 | change forget (INT4 - fp32) | change retain | change test | change MIA | avg_gap fp32 | avg_gap INT4 | share Q(theta_u)=Q(theta_o), INT4 | mean abs(delta)/step, INT4 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| S1c0 | Original |  |  | 100.00 | 100.00 | 100.00 | 100.00 | +0.00 | +0.00 | -0.33 | -0.60 | 29.05 | 28.82 | 1.000 | 0.000 |
| S1c0 | Retrain |  |  | 0.00 | 0.00 | 0.00 | 0.00 | +0.00 | -0.01 | -0.34 | -0.07 | 0.00 | 0.10 | 0.102 | 15.126 |
| S1c0 | M3b-R Random Relabeling |  | R1 | 0.00 | 0.00 | 0.00 | 0.01 | +0.01 | -0.01 | -0.30 | -0.02 | 1.19 | 1.26 | 0.241 | 0.200 |
| S1c0 | M3b-R Random Relabeling | yes | R1 | 0.00 | 0.00 | 0.00 | 0.00 | +0.00 | -0.00 | -0.27 | +0.50 | 1.17 | 1.36 | 0.357 | 0.144 |
| S2k359p10 | Original |  |  | 100.00 | 100.00 | 100.00 | 100.00 | +0.00 | +0.00 | -0.33 | +0.72 | 4.47 | 4.63 | 1.000 | 0.000 |
| S2k359p10 | Retrain |  |  | 91.63 | 91.63 | 91.46 | 90.59 | -1.04 | +0.00 | -0.28 | -0.16 | 0.00 | 0.37 | 0.104 | 17.137 |
| S2k359p10 | M3b-R Random Relabeling |  | R1-S2 | 95.16 | 95.26 | 95.14 | 93.78 | -1.38 | -0.77 | -0.72 | -0.53 | 3.91 | 3.80 | 0.180 | 0.758 |
| S2k359p10 | M3b-R Random Relabeling | yes | R1-S2 | 97.23 | 97.26 | 97.14 | 97.11 | -0.12 | -0.34 | -0.50 | -0.47 | 4.36 | 4.42 | 0.260 | 0.475 |
| S2k359p30 | Original |  |  | 100.00 | 100.00 | 100.00 | 100.00 | +0.00 | +0.00 | -0.33 | +0.72 | 4.75 | 4.85 | 1.000 | 0.000 |
| S2k359p30 | Retrain |  |  | 90.86 | 90.92 | 90.94 | 90.63 | -0.23 | -0.00 | -0.33 | +0.15 | 0.00 | 0.18 | 0.102 | 14.628 |
| S2k359p30 | M3b-R Random Relabeling |  | R1-S2 | 88.67 | 88.77 | 88.60 | 87.74 | -0.92 | -0.63 | -0.67 | -0.23 | 3.25 | 3.75 | 0.205 | 0.850 |
| S2k359p30 | M3b-R Random Relabeling | yes | R1-S2 | 94.35 | 94.47 | 94.40 | 93.92 | -0.43 | -0.41 | -0.40 | -0.37 | 3.10 | 3.10 | 0.274 | 0.550 |
| S2k359p50 | Original |  |  | 100.00 | 100.00 | 100.00 | 100.00 | +0.00 | +0.00 | -0.33 | +0.70 | 5.26 | 5.36 | 1.000 | 0.000 |
| S2k359p50 | Retrain |  |  | 88.74 | 88.75 | 88.69 | 87.99 | -0.75 | +0.00 | -0.33 | +0.03 | 0.00 | 0.28 | 0.101 | 14.567 |
| S2k359p50 | M3b-R Random Relabeling |  | R1-S2 | 81.07 | 81.08 | 81.14 | 79.86 | -1.20 | -1.00 | -0.99 | -0.25 | 5.43 | 6.17 | 0.208 | 0.861 |
| S2k359p50 | M3b-R Random Relabeling | yes | R1-S2 | 88.96 | 88.99 | 89.16 | 88.07 | -0.89 | -0.78 | -0.57 | -0.49 | 2.21 | 2.54 | 0.276 | 0.569 |

### 9.6 Observations
- MIA: 55.47% in S1 - the highest of all methods (Retrain 51.0): random wrong labels make the forget samples recognisably high-loss
  (Streisand effect); in S2 it is the closest to Retrain (53.2% vs 50).
- Exp 1, S1 (corrected setup): the only method with a clear partition effect, and in the opposite direction: the LOW partition is the
  hardest (forget acc 2.8 / 0.6 / 0.5%; with WS 17.4 / 1.9 / 0.8%, MIA 58.3% / with WS 62.0% on low). Low-entanglement airplanes are the
  most prototypical (farthest from all non-airplane images); random wrong labels fight the strongest class evidence there. avg_gap
  2.06 / 0.53 / 1.36 (WS 6.75 / 1.35 / 1.40). S2: partially closes the low-partition gap (4.3-5.5 vs Original 5.9-6.4) and adds damage on the high partition
  (S2-10 7.78, S2-30 3.33 vs 0.7).
- Exp 2: **sequential low -> high is better than one-shot in every S2 ratio** (S2-10 2.98 vs 3.91, S2-30 2.67 vs 3.25, S2-50 2.55 vs 5.43) -
  consistent with the RUM paper. S1 sequential fails (28.7): steps 1-2 cannot forget while airplanes remain in the retain set, and in step 3
  the forget pass (6 batches of 256 per epoch) is too weak against the unchanged 158-batch retain pass.
- Linear probe (R1): output 0.03%, probe 94.3%, NCC 82.1%, purity 89.2%, silhouette 0.175: the airplane cluster is fully intact - the
  forgetting is in the classifier head (Illusion-paper finding).
- Perturbation (main grid): S1 0%; S2-30 71.0% vs Retrain 78.4% - the closest of all methods.
- Quantization (R1 / R1-S2): the most quantization-stable method. S1 unchanged (0%). S2: INT4 forget -0.1 to -1.4 points, retain -0.3
  to -1.0, test -0.4 to -1.0; it moves the weights the most (0.48-0.86 INT4 steps) and INT4 flips only 2.3% of the forget images although
  18.6% are low-margin (S2-30 seed 0). avg_gap -0.11 to +0.74.
