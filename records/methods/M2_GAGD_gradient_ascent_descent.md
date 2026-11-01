# M2 - Gradient Ascent + Descent

Last updated: 2026-10-28. Chronology: `records/JOURNAL.md`. All runs: `records/results/all_runs.csv` (method GAGD), pilot tables
`records/results/pilot_all.csv`, Stage-5 table `records/results/improve_table.md`.

## 1. Method and implementation
**Code:** `src/mu/unlearn/methods.py::ga_gd` (reference: `sample_repos/RUM-main/RUM-main/unlearn/neggrad.py`). Config key: `GAGD`.
Iterate the retain loader; the forget loader is cycled alongside; per step
`loss = beta * CE(retain batch) - (1 - beta) * CE(forget batch)`; gradient clipping 1.0. One epoch = one pass over the retain set
(~158 steps); forget batches repeat |R|/|F| times per epoch. Option (ours): `joint_forward` = one forward pass on [retain; forget]
instead of two separate passes (BN then never sees a forget-only batch).

## 2. Hyperparameter sources
Teacher-Student paper eq. (4) ("NegGrad+"): large-scale beta 0.9999, 5 epochs, SGD lr 0.01, wd 5e-4 (small-scale beta 0.95, 10 ep,
wd 0.1; "small beta results in explosion"). RUM: beta [0.85, 0.99], 5 ep, lr [0.01, 0.05]. Illusion paper: 3 ep, lr 1e-4, clip 1.0.
The repo default alpha (= beta) 0.2 is unusable. Ours: 5 epochs, batch 256 / 256, SGD momentum 0.9, wd 5e-4, augmentation.

## 3. Configurations
| name | lr | beta | other | where |
|---|---|---|---|---|
| pilot grid | 0.01, 0.03 | 0.95, 0.99, 0.999, 0.9999 | | pilot (S1, S2-30) |
| **D1** main grid S1 | 0.01 | 0.99 | | main grid |
| **D1-S2** main grid S2 | 0.01 | 0.999 | | main grid |
| D2 | 0.01, 0.02 | 0.995, 0.998 | | imp (S2-30) |
| D1-joint | as D1 | | `joint_forward` | bnfix |
| D1 + recal | as D1 | | BN re-estimated on retain after unlearning (robustness check) | bnfix |

## 4. Results (avg_gap / sel_gap = mean |metric - Retrain|; best in bold)
S1 (Retrain: forget 0, retain 100, test other-cls 94.82, MIA 51.0):
| config | seeds | forget | retain | test other-cls | MIA | sel_gap | avg_gap |
|---|---|---|---|---|---|---|---|
| pilot beta 0.95, lr 0.01 | 1 | 0.00 | 95.04 | 87.82 | 48.90 | 2.82 | 3.52 |
| pilot beta 0.95, lr 0.03 | 1 | 0.00 | 24.96 | 24.21 | 50.00 | 35.12 | 35.07 |
| pilot beta 0.999, lr 0.01 | 1 | 31.02 | 99.91 | - | 57.20 | 9.27 | 9.60 |
| pilot beta 0.9999 (paper large-scale), lr 0.01 | 1 | 62.02 | 99.96 | - | 59.25 | 18.17 | 18.41 |
| D1 (main grid) | 3 | 0.00 | 99.09 | 91.87 | 49.25 | - | 1.34 |
| D1 + WS (main grid) | 3 | 0.00 | 99.43 | 92.54 | 50.67 | - | 0.81 |
| **D1-joint** | 1 | **0.00** | **99.88** | **94.03** | 54.95 | 1.18 | 1.02 |
| D1 + recal | 1 | 0.00 | 98.94 | 92.28 | 53.15 | 1.75 | 1.20 |

S2-30 (Retrain: forget 90.86, retain 100, test 94.46, MIA 49.95; Original sel_gap 4.51):
| config | seeds | forget | retain | test | MIA | sel_gap | avg_gap |
|---|---|---|---|---|---|---|---|
| pilot beta 0.99, lr 0.01 | 1 | 12.42 | 80.45 | 70.64 | 54.60 | 31.22 | 31.74 |
| **D1-S2 (main grid)** | 3 | 92.92 | 98.74 | 90.36 | 59.12 | - | **4.15** |
| D2 beta 0.998, lr 0.01 | 1 | 85.73 | 97.32 | 88.73 | 57.97 | 5.23 | 5.51 |
| D2 beta 0.998, lr 0.02 | 1 | 84.67 | 96.64 | 87.93 | 56.73 | 5.69 | 5.84 |
| D2 beta 0.995, lr 0.01 | 1 | 44.49 | 88.28 | 78.73 | 55.75 | 19.62 | 20.03 |
| D2 beta 0.995, lr 0.02 | 1 | 25.28 | 83.43 | 73.68 | 53.47 | 26.67 | 26.73 |
| D1-S2 joint forward | 1 | 100.00 | 100.00 | 94.61 | 60.23 | 4.88 | 4.67 |
| D1-S2 + recal | 1 | 98.74 | 98.87 | 92.44 | 57.65 | 4.13 | 4.53 |
Other S2 ratios (main grid, avg_gap): S2-10 3.52 (with WS 3.59), S2-50 3.80 (with WS 4.04).

Paper references: Teacher-Student paper Table 9 (ResNet CIFAR-10 class, pre-trained on CIFAR-100, no augmentation): NegGrad+
forget acc 12.78, retain 99.42, test 82.13. Illusion paper: NegGrad+ output forget 0, probe forget 67.09. Ours D1: forget 0.00,
retain 99.09; probe forget 82.0 (`records/results/repr_table.md`).

## 5. Issues and observations
- **beta is very sensitive** (paper's warning confirmed): beta 0.95 at lr 0.03 collapses (retain 25%), the paper's large-scale
  0.9999 under-forgets in S1 (62%). Selected 0.99 (S1) and 0.999 (S2).
- **BatchNorm:** the repo implementation does two separate forward passes, so BN is also updated on forget-only batches
  (BN running mean changes 23.9% in S1, similar to GA). Unlike GA, the forgetting is robust: re-estimating BN on retain keeps
  forget acc at 0.00 (S1) because the retain loss changes the weights substantially (15.9% relative weight change).
  A single joint forward pass (D1-joint) keeps BN representative and improves S1 (test other-cls 91.9 -> 94.0, avg_gap
  1.34 -> 1.02, seed 0). In S2 the joint pass removes the forgetting entirely (forget 100%): in S2 part of the main-grid
  forgetting came from BN drift towards classes 3/5/9.
- **S2:** forget acc can be matched to Retrain (92.9 vs 90.9) but the MIA stays ~59% (Retrain 50%) and same-class test acc drops;
  lower beta forgets more but destroys same-class retain/test samples (beta 0.995: forget 44%, test 79%).

## 6. Comparison (S1, 3 seeds, avg_gap): GA 4.58 (improved 1.07) | **GA+Descent 1.34 (+WS 0.81)** | Teacher-Student 0.60 |
Random Relabeling 1.19 | Boundary 6.92 (improved C1-mix 0.16). Details: `records/results/main_table.md`.

## 9. Evaluation results (updated 2026-10-30; full details `records/results/evaluation.md`)
Best configuration used: S1 D1 (lr 0.01, beta 0.99); S2 D1-S2 (lr 0.01, beta 0.999); main-grid models (D1 / D1-S2) for perturbation.
3 seeds, mean. avg_gap = mean |metric - Retrain| (forget, retain, test acc, MIA).

### 9.1 Entanglement Exp 1 - each partition unlearned separately
S1: corrected setup (other two partitions excluded from forget and retain; reference = S1 Retrain on the partition). S2: RUM protocol
(other partitions in the retain set; Retrain per partition). Per-partition accuracies and MIA:
| scenario | model | WS | forget acc: low | forget acc: medium | forget acc: high | S1 only: acc on the 2 excluded partitions, low run | S1 only: excluded acc, medium run | S1 only: excluded acc, high run | retain acc (mean of 3 runs) | test acc (mean of 3 runs) | MIA: low | MIA: medium | MIA: high |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| S1c0 | Retrain (reference) |  | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 100.0 | 85.4 | 53.1 | 49.2 | 49.7 |
| S1c0 | Original (no unlearning) |  | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 94.8 | 58.0 | 57.9 | 57.3 |
| S1c0 | M2 GA + Descent |  | 0.5 | 1.7 | 0.5 | 0.1 | 0.4 | 0.8 | 99.4 | 83.1 | 55.3 | 55.0 | 59.9 |
| S1c0 | M2 GA + Descent | yes | 0.7 | 1.5 | 0.5 | 0.2 | 0.3 | 0.9 | 99.6 | 83.5 | 53.4 | 53.7 | 56.9 |
| S2k359p10 | Retrain (reference) |  | 77.6 | 94.6 | 99.6 | - | - | - | 100.0 | 93.5 | 57.3 | 51.3 | 56.7 |
| S2k359p10 | Original (no unlearning) |  | 100.0 | 100.0 | 100.0 | - | - | - | 100.0 | 94.8 | 57.7 | 57.7 | 57.7 |
| S2k359p10 | M2 GA + Descent |  | 72.6 | 88.4 | 95.5 | - | - | - | 98.8 | 90.9 | 50.9 | 56.0 | 60.3 |
| S2k359p10 | M2 GA + Descent | yes | 76.8 | 90.5 | 95.8 | - | - | - | 98.8 | 90.9 | 48.3 | 56.9 | 60.0 |
| S2k359p30 | Retrain (reference) |  | 79.0 | 94.7 | 99.6 | - | - | - | 100.0 | 94.8 | 56.7 | 52.1 | 56.7 |
| S2k359p30 | Original (no unlearning) |  | 100.0 | 100.0 | 100.0 | - | - | - | 100.0 | 94.8 | 58.9 | 59.0 | 59.0 |
| S2k359p30 | M2 GA + Descent |  | 82.7 | 94.9 | 94.8 | - | - | - | 98.5 | 90.6 | 51.9 | 59.8 | 60.2 |
| S2k359p30 | M2 GA + Descent | yes | 85.3 | 96.1 | 95.2 | - | - | - | 98.6 | 90.7 | 52.6 | 60.7 | 60.0 |
| S2k359p50 | Retrain (reference) |  | 77.8 | 94.3 | 99.4 | - | - | - | 100.0 | 94.5 | 56.3 | 51.8 | 57.2 |
| S2k359p50 | Original (no unlearning) |  | 100.0 | 100.0 | 100.0 | - | - | - | 100.0 | 94.8 | 59.1 | 59.3 | 59.4 |
| S2k359p50 | M2 GA + Descent |  | 82.9 | 95.3 | 94.8 | - | - | - | 98.4 | 90.3 | 51.8 | 60.8 | 60.7 |
| S2k359p50 | M2 GA + Descent | yes | 85.3 | 96.3 | 94.7 | - | - | - | 98.5 | 90.5 | 52.5 | 61.3 | 60.5 |

avg_gap to the partition's Retrain:
| scenario | model | WS | low | medium | high | high - low |
|---|---|---|---|---|---|---|
| S1c0 | Original (no unlearning) | | 28.59 | 29.52 | 29.27 | +0.67 |
| S1c0 | (Retrain forget acc on partition, %) | | 0.0 | 0.0 | 0.0 | |
| S1c0 | M2 GA + Descent |  | 1.39 | 2.56 | 3.46 | +2.07 |
| S1c0 | M2 GA + Descent | yes | 1.17 | 2.07 | 2.44 | +1.27 |
| S2k359p10 | Original (no unlearning) | | 6.24 | 3.32 | 0.67 | -5.58 |
| S2k359p10 | (Retrain forget acc on partition, %) | | 77.6 | 94.6 | 99.6 | |
| S2k359p10 | M2 GA + Descent |  | 3.86 | 3.73 | 3.14 | -0.72 |
| S2k359p10 | M2 GA + Descent | yes | 4.27 | 3.47 | 2.94 | -1.34 |
| S2k359p30 | Original (no unlearning) | | 5.85 | 3.06 | 0.75 | -5.10 |
| S2k359p30 | (Retrain forget acc on partition, %) | | 79.0 | 94.7 | 99.6 | |
| S2k359p30 | M2 GA + Descent |  | 3.12 | 3.58 | 3.88 | +0.76 |
| S2k359p30 | M2 GA + Descent | yes | 3.55 | 3.91 | 3.64 | +0.09 |
| S2k359p50 | Original (no unlearning) | | 6.43 | 3.34 | 0.76 | -5.67 |
| S2k359p50 | (Retrain forget acc on partition, %) | | 77.8 | 94.3 | 99.4 | |
| S2k359p50 | M2 GA + Descent |  | 3.29 | 4.12 | 3.85 | +0.55 |
| S2k359p50 | M2 GA + Descent | yes | 3.68 | 4.45 | 3.72 | +0.04 |

### 9.2 Entanglement Exp 2 - sequential low -> medium -> high vs one-shot (final model, whole forget set)
| scenario | WS | mode | forget | retain | test | MIA | avg_gap |
|---|---|---|---|---|---|---|---|
| S1c0 |  | one-shot | 0.00 | 99.09 | 82.68 | 49.25 | 1.34 |
| S1c0 |  | sequential | 0.00 | 95.41 | 79.03 | 50.58 | 3.04 |
| S1c0 | yes | one-shot | 0.00 | 99.43 | 83.29 | 50.67 | 0.81 |
| S1c0 | yes | sequential | 0.00 | 94.10 | 77.67 | 50.95 | 3.56 |
| S2k359p10 |  | one-shot | 91.41 | 98.72 | 90.62 | 57.99 | 3.52 |
| S2k359p10 |  | sequential | 86.69 | 97.76 | 89.43 | 56.81 | 4.89 |
| S2k359p10 | yes | one-shot | 92.32 | 98.80 | 90.71 | 58.20 | 3.59 |
| S2k359p10 | yes | sequential | 88.57 | 98.20 | 89.95 | 56.94 | 4.21 |
| S2k359p30 |  | one-shot | 92.92 | 98.74 | 90.36 | 59.12 | 4.15 |
| S2k359p30 |  | sequential | 81.74 | 96.59 | 87.45 | 57.72 | 6.83 |
| S2k359p30 | yes | one-shot | 93.53 | 98.78 | 90.32 | 59.44 | 4.38 |
| S2k359p30 | yes | sequential | 87.51 | 97.72 | 88.83 | 58.26 | 4.89 |
| S2k359p50 |  | one-shot | 89.26 | 98.44 | 89.26 | 59.07 | 3.80 |
| S2k359p50 |  | sequential | 71.57 | 95.58 | 84.94 | 57.42 | 9.27 |
| S2k359p50 | yes | one-shot | 90.67 | 98.65 | 89.57 | 59.43 | 4.04 |
| S2k359p50 | yes | sequential | 80.81 | 97.09 | 86.98 | 58.20 | 6.26 |

### 9.3 Linear probe / embedding separability (S1; figure `records/figures/fig_embed_best_S1c0.png`)
| model | output forget-cls | probe forget-cls | NCC forget-cls | probe other-cls | silhouette class 0 | 10-NN purity class 0 |
|---|---|---|---|---|---|---|
| Original | 95.40 ± 0.00 | 95.50 ± 0.17 | 95.40 ± 0.00 | 94.69 ± 0.05 | 0.127 ± 0.000 | 93.00 ± 0.00 |
| Retrain | 0.00 ± 0.00 | 87.73 ± 0.87 | 41.43 ± 2.97 | 91.19 ± 0.08 | -0.040 ± 0.005 | 52.58 ± 1.06 |
| GA+Descent (D1) | 0.00 ± 0.00 | 81.80 ± 0.75 | 32.47 ± 1.26 | 92.10 ± 0.15 | -0.070 ± 0.004 | 32.38 ± 1.69 |
| GA+Descent (D1) + WS | 0.00 ± 0.00 | 82.87 ± 0.21 | 30.40 ± 0.72 | 92.64 ± 0.13 | -0.054 ± 0.003 | 32.73 ± 1.90 |

### 9.4 Perturbed forget samples (main-grid models; Gaussian tau, 20 perturbations per image)
| scenario | model | WS | clean forget acc | perturbed forget acc tau 0.03 | recovery tau 0.03 | perturbed tau 0.1 | recovery tau 0.1 |
|---|---|---|---|---|---|---|---|
| S1c0 | Original |  | 100.0 | 80.6 | 0.0 | 8.7 | 0.0 |
| S1c0 | Retrain |  | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| S1c0 | M2 GA + Descent |  | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| S1c0 | M2 GA + Descent | yes | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| S2k359p30 | Original |  | 100.0 | 91.6 | 0.0 | 8.2 | 0.0 |
| S2k359p30 | Retrain |  | 90.9 | 78.4 | 44.5 | 7.1 | 8.6 |
| S2k359p30 | M2 GA + Descent |  | 92.9 | 56.1 | 31.3 | 0.6 | 0.5 |
| S2k359p30 | M2 GA + Descent | yes | 93.5 | 56.4 | 29.7 | 0.6 | 0.5 |

### 9.5 Post-unlearning quantization (best configurations; weight-only per-channel, BN in fp32; details `records/results/quant_best.md`)
| scenario | model | WS | config | forget fp32 | forget INT8 | forget INT6 | forget INT4 | change forget (INT4 - fp32) | change retain | change test | change MIA | avg_gap fp32 | avg_gap INT4 | share Q(theta_u)=Q(theta_o), INT4 | mean abs(delta)/step, INT4 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| S1c0 | Original |  |  | 100.00 | 100.00 | 100.00 | 100.00 | +0.00 | +0.00 | -0.33 | -0.60 | 29.05 | 28.82 | 1.000 | 0.000 |
| S1c0 | Retrain |  |  | 0.00 | 0.00 | 0.00 | 0.00 | +0.00 | -0.01 | -0.34 | -0.07 | 0.00 | 0.10 | 0.102 | 15.126 |
| S1c0 | M2 GA + Descent |  | D1 | 0.00 | 0.00 | 0.00 | 0.00 | +0.00 | -0.33 | -0.43 | -0.37 | 1.34 | 1.62 | 0.288 | 0.124 |
| S1c0 | M2 GA + Descent | yes | D1 | 0.00 | 0.00 | 0.00 | 0.00 | +0.00 | -0.31 | -0.49 | +0.37 | 0.75 | 0.87 | 0.459 | 0.087 |
| S2k359p10 | Original |  |  | 100.00 | 100.00 | 100.00 | 100.00 | +0.00 | +0.00 | -0.33 | +0.72 | 4.47 | 4.63 | 1.000 | 0.000 |
| S2k359p10 | Retrain |  |  | 91.63 | 91.63 | 91.46 | 90.59 | -1.04 | +0.00 | -0.28 | -0.16 | 0.00 | 0.37 | 0.104 | 17.137 |
| S2k359p10 | M2 GA + Descent |  | D1-S2 | 91.41 | 91.28 | 91.26 | 87.09 | -4.32 | -0.84 | -0.87 | -1.37 | 3.46 | 4.63 | 0.304 | 0.062 |
| S2k359p10 | M2 GA + Descent | yes | D1-S2 | 92.32 | 92.44 | 92.27 | 88.49 | -3.83 | -0.75 | -0.85 | -0.88 | 3.59 | 4.38 | 0.477 | 0.041 |
| S2k359p30 | Original |  |  | 100.00 | 100.00 | 100.00 | 100.00 | +0.00 | +0.00 | -0.33 | +0.72 | 4.75 | 4.85 | 1.000 | 0.000 |
| S2k359p30 | Retrain |  |  | 90.86 | 90.92 | 90.94 | 90.63 | -0.23 | -0.00 | -0.33 | +0.15 | 0.00 | 0.18 | 0.102 | 14.628 |
| S2k359p30 | M2 GA + Descent |  | D1-S2 | 92.92 | 92.91 | 92.94 | 88.35 | -4.57 | -0.97 | -1.19 | -0.75 | 4.15 | 4.61 | 0.304 | 0.060 |
| S2k359p30 | M2 GA + Descent | yes | D1-S2 | 93.53 | 93.65 | 93.51 | 89.20 | -4.33 | -0.87 | -1.01 | -0.76 | 4.38 | 4.41 | 0.475 | 0.038 |
| S2k359p50 | Original |  |  | 100.00 | 100.00 | 100.00 | 100.00 | +0.00 | +0.00 | -0.33 | +0.70 | 5.26 | 5.36 | 1.000 | 0.000 |
| S2k359p50 | Retrain |  |  | 88.74 | 88.75 | 88.69 | 87.99 | -0.75 | +0.00 | -0.33 | +0.03 | 0.00 | 0.28 | 0.101 | 14.567 |
| S2k359p50 | M2 GA + Descent |  | D1-S2 | 89.26 | 89.20 | 88.92 | 84.45 | -4.80 | -0.85 | -1.13 | -0.76 | 3.73 | 4.97 | 0.304 | 0.058 |
| S2k359p50 | M2 GA + Descent | yes | D1-S2 | 90.67 | 90.58 | 90.47 | 87.03 | -3.64 | -0.64 | -1.03 | -0.51 | 4.04 | 4.27 | 0.473 | 0.037 |

### 9.6 Observations
- MIA: 49.25% (D1) / 50.67% (D1 + WS) in S1 - closest to Retrain (51.0) together with Boundary C1-mix; S2-30 59.1%.
- Exp 1, S1 (corrected setup): forget acc 0.5 / 1.7 / 0.5%, unused airplanes 0.1-0.8%; the MIA rises with entanglement (55.3 / 55.0 /
  59.9%), so avg_gap rises 1.39 / 2.56 / 3.46 (WS 1.17 / 2.07 / 2.44): ascent on boundary airplanes leaves them recognisably high-loss.
  S2: flat 3.1-4.1; on the high partition it adds damage (S2-30 3.88 vs Original 0.75).
- Exp 2: sequential is worse than one-shot everywhere (S1 3.04 vs 1.34; S2-50 9.27 vs 3.80); WS reduces the sequential damage
  (S2-30 6.83 -> 4.89, S2-50 9.27 -> 6.26).
- Linear probe (D1): output 0%, probe 81.8% (lowest of all methods), NCC 32.5%, purity 32.4%: the airplane cluster is dispersed.
- Perturbation (main grid): S2-30 56.1% vs Retrain 78.4% (over-unlearning); S1 0%.
- Quantization (D1 / D1-S2): S1 unchanged (0% forget at every bit width, test -0.43). S2: INT4 forget -3.6 to -4.8 points (S2-30
  92.92 -> 88.35%) with retain -0.64 to -0.97 and test -0.85 to -1.19; 12.6% of the correct forget images are low-margin and INT4 flips
  4.9% of the forget images (S2-30 seed 0). avg_gap worsens (S2-10 3.46 -> 4.63, S2-50 3.73 -> 4.97); no restoration.
