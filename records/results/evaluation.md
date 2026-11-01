# Evaluation (Objective 2 + robustness) - setup, results and observations

Last updated: 2026-10-29. All numbers are means over 3 seeds unless stated. Tables are generated from the run records
(`scripts/build_eval_tables.py`, `scripts/analyze_entangle_exps.py`, `scripts/probe_best_models.py`, `scripts/analyze_robustness.py`);
per-method summaries are in `records/methods/*.md`. Configuration names (G3, D1, T1, R1, C1-mix, ...) are defined on the method pages.

## 0. Common setup
- CIFAR-10, ResNet-18 (CIFAR stem), 45k train pool / 5k validation / 10k test. Original: 182 epochs, SGD lr 0.1, cosine, crop + flip.
- Scenarios: **S1** = forget all of class 0 (airplane, 4500 images); **S2-10/30/50** = forget 10/30/50% of each of classes 3, 5, 9
  (1350 / 4050 / 6750 images). Every scenario and seed has a Retrain-from-scratch reference (same recipe, retain data only).
- Metrics: forget / retain / test accuracy (test other-cls = test accuracy on the classes that were not forgotten), MIA accuracy,
  **avg_gap** = mean |metric - Retrain| over forget acc, retain acc, test acc and MIA acc (lower = closer to Retrain);
  **sel_gap** = same with validation acc and validation-negative MIA (used only for hyperparameter selection, never test data).
- Models evaluated: the main grid (paper-range settings) and the **best configuration of every method**
  (`records/results/method_comparison.md`): S1 GA = G3, GA+Descent = D1, Teacher-Student = T1, Random Relabeling = R1,
  Boundary Relabeling = C1-mix; each also with the Gradient based Weight Saliency mask (WS, ratio 0.5).

## 1. E1 - Accuracy (summary; full tables `records/results/main_table.md`, `method_comparison.md`)
Best configurations, S1 (Retrain: forget 0.00, retain 100, test other-cls 94.82, MIA 51.0):
| method (config) | forget | retain | test other-cls | MIA | avg_gap |
|---|---|---|---|---|---|
| M1 Gradient Ascent (G3) | 0.45 | 99.17 | 92.34 | 51.78 | 1.07 |
| M2 GA + Descent (D1 + WS) | 0.00 | 99.43 | 92.54 | 50.67 | 0.81 |
| M3a Teacher-Student (T1) | 0.33 | 99.99 | 94.30 | 52.58 | 0.60 |
| M3b-R Random Relabeling (R1 + WS) | 0.00 | 100.00 | 94.64 | 55.48 | 1.17 |
| **M3b-B Boundary Relabeling (C1-mix)** | **0.00** | **100.00** | **94.56** | **51.03** | **0.16** |
S2-30 (Retrain forget 90.86, MIA 49.95; Original avg_gap 4.75): best Random Relabeling + WS 3.10, Teacher-Student T4-best 3.34,
GA+Descent 4.15, GA G3 4.40, Boundary (C0) 9.92. In S2 no method reaches a low forget accuracy without dropping same-class
retain/test accuracy, because the Retrain model itself keeps ~89-92% forget accuracy.

## 2. E2 - Membership Inference Attack
**Implementation** (`src/mu/eval/mia.py::loss_mia`, called in `scripts/unlearn.py` / `scripts/train.py`):
1. Per-sample cross-entropy loss of the evaluated model, eval mode, no augmentation, for every forget sample and every test sample.
2. Positives ("member") = forget samples; negatives ("non-member") = test samples **of the same classes** as the forget set
   (S1: class-0 test images; S2: test images of classes 3, 5, 9) - the doc's rule, so the attack cannot exploit class differences.
3. Balanced: the larger side is randomly subsampled (seeded) to n = min(|forget|, |same-class test|) (S1: 1000 vs 1000).
4. Feature = scalar loss, clipped to [-400, 400] (doc; CE >= 0, so the clip is inactive in practice).
5. Attacker = sklearn `LogisticRegression()` (defaults), stratified 5-fold cross-validation (shuffle, seeded); reported MIA = mean
   held-out fold accuracy; AUC and ROC from out-of-fold probabilities. Ideal = Retrain's value (~50%).
6. `mia_val`: same with validation images as negatives - used only for selection.
**Sanity:** Retrain 49.6-51.0% in every scenario; Original 57.8% (S1) - 59.5% (S2). The Original is only moderately attackable
because the model generalises well (augmentation, 94.8% test acc). A 1-D logistic regression is a best-threshold attack on the loss
("Basic MIA" of the Teacher-Student paper); stronger attacks (SVC on confidence/entropy, LiRA) were not run.

MIA accuracy (%) for every scenario; main grid = paper-range settings, best config = improved configuration (last column).
| scenario | model | MIA main grid, no WS | MIA main grid, WS | MIA best config, no WS | MIA best config, WS | best config |
|---|---|---|---|---|---|---|
| S1c0 | Retrain | 51.02 | | | | |
| S1c0 | Original | 57.78 | | | | |
| S1c0 | M1 Gradient Ascent | 51.17 | 51.75 | 51.78 | 52.13 | G3 |
| S1c0 | M2 GA + Descent | 49.25 | 50.67 | 49.25 | 50.67 | D1 |
| S1c0 | M3a Teacher-Student | 52.58 | 52.18 | 52.58 | 52.18 | T1 |
| S1c0 | M3b-R Random Relabeling | 55.47 | 55.48 | 55.47 | 55.48 | R1 |
| S1c0 | M3b-B Boundary Relabeling | 54.28 | 56.50 | 51.03 | 51.90 | C1-mix |
| S2k359p10 | Retrain | 49.64 | | | | |
| S2k359p10 | Original | 58.95 | | | | |
| S2k359p10 | M1 Gradient Ascent | 59.70 | 59.78 | 60.10 | 60.40 | G3-S2 |
| S2k359p10 | M2 GA + Descent | 57.99 | 58.20 | 57.99 | 58.20 | D1-S2 |
| S2k359p10 | M3a Teacher-Student | 58.27 | 57.98 | 57.02 | 57.09 | T4-best S2 |
| S2k359p10 | M3b-R Random Relabeling | 55.62 | 57.07 | 55.62 | 57.07 | R1-S2 |
| S2k359p10 | M3b-B Boundary Relabeling | 59.25 | 59.21 | 59.25 | 59.21 | C0 |
| S2k359p30 | Retrain | 49.95 | | | | |
| S2k359p30 | Original | 59.46 | | | | |
| S2k359p30 | M1 Gradient Ascent | 59.76 | 59.75 | 59.62 | 59.64 | G3-S2 |
| S2k359p30 | M2 GA + Descent | 59.12 | 59.44 | 59.12 | 59.44 | D1-S2 |
| S2k359p30 | M3a Teacher-Student | 57.21 | 56.77 | 56.13 | 56.64 | T4-best S2 |
| S2k359p30 | M3b-R Random Relabeling | 53.16 | 55.04 | 53.16 | 55.04 | R1-S2 |
| S2k359p30 | M3b-B Boundary Relabeling | 59.52 | 59.42 | 59.52 | 59.42 | C0 |
| S2k359p50 | Retrain | 50.75 | | | | |
| S2k359p50 | Original | 59.47 | | | | |
| S2k359p50 | M1 Gradient Ascent | 58.69 | 58.74 | 59.46 | 59.51 | G3-S2 |
| S2k359p50 | M2 GA + Descent | 59.07 | 59.43 | 59.07 | 59.43 | D1-S2 |
| S2k359p50 | M3a Teacher-Student | 49.71 | 49.66 | 57.04 | 57.42 | T4-best S2 |
| S2k359p50 | M3b-R Random Relabeling | 51.78 | 53.74 | 51.78 | 53.74 | R1-S2 |
| S2k359p50 | M3b-B Boundary Relabeling | 59.92 | 59.68 | 59.92 | 59.68 | C0 |
ROC curves: `records/figures/fig_mia_roc_<scenario>.png` (main grid, one panel per seed).
**Observations:** in S1 every best configuration is within ~1 point of Retrain except Random Relabeling (55.5%: random wrong labels
make forget samples high-loss in a recognisable way - the "Streisand effect"); in S2 all methods stay at 53-60% (Retrain 50%):
the forget samples remain more "member-like" than unseen same-class images; Random Relabeling is the closest (53%).

## 3. E3 - Entanglement and unlearning difficulty
A detailed plain-language explanation of both experiments, with per-scenario tables for every method, is in
`records/results/entanglement_explained.md`; every run (including the WS variants) is listed in `records/results/entanglement_experiments.md`.
**Entanglement score** (slide 14): mean Euclidean distance from each forget sample to its k = 50 nearest RETAIN samples in the
Original model's 512-d penultimate features; smaller = more entangled. The forget set is ranked and split into equal thirds:
low (largest distance), medium, high (smallest). Partitions: `scripts/make_entangle_partitions.py`; sizes: S1 1500 per partition,
S2-10/30/50 450/1350/2250. Global ES (RUM formula) of partition vs its retain set, S1: low 1.6, medium 1.9, high 2.4.
All methods use their best configuration (`records/results/entanglement_experiments.md` lists every hyperparameter); Teacher-Student's
forget batch is divided by 3 (same number of forget updates per epoch as the confirmed configuration).

### 3.1 Exp 1 - each partition unlearned separately
Each method unlearns one partition at a time, starting from the Original model.
- **S1 (single-class forgetting), corrected setup (tag `entpart2`, 2026-10-30):** forget = one partition (1500 airplanes); retain = the
  normal S1 retain set (no airplane). The other two partitions (3000 airplanes) are excluded from both sets - they are not used during
  unlearning - and their accuracy is reported separately ("excluded"). Reference = the normal S1 Retrain evaluated on the partition.
  (The first S1 run kept the other two partitions in the retain set, which turned S1 into subset forgetting; it is superseded.)
- **S2 (subset forgetting), RUM protocol (tag `entpart`):** forget = one partition; retain = all other training data (the other two
  partitions included); a Retrain reference per partition (27 extra Retrains).

Per-partition accuracies and MIA (3 seeds, mean):
| scenario | model | WS | forget acc: low | forget acc: medium | forget acc: high | S1 only: acc on the 2 excluded partitions, low run | S1 only: excluded acc, medium run | S1 only: excluded acc, high run | retain acc (mean of 3 runs) | test acc (mean of 3 runs) | MIA: low | MIA: medium | MIA: high |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| S1c0 | Retrain (reference) |  | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 100.0 | 85.4 | 53.1 | 49.2 | 49.7 |
| S1c0 | Original (no unlearning) |  | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 94.8 | 58.0 | 57.9 | 57.3 |
| S1c0 | M1 Gradient Ascent |  | 0.4 | 0.4 | 0.7 | 0.2 | 0.4 | 5.4 | 99.1 | 83.2 | 51.4 | 50.5 | 49.6 |
| S1c0 | M1 Gradient Ascent | yes | 0.2 | 0.1 | 0.4 | 0.1 | 0.2 | 3.2 | 98.8 | 82.9 | 53.0 | 49.8 | 49.3 |
| S1c0 | M2 GA + Descent |  | 0.5 | 1.7 | 0.5 | 0.1 | 0.4 | 0.8 | 99.4 | 83.1 | 55.3 | 55.0 | 59.9 |
| S1c0 | M2 GA + Descent | yes | 0.7 | 1.5 | 0.5 | 0.2 | 0.3 | 0.9 | 99.6 | 83.5 | 53.4 | 53.7 | 56.9 |
| S1c0 | M3a Teacher-Student |  | 0.0 | 0.0 | 0.1 | 0.0 | 0.0 | 0.8 | 100.0 | 84.8 | 52.5 | 54.2 | 52.4 |
| S1c0 | M3a Teacher-Student | yes | 0.4 | 1.8 | 0.4 | 0.1 | 1.0 | 1.2 | 100.0 | 85.0 | 54.4 | 54.4 | 51.4 |
| S1c0 | M3b-R Random Relabeling |  | 2.8 | 0.6 | 0.5 | 0.2 | 1.4 | 2.2 | 100.0 | 85.4 | 58.3 | 50.6 | 54.4 |
| S1c0 | M3b-R Random Relabeling | yes | 17.4 | 1.9 | 0.8 | 5.1 | 4.0 | 3.2 | 100.0 | 85.7 | 62.0 | 52.6 | 54.2 |
| S1c0 | M3b-B Boundary Relabeling |  | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 100.0 | 85.2 | 50.3 | 51.0 | 53.1 |
| S1c0 | M3b-B Boundary Relabeling | yes | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 100.0 | 85.2 | 49.0 | 49.4 | 51.7 |
| S2k359p10 | Retrain (reference) |  | 77.6 | 94.6 | 99.6 | - | - | - | 100.0 | 93.5 | 57.3 | 51.3 | 56.7 |
| S2k359p10 | Original (no unlearning) |  | 100.0 | 100.0 | 100.0 | - | - | - | 100.0 | 94.8 | 57.7 | 57.7 | 57.7 |
| S2k359p10 | M1 Gradient Ascent |  | 94.4 | 100.0 | 99.3 | - | - | - | 99.5 | 92.2 | 54.3 | 63.1 | 59.7 |
| S2k359p10 | M1 Gradient Ascent | yes | 95.6 | 100.0 | 99.1 | - | - | - | 99.5 | 92.3 | 54.6 | 63.1 | 59.7 |
| S2k359p10 | M2 GA + Descent |  | 72.6 | 88.4 | 95.5 | - | - | - | 98.8 | 90.9 | 50.9 | 56.0 | 60.3 |
| S2k359p10 | M2 GA + Descent | yes | 76.8 | 90.5 | 95.8 | - | - | - | 98.8 | 90.9 | 48.3 | 56.9 | 60.0 |
| S2k359p10 | M3a Teacher-Student |  | 83.3 | 95.1 | 98.4 | - | - | - | 99.5 | 93.2 | 51.2 | 54.7 | 57.2 |
| S2k359p10 | M3a Teacher-Student | yes | 86.6 | 95.8 | 98.4 | - | - | - | 99.5 | 93.2 | 49.6 | 54.9 | 57.5 |
| S2k359p10 | M3b-R Random Relabeling |  | 82.9 | 93.6 | 95.3 | - | - | - | 92.8 | 86.6 | 52.3 | 54.8 | 60.5 |
| S2k359p10 | M3b-R Random Relabeling | yes | 88.7 | 96.3 | 94.7 | - | - | - | 96.2 | 89.2 | 49.2 | 56.1 | 59.3 |
| S2k359p10 | M3b-B Boundary Relabeling |  | 68.0 | 88.7 | 83.4 | - | - | - | 94.7 | 87.2 | 49.4 | 62.4 | 60.4 |
| S2k359p10 | M3b-B Boundary Relabeling | yes | 67.6 | 88.4 | 83.3 | - | - | - | 94.7 | 87.2 | 49.3 | 62.3 | 60.4 |
| S2k359p30 | Retrain (reference) |  | 79.0 | 94.7 | 99.6 | - | - | - | 100.0 | 94.8 | 56.7 | 52.1 | 56.7 |
| S2k359p30 | Original (no unlearning) |  | 100.0 | 100.0 | 100.0 | - | - | - | 100.0 | 94.8 | 58.9 | 59.0 | 59.0 |
| S2k359p30 | M1 Gradient Ascent |  | 77.7 | 93.2 | 99.2 | - | - | - | 98.4 | 90.8 | 50.5 | 60.7 | 60.5 |
| S2k359p30 | M1 Gradient Ascent | yes | 76.6 | 96.5 | 99.4 | - | - | - | 98.6 | 91.0 | 50.3 | 61.6 | 60.8 |
| S2k359p30 | M2 GA + Descent |  | 82.7 | 94.9 | 94.8 | - | - | - | 98.5 | 90.6 | 51.9 | 59.8 | 60.2 |
| S2k359p30 | M2 GA + Descent | yes | 85.3 | 96.1 | 95.2 | - | - | - | 98.6 | 90.7 | 52.6 | 60.7 | 60.0 |
| S2k359p30 | M3a Teacher-Student |  | 64.1 | 93.0 | 98.4 | - | - | - | 95.9 | 89.6 | 55.7 | 56.9 | 58.0 |
| S2k359p30 | M3a Teacher-Student | yes | 79.9 | 95.3 | 99.4 | - | - | - | 98.7 | 92.0 | 51.4 | 57.9 | 58.9 |
| S2k359p30 | M3b-R Random Relabeling |  | 89.5 | 96.9 | 98.9 | - | - | - | 97.8 | 90.9 | 49.4 | 56.9 | 61.3 |
| S2k359p30 | M3b-R Random Relabeling | yes | 92.2 | 98.6 | 99.7 | - | - | - | 99.0 | 91.9 | 51.5 | 57.4 | 62.0 |
| S2k359p30 | M3b-B Boundary Relabeling |  | 52.1 | 74.6 | 68.7 | - | - | - | 92.1 | 84.6 | 47.6 | 61.3 | 60.7 |
| S2k359p30 | M3b-B Boundary Relabeling | yes | 52.0 | 74.1 | 68.7 | - | - | - | 92.2 | 84.6 | 47.5 | 61.3 | 60.3 |
| S2k359p50 | Retrain (reference) |  | 77.8 | 94.3 | 99.4 | - | - | - | 100.0 | 94.5 | 56.3 | 51.8 | 57.2 |
| S2k359p50 | Original (no unlearning) |  | 100.0 | 100.0 | 100.0 | - | - | - | 100.0 | 94.8 | 59.1 | 59.3 | 59.4 |
| S2k359p50 | M1 Gradient Ascent |  | 76.5 | 94.2 | 99.0 | - | - | - | 98.4 | 90.7 | 50.6 | 61.4 | 60.5 |
| S2k359p50 | M1 Gradient Ascent | yes | 76.6 | 94.0 | 99.1 | - | - | - | 98.4 | 90.6 | 50.6 | 61.4 | 60.7 |
| S2k359p50 | M2 GA + Descent |  | 82.9 | 95.3 | 94.8 | - | - | - | 98.4 | 90.3 | 51.8 | 60.8 | 60.7 |
| S2k359p50 | M2 GA + Descent | yes | 85.3 | 96.3 | 94.7 | - | - | - | 98.5 | 90.5 | 52.5 | 61.3 | 60.5 |
| S2k359p50 | M3a Teacher-Student |  | 88.9 | 98.6 | 99.7 | - | - | - | 99.5 | 93.0 | 51.1 | 59.2 | 59.8 |
| S2k359p50 | M3a Teacher-Student | yes | 89.8 | 98.8 | 98.9 | - | - | - | 99.5 | 93.0 | 51.4 | 59.5 | 58.3 |
| S2k359p50 | M3b-R Random Relabeling |  | 86.0 | 97.1 | 99.3 | - | - | - | 98.2 | 91.1 | 48.8 | 56.5 | 61.0 |
| S2k359p50 | M3b-R Random Relabeling | yes | 89.9 | 97.9 | 99.9 | - | - | - | 99.1 | 92.0 | 50.8 | 57.5 | 61.8 |
| S2k359p50 | M3b-B Boundary Relabeling |  | 52.8 | 74.8 | 68.5 | - | - | - | 92.7 | 84.6 | 47.9 | 61.6 | 60.7 |
| S2k359p50 | M3b-B Boundary Relabeling | yes | 52.4 | 74.2 | 68.3 | - | - | - | 92.8 | 84.7 | 47.8 | 61.4 | 60.3 |

avg_gap to the partition's Retrain (lower = better); "Original" = no unlearning:
| scenario | model | WS | low | medium | high | high - low |
|---|---|---|---|---|---|---|
| S1c0 | Original (no unlearning) | | 28.59 | 29.52 | 29.27 | +0.67 |
| S1c0 | (Retrain forget acc on partition, %) | | 0.0 | 0.0 | 0.0 | |
| S1c0 | M1 Gradient Ascent |  | 1.50 | 1.30 | 0.84 | -0.66 |
| S1c0 | M1 Gradient Ascent | yes | 1.38 | 1.40 | 0.91 | -0.47 |
| S1c0 | M2 GA + Descent |  | 1.39 | 2.56 | 3.46 | +2.07 |
| S1c0 | M2 GA + Descent | yes | 1.17 | 2.07 | 2.44 | +1.27 |
| S1c0 | M3a Teacher-Student |  | 0.56 | 1.39 | 0.84 | +0.28 |
| S1c0 | M3a Teacher-Student | yes | 0.59 | 1.82 | 0.63 | +0.04 |
| S1c0 | M3b-R Random Relabeling |  | 2.06 | 0.53 | 1.36 | -0.69 |
| S1c0 | M3b-R Random Relabeling | yes | 6.75 | 1.35 | 1.40 | -5.34 |
| S1c0 | M3b-B Boundary Relabeling |  | 0.74 | 0.48 | 0.90 | +0.16 |
| S1c0 | M3b-B Boundary Relabeling | yes | 1.06 | 0.12 | 0.62 | -0.44 |
| S2k359p10 | Original (no unlearning) | | 6.24 | 3.32 | 0.67 | -5.58 |
| S2k359p10 | (Retrain forget acc on partition, %) | | 77.6 | 94.6 | 99.6 | |
| S2k359p10 | M1 Gradient Ascent |  | 5.42 | 4.67 | 1.28 | -4.13 |
| S2k359p10 | M1 Gradient Ascent | yes | 5.63 | 4.66 | 1.35 | -4.28 |
| S2k359p10 | M2 GA + Descent |  | 3.86 | 3.73 | 3.14 | -0.72 |
| S2k359p10 | M2 GA + Descent | yes | 4.27 | 3.47 | 2.94 | -1.34 |
| S2k359p10 | M3a Teacher-Student |  | 3.25 | 1.13 | 0.70 | -2.55 |
| S2k359p10 | M3a Teacher-Student | yes | 4.38 | 1.39 | 0.70 | -3.67 |
| S2k359p10 | M3b-R Random Relabeling |  | 4.26 | 4.63 | 7.78 | +3.52 |
| S2k359p10 | M3b-R Random Relabeling | yes | 6.17 | 3.35 | 4.81 | -1.36 |
| S2k359p10 | M3b-B Boundary Relabeling |  | 6.19 | 7.45 | 8.61 | +2.42 |
| S2k359p10 | M3b-B Boundary Relabeling | yes | 6.34 | 7.54 | 8.61 | +2.27 |
| S2k359p30 | Original (no unlearning) | | 5.85 | 3.06 | 0.75 | -5.10 |
| S2k359p30 | (Retrain forget acc on partition, %) | | 79.0 | 94.7 | 99.6 | |
| S2k359p30 | M1 Gradient Ascent |  | 3.37 | 4.42 | 1.91 | -1.46 |
| S2k359p30 | M1 Gradient Ascent | yes | 3.73 | 4.39 | 1.90 | -1.83 |
| S2k359p30 | M2 GA + Descent |  | 3.12 | 3.58 | 3.88 | +0.76 |
| S2k359p30 | M2 GA + Descent | yes | 3.55 | 3.91 | 3.64 | +0.09 |
| S2k359p30 | M3a Teacher-Student |  | 7.62 | 3.73 | 1.96 | -5.66 |
| S2k359p30 | M3a Teacher-Student | yes | 3.79 | 2.93 | 1.12 | -2.66 |
| S2k359p30 | M3b-R Random Relabeling |  | 5.49 | 3.26 | 3.33 | -2.16 |
| S2k359p30 | M3b-R Random Relabeling | yes | 5.39 | 3.03 | 2.79 | -2.60 |
| S2k359p30 | M3b-B Boundary Relabeling |  | 11.78 | 12.17 | 14.65 | +2.87 |
| S2k359p30 | M3b-B Boundary Relabeling | yes | 11.85 | 12.33 | 14.44 | +2.60 |
| S2k359p50 | Original (no unlearning) | | 6.43 | 3.34 | 0.76 | -5.67 |
| S2k359p50 | (Retrain forget acc on partition, %) | | 77.8 | 94.3 | 99.4 | |
| S2k359p50 | M1 Gradient Ascent |  | 3.07 | 4.44 | 1.76 | -1.31 |
| S2k359p50 | M1 Gradient Ascent | yes | 3.06 | 4.56 | 1.78 | -1.28 |
| S2k359p50 | M2 GA + Descent |  | 3.29 | 4.12 | 3.85 | +0.55 |
| S2k359p50 | M2 GA + Descent | yes | 3.68 | 4.45 | 3.72 | +0.04 |
| S2k359p50 | M3a Teacher-Student |  | 4.64 | 3.52 | 1.07 | -3.58 |
| S2k359p50 | M3a Teacher-Student | yes | 4.72 | 3.65 | 0.83 | -3.89 |
| S2k359p50 | M3b-R Random Relabeling |  | 4.95 | 3.29 | 2.49 | -2.46 |
| S2k359p50 | M3b-R Random Relabeling | yes | 5.09 | 3.23 | 2.24 | -2.85 |
| S2k359p50 | M3b-B Boundary Relabeling |  | 10.83 | 12.02 | 14.27 | +3.44 |
| S2k359p50 | M3b-B Boundary Relabeling | yes | 10.97 | 12.16 | 14.10 | +3.13 |

**Observations - S1 (corrected setup):**
- Every method forgets every partition almost completely: forget acc <= 2.8% on all partitions (Retrain 0%), with one exception
  (Random Relabeling + WS on the low partition, 17.4%). Retain acc stays 98.8-100% and test acc 82.9-85.7% (Retrain 85.4%).
- The forgetting spreads to the airplanes that were not used: accuracy on the 2 excluded partitions is 0-5.4% after unlearning any one
  partition. Removing one third of the class removes the whole class.
- **No increase of forget accuracy from low to high entanglement** for any method (e.g. Gradient Ascent 0.4 / 0.4 / 0.7%, Teacher-Student
  0.0 / 0.0 / 0.1%, Boundary Relabeling 0 / 0 / 0%). In single-class forgetting the hypothesis "highly entangled samples are harder to
  forget" is not supported at the accuracy level. The only clear partition effect goes the other way: Random Relabeling finds the LOW
  partition hardest (2.8%, with WS 17.4%, MIA 58-62%). Low-entanglement airplanes are the ones farthest from all non-airplane images
  (most prototypical airplanes); random wrong labels fight the strongest class evidence there.
- Entanglement shows up in two weaker ways: (1) unlearning the high partition leaves slightly more of the other airplanes recognised
  (excluded acc: Gradient Ascent 5.4% after the high run vs 0.2% after the low run; Random Relabeling 2.2% vs 0.2%) - forgetting
  boundary airplanes generalises less to the rest of the class; (2) GA + Descent's MIA rises from 55.3% (low) to 59.9% (high), so its
  avg_gap rises 1.39 -> 2.56 -> 3.46: ascent on boundary airplanes leaves them recognisably high-loss.

**Observations - S2:**
- **Forget accuracy rises with entanglement for every method**: highly entangled forget samples are harder to forget. S2-30: Gradient
  Ascent 77.7 / 93.2 / 99.2%, GA + Descent 82.7 / 94.9 / 94.8%, Teacher-Student 64.1 / 93.0 / 98.4%, Random Relabeling 89.5 / 96.9 / 98.9%,
  Boundary Relabeling 52.1 / 74.6 / 68.7%. (Retrain shows the same trend, 79.0 / 94.7 / 99.6%: high-entangled samples are predicted from
  the remaining same-class data.)
- Lowering forget accuracy on the medium / high partitions costs retain and test accuracy: Boundary Relabeling reaches 68.7% on the high
  partition but retain acc drops to 89.2% and test acc to 82.2% (S2-30); Random Relabeling S2-10 high: retain 88.1%. Methods that keep
  retain acc ~99% (Gradient Ascent, Teacher-Student) leave the high partition at 98-99% forget acc.
- MIA of the methods is 47.5-63.1% across all S2 partitions and variants (Retrain 51.3-57.3% per partition).
- WS changes the per-partition gaps by < 0.5 in most cells (exceptions: Teacher-Student S2-30 low 7.62 -> 3.79, Random Relabeling S2-10).
- Figure: `records/figures/fig_entangle_exp1.png`.

### 3.2 Exp 2 - sequential unlearning low -> medium -> high vs one-shot
Step 1 forget = low (medium + high in the retain set), step 2 forget = medium (high in the retain set), step 3 forget = high; each step
starts from the previous step's model (teacher / neighbour model / saliency mask taken at the start of the step); the final model is
compared with the base-scenario Retrain on the whole forget set. One-shot = the same configuration on the whole forget set at once.
avg_gap (lower is better; the better of the pair in bold):
| model | WS | S1c0 one-shot | S1c0 sequential | S2k359p10 one-shot | S2k359p10 sequential | S2k359p30 one-shot | S2k359p30 sequential | S2k359p50 one-shot | S2k359p50 sequential |
|---|---|---|---|---|---|---|---|---|---|
| M1 Gradient Ascent |  | **1.07** | 1.34 | **4.80** | 5.07 | **4.40** | 7.37 | **4.18** | 6.53 |
| M1 Gradient Ascent | yes | **1.18** | 1.53 | 5.01 | **4.70** | **4.54** | 6.98 | **4.18** | 6.44 |
| M2 GA + Descent |  | **1.34** | 3.04 | **3.52** | 4.89 | **4.15** | 6.83 | **3.80** | 9.27 |
| M2 GA + Descent | yes | **0.81** | 3.56 | **3.59** | 4.21 | **4.38** | 4.89 | **4.04** | 6.26 |
| M3a Teacher-Student |  | **0.60** | 30.43 | 3.81 | **2.78** | **3.34** | 63.87 | **3.62** | 61.64 |
| M3a Teacher-Student | yes | 1.02 | **0.98** | 3.87 | **2.85** | **3.47** | 62.38 | **3.43** | 60.48 |
| M3b-R Random Relabeling |  | **1.19** | 28.65 | 3.91 | **2.98** | 3.25 | **2.67** | 5.43 | **2.55** |
| M3b-R Random Relabeling | yes | **1.17** | 28.11 | 4.36 | **2.81** | 3.10 | **2.69** | 2.58 | **2.55** |
| M3b-B Boundary Relabeling |  | **0.16** | 27.91 | **11.22** | 13.51 | **9.92** | 14.95 | **8.98** | 13.33 |
| M3b-B Boundary Relabeling | yes | **0.28** | 28.19 | **11.31** | 13.48 | **10.12** | 14.80 | **8.76** | 13.05 |

**Observations:**
- **Relabeling and Teacher-Student benefit from the low -> high order in S2** (the RUM paper's recommendation): Random Relabeling is
  better sequentially in every S2 ratio (e.g. S2-50 5.43 -> 2.55), Teacher-Student in S2-10 (3.81 -> 2.78).
- **GA, GA+Descent and Boundary Relabeling are worse sequentially almost everywhere** (only exception: GA + WS on S2-10, 5.01 -> 4.70; e.g. GA S2-30 4.40 -> 7.37): each step pushes the model
  further and earlier partitions keep drifting (repeated passes accumulate collateral damage).
- **Teacher-Student collapses sequentially in S2-30/S2-50 and in S1 without WS** (retain 16-34%): each step adds 2 max-epochs relative to a
  new teacher, so the total push (6 max-epochs) exceeds the stable range found earlier (msteps 5 collapsed); with WS S1 stays stable (0.98).
- **S1, Random and Boundary Relabeling fail sequentially** (final forget 96-98%): in steps 1-2 the other airplanes are in the retain set,
  so the retain pass re-teaches class 0 (forget acc stays 100% on the step's partition); in step 3 the forget pass has only 1500 samples
  (6 batches per epoch instead of 18 one-shot) against an unchanged 158-batch retain pass, which is too weak (only the Teacher-Student
  forget batch was rescaled). GA (no retain loss) and GA+Descent (forget batch cycled with every retain batch) still reach 0%.

### 3.3 Earlier post-hoc analysis (superseded)
Before these experiments, main-grid runs were split post hoc by a kernel kNN score (`records/results/entanglement_partitions_k{10,50,100}.txt`).
In S1 the kernel score is 0 for every forget sample (all nearest neighbours are other airplanes), so that analysis fell back to the
distance-to-retain ranking used above.

## 4. E4 - Embedding-space separability (linear probe), best configurations, S1
Protocol (`scripts/probe_best_models.py`): frozen 512-d penultimate features; a fresh linear head (Adam lr 1e-3, batch 1024, 100 epochs,
standardised features) trained on all 45k train-pool images (retain + forget); evaluated per class on test. NCC = nearest class mean of
train-pool features. Silhouette (class 0 vs rest) and 10-NN purity (share of the 10 nearest test neighbours of class-0 test images
that are class 0) measure whether class 0 still forms a cluster.
| model | output forget-cls | probe forget-cls | NCC forget-cls | probe other-cls | silhouette class 0 | 10-NN purity class 0 |
|---|---|---|---|---|---|---|
| Original | 95.40 ± 0.00 | 95.50 ± 0.17 | 95.40 ± 0.00 | 94.69 ± 0.05 | 0.127 ± 0.000 | 93.00 ± 0.00 |
| Retrain | 0.00 ± 0.00 | 87.73 ± 0.87 | 41.43 ± 2.97 | 91.19 ± 0.08 | -0.040 ± 0.005 | 52.58 ± 1.06 |
| GA (G3) | 0.60 ± 0.00 | 86.20 ± 0.44 | 30.33 ± 1.12 | 92.14 ± 0.16 | 0.025 ± 0.003 | 32.63 ± 0.77 |
| GA (G3) + WS | 0.80 ± 0.61 | 85.60 ± 0.95 | 34.80 ± 1.82 | 92.43 ± 0.17 | -0.022 ± 0.017 | 29.19 ± 2.20 |
| GA+Descent (D1) | 0.00 ± 0.00 | 81.80 ± 0.75 | 32.47 ± 1.26 | 92.10 ± 0.15 | -0.070 ± 0.004 | 32.38 ± 1.69 |
| GA+Descent (D1) + WS | 0.00 ± 0.00 | 82.87 ± 0.21 | 30.40 ± 0.72 | 92.64 ± 0.13 | -0.054 ± 0.003 | 32.73 ± 1.90 |
| Teacher-Student (T1) | 0.17 ± 0.29 | 88.50 ± 0.26 | 45.60 ± 1.35 | 93.23 ± 0.26 | -0.049 ± 0.007 | 56.07 ± 0.27 |
| Teacher-Student (T1) + WS | 1.90 ± 2.34 | 88.20 ± 0.62 | 46.50 ± 1.28 | 93.38 ± 0.08 | -0.038 ± 0.008 | 57.77 ± 0.72 |
| Random Relabel (R1) | 0.03 ± 0.06 | 94.30 ± 0.69 | 82.10 ± 0.95 | 94.01 ± 0.08 | 0.175 ± 0.015 | 89.24 ± 0.64 |
| Random Relabel (R1) + WS | 0.03 ± 0.06 | 94.17 ± 0.12 | 80.33 ± 1.22 | 94.00 ± 0.08 | 0.198 ± 0.007 | 89.41 ± 0.64 |
| Boundary Relabel (C1-mix) | 0.00 ± 0.00 | 94.33 ± 0.64 | 74.77 ± 4.50 | 93.76 ± 0.03 | 0.044 ± 0.003 | 82.16 ± 1.02 |
| Boundary Relabel (C1-mix) + WS | 0.00 ± 0.00 | 94.27 ± 0.45 | 75.03 ± 3.87 | 93.85 ± 0.08 | 0.041 ± 0.004 | 82.94 ± 1.83 |
Figure (t-SNE of 3000 test images per model, class 0 in black): `records/figures/fig_embed_best_S1c0.png`.

**Observations:**
- All methods remove class 0 from the output (0-2%), but a linear probe recovers it at 82-94% - and so does Retrain (87.7%): features
  learned from the other classes still describe airplanes, so probe accuracy alone does not prove residual memory.
- The cluster measures separate the methods: GA, GA+Descent and Teacher-Student disperse the airplane cluster to Retrain level or further
  (NCC 30-46%, 10-NN purity 29-58%; Retrain 41% / 53%); Random Relabeling and Boundary C1-mix leave it intact (NCC 75-82%, purity 82-89%,
  silhouette 0.04-0.20 vs Retrain -0.04): their forgetting happens in the classifier layer (the Illusion paper's finding).
- Boundary C1-mix is the best on output-level metrics (avg_gap 0.16) but keeps a clear class-0 cluster in feature space.
- WS does not change representation-level separability.

## 5. M5 - Perturbed forget samples (main-grid models)
Protocol (`scripts/eval_perturb.py`; Unseen-Threat paper's Gaussian perturbation): x' = clip(x + tau * N(0, I), 0, 1) in pixel space,
20 perturbations per forget image, tau in {0.01, 0.02, 0.03, 0.05, 0.1}, identical noise for every model. Reported: accuracy on the
perturbed forget images, and recovery = % of forget images misclassified when clean but correctly classified under >= 1 perturbation.
These were run on the main-grid checkpoints (before the Stage-5 improvements).
| scenario | model | WS | clean forget acc | perturbed forget acc tau 0.03 | recovery tau 0.03 | perturbed tau 0.1 | recovery tau 0.1 |
|---|---|---|---|---|---|---|---|
| S1c0 | Original |  | 100.0 | 80.6 | 0.0 | 8.7 | 0.0 |
| S1c0 | Retrain |  | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| S1c0 | M1 Gradient Ascent |  | 5.3 | 0.0 | 0.0 | 0.0 | 0.0 |
| S1c0 | M1 Gradient Ascent | yes | 6.3 | 0.0 | 0.0 | 0.0 | 0.0 |
| S1c0 | M2 GA + Descent |  | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| S1c0 | M2 GA + Descent | yes | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| S1c0 | M3a Teacher-Student |  | 0.3 | 0.0 | 0.0 | 0.0 | 0.0 |
| S1c0 | M3a Teacher-Student | yes | 2.6 | 0.0 | 0.0 | 0.0 | 0.0 |
| S1c0 | M3b-R Random Relabeling |  | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| S1c0 | M3b-R Random Relabeling | yes | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| S1c0 | M3b-B Boundary Relabeling |  | 23.6 | 0.8 | 0.2 | 0.0 | 0.0 |
| S1c0 | M3b-B Boundary Relabeling | yes | 3.2 | 0.2 | 0.6 | 0.0 | 0.0 |
| S2k359p30 | Original |  | 100.0 | 91.6 | 0.0 | 8.2 | 0.0 |
| S2k359p30 | Retrain |  | 90.9 | 78.4 | 44.5 | 7.1 | 8.6 |
| S2k359p30 | M1 Gradient Ascent |  | 87.5 | 45.5 | 21.7 | 0.2 | 0.0 |
| S2k359p30 | M1 Gradient Ascent | yes | 87.7 | 45.7 | 21.2 | 0.2 | 0.0 |
| S2k359p30 | M2 GA + Descent |  | 92.9 | 56.1 | 31.3 | 0.6 | 0.5 |
| S2k359p30 | M2 GA + Descent | yes | 93.5 | 56.4 | 29.7 | 0.6 | 0.5 |
| S2k359p30 | M3a Teacher-Student |  | 99.6 | 87.5 | 71.6 | 8.3 | 5.9 |
| S2k359p30 | M3a Teacher-Student | yes | 98.8 | 83.4 | 63.0 | 7.0 | 12.7 |
| S2k359p30 | M3b-R Random Relabeling |  | 88.7 | 71.0 | 41.4 | 7.2 | 9.3 |
| S2k359p30 | M3b-R Random Relabeling | yes | 94.3 | 78.4 | 42.2 | 9.1 | 9.7 |
| S2k359p30 | M3b-B Boundary Relabeling |  | 75.5 | 33.0 | 20.2 | 0.0 | 0.1 |
| S2k359p30 | M3b-B Boundary Relabeling | yes | 75.0 | 32.9 | 19.7 | 0.0 | 0.1 |
Full table (all tau, all scenarios): `records/results/perturb_table.csv`; figures `records/figures/fig_perturb_<scenario>.png`.

**Observations:** S1: no residual knowledge under noise for any method (0% on perturbed forget images; Original 80.6% at tau 0.03).
S2-30: Retrain itself recognises 78.4% of perturbed forget images (44.5% recovery - natural generalisation), so the reference is not 0.
Teacher-Student (87.5%, recovery 71.6%) shows more residual knowledge than Retrain (it barely changed the forget samples); GA (45.5%) and
Boundary (33.0%) are below Retrain (over-unlearning); Random Relabeling (71-78%) is closest to Retrain.

## 6. Post-unlearning quantization (best configurations)
**Protocol** (`scripts/eval_quant.py`): simulated weight-only, per-output-channel symmetric quantization of every Conv/Linear weight to
8 / 6 / 4 bits (scale = max|w| of the channel / (2^(b-1) - 1)); biases, BatchNorm parameters and running statistics stay fp32; forget /
retain / test accuracy and MIA re-evaluated. Models: the best configuration of every method, with and without WS, all 4 scenarios,
3 seeds (`records/results/quant_best.md`, raw `quant_best_raw.csv`, means `quant_best_table.csv`). avg_gap is computed against the fp32
Retrain of the same scenario. Mechanism columns: share of weights whose INT4 value equals the INT4 value of the Original, and mean
|theta_u - theta_o| / INT4 step of the Original.

| scenario | model | WS | config | forget fp32 | forget INT8 | forget INT6 | forget INT4 | change forget (INT4 - fp32) | change retain | change test | change MIA | avg_gap fp32 | avg_gap INT4 | share Q(theta_u)=Q(theta_o), INT4 | mean abs(delta)/step, INT4 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| S1c0 | Original |  |  | 100.00 | 100.00 | 100.00 | 100.00 | +0.00 | +0.00 | -0.33 | -0.60 | 29.05 | 28.82 | 1.000 | 0.000 |
| S1c0 | Retrain |  |  | 0.00 | 0.00 | 0.00 | 0.00 | +0.00 | -0.01 | -0.34 | -0.07 | 0.00 | 0.10 | 0.102 | 15.126 |
| S1c0 | M1 Gradient Ascent |  | G3 | 0.46 | 0.44 | 0.41 | 0.39 | -0.07 | -0.42 | -0.71 | +0.23 | 1.06 | 1.39 | 0.310 | 0.012 |
| S1c0 | M1 Gradient Ascent | yes | G3 | 0.63 | 0.63 | 0.62 | 0.59 | -0.04 | -0.42 | -0.82 | +0.12 | 1.17 | 1.50 | 0.495 | 0.010 |
| S1c0 | M2 GA + Descent |  | D1 | 0.00 | 0.00 | 0.00 | 0.00 | +0.00 | -0.33 | -0.43 | -0.37 | 1.34 | 1.62 | 0.288 | 0.124 |
| S1c0 | M2 GA + Descent | yes | D1 | 0.00 | 0.00 | 0.00 | 0.00 | +0.00 | -0.31 | -0.49 | +0.37 | 0.75 | 0.87 | 0.459 | 0.087 |
| S1c0 | M3a Teacher-Student |  | T1 | 0.33 | 0.32 | 0.30 | 0.27 | -0.06 | -0.03 | -0.30 | +0.35 | 0.60 | 0.75 | 0.290 | 0.082 |
| S1c0 | M3a Teacher-Student | yes | T1 | 2.56 | 2.56 | 2.51 | 2.37 | -0.19 | -0.02 | -0.20 | -0.68 | 1.02 | 0.86 | 0.463 | 0.054 |
| S1c0 | M3b-R Random Relabeling |  | R1 | 0.00 | 0.00 | 0.00 | 0.01 | +0.01 | -0.01 | -0.30 | -0.02 | 1.19 | 1.26 | 0.241 | 0.200 |
| S1c0 | M3b-R Random Relabeling | yes | R1 | 0.00 | 0.00 | 0.00 | 0.00 | +0.00 | -0.00 | -0.27 | +0.50 | 1.17 | 1.36 | 0.357 | 0.144 |
| S1c0 | M3b-B Boundary Relabeling |  | C1-mix | 0.00 | 0.00 | 0.00 | 0.00 | +0.00 | -0.01 | -0.37 | -0.43 | 0.07 | 0.26 | 0.290 | 0.059 |
| S1c0 | M3b-B Boundary Relabeling | yes | C1-mix | 0.00 | 0.00 | 0.00 | 0.00 | +0.00 | -0.01 | -0.37 | -0.87 | 0.27 | 0.16 | 0.472 | 0.044 |
| S2k359p10 | Original |  |  | 100.00 | 100.00 | 100.00 | 100.00 | +0.00 | +0.00 | -0.33 | +0.72 | 4.47 | 4.63 | 1.000 | 0.000 |
| S2k359p10 | Retrain |  |  | 91.63 | 91.63 | 91.46 | 90.59 | -1.04 | +0.00 | -0.28 | -0.16 | 0.00 | 0.37 | 0.104 | 17.137 |
| S2k359p10 | M1 Gradient Ascent |  | G3-S2 | 95.38 | 95.43 | 94.81 | 91.43 | -3.95 | -1.00 | -0.98 | -0.98 | 4.79 | 4.15 | 0.313 | 0.003 |
| S2k359p10 | M1 Gradient Ascent | yes | G3-S2 | 96.74 | 96.79 | 96.42 | 93.43 | -3.31 | -0.87 | -1.02 | -1.05 | 5.01 | 4.40 | 0.489 | 0.002 |
| S2k359p10 | M2 GA + Descent |  | D1-S2 | 91.41 | 91.28 | 91.26 | 87.09 | -4.32 | -0.84 | -0.87 | -1.37 | 3.46 | 4.63 | 0.304 | 0.062 |
| S2k359p10 | M2 GA + Descent | yes | D1-S2 | 92.32 | 92.44 | 92.27 | 88.49 | -3.83 | -0.75 | -0.85 | -0.88 | 3.59 | 4.38 | 0.477 | 0.041 |
| S2k359p10 | M3a Teacher-Student |  | T4-best S2 | 97.53 | 97.53 | 97.48 | 96.10 | -1.43 | -0.27 | -0.55 | -0.52 | 3.82 | 3.54 | 0.282 | 0.073 |
| S2k359p10 | M3a Teacher-Student | yes | T4-best S2 | 97.85 | 97.88 | 97.88 | 96.67 | -1.19 | -0.21 | -0.46 | -0.12 | 3.87 | 3.71 | 0.442 | 0.050 |
| S2k359p10 | M3b-R Random Relabeling |  | R1-S2 | 95.16 | 95.26 | 95.14 | 93.78 | -1.38 | -0.77 | -0.72 | -0.53 | 3.91 | 3.80 | 0.180 | 0.758 |
| S2k359p10 | M3b-R Random Relabeling | yes | R1-S2 | 97.23 | 97.26 | 97.14 | 97.11 | -0.12 | -0.34 | -0.50 | -0.47 | 4.36 | 4.42 | 0.260 | 0.475 |
| S2k359p10 | M3b-B Boundary Relabeling |  | C0 | 73.56 | 73.98 | 72.10 | 65.53 | -8.02 | -2.13 | -1.67 | -0.68 | 11.22 | 14.01 | 0.314 | 0.001 |
| S2k359p10 | M3b-B Boundary Relabeling | yes | C0 | 73.23 | 73.58 | 71.78 | 65.31 | -7.93 | -2.14 | -1.75 | -0.86 | 11.31 | 14.05 | 0.490 | 0.001 |
| S2k359p30 | Original |  |  | 100.00 | 100.00 | 100.00 | 100.00 | +0.00 | +0.00 | -0.33 | +0.72 | 4.75 | 4.85 | 1.000 | 0.000 |
| S2k359p30 | Retrain |  |  | 90.86 | 90.92 | 90.94 | 90.63 | -0.23 | -0.00 | -0.33 | +0.15 | 0.00 | 0.18 | 0.102 | 14.628 |
| S2k359p30 | M1 Gradient Ascent |  | G3-S2 | 90.55 | 90.55 | 89.51 | 85.26 | -5.29 | -1.17 | -1.28 | -0.81 | 4.32 | 6.05 | 0.313 | 0.004 |
| S2k359p30 | M1 Gradient Ascent | yes | G3-S2 | 90.46 | 90.41 | 89.55 | 84.86 | -5.60 | -1.24 | -1.39 | -0.86 | 4.36 | 6.20 | 0.485 | 0.003 |
| S2k359p30 | M2 GA + Descent |  | D1-S2 | 92.92 | 92.91 | 92.94 | 88.35 | -4.57 | -0.97 | -1.19 | -0.75 | 4.15 | 4.61 | 0.304 | 0.060 |
| S2k359p30 | M2 GA + Descent | yes | D1-S2 | 93.53 | 93.65 | 93.51 | 89.20 | -4.33 | -0.87 | -1.01 | -0.76 | 4.38 | 4.41 | 0.475 | 0.038 |
| S2k359p30 | M3a Teacher-Student |  | T4-best S2 | 94.60 | 94.59 | 94.46 | 92.53 | -2.07 | -0.59 | -0.81 | -0.36 | 3.33 | 3.07 | 0.269 | 0.137 |
| S2k359p30 | M3a Teacher-Student | yes | T4-best S2 | 94.50 | 94.47 | 94.53 | 92.35 | -2.16 | -0.44 | -0.65 | -0.42 | 3.47 | 3.10 | 0.410 | 0.077 |
| S2k359p30 | M3b-R Random Relabeling |  | R1-S2 | 88.67 | 88.77 | 88.60 | 87.74 | -0.92 | -0.63 | -0.67 | -0.23 | 3.25 | 3.75 | 0.205 | 0.850 |
| S2k359p30 | M3b-R Random Relabeling | yes | R1-S2 | 94.35 | 94.47 | 94.40 | 93.92 | -0.43 | -0.41 | -0.40 | -0.37 | 3.10 | 3.10 | 0.274 | 0.550 |
| S2k359p30 | M3b-B Boundary Relabeling |  | C0 | 75.54 | 75.79 | 74.34 | 67.63 | -7.91 | -1.84 | -1.67 | -1.13 | 9.92 | 12.49 | 0.313 | 0.004 |
| S2k359p30 | M3b-B Boundary Relabeling | yes | C0 | 74.99 | 75.25 | 73.92 | 66.86 | -8.12 | -1.82 | -1.63 | -1.12 | 10.12 | 12.73 | 0.485 | 0.002 |
| S2k359p50 | Original |  |  | 100.00 | 100.00 | 100.00 | 100.00 | +0.00 | +0.00 | -0.33 | +0.70 | 5.26 | 5.36 | 1.000 | 0.000 |
| S2k359p50 | Retrain |  |  | 88.74 | 88.75 | 88.69 | 87.99 | -0.75 | +0.00 | -0.33 | +0.03 | 0.00 | 0.28 | 0.101 | 14.567 |
| S2k359p50 | M1 Gradient Ascent |  | G3-S2 | 88.31 | 88.35 | 86.96 | 82.00 | -6.31 | -1.10 | -1.70 | -0.88 | 4.06 | 6.11 | 0.313 | 0.005 |
| S2k359p50 | M1 Gradient Ascent | yes | G3-S2 | 88.05 | 88.07 | 86.81 | 81.98 | -6.07 | -1.06 | -1.55 | -0.81 | 4.19 | 6.15 | 0.482 | 0.004 |
| S2k359p50 | M2 GA + Descent |  | D1-S2 | 89.26 | 89.20 | 88.92 | 84.45 | -4.80 | -0.85 | -1.13 | -0.76 | 3.73 | 4.97 | 0.304 | 0.058 |
| S2k359p50 | M2 GA + Descent | yes | D1-S2 | 90.67 | 90.58 | 90.47 | 87.03 | -3.64 | -0.64 | -1.03 | -0.51 | 4.04 | 4.27 | 0.473 | 0.037 |
| S2k359p50 | M3a Teacher-Student |  | T4-best S2 | 94.10 | 94.04 | 93.79 | 91.60 | -2.50 | -0.43 | -0.88 | -0.44 | 3.62 | 3.21 | 0.276 | 0.099 |
| S2k359p50 | M3a Teacher-Student | yes | T4-best S2 | 91.67 | 91.65 | 91.94 | 88.44 | -3.23 | -0.53 | -1.04 | -0.52 | 3.42 | 3.02 | 0.422 | 0.061 |
| S2k359p50 | M3b-R Random Relabeling |  | R1-S2 | 81.07 | 81.08 | 81.14 | 79.86 | -1.20 | -1.00 | -0.99 | -0.25 | 5.43 | 6.17 | 0.208 | 0.861 |
| S2k359p50 | M3b-R Random Relabeling | yes | R1-S2 | 88.96 | 88.99 | 89.16 | 88.07 | -0.89 | -0.78 | -0.57 | -0.49 | 2.21 | 2.54 | 0.276 | 0.569 |
| S2k359p50 | M3b-B Boundary Relabeling |  | C0 | 75.11 | 75.37 | 73.81 | 67.84 | -7.26 | -1.23 | -1.58 | -0.95 | 8.98 | 11.26 | 0.311 | 0.006 |
| S2k359p50 | M3b-B Boundary Relabeling | yes | C0 | 75.54 | 75.77 | 74.36 | 68.35 | -7.19 | -1.22 | -1.57 | -0.89 | 8.76 | 11.03 | 0.481 | 0.004 |

**Margin diagnostic** (`records/results/quant_margin_diag.md`, S2-30, seed 0): p_y = fp32 softmax probability of the true label;
"low margin" = correctly classified with p_y < 0.9; "flipped" = correct at fp32, wrong at INT4; same-class test = test images of
classes 3, 5, 9.
| model | set | acc fp32 | acc INT | flipped (% of set) | median p_y all | % low-margin among correct | median p_y of flipped | median p_y of kept-correct |
|---|---|---|---|---|---|---|---|---|
| Retrain | forget | 91.48 | 90.69 | 1.51 | 0.999 | 5.3 | 0.682 | 0.999 |
| Retrain | same-class test | 90.93 | 89.67 | 1.90 | 0.999 | 6.2 | 0.673 | 0.999 |
| GA | forget | 91.78 | 87.04 | 5.48 | 0.989 | 16.5 | 0.681 | 0.993 |
| GA | same-class test | 72.13 | 69.17 | 3.97 | 0.970 | 17.7 | 0.641 | 0.993 |
| GAGD | forget | 91.98 | 88.12 | 4.91 | 0.996 | 12.6 | 0.664 | 0.997 |
| GAGD | same-class test | 74.30 | 71.60 | 3.90 | 0.989 | 13.2 | 0.628 | 0.997 |
| TS | forget | 94.42 | 90.77 | 4.20 | 0.996 | 10.6 | 0.697 | 0.997 |
| TS | same-class test | 83.57 | 80.17 | 4.63 | 0.993 | 14.4 | 0.651 | 0.997 |
| RLrand | forget | 90.30 | 88.69 | 2.32 | 0.987 | 18.6 | 0.580 | 0.991 |
| RLrand | same-class test | 82.87 | 81.27 | 2.67 | 0.981 | 20.6 | 0.539 | 0.992 |
| RLbnd | forget | 76.27 | 68.00 | 10.05 | 0.862 | 43.3 | 0.628 | 0.931 |
| RLbnd | same-class test | 58.37 | 53.47 | 6.53 | 0.700 | 40.4 | 0.584 | 0.937 |

Figures: `records/figures/fig_quant_best_forget_vs_bits.png` (forget acc vs bit width), `fig_quant_best_int4_delta.png` (INT4 - fp32
changes), `fig_quant_best_avg_gap.png` (avg_gap fp32 vs INT4), `fig_quant_best_mechanism.png` (INT4 forget-acc drop vs update size),
`fig_quant_margin_diag.png` (margin diagnostic), `fig_embed_quant_S1c0_int4.png` (t-SNE of test features, fp32 vs INT4, S1 seed 0).

**Observations:**
- **INT8 and INT6 change nothing that matters.** Largest forget-acc change: INT8 +0.42 (Boundary Relabeling S2-10), INT6 -1.46 (same model).
- **S1 (single-class forgetting): no relapse at any bit width.** INT4 changes forget acc by -0.19 to +0.01 for every method (all stay
  <= 2.4%), retain acc by -0.42 to 0, test acc by -0.82 to -0.20, MIA by -0.87 to +0.50. avg_gap changes by -0.16 to +0.33. The t-SNE
  of the test features looks the same at INT4 as at fp32 for every model - the airplane cluster neither reappears nor disappears.
- **S2 (subset forgetting): INT4 lowers forget accuracy further, for every method, but not selectively.** Forget acc change at INT4:
  Gradient Ascent -3.3 to -6.3, GA + Descent -3.6 to -4.8, Teacher-Student -1.2 to -3.2, Random Relabeling -0.1 to -1.4, Boundary
  Relabeling -7.2 to -8.1 (Retrain -0.2 to -1.0, Original 0). At the same time retain acc drops by 0.2-2.1, test acc by 0.4-1.8 and
  same-class test images are flipped at a similar rate as forget images (S2-30 seed 0: Gradient Ascent 5.5% of forget vs 4.0% of
  same-class test, Boundary 10.1% vs 6.5%). MIA moves 0.1-1.4 points towards 50% but stays at 51.5-59.4%.
- **Effect on avg_gap depends on whether the fp32 model was above or below Retrain's forget accuracy.** It improves for Teacher-Student in
  every S2 scenario (S2-10 3.82 -> 3.54, S2-30 3.33 -> 3.07, S2-50 3.62 -> 3.21) and for Gradient Ascent on S2-10 (4.79 -> 4.15; fp32
  forget 95.4% vs Retrain 91.6%). It worsens for Gradient Ascent on S2-30 / S2-50 (4.32 -> 6.05, 4.06 -> 6.11: forget acc falls to 85.3% /
  82.0%, below Retrain's 90.9% / 88.7%), for GA + Descent (+0.03 to +1.24), Boundary Relabeling (+2.3 to +2.8) and mostly for Random
  Relabeling (-0.11 to +0.74).
- **Why INT4 lowers S2 forget accuracy: unlearning leaves forget-class samples close to the decision boundary.** Among correctly
  classified forget images, the share with p_y < 0.9 is 5.3% for Retrain but 10.6% (Teacher-Student), 12.6% (GA + Descent), 16.5%
  (Gradient Ascent), 18.6% (Random Relabeling) and 43.3% (Boundary Relabeling); the same holds for same-class test images. The images
  flipped by INT4 have median p_y 0.54-0.70, the ones that stay correct 0.93-0.999. INT4 rounding acts like small weight noise and
  pushes these low-margin images over the boundary (Retrain flips 1.5% of forget images, the methods 2.3-10.1%).
- **The INT4 drop is largest for the methods with the smallest weight change** (`fig_quant_best_mechanism.png`): Gradient Ascent
  (mean |delta| = 0.002-0.005 INT4 steps) and Boundary C0 (0.001-0.006) lose 3.3-8.1 points; Random Relabeling (0.48-0.86 steps) loses
  0.1-1.4 and Retrain (15-17 steps) 0.2-1.0. Tiny updates move forget samples just to the boundary; Random Relabeling moves the weights
  more and its flipped share is low (2.3%) although its low-margin share is high (18.6%) - margin alone does not explain all of it.
- **Share of weights with the same INT4 value as the Original:** 0.18-0.31 without WS, 0.26-0.50 with WS (Retrain 0.10). The metric
  compares de-quantised values (integer code x channel scale), so a weight counts as "different" whenever the maximum |w| of its channel
  changed, even if its integer code is the same; this is why Gradient Ascent (mean |delta| ~0.003 steps) shows only ~31%. WS raises the
  share because half of the weights - and therefore many whole channels - are never updated.

**Conclusions:**
1. **H1 (quantization restores forgotten knowledge) is rejected** for weight-only quantization: no method and no scenario regains forget
   accuracy at INT8 / INT6 / INT4 (largest increase +0.42 at INT8). The S1 forgetting of all best configurations is stable under INT4.
2. **H2 (quantization helps forgetting) holds only as non-selective degradation:** INT4 lowers S2 forget accuracy together with retain,
   test and same-class test accuracy. It is not an unlearning tool; it brings a model closer to Retrain only when the fp32 model was
   under-forgetting (Teacher-Student, Gradient Ascent on S2-10), and over-forgets otherwise.
3. **INT8 is safe for every unlearned model; INT6 nearly so (<= 1.5 points); INT4 is not**, and least for methods that leave many
   low-margin forget-class samples (Boundary Relabeling, Gradient Ascent). Random Relabeling is the most quantization-stable method.
4. Weight-only quantization keeps BatchNorm statistics in fp32. For the main-grid Gradient Ascent / Boundary models, whose forgetting was
   stored mainly in the BN running statistics, a pipeline that re-estimates BN statistics would restore the forgotten class (see the
   BN analysis on the method pages); the best configurations (mixed BN batches) were designed to avoid this.

## 7. Summary tables (best configurations; one value per column)
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

### 7.1 S1 - class 0 (single-class forgetting)
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

### 7.2 S2 - random 10% / 30% / 50% of classes 3, 5, 9 (subset forgetting)
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

## 8. Cross-cutting conclusions
1. Output-level metrics (accuracy, MIA) rank Boundary C1-mix, Teacher-Student and GA+Descent best in S1; feature-level metrics show that
   the relabeling methods only change the classifier head.
2. Subset forgetting: highly entangled forget samples are harder to forget (forget accuracy rises low -> high for every method), and
   forcing them down costs retain / test accuracy. Single-class forgetting (corrected setup): every partition is forgotten almost
   completely and the forgetting spreads to the whole class, so entanglement does not make class forgetting harder at the accuracy level.
3. Sequential low -> high unlearning helps the relabeling methods in S2 but hurts methods that rely on repeated ascent steps.
4. Robustness: no residual knowledge under Gaussian noise in S1. Weight-only quantization of the best configurations never restores
   forgotten knowledge (S1 stable at INT4); in S2, INT4 lowers forget accuracy only together with retain / test accuracy, because
   unlearning leaves forget-class samples near the decision boundary. BN-statistics recalibration can restore the forgotten class of the
   main-grid forget-only models (see the BN analysis).
5. Open: the perturbation evaluation was run on the main-grid models; repeating it on the best configurations is pending.
