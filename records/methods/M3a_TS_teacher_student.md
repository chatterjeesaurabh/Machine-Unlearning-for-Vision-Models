# M3a - Teacher-Student Unlearning

Last updated: 2026-10-28. Full run list (every run, best row in bold): `records/results/runs_TS.md` (246 groups) and
`records/results/runs_TSjoint.md`. Chronology: `records/JOURNAL.md`.

## 1. Method and implementation
**Code:** `src/mu/unlearn/methods.py::teacher_student` (reference: `sample_repos/RUM-main/RUM-main/unlearn/scrub.py`), variant
`teacher_student_joint`. Config keys: `TS`, `TSjoint`.
Teacher = frozen Original (eval mode); student = copy of the Original. Training runs for `epochs` (= paper's STEPS) epochs:
- **max-step** (only in the first `msteps` epochs): one pass over the forget set minimising `-KL_T(teacher || student)` (push away);
- **min-step** (every epoch): one pass over the retain set minimising `alpha * KL_T(teacher || student) + gamma * CE(student, y)`;
- epochs after `msteps` are min-only (the paper's "extra min-steps"); lr x0.1 at `lr_decay_epochs`; KL with temperature T (x T^2).
So Table 3's "2 max-steps, 3 min-steps" = 3 epochs, the first two with a forget pass.
Our additions: `select_best` (keep the epoch-end checkpoint closest to Retrain on forget/retain/val acc; no test data) and
`TSjoint` (all three terms every step on a retain batch + a cycled forget batch, forget term weighted by `lam_f`; no alternation).
Repo bugs avoided: off-by-one in msteps (repo runs msteps+1 max epochs), unused `max_iter`.

## 2. Hyperparameter sources
- Teacher-Student paper Table 3 (ResNet, CIFAR-10): class: forget batch 512, retain batch 128, 2 max / 3 total epochs; selective
  (100 samples of one class): 16 / 64, 5 / 5; lr 5e-4 with x0.1 decay, wd 5e-4, momentum 0.9. alpha, gamma, T for SCRUB are not
  stated (T = 4 is stated only for the Bad-T baseline). Figure 5: sensitivity over gamma, alpha in {0.1, 0.5, 1, 2, 3}.
- RUM repo README example: `--kd_T 4.0 --msteps 4 --unlearn_lr 5e-3 --unlearn_epochs 5` (source of our T = 4).
- Illusion paper (CIFAR-10 ResNet-18): 3 epochs, batch 64, msteps 2.
- Common to all our runs: SGD momentum 0.9, weight decay 5e-4, lr 5e-4, random crop + flip, bf16.

## 3. Named configurations
| name | T | alpha (KL weight) | gamma (CE weight) | epochs | msteps | forget batch | retain batch | other | where |
|---|---|---|---|---|---|---|---|---|---|
| T0 paper Table 3 | 4 | 0.5 or 1 | 0.5 or 1 | 3 | 2 | 512 | 128 | | pilot |
| T0-sel paper "selective" | 4 | 1 | 1 | 5 | 5 | 16 | 64 | | pilot |
| **T1** main grid S1 | 4 | 0.1 | 1 | 3 | 2 | 128 | 128 | | pilot, main grid |
| T1-S2 main grid S2 | 4 | 0.001 | 1 | 5 | 2 | 64 | 64 | | pilot, main grid |
| T1-S2 main grid S2 + WS | 4 | 0.01 | 1 | 5 | 3 | 64 | 64 | | pilot, main grid |
| T2 10 epochs, S1 | 4 | 0, 1e-4, 1e-3 or 0.01 | 1 | 10 | 2 or 5 | 128 | 128 | lr x0.1 at epoch 8 | imp |
| T2 10 epochs, S2 | 4 | 0, 1e-3 or 0.01 | 1 | 10 | 2 or 5 | 128 | 64 | lr x0.1 at epoch 8 | imp, imp2 |
| T3 Fig.5 grid, Table-3 schedule | 1 or 4 | 0.1, 0.5, 1, 2 or 3 | 0.1, 0.5, 1, 2 or 3 | 3 | 2 | 512 | 128 | | tsA (S1) |
| **T4** Fig.5 grid, Illusion schedule | 1 or 4 | 0.1, 0.5, 1, 2 or 3 | 0.1, 0.5, 1, 2 or 3 | 3 | 2 | 64 | 64 | | tsA2 (S1), tsA (S2-30) |
| T4-best S1 | 4 | 0.5 | 1 | 3 | 2 | 64 | 64 | | imp3 (3 seeds) |
| **T4-best S2** (S2-10, S2-30) | 4 | 0.1 | 0.1 | 3 | 2 | 64 | 64 | | imp3 (3 seeds) |
| **T4-best S2** (S2-50) | 4 | 0.1 | 0.1 | 3 | 2 | 128 | 64 | | imp4 (3 seeds) |
| T5 best-epoch, S1 | 1 or 4 | 0.1, 0.5 or 1 | 1 | 10 | 5 | 512 | 128 | select_best | tsB |
| T5 best-epoch, S2 | 1 or 4 | 0.1, 0.5 or 1 | 1 | 10 | 5 | 64 | 64 | select_best | tsB |
| T6 joint loss, S1 | 1 or 4 | 0.1 or 1 | 1 | 5 | - | 128 | 128 | lam_f 1 or 0.1, select_best | tsC |
| T6 joint loss, S2 | 1 or 4 | 0.1 or 1 | 1 | 5 | - | 128 | 64 | lam_f 1 or 0.1, select_best | tsC |

## 4. Results - S1 (class 0), Retrain: forget 0.00, retain 100, test other-cls 94.82, MIA 51.0
Seed 0 unless "3 seeds". sel_gap (selection; no test data) and avg_gap (report): mean |metric - Retrain|. Best in **bold**.
| config | seeds | forget | retain | test other-cls | MIA | sel_gap | avg_gap |
|---|---|---|---|---|---|---|---|
| Original | 3 | 100.00 | 100.00 | 94.76 | 57.78 | 28.51 | 29.05 |
| T0 paper Table 3 (alpha 0.5, gamma 0.5) | 1 | 97.80 | 100.00 | 94.77 | 60.05 | 28.34 | 28.57 |
| T0 paper Table 3 (alpha 1, gamma 1) | 1 | 99.51 | 100.00 | 94.79 | 59.65 | 28.76 | 29.10 |
| T3 Fig.5 grid, Table-3 schedule: best of 50 (T 4, alpha 0.1, gamma 0.1) | 1 | 79.67 | 100.00 | 94.61 | 58.05 | 22.64 | 23.06 |
| T1 alpha 0.1, T=1 instead of 4 (bs 128) | 1 | 17.91 | 99.98 | 94.09 | 54.35 | 5.21 | 5.46 |
| T1 alpha 1, T=1 (bs 128) | 1 | 22.27 | 100.00 | 94.46 | 54.20 | 6.70 | 6.69 |
| **T1 (main grid)** | 3 | 0.33 | 99.99 | 94.30 | 52.58 | 1.86 | **0.60** |
| T1 + WS (main grid) | 3 | 2.56 | 99.99 | 94.26 | 52.18 | 2.01 | 1.02 |
| T2 10 ep, msteps 2, alpha 0.01 | 1 | 0.11 | 100.00 | 94.64 | 52.95 | 0.93 | 0.38 |
| T2 10 ep, msteps 5, alpha 0 | 1 | 0.00 | 76.62 | 75.37 | 51.35 | 11.27 | 10.31 |
| T2 10 ep, msteps 5, alpha 1e-4 | 1 | 0.00 | 46.06 | 46.40 | 52.20 | 25.25 | 24.51 |
| T2 10 ep, msteps 5, alpha 1e-3 | 1 | 0.00 | 76.59 | 75.43 | 49.55 | 11.59 | 10.75 |
| T2 10 ep, msteps 5, alpha 0.01 | 1 | 0.00 | 50.63 | 50.71 | 49.85 | 22.79 | 22.73 |
| T4 Illusion schedule, T=1, best (alpha 1, gamma 0.5) | 1 | 0.49 | 99.99 | 94.41 | 54.05 | 0.65 | 0.80 |
| **T4 Illusion schedule, T=4, alpha 0.5, gamma 1** | 1 | **0.00** | **99.88** | **94.11** | **52.35** | **0.41** | **0.35** |
| T4-best S1 | 3 | 0.01 | 99.84 | 94.10 | 54.18 | 1.50 | 1.00 |
| T4-best S1 + WS | 3 | 0.55 | 100.00 | 94.75 | 53.33 | 1.56 | 0.76 |
| T5 best-epoch, best (T 4, alpha 0.1) | 1 | 35.49 | 100.00 | 94.54 | 53.30 | 10.14 | 10.09 |
| T6 joint, T 1, alpha 1, lam_f 0.1 | 1 | 0.00 | 98.79 | 91.74 | 47.90 | 1.18 | 1.94 |
| T6 joint, T 4, alpha 0.1, lam_f 1 | 1 | 0.00 | 11.11 | 11.11 | 50.00 | 41.63 | 41.48 |

Fig.5 grid with the Illusion schedule (S1, forget acc %): T=1: all 25 alpha/gamma combinations forget (0.0-11.5%, test other
93.4-94.4); T=4: alpha <= 0.5 -> 0.0% for every gamma; alpha 1 -> 19-40%; alpha 2-3 -> 20-78%. Full grids in `records/results/runs_TS.md`.
Across 3 seeds T1 (main grid, avg_gap 0.60) and T4-best (1.00) are equivalent within noise (seed std of sel_gap ~0.8).

Paper reference (Teacher-Student paper Table 9, ResNet CIFAR-10 class unlearning, models pre-trained on CIFAR-100, no augmentation):
SCRUB test error 15.06 / retain error 0.12 / forget error 100 (forget acc 0); Retrain test error 13.97. Ours T1: forget 0.33, retain 99.99.
Illusion paper (CIFAR-10 class): SCRUB output forget acc 0.48, linear-probe forget acc 74.79. Ours T1: output 0.17, probe 88.23 (Retrain 87.70).

## 5. Results - S2 (random % of classes 3, 5, 9)
| scenario | config | seeds | forget | retain | test | MIA | sel_gap | avg_gap |
|---|---|---|---|---|---|---|---|---|
| S2-30 (Retrain forget 90.86, MIA 49.95; Original sel_gap 4.51) | T0-sel paper selective (forget bs 16, retain bs 64, 5 epochs, msteps 5) | 1 | 0.00 | 12.63 | 11.74 | 50.0 | 65.51 | 65.47 |
| S2-30 | T1-S2 (main grid) | 3 | 99.58 | 99.97 | 93.78 | 57.21 | 3.75 | 4.17 |
| S2-30 | T2 10 ep, msteps 5, alpha 1e-3 | 3 | 99.09 | 99.94 | 93.50 | 56.91 | 3.52 | 4.05 |
| S2-30 | T5 best-epoch best | 1 | 98.47 | 99.66 | 92.84 | 56.85 | 3.51 | 3.97 |
| S2-30 | T6 joint, T 1, alpha 0.1, lam_f 0.1 | 1 | 42.52 | 87.46 | 78.19 | 54.83 | 20.25 | 20.63 |
| S2-30 | T6 joint, T 1, alpha 0.1, lam_f 1 | 1 | 0.00 | 25.89 | 23.59 | 50.00 | 59.38 | 59.19 |
| S2-30 | T6 joint, T 4, alpha 0.1, lam_f 0.1 | 1 | 11.11 | 79.69 | 70.22 | 53.70 | 32.00 | 32.13 |
| S2-30 | T6 joint, T 4, alpha 0.1, lam_f 1 | 1 | 0.00 | 10.99 | 10.00 | 50.00 | 66.39 | 66.32 |
| S2-30 | T6 joint, T 1, alpha 1, lam_f 0.1 | 1 | 55.83 | 90.54 | 81.10 | 55.67 | 15.60 | 16.01 |
| S2-30 | T6 joint, T 1, alpha 1, lam_f 1 | 1 | 0.00 | 25.82 | 23.56 | 50.83 | 59.87 | 59.33 |
| S2-30 | T6 joint, T 4, alpha 1, lam_f 0.1 | 1 | 33.51 | 85.07 | 76.07 | 54.95 | 23.98 | 24.04 |
| S2-30 | T6 joint, T 4, alpha 1, lam_f 1 | 1 | 0.00 | 18.72 | 17.32 | 50.00 | 62.64 | 62.55 |
| S2-30 | **T4-best S2 (T 4, alpha 0.1, gamma 0.1)** | 3 | **94.60** | **98.85** | **92.17** | **56.13** | **2.78** | **3.34** |
| S2-30 | T4-best S2 + WS | 3 | 94.50 | 98.93 | 91.96 | 56.64 | 3.13 | 3.47 |
| S2-10 (Retrain forget 91.63) | T1-S2 (main grid) | 3 | 99.83 | 99.99 | 94.17 | 58.27 | 3.91 | 4.32 |
| S2-10 | **T4-best S2** | 3 | 97.48 | 99.55 | 93.07 | 57.02 | **3.29** | 3.81 |
| S2-50 (Retrain forget 88.74) | T1-S2 (main grid) | 3 | 0.00 | 26.89 | 23.83 | 49.71 | 58.22 | 58.20 |
| S2-50 | T4-best S2, forget bs 64 | 3 | 0.00 | 24.79 | 21.29 | 49.93 | 59.41 | 59.31 |
| S2-50 | **T4-best S2, forget bs 128** | 3 | **94.08** | **99.06** | **91.86** | **57.04** | **3.64** | **3.62** |
| S2-50 | T4-best S2, forget bs 128 + WS | 3 | 91.66 | 98.64 | 91.01 | 57.42 | 3.57 | 3.43 |

## 6. Issues found and how they were resolved
**(a) Paper settings did not forget class 0 (forget acc 98-100%).** Per-epoch logging (`forget_acc_after_max` in history.csv)
showed forget acc 0% right after each max-step but ~99% again after the min-step, although the S1 retain set has no airplanes.
Cause 1 - **soft-label re-teaching**: at T=4 the teacher gives class 0 a mean probability of 0.0496 on retain images (0.00011 at T=1);
class 0 is the teacher's second class for 44.7% of ships and 24.9% of birds (`records/results/ts_softlabel_diag.md`). Matching
these soft labels rebuilds a class-0 output that fires on airplane-like features. Evidence: T=1 with the same alpha forgets
much more (18-22% vs 98-99%).
Cause 2 - **too few max-step updates**: with Table 3's forget batch 512 an epoch has only 9 forget updates vs ~316 retain updates;
no alpha/gamma/T in the Fig.5 grid forgets with this schedule (T3), while with batch 64 (Illusion schedule, 70 forget updates)
almost all combinations do (T4).
**Resolution:** small KL weight (alpha 0.1) with forget batch 128 (T1, main grid) or the Illusion schedule with alpha <= 0.5 at
T=4 / any alpha at T=1 (T4).

**(b) S2 bistability / collapse.** In S2 the retain set contains the same classes, so the min-steps directly restore the forget
samples: with few max-steps nothing is forgotten (forget ~99-100%), with many max-steps the model collapses (retain 12-34%).
The number of max-step updates (|forget| / forget batch x msteps) is what matters: the paper's selective setting (bs 16, 5 max
epochs) gives 1265 updates on our 4050-sample forget set and collapses; the main-grid S2 setting (selected on S2-30, bs 64)
collapsed on S2-50 (larger forget set -> 70% more updates). **Resolution:** Illusion schedule with alpha 0.1, gamma 0.1 and the
forget batch scaled with the forget-set size (64 for S2-10/30, 128 for S2-50): stable on 3 seeds in all S2 ratios
(sel_gap 3.29 / 2.78 / 3.64), the first TS setting that clearly beats doing nothing (Original ~4.5) in S2.

**(c) 10 epochs.** Longer training only helps if msteps stays small: msteps 2 -> unchanged quality (3x slower); msteps 5 -> collapse.
The best-epoch checkpoint (T5) did not help because checkpoints are taken only at epoch ends (after the min-step); the moment
right after a max-step (forget 0%) is never a candidate.

**(d) Joint loss (T6).** Optimising all three terms together works only at T=1 with a small forget weight (lam_f 0.1) in S1
(sel_gap 1.18, retain 98.8%) and collapses in S2 within one epoch. The alternating schedule is better - consistent with the
paper's remark that the joint objective oscillates.

**(e) Representations.** Output forget acc 0.17% but linear-probe forget-class acc 88% (Retrain 87.7%): the forgotten class
remains linearly decodable from the features, as for Retrain (`records/results/repr_table.md`).

## 7. Comparison with the other methods (S1, 3-seed main grid; `records/results/main_table.md`)
| method | forget | retain | test other-cls | MIA | avg_gap |
|---|---|---|---|---|---|
| M1 Gradient Ascent | 5.33 | 93.91 | 86.99 | 51.17 | 4.58 |
| M2 GA + Descent | 0.00 | 99.09 | 91.87 | 49.25 | 1.34 |
| **M3a Teacher-Student (T1)** | 0.33 | 99.99 | 94.30 | 52.58 | **0.60** |
| M3b-R Random Relabeling | 0.00 | 100.00 | 94.54 | 55.47 | 1.19 |
| M3b-B Boundary Relabeling (main) | 23.59 | 99.48 | 92.31 | 54.28 | 6.92 |
S2-30 (3 seeds, avg_gap): GA 5.55, GA+Descent 4.15, Teacher-Student main 4.17 -> T4-best 3.34, Random Relabeling 3.25 (WS 3.10), Boundary 9.92.

## 8. Recommended settings
S1: T1 (main grid) or T4-best (equivalent across seeds). S2: T4-best S2 with forget batch scaled to keep ~70 forget updates per epoch.

## 9. Evaluation results (updated 2026-10-30; full details `records/results/evaluation.md`)
Best configuration used: S1 T1 (T 4, alpha 0.1, gamma 1, 3 ep, msteps 2; forget bs 43 for partitions); S2 T4-best S2 (T 4, alpha 0.1, gamma 0.1, Illusion schedule; forget bs 21 / 43); main-grid models (T1 / T1-S2) for perturbation.
3 seeds, mean. avg_gap = mean |metric - Retrain| (forget, retain, test acc, MIA).

### 9.1 Entanglement Exp 1 - each partition unlearned separately
S1: corrected setup (other two partitions excluded from forget and retain; reference = S1 Retrain on the partition). S2: RUM protocol
(other partitions in the retain set; Retrain per partition). Per-partition accuracies and MIA:
| scenario | model | WS | forget acc: low | forget acc: medium | forget acc: high | S1 only: acc on the 2 excluded partitions, low run | S1 only: excluded acc, medium run | S1 only: excluded acc, high run | retain acc (mean of 3 runs) | test acc (mean of 3 runs) | MIA: low | MIA: medium | MIA: high |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| S1c0 | Retrain (reference) |  | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 100.0 | 85.4 | 53.1 | 49.2 | 49.7 |
| S1c0 | Original (no unlearning) |  | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 94.8 | 58.0 | 57.9 | 57.3 |
| S1c0 | M3a Teacher-Student |  | 0.0 | 0.0 | 0.1 | 0.0 | 0.0 | 0.8 | 100.0 | 84.8 | 52.5 | 54.2 | 52.4 |
| S1c0 | M3a Teacher-Student | yes | 0.4 | 1.8 | 0.4 | 0.1 | 1.0 | 1.2 | 100.0 | 85.0 | 54.4 | 54.4 | 51.4 |
| S2k359p10 | Retrain (reference) |  | 77.6 | 94.6 | 99.6 | - | - | - | 100.0 | 93.5 | 57.3 | 51.3 | 56.7 |
| S2k359p10 | Original (no unlearning) |  | 100.0 | 100.0 | 100.0 | - | - | - | 100.0 | 94.8 | 57.7 | 57.7 | 57.7 |
| S2k359p10 | M3a Teacher-Student |  | 83.3 | 95.1 | 98.4 | - | - | - | 99.5 | 93.2 | 51.2 | 54.7 | 57.2 |
| S2k359p10 | M3a Teacher-Student | yes | 86.6 | 95.8 | 98.4 | - | - | - | 99.5 | 93.2 | 49.6 | 54.9 | 57.5 |
| S2k359p30 | Retrain (reference) |  | 79.0 | 94.7 | 99.6 | - | - | - | 100.0 | 94.8 | 56.7 | 52.1 | 56.7 |
| S2k359p30 | Original (no unlearning) |  | 100.0 | 100.0 | 100.0 | - | - | - | 100.0 | 94.8 | 58.9 | 59.0 | 59.0 |
| S2k359p30 | M3a Teacher-Student |  | 64.1 | 93.0 | 98.4 | - | - | - | 95.9 | 89.6 | 55.7 | 56.9 | 58.0 |
| S2k359p30 | M3a Teacher-Student | yes | 79.9 | 95.3 | 99.4 | - | - | - | 98.7 | 92.0 | 51.4 | 57.9 | 58.9 |
| S2k359p50 | Retrain (reference) |  | 77.8 | 94.3 | 99.4 | - | - | - | 100.0 | 94.5 | 56.3 | 51.8 | 57.2 |
| S2k359p50 | Original (no unlearning) |  | 100.0 | 100.0 | 100.0 | - | - | - | 100.0 | 94.8 | 59.1 | 59.3 | 59.4 |
| S2k359p50 | M3a Teacher-Student |  | 88.9 | 98.6 | 99.7 | - | - | - | 99.5 | 93.0 | 51.1 | 59.2 | 59.8 |
| S2k359p50 | M3a Teacher-Student | yes | 89.8 | 98.8 | 98.9 | - | - | - | 99.5 | 93.0 | 51.4 | 59.5 | 58.3 |

avg_gap to the partition's Retrain:
| scenario | model | WS | low | medium | high | high - low |
|---|---|---|---|---|---|---|
| S1c0 | Original (no unlearning) | | 28.59 | 29.52 | 29.27 | +0.67 |
| S1c0 | (Retrain forget acc on partition, %) | | 0.0 | 0.0 | 0.0 | |
| S1c0 | M3a Teacher-Student |  | 0.56 | 1.39 | 0.84 | +0.28 |
| S1c0 | M3a Teacher-Student | yes | 0.59 | 1.82 | 0.63 | +0.04 |
| S2k359p10 | Original (no unlearning) | | 6.24 | 3.32 | 0.67 | -5.58 |
| S2k359p10 | (Retrain forget acc on partition, %) | | 77.6 | 94.6 | 99.6 | |
| S2k359p10 | M3a Teacher-Student |  | 3.25 | 1.13 | 0.70 | -2.55 |
| S2k359p10 | M3a Teacher-Student | yes | 4.38 | 1.39 | 0.70 | -3.67 |
| S2k359p30 | Original (no unlearning) | | 5.85 | 3.06 | 0.75 | -5.10 |
| S2k359p30 | (Retrain forget acc on partition, %) | | 79.0 | 94.7 | 99.6 | |
| S2k359p30 | M3a Teacher-Student |  | 7.62 | 3.73 | 1.96 | -5.66 |
| S2k359p30 | M3a Teacher-Student | yes | 3.79 | 2.93 | 1.12 | -2.66 |
| S2k359p50 | Original (no unlearning) | | 6.43 | 3.34 | 0.76 | -5.67 |
| S2k359p50 | (Retrain forget acc on partition, %) | | 77.8 | 94.3 | 99.4 | |
| S2k359p50 | M3a Teacher-Student |  | 4.64 | 3.52 | 1.07 | -3.58 |
| S2k359p50 | M3a Teacher-Student | yes | 4.72 | 3.65 | 0.83 | -3.89 |

### 9.2 Entanglement Exp 2 - sequential low -> medium -> high vs one-shot (final model, whole forget set)
| scenario | WS | mode | forget | retain | test | MIA | avg_gap |
|---|---|---|---|---|---|---|---|
| S1c0 |  | one-shot | 0.33 | 99.99 | 84.89 | 52.58 | 0.60 |
| S1c0 |  | sequential | 0.00 | 34.11 | 30.69 | 49.88 | 30.43 |
| S1c0 | yes | one-shot | 2.56 | 99.99 | 85.02 | 52.18 | 1.02 |
| S1c0 | yes | sequential | 0.00 | 99.70 | 84.20 | 49.72 | 0.98 |
| S2k359p10 |  | one-shot | 97.48 | 99.55 | 93.07 | 57.02 | 3.81 |
| S2k359p10 |  | sequential | 90.07 | 98.40 | 91.63 | 53.91 | 2.78 |
| S2k359p10 | yes | one-shot | 97.85 | 99.62 | 93.19 | 57.09 | 3.87 |
| S2k359p10 | yes | sequential | 89.90 | 98.46 | 91.30 | 54.44 | 2.85 |
| S2k359p30 |  | one-shot | 94.60 | 98.85 | 92.17 | 56.13 | 3.34 |
| S2k359p30 |  | sequential | 0.00 | 15.93 | 14.62 | 49.82 | 63.87 |
| S2k359p30 | yes | one-shot | 94.50 | 98.93 | 91.96 | 56.64 | 3.47 |
| S2k359p30 | yes | sequential | 0.00 | 19.09 | 17.48 | 49.40 | 62.38 |
| S2k359p50 |  | one-shot | 94.08 | 99.06 | 91.86 | 57.04 | 3.62 |
| S2k359p50 |  | sequential | 0.00 | 19.62 | 17.16 | 49.91 | 61.64 |
| S2k359p50 | yes | one-shot | 91.66 | 98.64 | 91.01 | 57.42 | 3.43 |
| S2k359p50 | yes | sequential | 0.13 | 22.07 | 18.99 | 50.11 | 60.48 |

### 9.3 Linear probe / embedding separability (S1; figure `records/figures/fig_embed_best_S1c0.png`)
| model | output forget-cls | probe forget-cls | NCC forget-cls | probe other-cls | silhouette class 0 | 10-NN purity class 0 |
|---|---|---|---|---|---|---|
| Original | 95.40 ± 0.00 | 95.50 ± 0.17 | 95.40 ± 0.00 | 94.69 ± 0.05 | 0.127 ± 0.000 | 93.00 ± 0.00 |
| Retrain | 0.00 ± 0.00 | 87.73 ± 0.87 | 41.43 ± 2.97 | 91.19 ± 0.08 | -0.040 ± 0.005 | 52.58 ± 1.06 |
| Teacher-Student (T1) | 0.17 ± 0.29 | 88.50 ± 0.26 | 45.60 ± 1.35 | 93.23 ± 0.26 | -0.049 ± 0.007 | 56.07 ± 0.27 |
| Teacher-Student (T1) + WS | 1.90 ± 2.34 | 88.20 ± 0.62 | 46.50 ± 1.28 | 93.38 ± 0.08 | -0.038 ± 0.008 | 57.77 ± 0.72 |

### 9.4 Perturbed forget samples (main-grid models; Gaussian tau, 20 perturbations per image)
| scenario | model | WS | clean forget acc | perturbed forget acc tau 0.03 | recovery tau 0.03 | perturbed tau 0.1 | recovery tau 0.1 |
|---|---|---|---|---|---|---|---|
| S1c0 | Original |  | 100.0 | 80.6 | 0.0 | 8.7 | 0.0 |
| S1c0 | Retrain |  | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| S1c0 | M3a Teacher-Student |  | 0.3 | 0.0 | 0.0 | 0.0 | 0.0 |
| S1c0 | M3a Teacher-Student | yes | 2.6 | 0.0 | 0.0 | 0.0 | 0.0 |
| S2k359p30 | Original |  | 100.0 | 91.6 | 0.0 | 8.2 | 0.0 |
| S2k359p30 | Retrain |  | 90.9 | 78.4 | 44.5 | 7.1 | 8.6 |
| S2k359p30 | M3a Teacher-Student |  | 99.6 | 87.5 | 71.6 | 8.3 | 5.9 |
| S2k359p30 | M3a Teacher-Student | yes | 98.8 | 83.4 | 63.0 | 7.0 | 12.7 |

### 9.5 Post-unlearning quantization (best configurations; weight-only per-channel, BN in fp32; details `records/results/quant_best.md`)
| scenario | model | WS | config | forget fp32 | forget INT8 | forget INT6 | forget INT4 | change forget (INT4 - fp32) | change retain | change test | change MIA | avg_gap fp32 | avg_gap INT4 | share Q(theta_u)=Q(theta_o), INT4 | mean abs(delta)/step, INT4 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| S1c0 | Original |  |  | 100.00 | 100.00 | 100.00 | 100.00 | +0.00 | +0.00 | -0.33 | -0.60 | 29.05 | 28.82 | 1.000 | 0.000 |
| S1c0 | Retrain |  |  | 0.00 | 0.00 | 0.00 | 0.00 | +0.00 | -0.01 | -0.34 | -0.07 | 0.00 | 0.10 | 0.102 | 15.126 |
| S1c0 | M3a Teacher-Student |  | T1 | 0.33 | 0.32 | 0.30 | 0.27 | -0.06 | -0.03 | -0.30 | +0.35 | 0.60 | 0.75 | 0.290 | 0.082 |
| S1c0 | M3a Teacher-Student | yes | T1 | 2.56 | 2.56 | 2.51 | 2.37 | -0.19 | -0.02 | -0.20 | -0.68 | 1.02 | 0.86 | 0.463 | 0.054 |
| S2k359p10 | Original |  |  | 100.00 | 100.00 | 100.00 | 100.00 | +0.00 | +0.00 | -0.33 | +0.72 | 4.47 | 4.63 | 1.000 | 0.000 |
| S2k359p10 | Retrain |  |  | 91.63 | 91.63 | 91.46 | 90.59 | -1.04 | +0.00 | -0.28 | -0.16 | 0.00 | 0.37 | 0.104 | 17.137 |
| S2k359p10 | M3a Teacher-Student |  | T4-best S2 | 97.53 | 97.53 | 97.48 | 96.10 | -1.43 | -0.27 | -0.55 | -0.52 | 3.82 | 3.54 | 0.282 | 0.073 |
| S2k359p10 | M3a Teacher-Student | yes | T4-best S2 | 97.85 | 97.88 | 97.88 | 96.67 | -1.19 | -0.21 | -0.46 | -0.12 | 3.87 | 3.71 | 0.442 | 0.050 |
| S2k359p30 | Original |  |  | 100.00 | 100.00 | 100.00 | 100.00 | +0.00 | +0.00 | -0.33 | +0.72 | 4.75 | 4.85 | 1.000 | 0.000 |
| S2k359p30 | Retrain |  |  | 90.86 | 90.92 | 90.94 | 90.63 | -0.23 | -0.00 | -0.33 | +0.15 | 0.00 | 0.18 | 0.102 | 14.628 |
| S2k359p30 | M3a Teacher-Student |  | T4-best S2 | 94.60 | 94.59 | 94.46 | 92.53 | -2.07 | -0.59 | -0.81 | -0.36 | 3.33 | 3.07 | 0.269 | 0.137 |
| S2k359p30 | M3a Teacher-Student | yes | T4-best S2 | 94.50 | 94.47 | 94.53 | 92.35 | -2.16 | -0.44 | -0.65 | -0.42 | 3.47 | 3.10 | 0.410 | 0.077 |
| S2k359p50 | Original |  |  | 100.00 | 100.00 | 100.00 | 100.00 | +0.00 | +0.00 | -0.33 | +0.70 | 5.26 | 5.36 | 1.000 | 0.000 |
| S2k359p50 | Retrain |  |  | 88.74 | 88.75 | 88.69 | 87.99 | -0.75 | +0.00 | -0.33 | +0.03 | 0.00 | 0.28 | 0.101 | 14.567 |
| S2k359p50 | M3a Teacher-Student |  | T4-best S2 | 94.10 | 94.04 | 93.79 | 91.60 | -2.50 | -0.43 | -0.88 | -0.44 | 3.62 | 3.21 | 0.276 | 0.099 |
| S2k359p50 | M3a Teacher-Student | yes | T4-best S2 | 91.67 | 91.65 | 91.94 | 88.44 | -3.23 | -0.53 | -1.04 | -0.52 | 3.42 | 3.02 | 0.422 | 0.061 |

### 9.6 Observations
- MIA: T1 52.58% (S1, Retrain 51.0); S2-30 T4-best 56.13% (Retrain 49.95).
- Exp 1, S1 (corrected setup): forget acc 0.0 / 0.0 / 0.1% (WS 0.4 / 1.8 / 0.4%), unused airplanes 0-0.8%, MIA 52.5 / 54.2 / 52.4%,
  avg_gap 0.56 / 1.39 / 0.84 - no entanglement effect. S2: very good on the high and medium partitions (S2-10 0.70 / 1.13), worst on the low partition (S2-30 7.62, WS 3.79).
- Exp 2: sequential helps in S2-10 (2.78 vs 3.81) but collapses in S2-30 / S2-50 and in S1 without WS (retain 16-34%): every step adds
  2 max-epochs relative to a new teacher (6 in total), beyond the stable range (msteps 5 collapsed before); with WS S1 stays stable (0.98).
- Linear probe (T1): output 0.17%, probe 88.5%, NCC 45.6%, purity 56.1%, silhouette -0.049 - the closest to Retrain (87.7 / 41.4 / 52.6 /
  -0.040) of all methods at the representation level.
- Perturbation (main grid): S1 0%; S2-30 87.5% and recovery 71.6% vs Retrain 78.4% / 44.5% - residual knowledge (the main-grid S2 setting
  barely changed the forget samples).
- Quantization (T1 / T4-best S2): S1 no relapse (INT4 forget 0.33 -> 0.27%). S2: INT4 forget -1.2 to -3.2 points with retain -0.2 to
  -0.6 and test -0.5 to -1.0; because the fp32 model forgets less than Retrain, avg_gap improves in every S2 scenario (S2-10 3.82 -> 3.54,
  S2-30 3.33 -> 3.07, S2-50 3.62 -> 3.21). Margin (S2-30 seed 0): 10.6% low-margin, 4.2% of forget images flipped.
