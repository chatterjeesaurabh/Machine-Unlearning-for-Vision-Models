# Stage 5: method improvements (seed 0)

sel_gap / avg_gap: mean |metric - Retrain| (lower = closer to Retrain). Baseline = main-grid run (seed 0, no mask). For S2 the Retrain forget acc is ~89-92%, so very low forget acc is over-unlearning from the privacy view.

## GA - S1c0

| variant | forget | retain | val | test | test other-cls | MIA | sel_gap | avg_gap | time s | early stop | diverged |
|---|---|---|---|---|---|---|---|---|---|---|---|
| original | 100.00 | 100.00 | 95.44 | 94.82 | 94.76 | 57.75 | 28.51 | 28.88 | 0.00 | False | False |
| retrain | 0.00 | 100.00 | 86.22 | 85.34 | 94.82 | 51.70 | - | - | - | - | - |
| main-grid | 5.27 | 93.95 | 79.36 | 78.74 | 87.01 | 51.70 | 5.17 | 4.48 | 5.84 | False | False |
| imp_epochs10_lr0.001_stopit2_stop1 | 0.29 | 92.23 | 77.42 | 77.23 | 85.76 | 52.30 | 4.21 | 4.19 | 4.25 | True | False |
| imp_epochs10_lr0.001_bnfrozen_grad_clip1_stopit2_stop1 | 0.51 | 92.45 | 77.58 | 77.19 | 85.73 | 50.10 | 4.72 | 4.45 | 5.69 | True | False |
| imp_epochs10_lr0.0003_bnfrozen_grad_clip1_stopit2_stop1 | 0.64 | 89.75 | 75.44 | 75.18 | 83.47 | 52.35 | 5.67 | 5.43 | 8.19 | True | False |
| imp_epochs10_lr0.0001_stopit2_stop1 | 2.11 | 86.63 | 73.74 | 73.07 | 80.96 | 49.35 | 8.21 | 7.52 | 7.98 | False | False |
| imp_epochs10_lr0.0003_stopit2_stop1 | 1.16 | 81.08 | 69.82 | 68.92 | 76.48 | 51.65 | 10.17 | 9.14 | 6.44 | True | False |
| imp_bn_recalib1 | 92.00 | 98.22 | 91.38 | 90.30 | 91.63 | 57.55 | 25.74 | 26.15 | 8.44 | False | False |

## RLbnd - S1c0

| variant | forget | retain | val | test | test other-cls | MIA | sel_gap | avg_gap | time s | early stop | diverged |
|---|---|---|---|---|---|---|---|---|---|---|---|
| original | 100.00 | 100.00 | 95.44 | 94.82 | 94.76 | 57.75 | 28.51 | 28.88 | 0.00 | False | False |
| retrain | 0.00 | 100.00 | 86.22 | 85.34 | 94.82 | 51.70 | - | - | - | - | - |
| main-grid | 23.60 | 99.46 | 86.52 | 85.00 | 92.26 | 54.10 | 6.64 | 6.72 | 8.54 | False | False |
| imp_nb_searchlogit2_lr0.001_use_retain1 | 2.07 | 100.00 | 86.04 | 85.05 | 94.34 | 54.50 | 0.91 | 1.29 | 53.37 | False | False |
| imp_nb_searchpgd_lr0.001_use_retain1 | 1.38 | 100.00 | 85.90 | 85.07 | 94.43 | 55.95 | 1.40 | 1.47 | 78.31 | False | False |
| imp_nb_searchpgd_lr0.0001_use_retain0 | 1.02 | 99.97 | 85.78 | 84.92 | 94.21 | 57.20 | 1.52 | 1.74 | 36.89 | False | False |
| imp_nb_searchlogit2_lr0.0001_use_retain0 | 2.13 | 100.00 | 86.04 | 84.97 | 94.23 | 58.45 | 1.83 | 2.31 | 10.79 | False | False |
| imp_nb_searchfgsm_eps0.03_lr0.0001 | 1.22 | 99.86 | 84.84 | 84.39 | 93.61 | 56.55 | 2.04 | 1.79 | 15.41 | False | False |
| imp_nb_searchfgsm_lr0.001_use_retain1 | 13.16 | 100.00 | 86.88 | 85.87 | 94.48 | 55.85 | 4.13 | 4.46 | 55.79 | False | False |
| imp_nb_searchpgd_lr1e-05_use_retain0 | 19.47 | 99.72 | 86.42 | 85.34 | 93.03 | 55.45 | 5.61 | 5.87 | 40.78 | False | False |
| imp_nb_searchlogit2_lr1e-05_use_retain0 | 21.29 | 99.80 | 86.82 | 85.62 | 93.12 | 55.40 | 6.12 | 6.37 | 11.62 | False | False |
| imp_nb_searchfgsm_lr1e-05_use_retain0 | 23.67 | 99.45 | 86.50 | 85.01 | 92.27 | 54.00 | 6.65 | 6.71 | 14.33 | False | False |
| imp_nb_searchfgsm_lr0.0001_use_retain0 | 2.60 | 88.95 | 74.04 | 73.53 | 81.51 | 55.40 | 7.33 | 7.29 | 14.68 | False | False |
| imp_nb_searchfgsm_eps0.3_lr0.0001 | 1.60 | 81.95 | 67.22 | 66.71 | 73.89 | 55.50 | 9.84 | 10.52 | 14.36 | False | False |
| imp_nb_searchlogit2_lr0.0001_use_retain1 | 38.84 | 99.96 | 88.36 | 87.42 | 93.64 | 55.35 | 11.01 | 11.15 | 50.09 | False | False |
| imp_nb_searchpgd_lr0.0001_use_retain1 | 40.78 | 99.96 | 88.28 | 87.44 | 93.68 | 55.65 | 11.62 | 11.72 | 78.03 | False | False |
| imp_nb_searchfgsm_lr0.0001_use_retain1 | 78.40 | 99.98 | 91.96 | 91.06 | 93.43 | 53.00 | 21.69 | 21.36 | 50.85 | False | False |
| imp_bn_recalib1 | 99.91 | 98.71 | 92.50 | 91.52 | 90.77 | 52.75 | 26.95 | 27.11 | 15.83 | False | False |
| imp_nb_searchfgsm_lr1e-05_use_retain1 | 100.00 | 99.96 | 94.12 | 93.21 | 92.62 | 53.10 | 27.34 | 27.33 | 52.48 | False | False |
| imp_nb_searchpgd_lr1e-05_use_retain1 | 100.00 | 99.99 | 94.30 | 93.65 | 93.30 | 54.65 | 27.45 | 27.82 | 78.70 | False | False |
| imp_nb_searchlogit2_lr1e-05_use_retain1 | 100.00 | 99.94 | 94.22 | 93.10 | 92.62 | 54.35 | 27.51 | 27.62 | 51.40 | False | False |

## TS - S1c0

| variant | forget | retain | val | test | test other-cls | MIA | sel_gap | avg_gap | time s | early stop | diverged |
|---|---|---|---|---|---|---|---|---|---|---|---|
| original | 100.00 | 100.00 | 95.44 | 94.82 | 94.76 | 57.75 | 28.51 | 28.88 | 0.00 | False | False |
| retrain | 0.00 | 100.00 | 86.22 | 85.34 | 94.82 | 51.70 | - | - | - | - | - |
| main-grid | 0.02 | 99.98 | 85.68 | 84.96 | 94.40 | 52.75 | 0.89 | 0.37 | 20.15 | False | False |
| imp_epochs10_msteps2_alpha0.001_forgetbs128_retainbs128_dec8 | 0.04 | 100.00 | 86.00 | 85.11 | 94.57 | 54.70 | 0.89 | 0.82 | 65.02 | False | False |
| imp_epochs10_msteps2_alpha0.01_forgetbs128_retainbs128_dec8 | 0.11 | 100.00 | 85.90 | 85.18 | 94.64 | 52.95 | 0.93 | 0.38 | 64.99 | False | False |
| imp_epochs10_msteps2_alpha0.0001_forgetbs128_retainbs128_dec8 | 0.02 | 100.00 | 85.94 | 85.05 | 94.50 | 55.25 | 0.95 | 0.97 | 67.65 | False | False |
| imp_epochs10_msteps2_alpha0_forgetbs128_retainbs128_dec8 | 0.04 | 100.00 | 85.94 | 85.01 | 94.46 | 55.50 | 1.03 | 1.04 | 93.45 | False | False |
| imp_alpha0.1_kd_T1_forgetbs128 | 17.91 | 99.98 | 87.34 | 86.59 | 94.09 | 54.35 | 5.21 | 5.46 | 24.95 | False | False |
| imp_alpha1_kd_T1_forgetbs128 | 22.27 | 100.00 | 87.76 | 87.33 | 94.46 | 54.20 | 6.70 | 6.69 | 25.70 | False | False |
| imp_epochs10_msteps5_alpha0_forgetbs128_retainbs128_dec8 | 0.00 | 76.62 | 67.82 | 67.83 | 75.37 | 51.35 | 11.27 | 10.31 | 67.85 | False | False |
| imp_epochs10_msteps5_alpha0.001_forgetbs128_retainbs128_dec8 | 0.00 | 76.59 | 67.18 | 67.89 | 75.43 | 49.55 | 11.59 | 10.75 | 67.69 | False | False |
| imp_epochs10_msteps5_alpha0.01_forgetbs128_retainbs128_dec8 | 0.00 | 50.63 | 45.02 | 45.64 | 50.71 | 49.85 | 22.79 | 22.73 | 64.19 | False | False |
| imp_epochs10_msteps5_alpha0.0001_forgetbs128_retainbs128_dec8 | 0.00 | 46.06 | 40.08 | 41.76 | 46.40 | 52.20 | 25.25 | 24.51 | 68.38 | False | False |

## GA - S2k359p30

| variant | forget | retain | val | test | test other-cls | MIA | sel_gap | avg_gap | time s | early stop | diverged |
|---|---|---|---|---|---|---|---|---|---|---|---|
| original | 100.00 | 100.00 | 95.44 | 94.82 | 96.17 | 59.38 | 4.51 | 4.49 | 0.00 | False | False |
| retrain | 91.41 | 100.00 | 94.74 | 94.65 | 96.20 | 50.20 | - | - | - | - | - |
| main-grid | 87.73 | 97.14 | 89.08 | 88.29 | 96.84 | 59.50 | 5.35 | 5.55 | 3.78 | True | False |
| imp_epochs10_lr0.0001_stopit2_stopretrain | 91.58 | 98.05 | 90.04 | 89.23 | 96.93 | 59.73 | 4.01 | 4.27 | 4.20 | True | False |
| imp_epochs10_lr0.0003_bnfrozen_grad_clip1_stopit2_stopretrain | 90.99 | 97.95 | 89.68 | 89.10 | 96.29 | 59.07 | 4.16 | 4.22 | 6.45 | True | False |
| imp_epochs10_lr0.001_bnfrozen_grad_clip1_stopit2_stopretrain | 88.44 | 97.22 | 88.98 | 88.16 | 96.01 | 59.13 | 5.06 | 5.29 | 5.85 | True | False |
| imp_epochs10_lr0.0003_stopit2_stopretrain | 85.04 | 96.53 | 88.52 | 87.67 | 96.83 | 59.18 | 6.14 | 6.45 | 4.75 | True | False |
| imp_epochs10_lr0.001_stopit2_stopretrain | 59.28 | 90.38 | 82.06 | 81.91 | 96.50 | 57.27 | 15.33 | 15.39 | 4.96 | True | False |

## GAGD - S2k359p30

| variant | forget | retain | val | test | test other-cls | MIA | sel_gap | avg_gap | time s | early stop | diverged |
|---|---|---|---|---|---|---|---|---|---|---|---|
| original | 100.00 | 100.00 | 95.44 | 94.82 | 96.17 | 59.38 | 4.51 | 4.49 | 0.00 | False | False |
| retrain | 91.41 | 100.00 | 94.74 | 94.65 | 96.20 | 50.20 | - | - | - | - | - |
| main-grid | 92.02 | 98.56 | 91.06 | 90.16 | 96.97 | 58.67 | 3.47 | 3.75 | 24.33 | False | False |
| imp_beta0.998_lr0.01 | 85.73 | 97.32 | 89.42 | 88.73 | 96.86 | 57.97 | 5.23 | 5.51 | 44.24 | False | False |
| imp_beta0.998_lr0.02 | 84.67 | 96.64 | 88.80 | 87.93 | 95.40 | 56.73 | 5.69 | 5.84 | 44.26 | False | False |
| imp_beta0.995_lr0.01 | 44.49 | 88.28 | 79.68 | 78.73 | 96.71 | 55.75 | 19.62 | 20.03 | 40.87 | False | False |
| imp_beta0.995_lr0.02 | 25.28 | 83.43 | 74.06 | 73.68 | 95.19 | 53.47 | 26.67 | 26.73 | 44.68 | False | False |

## RLbnd - S2k359p30

| variant | forget | retain | val | test | test other-cls | MIA | sel_gap | avg_gap | time s | early stop | diverged |
|---|---|---|---|---|---|---|---|---|---|---|---|
| original | 100.00 | 100.00 | 95.44 | 94.82 | 96.17 | 59.38 | 4.51 | 4.49 | 0.00 | False | False |
| retrain | 91.41 | 100.00 | 94.74 | 94.65 | 96.20 | 50.20 | - | - | - | - | - |
| main-grid | 76.35 | 94.32 | 86.14 | 85.38 | 96.93 | 59.60 | 9.67 | 9.85 | 8.79 | False | False |
| imp_nb_searchfgsm_lr1e-05_use_retain1 | 100.00 | 100.00 | 95.18 | 94.37 | 94.99 | 58.53 | 4.23 | 4.30 | 54.83 | False | False |
| imp_nb_searchpgd_lr1e-05_use_retain1 | 100.00 | 100.00 | 95.28 | 94.51 | 95.56 | 59.60 | 4.62 | 4.53 | 82.42 | False | False |
| imp_nb_searchpgd_lr0.001_use_retain1 | 99.98 | 100.00 | 94.46 | 93.99 | 95.47 | 60.00 | 4.72 | 4.76 | 80.78 | False | False |
| imp_nb_searchpgd_lr0.0001_use_retain1 | 100.00 | 100.00 | 94.50 | 94.15 | 95.71 | 60.95 | 4.88 | 4.96 | 80.38 | False | False |
| imp_nb_searchfgsm_lr0.0001_use_retain1 | 100.00 | 99.99 | 94.62 | 93.84 | 95.11 | 61.00 | 4.89 | 5.05 | 55.49 | False | False |
| imp_nb_searchfgsm_lr0.001_use_retain1 | 99.95 | 100.00 | 95.22 | 94.29 | 95.46 | 61.00 | 5.03 | 4.93 | 54.60 | False | False |

## TS - S2k359p30

| variant | forget | retain | val | test | test other-cls | MIA | sel_gap | avg_gap | time s | early stop | diverged |
|---|---|---|---|---|---|---|---|---|---|---|---|
| original | 100.00 | 100.00 | 95.44 | 94.82 | 96.17 | 59.38 | 4.51 | 4.49 | 0.00 | False | False |
| retrain | 91.41 | 100.00 | 94.74 | 94.65 | 96.20 | 50.20 | - | - | - | - | - |
| main-grid | 99.63 | 99.97 | 94.42 | 93.73 | 95.57 | 57.18 | 3.92 | 4.04 | 58.50 | False | False |
| imp_epochs10_msteps5_alpha0.001_forgetbs128_retainbs64_dec8 | 99.11 | 99.94 | 94.06 | 93.72 | 95.57 | 56.85 | 3.70 | 3.84 | 101.68 | False | False |
| imp_epochs10_msteps5_alpha0_forgetbs128_retainbs64_dec8 | 99.19 | 99.92 | 94.28 | 93.52 | 95.46 | 56.85 | 3.73 | 3.91 | 94.79 | False | False |
| imp_epochs10_msteps5_alpha0.01_forgetbs128_retainbs64_dec8 | 99.51 | 99.97 | 94.08 | 93.81 | 95.51 | 57.62 | 3.93 | 4.10 | 93.60 | False | False |
| imp_epochs10_msteps2_alpha0.01_forgetbs128_retainbs64_dec8 | 99.98 | 100.00 | 95.20 | 94.49 | 95.73 | 59.02 | 4.35 | 4.39 | 94.97 | False | False |
| imp_epochs10_msteps2_alpha0_forgetbs128_retainbs64_dec8 | 99.95 | 100.00 | 95.08 | 94.47 | 95.71 | 59.07 | 4.36 | 4.40 | 93.59 | False | False |
| imp_epochs10_msteps2_alpha0.001_forgetbs128_retainbs64_dec8 | 99.95 | 100.00 | 95.20 | 94.49 | 95.71 | 59.13 | 4.40 | 4.41 | 93.81 | False | False |

## RLrand - S2k359p50

| variant | forget | retain | val | test | test other-cls | MIA | sel_gap | avg_gap | time s | early stop | diverged |
|---|---|---|---|---|---|---|---|---|---|---|---|
| original | 100.00 | 100.00 | 95.44 | 94.82 | 96.17 | 59.47 | 5.20 | 5.22 | 0.00 | False | False |
| retrain | 89.10 | 100.00 | 94.10 | 93.61 | 96.09 | 50.72 | - | - | - | - | - |
| main-grid | 81.97 | 94.58 | 88.94 | 88.12 | 92.69 | 52.20 | 4.76 | 4.88 | 26.33 | False | False |
| imp_lr0.01 | 98.83 | 99.85 | 94.04 | 93.39 | 95.21 | 57.28 | 4.23 | 4.17 | 43.33 | False | False |
| imp_lr0.005 | 99.97 | 100.00 | 95.02 | 94.30 | 95.99 | 61.12 | 5.53 | 5.49 | 49.22 | False | False |

## TS - S2k359p50

| variant | forget | retain | val | test | test other-cls | MIA | sel_gap | avg_gap | time s | early stop | diverged |
|---|---|---|---|---|---|---|---|---|---|---|---|
| original | 100.00 | 100.00 | 95.44 | 94.82 | 96.17 | 59.47 | 5.20 | 5.22 | 0.00 | False | False |
| retrain | 89.10 | 100.00 | 94.10 | 93.61 | 96.09 | 50.72 | - | - | - | - | - |
| main-grid | 0.00 | 26.31 | 23.48 | 23.34 | 33.34 | 49.65 | 58.43 | 58.53 | 52.32 | False | False |
| imp_epochs10_msteps2_alpha0.001_forgetbs128_retainbs64_dec8 | 99.91 | 100.00 | 94.88 | 94.20 | 96.14 | 58.57 | 4.74 | 4.81 | 91.33 | False | False |
| imp_epochs10_msteps2_alpha0.001_forgetbs256_retainbs64_dec8 | 99.97 | 100.00 | 95.02 | 94.39 | 96.13 | 59.27 | 4.95 | 5.05 | 86.14 | False | False |

