# E4: embedding-space separability of the forgotten class (S1 class 0), best configuration of every method, 3 seeds
output = model's own prediction; probe = fresh linear head on frozen features trained on all 45k train-pool images (retain+forget);
NCC = nearest class mean; silhouette = class 0 vs rest on 3000 test features (higher = more separable); 10-NN purity = % of the 10 nearest
test neighbours of class-0 test images that are class 0. All accuracies on the test set (%).

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
