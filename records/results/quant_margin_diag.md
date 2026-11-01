# INT4 margin diagnostic - S2k359p30 seed 0
p_y = fp32 softmax probability of the true label. 'low margin' = correctly classified with p_y < 0.9. 'flipped' = correct in fp32, wrong at INT4. Same-class test = test images of the forget classes [3, 5, 9].

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


**Interpretation (added 2026-11-01):** unlearning leaves many forget-class samples correctly classified but with low confidence
(p_y < 0.9: Retrain 5.3% of correct forget images, Teacher-Student 10.6%, GA + Descent 12.6%, Gradient Ascent 16.5%, Random Relabeling
18.6%, Boundary Relabeling 43.3%). INT4 rounding acts like small weight noise and flips exactly these low-margin samples (median p_y of
flipped images 0.54-0.70 vs 0.93-0.999 for images that stay correct). Same-class test images are flipped at a similar rate, so the INT4
forget-accuracy drop is degradation of the forget classes, not selective forgetting. Random Relabeling has many low-margin samples but few
flips (2.3%), so margin is not the only factor. Figure: `records/figures/fig_quant_margin_diag.png`.
