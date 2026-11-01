# Teacher soft labels on S1c0 retain images (classes 1-9): class-0 (airplane) content

| retain class | p(class0) T=1 | p(class0) T=4 | class0 is 2nd-ranked |
|---|---|---|---|
| automobile | 0.00010 | 0.0522 | 2.3% |
| bird | 0.00016 | 0.0546 | 24.9% |
| cat | 0.00008 | 0.0447 | 2.3% |
| deer | 0.00010 | 0.0495 | 3.6% |
| dog | 0.00005 | 0.0404 | 0.3% |
| frog | 0.00007 | 0.0466 | 1.6% |
| horse | 0.00011 | 0.0483 | 2.1% |
| ship | 0.00022 | 0.0584 | 44.7% |
| truck | 0.00011 | 0.0518 | 4.8% |

- mean p(class0) over all retain: T=1 0.00011, T=4 0.0496
- share of non-true-class probability mass on class 0: T=1 0.111, T=4 0.111 (uniform would be 1/9 = 0.111)
- class 0 is the 2nd-ranked class for 9.6% of retain images
