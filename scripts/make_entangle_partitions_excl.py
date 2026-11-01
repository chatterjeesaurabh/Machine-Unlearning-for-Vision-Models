"""Entanglement Exp 1 for single-class forgetting, corrected setup (2026-10-30): <base>-Ex<part>__s<seed>.npz
  forget = one entanglement partition of the class (same partitions as <base>-E<part>, scripts/make_entangle_partitions.py)
  retain = the base scenario's retain set (pool without ANY sample of the forgotten class)
  The other two partitions are excluded (neither forgotten nor retained); their accuracy is reported as acc_excluded.
  Reference = the base scenario's Retrain model evaluated on the partition (run '<base>-Ex<part>__retrain__s<seed>').
Usage: python scripts/make_entangle_partitions_excl.py
"""
import os

import numpy as np

from mu import SPLIT_DIR

BASES = ["S1c0"]
PARTS = ["low", "medium", "high"]

for base in BASES:
    for seed in (0, 1, 2):
        retain = np.load(os.path.join(SPLIT_DIR, f"{base}__s{seed}.npz"))["retain_idx"]
        for p in PARTS:
            src = np.load(os.path.join(SPLIT_DIR, f"{base}-E{p}__s{seed}.npz"))
            out = os.path.join(SPLIT_DIR, f"{base}-Ex{p}__s{seed}.npz")
            assert not np.isin(src["forget_idx"], retain).any()
            np.savez(out, forget_idx=src["forget_idx"], retain_idx=retain, knn_dist=src["knn_dist"])
            print(out, len(src["forget_idx"]), len(retain))
