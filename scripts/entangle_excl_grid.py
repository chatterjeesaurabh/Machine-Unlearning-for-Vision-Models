"""Entanglement Exp 1, single-class forgetting, corrected setup (tag 'entpart2', scenarios S1c0-Ex<part>, see
scripts/make_entangle_partitions_excl.py): forget = one partition, retain = S1 retain set (no airplane at all).
Phase A: Retrain reference (base S1c0 Retrain checkpoint evaluated on the partition -> run '<sc>__retrain__s<seed>') and the
         Original evaluated on the partition.  Phase B: 5 methods x (no WS, WS), best S1 configurations (same as scripts/entangle_grid.py).
Usage: python scripts/entangle_excl_grid.py A|B > outputs/excl_jobs_<phase>.txt   (one command per line; run with scripts/run_parallel.sh)
"""
import os
import sys

from mu import CKPT_DIR
from entangle_grid import BEST, PARTS, done, hp_for

phase = sys.argv[1]
L = []
for seed in (0, 1, 2):
    for p in PARTS:
        sc = f"S1c0-Ex{p}"
        if phase == "A":
            if not done(f"{sc}__retrain__s{seed}"):
                L.append(f"python scripts/unlearn.py --scenario {sc} --seed {seed} --method none --eval_label retrain "
                         f"--orig_ckpt {os.path.join(CKPT_DIR, f'S1c0__retrain__s{seed}.pt')}")
            if not done(f"{sc}__original__entpart2__s{seed}"):
                L.append(f"python scripts/unlearn.py --scenario {sc} --seed {seed} --method none --tag entpart2")
        else:
            for m in BEST:
                for mask in ("none", "saliency"):
                    rid = f"{sc}__{m}" + ("__sal0.5" if mask == "saliency" else "") + f"__entpart2__s{seed}"
                    if not done(rid):
                        L.append(f"python scripts/unlearn.py --scenario {sc} --seed {seed} --method {m} --mask {mask} --mask_ratio 0.5 "
                                 f"--hp {hp_for(m, 'S1c0')} --tag entpart2 --save_ckpt 0")
print("\n".join(L))
