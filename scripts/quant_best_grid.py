"""Emit the batch that runs post-unlearning quantization (scripts/eval_quant.py) on the BEST configuration of every method
(src/mu/best_configs.py), all scenarios, 3 seeds, with / without Weight Saliency, plus Original and Retrain.
Runs that already have quant_metrics.json (main-grid best configs, Original, Retrain) are skipped by eval_quant.py itself; runs
whose checkpoint is missing are reported here and skipped.
Usage: python scripts/quant_best_grid.py > outputs/quant_best_batch.sh; nohup bash scripts/gpu_batch.sh outputs/quant_best_batch.sh > outputs/logs/quant_best.log 2>&1 &
"""
import argparse
import os

from mu import CKPT_DIR, RECORDS_DIR, ROOT
from mu.best_configs import best_runs


def ckpt(rid):
    return os.path.join(CKPT_DIR, "orig__original__s0.pt") if "__original__" in rid else os.path.join(CKPT_DIR, f"{rid}.pt")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bits", nargs="+", type=int, default=[8, 6, 4])
    args = p.parse_args()
    todo, done, missing = [], [], []
    for sc, m, mask, seed, tag, rid in best_runs():
        if os.path.exists(os.path.join(RECORDS_DIR, "runs", rid, "quant_metrics.json")):
            done.append(rid)
        elif not os.path.exists(ckpt(rid)):
            missing.append(rid)
        else:
            todo.append(rid)
    print(f"cd {ROOT}; source env.sh")
    print(f"echo '[quant_best] {len(done)} done, {len(todo)} to run, {len(missing)} missing checkpoint'")
    for rid in missing:
        print(f"echo '[skip] no checkpoint: {rid}'")
    if todo:
        print(f"python scripts/eval_quant.py --bits {' '.join(map(str, args.bits))} --runs {' '.join(todo)} 2>&1 | grep -vE '^$'")
    print("python scripts/analyze_quant_best.py; python scripts/build_index.py; echo QUANT_BEST_DONE")


if __name__ == "__main__":
    main()
