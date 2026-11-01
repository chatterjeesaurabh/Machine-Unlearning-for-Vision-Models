"""Emit the main-grid batch: scenarios x seeds x (Original eval + 5 methods x {no mask, saliency 0.5}) with hyperparameters
from configs/final_hparams.yaml. Skips completed runs. Checkpoints are saved (needed for probe / perturbation / quantization).
Usage: python scripts/main_grid.py [--seeds 0 1 2] > outputs/main_batch.sh; bash scripts/gpu_batch.sh outputs/main_batch.sh
"""
import argparse
import os

import yaml

from mu import RECORDS_DIR, ROOT
from mu.splits import parse_scenario

SCENARIOS = ["S1c0", "S2k359p10", "S2k359p30", "S2k359p50"]
METHODS = ["GA", "GAGD", "TS", "RLrand", "RLbnd"]


def done(rid):
    return os.path.exists(os.path.join(RECORDS_DIR, "runs", rid, "metrics.json"))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--scenarios", nargs="+", default=SCENARIOS)
    args = p.parse_args()
    final = yaml.safe_load(open(os.path.join(ROOT, "configs", "final_hparams.yaml")))
    print("cd /anchor/anchor7/sachatt/cs6886/project; source env.sh")
    for seed in args.seeds:
        for sc in args.scenarios:
            st = parse_scenario(sc)["type"]
            if not done(f"{sc}__retrain__s{seed}"):  # never wait inside a lock-holding batch (would deadlock the ref queue)
                print(f"echo '[skip] {sc} s{seed}: retrain reference missing - regenerate later'")
                continue
            if not done(f"{sc}__original__s{seed}"):
                print(f"python scripts/unlearn.py --scenario {sc} --seed {seed} --method none 2>&1 | grep -E 'FINAL|Traceback|Error'")
            for m in METHODS:
                for mask in ("none", "saliency"):
                    ov = {k: v for k, v in final[m][mask][st].items() if not k.startswith("_")}
                    rid = f"{sc}__{m}" + ("__sal0.5" if mask == "saliency" else "") + f"__s{seed}"
                    if done(rid):
                        continue
                    hp = " ".join(f"{k}={v}" for k, v in ov.items())
                    print(f"echo '[run] {rid}'; python scripts/unlearn.py --scenario {sc} --seed {seed} --method {m} --mask {mask} "
                          f"--mask_ratio 0.5 {('--hp ' + hp) if hp else ''} 2>&1 | grep -E 'FINAL|Traceback|Error'")
    print("python scripts/aggregate.py; python scripts/build_index.py; echo MAIN_DONE")


if __name__ == "__main__":
    main()
