"""Ablation batches (tagged so they never mix with main-grid runs):
  A1 (tag ablA1): Weight Saliency update ratio {0.1, 0.3, 0.7, 0.9} (0.5 = main grid), all methods, S1c0 + S2k359p50, seeds 0, 1.
  A2 (tag ablA2): random mask with the same density (0.5), all methods, S1c0 + S2k359p50, seeds 0, 1, 2.
Hyperparameters = the saliency variant's final hyperparameters (configs/final_hparams.yaml). Checkpoints not saved.
Usage: python scripts/ablation_grid.py > outputs/abl_batch.sh; bash scripts/gpu_batch.sh outputs/abl_batch.sh
"""
import os

import yaml

from mu import RECORDS_DIR, ROOT
from mu.splits import parse_scenario

METHODS = ["GA", "GAGD", "TS", "RLrand", "RLbnd"]
SCEN = ["S1c0", "S2k359p50"]


def main():
    final = yaml.safe_load(open(os.path.join(ROOT, "configs", "final_hparams.yaml")))
    print("cd /anchor/anchor7/sachatt/cs6886/project; source env.sh")
    jobs = [("ablA1", "saliency", r, s) for s in (0, 1) for r in (0.1, 0.3, 0.7, 0.9)] + [("ablA2", "random", 0.5, s) for s in (0, 1, 2)]
    for tag, mask, ratio, seed in jobs:
        for sc in SCEN:
            st = parse_scenario(sc)["type"]
            for m in METHODS:
                ov = {k: v for k, v in final[m]["saliency"][st].items() if not k.startswith("_")}
                mt = f"__sal{ratio:g}" if mask == "saliency" else f"__randmask{ratio:g}"
                rid = f"{sc}__{m}{mt}__{tag}__s{seed}"
                if os.path.exists(os.path.join(RECORDS_DIR, "runs", rid, "metrics.json")):
                    continue
                hp = " ".join(f"{k}={v}" for k, v in ov.items())
                print(f"echo '[run] {rid}'; python scripts/unlearn.py --scenario {sc} --seed {seed} --method {m} --mask {mask} --mask_ratio {ratio} "
                      f"{('--hp ' + hp) if hp else ''} --tag {tag} --save_ckpt 0 2>&1 | grep -E 'FINAL|Traceback|Error'")
    print("python scripts/aggregate.py; python scripts/build_index.py; echo ABL_DONE")


if __name__ == "__main__":
    main()
