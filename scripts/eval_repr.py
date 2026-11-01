"""E4: linear probe + NCC on frozen embeddings for runs with checkpoints. Writes records/runs/<run_id>/repr_metrics.json.

Usage: python scripts/eval_repr.py --runs S1c0__GA__s0 S1c0__retrain__s0 ...   (or --pattern 'S1c0__*__s0')
For the Original model on a scenario use run id '<scenario>__original__s<seed>' (checkpoint orig__original__s0.pt).
"""
import argparse
import fnmatch
import glob
import json
import os

import numpy as np
import torch

from mu import CKPT_DIR, RECORDS_DIR
from mu.data import CIFAR10GPU
from mu.eval.embeddings import extract
from mu.eval.probe import probe_metrics
from mu.models import resnet18
from mu.splits import load_split
from mu.utils import load_ckpt, setup_device, run_with_oom_retry


def ckpt_for(run_id):
    if "__original__" in run_id:
        return os.path.join(CKPT_DIR, "orig__original__s0.pt")
    return os.path.join(CKPT_DIR, f"{run_id}.pt")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", nargs="*", default=[])
    p.add_argument("--pattern", default=None)
    p.add_argument("--skip_done", type=int, default=1)
    p.add_argument("--gpu", type=int, default=1)
    p.add_argument("--mem_frac", type=float, default=0.45)
    args = p.parse_args()
    runs = list(args.runs)
    if args.pattern:
        runs += [os.path.basename(d) for d in sorted(glob.glob(os.path.join(RECORDS_DIR, "runs", "*"))) if fnmatch.fnmatch(os.path.basename(d), args.pattern)]
    device = setup_device(args.gpu, args.mem_frac)
    data = CIFAR10GPU(device)
    for rid in runs:
        out_path = os.path.join(RECORDS_DIR, "runs", rid, "repr_metrics.json")
        if args.skip_done and os.path.exists(out_path):
            continue
        ck = ckpt_for(rid)
        if not os.path.exists(ck):
            print(f"[skip] no checkpoint for {rid}")
            continue
        cfg = json.load(open(os.path.join(RECORDS_DIR, "runs", rid, "config.json")))
        split = load_split(cfg["scenario"], cfg["seed"])
        sc = split["scenario"]
        forget_classes = [sc["class"]] if sc["type"] == "class" else sc["classes"]
        model = resnet18().to(device).to(memory_format=torch.channels_last)
        load_ckpt(model, ck, device)
        f_tr, y_tr = extract(model, data, "train", split["pool_idx"])
        f_te, y_te = extract(model, data, "test", np.arange(10000))
        m = probe_metrics(f_tr, y_tr, f_te, y_te, forget_classes, seed=cfg["seed"])
        m.update(protocol="probe: Adam lr1e-3 bs1024 100ep on standardized 512-d avgpool features of all 45k train-pool (retain+forget); NCC: train-pool class means")
        json.dump(m, open(out_path, "w"), indent=1)
        print(rid, {k: round(v, 2) for k, v in m.items() if isinstance(v, float)})


if __name__ == "__main__":
    run_with_oom_retry(main)
