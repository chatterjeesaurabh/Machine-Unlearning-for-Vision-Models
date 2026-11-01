"""M5 (evaluation): do unlearned models still recognise PERTURBED forget samples?

Perturbation = Gaussian noise as in the Unseen Threat paper (torchattacks.GN): x' = clip(x + tau * N(0, I), 0, 1), pixel space [0,1].
For each forget sample, c perturbations per tau. Identical noise for every model (deterministic per scenario/seed/tau/batch).
Metrics per tau: perturbed forget accuracy (mean over samples x perturbations), clean forget acc,
  recovery_any  = fraction of forget samples misclassified when clean but correctly classified under >= 1 perturbation,
  recovery_major = ... correctly classified under > 50% of perturbations.
Writes records/runs/<run_id>/perturb_metrics.json. (No residual-knowledge ratio, per project scope.)
Usage: python scripts/eval_perturb.py --pattern 'S1c0__*__s0' [--c 20] [--taus 0.01 0.02 0.03 0.05 0.1]
"""
import argparse
import fnmatch
import glob
import json
import os

import numpy as np
import torch

from mu import CKPT_DIR, RECORDS_DIR
from mu.data import CIFAR10GPU, GPULoader
from mu.models import resnet18
from mu.splits import load_split
from mu.utils import load_ckpt, run_with_oom_retry, setup_device


def ckpt_for(run_id):
    return os.path.join(CKPT_DIR, "orig__original__s0.pt") if "__original__" in run_id else os.path.join(CKPT_DIR, f"{run_id}.pt")


@torch.no_grad()
def perturbed_eval(model, data, forget_idx, taus, c, seed, batch_size=500):
    model.eval()
    res = {}
    for ti, tau in enumerate(taus):
        n_correct_clean, n_hits = [], []
        for bi, (x, y, _) in enumerate(GPULoader(data, "train", forget_idx, batch_size)):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                clean = model(x).argmax(1) == y
            g = torch.Generator(device=x.device).manual_seed(seed * 1_000_003 + ti * 10_007 + bi)
            hits = torch.zeros(len(y), device=x.device)
            for _ in range(c):
                xp = (x + tau * torch.randn(x.shape, device=x.device, generator=g)).clamp_(0, 1)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    hits += (model(xp.contiguous(memory_format=torch.channels_last)).argmax(1) == y).float()
            n_correct_clean.append(clean)
            n_hits.append(hits)
        clean, hits = torch.cat(n_correct_clean), torch.cat(n_hits)
        wrong = ~clean
        res[f"{tau:g}"] = {
            "clean_forget_acc": float(clean.float().mean() * 100),
            "perturbed_forget_acc": float((hits / c).mean() * 100),
            "recovery_any": float(((hits > 0) & wrong).float().sum() / max(1, wrong.sum().item()) * 100),
            "recovery_major": float(((hits > c / 2) & wrong).float().sum() / max(1, wrong.sum().item()) * 100),
            "frac_any_correct": float((hits > 0).float().mean() * 100),
            "n_clean_wrong": int(wrong.sum().item()),
        }
    return res


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", nargs="*", default=[])
    p.add_argument("--pattern", default=None)
    p.add_argument("--taus", nargs="+", type=float, default=[0.01, 0.02, 0.03, 0.05, 0.1])
    p.add_argument("--c", type=int, default=20)
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
        out = os.path.join(RECORDS_DIR, "runs", rid, "perturb_metrics.json")
        if (args.skip_done and os.path.exists(out)) or not os.path.exists(ckpt_for(rid)):
            continue
        cfg = json.load(open(os.path.join(RECORDS_DIR, "runs", rid, "config.json")))
        if cfg["scenario"] == "orig":
            continue
        split = load_split(cfg["scenario"], cfg["seed"])
        model = resnet18().to(device).to(memory_format=torch.channels_last)
        load_ckpt(model, ckpt_for(rid), device)
        r = perturbed_eval(model, data, split["forget_idx"], args.taus, args.c, cfg["seed"])
        json.dump({"protocol": f"Gaussian x+tau*N(0,I) clipped to [0,1], c={args.c} per forget sample, same noise for all models", "per_tau": r},
                  open(out, "w"), indent=1)
        print(rid, {t: round(v["perturbed_forget_acc"], 2) for t, v in r.items()})


if __name__ == "__main__":
    run_with_oom_retry(main)
