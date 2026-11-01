"""E3: compute global ES + per-sample E_i (kNN-restricted) for each scenario/seed on the ORIGINAL model embeddings.

Writes records/results/entanglement__<scenario>__s<seed>.npz (+ .json summary). Per-partition metrics of each unlearning run
are computed later on CPU from records/runs/<run_id>/per_sample.npz (forget_idx aligned).
"""
import argparse
import json
import os

import numpy as np
import torch

from mu import CKPT_DIR, RECORDS_DIR
from mu.data import CIFAR10GPU
from mu.eval.embeddings import extract
from mu.eval.entanglement import global_es, per_sample_entanglement, tertile_partition
from mu.models import resnet18
from mu.splits import load_split
from mu.utils import load_ckpt, setup_device, run_with_oom_retry


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scenarios", nargs="+", default=["S1c0", "S2k359p10", "S2k359p30", "S2k359p50"])
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--ks", nargs="+", type=int, default=[10, 50, 100])
    p.add_argument("--orig_ckpt", default=os.path.join(CKPT_DIR, "orig__original__s0.pt"))
    p.add_argument("--gpu", type=int, default=1)
    p.add_argument("--mem_frac", type=float, default=0.45)
    args = p.parse_args()
    device = setup_device(args.gpu, args.mem_frac)
    data = CIFAR10GPU(device)
    model = resnet18().to(device).to(memory_format=torch.channels_last)
    load_ckpt(model, args.orig_ckpt, device)
    os.makedirs(os.path.join(RECORDS_DIR, "results"), exist_ok=True)

    for sc in args.scenarios:
        for seed in args.seeds:
            split = load_split(sc, seed)
            phi_f, _ = extract(model, data, "train", split["forget_idx"])
            phi_r, _ = extract(model, data, "train", split["retain_idx"])
            out = {"forget_idx": split["forget_idx"]}
            summ = {"scenario": sc, "seed": seed, "global_ES": global_es(phi_r, phi_f), "n_forget": len(phi_f), "n_retain": len(phi_r)}
            for k in args.ks:
                r = per_sample_entanglement(phi_f, phi_r, k=k)
                out[f"E_k{k}"] = r["E"]
                out[f"retain_frac_k{k}"] = r["retain_frac"]
                out[f"mean_dist_retain_k{k}"] = r["mean_dist_retain_knn"]
                out[f"partition_k{k}"] = tertile_partition(r["E"], -r["mean_dist_retain_knn"])
                summ[f"k{k}"] = {"sigma": r["sigma"], "E_mean": float(r["E"].mean()), "E_quantiles": np.quantile(r["E"], [0, .25, .5, .75, 1]).tolist(),
                                 "mean_dist_retain_mean": float(r["mean_dist_retain_knn"].mean()), "frac_E_zero": float((r["E"] == 0).mean()),
                                 "corr_E_vs_dist_spearman": float(np.corrcoef(np.argsort(np.argsort(r["E"])), np.argsort(np.argsort(r["mean_dist_retain_knn"])))[0, 1])}
            base = os.path.join(RECORDS_DIR, "results", f"entanglement__{sc}__s{seed}")
            np.savez_compressed(base + ".npz", **out)
            json.dump(summ, open(base + ".json", "w"), indent=1)
            print(json.dumps({k: v for k, v in summ.items() if not k.startswith("k")}), {k: round(summ[k]["E_mean"], 4) for k in summ if k.startswith("k")})


if __name__ == "__main__":
    run_with_oom_retry(main)
