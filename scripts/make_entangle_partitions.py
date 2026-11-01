"""Entanglement partitions of each forget set (Original-model embeddings, 512-d avgpool, no augmentation).

Per-sample entanglement score (slide 14): mean Euclidean distance from a forget sample to its k nearest RETAIN samples
(k = 50). Smaller distance = more entangled. The forget set is ranked and split into equal thirds:
  low  = largest distances (least entangled), medium, high = smallest distances (most entangled).
Writes split files (indices into the original 50k train set):
  Experiment 1 (individual partitions, RUM protocol):  <base>-E{low,medium,high}__s<seed>.npz
      forget = partition, retain = pool \\ partition (the other two partitions join the retain set)
  Experiment 2 (sequential low -> medium -> high):     <base>-Eseq{1,2,3}__s<seed>.npz
      seq1: forget = low,    retain = pool \\ low               (medium + high still in retain)
      seq2: forget = medium, retain = pool \\ (low u medium)     (high still in retain)
      seq3: forget = high,   retain = pool \\ whole forget set   (= base retain)
Also writes records/results/entangle_partitions.json (sizes, distance ranges, global ES of each partition vs its retain set).
"""
import json
import os

import numpy as np
import torch

from mu import CKPT_DIR, RECORDS_DIR
from mu.data import CIFAR10GPU
from mu.eval.embeddings import extract
from mu.eval.entanglement import global_es
from mu.models import resnet18
from mu.splits import SPLIT_DIR, load_split
from mu.utils import load_ckpt, run_with_oom_retry, setup_device

SCEN = ["S1c0", "S2k359p10", "S2k359p30", "S2k359p50"]
K = 50


@torch.no_grad()
def knn_mean_dist(phi_f, phi_r, k=K, chunk=1024):
    out = []
    for s in range(0, len(phi_f), chunk):
        out.append(torch.cdist(phi_f[s:s + chunk], phi_r).topk(k, largest=False).values.mean(1))
    return torch.cat(out).cpu().numpy()


def main():
    device = setup_device(1, 0.45)
    data = CIFAR10GPU(device)
    model = resnet18().to(device).to(memory_format=torch.channels_last)
    load_ckpt(model, os.path.join(CKPT_DIR, "orig__original__s0.pt"), device)
    summary = {}
    for sc in SCEN:
        for seed in (0, 1, 2):
            sp = load_split(sc, seed)
            f, pool = sp["forget_idx"], sp["pool_idx"]
            phi_f, _ = extract(model, data, "train", f)
            phi_r, _ = extract(model, data, "train", sp["retain_idx"])
            d = knn_mean_dist(phi_f, phi_r)
            order = np.argsort(-d, kind="stable")  # largest distance first = least entangled
            thirds = np.array_split(order, 3)
            parts = {"low": np.sort(f[thirds[0]]), "medium": np.sort(f[thirds[1]]), "high": np.sort(f[thirds[2]])}
            info = {}
            for name, idx in parts.items():
                retain = np.setdiff1d(pool, idx)
                np.savez(os.path.join(SPLIT_DIR, f"{sc}-E{name}__s{seed}.npz"), forget_idx=idx, retain_idx=retain, knn_dist=d[np.isin(f, idx)])
                m = np.isin(f, idx)
                info[name] = {"n": int(len(idx)), "dist_min": float(d[m].min()), "dist_max": float(d[m].max()), "dist_mean": float(d[m].mean()),
                              "global_ES_vs_retain": global_es(phi_r, phi_f[torch.as_tensor(m, device=device)])}
            seq = [("seq1", parts["low"], np.setdiff1d(pool, parts["low"])),
                   ("seq2", parts["medium"], np.setdiff1d(pool, np.concatenate([parts["low"], parts["medium"]]))),
                   ("seq3", parts["high"], np.setdiff1d(pool, f))]
            for name, fi, ri in seq:
                np.savez(os.path.join(SPLIT_DIR, f"{sc}-E{name}__s{seed}.npz"), forget_idx=fi, retain_idx=ri)
            summary[f"{sc}__s{seed}"] = {"k": K, "partitions": info}
            print(sc, seed, {n: (v["n"], round(v["dist_mean"], 3), round(v["global_ES_vs_retain"], 1)) for n, v in info.items()})
    json.dump(summary, open(os.path.join(RECORDS_DIR, "results", "entangle_partitions.json"), "w"), indent=1)


if __name__ == "__main__":
    run_with_oom_retry(main)
