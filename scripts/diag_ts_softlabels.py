"""Diagnostic for Teacher-Student: how much class-0 (forgotten class) information do the TEACHER's soft labels carry on
RETAIN images (S1c0: classes 1-9 only)? Reports, per retain class, the mean teacher probability of class 0 at temperature
T=1 and T=4, and how often class 0 is the teacher's 2nd-ranked class. Writes records/results/ts_softlabel_diag.json/.md
"""
import json
import os

import numpy as np
import torch

from mu import CKPT_DIR, RECORDS_DIR
from mu.data import CLASSES, CIFAR10GPU, GPULoader
from mu.models import resnet18
from mu.splits import load_split
from mu.utils import load_ckpt, run_with_oom_retry, setup_device


@torch.no_grad()
def main():
    device = setup_device(1, 0.45)
    data = CIFAR10GPU(device)
    model = resnet18().to(device).to(memory_format=torch.channels_last)
    load_ckpt(model, os.path.join(CKPT_DIR, "orig__original__s0.pt"), device)
    model.eval()
    split = load_split("S1c0", 0)
    logits, labels = [], []
    for x, y, _ in GPULoader(data, "train", split["retain_idx"], 1000):
        logits.append(model(x).float())
        labels.append(y)
    logits, labels = torch.cat(logits), torch.cat(labels)
    out = {}
    for T in (1, 2, 4):
        p0 = (logits / T).softmax(1)[:, 0]
        out[f"T{T}_mean_p_class0_all_retain"] = float(p0.mean())
        out[f"T{T}_per_class_mean_p_class0"] = {CLASSES[c]: float(p0[labels == c].mean()) for c in range(1, 10)}
        # KL contribution scale: probability mass outside the true class that sits on class 0
        p = (logits / T).softmax(1)
        nontrue = 1 - p.gather(1, labels[:, None])[:, 0]
        out[f"T{T}_share_of_nontrue_mass_on_class0"] = float((p[:, 0] / nontrue.clamp_min(1e-12)).mean())
    masked = logits.clone()
    masked.scatter_(1, labels[:, None], float("-inf"))
    second = masked.argmax(1)
    out["frac_retain_with_class0_as_2nd"] = float((second == 0).float().mean())
    out["per_class_frac_class0_2nd"] = {CLASSES[c]: float((second[labels == c] == 0).float().mean()) for c in range(1, 10)}
    json.dump(out, open(os.path.join(RECORDS_DIR, "results", "ts_softlabel_diag.json"), "w"), indent=1)
    lines = ["# Teacher soft labels on S1c0 retain images (classes 1-9): class-0 (airplane) content", "",
             "| retain class | p(class0) T=1 | p(class0) T=4 | class0 is 2nd-ranked |", "|---|---|---|---|"]
    for c in range(1, 10):
        n = CLASSES[c]
        lines.append(f"| {n} | {out['T1_per_class_mean_p_class0'][n]:.5f} | {out['T4_per_class_mean_p_class0'][n]:.4f} | {out['per_class_frac_class0_2nd'][n]*100:.1f}% |")
    lines += ["", f"- mean p(class0) over all retain: T=1 {out['T1_mean_p_class0_all_retain']:.5f}, T=4 {out['T4_mean_p_class0_all_retain']:.4f}",
              f"- share of non-true-class probability mass on class 0: T=1 {out['T1_share_of_nontrue_mass_on_class0']:.3f}, T=4 {out['T4_share_of_nontrue_mass_on_class0']:.3f} (uniform would be 1/9 = 0.111)",
              f"- class 0 is the 2nd-ranked class for {out['frac_retain_with_class0_as_2nd']*100:.1f}% of retain images"]
    open(os.path.join(RECORDS_DIR, "results", "ts_softlabel_diag.md"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    run_with_oom_retry(main)
