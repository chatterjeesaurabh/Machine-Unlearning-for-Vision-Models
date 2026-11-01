"""E4 (embedding-space separability) on every method's BEST configuration (records/results/method_comparison.md), S1 class 0.

1. Linear probe + NCC from scratch for all seeds (scripts/eval_repr.py protocol, recomputed): frozen 512-d avgpool features of all 45k
   train-pool images (retain + forget), standardised, fresh linear head (Adam lr 1e-3, bs 1024, 100 epochs); evaluated per class on test.
2. Embedding plots (seed 0): t-SNE of test-set features (300 images per class) for every model in one grid; forgotten class 0 in black.
   Separability numbers per model (layout-free, 512-d): silhouette of class 0 vs the rest, and 10-NN purity of class-0 test images
   (fraction of their 10 nearest test neighbours that are also class 0).
Outputs: records/runs/<run>/repr_metrics.json (overwritten), records/results/probe_best.md/.csv, records/figures/fig_embed_best_S1c0.png
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score

from mu import CKPT_DIR, RECORDS_DIR
from mu.data import CLASSES, CIFAR10GPU
from mu.eval.embeddings import extract
from mu.eval.probe import probe_metrics
from mu.models import resnet18
from mu.splits import load_split
from mu.utils import load_ckpt, run_with_oom_retry, setup_device

MODELS = [  # (label, run-id stem; full id = stem + "__s<seed>")
    ("Original", "S1c0__original"), ("Retrain", "S1c0__retrain"),
    ("GA (G3)", "S1c0__GA__imp5_gamix"), ("GA (G3) + WS", "S1c0__GA__sal0.5__imp5_gamix"),
    ("GA+Descent (D1)", "S1c0__GAGD"), ("GA+Descent (D1) + WS", "S1c0__GAGD__sal0.5"),
    ("Teacher-Student (T1)", "S1c0__TS"), ("Teacher-Student (T1) + WS", "S1c0__TS__sal0.5"),
    ("Random Relabel (R1)", "S1c0__RLrand"), ("Random Relabel (R1) + WS", "S1c0__RLrand__sal0.5"),
    ("Boundary Relabel (C1-mix)", "S1c0__RLbnd__imp5_bndlogit2mix"), ("Boundary Relabel (C1-mix) + WS", "S1c0__RLbnd__sal0.5__imp5_bndlogit2mix"),
]


def ckpt(rid):
    return os.path.join(CKPT_DIR, "orig__original__s0.pt") if "__original" in rid else os.path.join(CKPT_DIR, f"{rid}.pt")


def main():
    device = setup_device(1, 0.45)
    data = CIFAR10GPU(device)
    y_test = data.y_test.cpu().numpy()
    rng = np.random.RandomState(0)
    sub = np.concatenate([rng.choice(np.where(y_test == c)[0], 300, replace=False) for c in range(10)])
    rows, emb = [], {}
    for label, stem in MODELS:
        for seed in (0, 1, 2):
            rid = f"{stem}__s{seed}"
            split = load_split("S1c0", seed)
            m = resnet18().to(device).to(memory_format=torch.channels_last)
            load_ckpt(m, ckpt(rid), device)
            f_tr, y_tr = extract(m, data, "train", split["pool_idx"])
            f_te, y_te = extract(m, data, "test", np.arange(10000))
            r = probe_metrics(f_tr, y_tr, f_te, y_te, [0], seed=seed)
            r["protocol"] = "probe: Adam lr1e-3 bs1024 100ep on standardised 512-d features of all 45k train-pool (retain+forget); NCC: train-pool class means"
            json.dump(r, open(os.path.join(RECORDS_DIR, "runs", rid, "repr_metrics.json"), "w"), indent=1)
            fs, ys = f_te[sub].cpu().numpy(), y_test[sub]
            is0 = (ys == 0).astype(int)
            sil = float(silhouette_score(fs, is0, sample_size=3000, random_state=0))
            nn = torch.cdist(f_te[sub], f_te[sub]).topk(11, largest=False).indices[:, 1:].cpu().numpy()
            purity = float((ys[nn[ys == 0]] == 0).mean() * 100)
            with torch.no_grad():
                out_pred = torch.cat([m.fc(f_te[i:i + 2000]).argmax(1) for i in range(0, 10000, 2000)]).cpu().numpy()
            rows.append({"model": label, "seed": seed, "output_forgetcls": float((out_pred[y_test == 0] == 0).mean() * 100),
                         "probe_forgetcls": r["probe_test_forgetcls_acc"], "ncc_forgetcls": r["ncc_test_forgetcls_acc"],
                         "probe_othercls": r["probe_test_othercls_acc"], "silhouette_class0": sil, "knn10_purity_class0": purity})
            if seed == 0:
                emb[label] = (TSNE(2, init="pca", perplexity=30, random_state=0).fit_transform(fs), ys)
            print(label, seed, {k: round(v, 2) for k, v in rows[-1].items() if isinstance(v, float)})
    t = pd.DataFrame(rows)
    t.to_csv(os.path.join(RECORDS_DIR, "results", "probe_best.csv"), index=False)
    g = t.groupby("model", sort=False).agg(["mean", "std"]).drop(columns="seed")
    lines = ["# E4: embedding-space separability of the forgotten class (S1 class 0), best configuration of every method, 3 seeds",
             "output = model's own prediction; probe = fresh linear head on frozen features trained on all 45k train-pool images (retain+forget);",
             "NCC = nearest class mean; silhouette = class 0 vs rest on 3000 test features (higher = more separable); 10-NN purity = % of the 10 nearest",
             "test neighbours of class-0 test images that are class 0. All accuracies on the test set (%).", "",
             "| model | output forget-cls | probe forget-cls | NCC forget-cls | probe other-cls | silhouette class 0 | 10-NN purity class 0 |", "|---|---|---|---|---|---|---|"]
    for mdl, r in g.iterrows():
        f = lambda c, d=2: f"{r[(c, 'mean')]:.{d}f} ± {r[(c, 'std')]:.{d}f}"
        lines.append(f"| {mdl} | {f('output_forgetcls')} | {f('probe_forgetcls')} | {f('ncc_forgetcls')} | {f('probe_othercls')} | {f('silhouette_class0', 3)} | {f('knn10_purity_class0')} |")
    open(os.path.join(RECORDS_DIR, "results", "probe_best.md"), "w").write("\n".join(lines) + "\n")
    n = len(emb)
    fig, axes = plt.subplots(2, (n + 1) // 2, figsize=(4 * ((n + 1) // 2), 8.4))
    for ax, (label, (z, ys)) in zip(axes.flat, emb.items()):
        for c in range(1, 10):
            ax.scatter(z[ys == c, 0], z[ys == c, 1], s=2, color=plt.cm.tab10(c), alpha=0.5, label=CLASSES[c])
        ax.scatter(z[ys == 0, 0], z[ys == 0, 1], s=3, color="black", label="airplane (forgotten)")
        pr = t[(t["model"] == label) & (t["seed"] == 0)].iloc[0]
        ax.set_title(f"{label}\noutput {pr.output_forgetcls:.1f}% | probe {pr.probe_forgetcls:.1f}% | NCC {pr.ncc_forgetcls:.1f}%", fontsize=8)
        ax.set_xticks([]), ax.set_yticks([])
    axes.flat[0].legend(fontsize=6, markerscale=3, loc="best")
    fig.suptitle("t-SNE of test-set features (300 per class, seed 0); black = forgotten class 0; title: class-0 test accuracy of output / linear probe / NCC", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(RECORDS_DIR, "figures", "fig_embed_best_S1c0.png"), dpi=130)
    print("\n".join(lines))


if __name__ == "__main__":
    run_with_oom_retry(main)
