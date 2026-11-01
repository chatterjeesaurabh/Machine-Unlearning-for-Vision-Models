"""Embedding check for Boundary Relabeling (any S1 run with a checkpoint): where do forget samples sit in feature space
BEFORE (Original) and AFTER unlearning, relative to the other classes - did they really move into their assigned
neighbour class?

Points: all forget samples + 500 random train-pool samples of every other class (no augmentation), 512-d avgpool features.
Plots (t-SNE fitted separately for each model): left = Original, right = unlearned; retain points coloured by class (small),
forget points coloured by their ASSIGNED neighbour label (from the Original model, same search as the run).
Quantitative (layout-free, in the unlearned model's 512-d feature space):
  - nn_class: majority class of the k=10 nearest retain points; centroid_class: nearest retain-class centroid
  - % forget samples whose nn_class / centroid_class / unlearned prediction == assigned neighbour label
  - same quantities for the Original model (baseline)
Writes records/figures/fig_embed_<run_id>.png and records/runs/<run_id>/embedding_check.json
Usage: python scripts/plot_embeddings.py --runs S1c0__RLbnd__s0 [...]
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.manifold import TSNE

from mu import CKPT_DIR, RECORDS_DIR
from mu.data import CLASSES, CIFAR10GPU, GPULoader
from mu.eval.embeddings import extract
from mu.models import resnet18
from mu.splits import load_split
from mu.unlearn.methods import neighbour_labels
from mu.utils import load_ckpt, run_with_oom_retry, setup_device


def neighbour_of(orig, data, idx, hp):
    labs = []
    for x, y, _ in GPULoader(data, "train", idx, 500):
        labs.append(neighbour_labels(orig, x, y, hp.get("eps", 0.1), hp.get("nb_search", "fgsm"), hp.get("nb_steps", 10)))
    return torch.cat(labs).cpu().numpy()


@torch.no_grad()
def predict_labels(model, data, idx):
    out = []
    for x, _, _ in GPULoader(data, "train", idx, 1000):
        out.append(model(x).argmax(1))
    return torch.cat(out).cpu().numpy()


def check(feat_f, feat_r, y_r, assigned, pred, k=10):
    cents = torch.stack([feat_r[y_r == c].mean(0) for c in range(10) if (y_r == c).any()])
    cls = torch.tensor([c for c in range(10) if (y_r == c).any()], device=feat_r.device)
    centroid_class = cls[torch.cdist(feat_f, cents).argmin(1)].cpu().numpy()
    nn = torch.cdist(feat_f, feat_r).topk(k, largest=False).indices
    nn_class = torch.mode(y_r[nn], 1).values.cpu().numpy()
    return {"pct_nn_class_eq_assigned": float((nn_class == assigned).mean() * 100),
            "pct_centroid_class_eq_assigned": float((centroid_class == assigned).mean() * 100),
            "pct_pred_eq_assigned": float((pred == assigned).mean() * 100),
            "nn_class_hist": np.bincount(nn_class, minlength=10).tolist()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--per_class", type=int, default=500)
    args = p.parse_args()
    device = setup_device(1, 0.45)
    data = CIFAR10GPU(device)
    orig = resnet18().to(device).to(memory_format=torch.channels_last)
    load_ckpt(orig, os.path.join(CKPT_DIR, "orig__original__s0.pt"), device)
    orig.eval()
    y_all = data.y_train.cpu().numpy()
    for rid in args.runs:
        cfg = json.load(open(os.path.join(RECORDS_DIR, "runs", rid, "config.json")))
        split = load_split(cfg["scenario"], cfg["seed"])
        hp = cfg.get("hparams") or {}
        fidx = split["forget_idx"]
        rng = np.random.RandomState(0)
        r = split["retain_idx"]
        ridx = np.concatenate([rng.choice(r[y_all[r] == c], args.per_class, replace=False) for c in range(10) if (y_all[r] == c).sum() >= args.per_class])
        assigned = neighbour_of(orig, data, fidx, hp)
        model = resnet18().to(device).to(memory_format=torch.channels_last)
        load_ckpt(model, os.path.join(CKPT_DIR, f"{rid}.pt"), device)
        model.eval()
        res, emb = {"assigned_hist": np.bincount(assigned, minlength=10).tolist(), "neighbour_search": hp.get("nb_search", "fgsm")}, {}
        for name, m in (("original", orig), ("unlearned", model)):
            ff, _ = extract(m, data, "train", fidx)
            fr, yr = extract(m, data, "train", ridx)
            res[name] = check(ff, fr, yr, assigned, predict_labels(m, data, fidx))
            z = TSNE(2, init="pca", perplexity=30, random_state=0).fit_transform(torch.cat([fr, ff]).cpu().numpy())
            emb[name] = (z[:len(ridx)], z[len(ridx):], yr.cpu().numpy())
        json.dump(res, open(os.path.join(RECORDS_DIR, "runs", rid, "embedding_check.json"), "w"), indent=1)
        fig, axes = plt.subplots(1, 2, figsize=(15, 7))
        cmap = plt.cm.tab10
        for ax, name in zip(axes, ("original", "unlearned")):
            zr, zf, yr = emb[name]
            for c in np.unique(yr):
                ax.scatter(zr[yr == c, 0], zr[yr == c, 1], s=3, color=cmap(c), alpha=0.35, label=f"{CLASSES[c]} (retain)")
            for c in np.unique(assigned):
                mk = assigned == c
                ax.scatter(zf[mk, 0], zf[mk, 1], s=7, color=cmap(c), marker="x", linewidths=0.6, label=f"forget -> {CLASSES[c]} ({mk.sum()})")
            st = res[name]
            ax.set_title(f"{'Original' if name == 'original' else rid}\nforget samples whose 10-NN class = assigned neighbour: "
                         f"{st['pct_nn_class_eq_assigned']:.1f}%, prediction = assigned: {st['pct_pred_eq_assigned']:.1f}%", fontsize=9)
            ax.set_xticks([]), ax.set_yticks([])
        axes[1].legend(fontsize=6, markerscale=2, loc="center left", bbox_to_anchor=(1, 0.5))
        fig.suptitle("t-SNE of 512-d features: dots = retain samples (by class), x = forget samples coloured by ASSIGNED neighbour class", fontsize=10)
        fig.tight_layout()
        fig.savefig(os.path.join(RECORDS_DIR, "figures", f"fig_embed_{rid}.png"), dpi=140)
        plt.close(fig)
        print(rid, json.dumps({k: v for k, v in res.items() if k != "assigned_hist"}))


if __name__ == "__main__":
    run_with_oom_retry(main)
