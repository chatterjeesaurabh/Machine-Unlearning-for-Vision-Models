"""E4 (embedding-space separability of the forgotten class) on QUANTIZED models: fp32 vs INT-b for the best configuration of every
method (src/mu/best_configs.py), S1 class 0, 3 seeds, +/- WS, plus Original and Retrain. Same protocol as scripts/probe_best_models.py:
  - linear probe: fresh linear head on frozen, standardised 512-d avgpool features of all 45k train-pool images (retain + forget),
    Adam lr 1e-3, bs 1024, 100 epochs; evaluated per class on the test set
  - NCC: nearest class mean of the train-pool features
  - output: the model's own class-0 test accuracy
  - silhouette of class 0 vs rest and 10-NN purity of class-0 images on 3000 test features (300 per class, fixed subset)
  - t-SNE of the same 3000 test features (seed 0 only), fp32 and INT-b side by side
Quantization = scripts/eval_quant.py protocol (simulated weight-only per-channel symmetric INT-b on Conv/Linear, BN fp32).
Outputs: records/runs/<run>/repr_quant_metrics.json; records/results/probe_quant_best.csv / .md;
         records/figures/fig_embed_quant_S1c0_int<b>.png (grid: row fp32, row INT-b) and records/figures/embed_quant/<model>_fp32_vs_int<b>.png
Usage: python scripts/probe_quant_best.py [--bits 4] [--seeds 0 1 2] [--ckpt_dirs ...]
"""
import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score

from mu import CKPT_DIR, RECORDS_DIR
from mu.best_configs import METHODS, conf_name, run_id
from mu.data import CLASSES, CIFAR10GPU
from mu.eval.embeddings import extract
from mu.eval.probe import probe_metrics
from mu.models import resnet18
from mu.splits import load_split
from mu.utils import load_ckpt, run_with_oom_retry, setup_device

sys.path.insert(0, os.path.dirname(__file__))
from eval_quant import quantized_copy  # noqa: E402

SC = "S1c0"
LABEL = {"GA": "GA", "GAGD": "GA+Descent", "TS": "Teacher-Student", "RLrand": "Random Relabel", "RLbnd": "Boundary Relabel"}


def models_list():
    out = [("Original", "original"), ("Retrain", "retrain")]
    for m in METHODS:
        for mask in ("none", "saliency"):
            out.append((f"{LABEL[m]} ({conf_name(m, SC)})" + (" + WS" if mask == "saliency" else ""), (m, mask)))
    return out


def rid_of(spec, seed):
    return f"{SC}__{spec}__s{seed}" if isinstance(spec, str) else run_id(SC, spec[0], spec[1], seed)


@torch.no_grad()
def separability(f_te_sub, ys):
    fs = f_te_sub.cpu().numpy()
    is0 = (ys == 0).astype(int)
    sil = float(silhouette_score(fs, is0, sample_size=3000, random_state=0))
    nn = torch.cdist(f_te_sub, f_te_sub).topk(11, largest=False).indices[:, 1:].cpu().numpy()
    purity = float((ys[nn[ys == 0]] == 0).mean() * 100)
    return sil, purity


def evaluate(m, data, split, seed, y_test, sub, bs=250):
    f_tr, y_tr = extract(m, data, "train", split["pool_idx"], batch_size=bs)
    f_te, y_te = extract(m, data, "test", np.arange(10000), batch_size=bs)
    r = probe_metrics(f_tr, y_tr, f_te, y_te, [0], seed=seed)
    sil, purity = separability(f_te[sub], y_test[sub])
    with torch.no_grad():
        out_pred = torch.cat([m.fc(f_te[i:i + 2000]).argmax(1) for i in range(0, 10000, 2000)]).cpu().numpy()
    row = {"output_forgetcls": float((out_pred[y_test == 0] == 0).mean() * 100), "output_othercls": float((out_pred[y_test != 0] == y_test[y_test != 0]).mean() * 100),
           "probe_forgetcls": r["probe_test_forgetcls_acc"], "ncc_forgetcls": r["ncc_test_forgetcls_acc"], "probe_othercls": r["probe_test_othercls_acc"],
           "ncc_othercls": r["ncc_test_othercls_acc"], "silhouette_class0": sil, "knn10_purity_class0": purity}
    return row, r, f_te[sub].cpu().numpy()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bits", type=int, default=4)
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--ckpt_dirs", nargs="*", default=[CKPT_DIR, "/anchor/anchor7/sachatt/cs6886/project/outputs/checkpoints"])
    p.add_argument("--gpu", type=int, default=1)
    p.add_argument("--skip_done", type=int, default=1)
    p.add_argument("--extract_bs", type=int, default=250, help="feature-extraction batch size (small: GPUs are shared and nearly full)")
    args = p.parse_args()
    b = args.bits

    def ckpt(rid):
        name = "orig__original__s0.pt" if "__original__" in rid else f"{rid}.pt"
        for d in args.ckpt_dirs:
            if os.path.exists(os.path.join(d, name)):
                return os.path.join(d, name)
        return None

    device = setup_device(args.gpu, 0.45)
    data = CIFAR10GPU(device)
    y_test = data.y_test.cpu().numpy()
    rng = np.random.RandomState(0)
    sub = np.concatenate([rng.choice(np.where(y_test == c)[0], 300, replace=False) for c in range(10)])
    fig_dir = os.path.join(RECORDS_DIR, "figures", "embed_quant")
    os.makedirs(fig_dir, exist_ok=True)
    rows, emb = [], {}
    for label, spec in models_list():
        for seed in args.seeds:
            rid = rid_of(spec, seed)
            path = ckpt(rid)
            if path is None:
                print("[skip] no checkpoint", rid)
                continue
            out = os.path.join(RECORDS_DIR, "runs", rid, "repr_quant_metrics.json")
            stem = label.replace(" ", "_").replace("/", "-")
            npz = {prec: os.path.join(fig_dir, f"{stem}_{prec}_s0.npz") for prec in ("fp32", f"int{b}")}
            cached = json.load(open(out)) if args.skip_done and os.path.exists(out) else {}
            if f"int{b}" in cached and (seed != 0 or all(os.path.exists(f) for f in npz.values())):
                for prec in ("fp32", f"int{b}"):
                    rows.append({"model": label, "seed": seed, "precision": prec, "run_id": rid, **cached[prec]["summary"]})
                    if seed == 0:
                        z = np.load(npz[prec])
                        emb[(label, prec)] = (z["z"], z["y"])
                continue
            split = load_split(SC, seed)
            m = resnet18().to(device).to(memory_format=torch.channels_last)
            load_ckpt(m, path, device)
            res = {"protocol": f"probe: Adam lr1e-3 bs1024 100ep on standardised 512-d features of all 45k train-pool (retain+forget); NCC: train-pool class means; "
                               f"silhouette/10-NN purity on 3000 test features; quantization: simulated per-channel symmetric weight-only INT{b} (Conv/Linear), BN fp32"}
            for prec, model in (("fp32", m), (f"int{b}", quantized_copy(m, b))):
                row, full, fs = evaluate(model, data, split, seed, y_test, sub, bs=args.extract_bs)
                res[prec] = {"summary": row, "probe_metrics": full}
                rows.append({"model": label, "seed": seed, "precision": prec, "run_id": rid, **row})
                print(label, seed, prec, {k: round(v, 2) for k, v in row.items()})
                if seed == 0:
                    z = TSNE(2, init="pca", perplexity=30, random_state=0).fit_transform(fs)
                    emb[(label, prec)] = (z, y_test[sub])
                    np.savez(npz[prec], z=z, y=y_test[sub])
            json.dump(res, open(out, "w"), indent=1)

    t = pd.DataFrame(rows)
    t.to_csv(os.path.join(RECORDS_DIR, "results", f"probe_quant_best_int{b}.csv"), index=False)
    cols = ["output_forgetcls", "probe_forgetcls", "ncc_forgetcls", "probe_othercls", "silhouette_class0", "knn10_purity_class0"]
    g = t.groupby(["model", "precision"], sort=False)[cols].agg(["mean", "std"])
    order = [lab for lab, _ in models_list() if lab in t["model"].values]
    L = [f"# E4 on quantized models: embedding-space separability of the forgotten class (S1 class 0), fp32 vs INT{b}, best configuration of every method",
         f"Seeds: {sorted(t.seed.unique().tolist())}. output = model's own class-0 test accuracy; probe = fresh linear head on frozen features trained on all 45k train-pool images",
         "(retain + forget); NCC = nearest class mean; silhouette = class 0 vs rest on 3000 test features (higher = more separable); 10-NN purity = % of the 10 nearest",
         f"test neighbours of class-0 test images that are class 0. INT{b} = simulated weight-only per-channel symmetric quantization of Conv/Linear weights (BN fp32).",
         f"Generated by `scripts/probe_quant_best.py --bits {b}`; per-run values in `records/runs/<run>/repr_quant_metrics.json`, table `probe_quant_best_int{b}.csv`.", "",
         "## Per model and precision (mean ± std over seeds)",
         "| model | precision | output forget-cls | probe forget-cls | NCC forget-cls | probe other-cls | silhouette class 0 | 10-NN purity class 0 |", "|---|---|---|---|---|---|---|---|"]
    f = lambda r, c, d=2: f"{r[(c, 'mean')]:.{d}f} ± {r[(c, 'std')]:.{d}f}"
    for lab in order:
        for prec in ("fp32", f"int{b}"):
            r = g.loc[(lab, prec)]
            L.append(f"| {lab} | {prec} | {f(r, 'output_forgetcls')} | {f(r, 'probe_forgetcls')} | {f(r, 'ncc_forgetcls')} | {f(r, 'probe_othercls')} | {f(r, 'silhouette_class0', 3)} | {f(r, 'knn10_purity_class0')} |")
    L += ["", f"## Change INT{b} - fp32 (mean over seeds)", "| model | d output forget-cls | d probe forget-cls | d NCC forget-cls | d probe other-cls | d silhouette | d 10-NN purity |", "|---|---|---|---|---|---|---|"]
    gm = t.groupby(["model", "precision"], sort=False)[cols].mean()
    for lab in order:
        d = gm.loc[(lab, f"int{b}")] - gm.loc[(lab, "fp32")]
        L.append(f"| {lab} | {d.output_forgetcls:+.2f} | {d.probe_forgetcls:+.2f} | {d.ncc_forgetcls:+.2f} | {d.probe_othercls:+.2f} | {d.silhouette_class0:+.3f} | {d.knn10_purity_class0:+.2f} |")
    open(os.path.join(RECORDS_DIR, "results", f"probe_quant_best_int{b}.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))

    # figures: grid (row fp32 / row INTb) + one side-by-side image per model
    def scatter(ax, z, ys, title):
        for c in range(1, 10):
            ax.scatter(z[ys == c, 0], z[ys == c, 1], s=2, color=plt.cm.tab10(c), alpha=0.5, label=CLASSES[c])
        ax.scatter(z[ys == 0, 0], z[ys == 0, 1], s=3, color="black", label="airplane (forgotten)")
        ax.set_title(title, fontsize=7)
        ax.set_xticks([]), ax.set_yticks([])

    def title(lab, prec):
        pr = t[(t.model == lab) & (t.seed == 0) & (t.precision == prec)].iloc[0]
        return f"{lab} [{prec}]\noutput {pr.output_forgetcls:.1f} | probe {pr.probe_forgetcls:.1f} | NCC {pr.ncc_forgetcls:.1f} | purity {pr.knn10_purity_class0:.0f}"

    labs = [lab for lab in order if (lab, "fp32") in emb]
    fig, axes = plt.subplots(2, len(labs), figsize=(3.6 * len(labs), 8), squeeze=False)
    for j, lab in enumerate(labs):
        for i, prec in enumerate(("fp32", f"int{b}")):
            scatter(axes[i][j], *emb[(lab, prec)], title(lab, prec))
        fig2, ax2 = plt.subplots(1, 2, figsize=(9, 4.6))
        for ax, prec in zip(ax2, ("fp32", f"int{b}")):
            scatter(ax, *emb[(lab, prec)], title(lab, prec))
        ax2[0].legend(fontsize=6, markerscale=3, loc="best")
        fig2.tight_layout()
        fig2.savefig(os.path.join(fig_dir, f"{lab.replace(' ', '_').replace('/', '-')}_fp32_vs_int{b}.png"), dpi=130)
        plt.close(fig2)
    axes[0][0].legend(fontsize=6, markerscale=3, loc="best")
    fig.suptitle(f"t-SNE of test-set features (300 per class, seed 0), S1 class 0; row 1 fp32, row 2 INT{b}; black = forgotten class; titles: class-0 test acc of output / probe / NCC, 10-NN purity", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(RECORDS_DIR, "figures", f"fig_embed_quant_{SC}_int{b}.png"), dpi=130)
    print("figures ->", os.path.join(RECORDS_DIR, "figures", f"fig_embed_quant_{SC}_int{b}.png"), "and", fig_dir)


if __name__ == "__main__":
    run_with_oom_retry(main)
