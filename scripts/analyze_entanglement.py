"""E3 analysis (CPU): per-partition (low / medium / high per-sample entanglement E_i, kNN k) unlearning performance.

Inputs: records/results/entanglement__<sc>__s<seed>.npz (scripts/compute_entanglement.py) and per_sample.npz of every main run.
Per partition: forget acc, |forget acc - Retrain forget acc| (same partition), MIA acc (partition forget losses vs same-class test).
Outputs: records/results/entanglement_partitions.csv, entanglement_summary.csv (global ES per forget set),
         records/figures/fig_entanglement_partitions_k<k>.png, fig_entanglement_hist.png
"""
import argparse
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from mu import RECORDS_DIR
from mu.data import load_cifar10_numpy
from mu.eval.entanglement import tertile_partition
from mu.eval.mia import loss_mia

RES, FIG = os.path.join(RECORDS_DIR, "results"), os.path.join(RECORDS_DIR, "figures")
SCEN = ["S1c0", "S2k359p10", "S2k359p30", "S2k359p50"]
METHODS = ["GA", "GAGD", "TS", "RLrand", "RLbnd"]
PART = ["low", "medium", "high"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--k", type=int, default=50)
    args = p.parse_args()
    df = pd.read_csv(os.path.join(RES, "all_runs.csv"))
    df = df[(df["tag"].fillna("") == "") & df["scenario"].isin(SCEN)]
    rows, summ = [], []
    y_train = load_cifar10_numpy()[1]
    for path in sorted(glob.glob(os.path.join(RES, "entanglement__*.npz"))):
        _, sc, s = os.path.basename(path)[:-4].split("__")
        seed = int(s[1:])
        ent = np.load(path)
        summ.append(json.load(open(path.replace(".npz", ".json"))))
        part = tertile_partition(ent[f"E_k{args.k}"], -ent[f"mean_dist_retain_k{args.k}"])  # rank tertiles, ties by retain distance
        fidx = ent["forget_idx"]
        for _, r in df[(df["scenario"] == sc) & (df["seed"] == seed)].iterrows():
            ps = os.path.join(RECORDS_DIR, "runs", r["run_id"], "per_sample.npz")
            if not os.path.exists(ps):
                continue
            d = np.load(ps)
            assert np.array_equal(d["forget_idx"], fidx), r["run_id"]
            lab = y_train[fidx]  # retrain per_sample.npz has no forget_label; use CIFAR labels (original 50k order)
            correct = d["forget_pred"] == lab
            key = (r["method"], r.get("mask", "none") if isinstance(r.get("mask"), str) else "none")
            for pi, pn in enumerate(PART):
                m = part == pi
                row = {"scenario": sc, "seed": seed, "method": key[0], "mask": key[1], "partition": pn, "k": args.k, "n": int(m.sum()),
                       "E_mean": float(ent[f"E_k{args.k}"][m].mean()), "forget_acc": float(correct[m].mean() * 100)}
                row["mia_acc"] = loss_mia(d["forget_loss"][m], lab[m], d["test_loss"], d["test_label"], seed=seed)["mia_acc"]
                rows.append(row)
    out = pd.DataFrame(rows)
    if out.empty:
        print("no data")
        return
    rt = out[out["method"] == "retrain"].set_index(["scenario", "seed", "partition"])
    for c in ("forget_acc", "mia_acc"):
        if c in out:
            out[f"gap_{c}"] = [abs(r[c] - rt.loc[(r["scenario"], r["seed"], r["partition"]), c]) if (r["scenario"], r["seed"], r["partition"]) in rt.index else np.nan
                               for _, r in out.iterrows()]
    out.to_csv(os.path.join(RES, f"entanglement_partitions_k{args.k}.csv"), index=False)
    pd.DataFrame(summ).to_csv(os.path.join(RES, "entanglement_summary.csv"), index=False)

    agg = out.groupby(["scenario", "method", "mask", "partition"])[["forget_acc", "gap_forget_acc", "gap_mia_acc"]].mean().reset_index()
    scs = [s for s in SCEN if s in set(agg["scenario"])]
    fig, axes = plt.subplots(2, len(scs), figsize=(4.5 * len(scs), 7), squeeze=False)
    for j, sc in enumerate(scs):
        for i, (c, lab) in enumerate((("gap_forget_acc", "|forget acc - Retrain|"), ("gap_mia_acc", "|MIA acc - Retrain|"))):
            ax = axes[i][j]
            a = agg[(agg["scenario"] == sc) & agg["method"].isin(METHODS)]
            labels = [f"{m}{'+WS' if k == 'saliency' else ''}" for m in METHODS for k in ("none", "saliency")]
            x = np.arange(len(labels))
            for pi, pn in enumerate(PART):
                v = [a[(a["method"] == m) & (a["mask"] == k) & (a["partition"] == pn)][c].mean() for m in METHODS for k in ("none", "saliency")]
                ax.bar(x + (pi - 1) * 0.27, v, 0.27, label=f"{pn} entanglement")
            ax.set_xticks(x, labels, rotation=60, fontsize=7)
            ax.set_title(f"{sc}: {lab}", fontsize=9)
    axes[0][0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, f"fig_entanglement_partitions_k{args.k}.png"), dpi=140)
    plt.close(fig)
    print(agg[agg["method"].isin(METHODS + ["retrain", "original"])].round(2).to_string(index=False))


if __name__ == "__main__":
    main()
