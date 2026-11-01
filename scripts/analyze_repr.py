"""E4 analysis (CPU): output forget-class test accuracy vs linear-probe and NCC forget-class accuracy (S1 class forgetting).
Outputs: records/results/repr_table.csv / .md, records/figures/fig_probe_vs_output.png
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from mu import RECORDS_DIR

RES, FIG = os.path.join(RECORDS_DIR, "results"), os.path.join(RECORDS_DIR, "figures")
ORDER = ["original", "retrain", "GA", "GAGD", "TS", "RLrand", "RLbnd"]


def main():
    df = pd.read_csv(os.path.join(RES, "all_runs.csv"))
    df = df[(df["tag"].fillna("") == "") & (df["scenario"] == "S1c0")].copy()
    df["mask"] = df["mask"].fillna("none")
    rows = []
    for _, r in df.iterrows():
        p = os.path.join(RECORDS_DIR, "runs", r["run_id"], "repr_metrics.json")
        if not os.path.exists(p):
            continue
        m = json.load(open(p))
        rows.append({"method": r["method"], "mask": r["mask"], "seed": r["seed"],
                     "output_forgetcls": r["acc_test_forgetcls"], "output_othercls": r["acc_test_othercls"],
                     "probe_forgetcls": m["probe_test_forgetcls_acc"], "probe_othercls": m["probe_test_othercls_acc"],
                     "ncc_forgetcls": m["ncc_test_forgetcls_acc"], "ncc_othercls": m["ncc_test_othercls_acc"]})
    t = pd.DataFrame(rows)
    if t.empty:
        print("no repr metrics yet")
        return
    g = t.groupby(["method", "mask"]).agg(["mean", "std"]).drop(columns="seed")
    g.columns = [f"{a}_{b}" for a, b in g.columns]
    g = g.reset_index()
    g["_o"] = g["method"].map({m: i for i, m in enumerate(ORDER)}) * 2 + (g["mask"] == "saliency")
    g = g.sort_values("_o").drop(columns="_o")
    g.to_csv(os.path.join(RES, "repr_table.csv"), index=False)
    lines = ["# E4: output vs representation-level accuracy on the forgotten class (S1c0, test set, mean ± std over seeds)",
             "Probe = new linear head on frozen 512-d features trained on all 45k train-pool samples (retain + forget). NCC = nearest class mean.",
             "| method | WS | output forget-cls | probe forget-cls | NCC forget-cls | output other-cls | probe other-cls | NCC other-cls |", "|---|---|---|---|---|---|---|---|"]
    for _, r in g.iterrows():
        f = lambda c: f"{r[c + '_mean']:.2f} ± {r[c + '_std']:.2f}"
        lines.append(f"| {r['method']} | {'yes' if r['mask'] == 'saliency' else 'no'} | {f('output_forgetcls')} | {f('probe_forgetcls')} | {f('ncc_forgetcls')} | "
                     f"{f('output_othercls')} | {f('probe_othercls')} | {f('ncc_othercls')} |")
    open(os.path.join(RES, "repr_table.md"), "w").write("\n".join(lines) + "\n")

    labels = [f"{m}{'+WS' if k == 'saliency' else ''}" for m, k in zip(g["method"], g["mask"])]
    x = np.arange(len(g))
    fig, ax = plt.subplots(figsize=(max(8, len(g) * 0.8), 4))
    for i, (c, lab) in enumerate((("output_forgetcls", "Output"), ("probe_forgetcls", "Linear probe"), ("ncc_forgetcls", "NCC"))):
        ax.bar(x + (i - 1) * 0.27, g[c + "_mean"], 0.27, yerr=g[c + "_std"], capsize=2, label=lab)
    ax.set_xticks(x, labels, rotation=40, fontsize=8)
    ax.set_ylabel("accuracy on forgotten class (test, %)")
    ax.set_title("S1 class forgetting: is the forgotten class still recoverable from the features?", fontsize=10)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_probe_vs_output.png"), dpi=150)
    plt.close(fig)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
