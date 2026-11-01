"""Main-grid analysis (CPU only): tables (mean +- std over seeds) and figures.

Outputs (records/results/, records/figures/):
  main_table.csv / main_table.md         per scenario x method x mask: forget/retain/test acc, MIA acc/AUC, avg_gap, time, memory
  saliency_gain.csv                      avg_gap(with mask) - avg_gap(without mask) per scenario x method
  fig_avg_gap_bars.png                   Avg-Gap per method, with/without Weight Saliency, per scenario
  fig_metrics_vs_ratio.png               forget acc / test acc / MIA / Avg-Gap vs forget ratio (S2 10/30/50)
  fig_mia_roc_<scenario>.png             MIA ROC curves (one subplot per seed), AUC in legend
  fig_cost_vs_gap.png                    unlearning time vs Avg-Gap (Pareto view)
"""
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from mu import RECORDS_DIR

RES, FIG = os.path.join(RECORDS_DIR, "results"), os.path.join(RECORDS_DIR, "figures")
SCEN = ["S1c0", "S2k359p10", "S2k359p30", "S2k359p50"]
SCEN_LABEL = {"S1c0": "S1: class 0", "S2k359p10": "S2: 10% of {3,5,9}", "S2k359p30": "S2: 30% of {3,5,9}", "S2k359p50": "S2: 50% of {3,5,9}"}
METHODS = ["GA", "GAGD", "TS", "RLrand", "RLbnd"]
NAME = {"original": "Original", "retrain": "Retrain", "GA": "M1 Gradient Ascent", "GAGD": "M2 GA + Descent", "TS": "M3a Teacher-Student",
        "RLrand": "M3b-R Random Relabel", "RLbnd": "M3b-B Boundary Relabel"}
COLS = ["acc_forget", "acc_retain", "acc_test", "acc_test_forgetcls", "acc_test_othercls", "mia_acc", "mia_auc", "avg_gap", "unlearn_time_s",
        "total_time_s", "train_time_s", "peak_mem_gb"]


def load_main():
    df = pd.read_csv(os.path.join(RES, "all_runs.csv"))
    df["tag"] = df["tag"].fillna("")
    df = df[(df["tag"] == "") & df["scenario"].isin(SCEN)].copy()
    df["mask"] = df["mask"].fillna("none")
    df.loc[df["method"] == "retrain", "avg_gap"] = 0.0
    df["variant"] = df["method"].map(NAME) + np.where(df["mask"] == "saliency", " + WS", "")
    return df


def table(df):
    g = df.groupby(["scenario", "method", "mask", "variant"])[COLS]
    mean, std, n = g.mean(), g.std(), g.size().rename("n_seeds")
    t = mean.join(std, rsuffix="_std").join(n).reset_index()
    order = {m: i for i, m in enumerate(["original", "retrain"] + METHODS)}
    t["_o"] = t["method"].map(order) * 2 + (t["mask"] == "saliency")
    t["_s"] = t["scenario"].map({s: i for i, s in enumerate(SCEN)})
    t = t.sort_values(["_s", "_o"]).drop(columns=["_o", "_s"])
    t.to_csv(os.path.join(RES, "main_table.csv"), index=False)
    lines = []
    for sc in SCEN:
        s = t[t["scenario"] == sc]
        if s.empty:
            continue
        lines += [f"\n### {SCEN_LABEL[sc]}\n", "| variant | n | forget acc | retain acc | test acc | MIA acc | MIA AUC | Avg-Gap | time (s) | peak mem (GB) |",
                  "|---|---|---|---|---|---|---|---|---|---|"]
        for _, r in s.iterrows():
            f = lambda c: f"{r[c]:.2f} ± {r[c + '_std']:.2f}" if pd.notna(r[c]) else "-"
            tm = r["train_time_s"] if r["method"] == "retrain" else r["total_time_s"]
            lines.append(f"| {r['variant']} | {r['n_seeds']} | {f('acc_forget')} | {f('acc_retain')} | {f('acc_test')} | {f('mia_acc')} | {f('mia_auc')} | "
                         f"{f('avg_gap')} | {tm:.1f} | {r['peak_mem_gb']:.2f} |")
    open(os.path.join(RES, "main_table.md"), "w").write("# Main results (mean ± std over seeds)\nAvg-Gap = mean |metric - Retrain| over forget acc, retain acc, test acc, MIA acc (lower is better). WS = Gradient based Weight Saliency (mask 0.5). Time: unlearning (+ mask) time; Retrain row = full training time.\n" + "\n".join(lines) + "\n")
    return t


def saliency_gain(t):
    u = t[t["method"].isin(METHODS)].pivot_table(index=["scenario", "method"], columns="mask", values="avg_gap").reset_index()
    if "saliency" in u and "none" in u:
        u["delta_avg_gap"] = u["saliency"] - u["none"]
    u.to_csv(os.path.join(RES, "saliency_gain.csv"), index=False)
    return u


def fig_bars(t):
    scs = [s for s in SCEN if s in set(t["scenario"])]
    fig, axes = plt.subplots(1, len(scs), figsize=(4.2 * len(scs), 3.6), squeeze=False)
    for ax, sc in zip(axes[0], scs):
        s = t[(t["scenario"] == sc) & t["method"].isin(METHODS)]
        x = np.arange(len(METHODS))
        for k, (mask, col) in enumerate((("none", "tab:blue"), ("saliency", "tab:orange"))):
            v = s[s["mask"] == mask].set_index("method").reindex(METHODS)
            ax.bar(x + (k - 0.5) * 0.38, v["avg_gap"], 0.38, yerr=v["avg_gap_std"], color=col, label="with WS" if mask == "saliency" else "no WS", capsize=2)
        o = t[(t["scenario"] == sc) & (t["method"] == "original")]["avg_gap"]
        if len(o):
            ax.axhline(o.iloc[0], color="k", ls="--", lw=1, label="Original (no unlearning)")
        ax.set_xticks(x, METHODS, rotation=30)
        ax.set_title(SCEN_LABEL[sc], fontsize=10)
        ax.set_ylabel("Avg-Gap to Retrain (lower = better)")
    axes[0][0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_avg_gap_bars.png"), dpi=150)
    plt.close(fig)


def fig_ratio(t):
    s = t[t["scenario"].str.startswith("S2")].copy()
    if s.empty:
        return
    s["ratio"] = s["scenario"].str[-2:].astype(int)
    mets = [("acc_forget", "forget acc"), ("acc_test", "test acc"), ("mia_acc", "MIA acc"), ("avg_gap", "Avg-Gap")]
    fig, axes = plt.subplots(2, 4, figsize=(16, 7))
    for row, mask in enumerate(("none", "saliency")):
        for ax, (c, lab) in zip(axes[row], mets):
            for m in ["original", "retrain"] + METHODS:
                v = s[(s["method"] == m) & ((s["mask"] == mask) | s["method"].isin(["original", "retrain"]))].sort_values("ratio")
                if len(v):
                    ax.errorbar(v["ratio"], v[c], yerr=v[c + "_std"], marker="o", capsize=2, label=NAME[m],
                                ls="--" if m in ("original", "retrain") else "-")
            ax.set_xlabel("forget ratio per class (%)")
            ax.set_title(f"{lab} ({'with WS' if mask == 'saliency' else 'no WS'})", fontsize=10)
    axes[0][0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_metrics_vs_ratio.png"), dpi=150)
    plt.close(fig)


def fig_roc(df):
    for sc in SCEN:
        runs = df[df["scenario"] == sc]
        seeds = sorted(runs["seed"].unique())
        if not len(seeds):
            continue
        fig, axes = plt.subplots(2, len(seeds), figsize=(4.5 * len(seeds), 8), squeeze=False)
        for j, seed in enumerate(seeds):
            for i, mask in enumerate(("none", "saliency")):
                ax = axes[i][j]
                sel = runs[(runs["seed"] == seed) & ((runs["mask"] == mask) | runs["method"].isin(["original", "retrain"]))]
                for _, r in sel.iterrows():
                    m = json.load(open(os.path.join(RECORDS_DIR, "runs", r["run_id"], "metrics.json")))
                    if "mia" not in m:
                        continue
                    ax.plot(m["mia"]["roc_fpr_grid"], m["mia"]["roc_tpr"], label=f"{NAME[r['method']]} ({m['mia']['mia_auc']:.1f})",
                            ls="--" if r["method"] in ("original", "retrain") else "-")
                ax.plot([0, 1], [0, 1], color="gray", lw=0.5)
                ax.set_title(f"{SCEN_LABEL[sc]} seed {seed} ({'with WS' if mask == 'saliency' else 'no WS'})", fontsize=9)
                ax.set_xlabel("FPR (test predicted as forgotten)")
                ax.set_ylabel("TPR (forget identified)")
                ax.legend(fontsize=6, title="AUC", title_fontsize=6)
        fig.tight_layout()
        fig.savefig(os.path.join(FIG, f"fig_mia_roc_{sc}.png"), dpi=130)
        plt.close(fig)


def fig_cost(t):
    s = t[t["method"].isin(METHODS)]
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    mk = {"S1c0": "o", "S2k359p10": "^", "S2k359p30": "s", "S2k359p50": "D"}
    for _, r in s.iterrows():
        ax.scatter(r["total_time_s"], r["avg_gap"], marker=mk[r["scenario"]], color=plt.cm.tab10(METHODS.index(r["method"])),
                   edgecolors="k" if r["mask"] == "saliency" else "none", s=40)
    rt = t[t["method"] == "retrain"]["train_time_s"].mean()
    ax.axvline(rt, color="k", ls="--", lw=1)
    ax.text(rt, ax.get_ylim()[1] * 0.9, " Retrain", fontsize=8)
    ax.set_xscale("log")
    for i, m in enumerate(METHODS):
        ax.scatter([], [], color=plt.cm.tab10(i), label=NAME[m])
    for sc, m in mk.items():
        ax.scatter([], [], marker=m, color="gray", label=SCEN_LABEL[sc])
    ax.set_xlabel("time (s, log)")
    ax.set_ylabel("Avg-Gap")
    ax.set_title("Cost vs quality (black edge = with WS)", fontsize=10)
    ax.legend(fontsize=6, ncol=2)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_cost_vs_gap.png"), dpi=150)
    plt.close(fig)


def main():
    os.makedirs(FIG, exist_ok=True)
    df = load_main()
    t = table(df)
    saliency_gain(t)
    fig_bars(t)
    fig_ratio(t)
    fig_roc(df)
    fig_cost(t)
    print(open(os.path.join(RES, "main_table.md")).read())


if __name__ == "__main__":
    main()
