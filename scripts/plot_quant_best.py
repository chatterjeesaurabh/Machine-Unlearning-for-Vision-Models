"""Figures for the post-unlearning quantization of the best configurations.
Inputs: records/results/quant_best_table.csv (mean over 3 seeds per scenario / method / mask / bits) and
        records/results/quant_margin_diag.md (INT4 margin diagnostic, S2-30 seed 0).
Outputs (records/figures/):
  fig_quant_best_forget_vs_bits.png  forget acc vs bit width, one panel per scenario
  fig_quant_best_int4_delta.png      INT4 - fp32 change of forget / retain / test acc and MIA per method, one panel per scenario
  fig_quant_best_avg_gap.png         avg_gap to Retrain (fp32 Retrain) at fp32 vs INT4
  fig_quant_best_mechanism.png       INT4 forget-acc drop vs mean |theta_u - theta_o| / INT4 step (S2 scenarios)
  fig_quant_margin_diag.png          INT4 margin diagnostic (flipped share, low-margin share)
Usage: python scripts/plot_quant_best.py
"""
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from mu import RECORDS_DIR

RES, FIG = os.path.join(RECORDS_DIR, "results"), os.path.join(RECORDS_DIR, "figures")
SC = ["S1c0", "S2k359p10", "S2k359p30", "S2k359p50"]
SCN = {"S1c0": "S1 (class 0)", "S2k359p10": "S2-10", "S2k359p30": "S2-30", "S2k359p50": "S2-50"}
M = ["GA", "GAGD", "TS", "RLrand", "RLbnd"]
MN = {"GA": "Gradient Ascent", "GAGD": "GA + Descent", "TS": "Teacher-Student", "RLrand": "Random Relabeling",
      "RLbnd": "Boundary Relabeling", "retrain": "Retrain", "original": "Original"}
COL = {"GA": "tab:blue", "GAGD": "tab:orange", "TS": "tab:green", "RLrand": "tab:red", "RLbnd": "tab:purple",
       "retrain": "black", "original": "gray"}
BITS = [32, 8, 6, 4]

t = pd.read_csv(os.path.join(RES, "quant_best_table.csv"))
t["mask"] = t["mask"].fillna("none")


def row(sc, m, mask, b):
    r = t[(t.scenario == sc) & (t.method == m) & (t["mask"] == mask) & (t.bits == b)]
    return r.iloc[0] if len(r) else None


def avg_gap(sc, m, mask, b):
    r, ref = row(sc, m, mask, b), row(sc, "retrain", "none", 32)
    return np.mean([abs(r[c] - ref[c]) for c in ("acc_forget", "acc_retain", "acc_test", "mia")])


# 1. forget acc vs bits
fig, axes = plt.subplots(1, 4, figsize=(22, 4.6))
for ax, sc in zip(axes, SC):
    for m in ["original", "retrain"] + M:
        for mask, ls in (("none", "-"), ("saliency", "--")):
            if m in ("original", "retrain") and mask == "saliency":
                continue
            y = [row(sc, m, mask, b)["acc_forget"] for b in BITS]
            ax.plot(range(4), y, ls, marker="o", color=COL[m], label=MN[m] + (" + WS" if mask == "saliency" else ""))
    ax.set_xticks(range(4), ["fp32", "INT8", "INT6", "INT4"])
    ax.set_title(SCN[sc]); ax.set_ylabel("forget acc (%)"); ax.grid(alpha=0.3)
    if sc != "S1c0":
        ax.set_ylim(60, 101)
axes[0].set_ylim(-2, 102)
axes[-1].legend(fontsize=7, loc="lower left", ncol=2)
fig.suptitle("Forget accuracy after weight-only quantization (best configurations, 3 seeds; dashed = with Weight Saliency)")
fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig_quant_best_forget_vs_bits.png"), dpi=130); plt.close(fig)

# 2. INT4 deltas
fig, axes = plt.subplots(1, 4, figsize=(22, 4.6), sharey=True)
mets = [("acc_forget", "forget acc"), ("acc_retain", "retain acc"), ("acc_test", "test acc"), ("mia", "MIA")]
for ax, sc in zip(axes, SC):
    models = ["original", "retrain"] + M
    x = np.arange(len(models)); w = 0.2
    for k, (c, lab) in enumerate(mets):
        d = [row(sc, m, "none", 4)[c] - row(sc, m, "none", 32)[c] for m in models]
        ax.bar(x + (k - 1.5) * w, d, w, label=lab)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(x, [MN[m] for m in models], rotation=40, ha="right", fontsize=8)
    ax.set_title(SCN[sc]); ax.grid(axis="y", alpha=0.3)
axes[0].set_ylabel("INT4 - fp32 (percentage points)")
axes[0].legend(fontsize=8)
fig.suptitle("Change caused by INT4 weight quantization (best configurations without WS, 3 seeds)")
fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig_quant_best_int4_delta.png"), dpi=130); plt.close(fig)

# 3. avg_gap fp32 vs INT4
fig, axes = plt.subplots(1, 4, figsize=(22, 4.6))
for ax, sc in zip(axes, SC):
    labels, g32, g4 = [], [], []
    for m in M:
        for mask in ("none", "saliency"):
            labels.append(MN[m] + (" + WS" if mask == "saliency" else ""))
            g32.append(avg_gap(sc, m, mask, 32)); g4.append(avg_gap(sc, m, mask, 4))
    x = np.arange(len(labels))
    ax.bar(x - 0.2, g32, 0.4, label="fp32"); ax.bar(x + 0.2, g4, 0.4, label="INT4")
    ax.set_xticks(x, labels, rotation=50, ha="right", fontsize=7)
    ax.set_title(SCN[sc]); ax.set_ylabel("avg_gap to Retrain (lower = better)"); ax.grid(axis="y", alpha=0.3)
axes[0].legend()
fig.suptitle("avg_gap to the fp32 Retrain model before and after INT4 quantization (best configurations, 3 seeds)")
fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig_quant_best_avg_gap.png"), dpi=130); plt.close(fig)

# 4. mechanism: INT4 forget drop vs update size (S2)
fig, ax = plt.subplots(figsize=(7.5, 5))
for sc, mk in zip(SC[1:], ("o", "s", "^")):
    for m in M + ["retrain"]:
        for mask in (("none", "saliency") if m in M else ("none",)):
            r4, r32 = row(sc, m, mask, 4), row(sc, m, mask, 32)
            ax.scatter(r4["mean_abs_delta_over_step"], r4["acc_forget"] - r32["acc_forget"], color=COL[m], marker=mk,
                       facecolors="none" if mask == "saliency" else COL[m], s=55)
ax.set_xscale("log"); ax.axhline(0, color="k", lw=0.8)
ax.set_xlabel("mean |theta_u - theta_o| / INT4 step (log)"); ax.set_ylabel("forget acc INT4 - fp32 (points)")
h = [plt.Line2D([], [], color=COL[m], marker="o", ls="", label=MN[m]) for m in M + ["retrain"]]
h += [plt.Line2D([], [], color="k", marker=mk, ls="", label=SCN[sc]) for sc, mk in zip(SC[1:], ("o", "s", "^"))]
h += [plt.Line2D([], [], color="k", marker="o", mfc="none", ls="", label="with WS (hollow)")]
ax.legend(handles=h, fontsize=7, loc="lower right"); ax.grid(alpha=0.3)
ax.set_title("S2: INT4 forget-acc drop vs size of the unlearning update")
fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig_quant_best_mechanism.png"), dpi=130); plt.close(fig)

# 5. margin diagnostic
txt = open(os.path.join(RES, "quant_margin_diag.md")).read().splitlines()
rows = [[c.strip() for c in l.strip("|").split("|")] for l in txt if l.startswith("| ") and not l.startswith("| model") and not re.match(r"\|\s*-", l)]
d = pd.DataFrame(rows, columns=["model", "set", "acc32", "acc4", "flipped", "med_all", "lowmargin", "med_flipped", "med_kept"])
for c in d.columns[2:]:
    d[c] = d[c].astype(float)
fig, axes = plt.subplots(1, 2, figsize=(13, 4.4))
models = list(dict.fromkeys(d.model))
x = np.arange(len(models))
for ax, col, lab in ((axes[0], "lowmargin", "% low-margin (p_y < 0.9) among correct, fp32"), (axes[1], "flipped", "% flipped by INT4 (correct fp32 -> wrong INT4)")):
    for k, s in enumerate(("forget", "same-class test")):
        v = [d[(d.model == m) & (d.set == s)][col].iloc[0] for m in models]
        ax.bar(x + (k - 0.5) * 0.38, v, 0.38, label=s)
    ax.set_xticks(x, models); ax.set_title(lab); ax.grid(axis="y", alpha=0.3); ax.legend()
fig.suptitle("INT4 margin diagnostic, S2-30 seed 0 (best configurations): unlearning leaves forget-class samples near the boundary; INT4 flips them")
fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig_quant_margin_diag.png"), dpi=130); plt.close(fig)
print("figures written")
