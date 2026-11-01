"""Analysis (CPU) of post-unlearning quantization on the BEST configuration of every method (src/mu/best_configs.py),
all scenarios, 3 seeds, with / without Weight Saliency, plus Original and Retrain. Reads records/runs/<run>/quant_metrics.json.
Outputs: records/results/quant_best_raw.csv (per run x bits), quant_best_table.csv (mean over seeds), quant_best.md (tables),
records/figures/fig_quant_best_<scenario>.png
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from mu import RECORDS_DIR
from mu.best_configs import METHODS, SCENARIOS, best_runs, conf_name

RES, FIG = os.path.join(RECORDS_DIR, "results"), os.path.join(RECORDS_DIR, "figures")
ORDER = ["original", "retrain"] + METHODS
NAME = {"original": "Original", "retrain": "Retrain", "GA": "M1 Gradient Ascent", "GAGD": "M2 GA + Descent", "TS": "M3a Teacher-Student",
        "RLrand": "M3b-R Random Relabeling", "RLbnd": "M3b-B Boundary Relabeling"}


def load_rows():
    rows, missing = [], []
    for sc, m, mask, seed, tag, rid in best_runs():
        f = os.path.join(RECORDS_DIR, "runs", rid, "quant_metrics.json")
        if not os.path.exists(f):
            missing.append(rid)
            continue
        q = json.load(open(f))
        base = {"scenario": sc, "method": m, "mask": mask, "seed": seed, "tag": tag, "config": conf_name(m, sc) if m in METHODS else "", "run_id": rid}
        for k, v in q.items():
            if k.startswith("w"):
                b = int(k[1:])
                mech = q["mechanism"].get(f"int{b}", {})
                rows.append({**base, "bits": b, "acc_forget": v["acc_forget"], "acc_retain": v["acc_retain"], "acc_test": v["acc_test"], "mia": v["mia"],
                             "rel_weight_change": q["mechanism"]["rel_weight_change"], "frac_weights_changed": q["mechanism"].get("frac_weights_changed"),
                             "frac_same_as_quantized_original": mech.get("frac_same_as_quantized_original"), "mean_abs_delta_over_step": mech.get("mean_abs_delta_over_step")})
    return pd.DataFrame(rows), missing


def md_table(g, sc):
    L = ["| model | WS | config | forget fp32 | forget INT8 | forget INT6 | forget INT4 | retain fp32 | retain INT4 | test fp32 | test INT4 | MIA fp32 | MIA INT4 | share Q(theta_u)=Q(theta_o), INT4 | mean abs(delta)/step, INT4 |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    s = g[g.scenario == sc]
    for m in ORDER:
        for mask in (("none",) if m in ("original", "retrain") else ("none", "saliency")):
            a = s[(s.method == m) & (s["mask"] == mask)].set_index("bits")
            if a.empty:
                continue
            cfg = a.iloc[0]["config"]
            L.append(f"| {NAME[m]} | {'yes' if mask == 'saliency' else ''} | {cfg} | {a.loc[32, 'acc_forget']:.2f} | {a.loc[8, 'acc_forget']:.2f} | {a.loc[6, 'acc_forget']:.2f} | "
                     f"{a.loc[4, 'acc_forget']:.2f} | {a.loc[32, 'acc_retain']:.2f} | {a.loc[4, 'acc_retain']:.2f} | {a.loc[32, 'acc_test']:.2f} | {a.loc[4, 'acc_test']:.2f} | "
                     f"{a.loc[32, 'mia']:.2f} | {a.loc[4, 'mia']:.2f} | {a.loc[4, 'frac_same_as_quantized_original']:.3f} | {a.loc[4, 'mean_abs_delta_over_step']:.3f} |")
    return L


def delta_table(g):
    """INT4 minus fp32, per scenario / method / WS (mean over seeds): does quantization move the model towards or away from Retrain?"""
    L = ["| scenario | model | WS | config | d forget acc (INT4 - fp32) | d retain acc | d test acc | d MIA | Retrain forget acc fp32 | avg_gap fp32 | avg_gap INT4 |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    for sc in SCENARIOS:
        s = g[g.scenario == sc]
        ref = s[(s.method == "retrain")].set_index("bits")
        for m in METHODS:
            for mask in ("none", "saliency"):
                a = s[(s.method == m) & (s["mask"] == mask)].set_index("bits")
                if a.empty:
                    continue
                gap = lambda b: sum(abs(a.loc[b, c] - ref.loc[32, c]) for c in ("acc_forget", "acc_retain", "acc_test", "mia")) / 4
                L.append(f"| {sc} | {NAME[m]} | {'yes' if mask == 'saliency' else ''} | {a.iloc[0]['config']} | {a.loc[4, 'acc_forget'] - a.loc[32, 'acc_forget']:+.2f} | "
                         f"{a.loc[4, 'acc_retain'] - a.loc[32, 'acc_retain']:+.2f} | {a.loc[4, 'acc_test'] - a.loc[32, 'acc_test']:+.2f} | {a.loc[4, 'mia'] - a.loc[32, 'mia']:+.2f} | "
                         f"{ref.loc[32, 'acc_forget']:.2f} | {gap(32):.2f} | {gap(4):.2f} |")
    return L


def main():
    t, missing = load_rows()
    if t.empty:
        print("no quant data for best configs")
        return
    t.to_csv(os.path.join(RES, "quant_best_raw.csv"), index=False)
    g = t.groupby(["scenario", "method", "mask", "config", "bits"]).mean(numeric_only=True).drop(columns="seed").reset_index()
    g.to_csv(os.path.join(RES, "quant_best_table.csv"), index=False)
    n_seeds = t.groupby(["scenario", "method", "mask"])["seed"].nunique()
    lines = ["# Post-unlearning quantization on the best configuration of every method",
             "Protocol (`scripts/eval_quant.py`): simulated weight-only per-output-channel symmetric quantization of every Conv/Linear weight to 8 / 6 / 4 bits;",
             "BatchNorm parameters and running statistics kept in fp32; accuracy and MIA re-evaluated. Mechanism columns: share of weights whose INT4 value equals the",
             "quantized Original's, and mean |theta_u - theta_o| / INT4 step. Configurations are the best ones from `records/results/method_comparison.md`",
             "(`src/mu/best_configs.py`); mean over seeds (" + f"{int(n_seeds.min())}-{int(n_seeds.max())} seeds per row). Generated by `scripts/analyze_quant_best.py`.", ""]
    if missing:
        lines += [f"Missing quant_metrics.json for {len(missing)} run(s): " + ", ".join(missing), ""]
    for sc in SCENARIOS:
        lines += [f"## {sc}"] + md_table(g, sc) + [""]
    lines += ["## Effect of INT4 (INT4 minus fp32) and distance to Retrain", "avg_gap = mean |metric - Retrain fp32| over forget acc, retain acc, test acc, MIA (Retrain of the same scenario, mean over seeds).", ""] + delta_table(g) + [""]
    open(os.path.join(RES, "quant_best.md"), "w").write("\n".join(lines))
    for sc in SCENARIOS:
        s = g[g.scenario == sc]
        if s.empty:
            continue
        cols = ["acc_forget", "acc_retain", "acc_test", "mia"]
        fig, axes = plt.subplots(2, len(cols), figsize=(5 * len(cols), 7), squeeze=False)
        for i, mask in enumerate(("none", "saliency")):
            for j, c in enumerate(cols):
                ax = axes[i][j]
                for m in ORDER:
                    v = s[(s.method == m) & ((s["mask"] == mask) | s.method.isin(["original", "retrain"]))].sort_values("bits")
                    if len(v):
                        lab = NAME[m] + (f" ({v.iloc[0]['config']})" if m in METHODS else "")
                        ax.plot(v["bits"], v[c], marker="o", label=lab, ls="--" if m in ("original", "retrain") else "-")
                ax.set_xlabel("weight bits (32 = fp)")
                ax.set_title(f"{sc}: {c} ({'with WS' if mask == 'saliency' else 'no WS'}), best configs", fontsize=9)
                ax.invert_xaxis()
        axes[0][0].legend(fontsize=6)
        fig.tight_layout()
        fig.savefig(os.path.join(FIG, f"fig_quant_best_{sc}.png"), dpi=130)
        plt.close(fig)
    print("\n".join(lines))
    print(f"\nrows {len(t)}; missing {len(missing)}")


if __name__ == "__main__":
    main()
