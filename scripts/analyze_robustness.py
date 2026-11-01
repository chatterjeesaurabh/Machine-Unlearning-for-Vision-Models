"""Analysis (CPU) of M5 perturbation evaluation and post-unlearning quantization for main-grid runs.
Outputs: records/results/perturb_table.csv, quant_table.csv; records/figures/fig_perturb_<scenario>.png, fig_quant_<scenario>.png
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from mu import RECORDS_DIR

RES, FIG = os.path.join(RECORDS_DIR, "results"), os.path.join(RECORDS_DIR, "figures")
SCEN = ["S1c0", "S2k359p10", "S2k359p30", "S2k359p50"]
ORDER = ["original", "retrain", "GA", "GAGD", "TS", "RLrand", "RLbnd"]


def main():
    df = pd.read_csv(os.path.join(RES, "all_runs.csv"))
    df = df[(df["tag"].fillna("") == "") & df["scenario"].isin(SCEN)].copy()
    df["mask"] = df["mask"].fillna("none")
    prow, qrow = [], []
    for _, r in df.iterrows():
        d = os.path.join(RECORDS_DIR, "runs", r["run_id"])
        base = {"scenario": r["scenario"], "seed": r["seed"], "method": r["method"], "mask": r["mask"]}
        if os.path.exists(os.path.join(d, "perturb_metrics.json")):
            for tau, v in json.load(open(os.path.join(d, "perturb_metrics.json")))["per_tau"].items():
                prow.append({**base, "tau": float(tau), **v})
        if os.path.exists(os.path.join(d, "quant_metrics.json")):
            q = json.load(open(os.path.join(d, "quant_metrics.json")))
            for k, v in q.items():
                if k.startswith("w"):
                    b = int(k[1:])
                    mech = q["mechanism"].get(f"int{b}", {})
                    qrow.append({**base, "bits": b, "acc_forget": v["acc_forget"], "acc_retain": v["acc_retain"], "acc_test": v["acc_test"],
                                 "mia": v["mia"], "rel_weight_change": q["mechanism"]["rel_weight_change"],
                                 "frac_same_as_quantized_original": mech.get("frac_same_as_quantized_original"),
                                 "mean_abs_delta_over_step": mech.get("mean_abs_delta_over_step")})
    for rows, name in ((prow, "perturb"), (qrow, "quant")):
        if not rows:
            print(f"no {name} data")
            continue
        t = pd.DataFrame(rows)
        t.to_csv(os.path.join(RES, f"{name}_raw.csv"), index=False)
        key = "tau" if name == "perturb" else "bits"
        g = t.groupby(["scenario", "method", "mask", key]).mean(numeric_only=True).drop(columns="seed").reset_index()
        g.to_csv(os.path.join(RES, f"{name}_table.csv"), index=False)
        for sc in SCEN:
            s = g[g["scenario"] == sc]
            if s.empty:
                continue
            cols = ["perturbed_forget_acc", "recovery_any"] if name == "perturb" else ["acc_forget", "acc_test", "mia"]
            fig, axes = plt.subplots(2, len(cols), figsize=(5 * len(cols), 7), squeeze=False)
            for i, mask in enumerate(("none", "saliency")):
                for j, c in enumerate(cols):
                    ax = axes[i][j]
                    for m in ORDER:
                        v = s[(s["method"] == m) & ((s["mask"] == mask) | s["method"].isin(["original", "retrain"]))].sort_values(key)
                        if len(v):
                            ax.plot(v[key], v[c], marker="o", label=m, ls="--" if m in ("original", "retrain") else "-")
                    ax.set_xlabel("noise std tau" if name == "perturb" else "weight bits (32 = fp)")
                    ax.set_title(f"{sc}: {c} ({'with WS' if mask == 'saliency' else 'no WS'})", fontsize=9)
                    if name == "quant":
                        ax.invert_xaxis()
            axes[0][0].legend(fontsize=7)
            fig.tight_layout()
            fig.savefig(os.path.join(FIG, f"fig_{name}_{sc}.png"), dpi=130)
            plt.close(fig)
        print(name, "rows", len(t))


if __name__ == "__main__":
    main()
