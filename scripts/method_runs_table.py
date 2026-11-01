"""Every run of a method (pilot, main grid, improvement stages, TS follow-up; not mask-ratio ablations) with its key
hyperparameters and results, per scenario, best (lowest sel_gap among no-mask runs that did not diverge) in bold.
Main-grid / imp2 rows are 3-seed means; all other rows are seed 0.
Usage: python scripts/method_runs_table.py GA RLbnd TS TSjoint  -> records/results/runs_<method>.md
"""
import json
import os
import sys

import pandas as pd

from mu import RECORDS_DIR

RES = os.path.join(RECORDS_DIR, "results")
KEYS = {"GA": ["lr", "epochs", "bn_mode", "grad_clip", "bn_mix_retain", "stop_every_iters", "bn_recalib"],
        "RLbnd": ["lr", "epochs", "nb_search", "eps", "nb_steps", "use_retain", "bn_mode", "grad_clip", "bn_mix_retain", "bn_recalib"],
        "TS": ["kd_T", "alpha", "gamma", "epochs", "msteps", "forget_batch_size", "retain_batch_size", "lr", "lr_decay_epochs", "select_best"],
        "GAGD": ["lr", "beta", "epochs", "joint_forward", "bn_recalib"],
        "RLrand": ["lr", "epochs", "bn_recalib"],
        "TSjoint": ["kd_T", "alpha", "gamma", "lam_f", "epochs", "forget_batch_size", "retain_batch_size", "lr", "select_best"]}
SHORT = {"forget_batch_size": "fbs", "retain_batch_size": "rbs", "lr_decay_epochs": "lr decay ep", "stop_every_iters": "iter stop",
         "select_best": "best ep", "use_retain": "retain pass", "bn_recalib": "BN recalib", "nb_search": "nb search",
         "bn_mix_retain": "BN mix retain", "grad_clip": "clip", "nb_steps": "nb steps"}
DEFAULTS = {"GA": {"bn_mode": "train", "grad_clip": 0, "bn_mix_retain": 0, "stop_every_iters": 0, "bn_recalib": 0},
            "RLbnd": {"nb_search": "fgsm", "use_retain": 0, "bn_mode": "train", "grad_clip": 0, "bn_mix_retain": 0, "bn_recalib": 0}}
MET = ["acc_forget", "acc_retain", "acc_val", "acc_test", "acc_test_othercls", "mia_acc", "sel_gap", "avg_gap"]


def stage(tag):
    if not tag:
        return "main grid"
    for p, n in (("imp5_", "imp5 (3 seeds)"), ("pilot", "pilot"), ("imp2_", "imp2 (3 seeds)"), ("imp3_", "imp3 (3 seeds)"), ("imp4_", "imp4 (3 seeds)"), ("bnfix", "BN fix"), ("imp_", "improve"), ("tsA2", "TS A2 (Fig.5, Illusion sched.)"), ("tsA", "TS A (Fig.5 grid)"), ("tsB", "TS B (10ep best)"),
                 ("tsC", "TS C (joint)"), ("abl", "ablation")):
        if tag.startswith(p):
            return n
    return tag


def main():
    df = pd.read_csv(os.path.join(RES, "all_runs.csv"))
    df["tag"] = df["tag"].fillna("")
    df["mask"] = df["mask"].fillna("none")
    for method in sys.argv[1:]:
        d = df[(df["method"] == method) & ~df["tag"].str.startswith("abl")].copy()
        if d.empty:
            continue
        hps = []
        for rid in d["run_id"]:
            h = json.load(open(os.path.join(RECORDS_DIR, "runs", rid, "config.json"))).get("hparams") or {}
            h = {**DEFAULTS.get(method, {}), **h}
            if method == "RLbnd":
                h["nb_steps"] = h.get("nb_steps", 10) if h["nb_search"] == "pgd" else "-"
                if h["nb_search"] == "logit2":
                    h["eps"] = "-"
            hps.append({k: (str(h.get(k)) if isinstance(h.get(k), list) else h.get(k)) for k in KEYS[method]})
        d = pd.concat([d.reset_index(drop=True), pd.DataFrame(hps)], axis=1)
        d["stage"] = d["tag"].map(stage)
        d["WS"] = (d["mask"] == "saliency").map({True: "yes", False: ""})
        grp = ["scenario", "stage", "tag", "WS"] + KEYS[method]
        agg = d.groupby(grp, dropna=False).agg({**{m: "mean" for m in MET}, "seed": "count", "diverged": "max", "best_epoch": "first"}).reset_index()
        lines = [f"# All {method} runs (seed 0 unless n_seeds > 1; best per scenario = lowest sel_gap among no-mask, non-diverged runs, in bold)",
                 "sel_gap = mean |metric - Retrain| over forget/retain/val acc + MIA(val negatives); avg_gap = same with test acc + MIA(test negatives).", ""]
        refs = df[df["method"].isin(["retrain", "original"]) & (df["tag"] == "")].groupby(["scenario", "method"])[MET[:6]].mean()
        for sc, s in agg.groupby("scenario"):
            s = s.sort_values("sel_gap")
            ok = s[(s["WS"] == "") & (s["diverged"] != True)]
            best = ok["sel_gap"].idxmin() if len(ok) else None
            cols = ["stage", "WS"] + KEYS[method] + ["n_seeds", "best_epoch"] + MET
            lines += [f"## {sc}", ""]
            for mname in ("retrain", "original"):
                if (sc, mname) in refs.index:
                    r = refs.loc[(sc, mname)]
                    lines.append(f"- {mname}: forget {r['acc_forget']:.2f}, retain {r['acc_retain']:.2f}, val {r['acc_val']:.2f}, test {r['acc_test']:.2f}, MIA {r['mia_acc']:.2f}")
            lines += ["", "| " + " | ".join(SHORT.get(c, c) for c in cols) + " |", "|" + "---|" * len(cols)]
            for i, r in s.iterrows():
                vals = []
                for c in cols:
                    v = r["seed"] if c == "n_seeds" else r.get(c)
                    vals.append("" if pd.isna(v) else (f"{v:.2f}" if isinstance(v, float) and c in MET else (f"{v:g}" if isinstance(v, float) else str(v))))
                if i == best:
                    vals = [f"**{v}**" if v else "" for v in vals]
                lines.append("| " + " | ".join(vals) + " |")
            lines.append("")
        open(os.path.join(RES, f"runs_{method}.md"), "w").write("\n".join(lines) + "\n")
        print(f"runs_{method}.md: {len(agg)} rows")


if __name__ == "__main__":
    main()
