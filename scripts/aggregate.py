"""Collect every run in records/runs into records/results/all_runs.csv, and the pilot selection into records/results/pilot_best.csv."""
import glob
import json
import os

import pandas as pd

from mu import RECORDS_DIR

MET = ["acc_forget", "acc_excluded", "acc_retain", "acc_val", "acc_test", "acc_test_forgetcls", "acc_test_othercls", "avg_gap", "sel_gap",
       "gap_acc_forget", "gap_acc_retain", "gap_acc_test", "gap_acc_val", "gap_mia_acc", "gap_mia_val_acc",
       "unlearn_time_s", "total_time_s", "train_time_s", "mask_time_s", "peak_mem_gb", "stopped_early", "diverged", "best_epoch"]


def main():
    rows = []
    for d in sorted(glob.glob(os.path.join(RECORDS_DIR, "runs", "*"))):
        if not os.path.exists(os.path.join(d, "metrics.json")):
            continue
        c, m = json.load(open(os.path.join(d, "config.json"))), json.load(open(os.path.join(d, "metrics.json")))
        r = {"run_id": os.path.basename(d), "kind": c.get("kind"), "method": c.get("method"), "mask": c.get("mask", "none"),
             "mask_ratio": c.get("mask_ratio"), "scenario": c.get("scenario"), "seed": c.get("seed"), "tag": c.get("tag", "")}
        r.update({f"hp.{k}": (str(v) if isinstance(v, list) else v) for k, v in (c.get("hparams") or {}).items()})
        r.update({k: m.get(k) for k in MET})
        for mk in ("mia", "mia_val"):
            if mk in m:
                r[f"{mk}_acc"], r[f"{mk}_auc"] = m[mk]["mia_acc"], m[mk]["mia_auc"]
        rows.append(r)
    df = pd.DataFrame(rows)
    out = os.path.join(RECORDS_DIR, "results")
    os.makedirs(out, exist_ok=True)
    df.to_csv(os.path.join(out, "all_runs.csv"), index=False)
    pil = df[df["tag"].fillna("").str.startswith("pilot")]
    if len(pil):
        best = pil.loc[pil.groupby(["scenario", "method", "mask"])["sel_gap"].idxmin()]
        cols = ["scenario", "method", "mask", "tag", "sel_gap", "avg_gap", "acc_forget", "acc_retain", "acc_val", "acc_test", "mia_acc", "mia_val_acc", "unlearn_time_s"]
        best[cols].to_csv(os.path.join(out, "pilot_best.csv"), index=False)
        pil[cols].sort_values(["scenario", "method", "mask", "sel_gap"]).to_csv(os.path.join(out, "pilot_all.csv"), index=False)
        print(best[cols].to_string(index=False))
    print(f"all_runs.csv: {len(df)} runs")


if __name__ == "__main__":
    main()
