"""Stage 5 analysis (CPU): every 'imp_' run vs the main-grid seed-0 run of the same method (no mask) and vs Retrain / Original.
Columns: forget / retain / val / test acc, MIA acc, sel_gap (selection; no test data), avg_gap (report), time.
Also 'removal view' for S2: the user's criterion (low forget acc with retain/test close to Original).
Outputs: records/results/improve_table.md / improve_table.csv
"""
import os

import pandas as pd

from mu import RECORDS_DIR

RES = os.path.join(RECORDS_DIR, "results")
C = ["acc_forget", "acc_retain", "acc_val", "acc_test", "acc_test_othercls", "mia_acc", "sel_gap", "avg_gap", "unlearn_time_s", "stopped_early", "diverged"]


def main():
    df = pd.read_csv(os.path.join(RES, "all_runs.csv"))
    df["tag"] = df["tag"].fillna("")
    df["mask"] = df["mask"].fillna("none")
    imp = df[df["tag"].str.startswith("imp_")]
    rows, lines = [], ["# Stage 5: method improvements (seed 0)", "",
                       "sel_gap / avg_gap: mean |metric - Retrain| (lower = closer to Retrain). Baseline = main-grid run (seed 0, no mask). "
                       "For S2 the Retrain forget acc is ~89-92%, so very low forget acc is over-unlearning from the privacy view.", ""]
    for (sc, m), g in imp.groupby(["scenario", "method"]):
        base = df[(df["scenario"] == sc) & (df["method"] == m) & (df["tag"] == "") & (df["mask"] == "none") & (df["seed"] == 0)]
        refs = df[(df["scenario"] == sc) & df["method"].isin(["retrain", "original"]) & (df["tag"] == "") & (df["seed"] == 0)]
        t = pd.concat([refs.assign(variant=refs["method"]), base.assign(variant="main-grid"), g.assign(variant=g["tag"])])
        t = pd.concat([t[t["variant"].isin(["retrain", "original", "main-grid"])], t[~t["variant"].isin(["retrain", "original", "main-grid"])].sort_values("sel_gap")])
        t["scenario"], t["method_"] = sc, m
        rows.append(t)
        lines += [f"## {m} - {sc}", "", "| variant | forget | retain | val | test | test other-cls | MIA | sel_gap | avg_gap | time s | early stop | diverged |",
                  "|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for _, r in t.iterrows():
            f = lambda c: "-" if pd.isna(r.get(c)) else (f"{r[c]:.2f}" if isinstance(r[c], float) else str(r[c]))
            lines.append(f"| {r['variant']} | " + " | ".join(f(c) for c in C) + " |")
        lines.append("")
    pd.concat(rows).to_csv(os.path.join(RES, "improve_table.csv"), index=False)
    open(os.path.join(RES, "improve_table.md"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
