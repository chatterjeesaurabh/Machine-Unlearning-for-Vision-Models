"""Run-table helper for the entanglement experiments.
Exp 1 uses tag 'entpart' (scenarios <base>-E<part>, other partitions in the retain set) for the subset scenarios and, for
single-class forgetting, the corrected setup tag 'entpart2' (scenarios S1c0-Ex<part>: other partitions excluded from forget AND
retain, reference = S1 Retrain evaluated on the partition). The superseded S1 'entpart' runs and their per-partition Retrains are
dropped here; afterwards both setups appear as tag 'entpart', base '<base>', part 'low' | 'medium' | 'high'."""
import pandas as pd

PARTS = ["low", "medium", "high"]
CLASS_BASES = ("S1c0",)


def prep_ent(df):
    df = df.copy()
    df["tag"] = df["tag"].fillna("")
    df["mask"] = df["mask"].fillna("none")
    df["base"] = df["scenario"].str.split("-E").str[0]
    raw = df["scenario"].str.split("-E").str[1]
    df["excl"] = raw.fillna("").str.startswith("x")
    df["part"] = raw.str.replace(r"^x", "", regex=True)
    old = df["base"].isin(CLASS_BASES) & df["part"].isin(PARTS) & ~df["excl"]
    df = df[~old].copy()
    df.loc[df["tag"] == "entpart2", "tag"] = "entpart"
    return df
