"""Data splits. All indices refer to the ORIGINAL 50k CIFAR-10 train order (test indices to the 10k test order).

base split : val (500/class, stratified) + train pool (45k)          -> data/splits/base.npz
scenario   : forget_idx / retain_idx (subsets of the train pool)       -> data/splits/<scenario>__s<seed>.npz
  orig            : forget = {}, retain = train pool
  S1c<c>          : forget = all pool samples of class c (seed-independent)
  S2k<K>p<pct>    : forget = random pct% of the pool samples of each class in K (seed-dependent)
"""
import json
import os

import numpy as np

from mu import SPLIT_DIR
from mu.data import load_cifar10_numpy

BASE_SPLIT_SEED = 0
VAL_PER_CLASS = 500


def make_base_split(y_train):
    rng = np.random.RandomState(BASE_SPLIT_SEED)
    val = np.concatenate([rng.choice(np.where(y_train == c)[0], VAL_PER_CLASS, replace=False) for c in range(10)])
    val = np.sort(val)
    pool = np.setdiff1d(np.arange(len(y_train)), val)
    return val, pool


def parse_scenario(name):
    if "-E" in name:  # entanglement partition / sequential step of a base scenario (files made by scripts/make_entangle_partitions.py)
        base, part = name.split("-E")
        return {**parse_scenario(base), "base": base, "partition": part}
    if name == "orig":
        return {"type": "orig"}
    if name.startswith("S1c"):
        return {"type": "class", "class": int(name[3:])}
    if name.startswith("S2k"):
        k, p = name[3:].split("p")
        return {"type": "subset_random", "classes": [int(ch) for ch in k], "pct": int(p)}
    raise ValueError(name)


def make_scenario(name, seed, y_train, pool):
    sc = parse_scenario(name)
    if sc["type"] == "orig":
        forget = np.array([], dtype=np.int64)
    elif sc["type"] == "class":
        forget = pool[y_train[pool] == sc["class"]]
    else:
        rng = np.random.RandomState(1000 + seed)
        parts = []
        for c in sc["classes"]:
            cls_idx = pool[y_train[pool] == c]
            parts.append(rng.choice(cls_idx, int(round(sc["pct"] / 100 * len(cls_idx))), replace=False))
        forget = np.sort(np.concatenate(parts))
    retain = np.setdiff1d(pool, forget)
    return sc, forget, retain


def split_path(name, seed):
    return os.path.join(SPLIT_DIR, f"{name}__s{seed}.npz")


def load_split(name, seed):
    """Returns dict with val_idx, pool_idx, forget_idx, retain_idx, scenario meta. Creates files if missing."""
    os.makedirs(SPLIT_DIR, exist_ok=True)
    base_path = os.path.join(SPLIT_DIR, "base.npz")
    _, y_train, _, _ = load_cifar10_numpy()
    if not os.path.exists(base_path):
        val, pool = make_base_split(y_train)
        np.savez(base_path, val_idx=val, pool_idx=pool)
    base = np.load(base_path)
    path = split_path(name, seed)
    if not os.path.exists(path):
        if "-E" in name:
            raise FileNotFoundError(f"{path}: run scripts/make_entangle_partitions.py first")
        sc, forget, retain = make_scenario(name, seed, y_train, base["pool_idx"])
        np.savez(path, forget_idx=forget, retain_idx=retain)
        with open(path.replace(".npz", ".json"), "w") as f:
            json.dump({"scenario": name, "seed": seed, **sc, "n_forget": int(len(forget)), "n_retain": int(len(retain)),
                       "forget_per_class": np.bincount(y_train[forget], minlength=10).tolist() if len(forget) else [0] * 10}, f, indent=1)
    s = np.load(path)
    return {"val_idx": base["val_idx"], "pool_idx": base["pool_idx"], "forget_idx": s["forget_idx"],
            "retain_idx": s["retain_idx"], "scenario": parse_scenario(name), "name": name, "seed": seed}
