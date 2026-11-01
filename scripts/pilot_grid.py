"""Emit the pilot hyperparameter grid (seed 0; S1c0 and S2k359p30; with/without Weight Saliency mask 0.5) as a bash batch.
Selection criterion: sel_gap (no test data). Usage:
  python scripts/pilot_grid.py > outputs/pilot_batch.sh && nohup bash scripts/gpu_batch.sh outputs/pilot_batch.sh > outputs/logs/pilot.log 2>&1 &
Skips runs whose metrics.json already exists.
"""
import itertools
import os

from mu import RECORDS_DIR

GRID = {
    "GA": {"common": {"lr": [1e-5, 3e-5, 1e-4, 3e-4, 1e-3]}},
    "GAGD": {"common": {"lr": [0.01, 0.03], "beta": [0.95, 0.99, 0.999, 0.9999]}},
    "TS": {"class": {"alpha": [0.5, 1.0], "gamma": [0.5, 1.0]},
           "subset_random": [{"forget_batch_size": 16, "msteps": 5}, {"forget_batch_size": 64, "msteps": 2},
                             {"forget_batch_size": 64, "msteps": 5}, {"forget_batch_size": 256, "msteps": 2},
                             {"forget_batch_size": 256, "msteps": 5}]},
    "RLrand": {"common": {"lr": [0.005, 0.01, 0.02, 0.05]}},
    "RLbnd": {"common": {"lr": [1e-6, 1e-5, 1e-4, 1e-3]}},
}
SCENARIOS = {"S1c0": "class", "S2k359p30": "subset_random"}

# Stage 2 (added after stage-1 results): Teacher-Student with alpha (KL weight on retain) in {0.5, 1} did not forget:
# forget acc is 0 right after each max-epoch but back to ~100% after the min-epoch, even in S1 where the retain set has no
# class-0 images -> the teacher's soft labels (T=4) on retain images re-teach the forgotten class. Explore small KL weights.
GRID2 = {"TS": {"class": {"alpha": [0.001, 0.01, 0.1], "forget_batch_size": [128, 512]},
                "subset_random": [{"alpha": a, "forget_batch_size": b, "msteps": 2} for a in (0.001, 0.01, 0.1) for b in (64, 256)]}}


def combos(spec, sc_type):
    g = spec.get(sc_type, spec.get("common"))
    if isinstance(g, list):
        return g
    keys = list(g)
    return [dict(zip(keys, v)) for v in itertools.product(*[g[k] for k in keys])]


# Stage 3: forget-only methods (GA, Boundary Relabeling) change forget acc strongly even at lr 1e-6 -> suspected BatchNorm
# running-stat drift from forget-only batches. Same lr grids with BN statistics frozen (bn_mode=frozen).
GRID3 = {"GA": {"common": {"lr": [1e-5, 1e-4, 1e-3], "bn_mode": ["frozen"]}},
         "RLbnd": {"common": {"lr": [1e-6, 1e-5, 1e-4, 1e-3], "bn_mode": ["frozen"]}}}


# Stage 4: Teacher-Student on S2 either collapses (msteps 5, forget bs <= 64) or does not forget (msteps 2) -> intermediate.
GRID4 = {"TS": {"class": [], "subset_random": [{"alpha": 0.01, "forget_batch_size": b, "msteps": m} for b in (64, 128) for m in (3, 4)]
                + [{"alpha": 0.01, "forget_batch_size": 128, "msteps": 5}]}}


def main():
    import sys
    grid = GRID2 if "--stage2" in sys.argv else GRID3 if "--stage3" in sys.argv else GRID4 if "--stage4" in sys.argv else GRID
    print("cd /anchor/anchor7/sachatt/cs6886/project; source env.sh")
    for sc, sc_type in SCENARIOS.items():
        for method, spec in grid.items():
            for hp in combos(spec, sc_type):
                tag = "pilot_" + "_".join(f"{k.replace('_batch_size', 'bs').replace('bn_mode', 'bn')}{v:g}" if not isinstance(v, str)
                                          else f"{k.replace('bn_mode', 'bn')}{v}" for k, v in hp.items())
                hp_args = " ".join(f"{k}={v}" for k, v in hp.items())
                for mask in ("none", "saliency"):
                    mtag = "__sal0.5" if mask == "saliency" else ""
                    rid = f"{sc}__{method}{mtag}__{tag}__s0"
                    if os.path.exists(os.path.join(RECORDS_DIR, "runs", rid, "metrics.json")):
                        continue
                    print(f"echo '[run] {rid}'; python scripts/unlearn.py --scenario {sc} --seed 0 --method {method} --mask {mask} "
                          f"--mask_ratio 0.5 --hp {hp_args} --tag {tag} --save_ckpt 0 2>&1 | grep -E 'FINAL|Error|error|Traceback'")
    print("python scripts/aggregate.py; python scripts/build_index.py; echo PILOT_DONE")


if __name__ == "__main__":
    main()
