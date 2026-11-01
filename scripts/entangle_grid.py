"""Entanglement experiments with each method's best configuration (records/results/method_comparison.md), +/- Weight Saliency.
Partitions: scripts/make_entangle_partitions.py (k=50 mean distance to retain, thirds low / medium / high).

Exp 1 (tag 'entpart', RUM protocol): forget = one partition, retain = everything else; Retrain reference per partition
       (train.py on <base>-E<part>, 182 epochs). Also the Original evaluated on each partition.
Exp 2 (tag 'entseq'): sequential low -> medium -> high; step k starts from the step k-1 checkpoint (teacher / neighbour model /
       saliency mask = the model at the start of the step); reference = base-scenario Retrain. Final model evaluated on the WHOLE
       forget set of the base scenario (run '<base>__seqfinal_<method>[_WS]__entseqfinal__s<seed>').
Teacher-Student forget batch scaled by 1/3 (partition = 1/3 of the forget set) to keep the number of forget updates per epoch of
the confirmed configuration. Emits one batch per seed (retrains -> exp 1 -> exp 2).
Usage: python scripts/entangle_grid.py SEED > outputs/ent_batch_sSEED.sh
"""
import os
import sys

from mu import CKPT_DIR, RECORDS_DIR

SCEN = ["S1c0", "S2k359p10", "S2k359p30", "S2k359p50"]
PARTS = ["low", "medium", "high"]
BEST = {  # method -> scenario type -> hp overrides (strings as passed to --hp)
    "GA": {"class": ["epochs=10", "lr=0.0003", "stop_every_iters=2", "stop_forget_acc=retrain", "stop_floor=1", "bn_mix_retain=1"],   # G3
           "subset": ["epochs=10", "lr=0.0001", "stop_every_iters=2", "stop_forget_acc=retrain", "bn_mix_retain=1"]},                  # G3-S2
    "GAGD": {"class": ["lr=0.01", "beta=0.99"], "subset": ["lr=0.01", "beta=0.999"]},                                                 # D1 / D1-S2
    "TS": {"class": ["epochs=3", "msteps=2", "alpha=0.1", "gamma=1", "kd_T=4", "forget_batch_size=43", "retain_batch_size=128", "lr_decay_epochs=[3]"],  # T1
           "subset": ["epochs=3", "msteps=2", "alpha=0.1", "gamma=0.1", "kd_T=4", "forget_batch_size={tsfbs}", "retain_batch_size=64", "lr_decay_epochs=[3]"]},  # T4-best S2
    "RLrand": {"class": ["lr=0.005"], "subset": ["lr=0.02"]},                                                                          # R1 / R1-S2
    "RLbnd": {"class": ["nb_search=logit2", "lr=0.001", "use_retain=1", "bn_mix_retain=1"],                                            # C1-mix
              "subset": ["nb_search=fgsm", "eps=0.1", "lr=0.00001"]},                                                                   # C0 (no good S2 config)
}
TS_FBS = {"S2k359p10": 21, "S2k359p30": 21, "S2k359p50": 43}  # 64/3, 64/3, 128/3


def done(rid):
    return os.path.exists(os.path.join(RECORDS_DIR, "runs", rid, "metrics.json"))


def hp_for(m, sc):
    hp = BEST[m]["class" if sc.startswith("S1") else "subset"]
    return " ".join(f"'{h.format(tsfbs=TS_FBS.get(sc, 43))}'" for h in hp)


def main():
    seed = int(sys.argv[1])
    G = "| grep -E 'FINAL|Traceback|Error'"
    L = ["cd /anchor/anchor7/sachatt/cs6886/project; source env.sh"]
    for sc in SCEN:  # retrain references for exp 1
        for p in PARTS:
            rid = f"{sc}-E{p}__retrain__s{seed}"
            if not done(rid):
                L.append(f"echo '[retrain] {rid}'; python scripts/train.py --scenario {sc}-E{p} --seed {seed} --gpu 1 --mem_frac 0.45 2>&1 | grep -E 'FINAL|MIA|Traceback'")
    for sc in SCEN:  # exp 1
        for p in PARTS:
            s = f"{sc}-E{p}"
            if not done(f"{s}__original__entpart__s{seed}"):
                L.append(f"python scripts/unlearn.py --scenario {s} --seed {seed} --method none --tag entpart 2>&1 {G}")
            for m in BEST:
                for mask in ("none", "saliency"):
                    rid = f"{s}__{m}" + ("__sal0.5" if mask == "saliency" else "") + f"__entpart__s{seed}"
                    if not done(rid):
                        L.append(f"echo '[exp1] {rid}'; python scripts/unlearn.py --scenario {s} --seed {seed} --method {m} --mask {mask} --mask_ratio 0.5 --hp {hp_for(m, sc)} --tag entpart --save_ckpt 0 2>&1 {G}")
    for sc in SCEN:  # exp 2
        for m in BEST:
            for mask in ("none", "saliency"):
                ms = "__sal0.5" if mask == "saliency" else ""
                prev = os.path.join(CKPT_DIR, "orig__original__s0.pt")
                for k in (1, 2, 3):
                    rid = f"{sc}-Eseq{k}__{m}{ms}__entseq__s{seed}"
                    if not done(rid):
                        L.append(f"echo '[exp2] {rid}'; python scripts/unlearn.py --scenario {sc}-Eseq{k} --seed {seed} --method {m} --mask {mask} --mask_ratio 0.5 "
                                 f"--hp {hp_for(m, sc)} --tag entseq --ref_scenario {sc} --orig_ckpt {prev} 2>&1 {G}")
                    prev = os.path.join(CKPT_DIR, f"{rid}.pt")
                label = f"seqfinal_{m}" + ("_WS" if mask == "saliency" else "")
                if not done(f"{sc}__{label}__entseqfinal__s{seed}"):
                    L.append(f"python scripts/unlearn.py --scenario {sc} --seed {seed} --method none --orig_ckpt {prev} --eval_label {label} --tag entseqfinal 2>&1 {G}")
    L.append(f"python scripts/aggregate.py; python scripts/build_index.py; echo ENT_S{seed}_DONE")
    print("\n".join(L))


if __name__ == "__main__":
    main()
