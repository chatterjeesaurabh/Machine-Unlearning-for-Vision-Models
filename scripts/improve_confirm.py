"""Stage 5b - confirm the improved configurations from Stage 5 on seeds 0-2, with and without Weight Saliency (mask 0.5,
same hyperparameters), on the scenarios where they improved. Tag 'imp2_<name>'. Checkpoints saved (robustness evals later).
Usage: python scripts/improve_confirm.py > outputs/imp2_batch.sh; nohup bash scripts/gpu_batch.sh outputs/imp2_batch.sh > outputs/logs/imp2.log 2>&1 &
"""
import os

from mu import RECORDS_DIR

CONF = [
    # name, method, scenarios, hp
    ("bndlogit2", "RLbnd", ["S1c0"], dict(nb_search="logit2", lr=1e-3, use_retain=1)),
    ("bndpgd", "RLbnd", ["S1c0"], dict(nb_search="pgd", lr=1e-4, use_retain=0)),
    ("gaitstop", "GA", ["S1c0"], dict(epochs=10, lr=1e-3, stop_every_iters=2, stop_forget_acc=1.0)),
    ("gaitstop", "GA", ["S2k359p10", "S2k359p30", "S2k359p50"], dict(epochs=10, lr=1e-4, stop_every_iters=2, stop_forget_acc="retrain")),
    ("ts10ep", "TS", ["S2k359p10", "S2k359p30", "S2k359p50"], dict(epochs=10, msteps=5, alpha=1e-3, forget_batch_size=128, retain_batch_size=64, lr_decay_epochs="[8]")),
    ("rllr01", "RLrand", ["S2k359p50"], dict(lr=0.01)),
]


def fmt(v):
    return f"{v:g}" if isinstance(v, float) else str(v)


def main():
    print("cd /anchor/anchor7/sachatt/cs6886/project; source env.sh")
    for seed in (0, 1, 2):
        for name, m, scs, hp in CONF:
            for sc in scs:
                for mask in ("none", "saliency"):
                    rid = f"{sc}__{m}" + ("__sal0.5" if mask == "saliency" else "") + f"__imp2_{name}__s{seed}"
                    if os.path.exists(os.path.join(RECORDS_DIR, "runs", rid, "metrics.json")):
                        continue
                    hps = " ".join(f"'{k}={fmt(v)}'" for k, v in hp.items())
                    print(f"echo '[run] {rid}'; python scripts/unlearn.py --scenario {sc} --seed {seed} --method {m} --mask {mask} --mask_ratio 0.5 "
                          f"--hp {hps} --tag imp2_{name} 2>&1 | grep -E 'FINAL|Traceback|Error'")
    print("python scripts/aggregate.py; python scripts/build_index.py; echo IMP2_DONE")


if __name__ == "__main__":
    main()
