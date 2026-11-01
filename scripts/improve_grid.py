"""Stage 5 - method improvements (seed 0; tag prefix 'imp_'), emitted as a bash batch (one job at a time under the GPU lock).
Each entry: (method, scenario, hp overrides, save_ckpt). Reasoning per block is in records/JOURNAL.md (2026-10-23).
Usage: python scripts/improve_grid.py > outputs/imp_batch.sh; nohup bash scripts/gpu_batch.sh outputs/imp_batch.sh > outputs/logs/imp.log 2>&1 &
"""
import os

from mu import RECORDS_DIR

S1, S30, S50 = "S1c0", "S2k359p30", "S2k359p50"
J = []

# --- Teacher-Student: 10 epochs total, smaller distillation (KL) weight, msteps 2 or 5 (rest = extra min-steps), lr x0.1 at epoch 8
for a in (0.0, 1e-4, 1e-3, 1e-2):
    for ms in (2, 5):
        J.append(("TS", S1, dict(epochs=10, msteps=ms, alpha=a, forget_batch_size=128, retain_batch_size=128, lr_decay_epochs="[8]"), 0))
for a in (0.0, 1e-3, 1e-2):
    for ms in (2, 5):
        J.append(("TS", S30, dict(epochs=10, msteps=ms, alpha=a, forget_batch_size=128, retain_batch_size=64, lr_decay_epochs="[8]"), 0))
for fbs in (128, 256):
    J.append(("TS", S50, dict(epochs=10, msteps=2, alpha=1e-3, forget_batch_size=fbs, retain_batch_size=64, lr_decay_epochs="[8]"), 0))
# mechanism check: paper-style alpha 1 but temperature T=1 (teacher soft labels carry ~no class-0 mass at T=1)
J.append(("TS", S1, dict(alpha=1.0, kd_T=1.0, forget_batch_size=128), 0))
J.append(("TS", S1, dict(alpha=0.1, kd_T=1.0, forget_batch_size=128), 0))

# --- Gradient Ascent: RUM's 10 epochs + iteration-level early stop (checked every 2 iterations on a 1k forget subset)
for sc, stop in ((S1, 1.0), (S30, "retrain")):
    for lr in (1e-4, 3e-4, 1e-3):
        J.append(("GA", sc, dict(epochs=10, lr=lr, stop_every_iters=2, stop_forget_acc=stop), 0))
    for lr in (3e-4, 1e-3):  # BN statistics frozen + gradient clipping (lr 1e-3 frozen diverged without clipping)
        J.append(("GA", sc, dict(epochs=10, lr=lr, bn_mode="frozen", grad_clip=1.0, stop_every_iters=2, stop_forget_acc=stop), 0))
# BN-drift diagnostic: same as main-grid runs, then re-estimate BN running statistics on retain data (no weight change)
J.append(("GA", S1, dict(bn_recalib=1), 0))
J.append(("RLbnd", S1, dict(bn_recalib=1), 1))

# --- Boundary Relabeling: neighbour search {fgsm (paper), pgd (iterative, first boundary crossed), logit2} x lr x retain pass
for nb in ("fgsm", "pgd", "logit2"):
    for lr in (1e-5, 1e-4):
        J.append(("RLbnd", S1, dict(nb_search=nb, lr=lr, use_retain=0), 1))
    for lr in (1e-5, 1e-4, 1e-3):
        J.append(("RLbnd", S1, dict(nb_search=nb, lr=lr, use_retain=1), 1))
for nb in ("fgsm", "pgd"):
    for lr in (1e-5, 1e-4, 1e-3):
        J.append(("RLbnd", S30, dict(nb_search=nb, lr=lr, use_retain=1), 0))
for eps in (0.03, 0.3):  # FGSM step size (paper: "relatively large", repo 0.1)
    J.append(("RLbnd", S1, dict(nb_search="fgsm", eps=eps, lr=1e-4), 1))

# --- Random Relabeling on S2-50 (main grid: lr 0.02 chosen on S2-30 gave forget 81 +- 5.6, high variance)
for lr in (0.005, 0.01):
    J.append(("RLrand", S50, dict(lr=lr), 0))
# --- GA + Descent on S2-30: between beta 0.99 (over-forgets) and 0.999 (chosen)
for b in (0.995, 0.998):
    for lr in (0.01, 0.02):
        J.append(("GAGD", S30, dict(beta=b, lr=lr), 0))


def fmt(v):
    return f"{v:g}" if isinstance(v, float) else str(v)


def main():
    print("cd /anchor/anchor7/sachatt/cs6886/project; source env.sh")
    seen = set()
    for m, sc, hp, ck in J:
        tag = "imp_" + "_".join(f"{k.replace('_batch_size', 'bs').replace('stop_every_iters', 'stopit').replace('stop_forget_acc', 'stop').replace('lr_decay_epochs', 'dec').replace('bn_mode', 'bn')}{fmt(v).replace('[', '').replace(']', '')}" for k, v in hp.items())
        rid = f"{sc}__{m}__{tag}__s0"
        if rid in seen or os.path.exists(os.path.join(RECORDS_DIR, "runs", rid, "metrics.json")):
            continue
        seen.add(rid)
        hps = " ".join(f"'{k}={fmt(v)}'" for k, v in hp.items())
        print(f"echo '[run] {rid}'; python scripts/unlearn.py --scenario {sc} --seed 0 --method {m} --hp {hps} --tag {tag} --save_ckpt {ck} 2>&1 | grep -E 'FINAL|Traceback|Error|early stop'")
    print("python scripts/aggregate.py; python scripts/build_index.py; echo IMP_DONE")


if __name__ == "__main__":
    main()
