"""BatchNorm follow-up (seed 0; tag prefix 'bnfix_') for the two forget-only methods.
Controls (same lr / epochs / early stop, one factor changed at a time) and candidate fixes:
  bn_mode=frozen       BN layers in eval mode during unlearning (running stats untouched; forgetting must come from weights)
  grad_clip=1          gradient-norm clipping (stability only)
  bn_mix_retain=1      forget batch forwarded together with a retain batch (BN sees representative statistics; loss on forget only)
  bn_recalib=1         robustness check: after unlearning, re-estimate BN running stats on retain data (no weight change).
                       If forget acc comes back, the forgetting lived in the BN statistics.
Usage: python scripts/bnfix_grid.py > outputs/bnfix_batch.sh; nohup bash scripts/gpu_batch.sh outputs/bnfix_batch.sh > outputs/logs/bnfix.log 2>&1 &
"""
import os

from mu import RECORDS_DIR

S1, S30 = "S1c0", "S2k359p30"
J = []
for sc, stop, best_lr in ((S1, 1.0, 1e-3), (S30, "retrain", 1e-4)):
    base = dict(epochs=10, stop_every_iters=2, stop_forget_acc=stop)
    for lr in (3e-4, 1e-3):
        J.append(("GA", sc, dict(base, lr=lr, bn_mode="frozen")))           # frozen, no clip (control for imp_ frozen+clip)
        J.append(("GA", sc, dict(base, lr=lr, grad_clip=1.0)))              # clip only
    for lr in (1e-4, 3e-4, 1e-3):
        J.append(("GA", sc, dict(base, lr=lr, bn_mix_retain=1)))            # mixed-batch BN
    J.append(("GA", sc, dict(base, lr=1e-3, bn_mix_retain=1, grad_clip=1.0)))
    J.append(("GA", sc, dict(base, lr=best_lr, bn_recalib=1)))              # robustness of the imp_ best
    J.append(("GA", sc, dict(base, lr=1e-3, bn_mode="frozen", grad_clip=1.0, bn_recalib=1)))
    J.append(("GA", sc, dict(base, lr=1e-3, bn_mix_retain=1, bn_recalib=1)))

BND = {"C0_fgsm0.1": dict(nb_search="fgsm", eps=0.1, lr=1e-5),                 # main grid / paper
       "C1_logit2_retain": dict(nb_search="logit2", lr=1e-3, use_retain=1),    # imp best (S1)
       "C2_pgd": dict(nb_search="pgd", eps=0.1, nb_steps=10, lr=1e-4),         # imp, iterative search
       "C3_fgsm0.03": dict(nb_search="fgsm", eps=0.03, lr=1e-4)}               # imp, smaller FGSM step
for name in ("C1_logit2_retain", "C2_pgd", "C3_fgsm0.03"):
    for fix in (dict(bn_recalib=1), dict(bn_mode="frozen"), dict(bn_mode="frozen", grad_clip=1.0), dict(bn_mix_retain=1)):
        J.append(("RLbnd", S1, dict(BND[name], **fix)))
J.append(("RLbnd", S1, dict(BND["C0_fgsm0.1"], bn_mix_retain=1)))
J.append(("RLbnd", S1, dict(BND["C0_fgsm0.1"], grad_clip=1.0)))
for name in ("C0_fgsm0.1", "C2_pgd", "C3_fgsm0.03"):
    J.append(("RLbnd", S30, dict(BND[name], bn_mode="frozen")))
    J.append(("RLbnd", S30, dict(BND[name], bn_mix_retain=1)))
J.append(("RLbnd", S30, dict(BND["C0_fgsm0.1"], bn_recalib=1)))
J.append(("RLbnd", S30, dict(BND["C2_pgd"], bn_recalib=1)))

SHORT = {"stop_every_iters": "stopit", "stop_forget_acc": "stop", "bn_mode": "bn", "grad_clip": "clip", "bn_mix_retain": "mix",
         "bn_recalib": "recal", "nb_search": "nb", "use_retain": "ret", "nb_steps": "steps"}


def fmt(v):
    return f"{v:g}" if isinstance(v, float) else str(v)


def main():
    print("cd /anchor/anchor7/sachatt/cs6886/project; source env.sh")
    for m, sc, hp in J:
        tag = "bnfix_" + "_".join(f"{SHORT.get(k, k)}{fmt(v)}" for k, v in hp.items())
        rid = f"{sc}__{m}__{tag}__s0"
        if os.path.exists(os.path.join(RECORDS_DIR, "runs", rid, "metrics.json")):
            continue
        hps = " ".join(f"'{k}={fmt(v)}'" for k, v in hp.items())
        print(f"echo '[run] {rid}'; python scripts/unlearn.py --scenario {sc} --seed 0 --method {m} --hp {hps} --tag {tag} --save_ckpt 0 2>&1 | grep -E 'FINAL|Traceback|Error|early stop'")
    print("python scripts/aggregate.py; python scripts/build_index.py; echo BNFIX_DONE")


if __name__ == "__main__":
    main()
