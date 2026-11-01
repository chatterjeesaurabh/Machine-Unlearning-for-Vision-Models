"""Teacher-Student follow-up (seed 0), tag prefix 'ts_':
 A  Teacher-Student paper Figure-5 grid: KL weight alpha x CE weight gamma in {0.1,0.5,1,2,3}^2, temperature T in {1,4},
    paper schedules: S1 = Table 3 class (3 epochs, msteps 2, forget bs 512, retain bs 128);
                     S2-30 = Illusion paper (3 epochs, msteps 2, batch 64/64) because Table 3 'selective' (bs 16, 5/5) collapses.
 B  10 epochs (msteps 5) with best-epoch checkpoint selection (no test data), alpha {0.1,0.5,1}, gamma 1, T {1,4}.
 C  joint loss (TSjoint: all three terms every step, no alternation), alpha {0.1,1} x T {1,4} x lam_f {1,0.1}, 5 epochs, best epoch.
Usage: python scripts/ts_grid.py > outputs/ts_batch.sh; nohup bash scripts/gpu_batch.sh outputs/ts_batch.sh > outputs/logs/ts.log 2>&1 &
"""
import os

from mu import RECORDS_DIR

S1, S30 = "S1c0", "S2k359p30"
SCHED = {S1: dict(epochs=3, msteps=2, forget_batch_size=512, retain_batch_size=128, lr_decay_epochs="[3]"),
         S30: dict(epochs=3, msteps=2, forget_batch_size=64, retain_batch_size=64, lr_decay_epochs="[3]")}
G = [0.1, 0.5, 1.0, 2.0, 3.0]
J = []
for sc in (S1, S30):  # A
    for T in (1.0, 4.0):
        for a in G:
            for g in G:
                J.append(("TS", sc, dict(SCHED[sc], alpha=a, gamma=g, kd_T=T), "A"))
for sc in (S1, S30):  # B
    for T in (1.0, 4.0):
        for a in (0.1, 0.5, 1.0):
            J.append(("TS", sc, dict(SCHED[sc], epochs=10, msteps=5, lr_decay_epochs="[8]", alpha=a, gamma=1.0, kd_T=T, select_best=1), "B"))
for sc in (S1, S30):  # C
    for T in (1.0, 4.0):
        for a in (0.1, 1.0):
            for lam in (1.0, 0.1):
                J.append(("TSjoint", sc, dict(alpha=a, gamma=1.0, kd_T=T, lam_f=lam, epochs=5, select_best=1), "C"))
# A2 (added after A-S1 results: with Table 3's forget bs 512 no alpha/gamma/T forgets): same Fig.5 grid on S1 with the
# Illusion-paper schedule (3 epochs, msteps 2, batch 64/64)
for T in (1.0, 4.0):
    for a in G:
        for g in G:
            J.append(("TS", S1, dict(SCHED[S30], alpha=a, gamma=g, kd_T=T), "A2"))
# run order: A-S1, B, C, A-S2, A2 (S1 first: fastest, answers the paper-grid question earliest)
ORDER = {("A", S1): 0, ("B", S1): 1, ("B", S30): 2, ("C", S1): 3, ("C", S30): 4, ("A", S30): 5, ("A2", S1): 6}
J.sort(key=lambda j: ORDER[(j[3], j[1])])

SHORT = {"forget_batch_size": "fbs", "retain_batch_size": "rbs", "lr_decay_epochs": "dec", "select_best": "best", "epochs": "ep"}


def fmt(v):
    return f"{v:g}" if isinstance(v, float) else str(v)


def main():
    print("cd /anchor/anchor7/sachatt/cs6886/project; source env.sh")
    for m, sc, hp, grp in J:
        tag = f"ts{grp}_" + "_".join(f"{SHORT.get(k, k)}{fmt(v).replace('[', '').replace(']', '')}" for k, v in hp.items())
        rid = f"{sc}__{m}__{tag}__s0"
        if os.path.exists(os.path.join(RECORDS_DIR, "runs", rid, "metrics.json")):
            continue
        hps = " ".join(f"'{k}={fmt(v)}'" for k, v in hp.items())
        print(f"echo '[run] {rid}'; python scripts/unlearn.py --scenario {sc} --seed 0 --method {m} --hp {hps} --tag {tag} --save_ckpt 0 2>&1 | grep -E 'FINAL|Traceback|Error|restored'")
    print("python scripts/aggregate.py; python scripts/build_index.py; echo TS_DONE")


if __name__ == "__main__":
    main()
