"""Builds the evaluation tables (outputs/eval_tables.json) used in records/results/evaluation.md and the method pages."""
import json, pandas as pd, numpy as np
R = 'records/results/'
e1 = pd.read_csv(R + 'entanglement_exp1.csv'); e2 = pd.read_csv(R + 'entanglement_exp2.csv')
pb = pd.read_csv(R + 'probe_best.csv'); pt = pd.read_csv(R + 'perturb_table.csv')
qb = pd.read_csv(R + 'quant_best_table.csv'); qb['mask'] = qb['mask'].fillna('none')
import sys; sys.path.insert(0, 'src')
from mu.entangle_runs import prep_ent
runs = prep_ent(pd.read_csv(R + 'all_runs.csv'))
SC = ['S1c0', 'S2k359p10', 'S2k359p30', 'S2k359p50']; P = ['low', 'medium', 'high']
M = ['GA', 'GAGD', 'TS', 'RLrand', 'RLbnd']
N = {'GA': 'M1 Gradient Ascent', 'GAGD': 'M2 GA + Descent', 'TS': 'M3a Teacher-Student', 'RLrand': 'M3b-R Random Relabeling', 'RLbnd': 'M3b-B Boundary Relabeling'}
PROBE = {'GA': 'GA (G3)', 'GAGD': 'GA+Descent (D1)', 'TS': 'Teacher-Student (T1)', 'RLrand': 'Random Relabel (R1)', 'RLbnd': 'Boundary Relabel (C1-mix)'}
orig1 = runs[(runs.tag == 'entpart') & (runs.method == 'original')].groupby(['base', 'part'])['avg_gap'].mean()
rf1 = runs[(runs.method == 'retrain') & runs.part.isin(P)].groupby(['base', 'part'])['acc_forget'].mean()

def exp1_table(methods, ws=True):
    L = ['| scenario | model | WS | low | medium | high | high - low |', '|---|---|---|---|---|---|---|']
    for sc in SC:
        o = [orig1[(sc, p)] for p in P]
        L.append(f'| {sc} | Original (no unlearning) | | {o[0]:.2f} | {o[1]:.2f} | {o[2]:.2f} | {o[2]-o[0]:+.2f} |')
        L.append(f'| {sc} | (Retrain forget acc on partition, %) | | {rf1[(sc,"low")]:.1f} | {rf1[(sc,"medium")]:.1f} | {rf1[(sc,"high")]:.1f} | |')
        for m in methods:
            for mask in (('none', 'saliency') if ws else ('none',)):
                v = e1[(e1.scenario == sc) & (e1.method == m) & (e1['mask'] == mask)].set_index('partition')['avg_gap']
                L.append(f'| {sc} | {N[m]} | {"yes" if mask == "saliency" else ""} | {v["low"]:.2f} | {v["medium"]:.2f} | {v["high"]:.2f} | {v["high"]-v["low"]:+.2f} |')
    return L

def exp1_acc_table(methods, scen=SC):
    """Per-partition forget acc (partition), acc on the excluded other partitions (S1 corrected setup only), retain / test acc
    (mean of the 3 partition runs) and MIA, from the Exp 1 runs."""
    e = runs[(runs.tag == 'entpart') | ((runs.method == 'retrain') & runs.part.isin(P))]
    g = e.groupby(['base', 'method', 'mask', 'part'])[['acc_forget', 'acc_excluded', 'acc_retain', 'acc_test', 'mia_acc']].mean()
    f1 = lambda v: '-' if pd.isna(v) else f'{v:.1f}'
    L = ['| scenario | model | WS | forget acc: low | forget acc: medium | forget acc: high | S1 only: acc on the 2 excluded partitions, low run | S1 only: excluded acc, medium run | S1 only: excluded acc, high run | retain acc (mean of 3 runs) | test acc (mean of 3 runs) | MIA: low | MIA: medium | MIA: high |',
         '|---|---|---|---|---|---|---|---|---|---|---|---|---|---|']
    for sc in scen:
        for m in ['retrain', 'original'] + list(methods):
            for mask in (('none', 'saliency') if m in methods else ('none',)):
                try:
                    r = {q: g.loc[(sc, m, mask, q)] for q in P}
                except KeyError:
                    continue
                nm = {'retrain': 'Retrain (reference)', 'original': 'Original (no unlearning)'}.get(m, N.get(m))
                L.append(f'| {sc} | {nm} | {"yes" if mask == "saliency" else ""} | ' + ' | '.join(f1(r[q].acc_forget) for q in P) + ' | '
                         + ' | '.join(f1(r[q].acc_excluded) for q in P) + f' | {f1(np.mean([r[q].acc_retain for q in P]))} | {f1(np.mean([r[q].acc_test for q in P]))} | '
                         + ' | '.join(f1(r[q].mia_acc) for q in P) + ' |')
    return L

def quant_best_table(methods, scen=SC):
    """Best configurations (records/results/quant_best_table.csv, 3 seeds): forget acc per bit width, INT4 - fp32 changes, avg_gap to
    the fp32 Retrain of the scenario at fp32 and INT4, INT4 mechanism columns."""
    L = ['| scenario | model | WS | config | forget fp32 | forget INT8 | forget INT6 | forget INT4 | change forget (INT4 - fp32) | change retain | change test | change MIA | avg_gap fp32 | avg_gap INT4 | share Q(theta_u)=Q(theta_o), INT4 | mean abs(delta)/step, INT4 |',
         '|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|']
    for sc in scen:
        ref = qb[(qb.scenario == sc) & (qb.method == 'retrain') & (qb.bits == 32)].iloc[0]
        gap = lambda r: np.mean([abs(r[c] - ref[c]) for c in ('acc_forget', 'acc_retain', 'acc_test', 'mia')])
        for m in ['original', 'retrain'] + list(methods):
            for mask in (('none', 'saliency') if m in methods else ('none',)):
                a = qb[(qb.scenario == sc) & (qb.method == m) & (qb['mask'] == mask)].set_index('bits')
                if len(a) == 0:
                    continue
                nm = {'original': 'Original', 'retrain': 'Retrain'}.get(m, N.get(m))
                cfg = a.loc[32, 'config'] if isinstance(a.loc[32, 'config'], str) else ''
                d = lambda c: f'{a.loc[4, c] - a.loc[32, c]:+.2f}'
                L.append(f'| {sc} | {nm} | {"yes" if mask == "saliency" else ""} | {cfg} | ' + ' | '.join(f'{a.loc[b, "acc_forget"]:.2f}' for b in (32, 8, 6, 4))
                         + f' | {d("acc_forget")} | {d("acc_retain")} | {d("acc_test")} | {d("mia")} | {gap(a.loc[32]):.2f} | {gap(a.loc[4]):.2f} | '
                         + f'{a.loc[4, "frac_same_as_quantized_original"]:.3f} | {a.loc[4, "mean_abs_delta_over_step"]:.3f} |')
    return L

MARGIN = [l for l in open(R + 'quant_margin_diag.md').read().splitlines() if l.startswith('|')]

def exp2_table(methods):
    L = ['| model | WS | ' + ' | '.join(f'{sc} one-shot | {sc} sequential' for sc in SC) + ' |', '|---|---|' + '---|' * 8]
    for m in methods:
        for mask in ('none', 'saliency'):
            cells = []
            for sc in SC:
                x = e2[(e2.scenario == sc) & (e2.method == m) & (e2['mask'] == mask)].set_index('mode')['avg_gap']
                o, s = x.get('one-shot', np.nan), x.get('sequential', np.nan)
                cells += [f'**{o:.2f}**' if o < s else f'{o:.2f}', f'**{s:.2f}**' if s < o else f'{s:.2f}']
            L.append(f'| {N[m]} | {"yes" if mask == "saliency" else ""} | ' + ' | '.join(cells) + ' |')
    return L

def seq_final_table(m):
    L = ['| scenario | WS | mode | forget | retain | test | MIA | avg_gap |', '|---|---|---|---|---|---|---|---|']
    for sc in SC:
        for mask in ('none', 'saliency'):
            for mode in ('one-shot', 'sequential'):
                r = e2[(e2.scenario == sc) & (e2.method == m) & (e2['mask'] == mask) & (e2['mode'] == mode)]
                if len(r):
                    r = r.iloc[0]
                    L.append(f'| {sc} | {"yes" if mask == "saliency" else ""} | {mode} | {r.acc_forget:.2f} | {r.acc_retain:.2f} | {r.acc_test:.2f} | {r.mia_acc:.2f} | {r.avg_gap:.2f} |')
    return L

def probe_table(labels):
    g = pb.groupby('model', sort=False).agg(['mean', 'std'])
    L = ['| model | output forget-cls | probe forget-cls | NCC forget-cls | probe other-cls | silhouette class 0 | 10-NN purity class 0 |', '|---|---|---|---|---|---|---|']
    for lab in labels:
        r = g.loc[lab]
        f = lambda c, d=2: f'{r[(c,"mean")]:.{d}f} ± {r[(c,"std")]:.{d}f}'
        L.append(f'| {lab} | {f("output_forgetcls")} | {f("probe_forgetcls")} | {f("ncc_forgetcls")} | {f("probe_othercls")} | {f("silhouette_class0",3)} | {f("knn10_purity_class0")} |')
    return L

def perturb_table(methods):
    L = ['| scenario | model | WS | clean forget acc | perturbed forget acc tau 0.03 | recovery tau 0.03 | perturbed tau 0.1 | recovery tau 0.1 |', '|---|---|---|---|---|---|---|---|']
    for sc in ('S1c0', 'S2k359p30'):
        for m in ['original', 'retrain'] + methods:
            for mask in (('none',) if m in ('original', 'retrain') else ('none', 'saliency')):
                a = pt[(pt.scenario == sc) & (pt.method == m) & (pt['mask'] == mask)].set_index('tau')
                if len(a) == 0:
                    continue
                nm = {'original': 'Original', 'retrain': 'Retrain'}.get(m, N.get(m))
                L.append(f'| {sc} | {nm} | {"yes" if mask == "saliency" else ""} | {a.loc[0.03,"clean_forget_acc"]:.1f} | {a.loc[0.03,"perturbed_forget_acc"]:.1f} | {a.loc[0.03,"recovery_any"]:.1f} | {a.loc[0.1,"perturbed_forget_acc"]:.1f} | {a.loc[0.1,"recovery_any"]:.1f} |')
    return L

json.dump({'exp1': exp1_table(M), 'exp1_acc': exp1_acc_table(M), 'exp2': exp2_table(M), 'probe': probe_table(['Original', 'Retrain'] + [x for m in M for x in (PROBE[m], PROBE[m] + ' + WS')]),
           'perturb': perturb_table(M), 'quant_best': quant_best_table(M), 'quant_margin': MARGIN,
           'per_method': {m: {'exp1': exp1_table([m]), 'exp1_acc': exp1_acc_table([m]), 'seq': seq_final_table(m), 'probe': probe_table(['Original', 'Retrain', PROBE[m], PROBE[m] + ' + WS']),
                              'perturb': perturb_table([m]), 'quant_best': quant_best_table([m])} for m in M}},
          open('outputs/eval_tables.json', 'w'))
print('ok')


# ---------------- summary tables (one value per column), added 2026-10-30 ----------------
BEST_TAG = {('GA', 'S1c0'): 'imp5_gamix', ('GA', 'S2'): 'imp5_gamix', ('GAGD', 'S1c0'): '', ('GAGD', 'S2'): '',
            ('TS', 'S1c0'): '', ('TS', 'S2'): 'imp3_tsill', ('TS', 'S2k359p50'): 'imp4_tsill_fbs128',
            ('RLrand', 'S1c0'): '', ('RLrand', 'S2'): '', ('RLbnd', 'S1c0'): 'imp5_bndlogit2mix', ('RLbnd', 'S2'): ''}
CONFNAME = {('GA', 'S1c0'): 'G3', ('GA', 'S2'): 'G3-S2', ('GAGD', 'S1c0'): 'D1', ('GAGD', 'S2'): 'D1-S2', ('TS', 'S1c0'): 'T1',
            ('TS', 'S2'): 'T4-best S2', ('RLrand', 'S1c0'): 'R1', ('RLrand', 'S2'): 'R1-S2', ('RLbnd', 'S1c0'): 'C1-mix', ('RLbnd', 'S2'): 'C0'}


def best_tag(m, sc):
    return BEST_TAG.get((m, sc), BEST_TAG[(m, 'S1c0' if sc == 'S1c0' else 'S2')])


def cname(m, sc):
    return CONFNAME[(m, 'S1c0' if sc == 'S1c0' else 'S2')]


def mean_of(sc, m, mask, tag, col):
    x = runs[(runs.scenario == sc) & (runs.method == m) & (runs['mask'] == mask) & (runs.tag == tag)]
    return x[col].mean(), len(x)


def ref_mean(sc, m, col):
    x = runs[(runs.scenario == sc) & (runs.method == m) & (runs.tag == '')]
    return x[col].mean()


def e1v(sc, m, mask, p):
    if m == 'original':
        return orig1[(sc, p)]
    v = e1[(e1.scenario == sc) & (e1.method == m) & (e1['mask'] == mask) & (e1.partition == p)]['avg_gap']
    return v.iloc[0] if len(v) else np.nan


def e2v(sc, m, mask, mode):
    v = e2[(e2.scenario == sc) & (e2.method == m) & (e2['mask'] == mask) & (e2['mode'] == mode)]['avg_gap']
    return v.iloc[0] if len(v) else np.nan


f2 = lambda v: '-' if pd.isna(v) else f'{v:.2f}'
pbm = pb.groupby('model', sort=False).mean(numeric_only=True)
PL = {'GA': 'GA (G3)', 'GAGD': 'GA+Descent (D1)', 'TS': 'Teacher-Student (T1)', 'RLrand': 'Random Relabel (R1)', 'RLbnd': 'Boundary Relabel (C1-mix)'}


def summary_s1():
    sc = 'S1c0'
    L = ['| model | WS | config | MIA acc (%) | probe acc on class 0 (%) | NCC acc on class 0 (%) | 10-NN purity of class 0 (%) | Exp 1 avg_gap: low partition | Exp 1 avg_gap: medium | Exp 1 avg_gap: high | Exp 2 avg_gap: one-shot | Exp 2 avg_gap: sequential |',
         '|---|---|---|---|---|---|---|---|---|---|---|---|']
    o = pbm.loc['Original']; r = pbm.loc['Retrain']
    L.append(f'| Retrain (reference) | | | {f2(ref_mean(sc, "retrain", "mia_acc"))} | {f2(r.probe_forgetcls)} | {f2(r.ncc_forgetcls)} | {f2(r.knn10_purity_class0)} | 0 | 0 | 0 | 0 | 0 |')
    L.append(f'| Original (no unlearning) | | | {f2(ref_mean(sc, "original", "mia_acc"))} | {f2(o.probe_forgetcls)} | {f2(o.ncc_forgetcls)} | {f2(o.knn10_purity_class0)} | '
             f'{f2(e1v(sc, "original", "none", "low"))} | {f2(e1v(sc, "original", "none", "medium"))} | {f2(e1v(sc, "original", "none", "high"))} | {f2(ref_mean(sc, "original", "avg_gap"))} | - |')
    for m in M:
        for mask in ('none', 'saliency'):
            mia, _ = mean_of(sc, m, mask, best_tag(m, sc), 'mia_acc')
            pr = pbm.loc[PL[m] + (' + WS' if mask == 'saliency' else '')]
            L.append(f'| {N[m]} | {"yes" if mask == "saliency" else "no"} | {cname(m, sc)} | {f2(mia)} | {f2(pr.probe_forgetcls)} | {f2(pr.ncc_forgetcls)} | {f2(pr.knn10_purity_class0)} | '
                     f'{f2(e1v(sc, m, mask, "low"))} | {f2(e1v(sc, m, mask, "medium"))} | {f2(e1v(sc, m, mask, "high"))} | {f2(e2v(sc, m, mask, "one-shot"))} | {f2(e2v(sc, m, mask, "sequential"))} |')
    return L


def summary_s2():
    L = ['| scenario | model | WS | config | MIA acc (%) | Exp 1 avg_gap: low partition | Exp 1 avg_gap: medium | Exp 1 avg_gap: high | Exp 2 avg_gap: one-shot | Exp 2 avg_gap: sequential |',
         '|---|---|---|---|---|---|---|---|---|---|']
    for sc in SC[1:]:
        L.append(f'| {sc} | Retrain (reference) | | | {f2(ref_mean(sc, "retrain", "mia_acc"))} | 0 | 0 | 0 | 0 | 0 |')
        L.append(f'| {sc} | Original (no unlearning) | | | {f2(ref_mean(sc, "original", "mia_acc"))} | {f2(e1v(sc, "original", "none", "low"))} | {f2(e1v(sc, "original", "none", "medium"))} | '
                 f'{f2(e1v(sc, "original", "none", "high"))} | {f2(ref_mean(sc, "original", "avg_gap"))} | - |')
        for m in M:
            for mask in ('none', 'saliency'):
                mia, _ = mean_of(sc, m, mask, best_tag(m, sc), 'mia_acc')
                L.append(f'| {sc} | {N[m]} | {"yes" if mask == "saliency" else "no"} | {cname(m, sc)} | {f2(mia)} | {f2(e1v(sc, m, mask, "low"))} | {f2(e1v(sc, m, mask, "medium"))} | '
                         f'{f2(e1v(sc, m, mask, "high"))} | {f2(e2v(sc, m, mask, "one-shot"))} | {f2(e2v(sc, m, mask, "sequential"))} |')
    return L


def mia_table():
    L = ['| scenario | model | MIA main grid, no WS | MIA main grid, WS | MIA best config, no WS | MIA best config, WS | best config |', '|---|---|---|---|---|---|---|']
    for sc in SC:
        L.append(f'| {sc} | Retrain | {f2(ref_mean(sc, "retrain", "mia_acc"))} | | | | |')
        L.append(f'| {sc} | Original | {f2(ref_mean(sc, "original", "mia_acc"))} | | | | |')
        for m in M:
            v = [mean_of(sc, m, mk, '', 'mia_acc')[0] for mk in ('none', 'saliency')] + [mean_of(sc, m, mk, best_tag(m, sc), 'mia_acc')[0] for mk in ('none', 'saliency')]
            L.append(f'| {sc} | {N[m]} | ' + ' | '.join(f2(x) for x in v) + f' | {cname(m, sc)} |')
    return L


def avg_gap_s2_table():
    L = ['| scenario | model | avg_gap main grid, no WS | avg_gap main grid, WS | avg_gap best config, no WS | avg_gap best config, WS | best config |', '|---|---|---|---|---|---|---|']
    for sc in SC[1:]:
        L.append(f'| {sc} | Original (no unlearning) | {f2(ref_mean(sc, "original", "avg_gap"))} | | | | |')
        for m in M:
            v = [mean_of(sc, m, mk, '', 'avg_gap')[0] for mk in ('none', 'saliency')] + [mean_of(sc, m, mk, best_tag(m, sc), 'avg_gap')[0] for mk in ('none', 'saliency')]
            L.append(f'| {sc} | {N[m]} | ' + ' | '.join(f2(x) for x in v) + f' | {cname(m, sc)} |')
    return L


def ws_effect_table():
    L = ['| evaluation (scenario) | model | without WS | with WS |', '|---|---|---|---|']
    for m in M:
        L.append(f'| MIA acc, best config (S1) | {N[m]} | {f2(mean_of("S1c0", m, "none", best_tag(m, "S1c0"), "mia_acc")[0])} | {f2(mean_of("S1c0", m, "saliency", best_tag(m, "S1c0"), "mia_acc")[0])} |')
    for m in M:
        L.append(f'| probe acc on class 0 (S1) | {N[m]} | {f2(pbm.loc[PL[m]].probe_forgetcls)} | {f2(pbm.loc[PL[m] + " + WS"].probe_forgetcls)} |')
    for m in M:
        L.append(f'| NCC acc on class 0 (S1) | {N[m]} | {f2(pbm.loc[PL[m]].ncc_forgetcls)} | {f2(pbm.loc[PL[m] + " + WS"].ncc_forgetcls)} |')
    for sc in SC:
        for p in P:
            for m in M:
                L.append(f'| Exp 1 avg_gap, {p} partition ({sc}) | {N[m]} | {f2(e1v(sc, m, "none", p))} | {f2(e1v(sc, m, "saliency", p))} |')
    for sc in SC:
        for mode in ('one-shot', 'sequential'):
            for m in M:
                L.append(f'| Exp 2 avg_gap, {mode} ({sc}) | {N[m]} | {f2(e2v(sc, m, "none", mode))} | {f2(e2v(sc, m, "saliency", mode))} |')
    for sc in ('S1c0', 'S2k359p30'):
        for m in M:
            a = pt[(pt.scenario == sc) & (pt.method == m) & (pt.tau == 0.03)].set_index('mask')['perturbed_forget_acc']
            L.append(f'| perturbed forget acc, tau 0.03, main grid ({sc}) | {N[m]} | {f2(a.get("none"))} | {f2(a.get("saliency"))} |')
    for sc in SC:
        for m in M:
            a = qb[(qb.scenario == sc) & (qb.method == m) & (qb.bits == 4)].set_index('mask')['frac_same_as_quantized_original']
            L.append(f'| share of weights with Q(theta_u) = Q(theta_o) at INT4, best config ({sc}) | {N[m]} | {f2(a.get("none"))} | {f2(a.get("saliency"))} |')
    for sc in SC:
        for m in M:
            a4 = qb[(qb.scenario == sc) & (qb.method == m) & (qb.bits == 4)].set_index('mask')['acc_forget']
            a32 = qb[(qb.scenario == sc) & (qb.method == m) & (qb.bits == 32)].set_index('mask')['acc_forget']
            L.append(f'| forget acc change INT4 - fp32, best config ({sc}) | {N[m]} | {a4.get("none") - a32.get("none"):+.2f} | {a4.get("saliency") - a32.get("saliency"):+.2f} |')
    return L


T2 = json.load(open('outputs/eval_tables.json'))
T2.update({'summary_s1': summary_s1(), 'summary_s2': summary_s2(), 'mia_full': mia_table(), 'avg_gap_s2': avg_gap_s2_table(), 'ws_effect': ws_effect_table()})
json.dump(T2, open('outputs/eval_tables.json', 'w'))
print('summary tables ok')
