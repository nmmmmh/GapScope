"""Final ledger for the GapScope manuscript. Offline assembly only: reads per-unit scores (predictions_summary_*.csv),
window files, PIT files, arm_metrics json rows, floors; no model is trained or tuned here.
Writes cv/final_numbers.csv, cv/final_consistency.txt.
Bootstrap: paired over compositions (pooled units grouped by composition), 4000 draws, seed 20261113 (one draw matrix per fold type)."""
import sys, glob, json, os, re
import numpy as np, pandas as pd
CV = '/data/czx/GapScope_Revision_20261113/cv/'
sys.path.insert(0, CV)
NB = 4000
rows = []
CONS = []


def add(section, fold, model, tier, metric, value, source, ref='', lo=None, hi=None, n=None, seed_set='', internal='', note=''):
    rows.append(dict(section=section, fold_type=fold, model=model, tier=tier, metric=metric, value=value, ci95_lo=lo, ci95_hi=hi, ref=ref,
                     n=n, seed_set=seed_set, internal_name=internal, source=source, note=note))


# ------------------------------------------------------------------ per-unit files
pfiles = sorted(glob.glob(CV + 'predictions_summary_v3*.csv') + glob.glob(CV + 'predictions_summary_v4*.csv') +
                glob.glob(CV + 'predictions_summary_v6*.csv') + glob.glob(CV + 'predictions_summary_vfinal*.csv'))
SFX = [f.split('predictions_summary')[1][:-4] for f in pfiles]
PRED = {}
for s, f in zip(SFX, pfiles):
    PRED[s] = pd.read_csv(f)
AV = {}
for s in SFX:
    for k in PRED[s].groupby(['variant', 'model']).groups:
        AV.setdefault(k, []).append(s)


def wfile(kind, s):
    p = CV + f'{kind}{s}.csv'
    return p if os.path.exists(p) else None


WIN, WINU, PIT = {}, {}, {}
for s in SFX:
    if wfile('windows', s):
        WIN[s] = pd.read_csv(wfile('windows', s))
    if wfile('windows_units', s):
        WINU[s] = pd.read_csv(wfile('windows_units', s))
    if wfile('pit', s):
        PIT[s] = pd.read_csv(wfile('pit', s))


def has(D, s, v, m):
    return s in D and ((D[s].variant == v) & (D[s].model == m)).any()


GEN = {'comp': ['_v4t1', '_v4a', '_v4eq', '_v4sgnz_frozen', '_v6t23', '_v6compX', '_v6comp6', '_v6compXeq', '_v4g_comp_GS_SELCAL_PRE_NZ',
                '_vfinal_enc', '_v3', '_v3b', '_v3c'],
       'chem': ['_v6chem', '_v4eqchem', '_v4chem_NZ', '_v6cheq', '_v4g_chem_GS_SELCAL_PRE_NZ', '_v3chem', '_v4chem_MT', '_v4g_chem_GS_SELCAL']}
BASEFIRST = {'comp': ['_v4a'], 'chem': ['_v4eqchem']}


def order(v, m):
    o = list(GEN[v])
    if m.startswith('ROOST') or m.startswith('EASYUQ_IDR') or m == 'EASYUQ_KS':
        o = BASEFIRST[v] + o
    rest = [s for s in AV.get((v, m), []) if s not in o]
    return [s for s in o + rest if s in AV.get((v, m), [])]


def choose(v, m):
    for s in order(v, m):
        if has(WIN, s, v, m) and has(WINU, s, v, m):
            return s
    o = order(v, m)
    return o[0] if o else None


MODELS = [
    ('comp', 'ROOST', 'ROOST', 'baseline'), ('comp', 'ROOST_CAL', 'ROOST_CAL', 'baseline'), ('comp', 'ROOST_CAL_CV', 'ROOST_CAL_CV', 'baseline'),
    ('comp', 'ROOST6_CAL', 'ROOST6_CAL', 'baseline'), ('comp', 'ROOSTX_CAL', 'ROOSTX_CAL', 'baseline'),
    ('comp', 'EASYUQ_KS', 'EASYUQ_KS', 'baseline'), ('comp', 'EASYUQ_IDR', 'EASYUQ_IDR', 'baseline'),
    ('comp', 'EASYUQ_X_KS', 'EASYUQ_KS_roostX', 'baseline'), ('comp', 'EASYUQ_X_IDR', 'EASYUQ_IDR_roostX', 'baseline'),
    ('comp', 'EASYUQ_ON_GS_SELCAL_PRE_NZ_KS', 'EASYUQ_ON_GS_KS', 'baseline'), ('comp', 'EASYUQ_ON_GS_SELCAL_PRE_NZ_IDR', 'EASYUQ_ON_GS_IDR', 'baseline'),
    ('comp', 'EASYUQ_ON_GS_SGNZ_frozen_X_KS', 'EASYUQ_ON_GS_SG_NZ_X_KS', 'baseline'), ('comp', 'EASYUQ_ON_GS_SGNZ_frozen_X_IDR', 'EASYUQ_ON_GS_SG_NZ_X_IDR', 'baseline'),
    ('comp', 'GS_SELCAL_PRE_NZ_SC', 'PRE_NZ', 'main'), ('comp', 'GS_SELCAL_PRE_NZ_6S_SC', 'PRE_NZ_6S', 'main'),
    ('comp', 'GS_SGNZ_frozen_SC', 'SG_NZ', 'variant'), ('comp', 'GS_SGNZ_frozenX_SC', 'SG_NZ_X', 'variant'), ('comp', 'GS_SGB_a_SC', 'SGB_a', 'variant'),
    ('comp', 'GS_SGB_b_SC', 'SGB_b', 'variant'), ('comp', 'GS_SGB_c_SC', 'SGB_c', 'variant'),
    ('comp', 'GS_SGC_l1_SC', 'SGC_l1', 'variant'), ('comp', 'GS_SGC_l5_SC', 'SGC_l5', 'variant'),
    ('comp', 'GS_SELCAL_SC', 'GS_SELCAL_SC', 'ablation'), ('comp', 'GS_SELCAL_HSE_SC', 'HSE_plain', 'ablation'),
    ('comp', 'GS_SELCAL_PRE_NEG_SC', 'PRE_NEG', 'ablation'), ('comp', 'GS_G20_L1_CRPS_SC', 'GS_BEST', 'ablation'), ('comp', 'GS_ELR_m0_SC', 'ENC_FROZEN', 'ablation'),
    ('chem', 'ROOST', 'ROOST', 'baseline'), ('chem', 'ROOST_CAL', 'ROOST_CAL', 'baseline'), ('chem', 'ROOST_CAL_CV', 'ROOST_CAL_CV', 'baseline'),
    ('chem', 'EASYUQ_KS', 'EASYUQ_KS', 'baseline'), ('chem', 'EASYUQ_IDR', 'EASYUQ_IDR', 'baseline'),
    ('chem', 'EASYUQ_ON_GS_SELCAL_PRE_NZ_KS', 'EASYUQ_ON_GS_KS', 'baseline'), ('chem', 'EASYUQ_ON_GS_SELCAL_PRE_NZ_IDR', 'EASYUQ_ON_GS_IDR', 'baseline'),
    ('chem', 'EASYUQ_ON_GS_SGNZ_frozen_KS', 'EASYUQ_ON_GS_SG_NZ_KS', 'baseline'), ('chem', 'EASYUQ_ON_GS_SGNZ_frozen_IDR', 'EASYUQ_ON_GS_SG_NZ_IDR', 'baseline'),
    ('chem', 'GS_SELCAL_PRE_NZ_SC', 'PRE_NZ', 'main'), ('chem', 'GS_SGNZ_frozen_SC', 'SG_NZ', 'variant'),
    ('chem', 'GS_SELCAL_SC', 'GS_SELCAL_SC', 'ablation'), ('chem', 'GS_BEST_SC', 'GS_BEST', 'ablation'),
]
PN = {(v, i): p for v, i, p, g in MODELS}
GRP = {(v, p): g for v, i, p, g in MODELS}


def tiers_of(g):
    a = g[g.tier == 'A']
    ca = []
    for k, gg in a.groupby('key'):
        w = gg.w.values / gg.w.sum()
        ca.append(dict(key=k, **{c: float((gg[c].values * w).sum()) for c in ('crps', 'nll', 'cov80', 'width80', 'mae')}))
    ca = pd.DataFrame(ca).sort_values('key').reset_index(drop=True)
    b = g[g.tier == 'B'][['key', 'crps', 'nll', 'cov80', 'width80', 'mae']].sort_values('key').reset_index(drop=True)
    return ca, b


UT = {}
MISSING = []
for v, i, p, g in MODELS:
    s = choose(v, i)
    if s is None:
        MISSING.append((v, p, i))
        continue
    ca, b = tiers_of(PRED[s][(PRED[s].variant == v) & (PRED[s].model == i)])
    UT[(v, p)] = (ca, b, s, i)

# ------------------------------------------------------------------ 1. metrics
MET = ('crps', 'nll', 'cov80', 'width80', 'mae')
for (v, p), (ca, b, s, i) in UT.items():
    src = f'predictions_summary{s}.csv'
    for t, df in (('A', ca), ('B', b), ('pooled', pd.concat([ca, b]))):
        for m in MET:
            add('metrics', v, p, t, m, float(df[m].mean()), src, n=len(df), internal=i)
    for s2 in AV.get((v, i), []):
        if s2 == s:
            continue
        ca2, b2 = tiers_of(PRED[s2][(PRED[s2].variant == v) & (PRED[s2].model == i)])
        d = max(abs(ca2[m].mean() - ca[m].mean()) for m in ('crps', 'nll')), max(abs(b2[m].mean() - b[m].mean()) for m in ('crps', 'nll'))
        if max(d) > 1e-9:
            CONS.append(f'{v} {i}: chosen {s} vs {s2}: max |diff| tier A {d[0]:.3g}, tier B {d[1]:.3g} (crps/nll)')

# ------------------------------------------------------------------ 2. bootstrap
REFS_DEF = ['ROOST_CAL', 'EASYUQ_KS', 'PRE_NZ']


def refs_for(v, p):
    if p in ('ROOSTX_CAL', 'EASYUQ_KS_roostX', 'EASYUQ_IDR_roostX', 'EASYUQ_ON_GS_SG_NZ_X_KS', 'EASYUQ_ON_GS_SG_NZ_X_IDR', 'SG_NZ_X'):
        return ['ROOSTX_CAL', 'EASYUQ_KS_roostX']
    r = list(REFS_DEF)
    if p == 'PRE_NZ_6S':
        r += ['ROOST6_CAL']
    if p == 'GS_BEST':
        r += ['GS_SELCAL_SC']
    return r


BOOT = {}
for v in ('comp', 'chem'):
    keys = sorted({k for (vv, p), (ca, b, s, i) in UT.items() if vv == v for k in list(ca.key) + list(b.key)})
    cid = {k: j for j, k in enumerate(keys)}
    rng = np.random.default_rng(20261113)
    cnt = rng.multinomial(len(cid), np.ones(len(cid)) / len(cid), size=NB).astype(float)
    for (vv, p), (ca, b, s, i) in UT.items():
        if vv != v:
            continue
        for ref in refs_for(v, p):
            if ref == p or (v, ref) not in UT:
                continue
            ra, rb, rs, ri = UT[(v, ref)]
            if not (len(ca) == len(ra) and (ca.key.values == ra.key.values).all() and len(b) == len(rb) and (b.key.values == rb.key.values).all()):
                CONS.append(f'{v} bootstrap {p} vs {ref}: unit keys differ, skipped')
                continue
            for t in ('A', 'B', 'pooled'):
                xa, xb = (ca, ra) if t == 'A' else (b, rb) if t == 'B' else (pd.concat([ca, b]), pd.concat([ra, rb]))
                ids = np.array([cid[k] for k in xa.key.values])
                W = cnt[:, ids]
                for met in ('crps', 'nll', 'mae'):
                    d = xa[met].values - xb[met].values
                    bs = (W @ d) / W.sum(1)
                    lo, hi = np.percentile(bs, [2.5, 97.5])
                    BOOT[(v, p, ref, t, met)] = (float(d.mean()), float(lo), float(hi))
                    add('bootstrap', v, p, t, f'd_{met}_vs_{ref}', float(d.mean()), f'predictions_summary{s}.csv & predictions_summary{rs}.csv (build_final_ledger.py, {NB} draws)',
                        ref=ref, lo=float(lo), hi=float(hi), n=len(xa), internal=i, note='model minus reference; negative = model better')

old = pd.concat([pd.read_csv(f).assign(f=os.path.basename(f)) for f in glob.glob(CV + 'bootstrap_v6_*.csv') + [CV + 'bootstrap_v4.csv']])
nchk, nbad = 0, 0
for r in old.itertuples():
    if r.metric not in ('crps', 'nll'):
        continue
    m_, ref = r.comparison.rsplit('_vs_', 1)
    pm = PN.get((r.variant, m_))
    rf = PN.get((r.variant, ref))
    if pm is None or rf is None:
        continue
    k = (r.variant, pm, rf, r.tier, r.metric)
    if k in BOOT:
        nchk += 1
        if abs(BOOT[k][0] - r.diff) > 1e-9:
            nbad += 1
            CONS.append(f'bootstrap point estimate differs {k}: ledger {BOOT[k][0]:.6f} vs {r.f} {r.diff:.6f}')
CONS.append(f'bootstrap cross-check: {nchk} (model, ref, tier, metric) point estimates compared with existing bootstrap_v4/v6 files, {nbad} differ')

# ------------------------------------------------------------------ 3. windows
WCOLS = ['pv_brier_prob', 'pv_brier_hard', 'pv_auc_prob', 'pv_auc_dist', 'pv_n_any', 'pv_top10_prob', 'pv_top10_dist', 'pv_top20_prob', 'pv_top20_dist', 'pv_top50_prob', 'pv_top50_dist',
         'visible_brier_prob', 'visible_brier_hard', 'visible_auc_prob', 'visible_auc_dist', 'visible_n_any', 'visible_top10_prob', 'visible_top10_dist',
         'visible_top20_prob', 'visible_top20_dist', 'visible_top50_prob', 'visible_top50_dist']


def ece(g, win):
    p = g[win + '_p'].values.astype(float)
    o = g[win + '_any'].values.astype(float)
    bi = np.clip(np.digitize(p, np.linspace(0, 1, 11)[1:-1]), 0, 9)
    return float(sum((bi == j).sum() / len(p) * abs(o[bi == j].mean() - p[bi == j].mean()) for j in range(10) if (bi == j).any()))


for (v, p), (ca, b, s, i) in UT.items():
    sw = s if has(WIN, s, v, i) else next((x for x in order(v, i) if has(WIN, x, v, i)), None)
    if sw is None:
        add('windows', v, p, 'all2522', 'window_metrics', 'not run', '-', internal=i)
        continue
    w = WIN[sw]
    w = w[(w.variant == v) & (w.model == i)].iloc[0]
    for c in WCOLS:
        add('windows', v, p, 'all2522', c, float(w[c]), f'windows{sw}.csv', n=int(w.n_comp), internal=i)
    su = sw if has(WINU, sw, v, i) else next((x for x in order(v, i) if has(WINU, x, v, i)), None)
    if su:
        g = WINU[su]
        g = g[(g.variant == v) & (g.model == i)]
        for win in ('pv', 'visible'):
            add('windows', v, p, 'all2522', f'ece_{win}', ece(g, win), f'windows_units{su}.csv', n=len(g), internal=i, note='10 equal-width probability bins; per-composition window probabilities')
        bq = float(g['pv_brier'].mean())
        if abs(bq - float(w['pv_brier_prob'])) > 1e-9:
            CONS.append(f'{v} {i}: pv_brier_prob summary {float(w["pv_brier_prob"]):.6f} vs recomputed from units {bq:.6f}')
    for s2 in order(v, i):
        if s2 != sw and has(WIN, s2, v, i):
            w2 = WIN[s2]
            w2 = w2[(w2.variant == v) & (w2.model == i)].iloc[0]
            dd = max(abs(float(w2[c]) - float(w[c])) for c in WCOLS)
            if dd > 1e-9:
                CONS.append(f'{v} {i}: windows differ between windows{sw}.csv and windows{s2}.csv (max |diff| {dd:.3g})')

# ------------------------------------------------------------------ 4. PIT
for (v, p), (ca, b, s, i) in UT.items():
    sp = next((x for x in order(v, i) if has(PIT, x, v, i)), None)
    if sp is None:
        for t in ('A', 'B'):
            add('pit', v, p, t, 'pit_bin_counts', 'not run', '-', internal=i)
        continue
    q = PIT[sp]
    q = q[(q.variant == v) & (q.model == i)]
    tot = np.zeros(10)
    for t in ('A', 'B'):
        r = q[q.tier == t].iloc[0]
        cnts = [int(r[f'bin{j}']) for j in range(1, 11)]
        tot += np.array(cnts)
        add('pit', v, p, t, 'pit_bin_counts', ' '.join(map(str, cnts)), f'pit{sp}.csv', n=sum(cnts), internal=i, note='10 equal-width bins; tier A per record, tier B per distinct value')
    add('pit', v, p, 'pooled', 'pit_bin_counts', ' '.join(str(int(x)) for x in tot), f'pit{sp}.csv', n=int(tot.sum()), internal=i, note='sum of tier A and tier B counts')
    for s2 in order(v, i):
        if s2 != sp and has(PIT, s2, v, i):
            q2 = PIT[s2]
            q2 = q2[(q2.variant == v) & (q2.model == i)]
            cols = [f'bin{j}' for j in range(1, 11)]
            if not (q2.sort_values('tier')[cols].values == q.sort_values('tier')[cols].values).all():
                CONS.append(f'{v} {i}: PIT counts differ between pit{sp}.csv and pit{s2}.csv')

# ------------------------------------------------------------------ 5/6. arm_metrics rows: S5, phase, SG on/off, validation
arm_rows = []
for f in sorted(glob.glob(CV + 'v4rows/*.json')):
    arm_rows.append(('comp', os.path.basename(f)[:-5], '17/29/41', f))
for f in sorted(glob.glob(CV + 'v6rows/*.json')):
    b_ = os.path.basename(f)[:-5]
    m = re.match(r'(comp|chem)_(.+)_(17,29,41)$', b_)
    arm_rows.append((m.group(1), m.group(2), '17/29/41', f))
PAPER_ARM = {'GS_SELCAL_PRE_NZ': 'PRE_NZ', 'GS_SGNZ_frozen': 'SG_NZ', 'GS_SGNZ_frozenX': 'SG_NZ_X', 'GS_SGB_a': 'SGB_a', 'GS_SGB_b': 'SGB_b', 'GS_SGB_c': 'SGB_c',
             'GS_SGC_l1': 'SGC_l1', 'GS_SGC_l5': 'SGC_l5', 'GS_F0': 'GS_F0', 'GS_ELR_m0': 'ENC_FROZEN'}
RUNS = {}
for f in glob.glob(CV + 'ft/out/gs_out_v2/*_info.json'):
    mm = re.match(r'(comp|chem)_(.+)_seed(\d+)_fold\d_info\.json', os.path.basename(f))
    if mm and int(mm.group(3)) in (17, 29, 41):
        RUNS[(mm.group(1), mm.group(2))] = RUNS.get((mm.group(1), mm.group(2)), 0) + 1
for fold, arm, ss, f in arm_rows:
    d = json.load(open(f))
    paper = PAPER_ARM.get(arm, arm)
    seedset = '53/67/79 (mp_seed 17/29/41)' if arm == 'GS_SGNZ_frozenX' else ss
    nr = RUNS.get((fold, arm), None)
    src = os.path.relpath(f, CV).replace('\\', '/')
    grp = 'frozen set' if arm in PAPER_ARM else 'appendix (earlier SG fixes, not frozen)'
    keyvals = [('s5_mae_full', d['s5_mae_full'], 'A'), ('s5_dir_full', d['s5_dir_full'], 'A'), ('s5_mae_formula_only', d['s5_mae_formula'], 'A'), ('s5_n_pairs', d['s5_n_pairs'], 'A'),
               ('s5_zero_ref', d['s5_zero_ref'], 'A'), ('phase_rho', d['phase_rho'], 'A'), ('phase_p', d['phase_p'], 'A'), ('phase_n', d['phase_n'], 'A'),
               ('tierA_crps_withSG_test', d['A_crps_sg'], 'A'), ('tierA_crps_noSG_test', d['A_crps_nosg'], 'A'), ('tierB_crps_test', d['B_crps'], 'B'),
               ('pooled_crps_formula_test', d['pooled_crps_formula'], 'pooled'), ('pooled_crps_withSG_test', d['pooled_crps_sg'], 'pooled'),
               ('tierA_crps_withSG_val', d.get('val_A_sg'), 'A'), ('tierA_crps_noSG_val', d.get('val_A_nosg'), 'A'), ('pooled_crps_val', d.get('val_crps_cal'), 'pooled')]
    for mname, val, t in keyvals:
        if val is None or (isinstance(val, float) and np.isnan(val)):
            val = 'not available'
        add('sg_s5_phase', fold, paper, t, mname, val, src, seed_set=seedset, internal=arm, n=nr, note=f'{grp}; n = (seed, fold) runs on disk')
for nm in ('ROOST', 'ROOST_CAL'):
    add('sg_s5_phase', 'comp', nm, 'A', 's5_mae_full', 0.3277, 'progress_v3.txt (S5 ROOST / S5 ROOST_CAL lines)', seed_set='17/29/41', note='equals zero reference: no polymorph resolution')
    add('sg_s5_phase', 'comp', nm, 'A', 's5_dir_full', 0.241, 'progress_v3.txt (S5 ROOST / S5 ROOST_CAL lines)', seed_set='17/29/41')
for f in sorted(glob.glob(CV + 'phase_summary_v6comp6.json')):
    for m, d in json.load(open(f)).items():
        for k, val in (('phase_rho', d['spearman_rho']), ('phase_p', d['p']), ('phase_n', d['n_comp'])):
            add('sg_s5_phase', 'comp', 'PRE_NZ_6S', 'A', k, val, os.path.basename(f), seed_set='17/29/41/53/67/79', internal=m)
        add('sg_s5_phase', 'comp', 'PRE_NZ_6S', 'A', 's5_mae_full', 'not run', '-', seed_set='17/29/41/53/67/79', note='S5 evaluator averages exactly 3 seeds')
NOTRUN = [
    ('comp', 'PRE_NZ_X', 'metrics, S5 and phase rows for PRE_NZ on seeds 53/67/79', 'never trained'),
    ('comp', 'SGB_a_X', 'metrics, S5 and phase rows for SGB_a on seeds 53/67/79', 'never trained'),
    ('chem', 'SGB_a', 'all rows for SGB_a on chem folds', 'never trained'),
    ('chem', 'PRE_NZ_6S', 'all rows for PRE_NZ with 6 seeds on chem folds', 'never trained'),
    ('chem', 'SG_NZ_X', 'all rows for SG_NZ on chem folds with seeds 53/67/79', 'never trained'),
    ('comp', 'NO_TIER_B', 'ablation "no tier-B term" under the final GS_SELCAL_SC recipe', 'not run (only the older H_AUX-recipe GS_noB exists in metrics.csv, a different setting)'),
    ('chem', 'NO_TIER_B', 'ablation "no tier-B term" on chem folds', 'not run'),
    ('chem', 'ENC_FROZEN', 'frozen-encoder ablation on chem folds', 'not run'),
    ('chem', 'HSE_plain', 'HSE pool with zeros kept on chem folds', 'not run'),
    ('chem', 'PRE_NEG', 'PRE_NEG on chem folds', 'not run'),
    ('comp', 'EASYUQ_ON_GS_SG_NZ', 'EasyUQ on the SG_NZ median for seed set 1 (17/29/41)', 'not run (only seed-set-2 and chem versions exist)'),
    ('chem', 'S5_baselines', 'S5 pair MAE/direction for ROOST and ROOST_CAL on chem folds', 'not run'),
    ('comp', 'S5_PRE_NZ_6S', 'S5 pair MAE/direction for PRE_NZ with 6 seeds', 'not run'),
]
for fold, m, what, why in NOTRUN:
    add('not_run', fold, m, '-', 'status', 'not run', '-', note=f'{what}: {why}')

# ------------------------------------------------------------------ 5. ablation ladder
LADDER = [('ENC_FROZEN', 'frozen encoder (enc lr x0, SG joint, SELCAL recipe)'), ('GS_SELCAL_SC', 'no HSE pretraining (MP init only)'),
          ('NO_TIER_B', 'no tier-B term'), ('HSE_plain', 'HSE pool with zeros kept (all 84,729 rows)'), ('PRE_NEG', 'PRE_NEG (negatives dropped, zeros kept)'),
          ('GS_BEST', 'early stopping on raw validation CRPS, patience 30 (GS_G20_L1_CRPS_SC = GS_BEST of the handover)')]
for p, desc in LADDER:
    if ('comp', p) not in UT:
        add('ablation', 'comp', p, 'pooled', 'd_crps_vs_PRE_NZ', 'not run', '-', ref='PRE_NZ', note=desc)
        continue
    for t in ('A', 'B', 'pooled'):
        d = BOOT[('comp', p, 'PRE_NZ', t, 'crps')]
        add('ablation', 'comp', p, t, 'd_crps_vs_PRE_NZ', d[0], f'predictions_summary{UT[("comp", p)][2]}.csv', ref='PRE_NZ', lo=d[1], hi=d[2], note=desc)
for t in ('A', 'B', 'pooled'):
    d = BOOT[('comp', 'GS_BEST', 'GS_SELCAL_SC', t, 'crps')]
    add('ablation', 'comp', 'GS_BEST', t, 'd_crps_vs_GS_SELCAL_SC', d[0], f'predictions_summary{UT[("comp", "GS_BEST")][2]}.csv', ref='GS_SELCAL_SC', lo=d[1], hi=d[2],
        note='early-stopping ablation: GS_BEST vs GS_SELCAL (neither has HSE pretraining)')

# ------------------------------------------------------------------ 7. floor, sweep
import evaluate_cv as ev
Aidx = {}
for j, k in enumerate(ev.Akey):
    Aidx.setdefault(k, []).append(j)
floorA, floorB = {}, {}
for k, ii in Aidx.items():
    y = np.array([float(ev.A[j]['gap_ev']) for j in ii])
    w = ev.wA[ii] / ev.wA[ii].sum()
    if len(set(np.round(y, 9))) >= 2:
        floorA[k] = 0.5 * float(w @ np.abs(y[:, None] - y[None]) @ w)
for k, vv in ev.Bvals.items():
    y = np.array(vv, float)
    if len(y) >= 2:
        floorB[k] = 0.5 * float(np.abs(y[:, None] - y[None]).mean())
fA, fB = float(np.mean(list(floorA.values()))), float(np.mean(list(floorB.values())))
fl = pd.read_csv(CV + 'floor_v3.csv')
chk = abs(fl[fl.tier == 'A'].floor.iloc[0] - fA) + abs(fl[fl.tier == 'B'].floor.iloc[0] - fB)
CONS.append(f'floor recomputed in the ledger script vs floor_v3.csv: summed absolute difference {chk:.3g}')
add('floor', 'comp', '-', 'A', 'floor_mean_abs_diff_half', fA, 'recomputed (analysis_v3.py definition); equals floor_v3.csv', n=len(floorA), note='units: compositions with >=2 distinct reported values')
add('floor', 'comp', '-', 'B', 'floor_mean_abs_diff_half', fB, 'recomputed (analysis_v3.py definition); equals floor_v3.csv', n=len(floorB), note='units: compositions with >=2 distinct values')
add('floor', 'comp', '-', 'pooled', 'floor_mean_abs_diff_half', (fA * len(floorA) + fB * len(floorB)) / (len(floorA) + len(floorB)), 'derived: unit-weighted mean of the tier A and tier B floors',
    n=len(floorA) + len(floorB))
for (v, p), (ca, b, s, i) in UT.items():
    if v != 'comp' or (GRP[(v, p)] == 'baseline' and p not in ('ROOST', 'ROOST_CAL', 'EASYUQ_KS')):
        continue
    mA = float(ca[ca.key.isin(floorA)].crps.mean())
    mB = float(b[b.key.isin(floorB)].crps.mean())
    for t, m, fm, n in (('A', mA, fA, len(floorA)), ('B', mB, fB, len(floorB))):
        add('floor', v, p, t, 'crps_on_floor_units', m, f'predictions_summary{s}.csv', n=n, internal=i)
        add('floor', v, p, t, 'excess_over_floor', m - fm, f'predictions_summary{s}.csv', n=n, internal=i, note='CRPS on multi-value units minus information floor; tier A = full (all-inputs) view')
for r in fl.itertuples():
    add('floor', 'comp', r.model, r.tier, 'floor_v3_formula_only_crps', float(r.crps_formula_only), 'floor_v3.csv', n=int(r.n_units), note='older analysis file, formula-only view, round-v3 models')
    add('floor', 'comp', r.model, r.tier, 'floor_v3_excess_formula_only', float(r.excess_formula_only), 'floor_v3.csv', n=int(r.n_units))
    add('floor', 'comp', r.model, r.tier, 'floor_v3_sg_gain', float(r.sg_gain), 'floor_v3.csv', n=int(r.n_units), note='formula-only minus formula+SG tier-A CRPS (negative: SG hurt); round-v3 recipe')
pc = pd.read_csv(CV + 'phase_check_v2_GS_SGNZ_frozen_SC.csv')
for r in pc.itertuples():
    add('phase_inputs', 'comp', '-', 'A', 'omega2', float(r.omega2), 'phase_check_v2_GS_SGNZ_frozen_SC.csv', n=int(r.n_reports), internal=r.formula,
        note=f'{r.formula}: {int(r.n_phases)} phases, {int(r.n_reports)} reports; omega2 = between-phase share of reported-gap variance (model independent)')
add('phase_inputs', 'comp', '-', 'A', 'omega2_mean', float(pc.omega2.mean()), 'phase_check_v2_GS_SGNZ_frozen_SC.csv', n=len(pc))
sw = pd.read_csv(CV + 'sweep_v4_agg.csv')
for r in sw.itertuples():
    for c in sw.columns:
        if c == 'enc_lr_mult':
            continue
        t = 'B' if c.startswith('B_') else ('pooled' if 'pooled' in c else 'A')
        add('enc_lr_sweep', 'comp', f'ENC_LR_x{r.enc_lr_mult:g}', t, c, float(getattr(r, c)), 'sweep_v4_agg.csv', seed_set='17/29/41', note='15 runs per multiplier; joint SG; recipe otherwise frozen')

# ------------------------------------------------------------------ 8. counts
orig = pd.Series([x['origin'] for x in ev.A])
akeys = set(ev.Akey)
bkeys = set(ev.Bvals.keys())
ball = pd.read_csv(CV + 'data/tierB_all_rows.csv')
tb = pd.read_csv(CV + 'data/tierB_rows.csv')
CNT = [
    ('A', 'records', len(ev.A), 'evaluate_cv.A (data/tierA.pt)'), ('A', 'records_core', int((orig == 'core').sum()), 'evaluate_cv.A origin'),
    ('A', 'records_external', int((orig != 'core').sum()), 'evaluate_cv.A origin'), ('A', 'compositions', len(akeys), 'evaluate_cv.Akey'),
    ('A', 'compositions_core', len({k for k, o in zip(ev.Akey, orig) if o == 'core'}), 'evaluate_cv.A origin'),
    ('A', 'compositions_with_>=2_distinct_values', len(floorA), 'floor units'),
    ('B', 'rows_release_EX_csv', len(ball), 'data/tierB_all_rows.csv'),
    ('B', 'distinct_(composition,value)_pairs', sum(len(x) for x in ev.Bvals.values()), 'evaluate_cv.Bvals'),
    ('B', 'compositions', len(bkeys), 'evaluate_cv.Bvals'), ('B', 'compositions_with_>=2_distinct_values', len(floorB), 'floor units'),
    ('B', 'rows_in_data/tierB_rows.csv', len(tb), 'data/tierB_rows.csv'),
    ('A+B', 'overlap_compositions', len(akeys & bkeys), 'intersection of keys'), ('A+B', 'union_compositions', len(akeys | bkeys), 'union of keys'),
]
hp = pd.read_csv('/data/czx/GapScope_Revision_20261113/hse/hse_pool.csv')
hc = json.load(open('/data/czx/GapScope_Revision_20261113/hse/hse_pool_counts.json'))
CNT += [('HSE pool', 'rows_before_filter', len(hp), 'hse/hse_pool.csv'), ('HSE pool', 'distinct_keys', int(hp.key.nunique()), 'hse/hse_pool.csv'),
        ('HSE pool', 'rows_target_>0.01 (PRE_NZ filter, before test-fold key removal)', int((hp.target > 0.01).sum()), 'hse/hse_pool.csv'),
        ('HSE pool', 'rows_target_==0', int((hp.target == 0).sum()), 'hse/hse_pool.csv'), ('HSE pool', 'rows_target_<0', int((hp.target < 0).sum()), 'hse/hse_pool.csv'),
        ('HSE pool', 'rows_0<target<=0.01', int(((hp.target > 0) & (hp.target <= 0.01)).sum()), 'hse/hse_pool.csv'),
        ('HSE pool', 'final_rows (hse_pool_counts.json)', hc['final_rows'], 'hse/hse_pool_counts.json'),
        ('HSE pool', 'zero_gap_rows (hse_pool_counts.json)', hc['final_zero_gap_rows'], 'hse/hse_pool_counts.json')]
for fold in ('comp', 'chem'):
    kept = []
    for f in glob.glob(CV + f'ft/out/gs_out_v2/{fold}_GS_SELCAL_PRE_NZ_seed*_fold*_info.json'):
        d = json.load(open(f))
        if d.get('hse'):
            kept.append(d['hse']['rows_kept'])
    if kept:
        CNT.append(('HSE pool', f'rows_kept_per_run_{fold}_PRE_NZ min-max (n_runs={len(kept)})', f'{min(kept)}-{max(kept)}', 'ft/out/gs_out_v2 info json (test-fold keys removed per fold)'))
import torch
ft = torch.load(CV + 'ft/ft_data.pt', weights_only=False)
sgm = np.asarray(ft['sg_mask']).astype(bool)
akey = np.array(ev.Akey)
sga = pd.read_csv(CV + 'sgB_pool_assignment.csv')
CNT += [('A', 'records_with_space_group', int(sgm.sum()), 'ft/ft_data.pt sg_mask'), ('A', 'compositions_with_space_group', len(set(akey[sgm])), 'ft/ft_data.pt sg_mask'),
        ('B', 'compositions_with_space_group (Materials Project)', len(sga), 'sgB_pool_assignment.csv'),
        ('B', 'compositions_with_unambiguous_space_group (n_sg_within_25meV==1; the SG_B labels)', int((sga.n_sg_within_25meV == 1).sum()), 'sgB_pool_assignment.csv'),
        ('B', 'compositions_with_exactly_one_MP_space_group', int((sga.n_sg == 1).sum()), 'sgB_pool_assignment.csv'),
        ('A+B', 'tier-B SG compositions that are also tier-A compositions', len(set(sga.key) & akeys), 'sgB_pool_assignment.csv & Akey')]
for t, m, vv, src in CNT:
    add('counts', '-', '-', t, m, vv, src)

df = pd.DataFrame(rows)
df.to_csv(CV + 'final_numbers.csv', index=False)
open(CV + 'final_consistency.txt', 'w').write('\n'.join(CONS) + '\nmissing models (no per-unit file): ' + str(MISSING) + '\n')
print(len(df), 'rows;', len(CONS), 'consistency lines; missing', MISSING)
