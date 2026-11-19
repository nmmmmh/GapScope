"""Renders cv/final_tables.md from cv/final_numbers.csv (no new numbers)."""
import pandas as pd, numpy as np
CV = '/data/czx/GapScope_Revision_20261113/cv/'
d = pd.read_csv(CV + 'final_numbers.csv', dtype={'value': object})
d['num'] = pd.to_numeric(d.value, errors='coerce')
out = []
P = out.append


def f4(x, k=4):
    return 'n/a' if x is None or (isinstance(x, float) and np.isnan(x)) else f'{x:.{k}f}'


def table(header, rows):
    P('| ' + ' | '.join(header) + ' |')
    P('|' + '|'.join(['---'] * len(header)) + '|')
    for r in rows:
        P('| ' + ' | '.join(str(x) for x in r) + ' |')
    P('')


def val(sec, fold, model, tier, metric, ref=None, col='num'):
    q = d[(d.section == sec) & (d.fold_type == fold) & (d.model == model) & (d.tier == tier) & (d.metric == metric)]
    if ref is not None:
        q = q[q.ref == ref]
    return None if q.empty else q.iloc[0]


def num(*a, **k):
    r = val(*a, **k)
    return float('nan') if r is None else r.num


def ci(fold, model, ref, tier, met='crps'):
    r = val('bootstrap', fold, model, tier, f'd_{met}_vs_{ref}', ref)
    if r is None:
        return 'not run'
    return f'{r.num:+.4f} [{r.ci95_lo:+.4f}, {r.ci95_hi:+.4f}]'


ORDER = {'comp': ['ROOST', 'ROOST_CAL', 'ROOST_CAL_CV', 'ROOST6_CAL', 'ROOSTX_CAL', 'EASYUQ_KS', 'EASYUQ_IDR', 'EASYUQ_KS_roostX', 'EASYUQ_IDR_roostX', 'EASYUQ_ON_GS_KS', 'EASYUQ_ON_GS_IDR',
                  'EASYUQ_ON_GS_SG_NZ_X_KS', 'EASYUQ_ON_GS_SG_NZ_X_IDR', 'PRE_NZ', 'PRE_NZ_6S', 'SG_NZ', 'SG_NZ_X', 'SGB_a', 'SGB_b', 'SGB_c', 'SGC_l1', 'SGC_l5',
                  'GS_SELCAL_SC', 'HSE_plain', 'PRE_NEG', 'GS_BEST', 'ENC_FROZEN'],
         'chem': ['ROOST', 'ROOST_CAL', 'ROOST_CAL_CV', 'EASYUQ_KS', 'EASYUQ_IDR', 'EASYUQ_ON_GS_KS', 'EASYUQ_ON_GS_IDR', 'EASYUQ_ON_GS_SG_NZ_KS', 'EASYUQ_ON_GS_SG_NZ_IDR',
                  'PRE_NZ', 'SG_NZ', 'GS_SELCAL_SC', 'GS_BEST']}
P('# GapScope final results ledger (tables)\n')
P('Rendered from `final_numbers.csv` (every cell has a row there with its source file). Fold types: `comp` = grouped by composition, `chem` = grouped by chemical system. '
  'Tiers: A = provenance-linked experimental records (85 compositions), B = compilation (2,467 compositions), pooled = equal weight per (tier, composition) unit. '
  'Paired bootstrap: over compositions, 4,000 draws, seed 20261113, model minus reference (negative = model better). `not run` = no run exists in that setting.\n')
P('Model names: ROOST_CAL = width factor fitted on Roost\'s internal 20% training-fold validation split; ROOST_CAL_CV = out-of-fold factor (footnote role); '
  'EASYUQ_KS/IDR = EasyUQ on the Roost 3-seed point prediction; EASYUQ_ON_GS_* = EasyUQ on the PRE_NZ median; PRE_NZ = main model (3 seeds), PRE_NZ_6S = 6 seeds; '
  'SG_NZ = space-group variant (sgp frozen); SG_NZ_X = SG_NZ trained on seeds 53/67/79 (mp_seed 17/29/41, 79->r41 an assumption), compared with ROOSTX_CAL / EASYUQ_KS_roostX (Roost on seeds 53/67/79); '
  'SGB_a = tier-B SG labels, sgp frozen, lambda 1; GS_SELCAL_SC = MP init, no HSE pretraining; HSE_plain = HSE pool with zeros kept; PRE_NEG = negatives dropped, zeros kept; '
  'GS_BEST = group20 validation, epoch on raw validation CRPS, patience 30 (arm GS_G20_L1_CRPS_SC); ENC_FROZEN = encoder lr x0 with the joint SG head (arm GS_ELR_m0_SC).\n')

P('## 1. CRPS, NLL, cov80, width80, MAE per tier\n')
for fold in ('comp', 'chem'):
    P(f'### 1.{1 if fold == "comp" else 2} {fold} folds\n')
    rows = []
    for m in ORDER[fold]:
        if val('metrics', fold, m, 'pooled', 'crps') is None:
            continue
        rows.append([m] + [f4(num('metrics', fold, m, t, k)) for k in ('crps', 'nll') for t in ('A', 'B', 'pooled')])
    table(['model', 'CRPS A', 'CRPS B', 'CRPS pooled', 'NLL A', 'NLL B', 'NLL pooled'], rows)
    rows = []
    for m in ORDER[fold]:
        if val('metrics', fold, m, 'pooled', 'crps') is None:
            continue
        rows.append([m] + [f4(num('metrics', fold, m, t, k), 3) for k in ('cov80', 'width80', 'mae') for t in ('A', 'B', 'pooled')])
    table(['model', 'cov80 A', 'cov80 B', 'cov80 pooled', 'width80 A', 'width80 B', 'width80 pooled', 'MAE A', 'MAE B', 'MAE pooled'], rows)

P('## 2. Paired composition bootstrap (CRPS; NLL and MAE pooled)\n')
for fold in ('comp', 'chem'):
    for ref in ('ROOST_CAL', 'EASYUQ_KS', 'PRE_NZ'):
        P(f'### {fold} folds, reference {ref}\n')
        rows = []
        for m in ORDER[fold]:
            if m == ref or val('bootstrap', fold, m, 'pooled', f'd_crps_vs_{ref}', ref) is None:
                continue
            rows.append([m] + [ci(fold, m, ref, t) for t in ('A', 'B', 'pooled')] + [ci(fold, m, ref, 'pooled', 'nll'), ci(fold, m, ref, 'pooled', 'mae')])
        table(['model', 'dCRPS A', 'dCRPS B', 'dCRPS pooled', 'dNLL pooled', 'dMAE pooled'], rows)
P('### comp folds, seed set 2 and 6-seed references\n')
rows = []
for m, ref in (('SG_NZ_X', 'ROOSTX_CAL'), ('SG_NZ_X', 'EASYUQ_KS_roostX'), ('EASYUQ_KS_roostX', 'ROOSTX_CAL'), ('EASYUQ_ON_GS_SG_NZ_X_KS', 'ROOSTX_CAL'), ('EASYUQ_ON_GS_SG_NZ_X_KS', 'EASYUQ_KS_roostX'),
               ('PRE_NZ_6S', 'ROOST6_CAL'), ('GS_BEST', 'GS_SELCAL_SC')):
    rows.append([m, ref] + [ci('comp', m, ref, t) for t in ('A', 'B', 'pooled')] + [ci('comp', m, ref, 'pooled', 'nll')])
table(['model', 'reference', 'dCRPS A', 'dCRPS B', 'dCRPS pooled', 'dNLL pooled'], rows)

P('## 3. Window screening (PV [1.1, 1.7] eV; visible [1.8, 3.1] eV; all 2,522 test compositions)\n')
for fold in ('comp', 'chem'):
    for win, nm in (('pv', 'PV'), ('visible', 'visible')):
        P(f'### {fold} folds, {nm} window\n')
        rows = []
        for m in ORDER[fold]:
            if val('windows', fold, m, 'all2522', f'{win}_brier_prob') is None:
                continue
            g = lambda k, dd=4: f4(num('windows', fold, m, 'all2522', f'{win}_{k}'), dd)
            gi = lambda k: str(int(num('windows', fold, m, 'all2522', f'{win}_{k}')))
            rows.append([m, g('brier_prob'), g('brier_hard'), g('auc_prob'), g('auc_dist'), f'{gi("top10_prob")}/{gi("top10_dist")}', f'{gi("top20_prob")}/{gi("top20_dist")}',
                         f'{gi("top50_prob")}/{gi("top50_dist")}', f4(num('windows', fold, m, 'all2522', f'ece_{win}'), 4), gi('n_any')])
        table(['model', 'Brier (prob)', 'Brier (hard)', 'AUC prob', 'AUC dist', 'top-10 hits prob/dist', 'top-20 prob/dist', 'top-50 prob/dist', 'ECE', 'n positives'], rows)

P('## 4. PIT histograms (10 equal-width bins; counts)\n')
for fold in ('comp', 'chem'):
    for t in ('A', 'B'):
        P(f'### {fold} folds, tier {t} ({"per record" if t == "A" else "per distinct value"})\n')
        rows = []
        for m in ORDER[fold]:
            r = val('pit', fold, m, t, 'pit_bin_counts')
            if r is None or r.value == 'not run':
                continue
            rows.append([m] + r.value.split())
        table(['model'] + [f'bin{j}' for j in range(1, 11)], rows)

P('## 5. Ablation ladder (comp folds; model minus main model PRE_NZ; positive = ablation worse)\n')
rows = []
for m in ('ENC_FROZEN', 'GS_SELCAL_SC', 'NO_TIER_B', 'HSE_plain', 'PRE_NEG', 'GS_BEST'):
    q = d[(d.section == 'ablation') & (d.model == m) & (d.metric == 'd_crps_vs_PRE_NZ')]
    desc = q.note.iloc[0] if len(q) else ''
    if q.empty or (q.value == 'not run').any():
        rows.append([m, desc, 'not run', 'not run', 'not run', 'not run'])
        continue
    pc = num('metrics', 'comp', m, 'pooled', 'crps')
    rows.append([m, desc, f4(pc)] + [ci('comp', m, 'PRE_NZ', t) for t in ('A', 'B', 'pooled')])
table(['ablation', 'what', 'CRPS pooled', 'vs PRE_NZ, tier A', 'tier B', 'pooled'], rows)
P(f'Main model PRE_NZ: CRPS A {f4(num("metrics", "comp", "PRE_NZ", "A", "crps"))}, B {f4(num("metrics", "comp", "PRE_NZ", "B", "crps"))}, pooled {f4(num("metrics", "comp", "PRE_NZ", "pooled", "crps"))}.\n')
P('Early-stopping ablation (GS_BEST versus GS_SELCAL_SC, same recipe otherwise):\n')
table(['comparison', 'dCRPS A', 'dCRPS B', 'dCRPS pooled'], [['GS_BEST - GS_SELCAL_SC'] + [ci('comp', 'GS_BEST', 'GS_SELCAL_SC', t) for t in ('A', 'B', 'pooled')]])

P('## 6. Space group: tier-A CRPS with / without SG, S5 pairs, phase check\n')
P('S5 zero reference (pair MAE of a model that resolves no polymorph contrast) = 0.3277; S5 has 22 pairs; phase check has 13 compositions. Test = out-of-fold test, val = mean over the 15 inner-validation runs.\n')
pairs = [('comp', 'PRE_NZ', '17/29/41'), ('comp', 'SG_NZ', '17/29/41'), ('comp', 'SG_NZ_X', '53/67/79 (mp_seed 17/29/41)'), ('comp', 'SGB_a', '17/29/41'), ('comp', 'SGB_b', '17/29/41'),
         ('comp', 'SGB_c', '17/29/41'), ('comp', 'SGC_l1', '17/29/41'), ('comp', 'SGC_l5', '17/29/41'), ('comp', 'GS_F0', '17/29/41'), ('comp', 'ENC_FROZEN', '17/29/41'),
         ('chem', 'PRE_NZ', '17/29/41'), ('chem', 'SG_NZ', '17/29/41')]
rows = []
for fold, m, ss in pairs:
    q = d[(d.section == 'sg_s5_phase') & (d.fold_type == fold) & (d.model == m) & (d.seed_set == ss)]
    g = lambda k, dd=4: (lambda r: 'n/a' if r.empty else (f'{float(r.value.iloc[0]):.{dd}f}' if r.value.iloc[0] not in ('not available',) else 'n/a'))(q[q.metric == k])
    rows.append([fold, m, ss, g('tierA_crps_withSG_test'), g('tierA_crps_noSG_test'), g('tierA_crps_withSG_val'), g('tierA_crps_noSG_val'), g('s5_mae_full'), g('s5_dir_full', 3),
                 g('phase_rho', 3), g('phase_p', 3), g('phase_n', 0), g('pooled_crps_formula_test')])
table(['fold', 'model', 'seed set', 'A test with SG', 'A test no SG', 'A val with SG', 'A val no SG', 'S5 MAE', 'S5 dir', 'phase rho', 'p', 'n', 'pooled formula-only test CRPS'], rows)
rows = []
for m in ('ROOST', 'ROOST_CAL'):
    rows.append([m, '0.3277', '0.241', 'progress_v3.txt'])
table(['baseline (comp)', 'S5 MAE', 'S5 dir', 'source'], rows)
P('Rows marked `not run` in the ledger for this section: ' + '; '.join(sorted(set(d[d.section == 'not_run'].note.str.split(':').str[0]))) + '.\n')

P('## 7. Information floor, encoder-lr sweep, phase-check inputs\n')
rows = []
for t in ('A', 'B', 'pooled'):
    r = val('floor', 'comp', '-', t, 'floor_mean_abs_diff_half')
    rows.append([t, int(r.n), f4(r.num)])
table(['tier', 'units (>=2 distinct values)', 'floor'], rows)
rows = []
for m in ('ROOST', 'ROOST_CAL', 'EASYUQ_KS', 'PRE_NZ', 'SG_NZ', 'SGB_a', 'GS_SELCAL_SC'):
    rows.append([m] + [f4(num('floor', 'comp', m, t, k)) for t in ('A', 'B') for k in ('crps_on_floor_units', 'excess_over_floor')])
table(['model', 'CRPS on multi-value units, A', 'excess A', 'CRPS B', 'excess B'], rows)
rows = []
q = d[(d.section == 'enc_lr_sweep')]
for mm in q.model.unique():
    g = lambda k, dd=4: f4(float(q[(q.model == mm) & (q.metric == k)].num.iloc[0]), dd)
    rows.append([mm, g('A_crps_nosg'), g('A_crps_sg'), g('B_crps'), g('pooled_crps_formula'), g('s5_mae_full'), g('s5_dir_full', 3), g('phase_rho', 3), g('enc_rel_dist'), g('enc_cka')])
table(['multiplier', 'A no SG', 'A with SG', 'B', 'pooled formula', 'S5 MAE', 'S5 dir', 'phase rho', 'enc rel. dist', 'enc CKA'], rows)
q = d[(d.section == 'phase_inputs') & (d.metric == 'omega2')]
P('Phase-check inputs (omega2 = between-phase share of the variance of reported gaps, per composition): ' + '; '.join(f'{r.internal_name} {r.num:.3f} (n={int(r.n)})' for r in q.itertuples()) +
  f'; mean {num("phase_inputs", "comp", "-", "A", "omega2_mean"):.3f}. A separate variance-decomposition file does not exist on the server beyond these per-composition values and floor_v3.csv.\n')

P('## 8. Dataset counts\n')
q = d[d.section == 'counts']
table(['tier / set', 'quantity', 'value', 'source'], [[r.tier, r.metric, r.value, r.source] for r in q.itertuples()])

P('## 9. Cells with no run\n')
q = d[d.section == 'not_run']
table(['fold type', 'item', 'what is missing'], [[r.fold_type, r.model, r.note] for r in q.itertuples()])
open(CV + 'final_tables.md', 'w').write('\n'.join(out))
print(len(out), 'lines')
