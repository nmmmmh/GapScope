"""PRIMARY QUESTION: what chemistry sets how WIDE a composition's experimental band-gap
distribution is?

Targets: per-composition spread of the reported experimental gaps, on the 237-record
provenance set and on the 3,728-row compilation set separately.
Predictors: per-composition SNUMAT polymorph statistics and composition-only chemistry.
Everything is evaluated with whole chemical systems held out.
Writes width.csv (one row per composition, both sets stacked) and tables/width_*.csv.
Usage: python -I width.py
"""
import sys, json, math, warnings
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from pathlib import Path
from scipy import stats as sps
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance

sys.path.insert(0, '/data/czx/GapScope_Revision_20261113/struct')
from desc_worker import classify, nval, rad
from pymatgen.core import Element

D = Path('/data/czx/GapScope_Revision_20261113/struct')
T = D / 'tables'
T.mkdir(exist_ok=True)
SUPP = Path('/data/czx/GapScope_Revision_20261113/suppdata/Supplementary_Data_1_GapScope_experimental_bandgaps.csv')
TIERB = Path('/data/czx/GapScope_Revision_20261113/cv/data/tierB_rows.csv')
RNG = np.random.default_rng(0)

HAL = {'F', 'Cl', 'Br', 'I'}
CHA = {'S', 'Se', 'Te'}
PNI = {'N', 'P', 'As', 'Sb', 'Bi'}

# spread of these descriptors across a composition's SNUMAT polymorphs
SPREAD_OF = ['cn_cat_mean', 'cn_cat_max', 'cn_all_mean', 'mean_shared_vertices', 'frac_corner',
             'frac_edge', 'frac_face', 'dim_cat_an', 'bond_cat_an_mean', 'distortion_index',
             'vol_per_atom', 'packing_fraction', 'n_symops', 'frac_isolated_poly']


def parse_key(k):
    out = {}
    for part in k.split('|'):
        el, fr = part.split(':')
        out[el] = float(fr)
    return out


def chem_features(key):
    try:
        fr = parse_key(key)
    except Exception:
        return None
    els = list(fr)
    try:
        cat, an, X = classify(els)
    except Exception:
        return None
    wc = sum(fr[e] for e in cat)
    wa = sum(fr[e] for e in an)
    o = dict(n_elements=len(els), anion_frac=wa)
    o['en_an_mean'] = sum(fr[e] * X[e] for e in an) / wa
    o['en_cat_mean'] = (sum(fr[e] * X[e] for e in cat) / wc) if wc > 0 else np.nan
    o['en_diff'] = o['en_an_mean'] - o['en_cat_mean'] if wc > 0 else np.nan
    o['ionicity'] = 1 - math.exp(-0.25 * o['en_diff'] ** 2) if wc > 0 else np.nan
    o['en_std_all'] = float(np.std([X[e] for e in els]))
    o['nval_per_atom'] = sum(fr[e] * nval(Element(e)) for e in els)
    o['radius_ratio'] = ((sum(fr[e] * rad(Element(e)) for e in cat) / wc) /
                         (sum(fr[e] * rad(Element(e)) for e in an) / wa)) if wc > 0 else np.nan
    o['mean_atomic_radius'] = sum(fr[e] * rad(Element(e)) for e in els)
    dom = max(an, key=lambda e: fr[e])
    o['is_halide'] = int(bool(an & HAL))
    o['is_oxide'] = int('O' in an)
    o['is_chalcogenide'] = int(bool(an & CHA) and 'O' not in an)
    o['is_pnictide'] = int(bool(an & PNI) and not (an & HAL) and 'O' not in an and not (an & CHA))
    o['is_intermetallic'] = int(not (an & (HAL | CHA | PNI | {'O', 'H', 'C', 'B'})))
    o['dom_anion_Z'] = Element(dom).Z
    o['elem_system'] = ','.join(sorted(els))
    o['cation_system'] = ','.join(sorted(cat)) if cat else ''
    o['anion_system'] = ','.join(sorted(an))
    return o


def snumat_features(d):
    """per composition: polymorph count, spread of the descriptors across polymorph cells,
    spread of the computed gaps"""
    rows = []
    for ck, g in d.groupby('composition_key'):
        cells = g.groupby('Space_group_rlx')
        rep = cells.apply(lambda q: q.iloc[(q.Band_gap_HSE - q.Band_gap_HSE.median()).abs().values.argsort()[0]])
        o = dict(composition_key=ck, n_polymorphs=g.Space_group_rlx.nunique(), n_snumat=len(g),
                 hse_mean=float(rep.Band_gap_HSE.mean()), hse_min=float(rep.Band_gap_HSE.min()),
                 n_crystal_systems=int(rep.crystal_system.nunique()),
                 any_noncentro=int((rep.centrosymmetric == 0).any()),
                 frac_direct=float((rep.Direct_or_indirect_HSE.astype(str) == 'Direct').mean()),
                 mixed_direct_indirect=int(rep.Direct_or_indirect_HSE.astype(str).nunique() > 1))
        for k in SPREAD_OF:
            v = rep[k].astype(float).values
            v = v[np.isfinite(v)]
            o['spr_' + k] = float(v.max() - v.min()) if len(v) > 1 else (0.0 if len(v) == 1 else np.nan)
            o['sd_' + k] = float(v.std(ddof=1)) if len(v) > 1 else (0.0 if len(v) == 1 else np.nan)
            o['mean_' + k] = float(v.mean()) if len(v) else np.nan
        for k, nm in (('Band_gap_HSE', 'hse'), ('Band_gap_HSE_optical', 'hseopt'), ('Band_gap_GGA', 'gga')):
            v = rep[k].astype(float).values
            o['gapspr_' + nm] = float(v.max() - v.min()) if len(v) > 1 else 0.0
            o['gapsd_' + nm] = float(v.std(ddof=1)) if len(v) > 1 else 0.0
        o['var_coordination'] = int(o['spr_cn_cat_mean'] >= 0.5) if np.isfinite(o['spr_cn_cat_mean']) else 0
        o['var_dimensionality'] = int(o['spr_dim_cat_an'] >= 1) if np.isfinite(o['spr_dim_cat_an']) else 0
        o['var_connectivity'] = int(max(o['spr_frac_face'] if np.isfinite(o['spr_frac_face']) else 0,
                                        o['spr_frac_edge'] if np.isfinite(o['spr_frac_edge']) else 0) >= 0.2)
        o['multi_polymorph'] = int(o['n_polymorphs'] > 1)
        rows.append(o)
    return pd.DataFrame(rows)


def targets(df, key, val, name):
    rows = []
    for ck, g in df.groupby(key):
        v = g[val].astype(float).dropna().values
        rows.append(dict(composition_key=ck, set=name, n_reports=len(v),
                         exp_range=float(v.max() - v.min()) if len(v) > 1 else np.nan,
                         exp_sd=float(v.std(ddof=1)) if len(v) > 1 else np.nan,
                         exp_iqr=float(np.percentile(v, 75) - np.percentile(v, 25)) if len(v) > 1 else np.nan,
                         exp_median=float(np.median(v))))
    return pd.DataFrame(rows)


def oof(X, y, groups, mk, nsplits=5):
    pred = np.full(len(y), np.nan)
    gk = GroupKFold(n_splits=min(nsplits, len(np.unique(groups))))
    for tr, te in gk.split(X, y, groups):
        m = mk()
        m.fit(X[tr], y[tr])
        pred[te] = m.predict(X[te])
    return pred


def score(y, pred):
    return dict(spearman=float(sps.spearmanr(pred, y).statistic),
                pearson=float(sps.pearsonr(pred, y).statistic),
                R2=float(1 - np.sum((y - pred) ** 2) / np.sum((y - y.mean()) ** 2)),
                MAE=float(np.abs(y - pred).mean()))


def main():
    d = pd.read_csv(D / 'descriptors.csv')
    d = d[d.Band_gap_HSE.notna() & d.Space_group_rlx.notna()]
    S = snumat_features(d)
    print('SNUMAT compositions', len(S))

    sup = pd.read_csv(SUPP)
    tb = pd.read_csv(TIERB)
    tp = targets(sup, 'composition_key', 'bandgap_ev', 'provenance')
    tc = targets(tb, 'composition_key', 'target_ev', 'compilation')
    print('provenance compositions %d (with >=2 reports %d); compilation compositions %d (with >=2 reports %d)'
          % (len(tp), int(tp.exp_range.notna().sum()), len(tc), int(tc.exp_range.notna().sum())))

    W = pd.concat([tp, tc], ignore_index=True)
    ch = pd.DataFrame([dict(composition_key=k, **(chem_features(k) or {})) for k in W.composition_key.unique()])
    W = W.merge(ch, on='composition_key', how='left').merge(S, on='composition_key', how='left')
    W['in_snumat'] = W.n_polymorphs.notna().astype(int)
    W.to_csv(D / 'width.csv', index=False)
    print('width.csv rows', len(W), 'matched to SNUMAT', int(W.in_snumat.sum()))

    # ---- 0. the initial-support claim, reproduced
    print('\n== initial claim: experimental gap range by number of SNUMAT space groups ==')
    for nm, g in W.groupby('set'):
        q = g[g.in_snumat == 1]
        for lab, m in (('1 space group', q.n_polymorphs == 1), ('>=2 space groups', q.n_polymorphs > 1)):
            r = q[m]
            print('%-12s %-17s n=%3d  median range %.3f  mean %.3f  (>=2 reports only: n=%d median %.3f)'
                  % (nm, lab, len(r), r.exp_range.fillna(0).median(), r.exp_range.fillna(0).mean(),
                     int(r.exp_range.notna().sum()), r.exp_range.median()))
        a = q[(q.n_polymorphs == 1)].exp_range.fillna(0)
        b = q[(q.n_polymorphs > 1)].exp_range.fillna(0)
        if len(a) > 3 and len(b) > 3:
            print('   Mann-Whitney p = %.4g (zeros kept for single-report compositions)'
                  % sps.mannwhitneyu(a, b).pvalue)
        a2 = q[(q.n_polymorphs == 1)].exp_range.dropna()
        b2 = q[(q.n_polymorphs > 1)].exp_range.dropna()
        if len(a2) > 3 and len(b2) > 3:
            print('   Mann-Whitney p = %.4g (>=2-report compositions only, n=%d vs %d)'
                  % (sps.mannwhitneyu(a2, b2).pvalue, len(a2), len(b2)))

    # ---- 1. the sample-size confound
    print('\n== how much of the observed spread is just the number of reports ==')
    for nm, g in W.groupby('set'):
        q = g[g.exp_range.notna()]
        print('%-12s n=%4d  Spearman(range, n_reports) = %+.3f   Spearman(sd, n_reports) = %+.3f'
              % (nm, len(q), sps.spearmanr(q.exp_range, q.n_reports).statistic,
                 sps.spearmanr(q.exp_sd, q.n_reports).statistic))

    # ---- 2. negative control: computed polymorph gap spread vs experimental spread
    print('\n== negative control: computed polymorph gap spread vs experimental spread ==')
    rows = []
    for nm, g in W.groupby('set'):
        q = g[(g.exp_range.notna()) & (g.in_snumat == 1)]
        for tgt in ('exp_range', 'exp_sd', 'exp_iqr'):
            for pr in ('gapspr_hse', 'gapsd_hse', 'gapspr_hseopt', 'gapspr_gga', 'n_polymorphs',
                       'spr_cn_cat_mean', 'spr_dim_cat_an', 'spr_mean_shared_vertices',
                       'spr_bond_cat_an_mean', 'spr_distortion_index', 'spr_vol_per_atom',
                       'var_coordination', 'var_dimensionality', 'var_connectivity', 'multi_polymorph',
                       'n_crystal_systems', 'hse_mean', 'exp_median', 'n_reports'):
                x = q[pr].astype(float).values
                y = q[tgt].values
                ok = np.isfinite(x) & np.isfinite(y)
                if ok.sum() < 10 or np.std(x[ok]) == 0:
                    continue
                r = sps.spearmanr(x[ok], y[ok])
                # partial Spearman controlling for the number of reports
                rn = sps.spearmanr(x[ok], q.n_reports.values[ok]).statistic
                ry = sps.spearmanr(y[ok], q.n_reports.values[ok]).statistic
                part = (r.statistic - rn * ry) / math.sqrt(max(1e-12, (1 - rn ** 2) * (1 - ry ** 2)))
                rows.append(dict(set=nm, target=tgt, predictor=pr, n=int(ok.sum()),
                                 spearman=float(r.statistic), p=float(r.pvalue),
                                 spearman_partial_on_n=float(part)))
    U = pd.DataFrame(rows)
    U.to_csv(T / 'width_univariate.csv', index=False)
    for nm in U.set.unique():
        print('\n-- %s, target exp_range, sorted by |Spearman| --' % nm)
        print(U[(U.set == nm) & (U.target == 'exp_range')].reindex(
            U[(U.set == nm) & (U.target == 'exp_range')].spearman.abs().sort_values(ascending=False).index
        ).head(14).to_string(index=False))

    # ---- 3. models
    chemf = ['n_elements', 'anion_frac', 'en_an_mean', 'en_cat_mean', 'en_diff', 'ionicity', 'en_std_all',
             'nval_per_atom', 'radius_ratio', 'mean_atomic_radius', 'is_halide', 'is_oxide',
             'is_chalcogenide', 'is_pnictide', 'is_intermetallic', 'dom_anion_Z']
    strucf = ['n_polymorphs', 'n_crystal_systems', 'any_noncentro', 'frac_direct', 'mixed_direct_indirect',
              'var_coordination', 'var_dimensionality', 'var_connectivity'] + \
             ['spr_' + k for k in SPREAD_OF] + ['mean_' + k for k in SPREAD_OF]
    gapf = ['gapspr_hse', 'gapsd_hse', 'gapspr_hseopt', 'gapspr_gga']
    levelf = ['hse_mean']
    nf = ['n_reports']
    sets = [('n_reports only (baseline)', nf),
            ('composition chemistry only', chemf),
            ('chemistry + n_reports', chemf + nf),
            ('chemistry + SNUMAT structure (no computed gap spread)', chemf + strucf + levelf),
            ('chemistry + structure + n_reports', chemf + strucf + levelf + nf),
            ('everything incl. computed gap spread', chemf + strucf + levelf + gapf),
            ('everything + n_reports', chemf + strucf + levelf + gapf + nf),
            ('computed gap spread only', gapf)]
    res = []
    for nm, g in W.groupby('set'):
        for snum_only in (True, False):
            q = g[g.exp_range.notna()]
            if snum_only:
                q = q[q.in_snumat == 1]
            if len(q) < 40:
                continue
            for tgt in ('exp_range', 'exp_sd'):
                y = q[tgt].values
                gr = q.elem_system.fillna('?').values
                for fname, fl in sets:
                    fl2 = [f for f in fl if f in q.columns]
                    X = q[fl2].values.astype(float)
                    if snum_only is False and any(f in strucf + gapf + levelf for f in fl2):
                        continue
                    Xr = np.where(np.isnan(X), np.nanmedian(np.where(np.isnan(X), np.nan, X), axis=0), X)
                    Xr = np.where(np.isnan(Xr), 0.0, Xr)
                    sc = StandardScaler().fit(Xr)
                    for mn, mk, Xu in (('ridge', lambda: Ridge(alpha=3.0), sc.transform(Xr)),
                                       ('gbm', lambda: HistGradientBoostingRegressor(
                                           max_iter=300, learning_rate=0.05, max_depth=3,
                                           min_samples_leaf=5, random_state=0), X)):
                        pred = oof(Xu, y, gr, mk)
                        r = score(y, pred)
                        r.update(set=nm, target=tgt, features=fname, model=mn, n=len(q),
                                 snumat_matched_only=snum_only, n_features=len(fl2))
                        res.append(r)
    R = pd.DataFrame(res)[['set', 'snumat_matched_only', 'target', 'features', 'model', 'n', 'n_features',
                           'spearman', 'pearson', 'R2', 'MAE']]
    R.to_csv(T / 'width_models.csv', index=False)
    print('\n== width models, whole element systems held out (5-fold GroupKFold) ==')
    for nm in R.set.unique():
        for so in (True, False):
            q = R[(R.set == nm) & (R.snumat_matched_only == so) & (R.target == 'exp_range')]
            if len(q):
                print('\n-- %s, target exp_range, %s --' % (nm, 'SNUMAT-matched compositions' if so else 'all compositions'))
                print(q[['features', 'model', 'n', 'n_features', 'spearman', 'R2', 'MAE']].to_string(index=False))
    print('\n-- target exp_sd, SNUMAT-matched --')
    print(R[(R.target == 'exp_sd') & (R.snumat_matched_only)][
        ['set', 'features', 'model', 'n', 'spearman', 'R2']].to_string(index=False))

    # ---- 4. which predictors carry the best model
    print('\n== permutation importance of the best feature set (chemistry + structure + gap spread) ==')
    for nm, g in W.groupby('set'):
        q = g[(g.exp_range.notna()) & (g.in_snumat == 1)]
        if len(q) < 40:
            continue
        fl = [f for f in chemf + strucf + levelf + gapf if f in q.columns]
        X = q[fl].values.astype(float)
        y = q.exp_range.values
        gr = q.elem_system.fillna('?').values
        imp = np.zeros(len(fl))
        gk = GroupKFold(n_splits=5)
        for tr, te in gk.split(X, y, gr):
            m = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_depth=3,
                                              min_samples_leaf=5, random_state=0).fit(X[tr], y[tr])
            imp += permutation_importance(m, X[te], y[te], n_repeats=10, random_state=0,
                                          scoring='neg_mean_absolute_error').importances_mean * len(te)
        s = pd.Series(-imp / len(y), index=fl).sort_values(ascending=False)
        print('\n-- %s (n=%d), eV of MAE --' % (nm, len(q)))
        print(s.head(12).round(4).to_string())
        s.to_csv(T / ('width_importance_%s.csv' % nm))

    # ---- 5. coarse categorical predictors, the part that could survive
    print('\n== coarse categorical contrasts (median experimental range, eV) ==')
    rows = []
    for nm, g in W.groupby('set'):
        q = g[(g.exp_range.notna()) & (g.in_snumat == 1)]
        for flag in ('multi_polymorph', 'var_coordination', 'var_dimensionality', 'var_connectivity',
                     'mixed_direct_indirect', 'any_noncentro', 'is_halide', 'is_oxide', 'is_intermetallic'):
            a = q[q[flag] == 0].exp_range
            b = q[q[flag] == 1].exp_range
            if len(a) < 5 or len(b) < 5:
                continue
            rows.append(dict(set=nm, flag=flag, n0=len(a), n1=len(b),
                             median0=float(a.median()), median1=float(b.median()),
                             mean0=float(a.mean()), mean1=float(b.mean()),
                             p_mw=float(sps.mannwhitneyu(a, b).pvalue),
                             median_n_reports0=float(q[q[flag] == 0].n_reports.median()),
                             median_n_reports1=float(q[q[flag] == 1].n_reports.median())))
        for k in (1, 2, 3):
            m = q.n_polymorphs >= k
            rows.append(dict(set=nm, flag='n_polymorphs>=%d' % k, n0=int((~m).sum()), n1=int(m.sum()),
                             median0=float(q[~m].exp_range.median()) if (~m).sum() else np.nan,
                             median1=float(q[m].exp_range.median()),
                             mean0=float(q[~m].exp_range.mean()) if (~m).sum() else np.nan,
                             mean1=float(q[m].exp_range.mean()),
                             p_mw=float(sps.mannwhitneyu(q[~m].exp_range, q[m].exp_range).pvalue)
                             if (~m).sum() > 4 and m.sum() > 4 else np.nan,
                             median_n_reports0=float(q[~m].n_reports.median()) if (~m).sum() else np.nan,
                             median_n_reports1=float(q[m].n_reports.median())))
    C = pd.DataFrame(rows)
    C.to_csv(T / 'width_categorical.csv', index=False)
    print(C.to_string(index=False))


if __name__ == '__main__':
    main()
