"""Stage 1 steps 3-4: same-composition polymorph pairs inside SNUMAT and what explains
the gap difference.  Writes pairs.csv, report_tables/*.csv and prints the report numbers.
Usage: python -I pairs_analysis.py
"""
import json, itertools, warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats as sps
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import StandardScaler

D = Path('/data/czx/GapScope_Revision_20261113/struct')
T = D / 'tables'
T.mkdir(exist_ok=True)
RNG = np.random.default_rng(0)

# descriptors that are a property of the structure (differ between polymorphs)
STRUCT = ['vol_per_atom', 'density', 'packing_fraction',
          'cn_cat_mean', 'cn_cat_min', 'cn_cat_max', 'cn_cat_std', 'cn_cat_spread',
          'cn_an_mean', 'cn_an_min', 'cn_an_max', 'cn_an_std', 'cn_an_spread', 'cn_all_mean',
          'dom_cn_cat', 'dom_poly_frac',
          'bond_cat_an_mean', 'bond_cat_an_std', 'bond_cat_an_rel_std',
          'distortion_index', 'distortion_index_max',
          'conn_per_cation', 'frac_corner', 'frac_edge', 'frac_face',
          'mean_shared_vertices', 'max_shared_vertices', 'frac_isolated_poly',
          'dim_all', 'dim_cat_an', 'bvs_cat_mean', 'bvs_dev_mean',
          'n_symops', 'centrosymmetric']
# descriptors fixed by the composition (identical inside a pair -> difference is 0 by construction)
CHEM = ['anion_frac', 'en_cat_mean', 'en_an_mean', 'en_diff', 'ionicity', 'en_std_all',
        'nval_per_atom', 'radius_ratio', 'mean_atomic_radius']


def build_cells(d):
    """one representative structure per (composition_key, relaxed space group) cell:
    the record whose HSE gap is closest to the cell median (benchmark/README uses the
    cell median; this keeps the descriptors those of a single real structure)."""
    rows = []
    for (ck, sg), g in d.groupby(['composition_key', 'Space_group_rlx']):
        med = g.Band_gap_HSE.median()
        r = g.iloc[(g.Band_gap_HSE - med).abs().values.argsort()[0]].to_dict()
        r['n_records_in_cell'] = len(g)
        r['cell_gap_spread'] = float(g.Band_gap_HSE.max() - g.Band_gap_HSE.min())
        for k in ('Band_gap_HSE', 'Band_gap_HSE_optical', 'Band_gap_GGA', 'Band_gap_GGA_optical'):
            r[k] = float(g[k].median())
        rows.append(r)
    return pd.DataFrame(rows)


def build_pairs(cells):
    out = []
    for ck, g in cells.groupby('composition_key'):
        if len(g) < 2:
            continue
        g = g.sort_values('Space_group_rlx')
        for (_, a), (_, b) in itertools.combinations(list(g.iterrows()), 2):
            r = dict(composition_key=ck, formula=a['formula'],
                     elem_system=','.join(sorted(set(a['cations'].split(',') if isinstance(a['cations'], str) and a['cations'] else []) |
                                                 set(a['anions'].split(',')))),
                     anion_system=a['anions'],
                     sg_a=int(a['Space_group_rlx']), sg_b=int(b['Space_group_rlx']),
                     id_a=a['snumat_id'], id_b=b['snumat_id'],
                     cs_a=a['crystal_system'], cs_b=b['crystal_system'],
                     same_crystal_system=int(a['crystal_system'] == b['crystal_system']))
            for k in ('Band_gap_HSE', 'Band_gap_HSE_optical', 'Band_gap_GGA'):
                r['d_' + k] = float(b[k]) - float(a[k])
                r[k + '_a'] = float(a[k])
                r[k + '_b'] = float(b[k])
            for k in STRUCT:
                r['d_' + k] = float(b[k]) - float(a[k])
                r['m_' + k] = 0.5 * (float(b[k]) + float(a[k]))
            for k in CHEM:
                r['c_' + k] = float(a[k])
                r['dchem_' + k] = float(b[k]) - float(a[k])
            out.append(r)
    p = pd.DataFrame(out)
    n = p.groupby('composition_key').size()
    p['weight'] = 1.0 / p.composition_key.map(n) / len(n)
    return p


def wR2(y, p, w):
    w = w / w.sum()
    ybar = w @ y
    return 1 - (w @ (y - p) ** 2) / (w @ (y - ybar) ** 2)


def grouped_eval(X, y, w, groups, model, name, feats, nsplits=5):
    gk = GroupKFold(n_splits=nsplits)
    pred = np.full(len(y), np.nan)
    for tr, te in gk.split(X, y, groups):
        m = model()
        Xtr, ytr, wtr = X[tr], y[tr], w[tr]
        # antisymmetric augmentation: the target is odd under swapping the two polymorphs
        Xtr = np.vstack([Xtr, -Xtr])
        ytr = np.concatenate([ytr, -ytr])
        wtr = np.concatenate([wtr, wtr])
        m.fit(Xtr, ytr, sample_weight=wtr)
        pred[te] = m.predict(X[te])
    r = dict(model=name, n=len(y),
             R2_w=float(wR2(y, pred, w)), R2_u=float(wR2(y, pred, np.ones(len(y)))),
             MAE_w=float((w / w.sum()) @ np.abs(y - pred)), MAE_u=float(np.abs(y - pred).mean()),
             zero_MAE_w=float((w / w.sum()) @ np.abs(y)), zero_MAE_u=float(np.abs(y).mean()),
             dir_agree_w=float((w / w.sum()) @ (pred * y > 0)),
             spearman=float(sps.spearmanr(pred, y).statistic))
    return r, pred


def main():
    d = pd.read_csv(D / 'descriptors.csv')
    print('descriptor rows', len(d))
    d = d[d.Band_gap_HSE.notna() & d.Space_group_rlx.notna()]
    cells = build_cells(d)
    print('cells (composition x relaxed sg)', len(cells), 'compositions', cells.composition_key.nunique())
    p = build_pairs(cells)
    multi = cells.groupby('composition_key').size()
    print('compositions with >1 relaxed sg', int((multi > 1).sum()), 'pairs', len(p))
    p.to_csv(D / 'pairs.csv', index=False)

    # chemistry differences must vanish inside a pair
    chk = p[['dchem_' + k for k in CHEM]].abs().max().max()
    print('max |chemistry descriptor difference| inside a pair', float(chk))

    feats = ['d_' + k for k in STRUCT] + ['same_crystal_system']
    y = p.d_Band_gap_HSE.values
    w = p.weight.values
    X = p[feats].values.astype(float)
    # ridge needs finite values: a missing descriptor difference is set to 0 with an indicator
    miss = np.isnan(X)
    Xr = np.where(miss, 0.0, X)
    ind = miss.any(0)
    Xr = np.hstack([Xr, miss[:, ind].astype(float)])
    featsr = feats + [f + '_missing' for f, m in zip(feats, ind) if m]
    sc = StandardScaler().fit(np.vstack([Xr, -Xr]))
    Xs = (Xr - sc.mean_) / sc.scale_

    res = []
    for tgt, yy in (('d_HSE', y), ('d_HSE_optical', p.d_Band_gap_HSE_optical.values),
                    ('d_GGA', p.d_Band_gap_GGA.values)):
        for nm, mk, Xu in (('ridge', lambda: Ridge(alpha=10.0, fit_intercept=False), Xs),
                           ('gbm', lambda: HistGradientBoostingRegressor(max_iter=400, learning_rate=0.06,
                                                                        max_depth=4, random_state=0), X)):
            for gname, gr in (('composition', p.composition_key.values),
                              ('chem_system', p.elem_system.values),
                              ('anion_system', p.anion_system.values)):
                r, pred = grouped_eval(Xu, yy, w, gr, mk, nm, featsr)
                r.update(target=tgt, heldout=gname)
                res.append(r)
                if tgt == 'd_HSE' and gname == 'composition':
                    p['pred_' + nm] = pred
    R = pd.DataFrame(res)[['target', 'model', 'heldout', 'n', 'R2_w', 'R2_u', 'MAE_w', 'zero_MAE_w',
                           'dir_agree_w', 'spearman']]
    R.to_csv(T / 'model_performance.csv', index=False)
    print('\n== model performance ==')
    print(R.to_string(index=False))

    # standardised ridge coefficients on the full set (antisymmetric augmentation)
    rg = Ridge(alpha=10.0, fit_intercept=False)
    rg.fit(np.vstack([Xs, -Xs]), np.concatenate([y, -y]), sample_weight=np.concatenate([w, w]))
    co = pd.DataFrame(dict(feature=featsr, std_coef_eV=rg.coef_))
    # permutation importance, out of fold, composition-held-out
    gk = GroupKFold(n_splits=5)
    imp = np.zeros(len(featsr))
    impg = np.zeros(len(feats))
    for tr, te in gk.split(Xs, y, p.composition_key.values):
        m = Ridge(alpha=10.0, fit_intercept=False)
        m.fit(np.vstack([Xs[tr], -Xs[tr]]), np.concatenate([y[tr], -y[tr]]),
              sample_weight=np.concatenate([w[tr], w[tr]]))
        imp += permutation_importance(m, Xs[te], y[te], n_repeats=10, random_state=0,
                                      scoring='neg_mean_absolute_error').importances_mean * len(te)
        g = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.06, max_depth=4, random_state=0)
        g.fit(np.vstack([X[tr], -X[tr]]), np.concatenate([y[tr], -y[tr]]),
              sample_weight=np.concatenate([w[tr], w[tr]]))
        impg += permutation_importance(g, X[te], y[te], n_repeats=10, random_state=0,
                                       scoring='neg_mean_absolute_error').importances_mean * len(te)
    co['perm_imp_ridge_eV'] = -imp / len(y)
    co = co.merge(pd.DataFrame(dict(feature=feats, perm_imp_gbm_eV=-impg / len(y))), how='left')
    # univariate: weighted Pearson / Spearman / OLS slope with composition-clustered bootstrap
    rows = []
    comps = p.composition_key.unique()
    cidx = {c: np.where(p.composition_key.values == c)[0] for c in comps}
    for f in feats:
        x = p[f].values.astype(float)
        ok = np.isfinite(x) & np.isfinite(y)
        if ok.sum() < 30 or np.nanstd(x[ok]) == 0:
            continue
        sl = np.sum(w[ok] * x[ok] * y[ok]) / np.sum(w[ok] * x[ok] ** 2)   # odd-function OLS, no intercept
        bs = []
        for _ in range(400):
            cs = RNG.choice(comps, len(comps))
            ii = np.concatenate([cidx[c] for c in cs])
            ii = ii[np.isfinite(x[ii])]
            bs.append(np.sum(w[ii] * x[ii] * y[ii]) / np.sum(w[ii] * x[ii] ** 2))
        rows.append(dict(feature=f, n=int(ok.sum()), slope_eV_per_unit=float(sl),
                         ci_lo=float(np.percentile(bs, 2.5)), ci_hi=float(np.percentile(bs, 97.5)),
                         pearson=float(sps.pearsonr(x[ok], y[ok]).statistic),
                         spearman=float(sps.spearmanr(x[ok], y[ok]).statistic),
                         r2_alone=float(wR2(y[ok], sl * x[ok], w[ok]))))
    U = pd.DataFrame(rows).sort_values('r2_alone', ascending=False)
    co = co.merge(U[['feature', 'slope_eV_per_unit', 'ci_lo', 'ci_hi', 'spearman', 'r2_alone']], how='left')
    co['abs'] = co.std_coef_eV.abs()
    co = co.sort_values('abs', ascending=False).drop(columns='abs')
    co.to_csv(T / 'descriptor_effects.csv', index=False)
    print('\n== top descriptor effects (standardised ridge coef in eV per 1 sd of the difference) ==')
    print(co.head(22).to_string(index=False))

    # ---- hypothesis contrasts
    H = []

    def contrast(name, mask_a, label, yy=y):
        """mean gap difference over a subset, oriented as stated in `label`"""
        m = np.asarray(mask_a)
        if m.sum() < 5:
            H.append(dict(hypothesis=name, contrast=label, n=int(m.sum()), mean_eV=np.nan,
                          ci_lo=np.nan, ci_hi=np.nan, frac_positive=np.nan))
            return
        v = yy[m]
        ww = w[m] / w[m].sum()
        bs = []
        for _ in range(400):
            cs = RNG.choice(comps, len(comps))
            ii = np.concatenate([cidx[c] for c in cs])
            ii = ii[m[ii]]
            if len(ii) == 0:
                continue
            bs.append(float(w[ii] @ yy[ii] / w[ii].sum()))
        H.append(dict(hypothesis=name, contrast=label, n=int(m.sum()), mean_eV=float(ww @ v),
                      median_eV=float(np.median(v)), ci_lo=float(np.percentile(bs, 2.5)),
                      ci_hi=float(np.percentile(bs, 97.5)), frac_positive=float(ww @ (v > 0))))

    # H1 dimensionality: orient each pair so that member b has the LOWER dimensionality
    dd = p.d_dim_cat_an.values
    yl = np.where(dd < 0, y, -y)       # gap(lower-dim member) - gap(higher-dim member)
    contrast('H1 lower dimensionality -> larger gap', np.abs(dd) > 0,
             'gap(lower dim) - gap(higher dim), any drop', yl)
    contrast('H1 lower dimensionality -> larger gap', np.abs(dd) == 2,
             'gap(lower dim) - gap(higher dim), drop of 2', yl)
    contrast('H1 lower dimensionality -> larger gap', np.abs(dd) == 1,
             'gap(lower dim) - gap(higher dim), drop of 1', yl)
    contrast('H1 lower dimensionality -> larger gap', dd == 0, 'same dimensionality (control, |mean| ~ 0)', np.abs(y))
    # H2 sharing: orient so that member b has MORE face+edge sharing
    fe = p.d_frac_face.values + p.d_frac_edge.values
    ys = np.where(fe > 0, y, -y)
    contrast('H2 more face/edge sharing -> larger gap', np.abs(fe) > 0.1,
             'gap(more face+edge) - gap(less), |d| > 0.1', ys)
    contrast('H2 more face/edge sharing -> larger gap', np.abs(fe) > 0.5,
             'gap(more face+edge) - gap(less), |d| > 0.5', ys)
    cs = (p.m_frac_corner.values > 0.6) & (np.abs(p.d_frac_corner.values) > 0.4)
    yc = np.where(p.d_frac_corner.values < 0, y, -y)
    contrast('H2 more face/edge sharing -> larger gap', cs,
             'corner-dominated vs not, gap(less corner) - gap(more corner)', yc)
    # H3 bond length: orient so b has the SHORTER mean bond
    db = p.d_bond_cat_an_mean.values
    yb = np.where(db < 0, y, -y)
    contrast('H3 shorter bonds -> smaller gap', np.abs(db) > 0.05,
             'gap(shorter bond) - gap(longer bond), |d| > 0.05 A', yb)
    contrast('H3 shorter bonds -> smaller gap', np.abs(db) > 0.2,
             'gap(shorter bond) - gap(longer bond), |d| > 0.2 A', yb)
    # H4 coordination: orient so b has the HIGHER mean cation CN
    dc = p.d_cn_cat_mean.values
    ycn = np.where(dc > 0, y, -y)
    contrast('H4 higher cation CN -> smaller gap', np.abs(dc) > 0.25,
             'gap(higher CN) - gap(lower CN), |d| > 0.25', ycn)
    contrast('H4 higher cation CN -> smaller gap', np.abs(dc) >= 1,
             'gap(higher CN) - gap(lower CN), |d| >= 1', ycn)
    # H5 distortion: orient so b is MORE distorted
    dq = p.d_distortion_index.values
    yq = np.where(dq > 0, y, -y)
    contrast('H5 larger distortion -> larger gap', np.abs(dq) > 0.01,
             'gap(more distorted) - gap(less), |d| > 0.01', yq)
    contrast('H5 larger distortion -> larger gap', np.abs(dq) > 0.05,
             'gap(more distorted) - gap(less), |d| > 0.05', yq)
    HH = pd.DataFrame(H)
    HH.to_csv(T / 'hypotheses.csv', index=False)
    print('\n== hypothesis contrasts (composition-weighted, composition-clustered 95% CI) ==')
    print(HH.to_string(index=False))

    # ---- small interpretable models
    small = ['d_dim_cat_an', 'd_mean_shared_vertices', 'd_bond_cat_an_mean', 'd_cn_cat_mean',
             'd_distortion_index', 'd_vol_per_atom']
    Xs2 = p[small].values.astype(float)
    Xs2 = np.where(np.isnan(Xs2), 0.0, Xs2)
    sc2 = StandardScaler().fit(np.vstack([Xs2, -Xs2]))
    Z = (Xs2 - sc2.mean_) / sc2.scale_
    rows = []
    for gname, gr in (('composition', p.composition_key.values), ('chem_system', p.elem_system.values)):
        r, _ = grouped_eval(Z, y, w, gr, lambda: Ridge(alpha=1.0, fit_intercept=False), 'ridge6', small)
        r.update(heldout=gname, target='d_HSE')
        rows.append(r)
    S = pd.DataFrame(rows)
    rg2 = Ridge(alpha=1.0, fit_intercept=False)
    rg2.fit(np.vstack([Z, -Z]), np.concatenate([y, -y]), sample_weight=np.concatenate([w, w]))
    print('\n== six-descriptor model ==')
    print(S[['heldout', 'n', 'R2_w', 'MAE_w', 'zero_MAE_w', 'dir_agree_w']].to_string(index=False))
    print(pd.DataFrame(dict(feature=small, std_coef_eV=rg2.coef_,
                            raw_coef=rg2.coef_ / sc2.scale_)).to_string(index=False))
    pd.DataFrame(dict(feature=small, std_coef_eV=rg2.coef_, raw_coef_eV_per_unit=rg2.coef_ / sc2.scale_,
                      mean=sc2.mean_, sd=sc2.scale_)).to_csv(T / 'six_descriptor_model.csv', index=False)
    S.to_csv(T / 'six_descriptor_perf.csv', index=False)
    np.save(D / 'ridge6_state.npy', np.vstack([rg2.coef_, sc2.mean_, sc2.scale_]))
    json.dump(dict(features=small, alpha=1.0), open(D / 'ridge6_features.json', 'w'))

    # descriptor spread inside a composition: how much of the gap difference is even there to explain
    print('\n== target scale ==')
    for t in ('d_Band_gap_HSE', 'd_Band_gap_HSE_optical', 'd_Band_gap_GGA'):
        v = p[t].values
        print(t, 'mean|d| %.4f  median|d| %.4f  sd %.4f  p90|d| %.4f' %
              (np.abs(v).mean(), np.median(np.abs(v)), v.std(), np.percentile(np.abs(v), 90)))


if __name__ == '__main__':
    main()
