"""Width question, part 2: how solid is the provenance-set result, and what would a
chemistry-predicted width be worth against a freely fitted one.

(a) bootstrap CI and a label-permutation null for the held-out Spearman of the width model;
(b) the coarse categorical rule (variable coordination) with a CI;
(c) a CRPS comparison on the provenance records: one global width, a two-group width from the
    coarse rule, and the GBM-predicted width - all out of fold with whole element systems held out.
Usage: python -I width2.py
"""
import warnings, math
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from pathlib import Path
from scipy import stats as sps
from scipy.stats import norm
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold

D = Path('/data/czx/GapScope_Revision_20261113/struct')
T = D / 'tables'
SUPP = Path('/data/czx/GapScope_Revision_20261113/suppdata/Supplementary_Data_1_GapScope_experimental_bandgaps.csv')
RNG = np.random.default_rng(1)
SPREAD_OF = ['cn_cat_mean', 'cn_cat_max', 'cn_all_mean', 'mean_shared_vertices', 'frac_corner',
             'frac_edge', 'frac_face', 'dim_cat_an', 'bond_cat_an_mean', 'distortion_index',
             'vol_per_atom', 'packing_fraction', 'n_symops', 'frac_isolated_poly']
CHEMF = ['n_elements', 'anion_frac', 'en_an_mean', 'en_cat_mean', 'en_diff', 'ionicity', 'en_std_all',
         'nval_per_atom', 'radius_ratio', 'mean_atomic_radius', 'is_halide', 'is_oxide',
         'is_chalcogenide', 'is_pnictide', 'is_intermetallic', 'dom_anion_Z']
STRUCF = (['n_polymorphs', 'n_crystal_systems', 'any_noncentro', 'frac_direct', 'mixed_direct_indirect',
           'var_coordination', 'var_dimensionality', 'var_connectivity'] +
          ['spr_' + k for k in SPREAD_OF] + ['mean_' + k for k in SPREAD_OF] + ['hse_mean'])


def gbm():
    return HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_depth=3,
                                         min_samples_leaf=5, random_state=0)


def oof(X, y, gr, mk=gbm, nsp=5):
    p = np.full(len(y), np.nan)
    for tr, te in GroupKFold(n_splits=min(nsp, len(np.unique(gr)))).split(X, y, gr):
        p[te] = mk().fit(X[tr], y[tr]).predict(X[te])
    return p


def crps_norm(sig, x):
    sig = np.maximum(sig, 1e-3)
    z = x / sig
    return sig * (z * (2 * norm.cdf(z) - 1) + 2 * norm.pdf(z) - 1 / math.sqrt(math.pi))


def main():
    W = pd.read_csv(D / 'width.csv')
    q = W[(W.set == 'provenance') & W.exp_range.notna() & (W.in_snumat == 1)].reset_index(drop=True)
    feats = [f for f in CHEMF + STRUCF if f in q.columns]
    X = q[feats].values.astype(float)
    y = q.exp_range.values
    gr = q.elem_system.fillna('?').values
    pred = oof(X, y, gr)
    rho = sps.spearmanr(pred, y).statistic
    print('provenance, SNUMAT-matched, n=%d compositions, %d element systems' % (len(q), len(set(gr))))
    print('held-out Spearman(predicted range, observed range) = %+.3f, R2 = %+.3f'
          % (rho, 1 - np.sum((y - pred) ** 2) / np.sum((y - y.mean()) ** 2)))
    bs = [sps.spearmanr(pred[i], y[i]).statistic
          for i in (RNG.integers(0, len(q), len(q)) for _ in range(2000))]
    bs = np.array([b for b in bs if np.isfinite(b)])
    print('bootstrap 95%% CI over compositions: [%.3f, %.3f]' % (np.percentile(bs, 2.5), np.percentile(bs, 97.5)))
    # permutation null: shuffle the target across compositions, refit out of fold
    nul = []
    for _ in range(200):
        ys = RNG.permutation(y)
        nul.append(sps.spearmanr(oof(X, ys, gr), ys).statistic)
    nul = np.array(nul)
    print('permutation null: mean %+.3f, 95th pct %+.3f, p(rho >= observed) = %.3f'
          % (nul.mean(), np.percentile(nul, 95), float((nul >= rho).mean())))

    # (b) coarse rule
    print('\n== coarse rule: variable cation coordination among the composition\'s SNUMAT polymorphs ==')
    for flag in ('var_coordination', 'var_connectivity', 'multi_polymorph'):
        a = q[q[flag] == 0].exp_range.values
        b = q[q[flag] == 1].exp_range.values
        d = []
        for _ in range(4000):
            d.append(np.median(RNG.choice(b, len(b))) - np.median(RNG.choice(a, len(a))))
        print('%-17s n0=%2d median %.3f eV | n1=%2d median %.3f eV | difference %.3f [%.3f, %.3f] | ratio %.2fx | MW p=%.4f'
              % (flag, len(a), np.median(a), len(b), np.median(b), np.median(b) - np.median(a),
                 np.percentile(d, 2.5), np.percentile(d, 97.5), np.median(b) / max(1e-6, np.median(a)),
                 sps.mannwhitneyu(a, b).pvalue))

    # (c) CRPS of the width itself, on the provenance records
    sup = pd.read_csv(SUPP)
    sup = sup[sup.composition_key.isin(q.composition_key)]
    med = sup.groupby('composition_key').bandgap_ev.transform('median')
    sup = sup.assign(dev=(sup.bandgap_ev - med).astype(float))
    key2row = {k: i for i, k in enumerate(q.composition_key)}
    idx = sup.composition_key.map(key2row).values
    dev = sup.dev.values
    grp = gr[idx]
    print('\n== CRPS of a zero-centred normal width on %d provenance records (%d compositions) ==' % (len(sup), len(q)))
    # observed within-composition sd as the regression target for the width model
    ysd = q.exp_sd.fillna(q.exp_range / 2).values
    predsd = np.clip(oof(X, ysd, gr), 0.02, None)
    rows = []
    for nm in ('global constant (freely fitted)', 'two-group rule (variable coordination)',
               'chemistry+structure model'):
        tot = np.zeros(len(sup))
        for tr, te in GroupKFold(n_splits=5).split(np.zeros(len(q)), ysd, gr):
            mtr = np.isin(idx, tr)
            mte = np.isin(idx, te)
            if nm.startswith('global'):
                s = np.full(mte.sum(), np.sqrt(np.mean(dev[mtr] ** 2)))
            elif nm.startswith('two-group'):
                f = q.var_coordination.values
                s0 = np.sqrt(np.mean(dev[mtr & (f[idx] == 0)] ** 2))
                s1 = np.sqrt(np.mean(dev[mtr & (f[idx] == 1)] ** 2))
                s = np.where(f[idx][mte] == 1, s1, s0)
            else:
                m = gbm().fit(X[tr], ysd[tr])
                s = np.clip(m.predict(X[idx[mte]]), 0.02, None)
                # rescale so the mean predicted variance matches the training residual scale
                k = np.sqrt(np.mean(dev[mtr] ** 2)) / max(1e-6, np.mean(np.clip(m.predict(X[tr]), 0.02, None)))
                s = s * k
            tot[mte] = crps_norm(s, dev[mte])
        rows.append(dict(width_model=nm, mean_CRPS_eV=float(tot.mean()),
                         median_CRPS_eV=float(np.median(tot))))
    C = pd.DataFrame(rows)
    C['improvement_vs_constant_%'] = 100 * (1 - C.mean_CRPS_eV / C.mean_CRPS_eV.iloc[0])
    C.to_csv(T / 'width_crps.csv', index=False)
    print(C.to_string(index=False))
    print('\nheld-out Spearman(predicted sd, observed sd) = %+.3f' % sps.spearmanr(predsd, ysd).statistic)


if __name__ == '__main__':
    main()
