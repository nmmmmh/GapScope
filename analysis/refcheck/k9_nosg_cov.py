"""Conformal baselines, full comparison.
A. Five fold assignments (cv_rep/s2026111x), tier-A reports, composition-equal weights. Normalized split conformal (cross-
   conformal over folds) with score |y - median| / (width80/2) from each model's own out-of-fold predictions: ROOST+conf and
   this work+conf; raw this work, ROOST_CAL, ROOST_CAL_CV for reference. Marginal coverage/width, and conditional coverage
   for multi-phase (>=2 phase labels) vs single-phase compositions and for floor terciles.
B. Original assignment: Roost Laplace mixture with b scaled per test fold by the conformal factor (q / ln 5), giving a full
   distribution; window promised/delivered at P >= 0.5 (both tiers, as main text) and pooled CRPS (tier A reports + tier B)."""
import sys, math, glob, numpy as np, pandas as pd
sys.path.insert(0, '/data/czx/GapScope_Revision_20261113/cv')
R = '/data/czx/GapScope_Revision_20261113/'
D1 = pd.read_csv(R + 'suppdata/Supplementary_Data_1_GapScope_experimental_bandgaps.csv', encoding='utf-8-sig')
nph = D1.assign(p=D1.phase_label.fillna('').astype(str).str.strip().str.lower()).query("p != ''").groupby('composition_key').p.nunique()
def wq(x, w, q):
    o = np.argsort(x); x, w = x[o], w[o]; c = np.cumsum(w) / w.sum(); return x[min(np.searchsorted(c, q), len(x) - 1)]
def wmed(v, w):
    o = np.argsort(v); v, w = v[o], w[o]; c = np.cumsum(w) / w.sum(); return v[np.searchsorted(c, 0.5)]
rows = []
for d in [R + "cv"]:
    for v in ('comp', 'chem'):
        P = pd.read_csv(f'{d}/predictions_summary_nosg_{v}.csv'); P = P[P.tier == 'A']
        fd = pd.read_csv(f'{d}/folds_{v}.csv').set_index('composition_key').fold.to_dict()
        r0 = P[P.model == 'ROOST']
        fl = {k: np.sum(g.w * np.abs(g.y - wmed(g.y.values, g.w.values))) / g.w.sum() for k, g in r0.groupby('key')}
        ter = pd.qcut(pd.Series(fl).rank(method='first'), 3, labels=['low', 'mid', 'high']).to_dict()
        for model, conf in (('ROOST', True), ("GS_SELCAL_PRE_NZ_SC", False), ("GS_SELCAL_PRE_NZ_NOSG_SC", False), ("ROOST_CAL_CV", False)):
            g = P[P.model == model].copy(); g['f'] = g.key.map(fd)
            g['wc'] = g.w / g.groupby('key').w.transform('sum')
            if conf:
                s = np.abs(g.y - g['median']) / (g.width80 / 2); cov = np.zeros(len(g)); wid = np.zeros(len(g))
                for f in range(1, 6):
                    tr, te = (g.f != f).values, (g.f == f).values
                    q = wq(s.values[tr], g.wc.values[tr], 0.80); cov[te] = s.values[te] <= q; wid[te] = g.width80.values[te] * q
                g['c'], g['wd'] = cov, wid
            else:
                g['c'], g['wd'] = g.cov80, g.width80
            ck = g.assign(c=g.c * g.wc, wd=g.wd * g.wc).groupby('key')[['c', 'wd']].sum()
            ck['multi'] = [nph.get(k, 0) >= 2 for k in ck.index]; ck['ter'] = [ter[k] for k in ck.index]
            rows.append(dict(rep=d[-8:], holdout=v, model=model + ('+conf' if conf else ''), cov=ck.c.mean(), width=ck.wd.mean(),
                             cov_multi=ck[ck.multi].c.mean(), cov_single=ck[~ck.multi].c.mean(),
                             cov_low=ck[ck.ter == 'low'].c.mean(), cov_mid=ck[ck.ter == 'mid'].c.mean(), cov_high=ck[ck.ter == 'high'].c.mean()))
T = pd.DataFrame(rows)
pd.set_option('display.width', 220)
S = T.drop(columns='rep').groupby(['holdout', 'model']).agg(['mean', 'std']).round(3)
print(S.to_string())
T.to_csv(R + 'refcheck/k9_nosg_table.csv', index=False)
