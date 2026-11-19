"""Conditional coverage by D_c group (computable before measurement) and a Mondrian conformal baseline grouped by D_c.
Tier-A traced reports, composition-equal weights, five fold assignments, both holdouts.
D_c: energy-only SNUMAT polymorphs within 0.1 eV/atom (main-text definition). Groups: 'none' (not in SNUMAT or one
accessible polymorph, D_c = 0), 'low' (0 < D_c < 0.3), 'high' (D_c >= 0.3). Also reports floor thirds and single/multi phase.
Mondrian: the 80% quantile of normalized out-of-fold scores is taken separately within each D_c group (other four folds)."""
import glob, numpy as np, pandas as pd
R = '/data/czx/GapScope_Revision_20261113/'
S = pd.read_csv(R + 'benchmark/snumat_by_comp_sg.csv')
def dc(c):
    c = c[c.hse.notna() & c.Free_energy.notna()]
    if len(c) < 2: return 0.0
    c = c[c.Free_energy - c.Free_energy.min() <= 0.10]; return c.hse.max() - c.hse.min() if len(c) >= 2 else 0.0
DC = {k: dc(c) for k, c in S.groupby('ckey')}
D1 = pd.read_csv(R + 'suppdata/Supplementary_Data_1_GapScope_experimental_bandgaps.csv', encoding='utf-8-sig')
nph = D1.assign(p=D1.phase_label.fillna('').astype(str).str.strip().str.lower()).query("p != ''").groupby('composition_key').p.nunique()
def grp(k): d = DC.get(k, 0.0); return 'none' if d == 0 else ('low' if d < 0.3 else 'high')
def wq(x, w, q):
    o = np.argsort(x); x, w = x[o], w[o]; c = np.cumsum(w) / w.sum(); return x[min(np.searchsorted(c, q), len(x) - 1)]
def wmed(v, w):
    o = np.argsort(v); v, w = v[o], w[o]; c = np.cumsum(w) / w.sum(); return v[np.searchsorted(c, 0.5)]
rows = []
for d in sorted(glob.glob(R + 'cv_rep/s2026111*')):
    for v in ('comp', 'chem'):
        P = pd.read_csv(f'{d}/predictions_summary_rep_{v}.csv'); P = P[P.tier == 'A']
        fd = pd.read_csv(f'{d}/folds_{v}.csv').set_index('composition_key').fold.to_dict()
        r0 = P[P.model == 'ROOST']
        fl = {k: np.sum(g.w * np.abs(g.y - wmed(g.y.values, g.w.values))) / g.w.sum() for k, g in r0.groupby('key')}
        ter = pd.qcut(pd.Series(fl).rank(method='first'), 3, labels=['low', 'mid', 'high']).to_dict()
        for model, mode in (('GS_SELCAL_PRE_NZ_SC', 'raw'), ('ROOST', 'conf'), ('ROOST', 'mondrian'), ('ROOST_CAL_CV', 'raw')):
            g = P[P.model == model].copy(); g['f'] = g.key.map(fd); g['G'] = g.key.map(grp)
            g['wc'] = g.w / g.groupby('key').w.transform('sum')
            if mode == 'raw':
                g['c'], g['wd'] = g.cov80, g.width80
            else:
                s = (np.abs(g.y - g['median']) / (g.width80 / 2)).values; cov = np.zeros(len(g)); wid = np.zeros(len(g))
                for f in range(1, 6):
                    for G in (['all'] if mode == 'conf' else ['none', 'low', 'high']):
                        selG = np.ones(len(g), bool) if G == 'all' else (g.G == G).values
                        tr = (g.f != f).values & selG; te = (g.f == f).values & selG
                        if te.sum() == 0: continue
                        q = wq(s[tr], g.wc.values[tr], 0.80); cov[te] = s[te] <= q; wid[te] = g.width80.values[te] * q
                g['c'], g['wd'] = cov, wid
            ck = g.assign(c=g.c * g.wc, wd=g.wd * g.wc).groupby('key')[['c', 'wd']].sum()
            ck['G'] = [grp(k) for k in ck.index]; ck['ter'] = [ter[k] for k in ck.index]; ck['multi'] = [nph.get(k, 0) >= 2 for k in ck.index]
            r = dict(holdout=v, model=model + '+' + mode, cov=ck.c.mean(), width=ck.wd.mean())
            for G in ('none', 'low', 'high'): r['Dc_' + G] = ck[ck.G == G].c.mean(); r['w_Dc_' + G] = ck[ck.G == G].wd.mean()
            for t in ('low', 'mid', 'high'): r['fl_' + t] = ck[ck.ter == t].c.mean()
            r['single'] = ck[~ck.multi].c.mean(); r['multi'] = ck[ck.multi].c.mean()
            rows.append(r)
T = pd.DataFrame(rows)
cnt = pd.Series([grp(k) for k in r0.key.unique()]).value_counts().to_dict()
print('D_c groups among 85 compositions:', cnt)
M = T.groupby(['holdout', 'model']).mean(numeric_only=True); Sd = T.groupby(['holdout', 'model']).std(numeric_only=True)
pd.set_option('display.width', 250)
print(M.round(3).to_string()); print('SD'); print(Sd[['cov', 'Dc_high', 'fl_high', 'multi']].round(3).to_string())
T.to_csv(R + 'refcheck/k4_rep_table.csv', index=False)
