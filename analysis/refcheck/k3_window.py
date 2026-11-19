"""Original fold assignment. Roost 3-seed Laplace mixture with every scale b multiplied, per test fold, by a conformal factor:
the composition-weighted 80% quantile, over the out-of-fold units of the other four folds (tier-A records and tier-B compiled
values, each composition equal mass), of |y - median| / (half-width80 of the raw mixture). Gives a full distribution.
Reports pooled coverage/width (composition-equal over 2552 units), CRPS pooled, and window promised/delivered at P >= 0.5
for PV (1.1-1.7) and VIS (1.8-3.1), to compare with Table S15 (this work chem PV 62/63, VIS 77/78; Roost-cal 74/59, 85/76)."""
import sys, numpy as np, pandas as pd
sys.path.insert(0, '/data/czx/GapScope_Revision_20261113/cv')
import evaluate_cv as ev
CV = ev.CV; A, Akey, Bkeys = ev.A, ev.Akey, ev.Bkeys; wA = np.asarray(ev.wA, float)
Brows = pd.read_csv(CV / 'data/tierB_rows.csv')
FOLDS = {v: pd.read_csv(CV / f'folds_{v}.csv').set_index('composition_key').fold.to_dict() for v in ('comp', 'chem')}
xs = np.linspace(-2, 14, 3201); dx = xs[1] - xs[0]
def cdf(mu, b): return (0.5 + 0.5 * np.sign(xs[:, None] - mu[None]) * (1 - np.exp(-np.abs(xs[:, None] - mu[None]) / b[None]))).mean(1)
def q(F, p): return xs[min(np.searchsorted(F, p), len(xs) - 1)]
def wq(x, w, p):
    o = np.argsort(x); x, w = x[o], w[o]; c = np.cumsum(w) / w.sum(); return x[min(np.searchsorted(c, p), len(x) - 1)]
for variant in ('chem', 'comp'):
    gA, gB = ev.load_roost(variant); rA = gA[0]
    units = []   # (key, fold, mu, b, ys, ws)
    for k in set(Akey):
        ii = [i for i, x in enumerate(Akey) if x == k]
        for i in ii: units.append((k, FOLDS[variant][k], rA[i][1], rA[i][2], np.array([float(A[i]['gap_ev'])]), np.array([wA[i] / wA[ii].sum()]), 'A'))
    for k in Bkeys:
        g = Brows[Brows.composition_key == k]
        if not len(g) or k not in gB: continue
        units.append((k, FOLDS[variant][k], gB[k][1], gB[k][2], g.target_ev.values.astype(float), g.within_composition_weight.values / g.within_composition_weight.sum(), 'B'))
    med = []; hw = []
    for u in units:
        F = cdf(u[2], u[3]); med.append(q(F, .5)); hw.append((q(F, .9) - q(F, .1)) / 2)
    med = np.array(med); hw = np.array(hw)
    sc, sw, sf = [], [], []
    for j, u in enumerate(units):
        for y, w in zip(u[4], u[5]): sc.append(abs(y - med[j]) / hw[j]); sw.append(w); sf.append(u[1])
    sc, sw, sf = map(np.array, (sc, sw, sf))
    fac = {f: wq(sc[sf != f], sw[sf != f], 0.80) for f in range(1, 6)}
    res = {k: dict(cov=[], wid=[], crps=[]) for k in ('A', 'B')}; win = []
    for j, u in enumerate(units):
        c = fac[u[1]]; F = cdf(u[2], u[3] * c); lo, hi = q(F, .1), q(F, .9)
        cov = float(np.sum(u[5] * ((u[4] >= lo) & (u[4] <= hi))))
        crps = float(np.sum([w * np.sum((F - (xs >= y)) ** 2) * dx for y, w in zip(u[4], u[5])]))
        res[u[6]]['cov'].append((u[0], cov)); res[u[6]]['wid'].append((u[0], hi - lo)); res[u[6]]['crps'].append((u[0], crps))
        for wn, (a, b_) in (('PV', (1.1, 1.7)), ('VIS', (1.8, 3.1))):
            p = np.interp(b_, xs, F) - np.interp(a, xs, F); o = float(np.sum(u[5] * ((u[4] >= a) & (u[4] <= b_))))
            win.append((u[0], u[6], wn, p, o))
    def comp_mean(lst):
        d = pd.DataFrame(lst, columns=['k', 'v']).groupby('k').v.sum(); return d
    covA = comp_mean(res['A']['cov']); covB = comp_mean(res['B']['cov'])
    widA = pd.DataFrame(res['A']['wid'], columns=['k', 'v']).groupby('k').v.mean(); widB = comp_mean(res['B']['wid'])
    crA = comp_mean(res['A']['crps']); crB = comp_mean(res['B']['crps'])
    pooled = lambda a, b: pd.concat([a, b]).mean()
    print('%s factors %s' % (variant, {f: round(v, 2) for f, v in fac.items()}))
    print('%s Roost+conformal: coverage A %.3f pooled %.3f | width A %.2f pooled %.2f | CRPS A %.3f pooled %.3f'
          % (variant, covA.mean(), pooled(covA, covB), widA.mean(), pooled(widA, widB), crA.mean(), pooled(crA, crB)))
    W = pd.DataFrame(win, columns=['k', 't', 'w', 'p', 'o'])
    # composition-level window probability: tier-A records of one composition share one distribution -> take first
    W = W.groupby(['k', 't', 'w']).agg(p=('p', 'mean'), o=('o', 'sum')).reset_index()
    W.loc[W.t == 'A', 'o'] = W[W.t == 'A'].o   # A: summed record weights within composition already normalised
    for wn in ('PV', 'VIS'):
        s = W[(W.w == wn) & (W.p >= 0.5)]
        print('   window %-3s n %4d promised %.3f delivered %.3f' % (wn, len(s), s.p.mean(), s.o.mean()))
