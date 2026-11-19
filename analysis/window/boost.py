import sys
import numpy as np, pandas as pd
sys.path.insert(0, '/data/czx/GapScope_Revision_20261113/cv'); sys.path.insert(0, '/data/czx/GapScope_Revision_20261113/s5')
import evaluate_cv_v4 as E
ev = E.ev
W = {'PV': (1.1, 1.7), 'VIS': (1.8, 3.1)}
def strad(y, a, b, d=0.0):
    i = (y >= a + d) & (y <= b - d); x = (y < a - d) | (y > b + d); return bool(i.any() and x.any())
S = pd.read_csv('/data/czx/GapScope_Revision_20261113/benchmark/snumat_by_comp_sg.csv')
acc = {}
for k, t in S.groupby('ckey'):
    te = t.dropna(subset=['Free_energy'])
    if len(te) >= 2:
        te = te[te.Free_energy <= te.Free_energy.min() + 0.1]
        if len(te) >= 2: acc[k] = (te.hse.values, te.sort_values('Free_energy').hse.values[0])
print('== A. HSE screen: compositions with >=2 accessible polymorphs', len(acc))
for wn, (a, b) in W.items():
    for d in (0.0, 0.1):
        n = sum(strad(v, a, b, d) for v, _ in acc.values())
        gs_in = sum((a <= g0 <= b) for _, g0 in acc.values()); flipped_in = sum((a <= g0 <= b) and strad(v, a, b, d) for v, g0 in acc.values())
        print(wn, 'margin', d, 'membership changes across accessible polymorphs: %d (%.1f%%); ground-state in window %d, of which %d have an accessible polymorph outside' % (n, 100 * n / len(acc), gs_in, flipped_in))
# B. concordance HSE straddle -> experimental straddle
Y = {}
for i, k in enumerate(ev.Akey): Y.setdefault(k, []).append(float(ev.A[i]['gap_ev']))
for k, v in ev.Bvals.items():
    if k not in Y: Y[k] = list(map(float, v))
print('== B. HSE polymorph straddle vs experimental straddle (compositions with >=2 reports and >=2 SNUMAT polymorphs)')
allpoly = {k: t.hse.values for k, t in S.groupby('ckey') if len(t) >= 2}
for name, H in (('accessible', {k: v for k, (v, _) in acc.items()}), ('all_sg', allpoly)):
    ks = [k for k in H if k in Y and len(Y[k]) >= 2]
    for wn, (a, b) in W.items():
        h = np.array([strad(H[k], a, b) for k in ks]); e = np.array([strad(np.array(Y[k]), a, b) for k in ks])
        # also exp straddle among exp comps NOT in SNUMAT multi
        base = np.mean([strad(np.array(y), a, b) for k, y in Y.items() if len(y) >= 2])
        print(name, wn, 'n %d | HSE straddle %d -> exp straddle %.2f | HSE no straddle %d -> exp straddle %.2f | all exp base %.2f' % (len(ks), h.sum(), e[h].mean() if h.any() else np.nan, (~h).sum(), e[~h].mean(), base))
# C. promised vs delivered hit rate when selecting P>=thr (unit-level, fraction of reports in window)
T = pd.read_csv('/data/czx/GapScope_Revision_20261113/window/par2/window_units_all.csv')
print('== C. select candidates with P >= thr: promised (mean P) vs delivered (mean fraction of measured gaps in window)')
for V in ('comp', 'chem'):
    for wn in W:
        for thr in (0.5, 0.7):
            s = []
            for m in ('GapScope', 'ROOST_CAL', 'EASYUQ_KS'):
                t = T[(T.holdout == V) & (T.model == m)]; q = t['P_' + wn] >= thr
                s.append('%s n=%d promised %.2f delivered %.2f' % (m, q.sum(), t.loc[q, 'P_' + wn].mean(), t.loc[q, 'frac_' + wn].mean()))
            print(V, wn, 'thr', thr, ' | '.join(s))
