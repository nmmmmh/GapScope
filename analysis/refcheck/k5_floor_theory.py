"""Tests of the derived floor.
F_c = min_y E|G - y| = E|G - med G|  (Bayes risk of a composition-only predictor under absolute loss).
Model: G = g_p + eps, p ~ pi_p (polymorph occurrence), eps ~ N(0, s_m) measurement scatter.
(T1) bound: F_c <= D_c/2 + E|eps|  (MAD about the median <= half range; triangle inequality for the noise).
(T2) computed floor with noise: pi_p = Boltzmann(T) over energy-only SNUMAT polymorphs, g_p = HSE06; E|G - med| by quadrature.
(T3) s_m from repeat pairs: mean|eps1-eps2| = 2 s_m / sqrt(pi) -> E|eps| = s_m sqrt(2/pi)."""
import numpy as np, pandas as pd
exec(open('/data/czx/GapScope_Revision_20261113/descr/d1_desc.py').read().split('pos_r, pos_f')[0])
S = pd.read_csv(R + 'benchmark/snumat_by_comp_sg.csv').rename(columns={'ckey': 'composition_key'})
S = S[S.composition_key.isin(keys) & S.Free_energy.notna() & S.hse.notna()]
rep = 0.043                      # mean |difference| of repeat pairs on one specimen (main text 2.1)
s_m = rep * np.sqrt(np.pi) / 2; e_noise = s_m * np.sqrt(2 / np.pi)
print('(T3) s_m %.3f eV, E|eps| %.3f eV (compare: single-phase floor of compositions measured more than once, 0.029 eV in 2.2)' % (s_m, e_noise))
kB = 8.617e-5; xs = np.linspace(-3, 10, 13001)
def floor_mix(g, w, s):
    w = w / w.sum()
    if s <= 0:
        o = np.argsort(g); c = np.cumsum(w[o]); m = g[o][np.searchsorted(c, 0.5)]; return np.sum(w * np.abs(g - m))
    pdf = (w[None] * np.exp(-0.5 * ((xs[:, None] - g[None]) / s) ** 2) / (s * np.sqrt(2 * np.pi))).sum(1); dx = xs[1] - xs[0]
    cdf = np.cumsum(pdf) * dx; m = xs[np.searchsorted(cdf, 0.5)]; return np.sum(pdf * np.abs(xs - m)) * dx
Dc = []; G = {}
for k in keys:
    c = S[S.composition_key == k]
    if len(c) >= 2:
        a = c[c.Free_energy - c.Free_energy.min() <= 0.1]; Dc.append(a.hse.max() - a.hse.min() if len(a) >= 2 else 0.0); G[k] = (c.hse.values, c.Free_energy.values - c.Free_energy.min())
    else: Dc.append(0.0)
Dc = np.array(Dc)
ok = F <= Dc / 2 + e_noise
print('(T1) F <= D_c/2 + E|eps| holds for %d of %d compositions; with D_c > 0: %d of %d; with D_c = 0: %d of %d'
      % (ok.sum(), len(F), ok[Dc > 0].sum(), (Dc > 0).sum(), ok[Dc == 0].sum(), (Dc == 0).sum()))
print('     mean F/(D_c/2+E|eps|) over D_c>0: %.2f' % np.mean(F[Dc > 0] / (Dc[Dc > 0] / 2 + e_noise)))
pos = F > 0.05
for T in (300, 600, 1000):
    for s in (0.0, s_m):
        x = np.array([floor_mix(G[k][0], np.exp(-G[k][1] / (kB * T)), s) if k in G else (e_noise if s > 0 else 0.0) for k in keys])
        print('(T2) T %4d noise %.3f | mean computed %.3f vs measured %.3f | positives %.3f vs %.3f | rho %.2f AUC %.3f | MAE(comp vs meas) %.3f'
              % (T, s, x.mean(), F.mean(), x[pos].mean(), F[pos].mean(), spearmanr(x, F).statistic, roc_auc_score(pos, x), np.abs(x - F).mean()))
# two-state check: compositions whose accessible set has exactly two polymorphs
two = [(k, f) for k, f in zip(keys, F) if k in G and (G[k][1] <= 0.1).sum() == 2]
print('(T2b) two-polymorph compositions within 0.1 eV/atom: %d' % len(two))
