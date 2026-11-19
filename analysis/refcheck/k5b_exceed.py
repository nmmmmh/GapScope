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
# k5b (2026-11-19): read-only listing of compositions outside the bound of eq 3, with their number of reports
D1 = pd.read_csv(R + "suppdata/Supplementary_Data_1_GapScope_experimental_bandgaps.csv", encoding="utf-8-sig")
fm = D1.drop_duplicates("composition_key").set_index("composition_key").formula; nr = D1.groupby("composition_key").size()
for lab, sel in (("D_c = 0, exceed by >= 0.04 eV", lambda f, d, o: d == 0 and not o and f - e_noise >= 0.04), ("D_c = 0, exceed by < 0.04 eV", lambda f, d, o: d == 0 and not o and f - e_noise < 0.04), ("D_c > 0, outside", lambda f, d, o: d > 0 and not o)):
    rows = [(fm.get(k, k), round(float(f), 3), round(float(f - d / 2 - e_noise), 3), int(nr.get(k, 0))) for k, f, d, o in zip(keys, F, Dc, ok) if sel(f, d, o)]
    print(lab, len(rows)); [print("   %-10s floor %.3f excess %.3f reports %d" % r) for r in sorted(rows, key=lambda r: -r[2])]
