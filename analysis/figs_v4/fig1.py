"""Figure 1 (v3): one accent colour = phase. A composition has a set of gaps, and the set is phase."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from gsfig import *

ACC = '#D55E00'   # phase
R0 = '/data/czx/GapScope_Revision_20261113/'
S = pd.read_csv(R0 + 'suppdata/Supplementary_Data_1_GapScope_experimental_bandgaps.csv')
S['ph'] = S.phase_label.fillna('').astype(str).str.strip().str.lower()
nph = S[S.ph != ''].groupby('composition_key').ph.nunique()
L = pd.read_csv(R0 + 'struct/tables/ladder_pairs.csv')
rng = np.random.default_rng(3)

W, H = 7.0, 3.95
fig = plt.figure(figsize=(W, H))
def ax_in(x, y, w, h): return fig.add_axes([x / W, 1 - (y + h) / H, w / W, h / H])
def letter(s, x, y): fig.text(x / W, 1 - y / H, '(%s)' % s, fontsize=8, fontweight='bold', ha='left', va='top')

# ======================= (a) each FAPbI3 phase sits above its own gap
E0, E1, X0, XW = 1.3, 2.6, 0.40, 2.60
def ein(E): return X0 + (E - E0) / (E1 - E0) * XW
axg = ax_in(X0, 1.30, XW, 0.20)
for sp in ('left', 'top', 'right'):
    axg.spines[sp].set_visible(False)
axg.set_yticks([]); axg.set_xlim(E0, E1); axg.set_ylim(0, 1)
axg.plot([1.95, 1.95], [0.0, 1.0], color=GREY, lw=0.7, ls=(0, (2, 1.5)))
axg.plot([1.47], [0.45], ls='none', marker='*', ms=7.5, color=ACC, mec='white', mew=0.3)
axg.plot([2.43], [0.45], ls='none', marker='D', ms=4.6, color=ACC, mec='white', mew=0.3)
axg.set_xticks([1.5, 2.0, 2.5]); axg.set_xlabel('$E_g$ (eV)', labelpad=1.5)
w1, w2 = 1.05, 0.80
ax1 = ax_in(ein(1.47) - w1 / 2, 0.24, w1, 0.98); ax2 = ax_in(ein(2.43) - w2 / 2, 0.24, w2, 0.98)
P, A = corner_sharing((2, 2, 1)); A = A + [(p.mean(0), 'Pb', 0.10) for p in P]
render(ax1, P, A, view(32, -58), VESTA['Pb'])
P2, A2 = [], []
for xy, ph in [((0, 0), 0), ((1.85, 0), 30)]:
    p, a = face_sharing_chain(xy, n=3, s=1.0, phase=ph); P2 += p; A2 += a + [(v.mean(0), 'Pb', 0.10) for v in p]
render(ax2, P2, A2, view(15, -80), VESTA['Pb'])
for ax, t in ((ax1, r'$\alpha$'), (ax2, r'$\delta$')):
    ax.text(0.5, 1.0, t, transform=ax.transAxes, ha='center', va='bottom', fontsize=8, color=INK)
letter('a', 0.02, 0.06)

# ======================= (b) within-study ladder
axe = ax_in(3.85, 0.18, 3.05, 1.55)
cats = [('repeat', 'Rep.'), ('specimen', 'Spec.'), ('phase_same_study', 'Phase'), ('phase_cross_study', 'Phase*')]
ci = {'repeat': (0.006, 0.090), 'specimen': (0.062, 0.162), 'phase_same_study': (0.223, 0.387), 'phase_cross_study': (0.086, 0.358)}
for x, (k, lab) in enumerate(cats):
    v = np.sort(L[L.kind == k].d.values)
    dens = np.array([np.sum(np.abs(v - t) < 0.04) for t in v], float)
    jit = (rng.random(len(v)) - 0.5) * 0.66 * dens / dens.max()
    isph = k.startswith('phase')
    axe.scatter(x + jit, v, s=2.0, color=ACC if isph else GREY, alpha=(0.55 if k == 'phase_same_study' else 0.28) if isph else 0.55,
                lw=0, zorder=2)
    m = v.mean(); lo, hi = ci[k]; c = ACC if k == 'phase_same_study' else INK
    axe.plot([x - 0.30, x + 0.30], [m, m], color=c, lw=1.4, zorder=3)
    axe.plot([x, x], [lo, hi], color=c, lw=0.8, zorder=3)
    axe.text(x + 0.34, m, '%.2f' % m, fontsize=6.5, va='center', color=c)
axe.set_xticks(range(4)); axe.set_xticklabels([c[1] for c in cats], fontsize=6.5)
axe.set_ylim(-0.03, 1.25); axe.set_xlim(-0.5, 3.7); axe.set_yticks([0, 0.5, 1.0])
axe.set_ylabel('|$\\Delta E_g$| (eV)'); axe.tick_params(axis='x', length=0, pad=2)
letter('b', 3.40, 0.06)

# ======================= (c) every composition with more than one value, as its set
axd = ax_in(0.42, 2.20, 6.48, 1.42)
rows = []
for k, gg in S.groupby('composition_key'):
    rgn = gg.bandgap_ev.max() - gg.bandgap_ev.min()
    if rgn > 0:
        rows.append((k, gg.formula.iloc[0], rgn, gg, nph.get(k, 0) >= 2))
rows.sort(key=lambda t: -t[2])
for i, (k, f, rgn, gg, multi) in enumerate(rows):
    c = ACC if multi else GREY
    axd.plot([i, i], [gg.bandgap_ev.min(), gg.bandgap_ev.max()], color=c, lw=2.4, solid_capstyle='round', alpha=0.85, zorder=1)
    axd.plot([i] * len(gg), gg.bandgap_ev, ls='none', marker='o', ms=1.6, mfc='white', mec='white', mew=0, zorder=2)
axd.set_xlim(-1, len(rows)); axd.set_ylim(0, 6.8); axd.set_xticks([]); axd.set_yticks([0, 2, 4, 6])
axd.set_ylabel('$E_g$ (eV)'); axd.set_xlabel('Composition', labelpad=3)
def is_fa(r):
    k = r[0]; return all(('%s:' % e) in k for e in ('Pb', 'I', 'C', 'N')) and 'Cs:' not in k and 'Br:' not in k
for name, dx, ty in (('FAPbI3', 1.0, 0.55), ('Ga2O3', 1.0, 6.1)):
    i = [j for j, r in enumerate(rows) if (is_fa(r) if name == 'FAPbI3' else r[1] == name)][0]
    gv = rows[i][3].bandgap_ev; yy = gv.max() + 0.12 if ty > 3 else gv.min() - 0.12
    axd.annotate(fmt(name), (i, yy), xytext=(i + dx, ty), fontsize=6, color=GREY,
                 arrowprops=dict(arrowstyle='-', lw=0.35, color=GREY, shrinkA=0.8, shrinkB=0.3), ha='left',
                 va='bottom' if ty > 3 else 'top')
letter('c', 0.02, 2.06)
save(fig, 'Fig1')
print('compositions shown', len(rows), 'multi-phase', sum(r[4] for r in rows))
