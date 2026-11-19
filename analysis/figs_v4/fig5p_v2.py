"""Figure 5 data panels (SVG) to sit beside the architecture: (b) coverage vs width under composition and chemical holdout, (c) density gallery."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gs5 import *
plt.rcParams['svg.fonttype'] = 'path'
import numpy as np, pandas as pd

D = os.path.dirname(os.path.abspath(__file__)); K = 7.0 / 1080.0
R0 = '/data/czx/GapScope_Revision_20261113/'
S = pd.read_csv(R0 + 'suppdata/Supplementary_Data_1_GapScope_experimental_bandgaps.csv')
S['ph'] = S.phase_label.fillna('').astype(str).str.strip().str.lower().str.split(',').str[0].str.replace(r'-(ga2o3|in2se3)', '', regex=True)
PC = PH

# ---------------- (b) coverage of the nominal 80% interval against its width (slot 370 x 322 pt)
fig = plt.figure(figsize=(370 * K, 322 * K))
ax = fig.add_axes([0.20, 0.17, 0.62, 0.78])
models = [('Roost', (0.42, 0.432), (0.45, 0.380), SLATE), ('Roost-cal', (0.86, 0.687), (0.66, 0.543), INDIGO),
          ('Roost-cal*', (1.06, 0.763), (0.91, 0.670), SAND), ('EasyUQ', (1.23, 0.824), (1.34, 0.854), '#7A68B0'),
          ('This work', (1.06, 0.790), (1.22, 0.821), TEAL2)]
ax.axhline(0.8, color=CORAL, lw=0.8, ls=(0, (3, 1.5)), zorder=0)
for name, a, b, c in models:
    ax.annotate('', xy=b, xytext=a, arrowprops=dict(arrowstyle='-|>', color=c, lw=1.2, mutation_scale=7, shrinkA=2.5, shrinkB=2.5), zorder=2)
    mk = 's' if name == 'EasyUQ' else 'o'; z = 4 if name == 'This work' else 3
    ax.plot(*a, mk, ms=4, mfc='white', mec=c, mew=1.0, zorder=z); ax.plot(*b, mk, ms=4, color=c, zorder=z)
off = {'Roost': (0.05, 0.0), 'Roost-cal': (0.03, -0.055), 'Roost-cal*': (0.03, -0.035), 'EasyUQ': (-0.02, 0.03), 'This work': (0.04, -0.045)}
ha = {'EasyUQ': 'right'}
for name, a, b, c in models:
    dx, dy = off[name]; ax.text(b[0] + dx, b[1] + dy, '%s %d%%' % (name, round(100 * b[1])), fontsize=6.5, color=c, ha=ha.get(name, 'left'), va='center',
                                fontweight='bold' if name == 'This work' else 'normal')
ax.set_xlim(0.3, 1.5); ax.set_ylim(0.34, 0.92); ax.set_xticks([0.4, 0.8, 1.2]); ax.set_yticks([0.4, 0.6, 0.8])
ax.set_xlabel('80% interval width (eV)'); ax.set_ylabel('Coverage')
for yy, filled, lab in ((0.445, False, 'Composition holdout'), (0.395, True, 'Chemical-family holdout')):
    ax.plot(0.98, yy, 'o', ms=4, mfc=(NAVY if filled else 'white'), mec=NAVY, mew=1.0)
    ax.text(1.02, yy, lab, fontsize=6.2, va='center', color=NAVY)
fig.savefig(os.path.join(D, 'fig5b.svg')); plt.close(fig)

# ---------------- (c) gallery of predicted distributions (slot 1060 x 240 pt)
dens = pd.read_csv(os.path.join(D, 'fig5_density.csv')); meta = pd.read_csv(os.path.join(D, 'fig5_meta.csv')).set_index('formula')
show = ['Ga2O3', 'In2Se3', 'SiC', 'CrSBr']
names = {'CH5N2PbI3': 'FAPbI3'}
fig = plt.figure(figsize=(1060 * K, 240 * K))
w = 0.215
for j, f in enumerate(show):
    ax = fig.add_axes([0.04 + j * 0.245, 0.22, w, 0.62])
    d = dens[dens.formula == f]; m = meta.loc[f]
    x, y = d.x.values, d.density.values; y = y / y.max()
    ax.fill_between(x, 0, y, color=LTEAL, lw=0); sel = (x >= m.p10) & (x <= m.p90)
    ax.fill_between(x[sel], 0, y[sel], color='#9CCFCD', lw=0); ax.plot(x, y, color=TEAL2, lw=1.0)
    g = S[S.formula == f]
    good = [p for p in g.ph.unique() if p and '/' not in p and 'mixed' not in p]
    order = sorted(good, key=lambda p: g[g.ph == p].bandgap_ev.median()); cm = {p: PC[i] for i, p in enumerate(order[:4])}
    seen = {}
    for _, r in g.sort_values('bandgap_ev').iterrows():
        n = seen.get(round(r.bandgap_ev, 2), 0); seen[round(r.bandgap_ev, 2)] = n + 1
        c = cm.get(r.ph)
        if c: ax.plot(r.bandgap_ev, -0.10 - 0.07 * n, 'o', ms=3.0, color=c, mec='white', mew=0.35, clip_on=False, zorder=4)
        else: ax.plot(r.bandgap_ev, -0.10 - 0.07 * n, 'o', ms=2.6, mfc='white', mec=GREYP, mew=0.6, clip_on=False, zorder=4)
    SPAN = 4.0; ctr = 0.5 * (m.p10 + m.p90); lo = max(0.0, ctr - SPAN / 2); hi = lo + SPAN   # same 4 eV span in every panel
    ax.set_xlim(lo, hi); ax.set_ylim(-0.16 - 0.07 * max(seen.values()), 1.12); ax.set_yticks([]); ax.spines['left'].set_visible(False)
    ax.locator_params(axis='x', nbins=4)
    ax.set_title(fmt(names.get(f, f)), fontsize=7, fontweight='bold', loc='left', pad=2)
    ax.text(1.0, 1.02, '%d/%d' % (m.inside80, m.n), transform=ax.transAxes, ha='right', va='bottom', fontsize=7, color=TEAL2, fontweight='bold')
fig.text(0.5, 0.01, '$E_g$ (eV)', ha='center', va='bottom', fontsize=7)
fig.savefig(os.path.join(D, 'fig5c.svg')); plt.close(fig)
print('ok')
