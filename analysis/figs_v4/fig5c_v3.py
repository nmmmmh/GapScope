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
    seen = {}; levels = []   # stack a dot one row lower whenever it sits within 0.06 eV of a dot already on that row
    for _, r in g.sort_values('bandgap_ev').iterrows():
        n = 0
        while n < len(levels) and r.bandgap_ev - levels[n] < 0.06: n += 1
        if n == len(levels): levels.append(r.bandgap_ev)
        else: levels[n] = r.bandgap_ev
        seen[len(seen)] = n
        c = cm.get(r.ph)
        if c: ax.plot(r.bandgap_ev, -0.10 - 0.07 * n, 'o', ms=3.0, color=c, mec='white', mew=0.35, clip_on=False, zorder=4)
        else: ax.plot(r.bandgap_ev, -0.10 - 0.07 * n, 'o', ms=2.6, mfc='white', mec=GREYP, mew=0.6, clip_on=False, zorder=4)
    SPAN = 4.0; ctr = 0.5 * (m.p10 + m.p90); lo = max(0.0, ctr - SPAN / 2); hi = lo + SPAN   # same 4 eV span in every panel
    ax.set_xlim(lo, hi); ax.set_ylim(-0.16 - 0.07 * len(levels), 1.12); ax.set_yticks([]); ax.spines['left'].set_visible(False)
    ax.locator_params(axis='x', nbins=4)
    ax.set_title(fmt(names.get(f, f)), fontsize=7, fontweight='bold', loc='left', pad=2)
    ax.text(1.0, 1.02, '%d/%d' % (m.inside80, m.n), transform=ax.transAxes, ha='right', va='bottom', fontsize=7, color=TEAL2)
fig.text(0.5, 0.01, '$E_g$ (eV)', ha='center', va='bottom', fontsize=7)
fig.savefig(os.path.join(D, 'fig5c.svg')); plt.close(fig)
print('ok')
