"""Figure 1 (v5 concept): provenance tree of one composition, within-study ladder as ridgelines, spread fan of all compositions."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gs5 import *
import numpy as np, pandas as pd
from matplotlib.path import Path
from matplotlib.patches import PathPatch

R0 = '/data/czx/GapScope_Revision_20261113/'
S = pd.read_csv(R0 + 'suppdata/Supplementary_Data_1_GapScope_experimental_bandgaps.csv')
def norm_ph(s):
    s = str(s).strip().lower() if isinstance(s, str) else ''
    for g in ('α', 'β', 'γ', 'δ', 'κ', 'ε'):
        if s in (g, g + '-ga2o3'): return g
    return s
S['ph'] = S.phase_label.apply(norm_ph)
L = pd.read_csv(R0 + 'struct/tables/ladder_pairs.csv')
PC = [TEAL, BLUE, PURP, AMBER]                     # phase 1-4 within a composition, ordered by median gap
LINK = '#B4C3D3'

def phase_colors(g):
    good = [p for p in g.ph.unique() if p and '/' not in p and 'mixed' not in p]
    order = sorted(good, key=lambda p: g[g.ph == p].bandgap_ev.median())
    return {p: PC[i] for i, p in enumerate(order[:4])}

W, H = 7.0, 4.55
fig = plt.figure(figsize=(W, H))
def ax_in(x, y, w, h): return fig.add_axes([x / W, 1 - (y + h) / H, w / W, h / H])
def letter(s, x, y): fig.text(x / W, 1 - y / H, '(%s)' % s, fontsize=8, fontweight='bold', ha='left', va='top')
def bez(ax, x0, y0, x1, y1, c, lw, a=1.0, z=1):
    dy = (y0 - y1) * 0.55
    p = Path([(x0, y0), (x0, y0 - dy), (x1, y1 + dy), (x1, y1)], [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4])
    ax.add_patch(PathPatch(p, fc='none', ec=c, lw=lw, alpha=a, zorder=z, capstyle='round'))

# ======================= (a) provenance tree of Ga2O3, leaves on the gap axis
axa = ax_in(0.55, 0.22, 3.75, 1.95)
G = S[S.formula == 'Ga2O3'].copy(); cmap = phase_colors(G)
YS, YP, YL = 1.0, 0.62, 0.0
srcs = list(G.groupby('source_group_id'))
spec_x, src_x = {}, {}
for sid, gs in srcs:
    xs = []
    for pid, gp in gs.groupby('specimen_group_id'):
        spec_x[pid] = gp.bandgap_ev.mean(); xs.append(spec_x[pid])
    src_x[sid] = np.mean(xs)
for sid, gs in srcs:
    for pid, gp in gs.groupby('specimen_group_id'):
        bez(axa, src_x[sid], YS, spec_x[pid], YP, LINK, 0.9)
        seen = {}
        for _, r in gp.iterrows():
            c = cmap.get(r.ph, GREYP); n = seen.get(r.bandgap_ev, 0); seen[r.bandgap_ev] = n + 1
            yl = YL + 0.07 * n
            bez(axa, spec_x[pid], YP, r.bandgap_ev, yl + 0.04, c, 1.0, 0.9, 2)
            axa.plot(r.bandgap_ev, yl, 'o', ms=4.2, color=c, mec='white', mew=0.5, zorder=4)
        axa.plot(spec_x[pid], YP, 'o', ms=2.6, mfc='white', mec=NAVY, mew=0.6, zorder=3)
    axa.plot(src_x[sid], YS, 'o', ms=4.0, color=NAVY, mew=0, zorder=3)
for y, t in ((YS, 'Source'), (YP, 'Specimen'), (YL, 'Report')):
    pass
axa.set_xlim(4.25, 5.42); axa.set_ylim(-0.12, 1.12); axa.set_yticks([])
for sp in ('left',):
    axa.spines[sp].set_visible(False)
axa.set_xticks([4.4, 4.6, 4.8, 5.0, 5.2, 5.4]); axa.set_xlabel('$E_g$ (eV)')
for p, c in cmap.items():
    xm = G[G.ph == p].bandgap_ev.median()
axa.text(5.40, 1.09, fmt('Ga2O3'), fontsize=7.5, ha='right', va='top', fontweight='bold', color='black')
lx = 5.40
for p in sorted(cmap, key=lambda q: -G[G.ph == q].bandgap_ev.median()):
    pass
for i, p in enumerate(['α', 'κ', 'γ', 'β']):
    if p in cmap:
        axa.text(5.40 - 0.075 * i, 0.93, p, fontsize=8, color=cmap[p], ha='right', va='top', fontweight='bold')
letter('a', 0.02, 0.06)

# ======================= (b) within-study ladder as ridgelines
axb = ax_in(4.95, 0.22, 1.98, 1.95)
cats = [('phase_same_study', 'Phase', TEAL2), ('specimen', 'Specimen', BLUE), ('repeat', 'Repeat', GREYP)]
xx = np.linspace(0, 1.05, 400)
for j, (k, lab, c) in enumerate(cats):
    v = L[L.kind == k].d.values; y0 = 2 - j
    bw = 0.055
    kde = np.exp(-0.5 * ((xx[:, None] - v[None, :]) / bw) ** 2).sum(1)
    kde += np.exp(-0.5 * ((xx[:, None] + v[None, :]) / bw) ** 2).sum(1)      # reflect at zero
    kde = kde / kde.max() * 0.82
    axb.fill_between(xx, y0, y0 + kde, color=c, alpha=0.30, lw=0, zorder=2 + j)
    axb.plot(xx, y0 + kde, color=c, lw=0.9, zorder=3 + j)
    axb.plot([0, 1.05], [y0, y0], color=c, lw=0.5, zorder=3 + j)
    m = v.mean(); km = np.interp(m, xx, kde)
    axb.plot([m, m], [y0, y0 + km], color=NAVY, lw=0.9, zorder=6)
    axb.text(m + 0.03, y0 + km + 0.04, '%.2f' % m, fontsize=6.5, color=NAVY, va='bottom', ha='left',
             fontweight='bold' if k.startswith('phase') else 'normal')
    axb.text(-0.03, y0 + 0.12, lab, fontsize=7, ha='right', va='bottom', color='black')
axb.set_xlim(0, 1.05); axb.set_ylim(-0.05, 3.0); axb.set_yticks([]); axb.spines['left'].set_visible(False)
axb.set_xticks([0, 0.5, 1.0]); axb.set_xlabel('Within-study |$\\Delta E_g$| (eV)')
letter('b', 4.35, 0.06)

# ======================= (c) spread fan: every composition's reports about its median
axc = ax_in(0.55, 2.78, 6.38, 1.42)
rows = []
for k, g in S.groupby('composition_key'):
    if g.bandgap_ev.max() > g.bandgap_ev.min():
        rows.append((g.bandgap_ev.max() - g.bandgap_ev.min(), k, g))
rows.sort(key=lambda t: -t[0])
for i, (rg, k, g) in enumerate(rows):
    med = g.bandgap_ev.median(); cm = phase_colors(g)
    axc.plot([i, i], [g.bandgap_ev.min() - med, g.bandgap_ev.max() - med], color=LINK, lw=1.4, solid_capstyle='round', zorder=1)
    for _, r in g.iterrows():
        c = cm.get(r.ph)
        if c:
            axc.plot(i, r.bandgap_ev - med, 'o', ms=2.9, color=c, mec='white', mew=0.35, zorder=3)
        else:
            axc.plot(i, r.bandgap_ev - med, 'o', ms=2.4, mfc='white', mec=GREYP, mew=0.6, zorder=2)
axc.axhline(0, color=NAVY, lw=0.5, ls=(0, (3, 2)), zorder=0)
axc.set_xlim(-1, len(rows)); YM = max(max(abs(r[2].bandgap_ev - r[2].bandgap_ev.median())) for r in rows) + 0.06; axc.set_ylim(-YM, YM); axc.set_yticks([-0.5, 0, 0.5]); axc.set_xticks([])
axc.set_ylabel('$E_g$ $-$ median (eV)'); axc.set_xlabel('Composition, by spread', labelpad=3)
def find(pred): return [i for i, r in enumerate(rows) if pred(r)][0]
iga = find(lambda r: r[2].formula.iloc[0] == 'Ga2O3')
ifa = find(lambda r: all(('%s:' % e) in r[1] for e in ('Pb', 'I', 'C', 'N')) and 'Cs:' not in r[1] and 'Br:' not in r[1])
for i, name, y in ((iga, 'Ga2O3', 0.66), (ifa, 'FAPbI3', -0.66)):
    pass
letter('c', 0.02, 2.66)
fig.savefig('Fig1.pdf'); fig.savefig('Fig1.png', dpi=300)
print('fan compositions', len(rows), 'Ga2O3 phases', cmap)
