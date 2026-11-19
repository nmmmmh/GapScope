"""Figure 1 lower panels (SVG): (b) Ga2O3 provenance tree, (c) within-study ladder ridgelines. Slots in slide points (1080 pt = 7.0 in)."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gs5 import *
plt.rcParams['svg.fonttype'] = 'path'
import numpy as np, pandas as pd
from matplotlib.path import Path
from matplotlib.patches import PathPatch

R0 = '/data/czx/GapScope_Revision_20261113/'
S = pd.read_csv(R0 + 'suppdata/Supplementary_Data_1_GapScope_experimental_bandgaps.csv')
L = pd.read_csv(R0 + 'struct/tables/ladder_pairs.csv')
K = 7.0 / 1080.0
def norm_ph(s):
    s = str(s).strip().lower() if isinstance(s, str) else ''
    for g in ('α', 'β', 'γ', 'δ', 'κ', 'ε'):
        if s in (g, g + '-ga2o3'): return g
    return s
S['ph'] = S.phase_label.apply(norm_ph)
PCOL = {'β': PH[0], 'γ': PH[1], 'κ': PH[2], 'α': PH[3]}
def bez(ax, x0, y0, x1, y1, c, lw, a=1.0, z=1):
    dy = (y0 - y1) * 0.55
    p = Path([(x0, y0), (x0, y0 - dy), (x1, y1 + dy), (x1, y1)], [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4])
    ax.add_patch(PathPatch(p, fc='none', ec=c, lw=lw, alpha=a, zorder=z, capstyle='round'))

# ---------------- (b) tree, slot 650 x 300 pt
fig = plt.figure(figsize=(650 * K, 300 * K))
ax = fig.add_axes([0.03, 0.17, 0.95, 0.80])
G = S[S.formula == 'Ga2O3']
YS, YP, YL = 1.0, 0.66, 0.40
spec_x, src_x = {}, {}
for sid, gs in G.groupby('source_group_id'):
    xs = []
    for pid, gp in gs.groupby('specimen_group_id'):
        spec_x[pid] = gp.bandgap_ev.mean(); xs.append(spec_x[pid])
    src_x[sid] = np.mean(xs)
for sid, gs in G.groupby('source_group_id'):
    for pid, gp in gs.groupby('specimen_group_id'):
        bez(ax, src_x[sid], YS, spec_x[pid], YP, '#C3CEDA', 0.6, 0.8)
        seen = {}
        for _, r in gp.iterrows():
            c = PCOL.get(r.ph, GREYP); n = seen.get(r.bandgap_ev, 0); seen[r.bandgap_ev] = n + 1
            yl = YL + 0.055 * n
            bez(ax, spec_x[pid], YP, r.bandgap_ev, yl + 0.03, c, 0.9, 0.85, 2)
            ax.plot(r.bandgap_ev, yl, 'o', ms=3.8, color=c, mec='white', mew=0.45, zorder=4)
        ax.plot(spec_x[pid], YP, 'o', ms=2.4, mfc='white', mec=NAVY, mew=0.55, zorder=3)
    ax.plot(src_x[sid], YS, 'o', ms=3.8, color=NAVY, mew=0, zorder=3)
# phase range strips under the leaves
order = ['α', 'κ', 'γ', 'β']
for j, p in enumerate(order):
    v = G[G.ph == p].bandgap_ev; y = 0.29 - 0.088 * j
    ax.plot([v.min(), v.max()], [y, y], color=PCOL[p], lw=2.6, solid_capstyle='round', zorder=3)
    ax.text(4.03, y, p, fontsize=6.5, color=PCOL[p], ha='left', va='center', fontweight='bold')
for y, t in ((YS, 'Source (%d)' % G.source_group_id.nunique()), (YP, 'Specimen (%d)' % G.specimen_group_id.nunique()), (YL, 'Report (%d)' % len(G))):
    ax.text(4.03, y, t, fontsize=6.5, color='black', ha='left', va='center')
ax.text(5.40, 1.04, fmt('Ga2O3'), fontsize=7.5, ha='right', va='center', fontweight='bold')
ax.set_xlim(4.02, 5.42); ax.set_ylim(-0.04, 1.08); ax.spines['bottom'].set_bounds(4.3, 5.42); ax.set_yticks([]); ax.spines['left'].set_visible(False)
ax.set_xticks([4.4, 4.6, 4.8, 5.0, 5.2, 5.4]); ax.set_xlabel('$E_g$ (eV)')
fig.savefig('fig1b.svg'); fig.savefig('fig1b.pdf'); plt.close(fig)

# ---------------- (c) ridgelines, slot 400 x 300 pt
fig = plt.figure(figsize=(400 * K, 300 * K))
ax = fig.add_axes([0.31, 0.17, 0.65, 0.80])
cats = [('phase_same_study', 'Phase', CORAL), ('specimen', 'Specimen', TEAL), ('repeat', 'Repeat', SLATE)]
xx = np.linspace(0, 1.05, 400)
for j, (k, lab, c) in enumerate(cats):
    v = L[L.kind == k].d.values; y0 = 2 - j; bw = 0.055
    kde = np.exp(-0.5 * ((xx[:, None] - v[None, :]) / bw) ** 2).sum(1) + np.exp(-0.5 * ((xx[:, None] + v[None, :]) / bw) ** 2).sum(1)
    kde = kde / kde.max() * 0.80
    ax.fill_between(xx, y0, y0 + kde, color=c, alpha=(0.38, 0.16, 0.12)[j], lw=0, zorder=2 + j)
    ax.plot(xx, y0 + kde, color=c, lw=(1.4, 0.8, 0.7)[j], zorder=3 + j)
    ax.plot([0, 1.05], [y0, y0], color=c, lw=0.5, zorder=3 + j)
    m = v.mean(); km = np.interp(m, xx, kde)
    ax.plot([m, m], [y0, y0 + km], color=NAVY, lw=0.9, zorder=6)
    ax.text(m + 0.03, y0 + km + 0.04, '%.2f' % m, fontsize=6.5, color=NAVY, va='bottom', fontweight='bold' if j == 0 else 'normal')
    ax.text(-0.03, y0 + 0.10, '%s (%d)' % (lab, len(v)), fontsize=7, ha='right', va='bottom')
ax.set_xlim(0, 1.05); ax.set_ylim(-0.05, 3.0); ax.set_yticks([]); ax.spines['left'].set_visible(False)
ax.set_xticks([0, 0.5, 1.0]); ax.set_xlabel('Within-study |$\\Delta E_g$| (eV)')
fig.savefig('fig1c.svg'); fig.savefig('fig1c.pdf'); plt.close(fig)
print('ok')
