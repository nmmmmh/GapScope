"""Two-dimensional maps for the SI.
FigS3: compositions, t-SNE of the 64-d Roost representation of the 2,467 compiled compositions.
FigS4: structures, t-SNE of six standardized local descriptors of the SNUMAT structures.
Run on c137-13: OMP_NUM_THREADS=1 <venv>/python -I maps2d.py"""
import sys, os
sys.path.insert(0, '/data/czx/GapScope_Revision_20261113/figs_v4')
from gs5 import *
import numpy as np, pandas as pd, torch
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from matplotlib.colors import LinearSegmentedColormap, LogNorm
plt.rcParams['svg.fonttype'] = 'path'
R = '/data/czx/GapScope_Revision_20261113/'
O = R + 'maps2d/'; os.makedirs(O, exist_ok=True)

# ---------------- data: compositions
d = torch.load(R + 'cv/data/tierB_features.pt', map_location='cpu', weights_only=False)
C = d['composition'].numpy(); keys = list(d['keys'])
tA = torch.load(R + 'cv/data/tierA.pt', map_location='cpu', weights_only=False)
CA = tA['features']['composition'].numpy(); kA = [r['composition_key'] for r in tA['records']]
seen = set(keys); extra = []
for i, k in enumerate(kA):
    if k not in seen: seen.add(k); extra.append(i); keys.append(k)
C = np.vstack([C, CA[extra]])
print('fraction block sums', np.round(C[:5, :118].sum(1), 3), 'added traced', len(extra))
emb = C[:, 118:182]
w = pd.read_csv(R + 'struct/width.csv')
wc = w[w.set == 'compilation'].set_index('composition_key')
wp = w[w.set == 'provenance'].set_index('composition_key')
a = pd.read_csv(R + 'cv/predictions_summary_v4a.csv'); g0 = a[(a.model == 'ROOST') & (a.tier == 'A')]
def wm(y, ww):
    o = np.argsort(y); y, ww = y[o], ww[o]; c = np.cumsum(ww) / ww.sum(); return y[np.searchsorted(c, 0.5)]
floor = {k: np.sum(g.w.values * np.abs(g.y.values - wm(g.y.values, g.w.values))) / g.w.sum() for k, g in g0.groupby('key')}
def dc(k):
    for t in (wc, wp):
        if k in t.index and t.multi_polymorph.get(k, 0) == 1: return t.gapspr_hse[k]
    return np.nan
Dc = np.array([dc(k) for k in keys], float)
is_tr = np.array([k in floor for k in keys])
print('compositions', len(keys), 'with Dc', np.isfinite(Dc).sum(), 'traced in map', is_tr.sum(), 'traced total', len(floor))

Ec = TSNE(2, perplexity=40, init='pca', learning_rate='auto', random_state=0, n_jobs=1).fit_transform(StandardScaler().fit_transform(emb))
pd.DataFrame({'key': keys, 'x': Ec[:, 0], 'y': Ec[:, 1], 'Dc': Dc, 'floor': [floor.get(k, np.nan) for k in keys]}).to_csv(O + 'comp_map.csv', index=False)

# ---------------- data: structures
S = pd.read_csv(R + 'struct/descriptors.csv')
F6 = ['cn_cat_mean', 'mean_shared_vertices', 'dim_cat_an', 'bond_cat_an_mean', 'distortion_index', 'vol_per_atom']
S = S.dropna(subset=F6 + ['Band_gap_HSE']).reset_index(drop=True)
Es = TSNE(2, perplexity=40, init='pca', learning_rate='auto', random_state=0, n_jobs=1).fit_transform(StandardScaler().fit_transform(S[F6].values))
S['x'], S['y'] = Es[:, 0], Es[:, 1]
S[['composition_key', 'formula', 'sg_number', 'Band_gap_HSE', 'x', 'y'] + F6].to_csv(O + 'struct_map.csv', index=False)
npoly = S.groupby('composition_key').size()
print('structures', len(S), 'compositions', S.composition_key.nunique(), 'with >=2 structures', (npoly >= 2).sum())

import re
def sub(f):
    f = f.replace('CH5N2', 'FA')
    return re.sub(r'(\d+(\.\d+)?)', r'$_{\1}$', f)
OFF3 = {'CsSnI3': (10, 8, 'left'), 'CsPbI3': (-22, 12, 'right'), 'CH5N2PbI3': (14, 16, 'left'),
        'NiFe2O4': (14, 12, 'left'), 'LuFeO3': (16, -4, 'left'), 'ZrO2': (-12, -6, 'right')}
OFF4 = {'CsSnI3': (6, 3, 'left'), 'ZrO2': (-4, 9, 'right'), 'BN': (-7, 3, 'right'), 'MoO3': (7, -6, 'left'),
        'Ta2O5': (6, 3, 'left'), 'SiC': (6, -10, 'left')}
def lab(A, name, xy, off):
    dx, dy, ha = off.get(name, (6, 4, 'left'))
    A.annotate(sub(name), xy, xytext=(dx, dy), textcoords='offset points', fontsize=6.4, color=NAVY, ha=ha, va='center', zorder=4,
               arrowprops=dict(arrowstyle='-', color=NAVY, lw=0.4, shrinkA=0, shrinkB=2))
fmlA = {r['composition_key']: r['formula'] for r in tA['records']}
cm_c = LinearSegmentedColormap.from_list('c', [LTEAL, TEAL, CORAL])
def clean(ax):
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values(): sp.set_visible(False)

# ---------------- FigS3
fig = plt.figure(figsize=(7.0, 3.1))
ax = fig.add_axes([0.02, 0.05, 0.40, 0.84]); bx = fig.add_axes([0.53, 0.05, 0.40, 0.84])
m = np.isfinite(Dc)
for A in (ax, bx): A.scatter(Ec[~m, 0], Ec[~m, 1], s=2.2, c='#D9DDE3', lw=0, zorder=1)
o = np.argsort(Dc[m]); xm, ym, dm = Ec[m, 0][o], Ec[m, 1][o], Dc[m][o]
sc = ax.scatter(xm, ym, s=4.5, c=np.clip(dm, 0.02, 3), cmap=cm_c, norm=LogNorm(0.02, 3), lw=0, zorder=2)
cb = fig.colorbar(sc, ax=ax, fraction=0.04, pad=0.02); cb.set_label('Polymorph gap spread (eV)', fontsize=7); cb.ax.tick_params(labelsize=6)
bx.scatter(xm, ym, s=2.2, c='#B8C0CC', lw=0, zorder=1)
ft = np.array([floor[k] for k in np.array(keys)[is_tr]])
o2 = np.argsort(ft)
sb = bx.scatter(Ec[is_tr, 0][o2], Ec[is_tr, 1][o2], s=26, c=ft[o2], cmap=cm_c, vmin=0, vmax=0.3,
                edgecolor=NAVY, lw=0.4, zorder=3)
kt = np.array(keys)[is_tr]
for j in np.argsort(-ft)[:6]:
    lab(bx, fmlA[kt[j]], tuple(Ec[is_tr][j]), OFF3)
cb2 = fig.colorbar(sb, ax=bx, fraction=0.04, pad=0.02); cb2.set_label('Floor (eV)', fontsize=7); cb2.ax.tick_params(labelsize=6)
for A in (ax, bx): clean(A)
fig.text(0.008, 0.93, '(a)', fontsize=8, fontweight='bold'); fig.text(0.515, 0.93, '(b)', fontsize=8, fontweight='bold')
for e in ('pdf', 'png', 'svg'): fig.savefig(O + 'FigS3.' + e, dpi=600)

# ---------------- FigS4
fig = plt.figure(figsize=(7.0, 3.1))
ax = fig.add_axes([0.02, 0.05, 0.40, 0.84]); bx = fig.add_axes([0.53, 0.05, 0.40, 0.84])
o = np.argsort(S.Band_gap_HSE.values)
sc = ax.scatter(S.x.values[o], S.y.values[o], s=2.0, c=S.Band_gap_HSE.values[o], cmap='viridis', vmin=0, vmax=8, lw=0)
cb = fig.colorbar(sc, ax=ax, fraction=0.04, pad=0.02); cb.set_label('HSE06 gap (eV)', fontsize=7); cb.ax.tick_params(labelsize=6)
bx.scatter(S.x, S.y, s=2.0, c='#D9DDE3', lw=0, zorder=1)
tr = [k for k in floor if npoly.get(k, 0) >= 3]
sel = sorted(tr, key=lambda k: -floor[k])[:6]
print('S4b compositions', [(fmlA[k], int(npoly[k]), round(floor[k], 3)) for k in sel])
for k in sel:
    g = S[S.composition_key == k].sort_values('Band_gap_HSE')
    bx.plot(g.x, g.y, color=NAVY, lw=0.6, alpha=0.6, zorder=2)
    bx.scatter(g.x, g.y, s=16, c=g.Band_gap_HSE, cmap='viridis', vmin=0, vmax=8, edgecolor=NAVY, lw=0.4, zorder=3)
    i = g.x.values.argmax(); lab(bx, fmlA[k], (g.x.values[i], g.y.values[i]), OFF4)
cb2 = fig.colorbar(sc, ax=bx, fraction=0.04, pad=0.02); cb2.set_label('HSE06 gap (eV)', fontsize=7); cb2.ax.tick_params(labelsize=6)
for A in (ax, bx): clean(A)
fig.text(0.008, 0.93, '(a)', fontsize=8, fontweight='bold'); fig.text(0.515, 0.93, '(b)', fontsize=8, fontweight='bold')
for e in ('pdf', 'png', 'svg'): fig.savefig(O + 'FigS4.' + e, dpi=600)

# ---------------- numbers for the captions / text
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr
mp = S[S.composition_key.map(npoly) >= 2]
dist, dg = [], []
for k, g in mp.groupby('composition_key'):
    dist += list(pdist(g[['x', 'y']].values)); dg += list(pdist(g[['Band_gap_HSE']].values))
rnd = pdist(S[['x', 'y']].sample(3000, random_state=0).values)
print('median map distance within composition %.1f, random pairs %.1f' % (np.median(dist), np.median(rnd)))
print('rho(map distance, |dEg|) within compositions %.3f' % spearmanr(dist, dg).statistic)
