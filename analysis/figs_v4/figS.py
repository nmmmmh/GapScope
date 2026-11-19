"""SI figures S1 and S2 in the main-figure style."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gs5 import *
import numpy as np, pandas as pd
from pymatgen.core import Element

R0 = '/data/czx/GapScope_Revision_20261113/'
S = pd.read_csv(R0 + 'suppdata/Supplementary_Data_1_GapScope_experimental_bandgaps.csv')
S['ph'] = S.phase_label.fillna('').astype(str).str.strip().str.lower()
nph = S[S.ph != ''].groupby('composition_key').ph.nunique()
def letter(fig, W, H, s, x, y): fig.text(x / W, 1 - y / H, '(%s)' % s, fontsize=8, fontweight='bold', ha='left', va='top')

# ======================= Figure S1
W, H = 7.0, 2.6
fig = plt.figure(figsize=(W, H)); ax_in = lambda x, y, w, h: fig.add_axes([x / W, 1 - (y + h) / H, w / W, h / H])
axa = ax_in(1.05, 0.25, 1.85, 1.95)
m = S.measurement_method.fillna('').str.lower()
fam = np.select([m.str.contains('reflect') & ~m.str.contains('thermoreflect'), m.str.contains('thermoreflect|modulat|electroreflect|photoreflect'),
                 m.str.contains(r'photolumin|\bpl\b|emission'), m.str.contains('ellipso'), m.str.contains('absor|transmi|uv')],
                ['Diffuse reflectance', 'Modulation', 'Photoluminescence', 'Ellipsometry', 'Absorption'], 'Other')
tr = np.select([m.str.contains('indirect'), m.str.contains('direct')], ['Indirect', 'Direct'], 'Not stated')
T = pd.crosstab(pd.Series(fam, name='f'), pd.Series(tr, name='t')); T = T.loc[T.sum(1).sort_values().index]
left = np.zeros(len(T))
for t, c in (('Direct', TEAL2), ('Indirect', '#7FA7BF'), ('Not stated', GREYP)):
    if t in T: axa.barh(range(len(T)), T[t].values, left=left, color=c, height=0.62, lw=0, label=t); left += T[t].values
axa.set_yticks(range(len(T))); axa.set_yticklabels(T.index, fontsize=6.5); axa.tick_params(axis='y', length=0)
axa.set_xlabel('Reports'); axa.legend(loc='lower right', fontsize=6.3, handlelength=0.9, title='Transition', title_fontsize=6.3)
letter(fig, W, H, 'a', 0.02, 0.06)
axb = ax_in(3.25, 0.20, 3.70, 2.20); axb.axis('off')
cnt = {}
for k in S.composition_key.unique():
    for e in [t.split(':')[0] for t in k.split('|')]: cnt[e] = cnt.get(e, 0) + 1
vmax = max(cnt.values()); cmap = matplotlib.colors.LinearSegmentedColormap.from_list('c', ['#EAF5F2', '#2A9D8F', '#264653'])
for Z in range(1, 87):
    el = Element.from_Z(Z); s = el.symbol; r, g = (7.75, Z - 57 + 3) if 57 <= Z <= 71 else (el.row, el.group)
    if s in cnt:
        f = np.log1p(cnt[s]) / np.log1p(vmax)
        axb.add_patch(plt.Rectangle((g - 1, -r), 0.92, 0.92, fc=cmap(f), ec='white', lw=0.5))
        axb.text(g - 1 + 0.46, -r + 0.46, s, fontsize=4.6, ha='center', va='center', color='white' if f > 0.5 else NAVY)
    else:
        axb.add_patch(plt.Rectangle((g - 1, -r), 0.92, 0.92, fc='#F4F6F8', ec='white', lw=0.5))
axb.set_xlim(-0.1, 18.05); axb.set_ylim(-7.8, -0.02); axb.set_aspect('equal')
cb = ax_in(4.10, 0.42, 1.10, 0.07); cb.imshow(np.linspace(0, 1, 64)[None, :], cmap=cmap, aspect='auto', extent=[0, 1, 0, 1])
cb.set_yticks([]); cb.set_xticks([0, 1]); cb.set_xticklabels(['1', str(vmax)], fontsize=6); cb.tick_params(length=0, pad=1)
for sp in cb.spines.values(): sp.set_visible(False)
cb.set_title('Compositions', fontsize=6.3, pad=2)
letter(fig, W, H, 'b', 3.10, 0.06)
fig.savefig('FigS1.pdf'); fig.savefig('FigS1.png', dpi=300); plt.close(fig)

# ======================= Figure S2
a = pd.read_csv(R0 + 'cv/predictions_summary_v4a.csv'); t = pd.read_csv(R0 + 'cv/predictions_summary_v4t1.csv')
MODELS = [('ROOST', a, 'Roost', '#9FBFD3'), ('ROOST_CAL', a, 'Roost-cal', '#7FA7BF'), ('EASYUQ_KS', a, 'EasyUQ', '#7A68B0'), ('GS_SELCAL_PRE_NZ_SC', t, 'This work', TEAL2)]
def wmedian(y, w):
    o = np.argsort(y); y, w = y[o], w[o]; c = np.cumsum(w) / w.sum(); return y[np.searchsorted(c, 0.5)]
per = {}
for mm, d, lab, _ in MODELS:
    g = d[(d.model == mm) & (d.tier == 'A')]; rows = []
    for k, gg in g.groupby('key'):
        y, w = gg.y.values, gg.w.values
        rows.append((k, np.sum(w * np.abs(y - gg['median'].values)) / w.sum(), np.sum(w * np.abs(y - wmedian(y, w))) / w.sum(),
                     len(gg), int(nph.get(k, 0) >= 2), round(y.max() - y.min(), 6)))
    per[mm] = pd.DataFrame(rows, columns=['key', 'mae', 'floor', 'n', 'multi', 'range'])
W, H = 7.0, 2.25
fig = plt.figure(figsize=(W, H)); ax_in = lambda x, y, w, h: fig.add_axes([x / W, 1 - (y + h) / H, w / W, h / H])
xm = 0.68
for i, (mm, _, lab, col) in enumerate(MODELS):
    ax = ax_in(0.45 + 1.30 * i, 0.30, 1.08, 1.45); P = per[mm]; xx = np.linspace(0, xm, 20)
    ax.fill_between(xx, 0, xx, color=LCORAL, alpha=0.6, lw=0, zorder=0); ax.plot([0, xm], [0, xm], color=CORAL, lw=0.7, ls=(0, (3, 2)))
    s = 2.5 + 2.4 * np.sqrt(P.n); sel = P.multi == 1
    ax.scatter(P.floor[~sel], P.mae[~sel], s=s[~sel], facecolor='white', edgecolor=GREYP, lw=0.5, zorder=2)
    ax.scatter(P.floor[sel], P.mae[sel], s=s[sel], color=col, alpha=0.85, lw=0, zorder=3)
    ax.set_xlim(-0.02, xm); ax.set_ylim(-0.04, 2.0); ax.set_xticks([0, 0.3, 0.6]); ax.set_yticks([0, 1, 2])
    ax.set_title(lab, fontsize=7, pad=3, color=NAVY)
    if i == 0: ax.set_ylabel('MAE (eV)')
    else: ax.set_yticklabels([])
    letter(fig, W, H, 'abcd'[i], 0.02 if i == 0 else 0.30 + 1.30 * i, 0.06)
fig.text((0.45 + 2.49) / W, 1 - 2.13 / H, 'Floor (eV)', ha='center', fontsize=7)
axe = ax_in(5.95, 0.30, 0.95, 1.45)
for i, (mm, _, lab, col) in enumerate(MODELS):
    P = per[mm]; y = 3 - i; full, kept = P.mae.mean(), P[P.range <= 0.1 + 1e-9].mae.mean()
    axe.plot([kept, full], [y, y], color=GREYP, lw=1.0, zorder=1)
    axe.plot(full, y, 'o', ms=4.5, color=col, zorder=2); axe.plot(kept, y, 'o', ms=4.5, mfc='white', mec=col, mew=1.0, zorder=2)
axe.set_yticks([3, 2, 1, 0]); axe.set_yticklabels([m[2] for m in MODELS], fontsize=6.3); axe.tick_params(axis='y', length=0)
axe.set_xlim(0.2, 0.48); axe.set_xticks([0.25, 0.35, 0.45]); axe.set_ylim(-0.6, 3.7); axe.set_xlabel('MAE (eV)')
nk = int((per['ROOST'].range <= 0.1 + 1e-9).sum()); na = len(per['ROOST'])
for xx_, filled, lab in ((0.212, True, 'All (%d)' % na), (0.352, False, 'Kept (%d)' % nk)):
    axe.plot(xx_, 4.05, 'o', ms=4.5, mfc=(NAVY if filled else 'white'), mec=NAVY, mew=1.0, clip_on=False)
    axe.text(xx_ + 0.02, 4.05, lab, fontsize=6.2, va='center', color=NAVY, clip_on=False)
letter(fig, W, H, 'e', 5.15, 0.06)
fig.savefig('FigS2.pdf'); fig.savefig('FigS2.png', dpi=300); plt.close(fig)
print('ok', int((per['ROOST'].range <= 0.1 + 1e-9).sum()), 'kept')
