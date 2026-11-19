"""Figure 3: chemistry sets how wide the set is."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gs5 import *
import numpy as np, pandas as pd
from sklearn.metrics import roc_curve, roc_auc_score
from pymatgen.core import Element

T = '/data/czx/GapScope_Revision_20261113/struct/tables/'
Wd = pd.read_csv(T + 'fig3_width.csv'); E = pd.read_csv(T + 'fig3_elements.csv')
lab = (Wd.exp_range > 0.2).values
rng = np.random.default_rng(5)

W, H = 7.0, 4.75
fig = plt.figure(figsize=(W, H))
def ax_in(x, y, w, h): return fig.add_axes([x / W, 1 - (y + h) / H, w / W, h / H])
def letter(s, x, y): fig.text(x / W, 1 - y / H, '(%s)' % s, fontsize=8, fontweight='bold', ha='left', va='top')
T0, HH = 0.24, 1.75

# (a) reported range, fixed vs variable cation coordination
axa = ax_in(0.50, T0, 1.55, HH)
for x, v, c, lab_ in ((0, 0, BLUE, 'Fixed'), (1, 1, CORAL, 'Variable')):
    r = np.sort(Wd[Wd.var_coordination == v].exp_range.values)
    dens = np.array([np.sum(np.abs(r - t) < 0.06) for t in r], float)
    jit = (rng.random(len(r)) - 0.5) * 0.55 * dens / dens.max()
    axa.scatter(x + jit, r, s=9, color=c, alpha=0.9, lw=0, zorder=2)
    m = np.median(r); axa.plot([x - 0.3, x + 0.3], [m, m], color=NAVY, lw=1.3, zorder=3)
    axa.text(x + 0.33, m, '%.2f' % m, fontsize=6.5, va='center')
axa.set_xticks([0, 1]); axa.set_xticklabels(['Fixed\n(%d)' % (Wd.var_coordination == 0).sum(), 'Variable\n(%d)' % (Wd.var_coordination == 1).sum()], fontsize=6.8)
axa.tick_params(axis='x', length=0, pad=2); axa.set_xlim(-0.5, 1.7); axa.set_ylim(-0.03, 1.3); axa.set_yticks([0, 0.4, 0.8, 1.2])
axa.set_ylabel('Reported range (eV)')
from scipy.stats import mannwhitneyu
pv = mannwhitneyu(Wd[Wd.var_coordination == 1].exp_range, Wd[Wd.var_coordination == 0].exp_range).pvalue
axa.text(-0.38, 1.24, '$P$ = %.4f' % pv, fontsize=6.5, va='top')
letter('a', 0.02, 0.06)


# (b) floor against the polymorph gap spread D_c (HSE06, polymorphs within 0.1 eV/atom), no fitting
Pd = pd.read_csv('/data/czx/GapScope_Revision_20261113/descr/d2_per_composition_energy.csv')   # energy-only D_c (2026-11-18)
axd = ax_in(2.78, T0, 1.55, HH)
big = (Pd.floor >= 0.05 - 1e-9).values   # floor of 0.05 eV or more (KNbO3 sits at 0.050)
axd.axhline(0.05, color=LBLUE, lw=0.8, ls=(0, (3, 2)), zorder=0)
axd.scatter(Pd.D_elec[~big], Pd.floor[~big], s=9, color=BLUE, lw=0, alpha=0.9, zorder=2)
axd.scatter(Pd.D_elec[big], Pd.floor[big], s=9, color=CORAL, lw=0, alpha=0.9, zorder=2)
pos = {'Ga2O3': (0.30, 0.30), 'SiC': (1.25, 0.31)}
for f_, (tx, ty) in pos.items():
    r_ = Pd[Pd.formula == f_].iloc[0]
    axd.annotate(fmt(f_), xy=(r_.D_elec, r_.floor), xytext=(tx, ty), fontsize=6.3, color=NAVY, ha='center', va='center',
                 arrowprops=dict(arrowstyle='-', color=SLATE, lw=0.5, shrinkA=1.5, shrinkB=2.5), zorder=4)
axd.text(0.03, 0.97, 'AUC %.2f' % roc_auc_score(big, Pd.D_elec), transform=axd.transAxes, fontsize=6.5, va='top', color=NAVY)
axd.set_xscale('symlog', linthresh=0.3, linscale=0.6); axd.set_xlim(-0.03, 7.5); axd.set_ylim(-0.02, 0.66)
axd.set_xticks([0, 0.3, 1, 3]); axd.set_xticklabels(['0', '0.3', '1', '3']); axd.xaxis.set_minor_locator(plt.NullLocator()); axd.set_yticks([0, 0.2, 0.4, 0.6])
axd.set_xlabel('$D_c$ (eV)'); axd.set_ylabel('Floor (eV)')
letter('b', 2.22, 0.06)

# (c) ROC: separating wide (> 0.2 eV) from narrow, cation families held out
axb = ax_in(5.20, T0, 1.65, HH)
axb.plot([0, 1], [0, 1], color=LBLUE, lw=0.8, ls=(0, (3, 2)))
aucs = []; grid = np.linspace(0, 1, 201); TPR = []
for i in range(20):
    f, t_, _ = roc_curve(lab, Wd['p%02d' % i]); TPR.append(np.interp(grid, f, t_)); aucs.append(roc_auc_score(lab, Wd['p%02d' % i]))
TPR = np.array(TPR); TPR[:, 0] = 0
axb.fill_between(grid, np.percentile(TPR, 10, 0), np.percentile(TPR, 90, 0), color=TEAL, alpha=0.22, lw=0)
axb.plot(grid, TPR.mean(0), color=TEAL2, lw=1.8)
for col, c, ls in (('var_coordination', CORAL, '-'), ('n_reports', SLATE, (0, (3, 1.5))), ('ent_max', PURP, (0, (1, 1.2)))):
    v = Wd[col].fillna(Wd[col].median()); f, t_, _ = roc_curve(lab, v); axb.plot(f, t_, color=c, lw=0.8, ls=ls)
rows = [('Model', np.mean(aucs), TEAL2), ('Coordination', roc_auc_score(lab, Wd.var_coordination), CORAL),
        ('Reports', roc_auc_score(lab, Wd.n_reports), SLATE), ('Elements', roc_auc_score(lab, Wd.ent_max.fillna(Wd.ent_max.median())), '#7A68B0')]
for j, (n, v, c) in enumerate(rows):
    axb.text(0.97, 0.05 + 0.095 * (3 - j), '%s %.2f' % (n, v), fontsize=6.3, color=c, ha='right', va='bottom', transform=axb.transAxes)
axb.set_xlim(0, 1); axb.set_ylim(0, 1.01); axb.set_xticks([0, 0.5, 1]); axb.set_yticks([0, 0.5, 1])
axb.set_xlabel('False positive rate'); axb.set_ylabel('True positive rate')
letter('c', 4.62, 0.06)

# (c) coordination repertoire of every cation across chemistry: large periodic table below
NONCAT = {'H', 'He', 'C', 'N', 'O', 'F', 'Ne', 'P', 'S', 'Cl', 'Ar', 'Se', 'Br', 'Kr', 'Te', 'I', 'Xe', 'At', 'Rn'}
E = E[~E.element.isin(NONCAT)]
nc = dict(zip(E.element, E.n_cn_any))
ramp = {1: '#EAF5F2', 2: '#A3D7CB', 3: '#2A9D8F', 4: '#264653'}
txt = {1: NAVY, 2: NAVY, 3: 'white', 4: 'white'}
axc = ax_in(0.35, 2.40, 5.30, 2.28); axc.axis('off')
for Z in range(1, 87):
    el = Element.from_Z(Z); s = el.symbol
    r, g = (7.75, Z - 57 + 3) if 57 <= Z <= 71 else (el.row, el.group)
    if s in nc:
        k = min(4, int(nc[s]))
        axc.add_patch(plt.Rectangle((g - 1, -r), 0.92, 0.92, fc=ramp[k], ec=(TEAL if k == 1 else 'white'), lw=(0.8 if k == 1 else 0.6)))
        axc.text(g - 1 + 0.46, -r + 0.46, s, fontsize=6.2, ha='center', va='center', color=txt[k], fontweight='bold' if k == 4 else 'normal')
    else:
        axc.add_patch(plt.Rectangle((g - 1, -r), 0.92, 0.92, fc='#F4F6F8', ec='white', lw=0.6))
        axc.text(g - 1 + 0.46, -r + 0.46, s, fontsize=5.4, ha='center', va='center', color='#C3CBD3')
axc.set_xlim(-0.1, 18.05); axc.set_ylim(-7.8, -0.02); axc.set_aspect('equal')
# legend: stepped scale, one swatch per count
axl = ax_in(5.85, 2.75, 1.05, 1.55); axl.axis('off'); axl.set_xlim(0, 1); axl.set_ylim(0, 1)
axl.text(0.0, 0.98, 'Coordination\nnumbers adopted', fontsize=6.5, va='top')
for j, k in enumerate((4, 3, 2, 1)):
    y = 0.62 - 0.14 * j
    axl.add_patch(plt.Rectangle((0.0, y), 0.22, 0.11, fc=ramp[k], ec=(TEAL if k == 1 else 'none'), lw=0.8))
    axl.text(0.30, y + 0.055, '$\geq$4' if k == 4 else str(k), fontsize=6.5, va='center')
    axl.text(0.62, y + 0.055, '%d' % sum(1 for v in nc.values() if min(4, int(v)) == k), fontsize=6.5, va='center', color=TEAL2 if k >= 2 else NAVY)
axl.text(0.62, 0.76, 'cations', fontsize=6, va='center')
axl.text(0.0, 0.02, '%d of %d cations\nadopt more than one' % (sum(1 for v in nc.values() if int(v) > 1), len(nc)), fontsize=6.5, va='bottom', color=TEAL2, fontweight='bold')
letter('d', 0.02, 2.30)
plt.rcParams['svg.fonttype'] = 'path'; fig.savefig('Fig3.pdf'); fig.savefig('Fig3.png', dpi=300); fig.savefig('Fig3.svg')
print([(n, round(v, 3)) for n, v, _ in rows], 'cations', len(nc), 'versatile', sum(1 for v in nc.values() if v >= 2))
