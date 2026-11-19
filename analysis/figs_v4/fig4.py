"""Figure 4: local structure orders the polymorphs of one composition only weakly (HSE pairs)."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gs5 import *
import numpy as np, pandas as pd
from sklearn.metrics import roc_curve, roc_auc_score

D = os.path.dirname(os.path.abspath(__file__))
P = pd.read_csv(os.path.join(D, 'fig5_scatter.csv')); M = pd.read_csv(os.path.join(D, 'fig5_matched.csv'))
rng = np.random.default_rng(2)
W, H = 7.0, 2.26
fig = plt.figure(figsize=(W, H))
def ax_in(x, y, w, h): return fig.add_axes([x / W, 1 - (y + h) / H, w / W, h / H])
def letter(s, x, y): fig.text(x / W, 1 - y / H, '(%s)' % s, fontsize=8, fontweight='bold', ha='left', va='top')
T0, HH = 0.24, 1.70

# (a) gap difference against coordination difference, every pair
axa = ax_in(0.45, T0, 2.45, HH)
Q = P[P.d_cn.notna()]; Q = pd.concat([Q, Q.assign(d_cn=-Q.d_cn, observed=-Q.observed)], ignore_index=True)  # both orders
bins = [-2, -1, -0.5, 0, 0.5, 1, 2]
Q = Q[Q.d_cn.isin(bins)]
axa.scatter(Q.d_cn + (rng.random(len(Q)) - 0.5) * 0.28, Q.observed, s=1.6, color=INDIGO, alpha=0.16, lw=0, zorder=1)
mx, my, lo, hi = [], [], [], []
for b in bins:
    v = Q[Q.d_cn == b]; w = v.weight.values; y = v.observed.values
    m = np.average(y, weights=w); bs = []
    for _ in range(1000):
        i = rng.integers(0, len(y), len(y)); bs.append(np.average(y[i], weights=w[i]))
    mx.append(b); my.append(m); lo.append(np.percentile(bs, 2.5)); hi.append(np.percentile(bs, 97.5))
axa.fill_between(mx, lo, hi, color=TEAL, alpha=0.30, lw=0, zorder=2); axa.plot(mx, my, color=TEAL2, lw=1.7, marker='o', ms=3.4, zorder=3)
axa.axhline(0, color=NAVY, lw=0.5, ls=(0, (3, 2)), zorder=0)
ga = P[P.formula == 'Ga2O3'].iloc[0]
axa.plot(ga.d_cn, ga.observed, marker='*', ms=10, color=CORAL, mec='white', mew=0.4, zorder=5)
axa.text(ga.d_cn + 0.12, ga.observed + 0.05, fmt('Ga2O3') + ' (α − β)', fontsize=6.5, va='bottom')
for xb, yb in ((-2, my[0]), (2, my[-1])): axa.text(xb, yb + (0.13 if xb < 0 else -0.13), ('%+.2f' % yb).replace('-', '−'), fontsize=6.5, color=TEAL2, ha='center', va='bottom' if xb < 0 else 'top', fontweight='bold', bbox=dict(fc='white', ec='none', alpha=0.85, pad=0.6))
axa.text(2.4, -1.45, '%s pairs' % format(len(P), ','), fontsize=6.5, color=NAVY, va='bottom', ha='right')
axa.set_xticks(bins); axa.set_xticklabels(['$-$2', '$-$1', '$-$0.5', '0', '0.5', '1', '2'], fontsize=6)
axa.set_xlim(-2.5, 2.5); axa.set_ylim(-1.6, 1.6); axa.set_yticks([-1.5, -1, -0.5, 0, 0.5, 1, 1.5])
axa.set_xlabel('$\\Delta$ cation coordination number'); axa.set_ylabel('$\\Delta E_g$ between polymorphs (eV)')
letter('a', 0.02, 0.06)

# (b) |gap difference|: all pairs against pairs matched in five local descriptors
axb = ax_in(3.45, T0, 1.55, HH)
allv = np.abs(P.observed.values); mat = M.abs_dgap.values
edges = np.linspace(0, 1.6, 33)
h1 = axb.hist(allv, bins=edges, density=True, color=LBLUE, lw=0, label='All pairs')[0]
h2 = axb.hist(mat, bins=edges, density=True, histtype='step', color=CORAL, lw=1.4, label='Matched')[0]; YT = 1.12 * max(h1.max(), h2.max())
for v, c in ((allv.mean(), INDIGO), (mat.mean(), CORAL)):
    axb.axvline(v, color=c, lw=0.8, ls=(0, (3, 1.5)))
axb.text(allv.mean() + 0.03, 0.62 * YT, '%.2f' % allv.mean(), fontsize=6.5, color='#3E7CB8')
axb.text(mat.mean() + 0.03, 0.74 * YT, '%.2f' % mat.mean(), fontsize=6.5, color=CORAL, fontweight='bold')
axb.set_xlim(0, 1.6); axb.set_xticks([0, 0.5, 1, 1.5]); axb.set_ylim(0, YT); axb.set_yticks([]); axb.set_ylabel('Density')
axb.spines['left'].set_visible(False); axb.set_xlabel('|$\\Delta E_g$| (eV)')
axb.text(1.58, 0.93 * YT, 'All pairs (%s)' % format(len(allv), ','), fontsize=6.3, color='#3E7CB8', ha='right'); axb.text(1.58, 0.84 * YT, 'Matched (%s)' % format(len(mat), ','), fontsize=6.3, color=CORAL, ha='right')
letter('b', 3.10, 0.06)

# (c) sign of the gap difference predicted from local structure
axc = ax_in(5.45, T0, 1.48, HH)
axc.plot([0, 1], [0, 1], color=LBLUE, lw=0.8, ls=(0, (3, 2)))
lab = P.observed > 0
f, t, _ = roc_curve(lab, P.predicted); a1 = roc_auc_score(lab, P.predicted)
axc.plot(f, t, color=TEAL2, lw=1.1)
big = np.abs(P.observed) > 0.5
f2, t2, _ = roc_curve(lab[big], P.predicted[big]); a2 = roc_auc_score(lab[big], P.predicted[big])
axc.plot(f2, t2, color=CORAL, lw=1.6)
axc.text(0.97, 0.16, 'All %.2f' % a1, fontsize=6.3, color=TEAL2, ha='right', transform=axc.transAxes)
axc.text(0.97, 0.06, '|$\\Delta E_g$| > 0.5 eV %.2f' % a2, fontsize=6.3, color=CORAL, ha='right', transform=axc.transAxes)
axc.set_xlim(0, 1); axc.set_ylim(0, 1.01); axc.set_xticks([0, 0.5, 1]); axc.set_yticks([0, 0.5, 1])
axc.set_xlabel('False positive rate'); axc.set_ylabel('True positive rate')
letter('c', 5.05, 0.06)
fig.savefig('Fig4.pdf'); fig.savefig('Fig4.png', dpi=300)
print('pairs', len(P), 'matched', len(M), 'mean all %.3f matched %.3f; AUC %.3f / %.3f' % (allv.mean(), mat.mean(), a1, a2))
print('bin means', [round(v, 3) for v in my])
