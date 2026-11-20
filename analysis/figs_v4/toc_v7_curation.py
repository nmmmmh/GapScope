"""TOC v7, curation variant of toc_v5.py: compositions removed by the 0.1 eV agreement rule drawn open, with their share of the floor.
TOC v4 (3.25 x 1.75 in): computed polymorphs (HSE06) of three compositions -> gap spread D_c -> D_c against the measured floor of 50 compositions."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gs5 import *
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
P = pd.read_csv('/data/czx/GapScope_Revision_20261113/descr/d2_per_composition_energy.csv')
HI, LO = '#E9A23B', '#2A9D8F'
ROWS = [('Ga2O3', 'r_ga2o3', 4.012, 3.538), ('SiC', 'r_sic', 2.920, 2.258), ('MoS2', 'r_mos2', 1.662, 1.662)]
def trim(path, pad=12):
    im = plt.imread(path)[..., :3]; m = (im < 0.97).any(-1); ys, xs = np.where(m)
    im = im[max(ys.min() - pad, 0):ys.max() + pad, max(xs.min() - pad, 0):xs.max() + pad]
    a = np.clip((1 - im.min(-1)) / 0.10, 0, 1)            # white -> transparent, smooth edge
    return np.dstack([im, a])
W, H = 3.25, 1.75
fig = plt.figure(figsize=(W, H))
def ax_in(x, y, w, h): return fig.add_axes([x / W, 1 - (y + h) / H, w / W, h / H])
ov = ax_in(0, 0, W, H); ov.set_xlim(0, W); ov.set_ylim(H, 0); ov.axis('off'); ov.set_zorder(-1)
PANEL = '#F2F6F7'
for x, w, lab in ((0.04, 1.96, 'Computed polymorphs'), (2.07, 1.14, 'Measured floor')):
    ov.add_patch(FancyBboxPatch((x, 0.05), w, 1.65, boxstyle='round,pad=0,rounding_size=0.06', fc=PANEL, ec='none'))
    ov.text(x + w / 2, 0.15, lab, fontsize=6.3, color=NAVY, ha='center', va='center')
X0, K = 1.43, 0.47
for j, (f, img, ghi, glo) in enumerate(ROWS):
    yc = 0.53 + 0.46 * j
    fig.text(0.09 / W, 1 - yc / H, fmt(f), fontsize=6.8, color=NAVY, ha='left', va='center')
    for suf, x in ((('_lo', 0.73),) if f == 'MoS2' else (('_hi', 0.50), ('_lo', 0.96))):
        im = trim(img + suf + '.png'); h_, w_ = im.shape[:2]; s = 0.40 / max(h_, w_ * 0.95)
        bw, bh = w_ * s, h_ * s
        ax = ax_in(x + (0.42 - bw) / 2, yc - bh / 2, bw, bh); ax.imshow(im, interpolation='lanczos'); ax.axis('off')
    D = ghi - glo; x1 = X0 + K * D
    ov.plot([X0, x1], [yc, yc], color=CORAL, lw=1.6, solid_capstyle='butt', zorder=3)
    if D > 0: ov.plot([x1, x1], [yc - 0.055, yc + 0.055], color=HI, lw=1.4, zorder=4)
    ov.plot([X0, X0], [yc - 0.055, yc + 0.055], color=LO, lw=1.4, zorder=5)
    ov.text(max(x1, X0 + 0.06) + 0.04, yc, '%.2f' % D if D > 0 else '0', fontsize=6.0, color=CORAL, ha='left', va='center')
ov.text(X0 + 0.18, 0.33, '$D_c$ (eV)', fontsize=6.4, color=NAVY, ha='center', va='center')
ov.add_patch(FancyArrowPatch((1.975, 0.99), (2.115, 0.99), arrowstyle='-|>', mutation_scale=7, lw=1.0, color=NAVY, zorder=5))
axr = ax_in(2.37, 0.36, 0.79, 1.06); axr.patch.set_alpha(0)
big = (P.floor >= 0.05 - 1e-9).values
rem = (P.rng > 0.1 + 1e-9).values   # removed by the 0.1 eV agreement rule
for m_, c_ in ((~big, BLUE), (big, CORAL)):
    axr.scatter(P.D_elec[m_ & ~rem], P.floor[m_ & ~rem], s=9, color=c_, lw=0, zorder=2)
    axr.scatter(P.D_elec[m_ & rem], P.floor[m_ & rem], s=8, facecolor='white', edgecolor=c_, lw=0.7, zorder=2)
share = P.floor[rem].sum() / P.floor.sum()
print('removed', int(rem.sum()), 'of', len(P), 'share of floor %.4f' % share)
for f, tx, ty in (('Ga2O3', 0.25, 0.47), ('SiC', 0.88, 0.44), ('MoS2', 0.40, 0.025)):
    r = P[P.formula == f].iloc[0]
    axr.scatter([r.D_elec], [r.floor], s=24, facecolor='none', edgecolor=NAVY, lw=0.7, zorder=3)
    axr.annotate(fmt(f), (r.D_elec, r.floor), xytext=(tx, ty), textcoords='axes fraction', fontsize=5.4, color=NAVY, ha='center', va='bottom', arrowprops=dict(arrowstyle='-', lw=0.45, color=NAVY, shrinkA=0.5, shrinkB=2.2))
axr.set_xscale('symlog', linthresh=0.3, linscale=0.6); axr.set_xlim(-0.06, 8); axr.set_ylim(-0.03, 0.68)
axr.set_xticks([0, 1, 3]); axr.set_xticklabels(['0', '1', '3']); axr.set_yticks([0, 0.3, 0.6]); axr.xaxis.set_minor_locator(plt.NullLocator())
axr.tick_params(labelsize=5.8, length=2, pad=1.2, colors=NAVY)
for sp in ('left', 'bottom'): axr.spines[sp].set_linewidth(0.7); axr.spines[sp].set_color(NAVY)
axr.set_xlabel('$D_c$ (eV)', fontsize=6.4, labelpad=1, color=NAVY); axr.set_ylabel('Floor (eV)', fontsize=6.4, labelpad=1, color=NAVY)
from sklearn.metrics import roc_auc_score as _auc
axr.text(0.06, 0.98, 'AUC %.2f' % _auc(big, P.D_elec), transform=axr.transAxes, fontsize=5.9, color=NAVY, va='top')
axr.text(0.06, 0.89, 'n = %d' % len(P), transform=axr.transAxes, fontsize=5.6, color=SLATE, va='top')
ov.scatter([2.30], [0.262], s=8, facecolor='white', edgecolor=CORAL, lw=0.7, zorder=6)
ov.text(2.34, 0.262, 'curated out, %d%% of floor' % round(100 * share), fontsize=5.0, color=NAVY, ha='left', va='center')
fig.savefig('TOC_v7.pdf', dpi=600); fig.savefig('TOC_v7.png', dpi=600); fig.savefig('TOC_v7.tif', dpi=600, pil_kwargs={'compression': 'tiff_lzw'})
print('ok')
