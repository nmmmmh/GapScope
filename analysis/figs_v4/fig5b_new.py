"""Figure 5b redesigned as a dumbbell chart: coverage under composition holdout (open) and chemical-family holdout (filled)."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gs5 import *
plt.rcParams['svg.fonttype'] = 'path'
D = os.path.dirname(os.path.abspath(__file__)); K = 7.0 / 1080.0
# name, coverage (composition, chemical family), width under chemical-family holdout (eV), colour
M = [('GapScope', 0.790, 0.821, 1.22, TEAL2), ('EasyUQ', 0.824, 0.854, 1.34, '#7A68B0'),
     ('Roost-cal*', 0.763, 0.670, 0.91, SAND), ('Roost-cal', 0.687, 0.543, 0.66, INDIGO), ('Roost', 0.432, 0.380, 0.45, SLATE)]
fig = plt.figure(figsize=(370 * K, 322 * K))
ax = fig.add_axes([0.26, 0.25, 0.53, 0.66])
ax.axvline(0.8, color=CORAL, lw=0.8, ls=(0, (3, 1.5)), zorder=0)
for i, (n, a, b, w, c) in enumerate(M):
    y = len(M) - 1 - i
    ax.annotate('', xy=(b, y), xytext=(a, y), arrowprops=dict(arrowstyle='-|>', color=c, lw=1.3, mutation_scale=7, shrinkA=3, shrinkB=3.2), zorder=2)
    ax.plot(a, y, 'o', ms=4.5, mfc='white', mec=c, mew=1.1, zorder=3); ax.plot(b, y, 'o', ms=4.5, color=c, zorder=3)
    bold = 'bold' if n == 'GapScope' else 'normal'
    ax.text(b + (0.022 if b > a else -0.022), y, '%d%%' % round(100 * b), color=c, fontsize=6.5, va='center', ha='left' if b > a else 'right', fontweight=bold)
    ax.text(-0.03, y, n, transform=ax.get_yaxis_transform(), ha='right', va='center', fontsize=6.8, color=c, fontweight=bold)
    ax.text(1.09, y, '%.2f' % w, transform=ax.get_yaxis_transform(), ha='left', va='center', fontsize=6.5, color=c, fontweight=bold)
ax.text(1.09, len(M) - 0.35, 'Width\n(eV)', transform=ax.get_yaxis_transform(), ha='left', va='bottom', fontsize=6.2, color=NAVY)
ax.text(0.8, len(M) - 0.35, '80%', ha='center', va='bottom', fontsize=6.2, color=CORAL)
ax.set_xlim(0.22, 0.95); ax.set_ylim(-0.6, len(M) - 0.4)
ax.set_xticks([0.4, 0.6, 0.8]); ax.set_xticklabels(['40%', '60%', '80%']); ax.set_yticks([])
ax.spines['left'].set_visible(False); ax.set_xlabel('Coverage of the 80% interval')
for xx, filled, lab in ((0.05, False, 'Composition holdout'), (0.50, True, 'Chemical-family holdout')):
    fig.lines.append(plt.Line2D([xx + 0.02], [0.045], marker='o', ms=4.5, mfc=(NAVY if filled else 'white'), mec=NAVY, mew=1.0, transform=fig.transFigure))
    fig.text(xx + 0.05, 0.045, lab, fontsize=6.2, va='center', color=NAVY)
fig.savefig(os.path.join(D, 'fig5b.svg')); fig.savefig(os.path.join(D, 'fig5b_check.png'), dpi=300); plt.close(fig)
print('ok')
