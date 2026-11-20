# SUPERSEDED by fig5b_new.py (Figure 5b as a dot plot of coverage per model, with the chemical-family width as a column).
"""Figure 5b in the original coverage-against-width style, relabelled so that no text or marker overlaps."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gs5 import *
plt.rcParams['svg.fonttype'] = 'path'
D = os.path.dirname(os.path.abspath(__file__)); K = 7.0 / 1080.0
EUQ = '#7A68B0'
# name, (width, coverage) composition holdout, (width, coverage) chemical-family holdout, colour, label position, alignment
models = [('Roost', (0.42, 0.432), (0.45, 0.380), SLATE, (0.50, 0.380), 'left'),
          ('Roost-cal', (0.86, 0.687), (0.66, 0.543), INDIGO, (0.70, 0.528), 'left'),
          ('Roost-cal*', (1.06, 0.763), (0.91, 0.670), SAND, (0.95, 0.652), 'left'),
          ('EasyUQ', (1.23, 0.824), (1.34, 0.854), EUQ, (1.34, 0.888), 'center'),
          ('This work', (1.06, 0.790), (1.22, 0.821), TEAL2, (1.39, 0.773), 'center')]
fig = plt.figure(figsize=(370 * K, 322 * K))
ax = fig.add_axes([0.19, 0.16, 0.76, 0.78])
ax.axhline(0.8, color=CORAL, lw=0.8, ls=(0, (3, 1.5)), zorder=0)
ax.text(0.33, 0.808, 'Nominal 80%', fontsize=6.2, color=CORAL, va='bottom')
for name, a, b, c, lp, ha in models:
    mk = 's' if name == 'EasyUQ' else 'o'; z = {'This work': 5, 'EasyUQ': 6}.get(name, 3)
    ax.annotate('', xy=b, xytext=a, arrowprops=dict(arrowstyle='-|>', color=c, lw=1.3, mutation_scale=7.5, shrinkA=3, shrinkB=3.2), zorder=z - 1)
    ax.plot(*a, mk, ms=4.6, mfc='white', mec=c, mew=1.1, zorder=z); ax.plot(*b, mk, ms=5.2, color=c, mec=c, mew=0.6, zorder=z)
    ax.text(*lp, '%s %d%%' % (name, round(100 * b[1])), fontsize=6.6, color=c, ha=ha, va='center')
ax.set_xlim(0.3, 1.55); ax.set_ylim(0.34, 0.92)
ax.set_xticks([0.4, 0.8, 1.2]); ax.set_yticks([0.4, 0.6, 0.8]); ax.set_yticklabels(['40%', '60%', '80%'])
ax.set_xlabel('80% interval width (eV)'); ax.set_ylabel('Coverage')
for yy, filled, lab in ((0.455, False, 'Composition holdout'), (0.405, True, 'Chemical-family holdout')):
    ax.plot(0.90, yy, 'o', ms=(5.2 if filled else 4.6), mfc=(NAVY if filled else 'white'), mec=NAVY, mew=1.0)
    ax.text(0.94, yy, lab, fontsize=6.2, va='center', color=NAVY)
fig.savefig(os.path.join(D, 'fig5b.svg')); fig.savefig(os.path.join(D, 'fig5b_check.png'), dpi=300); plt.close(fig)
print('ok')
