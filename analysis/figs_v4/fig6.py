"""Figure 6: band-gap windows. (a) HSE06 screen: polymorphic compositions whose lowest-energy polymorph lies in the window,
and how many of them have an accessible polymorph outside it. (b) Promised against delivered hit rate, chemical-family holdout."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gs5 import *
import pandas as pd
plt.rcParams['svg.fonttype'] = 'path'
D = os.path.dirname(os.path.abspath(__file__)); W = '/data/czx/GapScope_Revision_20261113/window/par2/'
EUQ = '#7A68B0'
H = pd.read_csv(W + 'hse_vs_exp.csv'); H = H[(H.part == 'A_screen') & (H.margin == 0.0)].set_index('window')
P = pd.read_csv(W + 'window_partitions.csv'); P = P[P.holdout == 'chem']
fig = plt.figure(figsize=(7.0, 2.45))
# (a)
ax = fig.add_axes([0.085, 0.22, 0.33, 0.62])
rows = [('VIS', 'Visible'), ('PV', 'Photovoltaic')]
for i, (w, lab) in enumerate(rows):
    n, f = int(H.loc[w, 'n_gs_in']), int(H.loc[w, 'n_gs_in_with_outside'])
    ax.barh(i, n - f, height=0.5, color=LTEAL, edgecolor=TEAL, lw=0.6)
    ax.barh(i, f, left=n - f, height=0.5, color=LCORAL, edgecolor=CORAL, lw=0.6)
    ax.text(n + 3, i, '%d/%d' % (f, n), va='center', fontsize=6.6, color=CORAL)
ax.set_yticks([0, 1]); ax.set_yticklabels([r[1] for r in rows])
ax.set_xlim(0, 215); ax.set_ylim(-0.6, 1.9); ax.set_xticks([0, 50, 100, 150, 200])
ax.set_xlabel('Compositions')
ax.barh(1.62, 14, left=2, height=0.16, color=LTEAL, edgecolor=TEAL, lw=0.6); ax.text(19, 1.62, 'All polymorphs inside', fontsize=6.2, va='center', color=NAVY)
ax.barh(1.36, 14, left=2, height=0.16, color=LCORAL, edgecolor=CORAL, lw=0.6); ax.text(19, 1.36, 'A polymorph outside', fontsize=6.2, va='center', color=NAVY)
fig.text(0.008, 0.93, '(a)', fontsize=8, fontweight='bold')
# (b)
bx = fig.add_axes([0.585, 0.20, 0.27, 0.70])
bx.plot([0.45, 0.95], [0.45, 0.95], color=GREYP, lw=0.8, ls=(0, (3, 1.5)), zorder=0)
models = [('This work', 'GapScope', TEAL2), ('Roost-cal', 'ROOST_CAL', INDIGO), ('EasyUQ', 'EASYUQ_KS', EUQ)]
lab_off = {('This work', 'PV'): (0.012, 0.03), ('Roost-cal', 'PV'): (0.015, -0.035), ('EasyUQ', 'PV'): (-0.012, 0.035),
           ('This work', 'VIS'): (-0.012, 0.035), ('Roost-cal', 'VIS'): (0.015, -0.035), ('EasyUQ', 'VIS'): (-0.012, 0.035)}
for name, key, c in models:
    for w, mk in (('PV', 'o'), ('VIS', 's')):
        t = P[(P.model == key) & (P.window == w)]
        x, y = t['promised_0.5'], t['delivered_0.5']
        bx.errorbar(x.mean(), y.mean(), xerr=x.std(), yerr=y.std(), fmt=mk, ms=5.0, color=c, mec=c, mfc=c, elinewidth=0.7, capsize=1.5, zorder=4)
        dx, dy = lab_off[(name, w)]
        if w == 'VIS': bx.text(x.mean() + dx, y.mean() + dy, name, fontsize=6.2, color=c, ha='left' if dx > 0 else 'right', va='center')
bx.set_xlim(0.45, 0.95); bx.set_ylim(0.45, 0.95)
bx.set_xticks([0.5, 0.6, 0.7, 0.8, 0.9]); bx.set_yticks([0.5, 0.6, 0.7, 0.8, 0.9])
bx.set_xticklabels(['50%', '60%', '70%', '80%', '90%']); bx.set_yticklabels(['50%', '60%', '70%', '80%', '90%'])
bx.set_xlabel('Promised hit rate'); bx.set_ylabel('Delivered hit rate')
for yy, mk, lab in ((0.53, 'o', 'Photovoltaic'), (0.49, 's', 'Visible')):
    bx.plot(0.70, yy, mk, ms=4.6, color=NAVY); bx.text(0.72, yy, lab, fontsize=6.2, va='center', color=NAVY)
fig.text(0.49, 0.93, '(b)', fontsize=8, fontweight='bold')
for e in ('pdf', 'svg', 'png'): fig.savefig(os.path.join(D, 'Fig6.' + e), dpi=600 if e == 'png' else None)
print('ok')
