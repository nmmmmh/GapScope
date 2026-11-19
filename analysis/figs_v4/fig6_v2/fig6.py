"""Figure 6: band-gap windows. (a) HSE06 screen: polymorphic compositions whose lowest-energy polymorph lies in the window,
and how many of them have an accessible polymorph outside it. (b) Promised against delivered hit rate, chemical-family holdout."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gs5 import *
import pandas as pd
plt.rcParams['svg.fonttype'] = 'path'
D = os.path.dirname(os.path.abspath(__file__)); W = '/data/czx/GapScope_Revision_20261113/window/par2/'
EUQ = '#7A68B0'
H = pd.read_csv(W + 'hse_screen_v2.csv').set_index('window')   # resolved ground state, 0.1 eV margin (refcheck/c8_resource.py)
P = pd.read_csv(W + 'window_partitions.csv'); P = P[P.holdout == 'chem']
Q = pd.read_csv(W + 'window_partitions_rcs.csv').rename(columns={'promised': 'promised_0.5', 'delivered': 'delivered_0.5'})   # Roost-cal* (refcheck/c16_rcs_sum.py)
P = pd.concat([P, Q[Q.holdout == 'chem']], ignore_index=True)
RCS = '#8FA3C7'
fig = plt.figure(figsize=(7.0, 2.45))
# (a)
ax = fig.add_axes([0.115, 0.22, 0.30, 0.56])
rows = [('VIS', 'Visible'), ('PV', 'Photovoltaic')]
# share with an accessible polymorph outside at margins 0 / 0.1 / 0.2 eV, resolved lowest-energy polymorph (refcheck/c7_out.txt)
SWEEP = {'PV': (14 / 28, 7 / 18, 4 / 11), 'VIS': (34 / 155, 20 / 131, 13 / 108)}
for i, (w, lab) in enumerate(rows):
    n, f = int(H.loc[w, 'n_gs_in']), int(H.loc[w, 'n_gs_in_with_outside']); s = f / n
    ax.barh(i, s, height=0.46, color=LCORAL, edgecolor=CORAL, lw=0.6, zorder=2)
    ax.barh(i, 1 - s, left=s, height=0.46, color=LTEAL, edgecolor=TEAL, lw=0.6, zorder=2)
    lo, hi = min(SWEEP[w]), max(SWEEP[w])
    ax.plot([lo, hi], [i, i], color=NAVY, lw=0.9, zorder=3, solid_capstyle='butt')
    for x in (lo, hi): ax.plot([x, x], [i - 0.09, i + 0.09], color=NAVY, lw=0.9, zorder=3)
    ax.text(1.03, i, '%d/%d' % (f, n), va='center', ha='left', fontsize=6.6, color=NAVY)
ax.set_yticks([0, 1]); ax.set_yticklabels([r[1] for r in rows])
ax.set_xlim(0, 1); ax.set_ylim(-0.5, 1.5)
ax.set_xticks([0, 0.25, 0.5, 0.75, 1]); ax.set_xticklabels(['0%', '25%', '50%', '75%', '100%'])
ax.set_xlabel('Share of candidates')
for x0, c, ec, lab in ((0.0, LCORAL, CORAL, 'A polymorph outside'), (0.52, LTEAL, TEAL, 'All polymorphs inside')):
    ax.add_patch(plt.Rectangle((x0, 1.13), 0.05, 0.075, transform=ax.transAxes, color=c, ec=ec, lw=0.6, clip_on=False))
    ax.text(x0 + 0.07, 1.17, lab, transform=ax.transAxes, fontsize=6.2, va='center', color=NAVY)
fig.text(0.008, 0.93, '(a)', fontsize=8, fontweight='bold')
# (b)
bx = fig.add_axes([0.585, 0.20, 0.27, 0.70])
bx.plot([0.45, 0.95], [0.45, 0.95], color=GREYP, lw=0.8, ls=(0, (3, 1.5)), zorder=0)
models = [('This work', 'GapScope', TEAL2), ('Roost-cal', 'ROOST_CAL', INDIGO), ('Roost-cal*', 'ROOST_CAL_CV', RCS), ('EasyUQ', 'EASYUQ_KS', EUQ)]
lab_off = {('This work', 'PV'): (0.012, 0.03), ('Roost-cal', 'PV'): (0.015, -0.035), ('EasyUQ', 'PV'): (-0.012, 0.035), ('Roost-cal*', 'PV'): (0.015, -0.035),
           ('This work', 'VIS'): (-0.012, 0.035), ('Roost-cal', 'VIS'): (0.015, -0.035), ('EasyUQ', 'VIS'): (-0.012, 0.035), ('Roost-cal*', 'VIS'): (0.02, 0.028)}
for name, key, c in models:
    for w, mk in (('PV', 'o'), ('VIS', 's')):
        t = P[(P.model == key) & (P.window == w)]
        x, y = t['promised_0.5'], t['delivered_0.5']
        bx.errorbar(x.mean(), y.mean(), xerr=x.std(), yerr=y.std(), fmt=mk, ms=5.0, color=c, mec=c, mfc=c, elinewidth=0.7, capsize=1.5, zorder=4)
        dx, dy = lab_off[(name, w)]
        if w == 'VIS': bx.text(x.mean() + dx, y.mean() + dy, name, fontsize=6.2, color=c, ha='left' if dx > 0 else ('right' if dx < 0 else 'center'), va='center')
bx.set_xlim(0.45, 0.95); bx.set_ylim(0.45, 0.95)
bx.set_xticks([0.5, 0.6, 0.7, 0.8, 0.9]); bx.set_yticks([0.5, 0.6, 0.7, 0.8, 0.9])
bx.set_xticklabels(['50%', '60%', '70%', '80%', '90%']); bx.set_yticklabels(['50%', '60%', '70%', '80%', '90%'])
bx.set_xlabel('Promised hit rate'); bx.set_ylabel('Delivered hit rate')
for yy, mk, lab in ((0.53, 'o', 'Photovoltaic'), (0.49, 's', 'Visible')):
    bx.plot(0.70, yy, mk, ms=4.6, color=NAVY); bx.text(0.72, yy, lab, fontsize=6.2, va='center', color=NAVY)
fig.text(0.49, 0.93, '(b)', fontsize=8, fontweight='bold')
for e in ('pdf', 'svg', 'png'): fig.savefig(os.path.join(D, 'Fig6.' + e), dpi=600 if e == 'png' else None)
print('ok')
