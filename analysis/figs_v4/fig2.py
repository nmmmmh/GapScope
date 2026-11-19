"""Figure 2 (v8): what the floor is (Ga2O3), where it sits and what curation removes, how much of model error it is."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gs5 import *
import numpy as np, pandas as pd

R0 = '/data/czx/GapScope_Revision_20261113/'
S = pd.read_csv(R0 + 'suppdata/Supplementary_Data_1_GapScope_experimental_bandgaps.csv')
def norm_ph(s):
    s = str(s).strip().lower() if isinstance(s, str) else ''
    for g in ('α', 'β', 'γ', 'δ', 'κ', 'ε'):
        if s in (g, g + '-ga2o3'): return g
    return s
S['ph'] = S.phase_label.apply(norm_ph)
nph = S[S.ph != ''].groupby('composition_key').ph.nunique()
a = pd.read_csv(R0 + 'cv/predictions_summary_v4a.csv'); t = pd.read_csv(R0 + 'cv/predictions_summary_v4t1.csv')
MODELS = [('ROOST', a, 'Roost', '#DCE8EE'), ('ROOST_CAL', a, 'Roost-cal', '#B4CDDB'), ('EASYUQ_KS', a, 'EasyUQ', '#7FA7BF'),
          ('GS_SELCAL_PRE_NZ_SC', t, 'This work', TEAL2)]
PCOL = {'β': PH[0], 'γ': PH[1], 'κ': PH[2], 'α': PH[3]}

def wmedian(y, w):
    o = np.argsort(y); y, w = y[o], w[o]; c = np.cumsum(w) / w.sum()
    return y[np.searchsorted(c, 0.5)]
per = {}
for m, d, lab, _ in MODELS:
    g = d[(d.model == m) & (d.tier == 'A')]; rows = []
    for k, gg in g.groupby('key'):
        y, w = gg.y.values, gg.w.values
        rows.append((k, np.sum(w * np.abs(y - gg['median'].values)) / w.sum(), np.sum(w * np.abs(y - wmedian(y, w))) / w.sum(),
                     int(nph.get(k, 0) >= 2), y.max() - y.min()))
    per[m] = pd.DataFrame(rows, columns=['key', 'mae', 'floor', 'multi', 'range']).set_index('key')
F = per['ROOST']

W, H = 7.0, 2.22
fig = plt.figure(figsize=(W, H))
def ax_in(x, y, w, h): return fig.add_axes([x / W, 1 - (y + h) / H, w / W, h / H])
def letter(s, x, y): fig.text(x / W, 1 - y / H, '(%s)' % s, fontsize=8, fontweight='bold', ha='left', va='top')
T0, HH = 0.24, 1.66

# ======================= (a) the floor of one composition: Ga2O3 reports and their distance to the median
axa = ax_in(0.30, T0, 1.65, HH)
G = S[S.formula == 'Ga2O3'].sort_values('bandgap_ev').reset_index(drop=True)
gkey = G.composition_key.iloc[0]; med = F.loc[gkey] if gkey in F.index else None
gw = a[(a.model == 'ROOST') & (a.tier == 'A') & (a.key == gkey)]
m0 = wmedian(gw.y.values, gw.w.values)
axa.axvline(m0, color=NAVY, lw=0.9, zorder=1)
for i, r in G.iterrows():
    c = PCOL.get(r.ph, GREYP)
    axa.plot([m0, r.bandgap_ev], [i, i], color=c, lw=1.2, alpha=0.75, solid_capstyle='round', zorder=2)
    axa.plot(r.bandgap_ev, i, 'o', ms=3.6, color=c, mec='white', mew=0.4, zorder=3)
axa.set_ylim(-1, len(G)); axa.set_yticks([]); axa.spines['left'].set_visible(False)
axa.set_xlim(4.25, 5.45); axa.set_xticks([4.4, 4.8, 5.2]); axa.set_xlabel('$E_g$ (eV)')
axa.text(m0 + 0.02, -0.6, 'median', fontsize=6.5, va='bottom', color=NAVY)
axa.text(4.27, len(G) - 0.2, fmt('Ga2O3'), fontsize=7.5, va='top', fontweight='bold')
axa.text(4.27, len(G) - 2.2, 'floor %.2f eV' % F.loc[gkey, 'floor'], fontsize=6.5, va='top')
for j, p in enumerate(['β', 'γ', 'κ', 'α']):
    axa.text(5.43, 2.0 + 1.8 * j, p, fontsize=7, color=PCOL[p], ha='right', va='center', fontweight='bold')
letter('a', 0.02, 0.06)

# ======================= (b) floor of every composition, widest first, with cumulative share and the 0.1 eV rule
axb = ax_in(2.50, T0, 2.20, HH)
Fs = F.sort_values('range', ascending=False); x = np.arange(1, len(Fs) + 1)
k_rule = int((Fs.range.round(6) > 0.1 + 1e-9).sum())
axb.axvspan(0.3, k_rule + 0.5, color=LCORAL, alpha=0.55, lw=0, zorder=0)
axb.bar(x, Fs.floor, width=0.8, color=[CORAL if m else GREYP for m in Fs.multi], lw=0, zorder=2)
axb.set_xlim(0, len(Fs) + 1); axb.set_ylim(0, 0.66); axb.set_yticks([0, 0.2, 0.4, 0.6]); axb.set_xticks([1, 20, 40, 60, 85])
axb.set_ylabel('Floor (eV)'); axb.set_xlabel('Composition, widest first')
ax2 = axb.twinx(); ax2.spines['right'].set_visible(True); ax2.spines['top'].set_visible(False)
cum = 100 * np.cumsum(Fs.floor.values) / Fs.floor.sum(); k = int((Fs.range.round(6) > 0.1 + 1e-9).sum())
ax2.plot(x, cum, color=NAVY, lw=1.0, zorder=3); ax2.set_ylim(0, 110); ax2.set_yticks([0, 50, 100])
ax2.set_ylabel('Cumulative floor (%)', rotation=270, labelpad=9)
ax2.plot([k + 0.5, k + 0.5], [0, 108], color=AMBER, lw=0.8, ls=(0, (3, 1.5)))
ax2.plot(k, cum[k - 1], 'o', ms=3, color=NAVY, zorder=4)
ax2.text(k + 2, cum[k - 1] - 5, '%.0f%%' % cum[k - 1], fontsize=6.5, va='top')
ax2.text(k + 2, 30, '0.1 eV rule removes %d' % k, fontsize=6.5, color=AMBER, va='center')
axb.text(60, 0.56, '$\\geq$2 phases (%d)' % int(Fs.multi.sum()), fontsize=6.5, color=CORAL, va='center')
axb.text(60, 0.50, '1 phase (%d)' % int((~Fs.multi.astype(bool)).sum()), fontsize=6.5, color=SLATE, va='center')
letter('b', 2.05, 0.06)

# ======================= (c) error = floor + rest, four models, by number of phases
axc = ax_in(5.55, T0, 1.40, HH)
bw = 0.16
for gi, gm in enumerate((1, 0)):
    shares, tops = [], []
    for mi, (m, _, lab, col) in enumerate(MODELS):
        P = per[m][per[m].multi == gm]; mae, fl = P.mae.mean(), P.floor.mean()
        xx = gi * 0.95 + (mi - 1.5) * (bw + 0.025)
        axc.bar(xx, fl, width=bw, color=CORAL, lw=0, zorder=2); axc.bar(xx, mae - fl, bottom=fl, width=bw, color=col, lw=0, zorder=2)
        shares.append(round(100 * fl / mae)); tops.append(mae)
    s = '%d%%' % shares[0] if min(shares) == max(shares) else '%d–%d%%' % (min(shares), max(shares))
    axc.text(gi * 0.95, max(tops) + 0.02, 'Floor ' + s, fontsize=6.3, ha='center', va='bottom', color=CORAL, fontweight='bold')
P0 = per[MODELS[0][0]]
axc.set_xticks([0, 0.95]); axc.set_xticklabels(['$\\geq$2 phases (%d)' % (P0.multi == 1).sum(), '1 phase (%d)' % (P0.multi == 0).sum()], fontsize=6.6); axc.tick_params(axis='x', length=0, pad=3)
axc.set_xlim(-0.45, 1.45); axc.set_ylim(0, 0.72); axc.set_yticks([0, 0.2, 0.4, 0.6]); axc.set_ylabel('MAE (eV)')
for mi, (m, _, lab, col) in enumerate(MODELS + [(None, None, 'Floor', CORAL)]):
    yy = 0.68 - 0.062 * mi
    axc.bar([0.80], [0.038], bottom=yy - 0.019, width=0.08, color=col, lw=0); axc.text(0.87, yy, lab, fontsize=6, va='center')
letter('c', 4.95, 0.06)
fig.savefig('Fig2.pdf'); fig.savefig('Fig2.png', dpi=300)
print('Ga2O3 median %.3f floor %.3f; removed %d -> %.1f%%' % (m0, F.loc[gkey, 'floor'], k, cum[k - 1]))
