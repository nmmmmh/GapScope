"""Counts for exact wording (2026-11-19): report-weighted floor share in the 38 removed compositions; Ga2O3 beta reports
below the weighted median; share of within-composition polymorph pairs on the structure map closer than random pairs."""
import numpy as np, pandas as pd, itertools
R = "/data/czx/GapScope_Revision_20261113/"
d = pd.read_csv(R + "cv/predictions_summary_v4t1.csv"); d = d[d.tier == "A"]; d = d[d.model == d.model.iloc[0]]
def wmed(v, w):
    o = np.argsort(v); v, w = v[o], w[o]; c = np.cumsum(w) / w.sum(); return v[np.searchsorted(c, 0.5)]
tot = rm_tot = 0.0; n = 0
for k, g in d.groupby("key"):
    y = g.y.values; m = np.median(y); s = np.abs(y - m).sum(); tot += s; n += len(y)
    if round(y.max() - y.min(), 6) > 0.1 + 1e-9: rm_tot += s
print("report-weighted floor %.3f over %d reports; share in removed %.3f" % (tot / n, n, rm_tot / tot))
D1 = pd.read_csv(R + "suppdata/Supplementary_Data_1_GapScope_experimental_bandgaps.csv", encoding="utf-8-sig")
G = D1[D1.formula == "Ga2O3"].copy()
ns = G.source_group_id.nunique()
G["w"] = [1.0 / ns / G[G.source_group_id == s].specimen_group_id.nunique() / ((G.source_group_id == s) & (G.specimen_group_id == p)).sum() for s, p in zip(G.source_group_id, G.specimen_group_id)]
gk = G.composition_key.iloc[0]; gp = d[d.key == gk]
print("Ga2O3 n %d; weights match predictions file: %s" % (len(G), np.allclose(np.sort(G.w.values / G.w.sum()), np.sort(gp.w.values / gp.w.sum()))))
m = wmed(G.bandgap_ev.values, G.w.values); ph = G.phase_label.fillna("").astype(str).str.strip().str.lower()
print("weighted median %.3f; phases %s" % (m, dict(ph.value_counts())))
for lab in sorted(ph.unique()):
    v = G.bandgap_ev.values[ph.values == lab]
    print("  %-12s n %2d below %2d at %2d above %2d" % (lab, len(v), (v < m - 1e-9).sum(), (np.abs(v - m) < 1e-9).sum(), (v > m + 1e-9).sum()))
S = pd.read_csv(R + "maps2d/struct_map.csv")
dw = []
for k, g in S.groupby("composition_key"):
    if len(g) < 2: continue
    xy = g[["x", "y"]].values
    for i, j in itertools.combinations(range(len(xy)), 2): dw.append(np.hypot(*(xy[i] - xy[j])))
dw = np.array(dw); rng = np.random.default_rng(0); xy = S[["x", "y"]].values
a, b = rng.integers(0, len(xy), 3000), rng.integers(0, len(xy), 3000); dr = np.hypot(*(xy[a] - xy[b]).T)
print("within-composition pairs %d, median %.1f; random median %.1f; share of pairs closer than random median %.3f; closer than 10 %.3f" % (len(dw), np.median(dw), np.median(dr), (dw < np.median(dr)).mean(), (dw < 10).mean()))
