"""Overlap of compositions removed by the 0.1 eV agreement rule with multi-phase compositions; floor shares; bound on kept floor."""
import numpy as np, pandas as pd
R = "/data/czx/GapScope_Revision_20261113/"
d = pd.read_csv(R + "cv/predictions_summary_v4t1.csv"); d = d[(d.tier == "A")]; d = d[d.model == d.model.iloc[0]]
D1 = pd.read_csv(R + "suppdata/Supplementary_Data_1_GapScope_experimental_bandgaps.csv", encoding="utf-8-sig")
nph = D1.assign(p=D1.phase_label.fillna("").astype(str).str.strip().str.lower()).query("p != \"\"").groupby("composition_key").p.nunique()
def wmed(v, w):
    o = np.argsort(v); v, w = v[o], w[o]; c = np.cumsum(w) / w.sum(); return v[np.searchsorted(c, 0.5)]
rows = []
for k, g in d.groupby("key"):
    y, w = g.y.values, g.w.values
    rows.append((k, np.sum(w * np.abs(y - wmed(y, w))) / w.sum(), round(y.max() - y.min(), 6) > 0.1 + 1e-9, nph.get(k, 1) >= 2, len(y)))
T = pd.DataFrame(rows, columns=["k", "F", "rm", "multi", "n"])
print("n", len(T), "removed", T.rm.sum(), "multi", T.multi.sum(), "removed&multi", (T.rm & T.multi).sum(), "removed&single", (T.rm & ~T.multi).sum(), "kept&multi", (~T.rm & T.multi).sum())
print("floor share removed %.3f | multi %.3f | removed&multi %.3f" % (T.F[T.rm].sum() / T.F.sum(), T.F[T.multi].sum() / T.F.sum(), T.F[T.rm & T.multi].sum() / T.F.sum()))
print("kept floor bound: 47 x 0.05 = %.2f vs total %.2f -> max share %.3f; actual kept share %.3f" % ((~T.rm).sum() * 0.05, T.F.sum(), (~T.rm).sum() * 0.05 / (T.F.sum()), T.F[~T.rm].sum() / T.F.sum()))
print("kept multi floor mean %.3f, removed multi floor mean %.3f" % (T.F[~T.rm & T.multi].mean(), T.F[T.rm & T.multi].mean()))
