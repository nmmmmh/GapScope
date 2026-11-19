"""MAE on the compositions the 0.1 eV rule keeps (47) vs removes (38), original fold assignment, with and without the space group."""
import numpy as np, pandas as pd
R = "/data/czx/GapScope_Revision_20261113/"
for v in ("comp", "chem"):
    d = pd.read_csv(R + f"cv/predictions_summary_nosg_{v}.csv"); d = d[d.tier == "A"]
    out = {}
    for m in ("GS_SELCAL_PRE_NZ_SC", "GS_SELCAL_PRE_NZ_NOSG_SC", "ROOST", "ROOST_CAL", "ROOST_CAL_CV"):
        g = d[d.model == m]
        if len(g) == 0: continue
        pc = g.groupby("key").apply(lambda x: pd.Series(dict(mae=np.sum(x.w * np.abs(x.y - x["median"])) / x.w.sum(), rm=round(x.y.max() - x.y.min(), 6) > 0.1 + 1e-9)))
        out[m] = (pc.mae[pc.rm.astype(bool)].mean(), pc.mae[~pc.rm.astype(bool)].mean(), pc.mae.mean(), int(pc.rm.sum()))
    for m, (a, b, c, n) in out.items(): print(f"{v} {m:26s} removed({n}) {a:.3f} kept {b:.3f} all {c:.3f}")
