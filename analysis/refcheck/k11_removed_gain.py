"""Relative gain on the 38 compositions the 0.1 eV rule removes, composition holdout, five fold assignments (c9_rank.csv; orig counted once)."""
import pandas as pd, numpy as np
T = pd.read_csv("c9_rank.csv")
T = T[(T.holdout == "comp") & (T.subset == "removed") & (T["assign"] != "orig")]
F = 0.169  # mean floor of the 38 removed compositions (b5_removed.txt)
others = [c for c in ("Roost", "Roost-cal", "EasyUQ") if c in T]
T["best_other"] = T[others].min(axis=1)
tw, ro, bo = T["This work"].mean(), T.Roost.mean(), T.best_other.mean()
print("assignments", len(T), "| this work %.3f Roost %.3f best other %.3f" % (tw, ro, bo))
print("lower than Roost by %.1f%%, than best other by %.1f%%" % (100 * (1 - tw / ro), 100 * (1 - tw / bo)))
print("above floor: this work %.3f Roost %.3f -> %.1f%% lower; best other %.3f -> %.1f%%" % (tw - F, ro - F, 100 * (1 - (tw - F) / (ro - F)), bo - F, 100 * (1 - (tw - F) / (bo - F))))
print("per assignment lower than Roost (%%):", (100 * (1 - T["This work"] / T.Roost)).round(1).tolist())
