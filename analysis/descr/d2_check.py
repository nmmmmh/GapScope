# SUPERSEDED (2026-11-19): reads the earlier D_c over all polymorphs, descr/d2_per_composition.csv.
# The paper uses the energy-only D_c (refcheck/c14_energy.py, descr/d2_per_composition_energy.csv written by refcheck/c15_write.py).
# Kept for comparison only; no number, table or figure of the paper comes from this script.
"""Window fixed at 0.1 eV/atom (the hull window of the pretraining pool). Bootstrap intervals, Mann-Whitney P, a rank-mean combination, examples."""
import numpy as np, pandas as pd
exec(open('/data/czx/GapScope_Revision_20261113/descr/d1_desc.py').read().split('pos_r, pos_f')[0])
from scipy.stats import rankdata
d = desc(0.10); Ds, De = d[:, 0], d[:, 1]
Dc = (rankdata(Ds) + rankdata(De)) / 2          # rank mean, no weights fitted
pos_r, pos_f = RNG > 0.2, F > 0.05
rng = np.random.default_rng(20261115)
def ci(x, lab):
    b = []
    for _ in range(4000):
        i = rng.integers(0, len(x), len(x))
        if lab[i].min() == lab[i].max(): continue
        b.append(roc_auc_score(lab[i], x[i]))
    return np.percentile(b, [2.5, 97.5])
for name, x in (('D_struct', Ds), ('D_elec', De), ('rank mean', Dc)):
    for tl, lab in (('floor>0.05', pos_f), ('range>0.2', pos_r)):
        lo, hi = ci(x, lab); P = mannwhitneyu(x[lab], x[~lab]).pvalue
        print('%-9s %-10s AUC %.3f [%.2f, %.2f] P=%.2g' % (name, tl, roc_auc_score(lab, x), lo, hi, P))
    print('%-9s rho floor %.3f (P=%.2g)' % (name, spearmanr(x, F).statistic, spearmanr(x, F).pvalue))
fm = pd.read_csv(R + 'suppdata/Supplementary_Data_1_GapScope_experimental_bandgaps.csv', encoding='utf-8-sig').drop_duplicates('composition_key').set_index('composition_key').formula
X = pd.DataFrame(dict(formula=[fm.get(k, k) for k in keys], floor=F, rng=RNG, D_struct=Ds, D_elec=De, n_acc=d[:, 2]))
X.to_csv(R + 'descr/d2_per_composition.csv', index=False)
print(X[X.formula.isin(['Ga2O3', 'SiC', 'CH5N2PbI3', 'FAPbI3', 'In2Se3', 'TiO2', 'ZnS', 'CdS', 'CrSBr', 'MoO3', 'BiVO4'])].round(3).to_string(index=False))
print('median floor by D_elec tercile:', X.groupby(pd.qcut(X.D_elec.rank(method="first"), 3, labels=["low", "mid", "high"])).floor.median().round(3).to_dict())
