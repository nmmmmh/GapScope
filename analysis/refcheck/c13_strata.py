# SUPERSEDED (2026-11-19): reads the earlier D_c over all polymorphs, descr/d2_per_composition.csv.
# The paper uses the energy-only D_c (refcheck/c14_energy.py, descr/d2_per_composition_energy.csv written by refcheck/c15_write.py).
# Kept for comparison only; no number, table or figure of the paper comes from this script.
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from pymatgen.core import Composition
R = '/data/czx/GapScope_Revision_20261113/'
D = pd.read_csv(R + 'descr/d2_per_composition.csv')
P = pd.read_csv(R + 'suppdata/Supplementary_Data_1_GapScope_experimental_bandgaps.csv')
n = P.groupby('formula').size(); D['nrep'] = D.formula.map(n)
miss = D.nrep.isna().sum()
y = D.floor >= 0.05
print('missing nrep', miss, '| all n %d AUC %.3f' % (len(D), roc_auc_score(y, D.D_elec)))
for lab, s in (('>=2 reports', D.nrep >= 2), ('>=3 reports', D.nrep >= 3), ('2-3 reports', D.nrep.between(2, 3)), ('>=4 reports', D.nrep >= 4)):
    d = D[s]; yy = d.floor >= 0.05
    print('%-12s n %2d pos %2d AUC %s' % (lab, len(d), yy.sum(), ('%.3f' % roc_auc_score(yy, d.D_elec)) if 0 < yy.sum() < len(d) else 'na'))
print('rho Dc-nrep %.2f' % D[['D_elec', 'nrep']].corr('spearman').iloc[0, 1])
