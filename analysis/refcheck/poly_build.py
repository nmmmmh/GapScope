"""Polymorph table for the polymorph-anchored mixture head (scheme 2): up to KP=4 lowest-energy MP polymorphs per composition
(raw MP HSE file, within 0.1 eV/atom of the hull): HSE-level gap g (eV), energy above the composition's lowest dE (eV/atom), mask."""
import numpy as np, pandas as pd, torch, warnings; warnings.filterwarnings('ignore')
from pymatgen.core import Composition
KP = 4
H = pd.read_csv('/home/czx/2024/spinels/data/mp_hse.csv')
H = H[(H.energy_above_hull <= 0.10) & H.target.notna() & H.spacegroup.between(1, 230)].sort_values('energy_above_hull', kind='mergesort').drop_duplicates(['formula', 'spacegroup'])
def pkey(f):
    c = Composition(f).fractional_composition; return '|'.join(f'{el}:{c[el]:.12f}' for el in sorted(c.as_dict()))
H['ck'] = [pkey(f) for f in H.formula]
K, G, E, M = [], [], [], []
for k, g in H.groupby('ck'):
    g = g.sort_values('energy_above_hull').head(KP); n = len(g)
    gg = np.zeros(KP); ee = np.zeros(KP); mm = np.zeros(KP)
    gg[:n] = g.target.values; ee[:n] = g.energy_above_hull.values - g.energy_above_hull.min(); mm[:n] = 1
    K.append(k); G.append(gg); E.append(ee); M.append(mm)
torch.save(dict(keys=K, g=torch.tensor(np.array(G), dtype=torch.float32), de=torch.tensor(np.array(E), dtype=torch.float32), m=torch.tensor(np.array(M), dtype=torch.float32), KP=KP),
           '/data/czx/GapScope_Revision_20261113/probe/poly_table_mp.pt')
print('compositions', len(K), 'multi-polymorph', int((np.array(M).sum(1) >= 2).sum()))
