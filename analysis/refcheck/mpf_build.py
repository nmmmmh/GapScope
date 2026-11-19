"""MP polymorph-ensemble token from the RAW MP HSE file (traced compositions are NOT excluded here; the pretraining pool excludes them) for gsft4m (same 5-feature layout as hf_table.pt): Boltzmann(300 K) mean HSE-level gap,
min, max, spread over the pool polymorphs of a composition (all within 0.1 eV/atom of the hull), log(1+n). Computed data
only, no measured gaps; standardized over all pool compositions."""
import numpy as np, pandas as pd, torch, warnings; warnings.filterwarnings('ignore')
from pymatgen.core import Composition
H = pd.read_csv('/home/czx/2024/spinels/data/mp_hse.csv'); H = H[(H.energy_above_hull <= 0.10) & H.target.notna() & H.spacegroup.between(1, 230)].sort_values('energy_above_hull', kind='mergesort').drop_duplicates(['formula', 'spacegroup'])
def pkey(f):
    c = Composition(f).fractional_composition; return '|'.join(f'{el}:{c[el]:.12f}' for el in sorted(c.as_dict()))
H['ck'] = [pkey(f) for f in H.formula]
kB = 8.617e-5; rows = {}
for k, g in H.groupby('ck'):
    e = g.energy_above_hull.values - g.energy_above_hull.min(); t = g.target.values; w = np.exp(-e / (kB * 300))
    rows[k] = [np.sum(w * t) / w.sum(), t.min(), t.max(), t.max() - t.min(), np.log1p(len(t))]
K = list(rows); X = np.array([rows[k] for k in K]); mu, sd = X.mean(0), X.std(0) + 1e-9
torch.save(dict(keys=K, feat=torch.tensor((X - mu) / sd, dtype=torch.float32), mu=mu, sd=sd, names=['boltz_mean', 'min', 'max', 'spread', 'log1p_n']),
           '/data/czx/GapScope_Revision_20261113/probe/hf_table_mp.pt')
import sys; sys.path.insert(0, '/data/czx/GapScope_Revision_20261113/cv'); import evaluate_cv as ev
print('pool compositions', len(K), '| traced covered', sum(k in rows for k in set(ev.Akey)), '/85 | compiled covered', sum(k in rows for k in ev.Bkeys), '/2467')
