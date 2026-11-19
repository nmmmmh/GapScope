"""How much of a functional's apparent error against experimental band gaps is phase mismatch?

For the measurements whose specimen was resolved to a specific structure, compare the HSE gap of
THAT structure with the measured gap, against the error obtained without phase information.
"""
import warnings, collections, sys
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, torch
from pymatgen.core import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

CONS = '/home/czx/2024/BandgapFusion_Consolidated'
IDX = pd.read_csv(CONS + '/results/physical_cases/resolved_structure_index.csv')
D = pd.read_csv('/data/czx/GapScope_Revision_20261113/struct/descriptors.csv')
T = torch.load('/data/czx/GapScope_Revision_20261113/cv/data/tierA.pt', weights_only=False)
R = T['records'] if isinstance(T, dict) else T
gap = {x['measurement_id']: float(x['gap_ev']) for x in R}
allrep = collections.defaultdict(list)
for x in R:
    allrep[x['composition_key']].append(float(x['gap_ev']))

pool = collections.defaultdict(list)
for _, r in D.iterrows():
    if np.isfinite(r.Band_gap_HSE):
        pool[r.composition_key].append((int(r.sg_number), float(r.Band_gap_HSE), r.snumat_id))
print('SNUMAT compositions with HSE', len(pool))

rows, nosg, nocomp, nomatch = [], 0, 0, 0
seen = {}
for _, r in IDX.iterrows():
    mid, key = r.measurement_id, r.composition_key
    if mid not in gap:
        continue
    if key not in pool:
        nocomp += 1; continue
    p = r.cif_path
    if p not in seen:
        try:
            st = Structure.from_file(CONS + '/' + p)
            seen[p] = SpacegroupAnalyzer(st, symprec=0.1, angle_tolerance=5).get_space_group_number()
        except Exception as e:
            seen[p] = None
    sg = seen[p]
    if sg is None:
        nosg += 1; continue
    cands = pool[key]
    hit = [g for s, g, _ in cands if s == sg]
    if not hit:
        nomatch += 1; continue
    rows.append(dict(mid=mid, key=key, formula=r.formula, sg=sg, y=gap[mid],
                     hse_matched=float(np.mean(hit)),
                     hse_all=[g for _, g, _ in cands], n_poly=len(cands)))
print('resolved measurements', len(IDX), '-> matched to a SNUMAT polymorph:', len(rows),
      '| no composition in pool:', nocomp, '| CIF unreadable:', nosg, '| no same-SG polymorph:', nomatch)
if not rows:
    sys.exit()
M = pd.DataFrame(rows)
M['err_matched'] = (M.hse_matched - M.y).abs()
M['err_blind'] = [np.mean([abs(g - y) for g in gs]) for gs, y in zip(M.hse_all, M.y)]
M['err_median_poly'] = [abs(np.median(gs) - y) for gs, y in zip(M.hse_all, M.y)]
M['err_collapsed'] = [abs(np.median(gs) - np.median(allrep[k])) for gs, k in zip(M.hse_all, M.key)]
multi = M[M.n_poly >= 2]
def rep(name, v):
    v = np.asarray(v, float)
    rs = np.random.RandomState(0)
    b = [v[rs.randint(0, len(v), len(v))].mean() for _ in range(4000)]
    print('  %-46s n=%3d  MAE=%.3f eV  [%.3f, %.3f]' % (name, len(v), v.mean(),
          np.percentile(b, 2.5), np.percentile(b, 97.5)))
print('\nall resolved measurements')
rep('HSE of the measured phase vs that measurement', M.err_matched)
rep('HSE averaged over the polymorphs vs that measurement', M.err_blind)
rep('median HSE over polymorphs vs median of all reports', M.err_collapsed)
print('\nonly compositions with >= 2 computed polymorphs')
rep('HSE of the measured phase', multi.err_matched)
rep('HSE without phase information', multi.err_blind)
rep('both sides collapsed to a median', multi.err_collapsed)
d = multi.err_blind - multi.err_matched
rs = np.random.RandomState(1)
b = [d.values[rs.randint(0, len(d), len(d))].mean() for _ in range(4000)]
print('\n  phase mismatch contributes %.3f eV [%.3f, %.3f] of the apparent error (n=%d)'
      % (d.mean(), np.percentile(b, 2.5), np.percentile(b, 97.5), len(d)))
M.drop(columns=['hse_all']).to_csv('/data/czx/GapScope_Revision_20261113/struct/bench_split.csv', index=False)
