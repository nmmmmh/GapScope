"""Madelung site potentials for every SNUMAT record (relaxed structure, oxidation states from bond valence, else the first
composition guess). Per record: mean potential at cation and anion sites (V), Madelung gap M = e(phi_anion - phi_cation),
Ewald energy per atom. Output refcheck/m1_madelung.csv. 24 processes, OMP_NUM_THREADS=1."""
import json, sys, warnings, numpy as np, pandas as pd
from multiprocessing import Pool
warnings.filterwarnings('ignore')
from pymatgen.io.vasp import Poscar
from pymatgen.analysis.ewald import EwaldSummation
from pymatgen.analysis.bond_valence import BVAnalyzer
D = json.load(open('/home/czx/2024/doped/data/snumat.json'))
def one(i):
    r = D[i]
    try:
        txt = r['Structure_rlx']; s = Poscar.from_str(txt).structure
        if len(s) > 160: return dict(i=i, err='large')
        try: s2 = BVAnalyzer().get_oxi_state_decorated_structure(s); how = 'bv'
        except Exception:
            g = s.composition.oxi_state_guesses(max_sites=-1)
            if not g: return dict(i=i, err='nooxi')
            s2 = s.copy(); s2.add_oxidation_state_by_element(g[0]); how = 'guess'
        q = np.array([sp.oxi_state for sp in s2.species])
        if (q == 0).any() or not ((q > 0).any() and (q < 0).any()): return dict(i=i, err='charge')
        ew = EwaldSummation(s2, compute_forces=False)
        phi = np.array([2 * ew.get_site_energy(j) / q[j] for j in range(len(s2))])   # eV/e
        return dict(i=i, how=how, phi_cat=phi[q > 0].mean(), phi_an=phi[q < 0].mean(), M=(phi[q < 0].mean() - phi[q > 0].mean()),
                    M_min=(phi[q < 0].min() - phi[q > 0].max()), e_ewald=ew.total_energy / len(s2), qmax=q.max(), n=len(s2))
    except Exception as e:
        return dict(i=i, err=type(e).__name__)
if __name__ == '__main__':
    with Pool(24) as p: out = p.map(one, range(len(D)), chunksize=20)
    O = pd.DataFrame(out)
    O['SNUMAT_id'] = [D[i]['SNUMAT_id'] for i in O.i]; O['ICSD_number'] = [D[i]['ICSD_number'] for i in O.i]
    O['sg'] = [int(D[i]['Space_group_rlx']) for i in O.i]; O['hse'] = [float(D[i]['Band_gap_HSE']) for i in O.i]
    O['hse_opt'] = [float(D[i]['Band_gap_HSE_optical']) for i in O.i]; O['gga'] = [float(D[i]['Band_gap_GGA']) for i in O.i]
    O['formula'] = [Poscar.from_str(D[i]['Structure_rlx']).structure.composition.reduced_formula if 'err' not in O.columns or pd.isna(O.err[k]) else '' for k, i in enumerate(O.i)]
    O.to_csv('/data/czx/GapScope_Revision_20261113/refcheck/m1_madelung.csv', index=False)
    print(O.err.value_counts(dropna=False) if 'err' in O else 'no errors', '\nhow', O.how.value_counts().to_dict())
