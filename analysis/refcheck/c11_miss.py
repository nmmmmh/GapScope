# SUPERSEDED (2026-11-19): reads the earlier D_c over all polymorphs, descr/d2_per_composition.csv.
# The paper uses the energy-only D_c (refcheck/c14_energy.py, descr/d2_per_composition_energy.csv written by refcheck/c15_write.py).
# Kept for comparison only; no number, table or figure of the paper comes from this script.
"""D_c misses: traced compositions with floor >= 0.05 eV but D_c < 0.3 eV. What drives their floor: phases (and are they in SNUMAT)
or specimens of one phase (sample form, method)?"""
import numpy as np, pandas as pd
from pymatgen.core import Composition
R = '/data/czx/GapScope_Revision_20261113/'
D = pd.read_csv(R + 'descr/d2_per_composition.csv')
P = pd.read_csv(R + 'suppdata/Supplementary_Data_1_GapScope_experimental_bandgaps.csv')
P['ph'] = P.phase_label.fillna('').astype(str).str.strip()
S = pd.read_csv(R + 'benchmark/snumat_by_comp_sg.csv')
def key(f):
    c = Composition(f).fractional_composition; return '|'.join(f'{el}:{c[el]:.12f}' for el in sorted(c.as_dict()))
def show(sel, lab):
    print('=====', lab, len(sel))
    for r in sel.sort_values('floor', ascending=False).itertuples():
        g = P[P.formula == r.formula]
        if len(g) == 0: g = P[P.composition_key == key(r.formula)]
        sn = S[S.ckey == key(r.formula)]
        print('%-10s floor %.3f Dc %.2f n_acc %d | reports %d phases %s | sg reported %s | SNUMAT sg %s | forms %s | methods %s | per-phase ranges %s' % (
            r.formula, r.floor, r.D_elec, r.n_acc, len(g), sorted(set(g.ph) - {''}), sorted(set(g.space_group.dropna().astype(str))),
            sorted(sn.sg.astype(int).tolist()), dict(g.sample_form.fillna('?').value_counts()), dict(g.measurement_method.fillna('?').value_counts()),
            {p: round(gg.bandgap_ev.max() - gg.bandgap_ev.min(), 2) for p, gg in g.groupby('ph')}))
show(D[(D.floor >= 0.05) & (D.D_elec < 0.3)], 'MISSES floor>=0.05, Dc<0.3')
show(D[(D.floor < 0.05) & (D.D_elec >= 0.3)], 'FALSE FLAGS floor<0.05, Dc>=0.3')
