import sys
sys.path.insert(0, '/data/czx/GapScope_Revision_20261113/cv'); sys.path.insert(0, '/data/czx/GapScope_Revision_20261113/s5')
import numpy as np, pandas as pd
import evaluate_cv_v4 as E
ev = E.ev
FA = 'C:0.083333333333|H:0.416666666667|I:0.250000000000|N:0.166666666667|Pb:0.083333333333'
MA = 'C:0.083333333333|H:0.500000000000|I:0.250000000000|N:0.083333333333|Pb:0.083333333333'
for key in (FA, MA):
    print('==', key)
    for i, k in enumerate(ev.Akey):
        if k == key:
            a = ev.A[i]; print(' A', a['formula'], a['gap_ev'], a['phase_label'], a['phase_assignment'], 'w=%.3f' % ev.wA[i], a['origin'])
    print(' B values', ev.Bvals.get(key))
S = pd.read_csv('/data/czx/GapScope_Revision_20261113/benchmark/snumat_by_comp_sg.csv')
for key in (FA, MA):
    print('SNUMAT', key[:30], S[S.ckey == key][['sg', 'hse', 'Free_energy']].to_string() if (S.ckey == key).any() else 'none')
# hybrid lead halides in compiled data: how many and their gaps
def els(k): return set(p.split(':')[0] for p in k.split('|'))
hal = [(k, v) for k, v in ev.Bvals.items() if {'Pb'} <= els(k) and els(k) & {'I', 'Br', 'Cl'} and 'C' in els(k)]
print('compiled organic Pb halides', len(hal))
for k, v in hal[:25]: print('  ', k, np.round(v, 2))
