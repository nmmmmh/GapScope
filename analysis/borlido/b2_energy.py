import re, numpy as np, pandas as pd
R = '/data/czx/GapScope_Revision_20261113/'
exec(open(R + 'borlido/b1_dc.py').read().split('G = {k')[0])
G = {k: g for k, g in S.groupby('ckey')}
T = pd.read_csv(R + 'borlido/b1_borlido_Dc.csv'); m = T[T.in_snumat].copy()
def full(f):
    c = G[key(f)]; return c.Free_energy.notna().all()
m['all_E'] = [full(f) for f in m.formula]
print('matched with energy for every polymorph', m.all_E.sum())
for th in (0.1, 0.3, 0.5): print(th, 'all', (m.D_c >= th).sum(), 'energy-complete subset', ((m.D_c >= th) & m.all_E).sum(), 'of', m.all_E.sum())
