import json, glob, numpy as np, re
Q = '/data/czx/GapScope_Revision_20261113/cv_rep/'
w = [json.load(open(f))['wall_s'] for f in glob.glob(Q + 's2026111[4-7]/ft/out/gs_out_v2/*_info.json')]
print('GS runs', len(w), 'wall min/median/max s', int(min(w)), int(np.median(w)), int(max(w)))
t = []
for f in glob.glob(Q + 's2026111[4-7]/logs/roost_*.log'):
    m = re.findall(r'^done \S+ \d+ \d+ ([\d.]+)', open(f, errors='ignore').read(), re.M); t += [float(x) for x in m]
print('Roost logs on 13 with done', len(t), 'median s', int(np.median(t)) if t else None)
