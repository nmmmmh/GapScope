"""Build tier-A / tier-B data, composition-grouped and chemical-system-grouped 5-fold assignments, inner validation sets."""
import sys, json, csv, math, time, random
from decimal import Decimal
from collections import defaultdict
from pathlib import Path
CV = Path('/data/czx/GapScope_Revision_20261113/cv')
sys.path.insert(0, '/data/czx/GapScope_Revision_20261113/hse')
from hse_common import *
C = CONSO
sys.path.insert(0, str(C/'runtime/roost_deps')); sys.path.insert(0, str(C/'models/official/MatterVial-MP-gap'))
import pandas as pd
from pymatgen.core import Composition
from mattervial.packages.roost.roost.model import Roost
from inference import _input_features
SEED = 20261113
(CV/'data').mkdir(exist_ok=True)

def key(formula):
    a = Composition(formula).get_el_amt_dict(); t = sum(a.values())
    return '|'.join(f'{e}:{a[e]/t:.12f}' for e in sorted(a))
def system(k): return '|'.join(x.split(':')[0] for x in k.split('|'))

# ---------------- tier A
m = rt.modules(); symtab = m['loader43'].symmetry_table()
rf = torch.load(SRC/'data/reports/features.pt', map_location='cpu', weights_only=True)
meta = json.loads((SRC/'data/reports/metadata.json').read_text(encoding='utf-8-sig'))
src = [json.loads(l) for l in (SRC/'data/reports/records.jsonl').read_text(encoding='utf-8-sig').splitlines() if l.strip()]
AUX = {"additional_optical_edges", "fundamental_gap_eV", "secondary_same_sample_gap_ev", "source_conflict_alternative_gap_ev"}
core = []
for mt, s in zip(meta, src):
    for k in AUX: s.pop(k, None)
    r = dict(mt); r.update(s); r['origin'] = 'core'; core.append(r)
assert len(core) == 205
core_feat = m['t35'].transform(rf, 'hybrid', symtab, set())
cohort = {r['measurement_id']: r for r in rt.read_json(C/'data/confirmation/cohort.json')['records'] if r['cohort'] == 'independent'}
enc = rt.read_json(C/'data/confirmation/encoded_features.json')
encs = [r for r in enc if r['measurement_id'] in cohort]
assert len(encs) == 32
ext = []
for r in encs:
    c = dict(cohort[r['measurement_id']]); c['origin'] = 'external'; ext.append(c)
raw = [_input_features(r) for r in encs]
ext_feat = {k: torch.cat([x[k] for x in raw]) for k in raw[0]}
feat = {k: torch.cat([core_feat[k], ext_feat[k]]) for k in ('composition', 'sg_features', 'structure', 'sg_mask', 'structure_mask')}
recs = core + ext
for r in recs: assert r['gap_ev'] > 0 and r['source_group_id'] and r['specimen_group_id']
# keep only json-safe fields needed downstream
keep = ('measurement_id', 'formula', 'composition_key', 'gap_ev', 'source_group_id', 'specimen_group_id', 'phase_label', 'origin', 'phase_assignment', 'sample_phase_group')
recsl = [{k: r.get(k) for k in keep} for r in recs]
for r in recsl:
    r['phase_assignment'] = json.dumps(r['phase_assignment']) if r['phase_assignment'] is not None and not isinstance(r['phase_assignment'], str) else r['phase_assignment']
    r['sample_phase_group'] = json.dumps(r['sample_phase_group']) if r['sample_phase_group'] is not None and not isinstance(r['sample_phase_group'], str) else r['sample_phase_group']
# key sanity: stored composition_key equals pymatgen key of formula
bad = [r['measurement_id'] for r in recsl if key(r['formula']) != r['composition_key']]
print('tier-A key mismatches vs pymatgen key:', len(bad), bad[:3])
torch.save(dict(records=recsl, features=feat), CV/'data/tierA.pt')
Akeys = sorted({r['composition_key'] for r in recsl})
print('tier A: records', len(recsl), 'core comps', len({r['composition_key'] for r in core}), 'ext comps', len({r['composition_key'] for r in ext}), 'union comps', len(Akeys))

# ---------------- tier B
EXP = Path('/home/czx/2024/BandgapFusion_RoostReproductionWork/sources/extracted_members/data/datasets/expt-non-metals.csv')
ex = pd.read_csv(EXP); assert len(ex) == 3895 and list(ex.columns)[:3] == ['id', 'composition', 'Eg'], (len(ex), list(ex.columns))
assert (ex.Eg > 0).all() and ex.Eg.notna().all()
ex['key'] = [key(f) for f in ex.composition]
g = defaultdict(lambda: defaultdict(list)); forms = {}
for r in ex.itertuples():
    g[r.key][Decimal(str(r.Eg))].append(int(r.id))
for k, grp in ex.groupby('key'):
    forms[k] = grp.sort_values('id').composition.iloc[0]
rowsB = []
for k in sorted(g):
    for v, ids in sorted(g[k].items()):
        rowsB.append(dict(composition_key=k, formula=forms[k], target_ev=format(v, 'f'), within_composition_weight=1/len(g[k]), n_dup_rows=len(ids)))
pd.DataFrame(rowsB).to_csv(CV/'data/tierB_rows.csv', index=False)
ex[['id', 'composition', 'Eg', 'key']].to_csv(CV/'data/tierB_all_rows.csv', index=False)
Bkeys = sorted(g)
print('tier B: rows', len(ex), 'distinct (key,value)', len(rowsB), 'comps', len(Bkeys))
dev = torch.device('cuda')
base = C/'models/official/MatterVial-MP-gap/mattervial/featurizers/custom_models'
ck = torch.load(base/'mpgap_checkpoint-r0.pth.tar', map_location='cpu', weights_only=False)
emb = json.loads((base/'matscholar-embedding.json').read_text())
encd = Roost(**ck['model_params'], device=dev).to(dev).eval(); encd.load_state_dict(ck['state_dict'])
nm_ = ck['normalizer_dict']['band_gap']
def batch(formulas):
    w, f, si, ni, ci = [], [], [], [], []; off = 0; fr = []
    for b, formula in enumerate(formulas):
        comp = Composition(formula); a = comp.get_el_amt_dict(); el = list(a); t = sum(a.values()); n = len(el)
        w += [[a[e]/t] for e in el]; f += [emb[e] for e in el]
        si += [off + i for i in range(n) for _ in range(n)]; ni += [off + j for _ in range(n) for j in range(n)]
        ci += [b]*n; off += n
        v = [0.]*118
        for e in comp.elements: v[e.Z-1] = a[str(e)]/t
        fr.append(v)
    T = lambda x, dt: torch.tensor(x, dtype=dt, device=dev)
    return (T(w, torch.float32), T(f, torch.float32), T(si, torch.long), T(ni, torch.long), T(ci, torch.long)), fr
rows = []
with torch.no_grad():
    for s in range(0, len(Bkeys), 1024):
        fm = [forms[k] for k in Bkeys[s:s+1024]]; inp, fr = batch(fm)
        hid = encd.material_nn(*inp); raw_ = list(encd(*inp))[0][:, 0]
        sc = (raw_.double()*float(nm_['std']) + float(nm_['mean'])).cpu().numpy(); hid = hid.cpu().numpy()
        for i in range(len(fm)): rows.append(fr[i] + hid[i].tolist() + [float(sc[i])])
cb = torch.tensor(rows, dtype=torch.float32); assert torch.isfinite(cb).all() and cb.shape == (len(Bkeys), 183)
torch.save(dict(keys=Bkeys, formulas=[forms[k] for k in Bkeys], composition=cb), CV/'data/tierB_features.pt')

# ---------------- groups, folds
Aset, Bset = set(Akeys), set(Bkeys)
coreK = {r['composition_key'] for r in core}; extK = {r['composition_key'] for r in ext}
allkeys = sorted(Aset | Bset)
def assign(group_of):
    gA = sorted({group_of[k] for k in Aset}); gB = sorted({group_of[k] for k in allkeys} - set(gA))
    random.Random(SEED).shuffle(gA); random.Random(SEED + 1).shuffle(gB)
    fg = {x: i % 5 for i, x in enumerate(gA)}
    fg.update({x: (i + len(gA)) % 5 for i, x in enumerate(gB)})
    return {k: fg[group_of[k]] for k in allkeys}
def inner_val(fold_of, group_of, f):
    gA = sorted({group_of[k] for k in Aset if fold_of[k] != f})
    n = max(1, round(0.1 * len(gA))); random.Random(SEED + 100 + f).shuffle(gA)
    vg = set(gA[:n]); return sorted(k for k in Aset if fold_of[k] != f and group_of[k] in vg)
nA = {k: sum(1 for r in recsl if r['composition_key'] == k) for k in Akeys}
nB = ex.groupby('key').size().to_dict()
summary = {}
for variant, gf in (('comp', lambda k: k), ('chem', system)):
    group_of = {k: gf(k) for k in allkeys}; fold_of = assign(group_of)
    iv = {f: inner_val(fold_of, group_of, f) for f in range(5)}
    df = pd.DataFrame([dict(composition_key=k, chem_system=system(k), group=group_of[k], fold=fold_of[k] + 1, in_tierA=k in Aset, in_tierB=k in Bset,
                            n_reports_A=nA.get(k, 0), n_rows_B=int(nB.get(k, 0))) for k in allkeys])
    df.to_csv(CV/f'folds_{variant}.csv', index=False)
    json.dump({str(f+1): v for f, v in iv.items()}, open(CV/f'data/inner_val_{variant}.json', 'w'))
    sm = dict(per_fold=[])
    for f in range(5):
        d = df[df.fold == f + 1]
        sm['per_fold'].append(dict(fold=f+1, comps_total=len(d), A_comps=int(d.in_tierA.sum()), A_reports=int(d.n_reports_A.sum()), B_comps=int(d.in_tierB.sum()), B_rows=int(d.n_rows_B.sum()),
                                   both=int((d.in_tierA & d.in_tierB).sum()), inner_val_A_comps=len(iv[f])))
    sm['total'] = dict(A_comps=int(df.in_tierA.sum()), A_reports=int(df.n_reports_A.sum()), B_comps=int(df.in_tierB.sum()), B_rows=int(df.n_rows_B.sum()), both=int((df.in_tierA & df.in_tierB).sum()), union=len(df))
    summary[variant] = sm
summary['tierA'] = dict(core_reports=205, core_comps=len(coreK), ext_records=32, ext_comps=len(extK), union_comps=len(Akeys))
summary['tierB'] = dict(rows=len(ex), distinct_key_value=len(rowsB), comps=len(Bkeys))
summary['overlap'] = dict(A_and_B_comps=len(Aset & Bset), core_and_B=len(coreK & Bset), ext_and_B=len(extK & Bset), core_and_ext=len(coreK & extK),
                          A_reports_in_B_comps=sum(nA[k] for k in Aset & Bset), B_rows_in_A_comps=int(sum(nB.get(k, 0) for k in Aset)))
json.dump(summary, open(CV/'fold_summary.json', 'w'), indent=1); print(json.dumps(summary))
