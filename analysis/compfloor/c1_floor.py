"""Error floor on the 3,895-row compilation: report- and composition-weighted, split by SNUMAT polymorph count."""
import json, numpy as np, pandas as pd, warnings
from pymatgen.core import Composition
from scipy.stats import mannwhitneyu
warnings.filterwarnings('ignore')
R='/data/czx/GapScope_Revision_20261113/'; O=R+'compfloor/'
def key(f):
    try:
        c=Composition(f).fractional_composition
        return '|'.join(f'{el}:{c[el]:.12f}' for el in sorted(c.as_dict()))
    except Exception: return None
A=pd.read_csv(R+'cv/data/tierB_all_rows.csv')
A=A.rename(columns={'Eg':'y'}); A['ckey']=A.key
n_raw=len(A)
# collapse transcription duplicates: identical value on one composition counts once
D=A.drop_duplicates(['ckey','y']).copy()
print('rows %d, after collapsing identical values %d, compositions %d'%(n_raw,len(D),D.ckey.nunique()))
# polymorph count per composition: distinct space groups over both SNUMAT exports
S=pd.read_csv(R+'benchmark/snumat_tidy.csv')[['ckey','sg']]
C=pd.read_csv(R+'benchmark/hse_csv_keyed.csv')[['ckey','Space_group_cif']].rename(columns={'Space_group_cif':'sg'})
SG=pd.concat([S,C]).dropna(); SG['sg']=SG.sg.astype(int)
nsg=SG.groupby('ckey').sg.nunique()
def comp_stats(df):
    g=df.groupby('ckey').y
    med=g.transform('median'); df=df.assign(ad=(df.y-med).abs())
    pc=df.groupby('ckey').agg(n=('y','size'),mad=('ad','mean'),sad=('ad','sum'),rng=('y',lambda s:s.max()-s.min()),med=('y','median')).reset_index()
    pc['nsg']=pc.ckey.map(nsg); pc['cls']=np.where(pc.nsg.isna(),'unmatched',np.where(pc.nsg>=2,'multi','single'))
    return pc
def summarise(pc,lab):
    rows=[]
    for cls in ['all','multi','single','unmatched']:
        p=pc if cls=='all' else pc[pc.cls==cls]
        if len(p)==0: continue
        rows.append(dict(set=lab,cls=cls,n_comp=len(p),n_rep=int(p.n.sum()),n_dup_comp=int((p.n>=2).sum()),
            floor_report=p.sad.sum()/p.n.sum(), floor_comp=p.mad.mean(),
            floor_dup_comp=p.loc[p.n>=2,'mad'].mean(), share_of_total=p.sad.sum()/pc.sad.sum()))
    return rows
out=[]
for lab,df in [('raw',A),('collapsed',D)]:
    pc=comp_stats(df); out+=summarise(pc,lab)
    if lab=='collapsed':
        pc.to_csv(O+'per_composition.csv',index=False)
        dup=pc[pc.n>=2]
        m,s=dup[dup.cls=='multi'].mad,dup[dup.cls=='single'].mad
        print('dup compositions multi %d single %d unmatched %d'%(len(m),len(s),(dup.cls=='unmatched').sum()))
        print('median per-composition floor multi %.3f single %.3f, MWU P=%.2g'%(m.median(),s.median(),mannwhitneyu(m,s).pvalue))
        rng=np.random.default_rng(0); bs=[]
        for _ in range(2000):
            a=m.sample(len(m),replace=True,random_state=rng.integers(1e9)).mean(); b=s.sample(len(s),replace=True,random_state=rng.integers(1e9)).mean(); bs.append(a/b)
        print('mean floor multi %.3f single %.3f ratio %.2f [%.2f, %.2f]'%(m.mean(),s.mean(),m.mean()/s.mean(),*np.percentile(bs,[2.5,97.5])))
        # Matbench-style cleaning: drop compositions whose values span more than 0.1 eV
        drop=pc[pc.rng>0.1]
        print('0.1 eV rule removes %d compositions (%d multi, %d single, %d unmatched), %.1f%% of the floor'%(len(drop),(drop.cls=='multi').sum(),(drop.cls=='single').sum(),(drop.cls=='unmatched').sum(),100*drop.sad.sum()/pc.sad.sum()))
        # floor vs number of space groups
        for k,lo,hi in [('1',1,1),('2',2,2),('3-4',3,4),('5+',5,99)]:
            p=dup[(dup.nsg>=lo)&(dup.nsg<=hi)]
            print('  nsg %s: %d compositions, mean floor %.3f, median %.3f'%(k,len(p),p.mad.mean(),p.mad.median()))
T=pd.DataFrame(out); T.to_csv(O+'floor_summary.csv',index=False)
pd.set_option('display.width',200); print(T.round(4).to_string(index=False))
