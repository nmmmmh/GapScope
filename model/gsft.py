"""GS_FT / GS_FT_HSE: release Roost encoder (MP-pretrained state, same as the ROOST baseline) trained end-to-end per fold
with the GS two-component lognormal mixture head (+ GS space-group input for tier-A records that have it).
Env: GSFT_ROOT = dir holding roostlib.py (release Roost code + W paths); GSFT_DIR = ft dir (ft_data.pt; outputs in out/)."""
import sys, os, json, time, math, argparse, random
import torch as _t
_ol = _t.load
_t.load = lambda *a, **k: _ol(*a, **{**k, "weights_only": False})
import types as _ty
class _Tr:
    def __init__(self, *a, **k): pass
    def __enter__(self): return self
    def __exit__(self, *a): return False
    def update(self, *a, **k): pass
    def set_postfix(self, *a, **k): pass
try:
    import tqdm.autonotebook
except Exception:
    _m = _ty.ModuleType('tqdm'); _a = _ty.ModuleType('tqdm.autonotebook'); _a.trange = _Tr; _m.autonotebook = _a
    import importlib.machinery as _im
    _m.__spec__ = _im.ModuleSpec('tqdm', None); _a.__spec__ = _im.ModuleSpec('tqdm.autonotebook', None); _m.__path__ = []
    sys.modules['tqdm'] = _m; sys.modules['tqdm.autonotebook'] = _a
try:
    import torch.utils.tensorboard
except Exception:
    _t2 = _ty.ModuleType('torch.utils.tensorboard'); _t2.SummaryWriter = object; sys.modules['torch.utils.tensorboard'] = _t2
sys.path.insert(0, os.environ['GSFT_ROOT'])
from roostlib import *          # W, S, EMB, Roost, CompositionData, collate_batch, parse_roost (release code)
import numpy as np, pandas as pd, torch
import torch.nn as nn, torch.nn.functional as F
FT = os.environ['GSFT_DIR']
OUT = f'{FT}/out'
torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
dev = torch.device('cuda')
K = 2
L2PI = 0.5 * math.log(2 * math.pi)
emb_keys = set(json.load(open(EMB)))


def graphs(formulas, tag):
    """Roost inputs via the release CompositionData (one item per formula)."""
    os.makedirs(f'{OUT}/tmp', exist_ok=True)
    p = f'{OUT}/tmp/{tag}_{os.getpid()}.csv'
    pd.DataFrame(dict(id=range(len(formulas)), composition=formulas, Eg=0.0)).to_csv(p, index=False)
    ds = CompositionData(p, EMB, 'regression')
    items = [ds[i] for i in range(len(ds))]
    os.remove(p)
    return items


def collate(items, idx):
    b = collate_batch([items[i] for i in idx])[0]
    return [t.to(dev) for t in b]


class GSFT(nn.Module):
    def __init__(self, ck, sg_dim):
        super().__init__()
        r = Roost(**ck['model_params'], device=dev)
        r.load_state_dict(ck['state_dict'])
        self.enc, self.out = r.material_nn, r.output_nn
        L = self.out.fcs[0].in_features if len(self.out.fcs) else self.out.fc_out.in_features
        self.out.fc_out = nn.Linear(self.out.fc_out.in_features, 3 * K)
        self.sgp = nn.Linear(sg_dim, L)
        nn.init.zeros_(self.sgp.weight); nn.init.zeros_(self.sgp.bias)   # formula-only at init

    def init_head(self, y):
        q = torch.quantile(torch.log(torch.as_tensor(y, dtype=torch.float32)), torch.tensor([.3, .7]))
        with torch.no_grad():
            self.out.fc_out.weight.mul_(0.1)
            b = self.out.fc_out.bias
            b[:K] = 0.; b[K:2 * K] = q; b[2 * K:] = math.log(math.expm1(0.4))

    def forward(self, b, sg=None, sgm=None):
        h = self.enc(*b)
        if sg is not None:
            h = h + sgm[:, None] * self.sgp(sg)
        o = self.out(h)
        lg, mu, s = o.split(K, 1)
        return F.log_softmax(lg, 1), mu.clamp(-8, 4), F.softplus(s) + 1e-2


def nll(p, y):
    lw, mu, sd = p
    ly = torch.log(y)[:, None]
    z = (ly - mu) / sd
    return -torch.logsumexp(lw - ly - torch.log(sd) - L2PI - 0.5 * z * z, 1)


def Phi(x):
    return 0.5 * (1 + torch.erf(x / math.sqrt(2)))


def crps(p, y):
    """closed-form lognormal-mixture CRPS (same formula as density_score.score)."""
    lw, mu, sd = p
    w = lw.exp()
    z = (torch.log(y)[:, None] - mu) / sd
    means = torch.exp(mu + 0.5 * sd * sd)
    absl = y[:, None] * (2 * Phi(z) - 1) + means * (1 - 2 * Phi(z - sd))
    dist = torch.sqrt(sd[:, :, None] ** 2 + sd[:, None, :] ** 2)
    mn = (means[:, :, None] * Phi((mu[:, None, :] - mu[:, :, None] - sd[:, :, None] ** 2) / dist)
          + means[:, None, :] * Phi((mu[:, :, None] - mu[:, None, :] - sd[:, None, :] ** 2) / dist))
    pair = means[:, :, None] + means[:, None, :] - 2 * mn
    return (w * absl).sum(1) - 0.5 * (w[:, :, None] * w[:, None, :] * pair).sum((1, 2))


def fold_sets(D, variant, fold):
    fd = D['folds'][variant]
    test = {k for k, v in fd.items() if v == fold}
    ivk = set(D['inner_val'][variant][str(fold)])
    A = D['A']
    tr = [r for r in A if r['key'] not in test and r['key'] not in ivk]
    va = [r for r in A if r['key'] in ivk]
    te = [r for r in A if r['key'] in test]
    B = D['B']
    trB = [i for i, k in enumerate(B['key']) if fd[k] != fold]
    teB = [k for k in D['Bkeys'] if fd[k] == fold]
    return test, tr, va, te, trB, teB


def pretrain_hse(ck, D, test, seed, epochs, tag):
    """hse/pretrain.py logic (standardised log1p target, MSE, 5% random validation, best state, patience) on the
    release Roost model (encoder + original output network, mu column), HSE pool minus test-fold keys."""
    H = D['hse']
    keep = [i for i, (k, f) in enumerate(zip(H['key'], H['formula'])) if k not in test and all(e in emb_keys for e in parse_roost(f)[0])]
    info = dict(rows_total=len(H['key']), rows_kept=len(keep), keys_dropped_test=len({k for k in H['key'] if k in test}))
    items = graphs([H['formula'][i] for i in keep], 'hse')
    y = torch.tensor([H['y'][i] for i in keep], dtype=torch.float32).log1p()
    lm, ls = float(y.mean()), float(y.std())
    z = ((y - lm) / ls).to(dev)
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(z), generator=g)
    nv = int(round(.05 * len(z)))
    vi, ti = perm[:nv], perm[nv:]
    torch.manual_seed(seed)
    r = Roost(**ck['model_params'], device=dev).to(dev)
    r.load_state_dict(ck['state_dict'])
    opt = torch.optim.AdamW(r.parameters(), lr=3e-4, weight_decay=1e-6)
    vb = [vi[s:s + 2048].tolist() for s in range(0, len(vi), 2048)]
    vcache = [collate(items, ix) for ix in vb]

    def val():
        r.eval()
        with torch.no_grad():
            return float(sum(((r(*b)[:, 0] - z[ix]) ** 2).sum() for b, ix in zip(vcache, vb)) / len(vi))
    best, best_ep, wait = val(), 0, 0
    best_state = {k: v.detach().clone() for k, v in r.state_dict().items()}
    t0 = time.time()
    for ep in range(1, epochs + 1):
        r.train()
        order = ti[torch.randperm(len(ti), generator=g)]
        for s in range(0, len(order), 256):
            ix = order[s:s + 256].tolist()
            opt.zero_grad(set_to_none=True)
            l = ((r(*collate(items, ix))[:, 0] - z[ix]) ** 2).mean()
            l.backward(); opt.step()
        v = val()
        print(f'hse ep {ep} val {v:.4f} {time.time() - t0:.0f}s', flush=True)
        if v < best - 1e-6:
            best, best_ep, wait = v, ep, 0
            best_state = {k: q.detach().clone() for k, q in r.state_dict().items()}
        else:
            wait += 1
            if wait >= 5:
                break
    info.update(selected_epoch=best_ep, best_val_mse_z=best, epochs_run=ep, wall_s=time.time() - t0)
    return dict(model_params=ck['model_params'], state_dict=best_state), info


def run(a):
    D = torch.load(f'{FT}/ft_data.pt', weights_only=False)
    t0 = time.time()
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed); torch.cuda.manual_seed_all(a.seed)
    ck = torch.load(f'{W}/original_comparison/models/original_mp/checkpoint-r{a.seed}.pth.tar', map_location='cpu')
    test, tr, va, te, trB, teB = fold_sets(D, a.variant, a.fold)
    info = dict(vars(a), A_train=len(tr), A_val=len(va), A_test=len(te), B_train_values=len(trB), B_test_comps=len(teB),
                A_train_comps=len({r['key'] for r in tr}))
    if a.hse:
        hs = a.hse_from if a.hse_from else a.seed
        hp = f'{OUT}/hse_states/{a.variant}_f{a.fold}_s{hs}_e{a.hse_epochs}.pt'
        if a.hse_from:      # one HSE pretrain per fold (seed hse_from); wait for it
            while not os.path.exists(hp + '.json'):
                time.sleep(30)
            time.sleep(5)
        if os.path.exists(hp):
            ck, info['hse'] = torch.load(hp, map_location='cpu'), json.load(open(hp + '.json'))
        else:
            ck, hinfo = pretrain_hse(ck, D, test, a.seed, a.hse_epochs, a.arm)
            os.makedirs(f'{OUT}/hse_states', exist_ok=True)
            torch.save(ck, hp); json.dump(hinfo, open(hp + '.json', 'w')); info['hse'] = hinfo
        torch.manual_seed(a.seed)
    sg_all, sgm_all = D['sg'], D['sg_mask']
    B = D['B']
    forms = [r['formula'] for r in tr] + [B['formula'][i] for i in trB]
    y = torch.tensor([r['y'] for r in tr] + [B['y'][i] for i in trB], dtype=torch.float32, device=dev)
    w = torch.tensor([r['wc'] for r in tr] + [B['w'][i] for i in trB], dtype=torch.float32, device=dev)
    if a.a_mass != 1.0:
        w[:len(tr)] *= a.a_mass
    nA = len(tr)
    sg = torch.zeros(len(forms), sg_all.shape[1], device=dev); sgm = torch.zeros(len(forms), device=dev)
    sg[:nA] = sg_all[[r['idx'] for r in tr]].to(dev); sgm[:nA] = sgm_all[[r['idx'] for r in tr]].to(dev)
    items = graphs(forms, 'train')
    model = GSFT(ck, sg_all.shape[1]).to(dev)
    model.init_head(y.cpu())
    enc_params = list(model.enc.parameters())
    ids = {id(p) for p in enc_params}
    head_params = [p for p in model.parameters() if id(p) not in ids]
    opt = torch.optim.AdamW([dict(params=enc_params, lr=a.lr * a.enc_lr_mult), dict(params=head_params, lr=a.lr)], weight_decay=1e-6)
    # inner-validation tier-A (all available inputs), composition mean of hierarchically weighted NLL
    vitems = graphs([r['formula'] for r in va], 'val')
    vb = collate(vitems, list(range(len(va))))
    vy = torch.tensor([r['y'] for r in va], device=dev)
    vw = torch.tensor([r['wc'] for r in va], device=dev) / len({r['key'] for r in va})
    vsg = sg_all[[r['idx'] for r in va]].to(dev); vsgm = sgm_all[[r['idx'] for r in va]].to(dev)

    def val():
        model.eval()
        with torch.no_grad():
            return float((nll(model(vb, vsg, vsgm), vy) * vw).sum())
    best, best_ep, wait = val(), 0, 0
    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    g = torch.Generator().manual_seed(a.seed)
    hist = []
    for ep in range(1, a.epochs + 1):
        model.train()
        order = torch.randperm(len(forms), generator=g)
        tl = 0.
        for s in range(0, len(order), a.batch):
            ix = order[s:s + a.batch]
            ixd = ix.to(dev)
            p = model(collate(items, ix.tolist()), sg[ixd], sgm[ixd])
            l = nll(p, y[ixd])
            if a.crps > 0:
                l = l + a.crps * crps(p, y[ixd])
            loss = (l * w[ixd]).sum() / w[ixd].sum()
            opt.zero_grad(set_to_none=True)
            loss.backward(); opt.step()
            tl += float(loss) * len(ix)
        v = val()
        hist.append((ep, tl / len(order), v))
        if v < best - 1e-6:
            best, best_ep, wait = v, ep, 0
            best_state = {k: q.detach().clone() for k, q in model.state_dict().items()}
        else:
            wait += 1
            if wait >= a.patience:
                break
        if ep % 10 == 0 or ep == 1:
            print(f'ep {ep} train {tl / len(order):.4f} val {v:.4f} best {best:.4f}@{best_ep} {time.time() - t0:.0f}s', flush=True)
    model.load_state_dict(best_state)
    model.eval()
    out = {}
    with torch.no_grad():
        def comp(p):
            lw, mu, sd = p
            w_ = np.clip(np.exp(lw.double().cpu().numpy()), 1e-12, None)
            return np.stack([w_ / w_.sum(1, keepdims=True), mu.double().cpu().numpy(), sd.double().cpu().numpy()])
        titems = graphs([r['formula'] for r in te], 'testA')
        tb = collate(titems, list(range(len(te))))
        tsg = sg_all[[r['idx'] for r in te]].to(dev); tsgm = sgm_all[[r['idx'] for r in te]].to(dev)
        v0 = comp(model(tb, tsg, torch.zeros_like(tsgm)))
        v1 = comp(model(tb, tsg, tsgm))
        out.update(A0=v0, A1=v1, A2=v0, A3=v1, A_idx=np.array([r['idx'] for r in te]))
        fB = dict(zip(D['Bkeys'], D['Bformulas']))
        bitems = graphs([fB[k] for k in teB], 'testB')
        parts = [comp(model(collate(bitems, list(range(s, min(s + 1024, len(teB))))))) for s in range(0, len(teB), 1024)]
        out['B'] = np.concatenate(parts, 1)
        out['B_keys'] = np.array(teB)
    os.makedirs(f'{OUT}/gs_out_v2', exist_ok=True)
    tag = f'{a.variant}_{a.arm}_seed{a.seed}_fold{a.fold}'
    np.savez_compressed(f'{OUT}/gs_out_v2/{tag}.npz', **out)
    info.update(selected_epoch=best_ep, best_val_nll=best, epochs_run=ep, wall_s=time.time() - t0, history=hist)
    json.dump(info, open(f'{OUT}/gs_out_v2/{tag}_info.json', 'w'))
    print('done', tag, best_ep, ep, f'{best:.4f}', f'{time.time() - t0:.0f}s', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--variant', default='comp'); p.add_argument('--fold', type=int, required=True); p.add_argument('--seed', type=int, required=True)
    p.add_argument('--arm', default='GS_FT'); p.add_argument('--hse', action='store_true'); p.add_argument('--hse_epochs', type=int, default=30); p.add_argument('--hse_from', type=int, default=0)
    p.add_argument('--lr', type=float, default=3e-4); p.add_argument('--enc_lr_mult', type=float, default=1.0)
    p.add_argument('--epochs', type=int, default=300); p.add_argument('--patience', type=int, default=40); p.add_argument('--batch', type=int, default=128)
    p.add_argument('--crps', type=float, default=0.0); p.add_argument('--a_mass', type=float, default=1.0)
    p.add_argument('--threads', type=int, default=2)
    a = p.parse_args()
    torch.set_num_threads(a.threads)
    run(a)
