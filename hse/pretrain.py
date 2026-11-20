"""HSE pretraining of the shared aggregate network (formula_only + formula+SG views), one process per seed."""
import sys, time, json, argparse
sys.path.insert(0, '/data/czx/GapScope_Revision_20261113/hse')
from hse_common import *
p = argparse.ArgumentParser(); p.add_argument('--seed', type=int, required=True); a = p.parse_args()
dev = rt.configure(); m = rt.modules(); seed = a.seed
pool = torch.load(H/'hse_pool_features.pt', map_location='cpu', weights_only=False)
y = pool.pop('targets'); keys = pool.pop('keys')
feats = m['t35'].transform(pool, 'hybrid', m['loader43'].symmetry_table(), set())
# composition scaler: one row per distinct composition key
first = {}
for i, k in enumerate(keys): first.setdefault(k, i)
idx = torch.tensor(sorted(first.values()))
cm, cs = m['t10'].standardize(feats['composition'][idx].to(dev))
views = m['t10'].four_views(feats, cm, cs, torch.zeros(160, device=dev), torch.ones(160, device=dev), dev)[[0, 1]]
t = y.to(dev).log1p(); lm, ls = float(t.mean()), float(t.std(unbiased=False)); z = ((t - lm) / ls).float()
N = len(z); g = torch.Generator().manual_seed(seed); perm = torch.randperm(N, generator=g)
nv = int(round(.05 * N)); vi, ti = perm[:nv].to(dev), perm[nv:].to(dev)
torch.manual_seed(seed)
model = m['interaction']('cross_attention', torch.zeros(184, dtype=torch.float64), 0., 1., mode='F').to(dev)
params = [q for n, q in model.named_parameters() if not n.startswith('report_head')]
opt = torch.optim.Adam(params, lr=1e-3)
def loss(ix): return (model.aggregate(views[:, ix]) - z[ix][None]).square().mean()
def val():
    model.eval()
    with torch.no_grad(): return float(sum(loss(vi[s:s+8192]) * len(vi[s:s+8192]) for s in range(0, len(vi), 8192)) / len(vi))
best = val(); best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}; best_ep = wait = 0; hist = [dict(epoch=0, val=best)]
t0 = time.time()
for ep in range(1, 101):
    model.train(); order = ti[torch.randperm(len(ti), generator=g).to(dev)]; tot = 0.
    for s in range(0, len(order), 2048):
        ix = order[s:s+2048]; opt.zero_grad(set_to_none=True); l = loss(ix); l.backward(); opt.step(); tot += float(l) * len(ix)
    v = val(); hist.append(dict(epoch=ep, train=tot / len(order), val=v)); print(f'seed={seed} ep={ep} train={tot/len(order):.4f} val={v:.4f}', flush=True)
    if v < best - 1e-6: best, best_ep, wait = v, ep, 0; best_state = {k: q.detach().cpu().clone() for k, q in model.state_dict().items()}
    else:
        wait += 1
        if wait >= 10: break
torch.save(dict(state=best_state, comp_mean=cm.cpu(), comp_std=cs.cpu(), log1p_mean=lm, log1p_std=ls, selected_epoch=best_ep,
                best_val=best, history=hist, seed=seed, n_train=len(ti), n_val=len(vi)), H/f'hse_pretrain_seed{seed}.pt')
json.dump(dict(seed=seed, selected_epoch=best_ep, best_val_mse=best, epochs_run=ep, n_train=len(ti), n_val=len(vi), seconds=time.time() - t0,
               val_mse_epoch0=hist[0]['val']), open(H/f'hse_pretrain_seed{seed}.json', 'w'))
print('DONE', seed, best_ep, best, flush=True)
