"""Round-2 GS arms (gsft.py building blocks, extended): init from the MP state (mp) or from the fold's trained ROOST
checkpoint (warm, head-first then whole network at lr x stage2_mult); validation = inner (7 tier-A comps), group20 (20% of
training compositions, tiers A+B, grouped by composition) or roostval (compositions whose rows all lie in the ROOST
checkpoint's internal 20% validation split); optional L1 on the mixture location; lognormal or Laplace components.
Env as gsft.py, plus ROOST_RUNS (dir with {variant}_f{fold}_s{seed}/models/...) for warm init."""
import os, sys, json, time, math, argparse, random
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gsft import *          # stubs, roostlib, graphs, collate, GSFT, nll, crps, fold_sets, pretrain_hse
import gsft as G1


def lap_nll(p, y):
    lw, mu, b = p
    return -torch.logsumexp(lw - torch.log(2 * b) - (y[:, None] - mu).abs() / b, 1)


def _pca8():
    import json as _j
    tab = torch.tensor([r['features'] for r in _j.load(open('/data/czx/GapScope_Revision_20261113/hse/runtime/modules/symmetry_table.json'))['groups']], dtype=torch.float32) \
        if os.path.exists('/data/czx/GapScope_Revision_20261113/hse/runtime/modules/symmetry_table.json') else torch.load(f'{FT}/symtab.pt')
    mu_, sd_ = tab.mean(0), tab.std(0).clamp_min(1e-6)
    U, S_, V = torch.linalg.svd((tab - mu_) / sd_, full_matrices=False)
    P = torch.zeros(290, 8)
    P[230:, :] = (V[:8].T / sd_[:, None])
    return P


PCA8 = _pca8()


class GSFT2(GSFT):
    def __init__(self, ck, sg_dim, comp, tierflag=False, sg_rank=0, iv=0):
        super().__init__(ck, sg_dim)
        self.comp = comp
        self.iv = iv
        if sg_rank > 0:
            Lh = self.sgp.out_features
            self.sgp = nn.Sequential(nn.Linear(sg_dim, sg_rank, bias=False), nn.Linear(sg_rank, Lh))
            nn.init.normal_(self.sgp[0].weight, std=0.1); nn.init.zeros_(self.sgp[1].weight); nn.init.zeros_(self.sgp[1].bias)
        if iv:                      # (iv) shift of the component locations and a log-width change, linear in the SG input
            self.sgs = nn.Linear(iv, 2)
            nn.init.zeros_(self.sgs.weight); nn.init.zeros_(self.sgs.bias)
            self.register_buffer('Pm', torch.eye(sg_dim) if iv == sg_dim else PCA8.clone())
        self.sdmax = 0.0
        self.tierflag = tierflag
        if tierflag:
            self.tier_emb = nn.Embedding(2, self.out.fcs[0].in_features)
            nn.init.zeros_(self.tier_emb.weight)

    def init_head(self, y):
        if self.comp == 'ln':
            return super().init_head(y)
        q = torch.quantile(torch.as_tensor(y, dtype=torch.float32), torch.tensor([.3, .7]))
        with torch.no_grad():
            self.out.fc_out.weight.mul_(0.1)
            b = self.out.fc_out.bias
            b[:K] = 0.; b[K:2 * K] = q; b[2 * K:] = math.log(math.expm1(0.4))

    def head(self, h, tier=None, sg=None, sgm=None):
        if self.tierflag:          # learned tier offset (0 = provenance-linked A, 1 = compilation B), zero at init
            h = h + self.tier_emb(tier)
        lg, mu, s = self.out(h).split(K, 1)
        if self.comp == 'ln':
            sd = F.softplus(s) + 1e-2
            if self.iv and sg is not None:
                d = sgm[:, None] * self.sgs(sg @ self.Pm)
                mu = mu + d[:, 0:1]; sd = sd * torch.exp(d[:, 1:2])
            return F.log_softmax(lg, 1), mu.clamp(-8, 4), (sd.clamp(max=self.sdmax) if self.sdmax else sd)
        return F.log_softmax(lg, 1), mu, F.softplus(s) + 1e-3

    def forward(self, b, sg=None, sgm=None, tier=None):
        h = self.enc(*b)
        if sg is not None and not self.iv:
            h = h + sgm[:, None] * self.sgp(sg)
        return self.head(h, tier, sg, sgm)

    def forward2(self, b, sg, sgm, tier):
        # two views from one encoder pass: formula-only (encoder + head trained) and SG view on detached encoder features
        h = self.enc(*b)
        return self.head(h, tier), self.head(h.detach() + sgm[:, None] * self.sgp(sg), tier)


def loss_fn(model, p, y, a):
    l = nll(p, y) if model.comp == 'ln' else lap_nll(p, y)
    if a.beta > 0 and model.comp == 'ln':     # beta-NLL: per-sample NLL x stopgrad(mixture mean width ^ (2 beta))
        l = l * ((p[0].exp() * p[2]).sum(1).detach() ** (2 * a.beta))
    if a.crps > 0 and model.comp == 'ln':
        l = l + a.crps * crps(p, y)
    if a.l1 > 0:
        lw, mu, _ = p
        loc = (lw.exp() * mu).sum(1)
        loc = loc.exp() if model.comp == 'ln' else loc
        l = l + a.l1 * (loc - y).abs()
    return l




def pretrain_hse4(ck, D, test, seed, epochs, arm):
    """HSE pretraining variants (gsft.pretrain_hse logic: 5% random val, AdamW 3e-4, batch 256, patience 5, best state).
    NEG: drop rows with target < 0 (control; zeros kept, MSE on standardised log1p).
    NZ : keep only target > 0.01 (non-metals), MSE on standardised log1p.
    MT : drop target < 0; shared encoder, gap head (original output net, MSE on non-metal rows only) + metal logit head
         (BCE on all kept rows), equal weights; epoch chosen on 5% val of (BCE + non-metal MSE)."""
    H = D['hse']
    ya = np.array(H['y'], dtype=np.float64)
    nrows = len(ya)
    ok = [i for i, (k, f) in enumerate(zip(H['key'], H['formula'])) if k not in test and all(e in emb_keys for e in parse_roost(f)[0])]
    keep = [i for i in ok if ya[i] >= 0] if arm in ('NEG', 'MT') else [i for i in ok if ya[i] > {'NZ': 0.01, 'T025': 0.25, 'T05': 0.5, 'T10': 1.0}.get(arm, 0.01)]
    yk = torch.tensor(ya[keep], dtype=torch.float32)
    nm = yk > 0.01                                  # non-metal flag
    info = dict(arm=arm, rows_total=nrows, rows_kept=len(keep), rows_dropped_test_or_emb=nrows - len(ok), rows_nonmetal=int(nm.sum()),
                rows_metal_le001=int((~nm).sum()), rows_negative_dropped=int((ya[ok] < 0).sum()))
    items = graphs([H['formula'][i] for i in keep], 'hse')
    ref = yk[nm] if arm == 'MT' else yk
    lt = ref.log1p(); lm, ls = float(lt.mean()), float(lt.std())
    z = ((yk.log1p() - lm) / ls).to(dev)
    nmd = nm.to(dev); lab = (~nm).float().to(dev)
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(z), generator=g)
    nv = int(round(.05 * len(z)))
    vi, ti = perm[:nv], perm[nv:]
    torch.manual_seed(seed)
    r = Roost(**ck['model_params'], device=dev).to(dev)
    r.load_state_dict(ck['state_dict'])
    L = r.output_nn.fcs[0].in_features if len(r.output_nn.fcs) else r.output_nn.fc_out.in_features
    mh = nn.Linear(L, 1).to(dev) if arm == 'MT' else None
    prm = list(r.parameters()) + (list(mh.parameters()) if mh is not None else [])
    opt = torch.optim.AdamW(prm, lr=3e-4, weight_decay=1e-6)

    def lossf(b, ix):
        ixd = torch.as_tensor(ix, device=dev)
        if arm != 'MT':
            return ((r(*b)[:, 0] - z[ixd]) ** 2).mean(), None
        h = r.material_nn(*b)
        pred = r.output_nn(h)[:, 0]; lg = mh(h)[:, 0]
        bce = F.binary_cross_entropy_with_logits(lg, lab[ixd])
        m = nmd[ixd]
        mse = ((pred - z[ixd]) ** 2)[m].mean() if bool(m.any()) else pred.sum() * 0
        return mse + bce, (float(mse), float(bce))
    vb = [vi[s:s + 2048].tolist() for s in range(0, len(vi), 2048)]
    vcache = [collate(items, ix) for ix in vb]

    def val():
        r.eval()
        if mh is not None: mh.eval()
        tot = 0.
        with torch.no_grad():
            for b, ix in zip(vcache, vb):
                tot += float(lossf(b, ix)[0]) * len(ix)
        return tot / len(vi)
    best, best_ep, wait = val(), 0, 0
    best_state = {k: v.detach().clone() for k, v in r.state_dict().items()}
    t0 = time.time()
    for ep in range(1, epochs + 1):
        r.train()
        if mh is not None: mh.train()
        order = ti[torch.randperm(len(ti), generator=g)]
        for s in range(0, len(order), 256):
            ix = order[s:s + 256].tolist()
            opt.zero_grad(set_to_none=True)
            l, _ = lossf(collate(items, ix), ix)
            l.backward(); opt.step()
        v = val()
        print(f'hse4 {arm} ep {ep} val {v:.4f} {time.time() - t0:.0f}s', flush=True)
        if v < best - 1e-6:
            best, best_ep, wait = v, ep, 0
            best_state = {k: q.detach().clone() for k, q in r.state_dict().items()}
        else:
            wait += 1
            if wait >= 5:
                break
    info.update(selected_epoch=best_ep, best_val=best, epochs_run=ep, wall_s=time.time() - t0)
    return dict(model_params=ck['model_params'], state_dict=best_state), info


def pretrain_hse6(ck, D, test, seed, epochs, arm):
    """hse/pretrain.py logic on the release Roost model with BOTH views: formula-only (encoder + output net) and
    formula+SG (hidden features + zero-initialised linear SG pathway sgp, one row = one HSE structure with its space group);
    loss = MSE(view0) + MSE(view1) on the standardised log1p target; 5% random val, AdamW 3e-4, batch 256, patience 5, best state.
    NEGSG: rows with target < 0 dropped, zeros kept. Test-fold composition keys removed."""
    H = D['hse']
    ya = np.array(H['y'], dtype=np.float64)
    sgn = torch.load(f'{FT}/hse_sg.pt').long()
    tab = torch.load(f'{FT}/symtab.pt').to(dev)
    ok = [i for i, (k, f) in enumerate(zip(H['key'], H['formula'])) if k not in test and all(e in emb_keys for e in parse_roost(f)[0])]
    keep = [i for i in ok if ya[i] >= 0]
    yk = torch.tensor(ya[keep], dtype=torch.float32)
    info = dict(arm=arm, rows_total=len(ya), rows_kept=len(keep), rows_negative_dropped=int((ya[ok] < 0).sum()), rows_le001=int((yk <= 0.01).sum()))
    items = graphs([H['formula'][i] for i in keep], 'hse')
    lt = yk.log1p(); lm, ls = float(lt.mean()), float(lt.std())
    z = ((lt - lm) / ls).to(dev)
    sgk = sgn[keep].to(dev)
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(z), generator=g)
    nv = int(round(.05 * len(z)))
    vi, ti = perm[:nv], perm[nv:]
    torch.manual_seed(seed)
    r = Roost(**ck['model_params'], device=dev).to(dev)
    r.load_state_dict(ck['state_dict'])
    Lh = r.output_nn.fcs[0].in_features if len(r.output_nn.fcs) else r.output_nn.fc_out.in_features
    sgp = nn.Linear(tab.shape[1] + 230, Lh).to(dev)
    nn.init.zeros_(sgp.weight); nn.init.zeros_(sgp.bias)
    opt = torch.optim.AdamW(list(r.parameters()) + list(sgp.parameters()), lr=3e-4, weight_decay=1e-6)

    def lossf(b, ix):
        ixd = torch.as_tensor(ix, device=dev)
        s_ = sgk[ixd] - 1
        v = torch.cat([F.one_hot(s_, 230).float(), tab[s_]], 1)
        h = r.material_nn(*b)
        p0 = r.output_nn(h)[:, 0]; p1 = r.output_nn(h + sgp(v))[:, 0]
        return ((p0 - z[ixd]) ** 2).mean() + ((p1 - z[ixd]) ** 2).mean()
    vb = [vi[s:s + 2048].tolist() for s in range(0, len(vi), 2048)]
    vcache = [collate(items, ix) for ix in vb]

    def val():
        r.eval()
        tot = 0.
        with torch.no_grad():
            for b, ix in zip(vcache, vb):
                tot += float(lossf(b, ix)) * len(ix)
        return tot / len(vi)
    def snap():
        return {k: v.detach().clone() for k, v in r.state_dict().items()}, {k: v.detach().clone() for k, v in sgp.state_dict().items()}
    best, best_ep, wait = val(), 0, 0
    best_state, best_sgp = snap()
    t0 = time.time()
    for ep in range(1, epochs + 1):
        r.train()
        order = ti[torch.randperm(len(ti), generator=g)]
        for s in range(0, len(order), 256):
            ix = order[s:s + 256].tolist()
            opt.zero_grad(set_to_none=True)
            l = lossf(collate(items, ix), ix)
            l.backward(); opt.step()
        v = val()
        print(f'hse6 {arm} ep {ep} val {v:.4f} {time.time() - t0:.0f}s', flush=True)
        if v < best - 1e-6:
            best, best_ep, wait = v, ep, 0
            best_state, best_sgp = snap()
        else:
            wait += 1
            if wait >= 5:
                break
    info.update(selected_epoch=best_ep, best_val=best, epochs_run=ep, wall_s=time.time() - t0, sgp_wnorm=float(sum((q ** 2).sum() for q in best_sgp.values()).sqrt()))
    return dict(model_params=ck['model_params'], state_dict=best_state, sgp_state=best_sgp), info


def run(a):
    D = torch.load(f'{FT}/ft_data.pt', weights_only=False)
    t0 = time.time()
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed); torch.cuda.manual_seed_all(a.seed)
    if a.init == 'warm':
        ck = torch.load(f'{os.environ["ROOST_RUNS"]}/{a.variant}_f{a.fold}_s{a.seed}/models/cv_s-0_t-1/checkpoint-r{a.seed}.pth.tar', map_location='cpu')
    else:
        ck = torch.load(f'{W}/original_comparison/models/original_mp/checkpoint-r{a.mp_seed or a.seed}.pth.tar', map_location='cpu')
    test, tr, va, te, trB, teB = fold_sets(D, a.variant, a.fold)
    B = D['B']
    if a.val != 'inner':            # inner-val compositions go back into the pool; validation drawn from the pool
        tr = tr + va
        keys = sorted({r['key'] for r in tr} | {B['key'][i] for i in trB})
        if a.val == 'group20':
            rng = np.random.default_rng(20261113 + 7 * a.fold)
            vk = set(rng.choice(keys, size=int(round(.2 * len(keys))), replace=False).tolist())
        else:
            vk = set(json.load(open(f'{FT}/roostval_{a.variant}_f{a.fold}.json')))
        va = [r for r in tr if r['key'] in vk]
        tr = [r for r in tr if r['key'] not in vk]
        vaB = [i for i in trB if B['key'][i] in vk]
        trB = [i for i in trB if B['key'][i] not in vk]
    else:
        vaB = []
    info = dict(vars(a), A_train=len(tr), A_val=len(va), A_test=len(te), B_train_values=len(trB), B_val_values=len(vaB), B_test_comps=len(teB))
    if a.hse_arm:
        hp = f'{OUT}/hse_states/{a.variant}_f{a.fold}_s{a.seed}_{a.hse_arm}.pt'
        os.makedirs(f'{OUT}/hse_states', exist_ok=True)
        if not a.pretrain_only and a.hse_arm.endswith('SG'):
            while not os.path.exists(hp + '.json'):
                time.sleep(20)
            time.sleep(3)
        if os.path.exists(hp + '.json'):
            ck, info['hse'] = torch.load(hp, map_location='cpu'), json.load(open(hp + '.json'))
        else:
            ck, hinfo = (pretrain_hse6 if a.hse_arm.endswith('SG') else pretrain_hse4)(ck, D, test, a.seed, 30, a.hse_arm)
            torch.save(ck, hp); json.dump(hinfo, open(hp + '.json', 'w')); info['hse'] = hinfo
        if a.pretrain_only:
            return
        torch.manual_seed(a.seed)
    if a.hse:
        hp = f'{OUT}/hse_states/{a.variant}_f{a.fold}_s{a.seed if a.hse_own else 17}_e30.pt'
        ck = torch.load(hp, map_location='cpu'); info['hse'] = json.load(open(hp + '.json'))
        torch.manual_seed(a.seed)
    sg_all, sgm_all = D['sg'], D['sg_mask']
    SGB = torch.load(f'{FT}/sgB_pool.pt', weights_only=False) if a.sg_extra else {}

    def tensors(Ar, Bi):
        forms = [r['formula'] for r in Ar] + [B['formula'][i] for i in Bi]
        y = torch.tensor([r['y'] for r in Ar] + [B['y'][i] for i in Bi], dtype=torch.float32, device=dev)
        w = torch.tensor([r['wc'] for r in Ar] + [B['w'][i] for i in Bi], dtype=torch.float32, device=dev)
        sg = torch.zeros(len(forms), sg_all.shape[1], device=dev); sgm = torch.zeros(len(forms), device=dev)
        if Ar:
            sg[:len(Ar)] = sg_all[[r['idx'] for r in Ar]].to(dev); sgm[:len(Ar)] = sgm_all[[r['idx'] for r in Ar]].to(dev)
        if a.sg_extra:
            for j, i in enumerate(Bi):
                v_ = SGB.get(B['key'][i])
                if v_ is not None:
                    sg[len(Ar) + j] = v_.to(dev); sgm[len(Ar) + j] = 1.0
        if a.sg_train == 'off':
            sgm.zero_()
        tier = torch.cat([torch.zeros(len(Ar), dtype=torch.long), torch.ones(len(Bi), dtype=torch.long)]).to(dev)
        return forms, y, w, sg, sgm, tier
    forms, y, w, sg, sgm, tier = tensors(tr, trB)
    if a.a_mass != 1.0:
        w[:len(tr)] *= a.a_mass
    items = graphs(forms, 'train')
    vforms, vy, vw, vsg, vsgm, vtier = tensors(va, vaB)
    nvk = len({r['key'] for r in va}) + len({B['key'][i] for i in vaB} - {r['key'] for r in va}) if a.val != 'inner' else len({r['key'] for r in va})
    # equal mass per (tier, composition) unit in validation, like pooled evaluation
    nunit = len({r['key'] for r in va}) + len({B['key'][i] for i in vaB})
    vw = vw / nunit
    vitems = graphs(vforms, 'val')
    vbs = [list(range(s, min(s + 1024, len(vforms)))) for s in range(0, len(vforms), 1024)]
    vcache = [collate(vitems, ix) for ix in vbs]
    iv_dim = 0 if not a.s2_iv else (sg_all.shape[1] if a.s2_iv == 'full' else 8)
    model = GSFT2(ck, sg_all.shape[1], a.comp, a.tierflag, a.sg_rank, iv_dim).to(dev)
    if 'sgp_state' in ck:
        model.sgp.load_state_dict(ck['sgp_state'])
        info['sgp_pretrained_wnorm'] = float(sum((q ** 2).sum() for q in model.sgp.parameters()).sqrt())
    model.init_head(y.cpu())
    import copy
    enc0 = copy.deepcopy(model.enc).eval()
    for q_ in enc0.parameters():
        q_.requires_grad_(False)
    model.sdmax = a.sdmax

    CG = torch.tensor(np.geomspace(0.4, 30.0, 41), dtype=torch.float32, device=dev)

    def val_all(cal=True, nosg=False):
        """weighted validation diagnostics: nll, crps, mae of the location (exp of weighted mu), mean width, and the same
        nll/crps after refitting one width factor c on the validation set (c minimises val nll)."""
        model.eval()
        P, Y, Wt = [], [], []
        with torch.no_grad():
            for b, ix in zip(vcache, vbs):
                ixd = torch.tensor(ix, device=dev)
                P.append([t for t in model(b, vsg[ixd], (vsgm[ixd] * 0 if nosg else vsgm[ixd]), vtier[ixd])]); Y.append(vy[ixd]); Wt.append(vw[ixd])
            lw = torch.cat([q[0] for q in P]); mu = torch.cat([q[1] for q in P]); sd = torch.cat([q[2] for q in P])
            Yv = torch.cat(Y); Wv = torch.cat(Wt); Wn = Wv / Wv.sum()
            p = (lw, mu, sd)
            if a.comp == 'ln':
                n_, c_ = nll(p, Yv), crps(p, Yv)
                loc = (lw.exp() * mu).sum(1).exp()
            else:
                n_ = lap_nll(p, Yv); c_ = n_ * 0
                loc = (lw.exp() * mu).sum(1)
            if a.sel == 'crps' and a.comp != 'ln':
                xs = torch.linspace(-3, 16, 1901, device=dev)
                zz = (xs[None, :, None] - mu[:, None, :]) / sd[:, None, :]
                Fx = (lw.exp()[:, None, :] * torch.where(zz < 0, 0.5 * torch.exp(zz.clamp(max=0)), 1 - 0.5 * torch.exp(-zz.clamp(min=0)))).sum(2)
                c_ = ((Fx - (xs[None] >= Yv[:, None]).float()) ** 2).sum(1) * (xs[1] - xs[0])
            r = dict(nll=float((n_ * Wv).sum()), crps=float((c_ * Wv).sum()), mae=float(((loc - Yv).abs() * Wn).sum()),
                     width=float(((lw.exp() * sd).sum(1) * Wn).sum()))
            if cal and a.comp == 'ln':
                best = None
                for c in CG:
                    pc = (lw, mu, sd * c)
                    nc = float((nll(pc, Yv) * Wv).sum())
                    if best is None or nc < best[0]:
                        best = (nc, float(c), pc)
                r['c'] = best[1]; pcd = tuple(t.double() for t in best[2]); cvv = crps(pcd, Yv.double()).clamp(min=0); r['crps_cal'] = float((cvv * Wv.double()).sum())
                nA_ = len(va); r['crps_cal_A'] = float((cvv[:nA_] * Wv[:nA_].double()).sum() / Wv[:nA_].double().sum())
                r['nll_A'] = float((nll(best[2], Yv)[:nA_] * Wv[:nA_]).sum() / Wv[:nA_].sum())
            else:
                r['c'] = 1.0; r['crps_cal'] = r['crps']
        return r

    def val():
        r = val_all(cal=(a.sel == 'selcal'))
        vlog[0] = r
        if a.sel == 'selcal':
            return r['crps_cal']
        return r['mae'] if a.sel == 'mae' else (r['crps'] if a.sel == 'crps' else r['nll'])
    vlog = [None]

    def sgstats():
        model.eval()
        with torch.no_grad():
            idx = (sgm > 0).nonzero().flatten()[:300]
            if len(idx) == 0:
                return {}
            b = collate(items, idx.tolist()); h = model.enc(*b)
            sgp_out = model.sgp(sg[idx]) if not model.iv else torch.zeros_like(h)
            p1 = model(b, sg[idx], sgm[idx], tier[idx]); p0 = model(b, sg[idx], sgm[idx] * 0, tier[idx])
            ww = w[idx]
            return dict(sg_wnorm=float(sum((q ** 2).sum() for q in sg_params).sqrt()), sg_rel=float((sgp_out.norm(dim=1) / h.norm(dim=1)).mean()),
                        n_sg_train_rows=len(idx), tr_sg_nll_with=float((nll(p1, y[idx]) * ww).sum() / ww.sum()), tr_sg_nll_without=float((nll(p0, y[idx]) * ww).sum() / ww.sum()))
    hist2 = []
    stt = {}
    g = torch.Generator().manual_seed(a.seed)
    hist = []

    def loop(opt, epochs, patience, stage):
        best, best_ep, wait = val(), 0, 0
        best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        ema = {k: v.detach().clone().float() for k, v in model.state_dict().items()} if a.ema > 0 else None
        for ep in range(1, epochs + 1):
            model.train()
            order = pool_ix[torch.randperm(len(pool_ix), generator=g)]
            tl = 0.; tmae = 0.; twt = 0.
            for s in range(0, len(order), a.batch):
                ix = order[s:s + a.batch]
                ixd = ix.to(dev)
                if a.sg_train == 'detach2v':
                    p, p1 = model.forward2(collate(items, ix.tolist()), sg[ixd], sgm[ixd], tier[ixd])
                    loss = (loss_fn(model, p, y[ixd], a) * w[ixd]).sum() / w[ixd].sum() + (loss_fn(model, p1, y[ixd], a) * w[ixd] * sgm[ixd]).sum() / w[ixd].sum()
                else:
                    p = model(collate(items, ix.tolist()), sg[ixd], sgm[ixd], tier[ixd])
                    loss = (loss_fn(model, p, y[ixd], a) * w[ixd]).sum() / w[ixd].sum()
                if a.sg_l2 > 0:
                    loss = loss + a.sg_l2 * sum((q ** 2).sum() for q in sg_params)
                opt.zero_grad(set_to_none=True)
                loss.backward(); opt.step()
                tl += float(loss) * len(ix)
                with torch.no_grad():
                    loc_ = (p[0].exp() * p[1]).sum(1).exp() if model.comp == 'ln' else (p[0].exp() * p[1]).sum(1)
                    tmae += float(((loc_ - y[ixd]).abs() * w[ixd]).sum()); twt += float(w[ixd].sum())
                if ema is not None:      # EMA of weights (floating tensors), selection and output use the EMA weights
                    with torch.no_grad():
                        for k, q in model.state_dict().items():
                            if q.dtype.is_floating_point:
                                ema[k].mul_(a.ema).add_(q.float(), alpha=1 - a.ema)
                            else:
                                ema[k].copy_(q)
            if ema is not None:
                raw = {k: q.detach().clone() for k, q in model.state_dict().items()}
                model.load_state_dict({k: ema[k].to(raw[k].dtype) for k in raw})
            v = val()
            hist.append((stage, ep, tl / len(order), v))
            if not a.nodiag:
                rr = vlog[0] if a.sel == 'selcal' else val_all(cal=False)
                hist2.append(dict(stage=stage, ep=ep, train_loss=tl / len(order), train_mae=tmae / twt, **rr))
            if a.sgdiag:
                hist2.append(dict(stage=stage, ep=ep, **sgstats(), v_with=val_all(False)['crps'], v_without=val_all(False, True)['crps'],
                                  v_with_nll=val_all(False)['nll'], v_without_nll=val_all(False, True)['nll']))
            stt['last'] = {k: q.detach().clone() for k, q in model.state_dict().items()}
            if v < best - 1e-6:
                best, best_ep, wait = v, ep, 0
                best_state = {k: q.detach().clone() for k, q in model.state_dict().items()}
            if ema is not None:
                model.load_state_dict(raw)
            else:
                wait += 1
                if wait >= patience:
                    break
            if ep % 10 == 0 or ep == 1:
                print(f'{stage} ep {ep} train {tl / len(order):.4f} val {v:.4f} best {best:.4f}@{best_ep} {time.time() - t0:.0f}s', flush=True)
        model.load_state_dict(best_state)
        stt['best'] = best_state
        return dict(selected_epoch=best_ep, epochs_run=ep, best_val=best)
    enc_ids = {id(p) for p in model.enc.parameters()}
    sg_params = list(model.sgs.parameters()) if model.iv else list(model.sgp.parameters())
    sg_ids = {id(p) for p in sg_params}
    pool_ix = torch.arange(len(forms))
    head = [p for p in model.parameters() if id(p) not in enc_ids and id(p) not in sg_ids]
    if a.stage2:       # frozen formula-only network (saved base arm), fit only the SG module on SG-labelled training rows
        ckp = f'{OUT}/ckpt/{a.variant}_{a.base_arm}_s{a.seed}_f{a.fold}_best.pt'
        while not os.path.exists(ckp):
            time.sleep(20)
        time.sleep(3)
        sd_ = torch.load(ckp, map_location=dev)
        model.load_state_dict({k: v for k, v in sd_.items() if not (k.startswith('sgp') or k.startswith('sgs') or k == 'Pm')}, strict=False)
        for q in model.parameters():
            q.requires_grad_(False)
        for q in sg_params:
            q.requires_grad_(True)
        pool_ix = (sgm.cpu() > 0).nonzero().flatten()
        info['n_stage2_rows'] = len(pool_ix)
    if a.stage1 > 0:
        model.enc.requires_grad_(False)
        info['stage1'] = loop(torch.optim.AdamW(head, lr=a.lr, weight_decay=a.wd), a.stage1, a.patience, 'head')
        model.enc.requires_grad_(True)
        lr2 = a.lr * a.stage2_mult
        opt = torch.optim.AdamW(model.parameters(), lr=lr2, weight_decay=a.wd)
    elif a.stage2:
        opt = torch.optim.AdamW(sg_params, lr=a.lr, weight_decay=0.0)
    else:
        opt = torch.optim.AdamW([dict(params=list(model.enc.parameters()), lr=a.lr * a.enc_lr_mult), dict(params=head, lr=a.lr),
                                 dict(params=sg_params, lr=a.lr * a.sg_lr_mult)], weight_decay=a.wd)
    if a.fixed_epochs > 0:      # fixed epoch count, last state
        n0 = a.patience
        a.patience = 10 ** 9
        info['stage2'] = loop(opt, a.fixed_epochs, a.patience, 'full')
    else:
        info['stage2'] = loop(opt, a.epochs, a.patience, 'full')
    info['final_val'] = val_all(cal=True)
    with torch.no_grad():          # how far the encoder moved: relative weight distance and linear CKA of embeddings vs init
        n0 = sum(float((q ** 2).sum()) for q in enc0.parameters()) ** .5
        dd = sum(float(((q - q0) ** 2).sum()) for q, q0 in zip(model.enc.parameters(), enc0.parameters())) ** .5
        info['enc_rel_dist'] = dd / n0
        model.eval()
        rg = torch.Generator().manual_seed(0)
        sel_ = torch.randperm(len(forms), generator=rg)[:1500].tolist()
        X, Y = [], []
        for s_ in range(0, len(sel_), 500):
            bb = collate(items, sel_[s_:s_ + 500])
            X.append(model.enc(*bb).double()); Y.append(enc0(*bb).double())
        X = torch.cat(X); Y = torch.cat(Y); X = X - X.mean(0); Y = Y - Y.mean(0)
        info['enc_cka'] = float(((X.T @ Y).norm() ** 2) / ((X.T @ X).norm() * (Y.T @ Y).norm()).clamp_min(1e-12))
    info['val_sg'] = info['final_val']; info['val_nosg'] = val_all(cal=True, nosg=True)
    info['sg_final_wnorm'] = float(sum((q ** 2).sum() for q in sg_params).sqrt())
    if a.interp:
        info['interp'] = []
        for al in np.linspace(0, 1, 11):
            model.load_state_dict({k: ((1 - al) * stt['best'][k] + al * stt['last'][k]).to(stt['best'][k].dtype) if stt['best'][k].dtype.is_floating_point else stt['best'][k] for k in stt['best']})
            info['interp'].append(dict(alpha=float(al), **val_all(cal=True)))
        model.load_state_dict(stt['best'])
    if a.save_ckpt:
        os.makedirs(f'{OUT}/ckpt', exist_ok=True)
        torch.save(stt['best'], f'{OUT}/ckpt/{a.variant}_{a.arm}_s{a.seed}_f{a.fold}_best.pt')
        torch.save(stt['last'], f'{OUT}/ckpt/{a.variant}_{a.arm}_s{a.seed}_f{a.fold}_last.pt')
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
        tA = torch.zeros(len(te), dtype=torch.long, device=dev)
        v0 = comp(model(tb, tsg, torch.zeros_like(tsgm), tA))
        v1 = comp(model(tb, tsg, tsgm, tA))
        out.update(A0=v0, A1=v1, A2=v0, A3=v1, A_idx=np.array([r['idx'] for r in te]), kind=np.array(a.comp))
        fB = dict(zip(D['Bkeys'], D['Bformulas']))
        bitems = graphs([fB[k] for k in teB], 'testB')
        def predB(ix):
            n_ = len(ix)
            return model(collate(bitems, ix), torch.zeros(n_, sg_all.shape[1], device=dev), torch.zeros(n_, device=dev), torch.ones(n_, dtype=torch.long, device=dev))
        out['B'] = np.concatenate([comp(predB(list(range(s, min(s + 1024, len(teB)))))) for s in range(0, len(teB), 1024)], 1)
        out['B_keys'] = np.array(teB)
    os.makedirs(f'{OUT}/gs_out_v2', exist_ok=True)
    tag = f'{a.variant}_{a.arm}_seed{a.seed}_fold{a.fold}'
    np.savez_compressed(f'{OUT}/gs_out_v2/{tag}.npz', **out)
    info.update(wall_s=time.time() - t0, history=hist, history2=hist2)
    json.dump(info, open(f'{OUT}/gs_out_v2/{tag}_info.json', 'w'))
    print('done', tag, info['stage2'], f'{time.time() - t0:.0f}s', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--variant', default='comp'); p.add_argument('--fold', type=int, required=True); p.add_argument('--seed', type=int, required=True)
    p.add_argument('--arm', required=True); p.add_argument('--init', default='mp', choices=['mp', 'warm']); p.add_argument('--hse', action='store_true')
    p.add_argument('--val', default='inner', choices=['inner', 'group20', 'roostval']); p.add_argument('--sel', default='nll', choices=['nll', 'crps', 'selcal', 'mae'])
    p.add_argument('--comp', default='ln', choices=['ln', 'lap'])
    p.add_argument('--lr', type=float, default=3e-4); p.add_argument('--enc_lr_mult', type=float, default=1.0)
    p.add_argument('--stage1', type=int, default=0); p.add_argument('--stage2_mult', type=float, default=0.1)
    p.add_argument('--epochs', type=int, default=300); p.add_argument('--patience', type=int, default=40); p.add_argument('--batch', type=int, default=128)
    p.add_argument('--fixed_epochs', type=int, default=0)
    p.add_argument('--crps', type=float, default=0.0); p.add_argument('--l1', type=float, default=0.0); p.add_argument('--a_mass', type=float, default=1.0)
    p.add_argument('--wd', type=float, default=1e-6); p.add_argument('--ema', type=float, default=0.0)
    p.add_argument('--hse_own', action='store_true', help='per-seed own HSE-pretrained encoder state'); p.add_argument('--beta', type=float, default=0.0)
    p.add_argument('--interp', action='store_true'); p.add_argument('--save_ckpt', action='store_true'); p.add_argument('--nodiag', action='store_true')
    p.add_argument('--hse_arm', default='', choices=['', 'NEG', 'NZ', 'MT', 'NEGSG', 'T025', 'T05', 'T10'])
    p.add_argument('--pretrain_only', action='store_true')
    p.add_argument('--sg_train', default='joint', choices=['joint', 'off', 'detach2v']); p.add_argument('--sg_l2', type=float, default=0.0)
    p.add_argument('--sg_lr_mult', type=float, default=1.0); p.add_argument('--sg_rank', type=int, default=0); p.add_argument('--sg_extra', action='store_true')
    p.add_argument('--sgdiag', action='store_true'); p.add_argument('--stage2', action='store_true'); p.add_argument('--base_arm', default='GS_F0')
    p.add_argument('--s2_iv', default='', choices=['', 'full', 'pca8'])
    p.add_argument('--tierflag', action='store_true'); p.add_argument('--mp_seed', type=int, default=0)
    p.add_argument('--threads', type=int, default=2); p.add_argument('--sdmax', type=float, default=0.0, help='cap on lognormal component sd (0 = none)')
    a = p.parse_args()
    torch.set_num_threads(a.threads)
    run(a)
