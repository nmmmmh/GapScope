

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
    keep = [i for i in ok if ya[i] >= 0] if arm in ('NEG', 'MT') else [i for i in ok if ya[i] > 0.01]
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

