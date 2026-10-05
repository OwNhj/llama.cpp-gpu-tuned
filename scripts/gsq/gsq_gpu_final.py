#!/usr/bin/env python3
"""GPU full-model in-place GSQ-style refinement for a production MXFP4 gguf.

Baseline codes = stored mix145 bytes (C++ grid search, unit-verified J-optimal).
Objective: minimize  tr(E^T H E) + lam * ||E||F^2  by reassigning E2M1 codes only.
H normalized per tensor (mean diag = 1) so lam is comparable across tensors.
Exponents untouched -> type 39 format and radiance fast path unaffected.

usage: gsq_gpu.py <lam> [rounds] [max_tensor_rows] [only_file]
Resumable via <out>.progress.json. Output = copy of PROD with refined bytes."""
import sys, os, json, time, shutil
import numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from gsq_core import Ggu, load_bf16, load_h, MAG
from gsq_pack import parse_blocks, pack_blocks

SRC = os.environ["GSQ_SRC"]
PROD = os.environ["GSQ_PROD"]
HD = os.environ.get("GSQ_HD", os.path.join(HERE, "hess"))
OUT = os.environ.get("GSQ_OUT", os.path.join(HERE, "gsq-refined.gguf"))

CV = torch.tensor(np.concatenate([MAG, -MAG]).astype(np.float64), device="cuda")


def dequant_gpu(codes, exps, K):
    nb = K // 32
    if torch.is_tensor(codes):
        codes = codes.cpu().numpy()
    N = codes.shape[0]
    mag = MAG[(codes & 7)].reshape(N, nb, 32)
    sgn = np.where((codes >> 3) & 1, -1.0, 1.0).reshape(N, nb, 32)
    sc = np.power(2.0, exps.astype(np.float64) - 127.0)[:, :, None]
    return torch.as_tensor((mag * sgn * sc).reshape(N, K)).double().cuda()


def cd_gpu(W, Hn, codes, exps, lam, rounds=2, seed=0):
    N, K = W.shape
    rng = np.random.default_rng(seed)
    Q = dequant_gpu(codes, exps, K)
    e = W - Q
    EH = e @ Hn
    sc_all = torch.pow(2.0, torch.as_tensor(exps.astype(np.int64), device="cuda") - 127.0).double()
    hj = torch.diagonal(Hn)
    cd = torch.as_tensor(codes, device="cuda").long().clone()
    ar = torch.arange(N, device="cuda")
    flips = 0
    for r in range(rounds):
        for j in rng.permutation(K):
            jj = int(j)
            sc = sc_all[:, jj // 32]
            cand = W[:, jj].reshape(-1, 1) - CV.reshape(1, 16) * sc.reshape(-1, 1)
            delta = cand - e[:, jj].reshape(-1, 1)
            dtr = 2.0 * delta * EH[:, jj].reshape(-1, 1) + delta * delta * hj[jj]
            dfro = cand * cand - (e[:, jj].reshape(-1, 1) ** 2)
            score = dtr + lam * dfro
            bs, bc = score.min(1)
            upd = bs < -1e-12
            nu = int(upd.sum())
            if nu == 0:
                continue
            uu = upd.nonzero(as_tuple=True)[0]
            bcu = bc[uu].long()
            cd[uu, jj] = bcu
            e[uu, jj] = cand[uu, bcu]
            d = delta[ar, bc]
            d[~upd] = 0.0
            EH[uu] += d[uu].reshape(-1, 1) * Hn[jj:jj + 1, :]
            flips += nu
        L = float((e * EH).sum()); F = float((e * e).sum())
    return cd.cpu().numpy(), L, F, flips


def main():
    lam = float(sys.argv[1]) if len(sys.argv) > 1 else 100.0
    rounds = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    only_file = sys.argv[3] if len(sys.argv) > 3 else None
    only = set(l.strip() for l in open(only_file)) if only_file else None
    if not os.path.exists(OUT):
        shutil.copyfile(PROD, OUT)
    pf = OUT + ".progress.json"
    prog = json.load(open(pf)) if os.path.exists(pf) else {}
    g_src = Ggu(SRC); g_prod = Ggu(PROD)
    f = open(OUT, 'r+b')
    g_out = Ggu(OUT)
    t_all = time.time()
    for name in sorted(g_prod.tensors):
        dims, dt, off = g_prod.tensors[name]
        if dt != 39 or name in prog:
            continue
        if only is not None and name not in only:
            continue
        hp = "%s/%s.bin" % (HD, name)
        if not os.path.exists(hp):
            continue
        ne0, ne1 = int(dims[0]), int(dims[1])
        if ne0 % 32 or ne1 * ne0 * 4 > 6_000_000_000:
            continue
        W_np = load_bf16(g_src, name)
        codes, exps = parse_blocks(
            bytes(g_prod.mm[g_prod.base + off: g_prod.base + off + ne1 * (ne0 // 32) * 17]), ne0, ne1)
        H = load_h(hp, ne0)
        mn = float(np.diagonal(H).mean())
        W = torch.as_tensor(W_np).double().cuda()
        Hn = torch.as_tensor((H / mn)).double().cuda()
        del H
        E0 = W - dequant_gpu(codes, exps, ne0)
        L0 = float((E0 * (E0 @ Hn)).sum()); F0 = float((E0 * E0).sum())
        t0 = time.time()
        cd, L1, F1, fl = cd_gpu(W, Hn, codes, exps, lam, rounds=rounds)
        del W, Hn, E0
        torch.cuda.empty_cache()
        newb = pack_blocks(cd, exps, ne0, ne1)
        f.seek(g_out.base + off); f.write(newb); f.flush()
        prog[name] = dict(tr=L1 / L0, fro=F1 / F0, flips=fl, secs=time.time() - t0)
        json.dump(prog, open(pf, 'w'))
        print("[%3d] %-38s trEHE %+7.2f%% fro %+6.2f%% flips=%9d (%.0fs, tot %.0fmin)" % (
            len(prog), name, 100 * (L1 / L0 - 1), 100 * (F1 / F0 - 1), fl,
            time.time() - t0, (time.time() - t_all) / 60), flush=True)
    f.close()
    tot_tr = sum(v["tr"] for v in prog.values()); n = len(prog)
    print("DONE %d tensors, mean trEHE ratio %.4f, %.0f min" % (n, tot_tr / max(n, 1), (time.time() - t_all) / 60))


main()
