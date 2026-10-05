#!/usr/bin/env python3
"""GSQ-MXFP4: Gumbel-Softmax refinement of MXFP4 (GGML type 39) assignments.

Format invariant: E2M1 codes + one E8M0 exponent per 32-block (17B/block) ->
radiance fast path / repack untouched. Only the discrete assignment (and later
block exponents) are optimised, against the TRUE layer objective tr(E^T H E)
with H from llama-imatrix --hessian-dir (real calibration activations).

GSQ mechanics (arXiv 2604.18556), adapted:
  - per-element logits over a local magnitude window around the grid-search init
    (their n-bit scheme: init + {-2..+2});
  - Gumbel-Softmax relaxation, tau annealed high->low, logit scale kappa low->high;
  - Lion optimizer, cosine LR; hard-round at the end;
  - per-block E8M0 exponent: keep from init here (joint exponent re-search is a
    separate step, see refine_exponents).
"""
import argparse, os, struct, sys, time, mmap
import numpy as np
import torch
import torch.nn.functional as F

DEV = "cuda" if torch.cuda.is_available() else "cpu"
MAG = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=np.float32)  # E2M1 magnitudes
QK = 32

# ---------------- GGUF helpers ----------------
SSZ = {0:1,1:1,2:2,3:2,4:4,5:4,6:4,7:1,10:8,11:8,12:8}
SC  = {0:'B',1:'b',2:'H',3:'h',4:'I',5:'i',6:'f',7:'?',10:'Q',11:'q',12:'d'}

class Ggu:
    def __init__(self, path):
        self.fh = open(path, 'rb')
        self.mm = mmap.mmap(self.fh.fileno(), 0, prot=mmap.PROT_READ)
        b = self.mm
        assert b[:4] == b'GGUF', path
        nten = struct.unpack_from('<Q', b, 8)[0]
        nkv = struct.unpack_from('<Q', b, 16)[0]
        o = 24
        def rstr(o):
            l = struct.unpack_from('<Q', b, o)[0]; return b[o+8:o+8+l], o+8+l
        align = 32
        for _ in range(nkv):
            k, o = rstr(o); t = struct.unpack_from('<I', b, o)[0]; o += 4
            if t == 8:
                _, o = rstr(o)
            elif t == 9:
                et = struct.unpack_from('<I', b, o)[0]; n = struct.unpack_from('<Q', b, o+4)[0]; o += 12
                if et == 8:
                    for _ in range(n): _, o = rstr(o)
                else:
                    o += n * SSZ[et]
            else:
                v = struct.unpack_from('<' + SC[t], b, o)[0]; o += SSZ[t]
                if k.decode() == 'general.alignment':
                    align = int(v)
        self.tensors = {}
        for _ in range(nten):
            name, o = rstr(o)
            nd = struct.unpack_from('<I', b, o)[0]; o += 4
            dims = list(struct.unpack_from('<%dQ' % nd, b, o)); o += 8 * nd
            dt = struct.unpack_from('<I', b, o)[0]; o += 4
            off = struct.unpack_from('<Q', b, o)[0]; o += 8
            self.tensors[name.decode()] = (dims, dt, off)
        self.base = (o + align - 1) // align * align

def rowsz(dt, dims):
    n = int(np.prod(dims[1:])) if len(dims) > 1 else 1
    ne = dims[0]
    if dt == 30: return n * ne * 2
    if dt == 0:  return n * ne * 4
    if dt == 39: return n * (ne // 32) * 17
    raise ValueError(dt)

def load_bf16(g, name):
    dims, dt, off = g.tensors[name]
    ne0, ne1 = dims[0], dims[1]
    a = np.frombuffer(g.mm, dtype=np.uint16, count=ne0 * ne1, offset=g.base + off)
    return ((a.astype(np.uint32) << 16).view(np.float32)).reshape(ne1, ne0)

def load_h(fp, K):
    with open(fp, 'rb') as f:
        k, ntok = struct.unpack('qq', f.read(16))
        d = np.frombuffer(f.read(), dtype=np.float32)
    H = np.zeros((K, K), dtype=np.float32)
    pos = 0
    for t in range(K):
        H[t, :t+1] = d[pos:pos+t+1]; pos += t+1
    H = H + np.tril(H, -1).T   # packed rows are the LOWER triangle; mirror to upper
    return H

# ---------------- baseline MXFP4 grid search (matches C++) ----------------
def grid_search(Wx, im, device=DEV):
    """Wx (N,K) f32 tensor, im (K,) f32 -> codes (N,K) int64 0..15, exps (N,nb) uint8.
    Nearest is over MAGNITUDES with the sign applied afterwards: matching signed x
    against the positive-only MAG table would quantize every negative to zero."""
    N, K = Wx.shape; nb = K // 32
    x = Wx.reshape(N, nb, 32)
    ax = x.abs()
    w = torch.as_tensor(im.copy()).float().to(device).reshape(1, nb, 32) if im.shape[0] == K else None
    mags = torch.tensor(MAG, device=device).view(1, 1, 1, 8)
    amax = ax.amax(-1)
    e0 = (amax.clamp_min(1e-30).log2().floor() - 2 + 127)
    best = torch.full(x.shape[:2], float('inf'), device=device); be = e0.long()
    for d in range(-6, 2):
        ei = (e0 + d).long()
        ok = (ei >= 0) & (ei <= 254)
        sc = torch.pow(2.0, (ei - 127).float()).unsqueeze(-1)
        mi = (ax.unsqueeze(-1) / sc.unsqueeze(-1) - mags).abs().argmin(-1)
        q = mags.view(1, 1, 1, 8)[0, 0, 0][mi] * sc
        sse = (((ax - q) ** 2) * (w if w is not None else 1.0)).sum(-1)
        upd = ok & (sse < best)
        best = torch.where(upd, sse, best); be = torch.where(upd, ei, be)
    sc = torch.pow(2.0, (be - 127).float()).unsqueeze(-1)
    mi = (ax.unsqueeze(-1) / sc.unsqueeze(-1) - mags).abs().argmin(-1)
    sgn = (x < 0).long() * 8
    codes = (mi + sgn).reshape(N, K)
    return codes.cpu().numpy().astype(np.int64), be.cpu().numpy().astype(np.uint8)

def dequant(codes, exps, ne0, ne1):
    nb = ne0 // 32
    mag = MAG[codes & 7].reshape(ne1, nb, 32).astype(np.float64)
    sgn = np.where((codes >> 3) & 1, -1.0, 1.0).reshape(ne1, nb, 32)
    scale = np.power(2.0, exps.astype(np.float64) - 127.0)[:, :, None]
    return (mag * sgn * scale).reshape(ne1, ne0).astype(np.float32)

def trEHE(E, H):
    # sum_n e_n^T H e_n, chunked; E numpy (N,K), H numpy (K,K)
    N = E.shape[0]; tot = 0.0
    H64 = H.astype(np.float64)
    for s in range(0, N, 512):
        Eb = E[s:s+512].astype(np.float64)
        tot += float((Eb @ H64 * Eb).sum())
    return tot

# ---------------- GSQ refinement ----------------
def gsq_refine(W, H, im, codes0, exps0, steps=200, lr=0.08, tau0=2.0, tau1=0.05,
               kappa0=0.5, kappa1=4.0, window=2, batch=2048, seed=0, verbose=False):
    N, K = W.shape; nb = K // 32
    torch.manual_seed(seed)
    Wt = torch.as_tensor(W).to(DEV)
    Ht = torch.as_tensor(H).to(DEV)
    mag_t = torch.tensor(MAG, device=DEV)
    mi0 = torch.as_tensor(codes0 & 7, device=DEV)                     # (N,K) init magnitude idx
    sgn0 = torch.as_tensor((codes0 >> 3) & 1, device=DEV).bool()
    off = torch.arange(-window, window + 1, device=DEV)               # (C,)
    C = 2 * window + 1
    cand_mi = (mi0.unsqueeze(-1) + off.view(1, 1, C)).clamp(0, 7)     # (N,K,C) magnitude idx
    cand_mag = mag_t[cand_mi]                                          # (N,K,C)
    scale = torch.as_tensor(np.power(2.0, exps0.astype(np.float64) - 127.0).astype(np.float32)).to(DEV)
    sc = scale.repeat_interleave(32, dim=1)                            # (N,K)
    logits = torch.zeros(N, K, C, device=DEV)
    logits[:, :, window] = 2.0                                         # init: center=nearest
    logits = logits.detach().requires_grad_(True)

    def softq(p=None, tau=None, kap=None):
        lg = logits if p is None else logits[p]
        cm = cand_mag if p is None else cand_mag[p]
        s  = sc if p is None else sc[p]
        u = torch.rand_like(lg)
        g = torch.log(u + 1e-20)
        y = F.softmax((kap * lg + g) / tau, dim=-1)
        return (y * cm).sum(-1) * s

    def hard_q():
        idx = logits.argmax(-1)
        mi = torch.gather(mi0.unsqueeze(-1).expand(-1, -1, C), -1, idx.unsqueeze(-1)).squeeze(-1)
        q = mag_t[mi]
        q = torch.where(sgn0, -q, q)
        return q * sc

    def loss_of(q, rows=None):
        E = (Wt if rows is None else Wt[rows]) - (q if rows is None else q)
        return (E * (E @ Ht)).sum()

    try:
        opt = torch.optim.Lion([logits], lr=lr)
    except Exception:
        opt = torch.optim.Adam([logits], lr=lr)
    Hdev = Ht
    with torch.no_grad():
        base = loss_of(hard_q()).item()
    init_codes_q = hard_q()
    t0 = time.time()
    for s in range(steps):
        f = s / max(steps - 1, 1)
        tau = tau0 * (tau1 / tau0) ** f
        kap = kappa0 * (kappa1 / kappa0) ** f
        lrcur = lr * 0.5 * (1 + np.cos(np.pi * f))
        for gp in opt.param_groups: gp['lr'] = lrcur
        rows = torch.randperm(N, device=DEV)[:batch]
        q = softq(p=rows, tau=tau, kap=kap)
        E = Wt[rows] - q
        L = (E * (E @ Hdev)).sum()
        opt.zero_grad(); L.backward(); opt.step()
        if verbose and (s % 50 == 0 or s == steps - 1):
            with torch.no_grad():
                cur = loss_of(hard_q()).item()
            print("    step %3d tau %.3f L=%.4e hard trEHE=%.5e (%+.2f%% vs init)" %
                  (s, tau, L.item(), cur, 100 * (cur / base - 1)), flush=True)
    with torch.no_grad():
        idx = logits.argmax(-1)
        mi = torch.gather(mi0.unsqueeze(-1).expand(-1, -1, C), -1, idx.unsqueeze(-1)).squeeze(-1)
        codes_new = ((mi + sgn0.long() * 8)).cpu().numpy().astype(np.int64)
        final = loss_of(hard_q()).item()
    return codes_new, base, final, time.time() - t0
