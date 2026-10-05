#!/usr/bin/env python3
"""In-place MXFP4 code refinement packer.

Pipeline: copy B145 gguf -> for each type-39 tensor with an available Hessian,
run exact greedy CD over code assignments (fixed per-block exponents read from
the stored file) -> rewrite the 17-byte blocks in place. Header/metadata and all
other tensors untouched; format stays byte-type 39 so radiance is unaffected.

Self-test mode: unpack+repack stored blocks and assert zero byte diff.
"""
import sys, time, shutil, struct
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gsq_core import Ggu, load_bf16, load_h, dequant, MAG

CV = np.concatenate([MAG, -MAG]).astype(np.float64)


def parse_blocks(raw, ne0, ne1):
    nb = ne0 // 32
    b = np.frombuffer(raw, dtype=np.uint8)[:ne1 * nb * 17].reshape(ne1, nb, 17)
    exps = b[:, :, 0].copy()
    qs = b[:, :, 1:17].astype(np.int64)
    codes = np.concatenate([qs & 0xF, qs >> 4], axis=2)   # (ne1,nb,32) elem j / j+16
    return codes.reshape(ne1, ne0), exps


def pack_blocks(codes, exps, ne0, ne1):
    nb = ne0 // 32
    c = codes.reshape(ne1, nb, 32).astype(np.uint8)
    lo = c[:, :, :16]; hi = c[:, :, 16:]
    qs = lo | (hi << 4)
    out = np.empty((ne1, nb, 17), dtype=np.uint8)
    out[:, :, 0] = exps
    out[:, :, 1:] = qs
    return out.reshape(-1).tobytes()


def cd_refine(Ws, H, codes, exps, rounds=2, seed=0):
    import os
    os.environ.setdefault("GSQ_CPU", "1")
    B, K = Ws.shape
    rng = np.random.default_rng(seed)
    H64 = H.astype(np.float64)
    hj = np.diagonal(H64)
    sc_all = np.power(2.0, exps.astype(np.int64) - 127)
    Q = dequant(codes, exps, K, B).astype(np.float64)
    e = Ws.astype(np.float64) - Q
    EH = e @ H64
    cd = codes.copy()
    L = float((e * EH).sum())
    for r in range(rounds):
        flips = 0
        for j in rng.permutation(K):
            sc = sc_all[:, j // 32]
            newe = Ws[:, j].astype(np.float64).reshape(-1, 1) - CV.reshape(1, 16) * sc.reshape(-1, 1)
            delta = newe - e[:, j].reshape(-1, 1)
            gain = -(2.0 * delta * EH[:, j].reshape(-1, 1) + delta * delta * hj[j])
            bg = gain.max(1); bc = gain.argmax(1)
            upd = bg > 1e-30
            if not upd.any():
                continue
            cd[upd, j] = bc[upd]
            e[upd, j] = newe[upd, bc[upd]]
            d = delta[np.arange(B), bc]; d[~upd] = 0.0
            EH[upd] += d[upd].reshape(-1, 1) * H64[j].reshape(1, -1)
            flips += int(upd.sum())
        L = float((e * EH).sum())
    return cd, L


def selftest(path):
    g = Ggu(path)
    n_ok = 0
    for name, (dims, dt, off) in list(g.tensors.items())[:400]:
        if dt != 39:
            continue
        ne0, ne1 = int(dims[0]), int(dims[1])
        if ne0 % 32:
            continue
        raw = g.mm[g.base + off: g.base + off + ne1 * (ne0 // 32) * 17]
        codes, exps = parse_blocks(raw, ne0, ne1)
        back = pack_blocks(codes, exps, ne0, ne1)
        assert back == bytes(raw), "byte mismatch in %s" % name
        n_ok += 1
    print("ROUNDTRIP OK for %d mxfp4 tensors" % n_ok)


if __name__ == "__main__":
    if sys.argv[1] == "selftest":
        import os
        selftest(os.environ["GGUF_IN"])
        sys.exit(0)
