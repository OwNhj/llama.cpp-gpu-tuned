#!/usr/bin/env python3
"""Full tt-errors comparison: B145 stored codes vs refined codes, all 280 tensors,
parallel (each worker opens its own GGUF mmaps). im_rmse follows the C++ definition
(sqrt(sum w e^2 / sum w W^2) over the tensor)."""
import sys, os, json
os.environ["GSQ_CPU"] = "1"
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from multiprocessing import Pool
from gsq_core import Ggu, load_bf16, dequant
from gsq_pack import parse_blocks
from gguf import GGUFReader

SRC = os.environ["GSQ_SRC"]
PROD = os.environ["GSQ_PROD"]
REF = os.environ.get("GSQ_REF", os.path.join(HERE, "gsq-refined.gguf"))
OUT = os.environ.get("GSQ_TTOUT", os.path.join(HERE, "tt_compare_full.json"))
G = {}


def init():
    G["src"] = Ggu(SRC); G["prod"] = Ggu(PROD); G["ref"] = Ggu(REF)
    ri = GGUFReader(os.environ["GSQ_IMATRIX"])
    G["ima"] = {t.name[:-8]: np.ascontiguousarray(np.asarray(t.data, dtype=np.float32))
                for t in ri.tensors if t.name.endswith(".in_sum2")}


def metrics(g, name, ne0, ne1):
    dims, dt, off = g.tensors[name]
    c, e = parse_blocks(bytes(g.mm[g.base + off: g.base + off + ne1 * (ne0 // 32) * 17]), ne0, ne1)
    W = load_bf16(G["src"], name).astype(np.float32)
    Q = dequant(c, e, ne0, ne1).astype(np.float32)
    E = (W.astype(np.float32) - Q).astype(np.float64)
    Wd = W.astype(np.float64)
    im = G["ima"][name].astype(np.float64)
    rmse = float(np.sqrt((E * E).sum() / (Wd * Wd).sum()))
    wim = float((Wd * Wd * im).sum())
    imr = float(np.sqrt((E * E * im).sum() / wim)) if wim > 0 else -1.0
    return rmse, imr


def work(name):
    dims, dt, off = G["prod"].tensors[name]
    ne0, ne1 = int(dims[0]), int(dims[1])
    r0, i0 = metrics(G["prod"], name, ne0, ne1)
    r1, i1 = metrics(G["ref"], name, ne0, ne1)
    return name, i0, i1, r0, r1


if __name__ == "__main__":
    prog = set(json.load(open(REF + ".progress.json")))
    names = sorted(n for n, (d, dt, o) in Ggu(PROD).tensors.items() if dt == 39 and n in prog)
    with Pool(8, initializer=init) as p:
        res = p.map(work, names, chunksize=3)
    J0 = sum(x[1] ** 2 for x in res); J1 = sum(x[2] ** 2 for x in res)
    json.dump(res, open(OUT, "w"))
    worse = sum(1 for x in res if x[2] > x[1])
    print("refined mxfp4 tensors=%d  per-tensor worse: %d" % (len(res), worse))
    print("J(im_rmse^2 over refined subset): prod=%.4f refined=%.4f  ratio %.4f (%+.2f%%)" % (
        J0, J1, J1 / J0, 100 * (J1 / J0 - 1)))
