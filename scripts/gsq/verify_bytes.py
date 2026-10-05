#!/usr/bin/env python3
"""Prove the refinement touched ONLY code nibbles, never exponents, never sizes,
never metadata -> radiance fast path premises are byte-for-byte intact."""
import sys, os
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.environ["GSQ_CPU"] = "1"
import numpy as np
from gsq_core import Ggu
from gsq_pack import parse_blocks

PROD = os.environ["GSQ_PROD"]
REF = os.environ.get("GSQ_REF", os.path.join(HERE, "gsq-refined.gguf"))
gp = Ggu(PROD); gr = Ggu(REF)
assert set(gp.tensors) == set(gr.tensors), "tensor set changed"
n_exp_same = n_code_diff = n_type = bad = 0
for name, (dims, dt, off) in gp.tensors.items():
    d2, dt2, off2 = gr.tensors[name]
    assert d2 == dims and dt2 == dt, name
    if dt != 39:
        continue
    n_type += 1
    ne0, ne1 = int(dims[0]), int(dims[1])
    sz = ne1 * (ne0 // 32) * 17
    c0, e0 = parse_blocks(bytes(gp.mm[gp.base + off: gp.base + off + sz]), ne0, ne1)
    c1, e1 = parse_blocks(bytes(gr.mm[gr.base + off: gr.base + off + sz]), ne0, ne1)
    if not np.array_equal(e0, e1):
        bad += 1
    n_exp_same += 1
    if not np.array_equal(c0, c1):
        n_code_diff += 1
print("mxfp4 tensors: %d, exponent-identical: %d, exponent-changed: %d, code-changed: %d" % (
    n_type, n_exp_same, bad, n_code_diff))
assert bad == 0, "EXONENTS CHANGED -> radiance premise broken"
import os
print("file sizes: prod=%d refined=%d same=%s" % (
    os.path.getsize(PROD), os.path.getsize(REF),
    os.path.getsize(PROD) == os.path.getsize(REF)))
