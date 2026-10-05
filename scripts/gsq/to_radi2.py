#!/usr/bin/env python3
# convert B145 mxfp4 -> mxfp4_rad FULL PLANE layout:
#   per tensor: [N*nb*16 interleaved codes][N*nb e8m0 scales (row-major n*nb+b)]
# same total bytes as mxfp4. header: type 39 -> 47.
import struct, sys, os, numpy as np

SRC = os.environ["GGUF_IN"]
DST = os.environ["GGUF_OUT"]

data = open(SRC, "rb").read()
ver, n_tensors, n_kv = struct.unpack_from("<IQQ", data, 4)

def rstr(buf, o):
    n = struct.unpack_from("<Q", buf, o)[0]; o += 8
    return buf[o:o+n].decode(), o + n

def rval(buf, o, t):
    if t == 8: return rstr(buf, o)
    if t == 9:
        et = struct.unpack_from("<I", buf, o)[0]; o += 4
        n = struct.unpack_from("<Q", buf, o)[0]; o += 8
        vals = []
        for _ in range(n):
            v, o = rval(buf, o, et); vals.append(v)
        return (et, vals), o
    fmt = {0:"B",1:"b",2:"H",3:"h",4:"I",5:"i",6:"f",7:"B",10:"Q",11:"q",12:"d"}[t]
    v = struct.unpack_from("<"+fmt, buf, o)[0]
    return v, o + struct.calcsize(fmt)

off = 24
kvs = {}
for _ in range(n_kv):
    k, off = rstr(data, off)
    t = struct.unpack_from("<I", data, off)[0]; off += 4
    v, off = rval(data, off, t)
    kvs[k] = (t, v)

infos = []
for _ in range(n_tensors):
    name, off = rstr(data, off)
    nd = struct.unpack_from("<I", data, off)[0]; off += 4
    dims = struct.unpack_from("<"+"Q"*nd, data, off); off += 8*nd
    tt = struct.unpack_from("<I", data, off)[0]; off += 4
    toff = struct.unpack_from("<Q", data, off)[0]; off += 8
    infos.append([name, nd, dims, tt, toff])
data_start = (off + 31) // 32 * 32

def tsize(dims, tt):
    n = 1
    for d in dims: n *= d
    if tt == 39: return (n + 31)//32 * 17
    if tt == 45: return (n + 31)//32 * 25
    if tt == 43: return (n + 31)//32 * 33
    if tt == 0:  return n*4
    if tt == 30: return n*2
    if tt == 47: return (n + 31)//32 * 17
    raise ValueError(tt)

prev = 0
for name, nd, dims, tt, toff in infos:
    sz = tsize(dims, tt)
    assert toff == prev, (name, toff, prev)
    prev = toff + sz
print("contiguous ok; data bytes", prev)

LO = np.arange(256, dtype=np.uint8) & 0x0F
HI = (np.arange(256, dtype=np.uint8) >> 4)

body = bytearray()
new_infos = []
nconv = 0
for name, nd, dims, tt, toff in infos:
    if tt == 39 and name != "token_embd.weight":
        K, N = dims[0], dims[1] if nd >= 2 else 1
        nb = K // 32
        src = np.frombuffer(data, dtype=np.uint8, count=N*nb*17, offset=data_start+toff)
        src = src.reshape(N, nb, 17)
        e  = src[:, :, 0]
        qs = src[:, :, 1:]
        lo = LO[qs]; hi = HI[qs]
        outb = np.empty((N, nb, 16), dtype=np.uint8)
        outb[:, :, 0:8]  = lo[:, :, 0::2] | (lo[:, :, 1::2] << 4)
        outb[:, :, 8:16] = hi[:, :, 0::2] | (hi[:, :, 1::2] << 4)
        body += outb.tobytes()          # code plane [N][nb*16]
        body += e.tobytes()             # scale plane [N][nb] (row-major)
        new_infos.append([name, nd, dims, 47, toff])
        nconv += 1
        if nconv % 60 == 0: print(f"  {nconv}...", flush=True)
    else:
        sz = tsize(dims, tt)
        body += data[data_start+toff : data_start+toff+sz]
        new_infos.append([name, nd, dims, tt, toff])
print("converted", nconv)

def pstr(s):
    b = s.encode(); return struct.pack("<Q", len(b)) + b

def wval(buf, t, v):
    if t == 8: return buf + pstr(v)
    if t == 9:
        et, vals = v
        buf += struct.pack("<IQ", et, len(vals))
        for item in vals: buf = wval(buf, et, item)
        return buf
    fmt = {0:"B",1:"b",2:"H",3:"h",4:"I",5:"i",6:"f",7:"B",10:"Q",11:"q",12:"d"}[t]
    return buf + struct.pack("<"+fmt, v)

hdr = bytearray(b"GGUF" + struct.pack("<IQQ", ver, n_tensors, n_kv))
for k, (t, v) in kvs.items():
    hdr += pstr(k) + struct.pack("<I", t)
    hdr = wval(hdr, t, v)
for name, nd, dims, tt, toff in new_infos:
    hdr += pstr(name) + struct.pack("<I", nd) + struct.pack("<"+"Q"*nd, *dims) + struct.pack("<IQ", tt, toff)
while len(hdr) % 32: hdr += b"\x00"
assert len(hdr) == data_start, (len(hdr), data_start)
assert len(hdr) + len(body) == len(data), (len(hdr)+len(body), len(data))
with open(DST, "wb") as f:
    f.write(hdr); f.write(bytes(body))
print("wrote", DST, len(data))
