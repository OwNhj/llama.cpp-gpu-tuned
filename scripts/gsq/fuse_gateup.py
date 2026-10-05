#!/usr/bin/env python3
# fuse ffn_gate + ffn_up -> ffn_gate_up (byte-exact mxfp4 concat) in a GGUF
# reads SRC, writes DST. gate rows first, then up rows.
import struct, sys, json, os

SRC = os.environ["GGUF_IN"]
DST = os.environ["GGUF_OUT"]

f = open(SRC, "rb")
data = f.read()
f.close()

# --- parse GGUF v3 header ---
off = 0
magic = data[0:4]; assert magic == b"GGUF", magic
off = 4
ver, n_tensors, n_kv = struct.unpack_from("<IQQ", data, off)
off += 20
print(f"ver={ver} n_tensors={n_tensors} n_kv={n_kv}")

def read_str(buf, o):
    n = struct.unpack_from("<Q", buf, o)[0]; o += 8
    s = buf[o:o+n].decode("utf-8"); o += n
    return s, o

def read_val(buf, o, t):
    if t == 0: v = struct.unpack_from("<B", buf, o)[0]; return v, o+1
    if t == 1: v = struct.unpack_from("<b", buf, o)[0]; return v, o+1
    if t == 2: v = struct.unpack_from("<H", buf, o)[0]; return v, o+2
    if t == 3: v = struct.unpack_from("<h", buf, o)[0]; return v, o+2
    if t == 4: v = struct.unpack_from("<I", buf, o)[0]; return v, o+4
    if t == 5: v = struct.unpack_from("<i", buf, o)[0]; return v, o+4
    if t == 6: v = struct.unpack_from("<f", buf, o)[0]; return v, o+4
    if t == 7: v = struct.unpack_from("<B", buf, o)[0]; return v, o+1
    if t == 8: return read_str(buf, o)
    if t == 9:
        etype = struct.unpack_from("<I", buf, o)[0]; o += 4
        n = struct.unpack_from("<Q", buf, o)[0]; o += 8
        vals = []
        for _ in range(n):
            v, o = read_val(buf, o, etype)
            vals.append(v)
        return (etype, vals), o
    if t == 10: v = struct.unpack_from("<Q", buf, o)[0]; return v, o+8
    if t == 11: v = struct.unpack_from("<q", buf, o)[0]; return v, o+8
    if t == 12: v = struct.unpack_from("<d", buf, o)[0]; return v, o+8
    raise ValueError(f"type {t}")

kvs = {}
for _ in range(n_kv):
    k, off = read_str(data, off)
    t = struct.unpack_from("<I", data, off)[0]; off += 4
    v, off = read_val(data, off, t)
    kvs[k] = (t, v)

tensors = []
for _ in range(n_tensors):
    pass  # handled in second pass below

# simpler: tensor info is name, n_dims, dims, type, offset(u64)
off = 4 + 20
for _ in range(n_kv):
    k, off = read_str(data, off)
    t = struct.unpack_from("<I", data, off)[0]; off += 4
    v, off = read_val(data, off, t)

tensor_infos = []
for _ in range(n_tensors):
    name, off = read_str(data, off)
    nd = struct.unpack_from("<I", data, off)[0]; off += 4
    dims = struct.unpack_from("<" + "Q"*nd, data, off); off += 8*nd
    ttype = struct.unpack_from("<I", data, off)[0]; off += 4
    toff = struct.unpack_from("<Q", data, off)[0]; off += 8
    tensor_infos.append((name, nd, dims, ttype, toff))

data_start = off
# align pad to 32
ALIGN = 32
data_start = (data_start + ALIGN - 1) // ALIGN * ALIGN
print(f"data_start={data_start} (0x{data_start:x})")

def tsize_bytes(n_dims, dims, ttype):
    # type 39 mxfp4: 17 bytes per 32 elems; type 45 mxfp6: 25 per 32; type 30 bf16: 2/elem; 0 f32: 4; 1 f16: 2
    # 43: Q8_0? no -> 8_0 is 8. type ids: 8=Q8_0 (32 elems: 32+2 bytes), 12..: fallback sizes
    n = 1
    for d in dims: n *= d
    if ttype == 39: return (n + 31)//32 * 17
    if ttype == 45: return (n + 31)//32 * 25
    if ttype == 30: return n*2
    if ttype == 0:  return n*4
    if ttype == 1:  return n*2
    if ttype == 8:  return (n + 31)//32 * 34   # Q8_0
    if ttype == 43: return (n + 31)//32 * 33   # mxfp8 (eh_proj: 33B/32, id 43)    if ttype == 2:  return (n + 31)//32 * 18   # Q4_0
    raise ValueError(f"ttype {ttype}")

# group ffn pairs
gate_idx = {}
up_idx = {}
for i,(name,nd,dims,tt,toff) in enumerate(tensor_infos):
    m = name.rsplit(".", 2)
    if len(m)==3 and m[1]=="ffn_gate" and m[2]=="weight":
        gate_idx[m[0]] = i
    if len(m)==3 and m[1]=="ffn_up" and m[2]=="weight":
        up_idx[m[0]] = i

assert set(gate_idx) == set(up_idx), "gate/up mismatch"
print(f"pairs: {len(gate_idx)}")

# verify contiguity and sizes
tmap = {name:(nd,dims,tt,toff) for name,nd,dims,tt,toff in tensor_infos}
for blk in gate_idx:
    g = tmap[blk+".ffn_gate.weight"]; u = tmap[blk+".ffn_up.weight"]
    assert g[2]==39 and u[2]==39, (g[2],u[2])
    assert g[1]==u[1], blk  # dims equal
    gs = tsize_bytes(*g[:1], g[1], g[2], g[3]) if False else tsize_bytes(2, g[1], 39)
    assert tsize_bytes(2, g[1], 39) == tsize_bytes(2, u[1], 39)

# verify all tensors are contiguous in file order (needed for byte-move plan)
order = sorted(tensor_infos, key=lambda t: t[4])
prev_end = 0
for name,nd,dims,tt,toff in order:
    sz = tsize_bytes(nd, dims, tt)
    assert toff == prev_end, f"gap before {name}: {toff} != {prev_end}"
    prev_end = toff + sz
print("data section fully contiguous, total", prev_end)

# build new tensor list: for each blk with pair -> one fused tensor at gate's offset
fused_names = set()
new_infos = []
for (name,nd,dims,tt,toff) in tensor_infos:
    m = name.rsplit(".", 2)
    if len(m)==3 and m[1] in ("ffn_gate","ffn_up") and m[2]=="weight":
        blk = m[0]
        if blk in fused_names:
            continue
        if m[1]=="ffn_gate":
            g = tmap[blk+".ffn_gate.weight"]; u = tmap[blk+".ffn_up.weight"]
            fused_dims = (g[1][0], g[1][1]*2)
            new_infos.append((blk+".ffn_gate_up.weight", 2, fused_dims, 39, g[3]))
            fused_names.add(blk)
        # ffn_up entry skipped
    else:
        new_infos.append((name,nd,dims,tt,toff))

print(f"new tensor count: {len(new_infos)} (was {n_tensors})")

# data section: since gate comes before up and all pairs are adjacent? verify:
# gate off + gsize == up off?
for blk in sorted(gate_idx, key=lambda b: int(b.split(".")[1])):
    g = tmap[blk+".ffn_gate.weight"]; u = tmap[blk+".ffn_up.weight"]
    gs = tsize_bytes(2, g[1], 39)
    if g[3] + gs != u[3]:
        print(f"NOT adjacent: {blk} gate@{g[3]}+{gs} vs up@{u[3]}")
        break
else:
    print("all gate/up pairs byte-adjacent (gate first)")

# write header
def pack_str(s):
    b = s.encode("utf-8")
    return struct.pack("<Q", len(b)) + b

hdr = bytearray()
hdr += b"GGUF"
hdr += struct.pack("<IQQ", ver, len(new_infos), n_kv)
for k,(t,v) in kvs.items():
    hdr += pack_str(k) + struct.pack("<I", t)
    # write value
    def wval(buf, t, v):
        if t == 0: buf += struct.pack("<B", v)
        elif t == 1: buf += struct.pack("<b", v)
        elif t == 2: buf += struct.pack("<H", v)
        elif t == 3: buf += struct.pack("<h", v)
        elif t == 4: buf += struct.pack("<I", v)
        elif t == 5: buf += struct.pack("<i", v)
        elif t == 6: buf += struct.pack("<f", v)
        elif t == 7: buf += struct.pack("<B", v)
        elif t == 8: buf += pack_str(v)
        elif t == 9:
            etype, vals = v
            buf += struct.pack("<I", etype) + struct.pack("<Q", len(vals))
            for item in vals:
                buf = wval(buf, etype, item)
        elif t == 10: buf += struct.pack("<Q", v)
        elif t == 11: buf += struct.pack("<q", v)
        elif t == 12: buf += struct.pack("<d", v)
        return buf
    hdr = wval(hdr, t, v)

for (name,nd,dims,tt,toff) in new_infos:
    hdr += pack_str(name)
    hdr += struct.pack("<I", nd)
    hdr += struct.pack("<" + "Q"*nd, *dims)
    hdr += struct.pack("<I", tt)
    hdr += struct.pack("<Q", toff)

# align
while len(hdr) % ALIGN != 0:
    hdr += b"\x00"

new_data_start = len(hdr)
print(f"new data_start={new_data_start}, header shrink={data_start-new_data_start}")

# data section: copy whole block; only fix up: fused tensor occupies gate..up end
# Since gate..up are adjacent and we keep offsets, the data region between them is
# exactly gate+up bytes -> byte-identical data section, nothing to move!
out = open(DST, "wb")
out.write(hdr)
out.write(data[data_start:])
out.close()
print("wrote", DST, os.path.getsize(DST))
