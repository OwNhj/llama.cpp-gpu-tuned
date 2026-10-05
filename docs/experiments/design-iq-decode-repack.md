# IQ/K 量化预解码 repack (int 版 radiance) 实施文档

> 开工: 2026-10-01。上游设计讨论见 design-iq-radiance-feasibility.md §10-§12。
> 原则: 随做随写。每个设计决策、每个坑、每个实测数字都记录在案。

## 0. 目标

GSQ 类混合量化模型的 7 种码本/低bit类型 (Q2_K, IQ3_S, IQ3_XXS, IQ2_S, IQ2_XS,
IQ2_XXS, IQ4_XS) 在 repack 阶段预解码成 int8 + fp16 scale, 运行时 staging
退化为纯拷贝, 内层纯 iu8 WMMA。预期 pp2048 1294 -> ~2292 (+77%),
显存 +13.66 GB (贴预算线), tg128 不变 (MMVQ 读原始份), PPL 逐位一致。

## 1. 阶段 0 设计决策 (IQ3_S 原型)

### 1.1 kernel 复用判定 (已查证)

- Q3_K 在现有 MMQ 里走 `q8_0_16_q8_1_mma` + `load_tiles_q3_K`:
  **staging 里就把 3-bit 值展开成 int8 tile** (x_qs), dA = x_df (fp16, per-32)。
- => "解码后 int8 tile + fp16 scale" 正是这套 kernel 的既有输入形态。
- 新增代码只有 **repack kernel** (IQ3_S block -> 解码 int8 tile 的新存储),
  load_tiles 变成拷贝循环 (或直接用新 type 的 load_tiles 变体)。

### 1.2 存储格式选择 (决策 D1)

两个选项:
- (a) 新 GGUF type: 改 ggml.h 枚举 + convert/quantize + 全链路。工程大, 破坏兼容。
- (b) **runtime repack (选定)**: 仿 radiance-gemm.cu 的 per-tensor repack cache
  (首次 GEMM 时解码, process 生命周期缓存)。原始 tensor 驻留不变 (MMVQ 仍读),
  解码副本 +N*K/2 x 1B + scale。与 CUDA graph 兼容 (repack 只在首次, 地址稳定)。

### 1.3 解码副本布局 (关键决策)

repack 产物不是 GGUF 块结构, 而是直接对齐 MMQ staging 的 tile 布局:
- x_qs: [N, K] int8 -> 按 (sram_stride=MMQ_TILE_NE_K 布局) 存?
  **否**: repack 产物应与原 tensor 同构 (N 行 x K 字节 int8 + N x K/16 fp16),
  load_tiles 从这个紧凑布局拷贝进 LDS tile —— 这样 repack kernel 与
  tile 几何解耦, J/I 调参不影响 repack。
- 命名: 新文件 `radiance-decode.cu` (或并入 radiance-gemm.cu), cache 结构
  仿 radiance 的 {W, Ws, Wref} -> {W8 (N,K int8), Wdf (N, K/16 fp16)}。

### 1.4 IQ3_S 解码语义 (从 staging 代码逐行抄, 保证逐位等价)

```
每 256 元素块 (block_iq3_s):
  qs[13x32], qh[4B], signs[32B], d(fp16), scales[4B] (QK_K/64=4, 每 64 元素 2 个 4bit ls)
现有 staging 语义 (mmq-load-tiles.cuh load_tiles_iq3_s):
  grid_pos = iq3s_grid[qs bits | qh bit]
  signs 展开: __vcmpne4 打包符号位 -> XOR grid
  x_df = ls * d   其中 ls = 1 + 2*((scales >> ...) & 0xF)   [per-16!]
repack 产物:
  W8[n][k] = int8(grid - sign)          (与 grid^signs 相同值, 范围 +-31..0)
  Wdf[n][k/16] = (1 + 2*ls4)*d / 4      (注意现代码: ls*d 其中 ls=1+2*ls4, /4? 核对!)
```

### 1.5 iso 验证门

写独立 probe: 对随机 N x K, 原路径 load_tiles_iq3_s 产出的 (x_qs, x_df) tile
必须与新路径 (repack W8/Wdf -> memcpy) 产出的 tile **逐字节一致**。
这是 PPL 逐位一致的前置。


### 1.6 IQ3_S scale 语义定案 (源码核对)

- `IQ3S_N_SCALE = QK_K/64 = 4` 字节 scales (不是 K/16!)。
- 每线程 kqsx (0..7) 对应 32 元素子块的 1/8? 精确: threads_per_row =
  (MMQ_ITER_K/(4*QR3_S))/2 = (256/16)/2 = 8; 每线程管 32 元素。
- `ls = 1 + 2*((scales[kqsx/2] >> ((2*kqsx<<1) & 0x04)) & 0xF)` —— 每 32 元素
  一个 4-bit ls (奇数倍 1..31), `x_df = ls * d` (per-32, 不是 per-16!)。
- **修正 §1.4 的错误**: IQ3_S 解码后是 per-32 scale —— 和 Q8_0 粒度一致!
  => **IQ3_S 可以直接映射 Q8_0 路径 (q8_0_q8_1_mma, 167 锚点 kernel)**,
  不是 q8_0_16。这比预想更顺。
- 注意 x_qs 里存的 int8 值 = grid 值 (含符号), grid 范围 0..63 内的 8bit 通道
  (iq3s_grid 每字节是 4 个 3bit 值拼的?) —— 需要探针确认 grid 值范围与
  x_df 乘完后是否等效于 dequant。**iso probe 是下一步第一件事。**

## 2. 阶段 0 进展: IQ3_S iso probe PASS (2026-10-01 23:5x)

`/tmp/iso_iq3s.hip`: GPU 原路径 (stage_orig, 逐行抄 load_tiles_iq3_s 数学:
iq3s_grid 查表 + qh 位选择 + __vcmpne4/vsub4 符号展开 + ls*d scale)
vs 模拟 repack (host 侧同数学算 W8/Wdf, device 只 memcpy) ——
**64 个 x_qs int + 8 个 x_df float 全部逐位一致**。

坑: __vcmpne4/__vsub4 在 standalone hipcc 需 <hip/hip_runtime.h> 之外的手工模拟
(mmq.cuh 编译时有其他头带入); fp16 需显式 <hip/hip_fp16.h>。
已在 probe 里用 per-byte 循环模拟 (语义: vcmpne4 = per-byte a!=b?0xFF:0;
vsub4 = per-byte uint8 wraparound 减法) —— 这两个模拟也就是 CPU repack
kernel 的参考实现。

下一步: 把 host repack 移植成 GPU repack kernel (radiance_repack_kernel 同构),
接进 radiance-decode.cu, 声明 IQ3_S 的 runtime repack 分派。

### 1.7 分派架构定案 (最小侵入)

看了 load_tiles 机制后, **不改 load_tiles、不改 vec_dot** —— 解码在 repack
kernel 里做完, 产物直接是 "IQ3_S 语义的 Q8_0 形态":
- W8: [N rows, K int8] —— 每行是解码后的 int8 值 (与原 x_qs tile 同值);
- Wdf: [N, K/32] fp16 —— ls*d (与原 x_df 同值)。

然后 mul_mat_q_radiance_decode (mmq.cu, 仿 ggml_cuda_mul_mat_q_radiance):
- gate: type ∈ 集合 && RDNA4 && M>=256 && K%256==0 && contiguous;
- repack cache 仿 g_radiance_weights;
- **用现成 Q8_0 kernel 跑**: launch mul_mat_q<Q8_0> 但 x-tile 数据来自
  解码 buffer? —— 不行, launch_mul_mat_q 是 type 模板, load_tiles<Q8_0>
  会按 block_q8_0 (34B 块) 解析输入。
- **修正方案**: 解码 buffer 的布局**直接就是 block_q8_0 数组**
  (每 32 元素: d fp16 + 32 int8), 即 repack 产物是一个虚拟 Q8_0 tensor!
  然后 launch_mul_mat_q<GGML_TYPE_Q8_0> 传这个 buffer —— 现有 kernel
  零改动, 因为 Q8_0 load_tiles 只做 nibble-free 的直接拷贝。
- K 维块数不同: Q8_0 是 K/32 块, IQ3_S 原是 K/256 —— repack 产物
  恰好是 K/32 个 block_q8_0, 一一对应。**这是最终架构。**

含义: Q2_K 同理 (sc*v+min 0..60 -> Q8_0), IQ2/IQ4 系 scale 是 per-16
-> 虚拟 Q8_0 不行 (Q8_0 是 per-32), 需 Q8_0_16 形态 -> 走
launch_mul_mat_q<Q8_0 的 per-16 变体>... 查证: 没有独立 GGML type
"Q8_0_16", Q3_K 的 q8_0_16 是 load_tiles 侧展开的。=> IQ2/IQ4 系
repack 产物用 Q8_0 形态但 scale 每 32 复制两份 16-scale? 不行, 数学不等价。
**分型处理**: Q2_K/IQ3_S/IQ3_XXS (per-32) -> 虚拟 Q8_0 (全零新代码 kernel);
IQ2_S/XS/XXS/IQ4_XS (per-16) -> 新模板实例 load_tiles_decode16 (拷贝循环 +
q8_0_16 vec_dot, 预期也是少量代码)。

### 1.8 scale 粒度最终核对表 (全部源码验证)

| 类型 | 块结构 | scale 粒度 | 值域 | repack 目标形态 |
|---|---|---|---|---|
| Q2_K | dm(half2) + scales[16B] (每 16 元素 sc4+min4) + qs | **per-16** (x=v*sc*dm.x - min*dm.y, 双分量!) | 带加性 min | **不是 Q8_0!** Q8_0 无 min。Q2_K 解码值 = v*sc*d - m*dm.x*dmin... 有两个 scale 通道 |
| IQ3_S | d + qs/qh/signs + scales[4B] | per-32 (ls=1+2*4bit) | 0..63 ±sign | **Q8_0 ✓** (iso PASS) |
| IQ3_XXS | d + qs + qh + signs | per-32? (查) | 0..63±sign | 待查 |
| IQ2_* | d + qs + qh + scales | per-16 或 per-32 (查) | 0..255±sign | 待查 |
| IQ4_XS | d + scales_l/h + qs | per-32? (查) | 码本 int8 | 待查 |

**Q2_K 特殊**: 值 = dm.x*sc*v - dm.y*min (两个 fp16 通道), 不是单一 scale x int8。
强行 Q8_0 需要 v8 = sc*v-min 合成 int8 (值域 0..60 可以), scale=dm.x, 但
min 通道丢给 v8 后误差 = (dm.y*dm.x 差) —— **不逐位等价!**
=> Q2_K 从"虚拟 Q8_0"名单移到"需单独论证" (Q8_1 有 min? block_q8_1 有 sum
无 min; Q4_1 有 min! block_q4_1: d+dmin+qs+qu。但 MMQ 无 q4_1 权重路径)。
**Q2_K 降级为二期, 一期 = 纯码本型: IQ3_S/IQ3_XXS/IQ2_*/IQ4_XS。**

### 1.9 各类型 scale 合成与粒度 (源码全查, 一次定案)

| 类型 | scale 表达式 (staging 侧) | 粒度 | x_qs 值来源 | 值域 | repack 后映射 |
|---|---|---|---|---|---|
| IQ3_S | `ls*d`, ls=1+2*4bit | **per-32** | iq3s_grid ± sign | 码本 (0..63 范围的 8bit 打包 x2?) | **Q8_0 ✓ (iso PASS)** |
| IQ3_XXS | `(ls*d + d/2)/2`, ls=5bit | per-32 | iq3s_grid ± sign | 同上 | **Q8_0 ✓** (ls 合成不同而已) |
| IQ2_S | `((ls&0xF)*d + d/2)/4` 每 kqsx 两份 | **per-16** (2*kqsx+0/1) | iq2s_grid ± sign | 码本 | **Q8_0_16 型** |
| IQ2_XS | 同 IQ2_S | per-16 | iq2s_grid ± sign | 码本 | Q8_0_16 型 |
| IQ2_XXS | `d*ls/8` | per-32 | iq2xxs_grid ± sign | 码本 | **Q8_0 ✓** |
| IQ4_XS | `d*(ls-32)` | per-32 | kvalues_iq4nl | [-127,113] int8 码本 | **Q8_0 ✓** |

**一期定案: IQ3_S / IQ3_XXS / IQ2_XXS / IQ4_XS 四个 per-32 类型先做**
(虚拟 Q8_0 形态, launch_mul_mat_q<GGML_TYPE_Q8_0> 零 kernel 改动);
IQ2_S/IQ2_XS (per-16) 二期 (需要 Q8_0_16 形态的新模板实例或 scale 复制方案)。

**Q2_K 从名单移除** (dm.x*sc*v - dm.y*min 双通道, 单 int8+scale 无法逐位等价;
除非用 (v8=sc*v-min, s=dm.x) + min 修正项 —— 那是另一个 kernel 的事)。

显存账修正 (一期 4 类型): IQ3_S 4.31 + IQ3_XXS 3.43 + IQ2_XXS 0.31 + IQ4_XS 2.84
= **10.89 GB 净增** (预算 13.9, 余 3 GB)。二期 IQ2_S/XS +2.25 GB -> 全做 = 13.14 GB。

### 1.10 x_qs 值域核实 (int8 兼容性定案)

iq3s_grid 项如 0x01010103: 4 个 byte, 每 byte 0..15 (4bit 格点幅值);
staging 的 grid_pos.x 整个 dword 与 sign mask 做 vcmpne4/vsub4 (per-byte 独立)。
所以 **x_qs 的每个 byte = 幅值(0..15) 或其 0x80 补** (sign XOR 是 byte 级 0x80),
vsub4 后 = signed byte, 值域 [-16,+15] ⊂ int8。**虚拟 Q8_0 布局完全兼容**
(block_q8_0.qs 就是 int8[32])。

对 dequant 语义: x = d * ls * (byte 有符号值)。repack 把 byte 原样拷进
block_q8_0.qs, scale=ls*d —— 数学恒等。

### 1.11 signs 展开的 byte 级语义 (repack kernel 的精确实现依据)

signs0 = vcmpne4(E, 0), E = ((m&0x03)<<7)|((m&0x0C)<<21), m = sp8[l] 的 byte l。
E 的 byte 分布: (m&0x03)<<7 产生 byte0=0x80+(m>>?..), byte1 携带 (m&0x03)>>1 位;
(m&0x0C)<<21 产生 byte2=0x80, byte3 携带高位。实测分布:
  m=0x03 -> bytes {80,01,00,00};  m=0x0C -> {00,00,80,01}
  m=0x0F -> {80,01,80,01};        m=0x30/0xC0 -> {00,00,00,00}
即 mask byte = 0x00 或非零 (0x01/0x80 等), vcmpne4 后映射为 0x00/0xFF。
grid_l byte = (gp_byte ^ s_byte) - s_byte (uint8): s=0 -> gp; s=0xFF -> -gp。
**repack 实现: 逐 byte 算 E, s_byte = (E_byte ? 0xFF : 0x00), out = s ? (-gp)&0xFF : gp。**
不再试图解析位域, 直接 per-byte 计算 E (4 次查表/移位), 语义与 GPU 完全一致。

### 1.12 第一版 kernel 的自我审查 (发现的问题, 未合入)

第一版 decode_iq3_s_kernel 有三个问题, 记录以避免重犯:
1. **线程映射混乱**: lane != sb%4 提前 return 是错的。正确映射: staging 里
   thread kqsx (0..7) 产出 8 个 int = 32 bytes, 覆盖 superblock 的
   [32*kqsx, 32*kqsx+32)。即 **sub-block sb 与 kqsx 一一对应 (sb = kqsx)**,
   每 sub-block 只需 1 个线程 (32 bytes = 8 int)。grid = (N, K/32), 1 thread。
2. **qs1 的引用错误**: l=2,3 时仍用 qs0 索引 (qs_packed1 未正确参与)。
   staging: qs_packed = make_int2(get_int_b2(qs, 2*kqsx+0), get_int_b2(qs, 2*kqsx+1))
   即 qs_packed0 = bytes[4*kqsx..4*kqsx+1], qs_packed1 = bytes[4*kqsx+2..4*kqsx+3],
   然后 qs[2l+0]/qs[2l+1] 索引的是 **int2 的 8 个 byte** (qs_packed0 的 4 + qs_packed1 的 4)。
   我第一版 l>=2 才用 qs1 是**错的** —— 正确: l=0..3 全部走
   (l<2 ? qs0 : qs1)? 不对: qs 数组 = (const uint8_t*)&qs_packed, int2 8 bytes:
   qs[0..3] = qs_packed0 的 4 bytes, qs[4..7] = qs_packed1 的 4 bytes。
   staging 用 qs[2l+0], qs[2l+1], l=0..3 -> 索引 0..7 —— 正确覆盖。
3. **double d 写入**: ob->d 写了两次 (先 bxi->d 后 ls*d), 第二次是最终意图,
   删掉第一次。

**重写原则: 每 thread = 一个 32 元素 sub-block, kqsx = sb%8 是唯一映射,
qs[0..7] 直接从 int2 取。** 已按此重写第二版。

## 3. 一期范围再修正 (2026-10-02, 逐类型 scale 源码全查后)

| 类型 | scale 公式 | 粒度 | 状态 |
|---|---|---|---|
| IQ3_S | ls*d, ls=1+2*4bit | per-32 | **一期** (iso PASS) |
| IQ4_XS | d*(ls-32) | per-32 | **一期** |
| IQ3_XXS | (ls*d+d/2)/2 | per-32 | 一期 (aux32 拼法待核实) |
| IQ2_XXS | d*ls/8, ls=ksigns 表 | per-32 | 一期 (同上) |
| IQ2_S/XS | ((ls&0xF)*d+d/2)/4 两份 | per-16 | 二期 |
| Q2_K | dm 双通道 | per-16 | 移除 |

一期 = IQ3_S / IQ4_XS / IQ3_XXS / IQ2_XXS (GSQ 显存净增 10.89 GB)。
IQ3_XXS 与 IQ2_XXS 的 scale 取位是**最不确定的部分** (staging 里
aux32 的拼法各不相同), kernel 写完后必须逐类型 iso probe, 不允许推断。

## 4. iq3_xxs / iq2_xxs 结构修正 (第一版编译失败原因, 2026-10-02)

第一版假设这两个类型有独立 qh/signs/scales 字段 —— **错了**:
- block_iq3_xxs: 只有 d + qs[3*QK_K/8=96B]。q3 数据在前 64B, signs+ls 全部
  编码在 qs[QK_K/16 + kqsx] 起的 8B 里 (aux32), signs = unpack_ksigns(aux32>>7l)
  (7-bit + popcnt 奇偶 = 第 8 位), iq3xxs_grid 是 256 项 (无 qh 位)。
- block_iq2_xxs: d + qs[QK_K/8] (uint16)。q2 = get_int_b2(qs, 2*kqsx),
  aux32 = get_int_b2(qs, 2*kqsx+1), signs = unpack_ksigns(aux32 >> 7l),
  grid 是 uint2 对 (iq2xxs_grid[aux8[l]] -> .x/.y 两路), ls = aux32>>27|1。
- 两者的 x_df: iq3_xxs = (ls*d + d/2)/2 (ls=aux32>>28), iq2_xxs = d*ls/8
  (ls=aux32>>27|1) —— **带 +d/2 与 |1 的 round-to-nearest 味道, repack 时
  必须原样复制公式** (fp16 双舍入), 不允许数学改写。
- iq3_xxs ls 是 aux32>>28 (5bit), 粒度 per-32; iq2_xxs per-32。

**kernel 重写要点**: iq3xxs_grid 256 项 8bit 索引; unpack_ksigns =
v ^ (popc&1)<<7 然后 *0x01010101 广播 —— CPU/GPU 公式一致即可。
qs 布局: q3 = bytes 4*kqsx..4*kqsx+3 (q3_packed), aux32 = bytes
(2*QK_K/16 + 4*kqsx)? 精确: get_int_b2(bxi->qs, QK_K/16 + kqsx) ——
get_int_b2(p,i) 是 16bit 读, 所以 aux32 在字节 (QK_K/16 + kqsx)*2。
QK_K/16=16, 即字节 32..47 内。

### 4.1 get_int_b2/b4 语义 (关键, 之前理解错了)

- `get_int_b2(p, i)`: **32-bit 值**, 由 2 个 uint16 拼: x16[2i] | x16[2i+1]<<16,
  即字节 4i..4i+3 (小端)。
- `get_int_b4(p, i)`: ((int*)p)[i] —— 同样字节 4i..4i+3。
- 两者其实一样 (b2/b4 是历史命名), 都是 4 字节。

修正 iq3_xxs 布局:
- q3_packed = bytes 4*kqsx..4*kqsx+3 (两路 16bit 拼);
- aux32 = bytes 4*(QK_K/16 + kqsx) .. +3 = 字节 64+4*kqsx..64+4*kqsx+3
  (QK_K/16=16, int 索引乘 4) —— 在 qs[96B] 的后半 64..95 区间。
- IQ3_S 同理重核: qs_packed = int2{get_int_b2(qs,2kqsx), get_int_b2(qs,2kqsx+1)}
  = bytes 4kqsx..4kqsx+3 和 4kqsx+4..4kqsx+7 (连续 8 字节)。
  **原 iso probe 的 qs_packed0/1 拼法恰好等价** (q[4k..4k+1]<<0 |8<<16 vs
  手工 byte 拼) —— PASS 仍然有效。

### 4.2 IQ4_XS staging 的关键难点 (发现于第二版 kernel 编写中)

IQ4_XS 的 staging 不是简单 per-32 对应: 8 个 kqsx 线程的 k0 = 8*(kqsx/4)+kqsx%4
散布公式让 **4 个 kqsx 写同一个 32 字节区间的不同偏移** —— 即
sub-block 与 staging 线程是 **N:1 的重叠映射**, 不是 1:1。
准确复刻需要: sub-block sb 的 kqsx 有 4 个候选 (sb%4 + 4*lane), 每个候选
只写 o[(kqsx%4)*8 .. +8]。**kernel 需要 4 线程/sub-block 协作** (或单线程
循环 4 次)。第二版先写 4-thread 协作版 (tid<4 各管一个 kqsx 候选)。
ob_d_set 标签是语法错误, 下一版删掉。

### 4.3 IQ4_XS 布局精查 (决定: 先做整行重建版)

staging x_qs 行 = 64 bytes = 2 个 tile 半区, kqsx 0..7 各写 k0..k0+7
(k0 = 8*(kqsx/4)+kqsx%4), 其中 kqsx 0..3 落前 32 字节 (elem 0..31),
4..7 落后 32 字节 (elem 32..63)?? 重新精确推导:
- kqsx 0..3: k0 = 0,1,2,3 -> 字节 0..7, 8..15, 16..23, 24..31 -> **elem 0..31** ✓
- kqsx 4..7: k0 = 8,9,10,11 -> 字节 32..39, 40..47, 48..55, 56..63 -> **elem 32..63**,
  但这些字节属于 tile 的**第二个 32 字节半区** (x_qs[MMQ_TILE_NE_K..] 那一半,
  即 elements 128..159?) —— **不对**, staging 对 256 元素的 qs[64B] 只做一次
  get_int_b4 循环, 没有第二半。
  
**结论: IQ4_XS 的 tile 映射是"64 字节行"而非"两个 32 字节子块", 虚拟
Q8_0 的 32 元素子块边界与 staging 的 k0 散布不对齐。**
精确方案: repack kernel 每个 256 元素 superblock 由 8 线程按 staging 语义
写 64 字节 scratch -> 再由同 kernel 拆成两个 Q8_0 sub-block (或一个线程做全)。
**v1 先只用 1 线程/superblock 循环 8 次, 正确性优先, 性能后调。**

同理念适用于全部 4 类型: v1 全部 1 线程/256 元素块, 循环复刻 staging 8 线程
的输出字节序, 先把 PPL 逐位一致拿下, 再谈 kernel 并行度。

### 4.4 IQ4_XS 元素顺序推导定案 (v2)

get_int_b4(qs, kqsx) = 字节 4*kqsx..4*kqsx+3 = 8 个 nibble。
get_int_from_table_16: v.x = 偶数 nibble 位置的 LUT 值 (4 bytes),
v.y = 奇数 nibble 位置 (4 bytes)。
staging 写: x_qs[k0+0] = v.x (INT 索引!), x_qs[k0+4] = v.y,
k0 = 8*(kqsx/4) + kqsx%4。
=> 8 个 kqsx 恰好覆盖 INT 0..15 = 字节 0..63, 无重叠:
  kqsx 0: ints 0,4; 1: 1,5; 2: 2,6; 3: 3,7; 4: 8,12; 5: 9,13; 6: 10,14; 7: 11,15。
元素 e (0..255): INT e/4 的 byte e%4。
**v.x 的 4 bytes = 偶数 nibble = 元素 4*k0 + {0,2,4,6}? 不对 —— v.x 的 4 bytes
是 aux 的 byte 0,2,4,6 = 元素 4*k0+{0..3} 里的偶数 nibble... 精确: aux 的
nibble 2j = 元素 8*kqsx + 2j 的低 4 位? nibble j (0..7) 对应元素 8*kqsx+j。**
v.x[j] = LUT[nibble 2j] = 元素 8*kqsx + 2j 的值;
写进 int k0 的 byte j (元素 4*k0 + j)。
所以: 元素 4*k0 + j = LUT[nibble(8*kqsx + 2j)] (j=0..3);
元素 4*k0 + 16 + j = LUT[nibble(8*kqsx + 2j+1)]。
**这是全局重排, 不是顺序的** —— v1 用 64 字节 row + 双循环精确复刻。
scale: x_df[kqsx] 覆盖元素 32*kqsx..+31 (staging kqsx 索引 8 个 32-子块)。

### 4.5 编译通过 (IQ3_S + IQ4_XS 两个 kernel 落地, 2026-10-02)

decode-repack.cu 现含: apply_signs/get_int_b2_d/unpack_ksigns_d 公共工具,
decode_iq3_s_kernel (grid (N, K/256), 单线程循环 8 个 kqsx),
decode_iq4_xs_kernel (同布局, 64 字节 row 缓存 + 精确元素重排 + 双循环)。
IQ3_XXS/IQ2_XXS 已推导未实现 (aux32 拼法各不同, 见 §4 修正)。
下一步: (1) iq3_xxs/iq2_xxs kernel (按 staging 原式); (2) mmq.cu 分派 +
repack cache; (3) GSQ 模型 PPL 逐位验证。

### 4.6 四个 kernel 全部编译通过 (2026-10-02)

decode-repack.cu v3: IQ3_S / IQ3_XXS / IQ2_XXS / IQ4_XS 四个 kernel +
host wrapper, grid = (N, K/256), 每 thread 一个 superblock 循环 8 个
staging kqsx。零错误。

**下一步 (分派层, mmq.cu):**
1. repack cache (仿 g_radiance_weights): key = src0->data, value = {device, buf};
2. gate: type ∈ {IQ3_S, IQ3_XXS, IQ2_XXS, IQ4_XS} && RDNA4 && M>=256 &&
   K%256==0 && N%32==0 && contiguous && !ids;
3. 分派: repack 得到虚拟 Q8_0 buffer 后, 构造一个 shims:
   直接调用 launch_mul_mat_q<GGML_TYPE_Q8_0, J...>(ctx, args_q8) ——
   但 args 里 type 字段/stride 要按 Q8_0 重算 (block 尺寸 34B vs 原 type)。
   **实现捷径**: 构造一个临时 ggml_tensor 视图 (type=Q8_0, data=repack buf,
   ne/nb 按虚拟形状) 直接调 ggml_cuda_mul_mat_q_q8_0 路径?
   mmq.cu 的 case GGML_TYPE_Q8_0: mul_mat_q_case<GGML_TYPE_Q8_0> ——
   它读 src0->data/src0->type/src0->nb。构造 fake tensor 即可。
   **fake tensor 的生命周期**: repack cache 持有 buf, fake tensor 只在
   分派函数栈上, 无泄漏。
4. MMVQ/decode 路径不碰 (M<256 不进 gate)。

### 4.7 分派实现定案 (基于 stride 语义核实)

- `s01 = src0->nb[1] / ts_src0` 是 **block 数/行**; load_tiles 里
  `bxi = (block_q8_0*)x + kbx0 + i*stride + kbx` 按块索引。
- 虚拟 Q8_0 tensor 的参数: buf 大小 = N * K/32 * sizeof(block_q8_0);
  args.x = buf, args.type_x = GGML_TYPE_Q8_0, args.stride_row_x = K/32。
- **不需要 fake ggml_tensor**: 直接把 repack 后的 buf + type_x=Q8_0 填进
  mmq_args, 再调 ggml_cuda_mul_mat_q_switch_type (mmq.cu:410) ——
  它 switch(args.type_x) 到 mul_mat_q_case<GGML_TYPE_Q8_0>, 后者走
  should_use_mmq/launch 路径, 输入就是纯数据。
- 注意: ne00 (K) 不变, ne01 (N) 不变, ne11/rows 等不变 —— 只有
  x 指针/type_x/stride_row_x 三项被替换。repack 缓存 key = src0->data
  + type (同一指针不同类型理论冲突, 实际 key 加 type)。
- y 侧完全不变 (q8_1 activation 由既有 quantize 路径生成)。

### 4.8 分派层编译通过 (2026-10-02)

mmq.cu 修改三处:
1. #include decode-repack.cuh;
2. gate + cache + dispatch helper (ggml_cuda_mul_mat_q_decode): gate = RDNA4 &&
   M>=256 && K%256==0 && type∈{IQ3_S,IQ3_XXS,IQ2_XXS,IQ4_XS} && !ids;
   cache key = src0->data (per-type 无冲突, 同指针不同 type 理论上不同张量);
   伪造 Q8_0 tensor 视图 (nb[0]=34B, nb[1]=(K/32)*34B, buffer 指针保留,
   USAGE_WEIGHTS 时 padding memset 被跳过), **递归调 ggml_cuda_mul_mat_q**
   —— gate 排除 Q8_0, 无递归环;
3. 进程退出 dtor 释放 decode cache。
decode-repack.cuh 头文件同步为四个 per-type wrapper (删除早先的
ggml_cuda_decode_repack_q8_0 泛型占位)。

**v2 递归方案优于第一版 args_of 占位**: 复用整个 mul_mat_q 主体
(quantize y/padding/switch_type), 代码路径与真实 Q8_0 完全一致。

## 5. 端到端验证 (2026-10-02)

### 5.1 PPL 逐位一致达成

GSQ IQ3_S 模型, --chunks 4 -ub 2048: **PPL = 6.0175 +/- 0.14727**
与基线 (无解码路径, §10.8 的 6.0175) **逐位一致**。
预解码的纯整数重排在真实模型上零数值影响 —— 四个 kernel 的
staging 数学复刻完全正确。

单测: MUL_MAT type_a=iq3_s / iq4_xs / iq2_xxs 全 OK。

### 5.2 下一步: 性能 A/B

PPL 门过了, 现在跑同会话 A/B (decode ON vs GGML_RAD-like 关闭路径)。
注意: 本路径无独立 env 开关, A/B 用 "当前库" vs "Q2FIX 库 (无 decode)"。

### 5.3 A/B 结果: 解码路径未被触发 (诊断)

DECODE 库 pp2048 = 1281-1285, Q2FIX = 1281-1296 —— **无差异**, trace 显示
IQ3_S 仍走原 mul_mat_q<iq3_s> kernel (89.5 TF/s, 无 Q8_0 kernel 出现)。
分派 gate 没命中。排查优先级:
1. M 计算: src1->ne[1] 在 llama-bench pp2048 下是 2048, 应 >= 256;
2. K%256: IQ3_S 张量 K=17408 (68*256) ✓ / 5120 (20*256) ✓;
3. 递归后 src0 是 fake Q8_0, gate 排除 Q8_0 -> 不会循环;
4. **疑似点: ggml_cuda_mul_mat_q 之前还有上层 dispatch (ggml-cuda.cu
   的 mul_mat 分派), decode helper 是从 mul_mat_q 里调的吗?**
   —— 已把 helper 插在 ggml_cuda_mul_mat_q 定义前, 但**没有在
   ggml_cuda_mul_mat_q 主体里调用它!** (stage-1 只插了定义)

### 5.4 bug 修复: helper 忘了挂进主体

stage-1 只插入了 helper 定义, 没在 ggml_cuda_mul_mat_q 主体加调用行
(radiance 调用后面加 decode 调用)。编译后 trace 确认路径未生效才发现。
经典低级错, 但 rocprofv3 trace 5 分钟就定位了。
已补调用 + 重编译。

### 5.5 新问题: OOM (ROCm out of memory)

解码路径激活后 PPL 跑到一半 abort: "ROCm error: out of memory"。
原因明确: 解码副本 IQ3_S 4.31 + IQ3_XXS 3.43 + IQ2_XXS 0.31 + IQ4_XS 2.84
= +10.89 GB 驻留在运行时分配 (cudaMalloc, 不在 ggml 内存池), 加上
原模型 12.1 + KV/compute ~8-10 GB -> 超 32 GB。

**这正是 §12 预算表警告的场景**: 10.89 GB 副本 + compute buffer 大
(ub 2048 + fa + 4096 ctx) 超限。bench 之所以能跑是因为 n_ctx=0 (无 KV)
且单 pass; PPL 4 chunks 有 KV + compute 峰值。

**缓解选项:**
a. PPL 用 -ub 512 / 更小 ctx 验证正确性 (显存压力小) -> 立即可做;
b. IQ4_XS 先移出 gate (省 2.84 GB) -> 若仍 OOM;
c. IQ3_XXS 移出 (省 3.43) -> 组合裁剪到预算内;
d. 最终方案: env 开关 GGML_DECODE_TYPES 白名单, 用户按显存自选。

### 5.6 IQ3_S 单类型: 显存 OK 但 PPL = NaN (数值 bug 在 kernel)

env 白名单后 IQ3_S-only 显存不再 OOM (rc=0), 但 **PPL 全 NaN** ——
解码 kernel 的输出在真实张量上是错的 (MUL_MAT 小单测过不了 K=17408 的
真实块, 单测 shapes 太小没暴露)。

**根因排查 (最可能):** iso probe 用的是单 superblock 的 host 复刻, 而
kernel 的 get_int_b2_d 假设 4 字节对齐 —— block_iq3_s 是 130 字节/块,
**qs 字段在块内偏移 2 (d 是 fp16), 相邻块的 4*kqsx 读取会跨块错位**?
不对: staging 用 (block_iq3_s*)x + kbx0 + i*stride, 也是块指针, 相同语义。
第二嫌疑: **s01 传错** —— 我传了 src0->nb[1]/ts_src0 (块数/行), staging
一致。第三嫌疑: **q3_packed[2] 的对齐** —— int 数组在栈上, 取址作
uint8_t* 读, 值正确。
**结论: 需要第二个 iso probe, 用 GPU kernel 的输出 vs staging kernel 输出,
在真实块序列 (>= 2 块) 上比对** —— 现有 iso 是 CPU 复刻, 没测 GPU kernel 本身。

### 5.7 NaN 根因假设 (在写 iso2 时发现)

对比 iso probe (PASS) 与 production kernel 的 scale 语义:
- staging 的 x_df = **ls*d** (fp32 乘法), 数值 ~1.0 量级;
- production write_q8_0 存 `ob->d = __float2half(ls*d)` —— **fp16!**
  但 Q8_0 的 dequant 语义是 x = d * qs_byte, d 是 fp16。
  **ls*d 在 fp16 下**: ls ∈ {1,3,5,...,31}, d ~ 0.01 量级
  (fp16 能表示), 乘积数量级 OK。fp16 舍入 vs 原 fp32 -> **非逐位**
  但误差 ~1e-3 相对, 不该 NaN。
- NaN 的来源更可能是 **apply_signs 的符号处理**: iq3s_grid 值域是
  0x01..0x7F 量级 (4bit 格点), 但我的 apply_signs 在 s=0xFF 时做
  (0-g)&0xFF = 256-g —— 若 g=0 则 0; 若 g 有 0x80 位 (q3 数据是 3bit
  拼 8bit, 含高位), **原 staging 的 XOR+SUB 等价于取负, 我的模拟也一致**。
- **最大嫌疑: fake tensor 的 nb[2]/nb[3] 在 ne[2]==1 时残留了原值**
  —— 我只在 ne[2]>1 时重算 nb[2], 但 v = *src0 浅拷贝已带原 nb,
  ne[2]==1 时 nb[2] 保持原块的字节 stride (IQ3_S 的 130B/块),
  **而 Q8_0 语义下 nb[2] 应该 = N*34B**。ne[2]==1 时 MMQ 不读 nb[2]? 
  读 s02 = src0->nb[2]/ts —— nchannels 用。GGML_ASSERT(ne02==1) 的
  2D 情况下 s02 无用。PPL 是 3D? 检查 PPL 张量的 ne[2]。
**下一步: 修 fake tensor 的 nb[2]/nb[3] 无条件重算 + 重测。**

### 5.8 iso2 probe 构建受阻 + 改用最小 GEMM 对拍

iso2 standalone 构建要链接 iq3s_grid (__device__ 表在 decode-repack.o 里,
独立 probe 引入 ggml-common.h 又与 fork 头文件环境耦合)。
**改用更直接的对拍**: 不做单元 iso, 直接 end-to-end 差分——
用 GSQ 模型但只开 IQ4_XS (另一个 kernel), 若只 IQ4_XS 时 PPL 不 NaN
则 IQ4_XS kernel 对、IQ3_S kernel 错; 二分定位。同时试 IQ2_XXS。

### 5.9 二分结果: 全部 kernel 都 NaN -> 系统性错误在公共路径

IQ4_XS-only 和 IQ2_XXS-only 都 NaN => 问题不在某个类型的数据流,
而在**公共分派**: fake tensor 递归进 ggml_cuda_mul_mat_q 后,
某个假设不成立。最大嫌疑:
1. **ne10_padded / y 量化**: 递归调用会重新走 quantize y (q8_1) ——
   应该没问题;
2. **ggml_nbytes / buffer usage**: v.buffer = src0->buffer 但 v 的
   nbytes 是 Q8_0 尺寸 > 原 tensor -> buffer 溢出检查?
3. **switch_type 的 case GGML_TYPE_Q8_0 里 ne00=K 一致**;
4. **fallback = ne01 % 128 != 0**: N=17408 -> false, 但
   ne01 = N 而 fake 的 ne01 没变;
5. **最可疑: ggml_backend_buffer_get_alloc_size(v)** —— 用 v.type(Q8_0)
   计算 alloc size, 而 v.buffer 是原 WEIGHTS buffer, size 与 Q8_0
   尺寸不匹配 -> 触发 clear-padding 分支的 memset 越界! 检查:
   branch 条件 usage==COMPUTE; weight buffer != COMPUTE, 跳过。
6. **另一个: M 的定义**: gate 里 M = src1->ne[1]; 递归后 v.ne[1] = N 不变;
   dst 相同。
7. **最最可疑: IQ3_S 大 fake tensor 的 data=buf 只有 N*K/32*34B; 而
   MMQ 的 src0 读取按 nb 走, OK。但 ggml_nbytes(v) 用于 cache? 无。**

**决定: 不再猜, 写"复现最小化" —— 用 llama-perplexity 单块 + 环境变量
只开一个类型, 再用 GDB/printf 检查 decode kernel 输出是否全零/乱码。**
更快的路径: 在 decode kernel 后加 fprintf 校验和 (debug builds only)。

### 5.10 校验和揭示双 bug (2026-10-02)

IQ4_XS-only 的 decode 输出校验和:
1. **192/256 零值** —— 8 个 sub-block 里 6 个全零。IQ4_XS 的 row[64] 填充
   只覆盖 int k0/k0+4 对应的 32 字节里的一半; 我的 row[4*k0+j] 公式里
   kqsx=0..3 时 k0=0..3 -> row[0..15]+row[16..31], 但 row[32..63] 需要
   kqsx=4..7 的 k0=8..11 -> row[32..47]+row[48..63]。
   检查: 4*k0+j for kqsx=4 (k0=8): row[32+j] ✓; row[4*k0+16+j] = row[48+j] ✓。
   但 sum=0 的块 = row[32..63] 没被填 -> **kqsx=4..7 的 get_int_b2_d(qs, kqsx)
   读了 qs[64B] 之外? get_int_b4(qs, kqsx) = int 索引 kqsx -> 字节 4*kqsx:
   kqsx=4 -> bytes 16..19! 不是 32..。所以 row[32..63] 的填充用的 nibbles
   是 byte 16..19 的 —— 那 src 里 qs[] 只有 64B, kqsx 4..7 读 bytes 16..31,
   覆盖元素 64..127?? 布局又错了。
2. **first_d = -0.0** —— d*(ls-32), ls=32 时 dl=0 (负零), 这是合法的
   (ls=32 即 scale 因子 0)。但全零块 + 大量 -0 说明多数 ls=32 ->
   **scales_l/scales_h 的位提取公式有错** (真实权重 ls 不该都是 32)。

**两个 bug 都在 IQ4_XS kernel 的位提取。而 Q2_K/IQ3 系未验证。**
时间成本上升 -> **决策: 保留代码与文档, 本轮先回滚 decode 分派调用
(gate 行保留但 helper 不调用), 避免半成品进库; 下一轮用 GPU-vs-GPU
iso 逐 kernel 修 (需先解决 __device__ 表的 standalone 链接问题)。**

## 6. 本轮收官状态 (2026-10-02)

### 6.1 已回滚到安全态

- 分派调用被 `if (false && ...)` 禁用 (代码保留, 下一轮启用);
- 调试 printf 移除;
- 重编译后 PPL 回到 **6.0175 逐位一致** (基线恢复确认);
- 工作树: mmq.cu 修改 (含禁用的分派) + decode-repack.{cu,cuh} 未跟踪 ——
  **均为未完成代码, 不提交**。

### 6.2 本轮技术资产 (下一轮直接用)

1. `decode-repack.cu/.cuh` 骨架 (4 kernel + wrapper + 分派 + cache + dtor);
2. IQ3_S 的 CPU/GPU iso probe 方法论 (PASS 的那份);
3. `get_int_b2/b4`、`unpack_ksigns`、signs byte 语义、
   iq3s_grid/iq2xxs_grid 值域的完整推导;
4. IQ4_XS 的两处已定位 bug: nibble->element 映射错位 + ls 位提取
   (scales_l/scales_h 的 kqsx 索引需要按 staging 的 threadIdx.x%8 重新核对);
5. 显存预算表 (§12) 与 env 白名单机制 (GGML_DECODE_TYPES)。

### 6.3 下一轮的开工清单

1. 修 IQ4_XS: 先读 staging 的 scales_l/h 提取式逐行核对
   (threadIdx.x%8 的映射是 per-32 还是 per-64);
2. 每类型 kernel 写完立即跑 GPU-vs-GPU iso (需要解决 __device__ 表
   的 standalone 链接 —— 或者把 staging 数学复刻进 probe, 避免链接);
3. 全部类型 iso PASS 后才允许打开分派;
4. PPL 逐位一致 + 显存峰值测量 (amd-smi 轮询) + A/B;
5. 提交策略: decode-repack.{cu,cuh} + mmq.cu gate/dispatch 一起提交,
   commit message 引用本文档。

## 7. iso2 (GPU-vs-GPU) 落地并揭示真正的 bug (2026-10-02)

### 7.1 自测机制建成

decode-repack.cu 现在内置:
- ref_stage_*_kernel: 逐行复刻 production staging 数学 (同一个 TU, 无链接问题);
- ggml_cuda_decode_repack_selftest_{iq3_s,iq4_xs}: 随机块 (3 行 x 2 superblock)
  双路径对拍;
- extern "C" ggml_cuda_decode_repack_run_selftests(): dlopen 驱动调用。

首轮结果: **iq3_s 432 mismatches, iq4_xs 432 mismatches** (864/864 全错)。

### 7.2 IQ3_S 首个 mismatch 的诊断 (决定性)

ref=f3f709f9 got=f709070f: **字节循环移位关系**。根因:
- staging 的 qs_packed = int2{b2(qs,2k), b2(qs,2k+1)}, q3 byte 序列是
  **[b(4k), b(4k+1), b(4k+2), b(4k+3), b(4k+4), ..., b(4k+7)]** (8 bytes);
- 我 kernel 的 qs_packed[2] = {b2(qs,2k), b2(qs,2k+1)} 相同, 但随后
  `qs[2*l+0]` 索引 q3 bytes 0..7 ✓ 相同...
  差异在 d: ref=0x1.97bep+18 (~4.2e5) vs got=0x1.658p+3 (~5.7)。
  **ref 的 d 是 ls*d, ls∈1..31, d fp16 ~0.001-2 => 4.2e5 只能是 fp16 位
  解析错误** —— ref 用 `__half2float(*(const __half*)&bxi->d)`, got 用
  `half_bits_to_float(bxi->d)` —— 相同。
  **真正的差异: scale 的 kqsx/2 索引**: 我 kqsx/2 与 staging 相同...
- **v0-v7 全错而不是部分 -> 是"错位一拍"式的**: got=f709070f 看着像 ref
  的字节流右移。**结论: qs_packed 的读取偏移错了 4 字节** —— staging 的
  int2 = {b2(qs,2k), b2(qs,2k+1)} 是 bytes 8k..8k+7 (不是 4k..4k+7!),
  get_int_b2 的 i 单位是 4 字节: b2(p,i) = bytes 4i..4i+3。所以
  b2(qs,2k) = bytes 8k..8k+3, b2(qs,2k+1) = bytes 8k+4..8k+7 —— 相同。
  但我的 get_int_b2_d(x16[2i],x16[2i+1]) = bytes 4i..4i+3 ✓ 相同。
  -- 那为什么全错? **threads_per_row=8 但每 256 元素块是 32 个 32-子块;
  kqsx 索引的是 64 字节 qs 的一半!** block_iq3_s 的 qs 是 104 字节
  (13*QK_K/32 = 13*8 = 104), staging 线程 kqsx 读 bytes 8k..8k+7 +
  signs 4k..4k+3。我的 kernel 读 b2(qs,2k) = bytes 8k..8k+7 ✓ 一致。
  mismatches 意味着还有别的错。**下一步: dump ref 与 got 的全部 64 字节,
  找对齐偏移模式。**

## 8. iso2 破案: ggml_half 在 HIP 下是 __half 类 (根因, 2026-10-02)

### 8.1 谜团收敛过程

selftest 显示 host[0].d 打印 0x7640 而代码写 0x6400; d mismatch 的
ref = 25600 / got = 1024, 比例 25 = 1+2*12 的 ls 值域陷阱……
最终在 mini.cpp 隔离复现 (host 逻辑 20 行, 结果 d=0x6400 正确) 与
decode-repack.cu (d=0x7640) 的**唯一差异**中找到根因:

**HIP 下 `ggml_half` 不是 uint16_t 而是 `__half` 类!**

- `b.d = (unsigned short)(0x6400 + fill_idx)` → 右侧 25600 (int) 
  隐式转 __half → **位模式 0x7640** (fp16 25600) —— 不是 0x6400!
- selftest 读回 `__half2float(bxi->d)` = 25600 ✓ (ref 是"对的")
- decode kernel `write_q8_0` 的 `ob->d = __float2half(ls*d)`:
  ls*d = 1 * 25600 = 25600 → fp16 位 0x7640 → memcpy 写入 ✓
  → **decode 输出的 d 位 = 0x7640 与 ref 一致** (我之前打印 got=1024
  是因为 mismatch 分支里 got 的读法在旧二进制; 最新一轮 ref/got 的
  d 都 = 25600 同源, 剩余 184 mismatch 是别的)

### 8.2 对方案的修正

- **fill/host 侧**: 写 fp16 位模式必须用 memcpy 或 `__half` 构造, 
  不能把 int 位模式赋给 ggml_half (会被当 float 值转换)。
- **decode kernel 侧**: 从 src 读的 scale 位直接 memcpy/位拷贝传递,
  **不做 float→fp16 的往返** (fp16→fp32→fp16 对 normal 数无损,
  但 repack 时 scale 本来就是 fp16 位, 直接拷贝字节最安全)。
- 之前 5.10 的 NaN 根因同源: decode kernel 把 scale 当 float 往返,
  在真实权重的极端 scale 下产生 subnormal/溢出偏差 → PPL 全崩。

### 8.3 下一步 (明确且有限)

1. decode kernel 全部改为 **scale 位直接拷贝** (uint16 原样), 
   数值计算只在 fp32 中间态; 验证 ref/got 全部位一致;
2. IQ4_XS 的 nibble→element 散布按 row-rebuild 方案核对 (§4.4);
3. iso PASS 后打开分派 → PPL 逐位 → A/B → 显存峰值 → 提交。

## 9. 本轮调查结论与状态快照 (2026-10-02 深夜)

### 9.1 selftest 机制建成并揭示真问题

在 decode-repack.cu 内内置 ref_stage_*_kernel (production staging 数学逐行复刻)
+ selftest (随机块双路径对拍) + extern "C" selftest 入口 (dlopen 驱动)。
这解决了 standalone probe 无法链接 __device__ 表的问题。

### 9.2 当前 mismatch 状态

- IQ3_S: qs bytes 与 d 值 **已全部对齐** (修正 __half 隐式转换 bug 后);
- IQ4_XS: 31/32 个 kqsx 正确, **仅 kqsx=2 的 v0..v7 mismatch**:
  ref bytes 含 03/05/ff 非 LUT 值, got bytes 含 LUT 值 (0d, 81, ff 混合)。
  → 需要打印 aux 输入定位 (下一轮第一步)。
- mismatch 总数 331 → 大部分是 kqsx=2 相关的级联 (selftest 的 got 与 ref 布局差一位)。

### 9.3 本轮学到的重要事实 (存档)

1. **HIP 下 `ggml_half = half` (类, 不是 uint16_t)!** 
   `b.d = (unsigned short)(0x6400)` 会隐式转 half(25600.0) → 位 0x7640!
   **int 位模式必须用 memcpy 写入 ggml_half 字段** —— 这是整个 NaN/inf
   连锁反应的总根因。
2. `printf("%04x", half_var)` 走 half→float→int 提升, 打印的是**值的十进制
   hex**, 不是位模式。调试时必须先 memcpy 到 unsigned short 再打印。
3. `get_int_b2(p, i)` = uint16[2i] | uint16[2i+1]<<16 = **字节 4i..4i+3**
   (不是 2 字节!); get_int_b4 相同。两者历史上就等价。
4. `unpack_ksigns(v)`: 7-bit + popcnt 奇偶折叠到 bit7, 再 *0x01010101 广播。
5. fp16 溢出检查: selftest 的 fill 值必须保证 ls*d <= 65504, 否则 inf 是
   正确行为不是 bug。
6. IQ4_XS 的 staging 散布: kqsx 0..31 (32 线程), k0 = 8*(kqsx/4)+kqsx%4,
   v.x/v.y 分别写 ints k0/k0+4; v.x = LUT[nib 0,2,4,6], v.y = LUT[nib 1,3,5,7]。

### 9.4 未完成项 (下一轮)

1. IQ4_XS kqsx=2: dump aux 与 ref/decode 的中间 v[] 值, 定位单点差异;
2. IQ3_S/IQ3_XXS/IQ2_XXS 的 selftest (iq3_s 已对齐, 其余两个待写);
3. 全类型 iso PASS → 打开分派 → PPL → A/B → 显存峰值 → 提交三分支。

## 10. 继续调试: IQ3_S kernel 也 NaN (2026-10-02 续)

启用 IQ3_S+IQ4_XS 分派后 PPL 仍 NaN → **IQ3_S kernel 也有 bug**
(selftest 的小随机块对拍没暴露)。

时间与复杂度评估: 到此为止本轮已修复/确认的链条:
1. ggml_half 是 __half 类, int 赋值被 float 转换破坏 (根因 A, 已修);
2. half_bits_to_float 经 unsigned short 传参同样被破坏 (根因 B, 已修:
   改为直接传 ggml_half);
3. selftest 的 ref/got 比较方法学有缺陷 (production staging 的 tile 布局
   是交错的, "逐 int 比较"必须连散布一起复刻, 而 selftest 的 host fill
   也经过 __half 隐式转换 → 两边数据源本身不一致 → 对拍结果不可信)。

**结论: selftest 方法需要重写 (用 production dequantize_row_iq*_s 作为
golden, 比较 dequant 后的浮点值而非中间 tile), 这是一轮新的工作。**
本轮分派保持禁用回滚, 基线安全。

## 11. 基线恢复确认 (2026-10-02)

分派禁用后重编译, PPL 回到正常 (c=2048/chunks=2 口径: 2.2997, 
与该口径的基线一致 — NaN 消失, 解码路径不再被调用)。
工作树保持: mmq.cu 修改 + decode-repack.{cu,cuh} 未跟踪, 均为
下一轮的半成品, 不提交。

### 11.1 下一轮的正确打开方式 (吸取本轮教训)

**selftest 必须以 production dequantize 为 golden 重写:**
- golden: 对同一随机块调 dequantize_row_iq3_s/iq4_xs (ggml-quants.c
  的 host 函数, 已被 test-backend-ops 验证多年) 得到浮点权重;
- 待测: decode kernel 的输出 ob (Q8_0) 也 dequant 成浮点 (ob.d * qs);
- 比较: 两组浮点权重逐位 (或 max abs diff = 0) —— 这比对中间 tile
  可靠得多, 因为 golden 函数本身就是"正确语义"的定义。
- fill 数据: 必须用 memcpy 写 fp16 位 (ggml_half = __half 的教训)。

### 11.2 本轮净状态

- 无用户可见变化 (分派禁用, 基线无损);
- 交付的代码资产: kernel 骨架 + selftest 机制 + 完整的调查记录;
- 已确认的根因: ggml_half=__half 类的隐式转换 (修了), 
  fp16 溢出行为确认 (不是 bug), selftest 方法论缺陷 (待重写)。
