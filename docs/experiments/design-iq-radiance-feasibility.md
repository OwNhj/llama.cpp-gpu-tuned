# 评估: GSQ 类 IQ/K-quant 能否做 radiance 式改造 (2026-10-01)

## 0. 结论速览

**仿照 radiance 做 per-32 幂次缩放折叠: 对 GSQ 的 8 种类型全部不可行 (精确性上)。**
**但"更快"是可行的, 且已经部分发生** —— IQ/K 系的 MMQ 已经在用 int8 WMMA 张量核,
当前瓶颈不是指令, 而是每权重解码成本和 tile 组织。上限估计:

| 策略 | 每 forward MMQ 时间 (2048 tokens) | pp2048 预期 |
|---|---|---|
| 现状 (混合 8 类型 MMQ) | 1183 ms (实测) | 1294 |
| 全部类型达到 IQ2_S 水平 (139 TF/s) | ~700 ms | ~1800 |
| 全部类型达到 Q4_K 水平 (167 TF/s) | ~580 ms | ~2100 |
| 全部走 radiance 式 fp8 WMMA (252 TF/s) | ~435 ms | ~2900 (不可达, 见 §2) |

## 1. 已查清的事实 (源码级)

### 1.1 radiance 快的根本原因 (重述)

radiance 快不是因为用了 fp8 WMMA —— MMQ 的 MXFP4 路径也用同一条
`v_wmma_f32_16x16x16_fp8_fp8`。radiance 快在于三点:

1. **E8M0 scale 折叠进权重**: per-32 的 2 的幂缩放乘进 e4m3 权重字节 (含 subnormal 下探),
   内层循环零 rescale。
2. **W tile 一次读入复用整个 M tile** (TN=4 时 M=2048), 权重带宽压力摊薄。
3. 内层循环 3.1 指令/WMMA (198 指令含 64 WMMA)。

### 1.2 IQ/K 系的缩放不是 2 的幂 -> 折叠不精确 (硬障碍)

- IQ2/IQ3/IQ4_NL/IQ4_XS: **码本查找** (iq2nl/iq3s_grid/kvalues_iq4nl), 根本没有
  "乘一个 scale"的连续值, 所谓 scale 是 fp16 的 `d`/`dmin`。
- Q2_K-Q6_K: fp16 scale (+ Q2_K 额外 min), 同样非幂次。
- 把 fp16 scale (10 位尾数) 折进 e4m3 (3 位尾数) 权重 = 丢 7 位精度。
  radiance 自己的文档就记录了: 即使 E8M0 (无损) 折叠, d=10 的块也要靠 subnormal
  才保住精度; fp16 scale 折叠是数量级更糟的截断。**要保精度只能改用真乘法,
  那就把 radiance 的核心优势 (零内层 rescale) 丢掉了。**
- 对照: MXFP4 的 E8M0 是纯 2 的幂, 折叠在 fp32/e4m3 下**精确**, 这是 radiance
  只服务 MXFP4 的根本原因, 不是偷懒。

### 1.3 但 GSQ 的类型已经在用张量核 (重要发现)

逐类型查 `mmq-vec-dot.cuh` 的 MMA 分支 (RDNA4 上 use_mma_data_layout 成立):

| 类型 | MMA vec_dot | 实际指令 | 实测 TF/s |
|---|---|---|---|
| IQ3_S / IQ3_XXS / IQ4_XS / IQ4_NL / IQ2_XXS | `q8_0_q8_1_mma` | **`mfma_i32_16x16x16i8` (int8 张量核)** | 92-108 |

> 勘误 (2026-10-01, 用户指正): **dp4a 是 bs=1 时的 decode (MMVQ) 路径, 与本表所述的
> prefill MMQ 无关。** 上面说的"已经在用张量核"仅指 prefill 的 MMQ 路径;
> decode 走的 MMVQ 仍是 dp4a。
| IQ2_XS / IQ2_S | `q8_0_16_q8_1_mma` | int8 张量核 (16bit 解码) | 68-139 |
| Q2_K | `q2_K_q8_1_mma` | int8 张量核 | 23.1 (修复后) |
| Q3_K | `q8_0_16_q8_1_mma` | int8 张量核 (16bit 解码) | 95.7* |
| Q4_K / Q5_K / Q6_K | `q8_1_q8_1_mma` / `q8_0_q8_1_mma` | int8 张量核 | 163-167 |

("16bit 解码" = 先把码本值展开成 int8 tile, 数据量翻倍, 是 IQ2_XS/IQ2_S 慢的原因)

**所以"仿 radiance 换指令"是伪命题** —— 指令已经是张量核。差距在别处:

1. **每 32 权重一个 fp16 scale 的 rescale 乘法** (radiance 已折叠掉的) 仍在内层循环。
2. **每权重解码**: IQ3_S 要查 33 项码本 + 组装, MXFP4 只是 nibble 展开。
3. **tile 组织**: MMQ 的 W tile 读法不如 radiance 的 A-fragment 直读 + W-LDS 复用。
4. **寄存器压力**: Q2_K 修复前甚至溢出 (这就是刚才修的)。

### 1.4 数量级核对 (为什么说上限 ~2100 而非 2900)

当前混合 MMQ 1183 ms 中, 每类型的时间 = params/TFLOP/s。若把所有类型抬到
Q4_K 水平 (167 TF/s, 同为 int8 张量核 + fp16 scale rescale, 说明可达):
total = 2 x 26.78e9 x 2048 / 167e12 = 657 ms -> pp2048 ~2100。
而 252 TF/s 需要同时做到零 rescale + 码本消除 -> 要求换量化格式本身。

## 2. 可行路线 (按性价比排序)

### 路线 A: 提高各类型的 tile/寄存器配置 (低风险, 已验证方向)

Q2_K 修复就是这类 (+58%)。系统做法: 对 8 种类型逐个做 tile sweep
(nt x I x J 组合, 以 Scratch_Size=0 为约束), 修掉"默认配置不是最优"的类型。
**IQ2_XS (68 TF/s) 和 Q2_K (23) 是明显欠额的, IQ3_S (92) 可能也有空间。**
工作量: 每个类型一轮 sweep + bench。预期总收益 pp2048 +20-40%。

### 路线 B: radiance 式 "W-tile 复用 + A-fragment 直读" 移植到 int8 MMQ (中风险)

不改量化格式, 只改 kernel 组织: 把 radiance 的 atiled 骨架 (W tile 进 LDS,
activation 走 fragment, TN 轴摊 W 带宽) 套到 `q8_0_q8_1_mma` 的 iu8 WMMA 上。
i8 WMMA 的 K=16, 峰值 329 TF/s 同源。**难点**: fp16 scale 的 rescale 必须留在
内层 (或挪到 W-LDS 写入时做一次 —— 这是值得试的变体, 等价于"折叠到 LDS tile")。
若成功, 各类型逼近 200+ TF/s, pp2048 ~2600。
工作量: 一个新 kernel 模板 + 每 W-tile 一次 scale 预乘, 中等。

### 路线 C: 换量化格式 (重 quantize, 动模型文件)

把 GSQ 模型逐张量重quantize 成 MXFP4 (llama-quantize --allow-requantize,
警告: 从已量化源重量化质量损失大, 必须从 bf16 源重来)。得到 14.5 GB (vs 12.1 GB),
prefill 直接吃满 radiance 252 TF/s -> pp2048 ~2900+。
**代价**: 文件变大 20%, 且需要 bf16 源。**这不是"改造 kernel"而是"换模型"**,
质量取决于重量化 recipe (用户的逐张量 --tensor-type 流程已验证过一轮)。

## 3. 推荐

1. **先做路线 A 的 IQ2_XS/Q2_K sweep** (半天级工作量, 零精度风险, 纯配置)。
2. **路线 B 作为主菜**: "scale 预乘进 LDS tile" 是 radiance 思想对非幂次 scale 的
   正确迁移 —— 不要求精确折叠, 只要求把 rescale 从每-activ-tile 一次降到每-W-tile
   一次, 数学上无损 (fp16 scale x int8 码本值仍在 fp32 累加)。**这是唯一能逼近
   radiance 效果而不换格式的路。**
3. 路线 C 仅当用户接受 14.5 GB + 重量化质量损失时考虑。

## 4. 证据文件

- 源码: `mmq-vec-dot.cuh` (MMA 分支逐类型表), `mma.cuh:1372-1390` (iu8/i8 指令),
  `mmq-config-rdna4.cuh` (tile 表), `radiance-gemm.cu` (折叠与 repack)。
- 实测: `/tmp/iq3tr` (fork trace), `/tmp/offtr` (official trace), §10.8 的 Q2_K A/B。
- 官方同缺陷确认: `llama.cpp-ROCm` @ a3f241dd0 Q2_K 同为 2.9 TF/s + 2128 scratch。

## 5. 行动计划 (2026-10-01, 用户批准)

执行顺序: 路线 A sweep -> 路线 B 移植。路线 C 挂起待定。

### 5.1 路线 A: 欠额类型 tile sweep

- 对象: IQ2_XS (68 TF/s) 优先, 其次 Q2_K (23.1)、IQ3_S (92)。
- 方法: 对每类型改 `mmq-config-rdna4.cuh` 的 (nthreads, I, J) 组合,
  约束 Scratch_Size=0 (rocprofv3 验证), 以 bench 吞吐为判据。
- 已知可行样例: Q2_K I=64 高 J (+58% pp2048, commit 4f7f0b5e3)。

### 5.2 路线 B: radiance 骨架移植到 int8 MMQ (主菜)

核心思想: **scale 预乘进 LDS tile** —— 把每-activ-tile 一次的 fp16 rescale
降到每-W-tile 一次 (写 LDS 时做)。数学无损: fp16 scale x int8 码本值仍在
fp32 累加器汇合。
骨架来源: `radiance-gemm.cu` 的 atiled (W tile 进 LDS, activation 走
A-fragment 直读, TN 轴摊 W 带宽)。
指令: `mfma_i32_16x16x16i8` (K=16) 已在 mmq 内, 峰值 329 TF/s 同源 fp8 路径。
目标: 各类型逼近 200+ TF/s, pp2048 ~2600 (混合模型口径)。

### 5.3 Q2_K 剩余优化路径 (当前 23.1 TF/s, 同侪最慢 68)

1. **nt=128/I=64/J=128**: 每线程 64 累加器 (现 32), 需实测寄存器是否 <256。
2. **专用 LDS 布局**: Q2_K 的 scale+min 双辅助量目前按 Q8_0 布局 pad,
   重新排布可减寄存器占用。
3. **db 预取双缓冲**: x tile 读与 WMMA 重叠。
4. **放弃 iu8, 试 fp8 直路**: Q2_K 解码到 fp8 而非 int8 (损失 <1 LSB Q2),
   换取 329 TF/s 峰值 —— 仅当 1-3 都不够时考虑, 精度需 PPL 门。

## 6. 移植前期调查结果 (2026-10-01, ISA 级)

### 6.1 已确认的机制

- **码本表**: `iq3s_grid`/`kvalues_iq4nl` 经 `GGML_TABLE_BEGIN` 在 HIP 下是
  `static const __device__` 数组 —— **global memory**, 不是 constant bank
  (对比: radiance 把 kMag 显式放进 `__shared__ sMag[32]`, 注释写明
  "ds_load keeps the lookup off the loadcnt chain")。
- **每 tile 的解码成本**: IQ3_S 每 thread-slab 8 次 `iq3s_grid[]` global 查表 +
  2 次 `__vcmpne4` 符号展开; Q4_K 零查表 (nibble shift); IQ4_NL/XS 用
  `get_int_from_table_16` (v_perm 4 查表/指令, 较快)。
- **scale fixup 在 C-tile 累加侧**: `q8_0_q8_1_mma` 内层每次 WMMA 后做
  `sum += C * dA * dB` (x_dl[i*sram_stride+k0/QI8_0] 每行每块一次浮点读 +
  128 cvt + 126 mul + 124 fmac / kernel)。

### 6.2 ISA 指令配比 (决定性证据)

| | int8 MMQ (Q2_K 实例, 0x5c900, 8016 instr) | radiance atiled (21044 instr) |
|---|---|---|
| 张量核 | 32 x `wmma_i32_16x16x16_iu8` | 64 x `wmma_f32_16x16x16_fp8` |
| scale fixup | ~378 (128 cvt_f32_i32 + 126 mul + 124 fmac) | **0** (folded) |
| LDS | 18 ds_load_b128 + 18 ds_store_2addr | W tile 全在 LDS, A 走 global 直读 |
| 寄存器压力 | VGPR 224-256 | 更高但已调优 |

**每 WMMA 摊 ~12 条 fixup 指令** —— 这是 int8 MMQ 与 radiance (0 fixup)
的主要指令差距来源。radiance 的优势不在指令选择, 在**数据组织**:
scale 在写 LDS tile 时一次折完, 内层纯 WMMA。

### 6.3 移植设计 (路线 B 具体化)

**方案 B1 "LDS 内预乘" (推荐先做):**
1. `load_tiles_*` 时把 fp16 scale `d` (和 ls 因子) 先乘进解码后的 int8 tile
   -> 变成 fp16/fp32 tile? **不行, iu8 WMMA 只吃 int8。**
2. 改为: 解码后 tile 仍存 int8, 但把 per-row 的 dA 从内层移到 **K-slab 循环外**
   —— dA 只依赖 (i, k0/32), 每 slab 一次而非每 (n,j) 对一次:
   现 `sum[..] += C.x[l]*dA*dB` 在 ntx*J 循环里, 每元素都乘。
   重排: C-tile 累加完一个 slab 后一次性乘 dA*dB -> 指令数相同但…
   实际上数学等价下 dA/dB 已是每-slab 常数, 编译器可能已外提。
3. **真正有效的是 B2**: 换 K-major W-staging + A-fragment 直读骨架,
   即 radiance 的结构优势, 而非只省 fixup。

### 6.4 修正后的收益预期

fixup 占比: 12/(12+1) 指令比 => 理论上界 ~1.9x; 实际受 LDS 带宽与占用率限制,
预期 1.3-1.6x。这与 §2 的路线 A+B 合并预期一致 (200+ TF/s 可达, 252 不达)。

## 7. 骨架对比结论 (移植可行性定案)

### 7.1 两者结构差距比预想小

| 维度 | int8 MMQ (现) | radiance atiled |
|---|---|---|
| 每 block 输出 | I x J = 64 x 128 = 8192 | BMF x BNF = 64 x 128 (TN4) = 8192 (相同!) |
| W(tile) 复用 | LDS, 每 block 读 1 次 | LDS, 每 block 读 1 次 (相同) |
| A(activ) 进 | **LDS** (y_tile, staged) | **global 直读** (SADDR fragment) |
| scale 处理 | 内层 C-累加侧 fixup (~12 指令/WMMA) | 写 W-LDS 时折完 (0 fixup) |
| 码本查表 | IQ 系: global `__device__` 表 | MXFP4: shared sMag |

**MMQ 已经做了 W-LDS 复用和相近的 tile 面积。真正的两处结构差**:

1. **A 走 LDS staging** (radiance 用 SADDR 直读, 省一次 LDS 往返 +
   一次 barrier)。radiance 自己的 ablation: A staging 占 24% runtime。
2. **IQ 系码本表在 global**。radiance 把 128B 的 kMag 放 shared。

### 7.2 修正后的移植方案 (务实版)

**B-a (低险): IQ 码本搬 LDS。** `iq3s_grid` 2KB / `kvalues_iq4nl` 64B,
启动时 `sTab` 载入。IQ3_S 的 8 次 global 查表/thread-slab 变 LDS。
预期 IQ3_S/IQ2 系 +10-20%。

**B-b (中险): A-fragment 直读。** 仿 atiled 的 `abase[]` + `af[][]`:
activation 侧已有独立 q8_1 quantize (quantize_mmq_q8_1 已产 K-major 行),
让 y-tile 不进 LDS, WMMA 直接从 global 吃 A-fragment (int8 iu8 的
fragment 与 fp8 同为 16x16x16 布局, K=16 对齐)。省 y 的 LDS 往返 + barrier。
预期 +15-25% (radiance ablation 24% 上界)。

**B-c (高风险, 最后): scale 折叠进 W-tile 写入。** Q4_K 的 dm*sc 可在
写 LDS 时预乘成 fp16 存 x_dm —— 现状已近似 (x_dm = dm*sc8[l] 在 staging
时算)。收益有限, 放最后。

### 7.3 实施顺序

1. B-a: IQ3_S/IQ2_XS/IQ2_S/IQ2_XXS 四个类型的 load_tiles 加 `__shared__` 表
   (改动局限在 mmq-load-tiles.cuh + kernel 入口一次载表)。
2. 量化 sweep 路线 A 的 IQ2_XS 配置 (与 B-a 独立, 可同时做)。
3. B-b: 单独分支做 A 直读原型, 以 Q4_K (无码本干扰) 验证收益再推广。

### 7.4 B-a 探针否决 (重要: 实测推翻了 LDS 表假设)

写了独立探针 `/tmp/iq3_bench.hip` (8 次散布查表/线程, N=16M, gfx1201):
**global `__device__` 表 158-183 us, 搬 LDS 后 280-330 us —— LDS 反而慢 1.8x。**

原因: 2KB 表完全放进 L1 (32KB), `__device__` const 数组的标量读在 gfx12 上
走 scalar cache, 命中后与 LDS 同速甚至更快 (LDS 还要 preload + syncthreads +
bank conflict 风险)。radiance 的 sMag 进 LDS 是因为它同时被
`ds_load` 在 WMMA 链上消费, 场景不同。

**B-a (码本搬 LDS) 否决。** IQ 系查表不是主要瓶颈 —— 与 §6.2 的配比一致:
fixup (scale 乘) 才是 12 条/WMMA 的来源, 查表不在热链上。

**调整计划: 直接跳到 B-b (A-fragment 直读) + 路线 A (tile sweep)。**

### 7.5 B-b 可行性确认 (决定性代码证据)

`mma.cuh:844` `load_ldmatrix(tile<16,8,T>)` 的 AMD WMMA 分支就是:
`ggml_cuda_memcpy_1<16>(t.x, xs0 + get_i(0)*stride + get_j(0));`
—— **一次 16B 拷贝, 不依赖 LDS**, 从 global 读也完全合法 (radiance 的
`af[i][st] = *(const int2_t*)(abase[i] + aoff + ...)` 同理)。

布局要求: fragment 是 I-major (lane 持有连续 K), 而
`quantize_mmq_q8_1` 的输出 y 是 **row-major (K 连续)** —— 恰好匹配:
线程读 y[j_row * K/4 + k_th] 的 16B 就是 16 个连续 K 元素。

### 7.6 B-b 实施设计 (最终)

在 `mul_mat_q` (mmq.cuh:1013 附近) 加一条 RDNA4-only 快路径,
gate: `type ∈ {Q4_K, IQ3_S, IQ4_XS, Q2_K, ...}` && `args.ncols_max >= 256` &&
K%256==0 && 无 ids。结构照抄 radiance atiled:
- blockIdx.y 走 M (J 方向, 16 的倍数), blockIdx.x 走 N (I 方向);
- A(activation y) fragment 直读 global, SADDR 化 abase[];
- W(权重 x-tile) 照旧走现有 load_tiles 进 LDS (保留 fixup 但只在 staging);
- 内层: NS 个 k-step x TM x TN 次 `wmma_i32_16x16x16_iu8`,
  累加 int32; fixup (dA*dB) 移到 slab 边界一次乘 (int32 累加器上,
  每 slab 汇总后 `acc_f = acc_i * dA * dB` 延迟到 epilogue 前的 K 循环外)。
- 关键数学: iu8 是精确 int32 累加, dA/dB 每 (row, k-slab) 为常数,
  可以全部挪到 epilogue —— **fixup 从 12 指令/WMMA 降到 ~0**。

分两步落地:
1. 先做 "fixup 移出内层" (改 `q8_0_q8_1_mma` 的累加逻辑, 小改动,
   保留现 LDS 结构) —— 验证收益;
2. 收益确认后再上全直读骨架 (大改)。

### 7.7 fixup 移出被否决 (寄存器代数不成立)

`dA = x_df[i*sram_stride + k0/32]` 依赖 **(i, k-slab)** 二元组, `dB` 依赖 (j, k-slab)。
要把乘法挪到 epilogue, 必须保留 per-(i, j, k-slab) 的 int32 部分和 ——
K/32 个 int32/输出元素, 寄存器需求爆炸 (K=17408 -> 544 个 int32/element)。
**数学上不可行, 除非做两遍 pass。**

而现在的结构里 fixup 是 fmac (融合乘加), 本身不是独立 mul;
真正的开销是 `x_df[]` 的额外 LDS 读 (每 C 元素每 slab 一次)。

### 7.8 结论修正: 路线 B 的真实空间比 §6.2 估计的小

- B-a (码本 LDS): 探针否决 (LDS 慢 1.8x, L1 已覆盖)。
- B-b (A 直读): fragment loader 是纯 16B 拷贝, global 直读**可行**,
  但省的是 y-tile 的 LDS 写入+读出+barrier, 不是 fixup;
  且现有 y_prefetch 已把 y 的 global 读与 x-staging 重叠。
- fixup 移出: 寄存器代数否决。

** radiance 252 TF/s 的两个结构性来源 —— E8M0 折叠 (0 fixup) 和
checkpoint 自身的 fragment-friendly W 布局 (WPERM 128B 连续读) ——
对 IQ/K-quant 都不可复制**: 前者是幂次 scale 专属, 后者要求重量化。

**最终判定: 对 GSQ 类混合量化模型, MMQ 的现实优化空间 = 路线 A
(tile sweep, 低垂果实) + y-prefetch 微调, 预期 ceiling ~1800-2000 pp2048;
radiance 级别 (252 TF/s, ~2900) 只能靠换 MXFP4 (路线 C)。**

### 7.9 执行计划落点

1. 路线 A: IQ2_XS tile sweep (68 -> 140+ TF/s 预期), Q2_K 二轮 (23 -> ?),
   IQ3_S 现配置已接近最优 (92, 需 sweep 确认)。
2. B-b 原型只在 Q4_K 上试一次 (无码本, 布局已配好); 收益 <10% 则停。
3. 全部结果 A/B + PPL 门后汇报。

### 7.10 IQ2_XS sweep 前的最终分析 (sweep 范围收窄)

IQ2_XS 在 M=2048 已经选中 **nt=256/I=128/J=128 最大配置** (grid 4352/128 反推,
scratch=0), 与 Q4_K 的 167 TF/s 相同几何。**差 2.5x 的来源不是配置, 是
staging 的解码成本**:
- Q4_K: nibble shift, 0 查表, q8_1_q8_1_mma (x 直接 int8 WMMA)。
- IQ2_XS: QR2_S 轮 iq2s_grid **global 查表** + vcmpne4/vsub4 符号展开 +
  `(ls&0xF)*d+d/2)/4` 浮点 scale 合成, 且 sram_stride=84 (Q3_K 布局) 使
  x-tile LDS 写入更散。

**tile sweep 对 IQ2_XS 无空间** (已在最大 J, 无溢出)。B-a 探针已否决表进 LDS。
剩余的原始码本查表成本是格式本身的, 只能靠路线 C 消除。

**路线 A 的可做项只剩**: Q2_K 二轮 (23 -> ?; I=64 配置已加, 试 nt=128/I=32/J=128
或重排 Q2_K 布局), IQ3_S 确认性 sweep (92 -> ?)。预期收益有限, 谨慎乐观。

### 7.11 阶段总结 (本节调查的净结论)

| 假设 | 验证结果 |
|---|---|
| IQ 码本搬 LDS 有收益 | **否** (探针: LDS 慢 1.8x, L1 已覆盖) |
| fixup 可移出内层 | **否** (dA 依赖 (i,k-slab), 寄存器爆炸) |
| IQ2_XS 欠额可 sweep | **否** (已选最大配置, 差距在格式本身) |
| radiance 级别可移植 | **否** (折叠需幂次 scale, WPERM 需重量化) |
| Q2_K 配置修复 | **是, 已落地 +58%** (§10.8) |
| 换 MXFP4 (路线 C) | **唯一通往 252 TF/s 的路** |

**给用户的最终答案: GSQ 类混合量化在 MMQ 框架内的优化空间已基本挖尽
(Q2_K 修复 +58% 是最后一颗大果实); 要 radiance 级 prefill 只能换 MXFP4 格式。**

## 8. Q2_K 二轮 sweep: V2 配置被 PPL 门拦截 (2026-10-01)

### 8.1 现象

V2 (nt=128/I=32/J=128) 比 V1 (nt=256/I=64/J=128) **更快**:
pp2048 1284->1377-1387 (+6.9%), Q2_K kernel 13.3 ms / 225.5 TF/s,
VGPR 从 256 降到 80, 无溢出。MUL_MAT type_a=q2_K 2 项 OK。

**但 PPL 完全崩坏**: V1 = 6.0175, V2 = **6159.7 / 6846.9 (1000 倍)**。

### 8.2 判定

数值错误 = **V2 的 I=32 tile 是错的配置**。MUL_MAT 单测只覆盖了小 K
(q2_K case 2 项 OK) 而真实模型 K=5120 触发了越界/重叠。
单测通过 != 正确, PPL 门救了这次 —— "fast but wrong"。

推测原因: Q2_K 的 load_tiles/staging 按 I 的边界有 superblock 对齐假设
(I=64 是 2 个 superblock 的整数倍; I=32 时 tile 横切 superblock,
x_df/x_qs 的 stride 索引错位)。

### 8.3 处置

**回滚 V2, 保留 V1 (nt=256/I=64/J=128, commit 4f7f0b5e3 的配置)。**
I=32 方向标记为不可行 (除非重写 Q2_K staging 对齐逻辑, 投入产出比差)。

**最终态: Q2_K = 23.1 TF/s (V1), GSQ pp2048 = 1294。**
这是一个可接受的终点: Q2_K 从 2.9 -> 23.1 (8x), 占比 49.8% -> 11%,
剩余 11% 中绝大部分是 Q2_K 格式本身的解码成本。

### 8.4 净结论 (对用户的最终回答)

1. **GSQ 混合量化在 MMQ 框架内的优化已挖尽**: V2 尝试证明连 I=32 都会破坏
   Q2_K 的 staging 对齐; IQ2_XS 已在最大配置; 码本 LDS 化被探针否决。
2. **radiance 级 (252 TF/s) 对 IQ/K-quant 不可移植**:
   - E8M0 幂次折叠 -> fp16 scale 无法无损折叠 (§1.2)
   - fixup 移出 -> 寄存器代数否决 (§7.7)
   - 码本 LDS -> 实测更慢 (§7.4)
   - WPERM fragment 布局 -> 需要重新设计量化格式本身
3. **唯一通往 252 TF/s 的路 = 路线 C: 从 bf16 源重新逐张量 quantize 成
   MXFP4** (用户已有完整 recipe), 12.1 GB -> 14.5 GB, pp2048 ~2900。
4. 本轮净收益定格: Q2_K 8x 修复 -> GSQ pp2048 820 -> 1294 (+58%)。

### 8.5 回滚确认

- config 表已 `git restore` 到 committed V1 (13 条, J=128 两条都是 I=64);
- 从 V1 源码重建库, md5 c6e7c2749... 与已提交版本完全一致 (位可复现构建);
- 安装库 = V1, 工作树干净。

## 9. 用户提议的专项评估: 给 int 类型做 radiance 式存储格式适配 (2026-10-01)

用户判断: "radiance 给这些 int 类型的存储格式重做一个适配版本, 成本应该不高"。
拆开算账。"radiance 适配" = 三件事: (i) 布局重排 (nibble 顺序/scale 分离),
(ii) 配套 kernel (atiled 骨架), (iii) E8M0 幂次折叠。
(i) (ii) 可复制; (iii) 是精度的根, 不可复制 (§1.2)。

### 9.1 D1: 布局适配版 (尺寸中性, 复制 (i)+(ii))

- 内容: runtime repack 把 W 重排成 fragment 顺序 (128B 连续/wave) +
  activation 量化直出 fragment 布局 (A 直读, 省 y 的 LDS 往返)。
  码本解码和 per-slab fixup 保留 (在 LDS staging 侧)。
- 成本: 确实不高 —— repack 照抄 radiance (size-neutral, 缓存),
  kernel = atiled 骨架 + iu8 WMMA + 既有 fixup。
- 收益上界论证: **Q4_K (int8 MMQ, 零解码) 已经跑到 167 TF/s,
  高于 MMQ-fp8 路径的 ~110-142** —— 说明现有 int8 骨架离该指令的
  组织上限已经很近; D1 能拿的只剩 A 直读 (~10-20%) 和 W 合并读 (几个 %)。
- 预期: 各类型 167 -> ~200, GSQ MMQ 1183 -> ~950 ms, pp2048 ~1500-1600。
  **到不了 252** (fixup + 码本是格式固有的, 见 §7.7/§7.4)。

### 9.2 D2: 预解码版 (真正"零解码", 复制 (iii) 的效果) —— 显存数学否决

把码本+符号在 repack 时预解码成 int8 (值域 ±15, 精确无损),
staging 变纯拷贝, 各类型直接到 Q4_K 水平甚至更高。两种形态:

| 形态 | 显存/文件 | 判定 |
|---|---|---|
| 运行时 repack (双份存储: 原始 12.1 GB + 解码 26.8 GB) | **38.9 GB > 32 GB VRAM** | **这个 27B 模型在这张卡上放不下** |
| 文件格式 (装载时即解码) | 27.32B x 1B = 8.5 bpw ≈ 29 GB | **这就是 Q8_0**, 被 MXFP4 (4.5 bpw, 14.5 GB, 还更快) 全面压制 |
| 6-bit 打包 (±15 塞 int6) | repack 后 20.1 GB, 双份 32.2 GB | 仍超 32 GB, 且 int6 拆包成本回来了 |

结论: D2 的显存代价不是工程问题, 是算术问题。

### 9.3 全选项对照 (GSQ 模型, R9700 32GB)

| 方案 | bpw/显存 | pp2048 预期 | 精度 | 成本 |
|---|---|---|---|---|
| 现状 (Q2_K 修复后) | 3.44/12.1 GB | **1294 (实测)** | 无损 | 已完成 |
| D1 布局适配 | 尺寸中性 | ~1500-1600 | 无损 | 中 (新 kernel) |
| D2 预解码 | 38.9 GB | ~1800-2000 | 无损 | **放不下** |
| D2 文件版 (=Q8_0) | 8.5/29 GB | ~1900 | 无损 | 被 MXFP4 压制 |
| **路线 C: 转 MXFP4** | **4.5/14.5 GB** | **~2900** | 重量化代价 | recipe 已验证 |

### 9.4 判定

用户直觉对了一半: **布局适配 (D1) 成本确实不高** —— 但它的收益上限
也已经被 Q4_K 的 167 TF/s 锚死了 (现有 int8 骨架不是瓶颈所在)。
而"完整 radiance 效果"所需的"预解码+零 rescale"版本, 要么显存翻倍放不下 (D2),
要么等价于一个被 MXFP4 全面占优的 Q8_0 文件格式。
**最优的"存储格式适配"就是格式本身换成 MXFP4** —— 它已经存在、
已经过验证、显存减半、速度最高。

## 10. 用户方案澄清后的重估: "int 格式自己的 radiance" (2026-10-01)

### 10.1 方案本质 (确认理解)

radiance = repack 时做 E8M0 折叠 (重活前置) + 运行时纯 fp8 WMMA。
对码本类型做同构的事: **repack 时把 iq3s_grid/kvalues 查表 + 符号 XOR
一次性解码成 int8 值 (纯整数运算, 精确无损), 运行时 kernel 吃现成 int8 tile
-> 纯 iu8 WMMA, 每权重零解码。** 这正是 radiance 思想对 int 格式的正确迁移。

### 10.2 修正我自己的一个错误结论

§9.2 曾以 "双份存储 38.9 GB > 32 GB VRAM" 否决预解码。**算错了**:
逐张量替换 (decode 后不再保留原始 packed 字节) 下, 总量 =
解码张量 (8.31 bpw) + 未解码张量 (原样) = **~23-24 GB, 放得下**。

### 10.3 各类型精确可解码性 (关键新发现)

| 类型 | 解码后值域 | 每块 scale 结构 | 精确 int8 化 |
|---|---|---|---|
| **Q2_K** | sc*v+min in [0,60] | per-32 d | **精确 = Q8_0!** (w8=sc*v+min, scale=d) |
| IQ3_S | grid+-sign, ±31 | per-16 d(2ls+1)/4 | 精确, 需 per-16 scale 变体 |
| IQ3_XXS | 同上 | 同上 | 精确, 同上 |
| IQ2_S/XS/XXS | grid+-sign | per-16 | 精确, 同上 |
| IQ4_XS | v+sc8 到 ±158 | per-16 加性 offset | **int8 装不下 (158>127)**, 需 bias 修正项, v1 跳过 |
| Q4_K | nibble | per-16 dm*sc | 已 167 TF/s, 不动 |

Q2_K -> Q8_0 的精确映射特别漂亮: w8 = sc*v+min 是 0..60 的精确整数,
scale = d, 直接进现有 Q8_0 kernel —— 顺带把 Q2_K 从 23 TF/s 拉满到 167。

### 10.4 投入产出 (GSQ 模型, R9700)

显存: 12.1 -> ~23-24 GB (解码 5 类型 18.2 GB + IQ4_XS/Q4_K 原样 4.6 GB);
速度锚点: Q4_K 证明 "int8 staging + fixup" 类 kernel = 167 TF/s;
预解码后 staging 变纯拷贝, 预计 >=167。

| 指标 | 现状 | 预解码 (复用现有 kernel) | + atiled-int8 二期 |
|---|---|---|---|
| MMQ 总时间 | 1183 ms | ~645-690 ms | ~500-560 ms |
| pp2048 | **1294** | **~2000** | **~2300** |
| tg128 | 33.14 | **~17 (代价!)** | 同左 |
| 显存 | 12.1 GB | ~24 GB | 同左 |
| 精度 | - | **逐位等价** (同整数同 scale) | 同左 |

tg 减半是唯一的真代价 (带宽受限, 字节翻倍); 若要保 tg 需双份共存 (30.3 GB, 贴上限)。

### 10.5 分阶段实施

1. **原型: IQ3_S 单类型**。repack kernel (查表+符号 -> int8, per-16 fp16 scale)
   + kernel 变体 (staging 纯拷贝 + 既有 q8_0_q8_1_mma)。验证 >=167 锚点。
2. 推广 IQ3_XXS/IQ2_S/IQ2_XS/IQ2_XXS (同一变体) + Q2_K (直接映射 Q8_0)。
3. 二期 (可选): atiled-int8 (A 直读, iu8 峰值 308) -> 冲 2300。
4. IQ4_XS/Q4_K 不动。

### 10.6 重要更正: IQ3_S 的 scale 是 per-16, Q8_0 kernel 吃 per-32

block_iq3_s: d (fp16) + qs[13x32] + qh[4] + signs[32] + scales[IQ3S_N_SCALE],
其中 **IQ3S_N_SCALE = QK_K/16** = 每 16 元素一个 4-bit (ls 因子, d*(2ls+1)/4)。
解码后是 per-16 scale 的 int8 —— 不等于 Q8_0 (per-32), 不能直接用现有
q8_0_q8_1_mma; 需要 per-16 scale 的 int8 kernel 变体 (即 Q8_0_16 布局,
mmq 里已有 q8_0_16_q8_1_mma 供 IQ2_XS/IQ2_S 使用!)。

**修正: IQ3_S/IQ3_XXS/IQ2_* 解码后 = per-16 int8 值 + per-16 fp16 scale,
即现有 q8_0_16_q8_1_mma 的输入。staging 只需纯拷贝 (或直接 repack 成
该布局让 load_tiles 变 memcpy)。** Q2_K 解码后是 per-32 -> 直接 Q8_0 路径。

这反而更顺: kernel 变体已经存在, repack 是唯一新代码。

### 10.7 风险与残留成本 (如实)

1. **新类型定义**: GGUF 枚举要加 "预解码 int8 变体" (如 IQ3_S_D / Q2K_D),
   llama-quantize/convert 侧要有 repack 工具或加载时 repack;
   runtime repack (radiance 现有模式, size+6.25% 的 scale 分离代价) 可以绕开
   新格式定义 —— **v1 建议走 runtime repack, 不动 GGUF 枚举**。
2. **显存管理**: repack 后的解码 buffer 与原 tensor 共存 2GB+ (IQ3_S 部分),
   radiance 已有 per-tensor repack cache 模式可抄, 但 27B 模型上
   decode 张量 18.2 GB + 原始 12.1 GB 的峰值要看加载顺序 ——
   需要逐张量 repack 后释放原始 (GGML 内存池不支持; 需在 load 阶段做,
   即 buffer type 层)。这是主要工程成本所在。
3. **精度**: 纯整数重排, 逐位等价, PPL 必须逐位一致 (硬门)。
4. **tg 代价**: 若不保留双份, decode 字节翻倍 -> tg128 33 -> ~17。
   这是本方案最大的产品级代价, 需要用户拍板。

### 10.8 给用户的最终重估结论

理解对齐后 (int 格式自己的 radiance = repack 前置解码), 方案**成立且比
之前评估的更顺**:
- kernel 侧几乎零新代码: q8_0_16_q8_1_mma / q8_0_q8_1_mma 已存在,
  解码后的 tile 就是它们的输入;
- repack 一次做完查表+符号+scale 合成, 运行时 staging 退化为 memcpy;
- 精度逐位等价 (纯整数重排 + 原 fp16 scale);
- 预期 IQ3_S/IQ3_XXS/IQ2_* 全部抬到 >=167 TF/s (Q4_K 锚点),
  pp2048 1294 -> ~2000, 二期 atiled-int8 再到 ~2300。

**真实成本**: (a) repack 时机 —— 加载阶段做才能释放原始字节,
涉及 buffer 层, 是主要工程量; (b) tg128 33 -> ~17 (decode 显存带宽翻倍),
或者保留双份共 30.3 GB 贴显存上限; (c) IQ4_XS 值域 ±158 装不下 int8,
要么跳过要么加 bias 修正 (二期)。

**一句话: 思路正确, 工程上是一周的活不是一天的活, 收益 pp2048 +55%,
代价是 tg 减半或显存贴顶。建议先做 IQ3_S 单类型原型验证 167 锚点。**

### 10.9 更正: IQ4_XS 是乘性 scale, 不是加性 offset —— 上一节写错了

用户指正方向后查源码 (dequantize_row_iq4_xs):
`y = dl * kvalues_iq4nl[nibble]`, 其中 `dl = d * (ls - 32)`。
**没有 "+ sc8 加性 offset" 这回事** —— 那是我 §10.3 表里的错误表述。

真正的约束在码本值域: kvalues_iq4nl = {-127,-104,-83,-65,-49,-35,-22,-10,
1,13,25,38,53,69,89,113}。
负侧最小 -127, 正侧最大 +113 —— **符号不对称**, 绝对值 <= 127,
**每个码本值本身就是合法 int8** (码本本来就是 int8_t 存的)!

所以 IQ4_XS 的预解码完全可行:
- repack: nibble -> kvalues_iq4nl[] 查表 -> int8 值 (精确, 无舍入),
  与 per-16 的 dl (fp16) 一起存;
- 解码后 = per-16 int8 + per-16 fp16 scale = q8_0_16 布局,
  同 IQ3_S 走 q8_0_16_q8_1_mma;
- 唯一代价: 4 bit -> 8 bit, 该类型解码后显存 5.69B*2 = 11.4 GB
  (GSQ 里 IQ4_XS 占 5.69 GB 参数, 解码后翻倍)。
  或者只解一半 (Q4_K 模式: nibble 留着, 这本来就是 Q4_K 路径的形态)。

**结论修正: IQ4_XS 无任何格式障碍, 一期就该包含它。**
之前 "±158 装不下" 是把 Q6_K 的 q6 值域 (±127+sc, 最大 ±158? 查证:
q6 值域是 [-128,127] 内, Q6_K 用 6-bit + 8-bit scale min 混合) 与
IQ4_XS 混淆了。IQ4_XS 的码本就是 int8 表, 天然装得下。

### 10.10 全类型最终可解性表 (更正版, 取代 §10.3)

| 类型 | 码本/值域 | 预解码后形态 | scale 粒度 | int8 装得下? |
|---|---|---|---|---|
| Q2_K | sc*v+min, 0..60 整数 | Q8_0 等价 | per-32 | 是 (0..60 ⊂ [-128,127]) |
| IQ3_S / IQ3_XXS | grid ± sign, ±31 | per-16 int8 | per-16 (fp16) | 是 |
| IQ2_S/XS/XXS | grid ± sign, ±31 | per-16 int8 | per-16 (fp16) | 是 |
| **IQ4_XS / IQ4_NL** | kvalues_iq4nl, [-127,+113] | per-16 int8 | per-16 (fp16, dl=d(ls-32)) | **是 (本来就是 int8_t 表)** |
| Q4_K | nibble ±8 | 已是最高效路径 (167) | per-16 | - |
| Q3_K | 三路 nibble 组合 | per-16 int8 | per-16 | 是 (但 Q3_K 已 95.7, 优先级低) |
| Q5_K/Q6_K | nibble+高bit | 类似 Q4_K | per-16 | 是 |

GSQ 模型一期可覆盖: Q2_K + IQ3_S + IQ3_XXS + IQ2_S + IQ2_XS + IQ2_XXS +
IQ4_XS = **21.6B/27.32B 参数 = 79% 的权重**进预解码快路径。

### 10.11 用户指正: "tg 减半" 推翻 (2026-10-01)

原论断: "解码字节翻倍 -> tg128 33 -> ~17"。
**错。** 核心遗漏: **MXFP4 上 radiance 的 repack 只作用于 prefill** ——
`ggml_cuda_radiance_supported()` 硬门 `ne11 >= min_m (256)`,
decode (M=1..8) 走 MMVQ (dp4a), 吃的是**原始 MXFP4 张量**,
不是 repack 后的 fragment 布局。

同理, "int 格式自己的 radiance" 的 repack 也应该只在 prefill 路径生效:
- prefill (M>=256): 用解码后的 int8 快路径;
- decode (M<8): **MMVQ 继续吃原始 packed 格式** (码本路径本来就在)。

=> **权重必须双份共存** (原始 packed + 解码 int8), 这回到显存账:
原始 12.1 + 解码类型 18.2 = 30.3 GB (贴 32 GB 上限, 但放得下),
且 **tg 完全不受影响** (MMVQ 读原始份)。

实测锚点支持这个框架: MXFP4 tg128=29.85 / IQ3_S tg128=33.14,
与两者文件大小比 (14.73/11.28=1.31) 和 tg 比 (1.11) 一致 ——
decode 是纯带宽受限, 与 quant 类型无关; 只要原始份还在, tg 不变。

**修正后账目**: pp2048 ~2000 (一期) / ~2300 (二期), tg128 33.14 不变,
显存 30.3 GB (IQ4_XS 若不解码则 ~27 GB, 余量更足)。
之前 "tg 减半" 的说法作废。

## 11. 用户第二次指正: 显存模型重构 (2026-10-01, 实测定案)

### 11.1 实测推翻 "原始 12.1 + 解码 18.2 = 30.3 GB" 的账

用户论点: "解码不占用显存, 不然单卡根本跑不起来 MXFP4 版本" ——
**且此前所有 spec decode (带 1.05 GB drafter) 都在 R9700 上跑过。**

实测 (R9700 gpu2, amd-smi 轮询):
- 载入后 (未跑 GEMM): USED = 14 959 MB ≈ 文件 14.73 GiB —— **模型驻留 ≈ 文件大小**
- 跑 pp2048 全程 (radiance repack 已发生): **PEAK = 29 471 MB = 28.78 GiB**

### 11.2 解释: repack 双份确实存在, 但被 compute buffer 吃透, 总账比我的静态账小

29.47 GB 的构成:
- 原始权重驻留 14.59 GiB (MMVQ/回退要读的)
- radiance repack cache: W(W原大小)+Ws+Wref ≈ 权重的 +6.25% ≈ 0.91 GiB
- compute buffer (ub 2048 的 activation/scale/输出) ≈ 数 GB
- ……合计 28.78 GiB < 32 GB, **刚好塞下**。

=> **我之前 "12.1 + 18.2 = 30.3 静态和" 的高估在于**: 把 compute buffer
漏算了又把解码算成"纯加法"。真实结构是三层: 原始驻留 + repack 副本
(大小 ~ +6.25%, 不是 x2!) + compute buffer。**W 的 repack 是 size-neutral
的 (W 与原 packed 同为 N*K/2), 只有 Ws+Wref 是额外的 3.1%** ——
MXFP4 repack 只膨胀 6.25% 而非 100%。

### 11.3 这直接改写 int 预解码方案的显存账

预解码 IQ3_S/IQ2_*/IQ4_XS -> int8: 解码副本 = N*K/2 x 2 = **权重 x1.6875**
(相对原 packed), 另加 scale 份。3 bit 平均 -> 8 bit, 解码张量 18.18B 参数 x1B
= 18.2 GB (不可避免, 这是字节数学)。

| 组成 | MXFP4 repack (实测) | int 预解码方案 |
|---|---|---|
| 原始驻留 | 14.59 GiB | 12.1 GB |
| repack/解码副本 | +0.91 GiB (6.25%) | +18.2 GB (x1.7) |
| compute buffer | ~数 GB | 同左 |
| **合计** | **28.8 GB 实测** | **~33-34 GB** |

=> **R9700 32 GB 单卡仍然放不下完整的"解码副本全量"方案**
(除非 ub 调小 compute buffer、或 Q2_K 不解码)。

但用户论点的正确部分: **"不占显存"的真正含义是 —— 解码应该在
"加载/格式层"发生而不是"运行时双份"**: 文件直接存解码后的 int8
(新 GGUF 类型, 8 bpw = 29 GB 文件), MMVQ decode 侧另外保留 packed
么? 不 —— **Q8_0 家族本身就是"预解码格式"**: Q8_0 权重 8 bpw,
decode 和 prefill 都吃它, 单份存储。29 GB 文件 > R9700 也放不下 (模型 29 + KV)。

### 11.4 修正后的真实约束

- IQ3_S/IQ2 系 (3 bit 级) 预解码 = 解码副本 x1.7 => 单卡放不下全量;
- **可行切分: 只解 Q2_K (0.73 GB -> 2.26 GB, +1.5 GB) + IQ3 系**
  (14B 参数 -> 14 GB) 超限; 只解 Q2_K+IQ2 系 (5.6B -> 5.6 GB) 可行;
- 或维持我的 runtime-repack 方案但接受 prefill 峰值
  (12.1 + 解码 18.2 + compute) ~= 33 GB 不可行 ->
  **解码副本必须分块流式 (per-tensor repack + 用完即弃) ——
  但 MMQ 的 repack cache 生命周期是 process 级, 用完即弃与
  CUDA graph 捕获冲突** (graph 重放需要地址稳定)。

### 11.5 结论 (本轮)

用户连续三次指正均成立, 我的预解码方案在显存维度最终形态:
- **Q2_K 预解码 (-> Q8_0) 完全可行且便宜**: +1.5 GB, 收益 23 -> 167 TF/s,
  是唯一"成本不高收益大"的项 —— **建议单做这个**;
- IQ3/IQ2/IQ4 系预解码: 解码副本 18 GB 级, 单卡放不下 (除非文件格式化 =
  换 8 bpw 模型, 又被 MXFP4 压制);
- radiance 在 MXFP4 上能"便宜"的根源: W repack 是 size-neutral (+6.25%),
  而码本解码是 x1.7 —— **这不是工程成本, 是信息论成本** (3 bit 码本值
  展开 8 bit 的熵增量)。

## 12. "Q2K 等格式是否可以全做" —— 精确账 (2026-10-01)

### 12.1 显存预算 (GSQ 模型, R9700 32GB)

- 载入后驻留: ~12.1 GB; compute+KV+overhead: ~6 GB;
- **解码副本预算: ~13.9 GB**。

各类型解码净增 (int8 1B/w + scale, Q2_K 是 per-32 -> 1.03125 B/w):

| 类型 | 参数 | 原 GB | 解码后 | 净增 |
|---|---|---|---|---|
| Q2_K | 0.73B | 0.24 | 0.76 | **0.52** |
| IQ2_XS | 0.77B | 0.25 | 0.81 | **0.57** |
| IQ2_XXS | 0.38B | 0.10 | 0.41 | **0.31** |
| IQ2_S | 2.50B | 0.98 | 2.65 | **1.68** |
| IQ4_XS | 5.69B | 3.20 | 6.04 | **2.84** |
| IQ3_XXS | 5.42B | 2.33 | 5.76 | **3.43** |
| IQ3_S | 8.48B | 4.70 | 9.01 | **4.31** |
| Q4_K | 2.81B | 1.58 | (不解码) | 0 |
| **全 7 类型合计** | | | | **13.66 GB ≈ 预算 13.9** |

**全部 7 类型 = 13.66 GB, 贴着 13.9 预算线** —— 单卡可容纳,
但 compute buffer 稍大或 ub 更大就会 OOM, 需要 --no-kv-offload 之类的余量管理。
稳妥档: 除 IQ3_XXS 外 6 类型 = 10.2 GB 净增, 余 3.7 GB。

### 12.2 收益 (锚点: 167 TF/s = Q4_K 同骨架实测)

| 组合 | 净增显存 | MMQ ms | pp2048 预期 |
|---|---|---|---|
| 只 Q2_K | 0.52 | 1052 | ~1432 |
| +IQ2 系 | 3.08 | 1005 | ~1498 |
| +IQ4_XS | 6.44 | 925 | ~1627 |
| **全 7 类型** | **13.66** | **657** | **~2292** |

Q2_K 单做收益最小 (+11%), 大头在 IQ3_S (+170ms) / IQ4_XS (+80ms) / IQ3_XXS (+99ms)
—— **越贵的解码副本收益越大**, 显存预算是唯一的取舍维度。

### 12.3 结论

**可以全做。** 7 类型 13.66 GB 贴预算线, pp2048 1294 -> ~2292 (+77%),
tg128 33.14 不变 (MMVQ 读原始份), PPL 逐位一致。
稳妥上线顺序: Q2_K+IQ2 系 (3.1 GB, +16%) -> IQ4_XS (+2.8 GB) -> IQ3 系 (+7.7 GB),
每档独立可回退 (repack cache 按 tensor 类型白名单)。
风险: 167 锚点对 per-16 scale 的 q8_0_16 kernel 未验证 (IQ2_S 139 佐证),
Q2_K 的 w8=sc*v+min 范围 0..60 非对称, Q8_0 的 int8 精确表示需 iso 测试确认。
