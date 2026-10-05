# Qwen3.8-27B RDNA4 GEMM 专项

> **压缩上下文后先读这里。当前状态 (2026-10-01):**
> - **pp512 2318.7 / pp2048 2577.7 / tg128 28.26 / PPL 6.9318**; 相对最初基线 1067/1134 = **+127%**。
>   **运行必须带 `-ub 2048`** (默认 ub 512 时 pp2048 只有 2440.8)。
> - 本会话完成三件事: **A2 GDN chunked bf16 WMMA** (+12.2%, 已默认开)、
>   **GEMM W fragment 改 LDS.64** (-2.4%)、**ubatch 512->2048** (+5.6%)。
> - **GEMM 那 51% 已到架构上限** (作者的 ablation: 超出 WMMA 下限的 ~40% 是结构性的, 见第 8 节)。
> - **concat 优化三条路线已全部否决**, `concat.cu` 保持 pristine (见第 5 节)。
> - **TG 侧已完成 (2026-10-01, 见 `design-mtp-rdna4.md`)**: `output.weight`/`token_embd.weight`
>   重量化为 MXFP8 + 投机解码 -> **tg 28.25 -> 50.7 t/s (1.79x)**。
>   PPL 6.9318 -> 6.9633 (+0.45%), greedy 逐字节一致, pp2048 无退化。
>   新模型 `Qwen3.8-27B-Rad-MXFP4-f8out.gguf` (15088 MiB, 4.63 BPW), 原模型保留不动。
>   投机用 **DFlash2 drafter** 量化到 `f4body-f8head` (1.05 GB, 4.52 BPW):
>   body MXFP4 + selector 码本 MXFP8 (selector 是离散码本, 压到 4bit 会破坏排序)。
> - **新增 MMVQ 阈值补丁 (2026-10-01)**: `MMVQ_RDNA4_MAX_BATCH_SIZE = 4` —— RDNA4 上
>   ncols in [5,8] 从 dp4a 改走 WMMA。mmvq.cu/.cuh + test-backend-ops.cpp 共 +7/-1 行。
>   M=1 的生产路径不受影响, 全部红线保持 (PPL 6.9633 逐位不变 / 单测全绿)。
>   这是 dflash 提速的关键 (n_max=7 时一轮恰好 8 个位置, 原阈值 8 让它永远走 dp4a)。
> - **F32 WMMA 已彻底证伪 (2026-10-01)**: RDNA4 连小块 f32 WMMA 都没有 (硬件实测
>   `ILLEGAL_INSTRUCTION`), 且本模型 F32 matmul = 0 次 -> 收益精确为 0。五重证据见 §1,
>   **不要再投入这条线**。真想优化 F32, 对象是 §5 那 15.9% 的向量 kernel, 但都是零头。
> - 硬件: R9700 是 **64 CU** (HIP 误报 32), 每 CU **64KB LDS**; RDNA4 只有 11 个 WMMA, 全是 16x16x16
>   (f32/大 K/f4/scale 变体都属 gfx1250 = CDNA5, 与本机无关, 别再查)。

目标: 压缩 radiance MXFP4×FP8 GEMM 的 459ms (pp2048 单次约 863ms 中的 53%)。
上游文档: `design-gdn-rdna4.md` (PP 战线总账 / GDN 交接记录)。

---

## 1. RDNA4 (gfx1200/gfx1201) WMMA 指令集权威清单

来源: `ROCm 10.0 /opt/rocm/core-10.0/lib/llvm/include/clang/Basic/BuiltinsAMDGPU.inc`
(builtin + 类型签名 + target feature 三元组), 并用 `llvm-mc -mcpu=gfx1201` 逐条验证。

**R9700 (gfx1201) 上可用的 WMMA 只有 11 个, 全部是 16x16x16 (iu4 另有 K=32 变体):**

| builtin (gfx1201) | 输入 | K | tile MAC | 实测 T MAC/s | MAC/操作数字节 |
|---|---|---|---|---|---|
| `wmma_f32_16x16x16_bf16_w32_gfx12` | bf16 | 16 | 4096 | 97.5 | 128 |
| `wmma_f32_16x16x16_f16_w32_gfx12` | f16 | 16 | 4096 | 91.8 | 128 |
| `wmma_f16_16x16x16_f16_w32_gfx12` | f16→f16 | 16 | 4096 | (未测) | 128 |
| `wmma_bf16_16x16x16_bf16_w32_gfx12` | bf16→bf16 | 16 | 4096 | (未测) | 128 |
| `wmma_f32_16x16x16_fp8_fp8_w32_gfx12` | e4m3 x e4m3 | 16 | 4096 | **180.6** | 256 |
| `wmma_f32_16x16x16_fp8_bf8_w32_gfx12` | e4m3 x e5m2 | 16 | 4096 | (未测) | 256 |
| `wmma_f32_16x16x16_bf8_fp8_w32_gfx12` | e5m2 x e4m3 | 16 | 4096 | (未测) | 256 |
| `wmma_f32_16x16x16_bf8_bf8_w32_gfx12` | e5m2 x e5m2 | 16 | 4096 | (未测) | 256 |
| `wmma_i32_16x16x16_iu8_w32_gfx12` | int8 | 16 | 4096 | (未测) | 256 |
| `wmma_i32_16x16x16_iu4_w32_gfx12` | int4 | 16 | 4096 | (未测) | 512 |
| **`wmma_i32_16x16x32_iu4_w32_gfx12`** | int4 | **32** | **8192** | **347.1** | **512** |

### 不在 RDNA4 上的 (避免再走弯路)
- **f32 输入 (2026-10-01 彻底证伪, 勿再投入)**: RDNA4 上**没有任何** f32 输入的 WMMA,
  小块 (K=4) 也没有。五重独立证据:
  1. `llvm-mc -triple=amdgcn-amd-amdhsa -mcpu=gfx1201` 对 `v_wmma_f32_16x16x4_f32` 报
     `instruction not supported on this GPU`; `-mcpu=gfx1250/gfx1251` 通过。
  2. builtin `__builtin_amdgcn_wmma_f32_16x16x4_f32` 的 target feature 是
     `gfx1250-insts,wavefrontsize32`。
  3. **硬件实测**: 用 `.long 0xcc5d0000 / 0x1c321508` 直接发射该编码, gfx1201 抛
     `HSA_STATUS_ERROR_ILLEGAL_INSTRUCTION`。阴性对照 `v_wmma_f32_16x16x32_bf16` 行为一致,
     空 kernel 无错 -> **不是 LLVM 门控问题, 是硬件不认**。
  4. CK Tile `/opt/rocm/include/ck_tile/ops/gemm/warp/warp_gemm_attribute_wmma_impl.hpp:127`
     里 `WarpGemmAttributeWmmaImpl_f32_16x16x4_f32` 绑的是 **`gfx125_t`**, 不是 `gfx120_t`。
     `ck_tile/core/arch/arch.hpp`: `gfx120_t` = LDS 64KB / VGPR 256 / 32 banks (RDNA4),
     `gfx125_t` = **LDS 320KB / VGPR 1024 / 64 banks** (CDNA5)。
  5. 同文件里标 `gfx120_t` 的 WMMA 只有 fp8/bf8 的 16x16x16; f16/bf16/i8 用 `DeviceIp`。
- **易混来源 (回答"ROCm 10 明明有 1201 的 F32")**: hipBLASLt 确实为 gfx1201 提供 FP32 GEMM
  kernel (`/opt/rocm/lib/hipblaslt/library/gfx1201/TensileLibrary_SS_SS_..._Type_SS_Contraction_*.co`,
  SS = single precision), 但内部是 **SIMD v_fma**, 不是 WMMA。所以"F32 走 SIMD"和
  "ROCm 10 有 gfx1201 的 F32"两句都对, 只有"RDNA4 有 F32 WMMA"不成立。
- **本模型收益精确为 0**: pp2048 稳态 9287 次 dispatch 里 **F32/rocBLAS matmul = 0 次**。
  360 个 F32 张量共 267 万参数 (占 27.32B 的 0.01%), 全是 `ssm_conv1d.weight [4,10240]` /
  norm `[5120]` / `ssm_a [48]` / `ssm_dt.bias [48]`, **无一参与 matmul**;
  两个大权重 `output.weight` / `token_embd.weight` 是 **BF16** (走 `mul_mat_vec_f`, 占 1.0%)。
  -> 就算硬件支持, 也无对象可优化。
- **大 K 变体** (K=32/64/128 的 16 位与 8 位)、**f4/f8f6f4 混合精度**、**`wmma_scale_*` 硬件块缩放**:
  同样全部是 `gfx1250-insts`。
- **gfx1250 = CDNA5 (MI455X, Helios 机架)**, gfx1251 = RDNA4.5 APU。**与 R9700 无关, 不要再查。**
- 命名陷阱: builtin 名字里的 `f32` 是**累加器**类型 (`wmma_<acc>_<M>x<N>x<K>_<input>`),
  不是输入类型。`wmma_f32_16x16x16_bf16` = f32 累加 + bf16 输入。

### 硬件事实 (顺手确认, 影响所有 occupancy 计算)
- **R9700 是 64 CU**。`hipDeviceProp_t.multiProcessorCount` **误报 32** (rocminfo 报 64 才对)。
  所有按 32 CU 折算的旧推算都要翻倍修正。
- **每 CU 只有 64KB LDS** (`sharedMemPerMultiprocessor=65536`)。
  这直接解释了 GDN kernel (38.1KB) 每 CU 只能驻留 1 个 workgroup。
- **MMVQ(dp4a) vs MMQ(WMMA) 的分派边界 = `ne11 <= MMVQ_MAX_BATCH_SIZE (8)`**
  (`mmvq.cu:445` 的 `should_use_mmvq`)。WMMA 的 tile 是 16 行, 所以 batch 填不满 16 行时
  dp4a 反而更省。**推论**: 任何 batch <= 8 的 kernel **一次都吃不到 WMMA** ——
  普通 decode (M=1)、MTP draft (M=1) 都在此列。
  `MMVQ_RDNA4_MAX_BATCH_SIZE = 4` (2026-10-01 补丁) 把 RDNA4 的边界收到 4,
  让 ncols in [5,8] 走 WMMA; 实测 mxfp8 ncols=8 从 26.93 -> 37.18~53.57 t/s (dflash 场景)。
  注意激活量化会从 q8_1 变成 e4m3 (误差 5.1e-4~7.6e-4, 见 test-backend-ops
  `mmq_activation_is_coarse` 的 2e-3 容差)。

## 2. 吞吐模型: 瓶颈是操作数带宽, 不是 MAC 吞吐

参数扫描 (NACC 4->8->16, nblk 64->1024) 证明**已饱和**, 不是延迟受限: bf16 96.5->97.5 (+1%),
fp8 179->180 (+0.6%), iu4 345->347。汇编确认循环体未被折叠 (8 条 v_wmma 在循环内)。

把实测吞吐折算成"每 lane 每 cycle 的操作数字节":

| 输入 | MAC/CU/cyc | 每 WMMA 每 lane 操作数 | 折算 B/lane/cyc |
|---|---|---|---|
| bf16 | 648 | 32 B | 5.06 |
| fp8 | 1256 | 16 B | 4.91 |
| iu4 (K=32) | 2304 | 16 B | 4.50 |

**三个数字几乎相同 (~4.5-5.1)** -> 矩阵单元的墙是**操作数从 VGPR 喂进矩阵单元的带宽**,
不是 MAC 数。于是: **吞吐 ≈ 常数 x (MAC / 操作数字节)**。

这条规律统一解释了所有观测, 并给出唯一可行的提速杠杆:
**降低操作数位宽**。RDNA4 上:
- 16 位操作数 -> 128 MAC/byte -> ~97 T MAC/s
- 8 位操作数 -> 256 MAC/byte -> ~180 T MAC/s
- **4 位操作数 -> 512 MAC/byte -> ~347 T MAC/s**

## 3. 模型结构 (决定计算量口径)

`gguf_meta.py` 读出的权威结构 (非 MoE, dense):
```
block_count 65, embedding_length 5120, feed_forward_length 17408
attention: 24 heads / 4 kv heads / key_length = value_length = 256
ssm(GDN): group_count 16, inner_size 6144, state_size 128, time_step_rank 48, conv_kernel 4
nextn_predict_layers 1  (MTP 层存在, P4 的抓手)
```
FFN 参数 = 65 x 3 x 5120 x 17408 = 17.4B, 与 llama-bench 报的 27.32B 总参数吻合。
每 token 计算量 ~54.6 GFLOP (全部参数参与) -> pp2048 = **111.9 TFLOP = 55.95 T MAC**。

## 4. 成果 (2026-10-01, 本会话实测)

| 指标 | 起点 (GDN A2 后) | 现在 (默认 ubatch) | 现在 (`-ub 2048`) |
|---|---|---|---|
| pp512 | 2165.4 | 2318.7 | 2272.0 |
| pp2048 | 2388.8 | 2440.8 | **2577.7 (+7.9%)** |
| tg128 | 28.26 | 28.29 | 28.25 |
| PPL 8-chunk | 6.9318 | - | **6.9318 (逐位不变)** |
| MXFP4 MUL_MAT | 60/60 | 60/60 | 60/60 |
| GDN 单测 | 39/39 | 39/39 | 39/39 |

相对最初基线 1067/1134: **pp2048 1134 -> 2578 (+127%)**。

### 有效改动
1. **W fragment 合并为一次 LDS.64** (`radiance-gemm.cu` atiled 内循环)。
   原来是 `wf[j][0]=*(int*)p; wf[j][1]=*(int*)(p+4);` 两次 4 字节读。
   地址恒 8 字节对齐的理由: 行距 `LWSTR = LBK + PAD = 72` 是 8 的倍数, 且
   `kk = step*16 + kb8` (kb8 属于 {0,8})。改后 GEMM 每 call **0.2315 -> 0.226 ms (-2.4%)**。
2. **ubatch 512 -> 2048** (运行时参数, 未改代码默认)。端到端 pp2048 **+5.6%**。
   真正原因**不是** kernel 变快, 而是:
   - `launch_at_f32` 的门槛是 `M >= 2048`, ubatch=512 时 M 只有 512, 落到
     `atiled<TN=2, LBK=128>` (VGPR 230, occupancy 最低的那个);
     ubatch=2048 时 M=2048 命中 `atiled<TN=4, LBK=64>` (VGPR ~207)。
   - 更关键: 2048/512 = 4 个 ubatch 各自完整跑一遍全部 65 层, **权重被读 4 遍**;
     ubatch=2048 只读 1 遍。GEMM 调用数 3968 -> 800。
   - GEMM 单 pass 从 ~449ms 降到 ~404ms; 效率 121.9 -> **138.7 T MAC/s (77%)**。
   **注意: 不要为此降低 TN=4 的 M 门槛。** 作者的实测数据 (源码 238-243 行注释) 表明
   M=512 时 TN=4 反而**慢 8.8%** (M=2048 +1.3%, M=1024 -4.2%, M=512 -8.8%),
   原因是宽 tile 在短 M 下填不满且 N 方向 block 数减半。门槛 2048 是对的。

### 无效实验 (已回退, 勿重复)
- **去掉内循环的 `__builtin_amdgcn_sched_barrier(0)`**: 0.9810 -> 0.9868 ms/call (略差)。
  该屏障用于把同时活跃的 fragment 从 4 个 k-step 压到 1 个, 移除后 ILP 无提升而 VGPR 压力上升。
- **TN=4 路径 LBK 64 -> 128** (barrier 次数减半): 0.9810 -> 0.9792 (噪声内)。
  说明 `radiance_lds_barrier()` 的开销不是瓶颈, 这也意味着 K 循环双缓冲的预期收益不乐观
  (K 循环目前无双缓冲: 加载 A + 加载/转换 W -> barrier -> WMMA -> barrier, 加载与计算不重叠)。

## 5. rocprof 分解 (ubatch 2048, 单 pass 约 771ms)

数据源 `/tmp/pf32_results.db` (2026-10-01; `llama-bench -p 2048 -n 1 -r 1 -fa 1 -ub 2048`)。
总计 1590.5 ms / 9287 dispatches, 含 warmup 与一次性的 `radiance_repack` (47.5 ms / 496 calls,
模型加载时执行, **不计入稳态**)。分母取 1543 ms = 1590.5 - 47.5。

| kernel | 占比 | 每 pass | 备注 |
|---|---|---|---|
| `radiance_mxfp4_fp8_gemm_atiled<4,false,64,false,true>` | 50.2% | 388 ms | 效率 77% |
| `gdn_chunked_wmma_cuda<32>` | 9.2% | 71 ms | A2 成果, 已达 WMMA 吞吐 ~85% |
| `swiglu_quant_fused_kernel` | 6.9% | 53 ms | 已融合 silu+mul+quant |
| `quantize_tokens_fp8` | 5.1% | 39 ms | 432 calls/pass, ~569 GB/s 已近峰值 |
| `concat_non_cont` | 4.1% | 32 ms | 纯浪费, 见下 |
| `k_bin_bcast<op_add>` | 3.4% | 27 ms | F32 |
| `flash_attn_ext_f16` | 2.7% | 21 ms | |
| `rms_norm_f32<256,1,0>` | 2.6% | 20 ms | F32 |
| `mul_mat_vec_q<MXFP4,1,1,0,0>` | 2.1% | 16 ms | |
| `rms_norm_f32<1024,1,0>` | 2.1% | 16 ms | F32, ~706 GB/s 已超 DRAM 靠 L2 |
| `ssm_conv_long_token_f32` | 2.0% | 16 ms | F32 |
| `unary_gated_op<op_silu>` | 1.6% | 12 ms | F32 |
| `rms_norm_f32<256,0,0>` | 1.4% | 11 ms | F32 |
| `mul_mat_vec_q<MXFP4,1,0,0,0>` | 1.1% | 9 ms | |
| `mul_mat_vec_f<bf16>` | 1.0% | 8 ms | `output.weight` (BF16) 的 GEMV, 4 calls |
| `cpy_scalar` | 1.0% | 8 ms | F32 |
| `rope_multi` | 0.9% | 7 ms | F32 |
| `radiance_mxfp4_fp8_gemm_atiled<2,false,128,false,true>` | 0.6% | 5 ms | |
| `unary_gated_op<op_sigmoid>` | 0.5% | 4 ms | F32 |
| `scale_f32` | 0.4% | 4 ms | F32 |

### F32 向量运算合计 15.9% (新发现, 2026-10-01)

| 项 | 占比 |
|---|---|
| `rms_norm_f32` (三个变体) | **6.1%** |
| `k_bin_bcast<op_add>` | 3.4% |
| `unary_gated_op` (silu+sigmoid) | 2.1% |
| `ssm_conv_long_token_f32` | 2.0% |
| `cpy_scalar` | 1.0% |
| `rope_multi` | 0.9% |
| `scale_f32` | 0.4% |
| **合计** | **15.9%** |

这批全是 **elementwise / reduction**, 不是 matmul —— 这正是"F32 没走 WMMA"观感的来源,
但 WMMA 结构上用不上。两项还有带宽余量的已逐一核过:
- `rms_norm_f32<1024>` 已跑到 ~706 GB/s (超 DRAM 640, 靠 L2), **无空间**;
- `rms_norm_f32<256>` 是 one-block-per-row + **strided 标量 load, 且 x 被读两遍**,
  是唯一明显可改的, 但只占 4.0%, 优化 2x 也只有 ~2% 端到端 -> 归入零头。

### 已定位的下一个目标: GDN conv1d 输入拼接 (4.2%) —— 注意不是 QKV bias

**先说一个被证伪的假设 (勿重复)**: 最初以为 4.1% 来自 `src/llama-graph.cpp:1658` 的
QKV bias concat。**实测证伪**: 把该分支条件改成 `false &&` 后, `concat_non_cont` 的调用数
**完全不变** (96 calls / 63.4 ms)。原因是本模型的 GGUF 带 fused `wqkv_b`
(`llama-model.cpp:3291` 的 `create_tensor_qkv`), 走的是 `if (layer.wqkv_b)` 快分支,
那段 `else if` 的 concat 从未执行。**QKV bias 在这条链路上开销为零。**

**真正的来源**: `src/models/delta-net-base.cpp:472`
```cpp
conv_states = ggml_reshape_3d(ctx0, conv_states, conv_kernel_size - 1, conv_channels, n_seqs);
qkv_mixed   = ggml_transpose(ctx0, qkv_mixed);          // [C, n_tokens] -> [n_tokens, C]
ggml_tensor * conv_input = ggml_concat(ctx0, conv_states, qkv_mixed, 0);   // [3 + n_tokens, C]
```
这是 GDN 每层的 conv1d 因果输入准备 (历史 state + 当前 token)。证据链:
- `concat_non_cont` 96 calls / 1.95 passes = **49 calls/pass**, 与 **48 个 GDN 层**吻合
  (time_step_rank=48), 而不是 65 层或 MTP 的 1 层。
- 删掉 `qwen35.cpp:546` (MTP 图的 concat) 后调用数仍为 96 -> **也不是 MTP**。
- `ggml_transpose` 产生非连续输入, 因此落到 `concat.cu` 的 `concat_non_cont`,
  该 kernel 自己的注释就写着 "non-contiguous kernel (slow)"。
- 耗时与 token 数严格成正比 (ubatch 512 时 0.11 ms/call, ubatch 2048 时 0.66 ms/call)。

实测带宽: 约 50 MB 读 + 50 MB 写 / 0.66 ms ≈ **152 GB/s**, 只有 DRAM (约 640 GB/s) 的 24%
—— 逐元素 + 字节 stride 访问的典型水平。

### 优化尝试结果 (2026-10-01): 三条路线全部否决, 这条线关闭

**先纠正前提**: 真实布局是在 `concat_cuda` 里打印 ne/nb 实测得到的:
```
src0 ne=[3,10240]  nb=[4,12]      通道慢变, 3 个时间步连续
src1 ne=[2,10240]  nb=[40960,4]   通道连续 (真转置视图)
dst  ne=[5,10240]  nb=[4,20]      时间步连续, 通道慢变
```
即 `dst(k+t, c) = src1(t, c)` 是**真正的转置** (`src1` 的 (t,c) 在 flat `t*C+c`,
`dst` 的在 flat `(k+t)+c*(k+T)`), **不是行拷贝**。

1. **memcpy 方案 -> 否决 (而且是错的)**: 误判为"src1 连续且顺序即 dst 所需", 用两次
   `cudaMemcpyAsync` 替换。PPL 从 6.9318 直接爆到 **424051**。**CONCAT 单测 177/177 全绿
   也拦不住** —— 测试集里没有一个用例满足"转置视图 src1"的条件, 新路径零覆盖。
   这是"测试 0 覆盖"陷阱在本项目的第三次重演 (前两次: GDN chunked 36/36 假通过、
   atiled 隔离台尺寸 bug)。**教训: 通用算子加特化分支后必须构造真实形状用例, 或以 PPL 为准。**
2. **tile 转置方案 -> 否决 (正确但更慢)**: 修正布局推导后重写 (共享内存 32x33 tile +
   `cudaMemcpy2DAsync` 处理 src0 头部), PPL 恢复 6.9318, 但 kernel 实测
   **1.0228 ms/call, 比原版 `concat_non_cont` 的 0.660 慢 55%**, pp2048 掉到 2545。
   原因: 原版**写是完全连续的**, 读虽 strided 但 320 个 block 并发读相邻地址时 **L2 会
   合并请求**, 实测已达 **254 GB/s** (DRAM 的 40%); tile 版反而引入共享内存往返、
   每 block 一次 `__syncthreads`、1024 线程的调度开销, 净亏。
   **教训: "strided 读一定慢"是错的假设, 先测出原版带宽再决定要不要重写。**
3. **环形缓冲 -> 未实施, 判定不划算**: 能连 transpose 一起消除, 上限 4.2%
   (concat 占单 pass 约 33ms / 790ms), 但要改 `ssm_conv` kernel 的索引和 state 更新路径
   (`delta-net-base.cpp:472-496` 还要从 concat 结果尾部 view 出 state), 风险与收益不成比例。
   **路线 2 复核后确认不可行**: conv 沿时间卷积必须是 `[时间连续]` 布局, 而 GEMM 输出天然是
   `[通道,时间]`, transpose 绕不开。

**结论: `concat.cu` 已回退到 pristine, 这条线关闭。**

## 6. 复现命令
```bash
# 正确性
./build/bin/test-backend-ops -b ROCm0 -o MUL_MAT -p "type_a=mxfp4" test     # 60/60
./build/bin/test-backend-ops -b ROCm0 -o GATED_DELTA_NET test               # 39/39
# 性能 (推荐带 -ub 2048)
./build/bin/llama-bench -m <gguf> -dev ROCm0 -p 512,2048 -n 128 -r 5 -fa 1 -ub 2048
# 数值门
./build/bin/llama-perplexity -m <gguf> -f /media/seirin/HDD500G/corpus.txt -dev ROCm0 -c 4096 -ngl 99 -fa 1 --chunks 8 -ub 2048
```

## 5. 复现命令
```bash
# 指令集验证 (权威)
grep -c wmma /opt/rocm/core-10.0/lib/llvm/include/clang/Basic/BuiltinsAMDGPU.inc
echo 'v_wmma_f32_16x16x4_f32 v[0:7], v[8:9], v[10:11], v[12:19]' > /tmp/i.s
/opt/rocm/core-10.0/lib/llvm/bin/llvm-mc -arch=amdgcn -mcpu=gfx1201 -filetype=obj -o /tmp/i.o /tmp/i.s
# 峰值微基准 (源在 .49:/tmp/wmma_peak2.hip, /tmp/scan.hip, /tmp/scan4.hip)
```

## 7. 推荐运行配置 (2026-10-01 结论)

**ubatch 用 2048, 不要用默认的 512。** 实测 +5.6% (pp2048 2440.8 -> 2577.7)。

两条独立机制叠加:
1. `launch_at_f32` 的 TN=4 门槛是 `M >= 2048`。ubatch=512 时 M=512, 落到最慢的
   `atiled<TN=2, LBK=128>` (VGPR 230, occupancy 最低的那一档)。
2. **更主要**: 2048/512 = 4 个 ubatch 各自完整跑一遍全部 65 层, **权重被从显存读 4 遍**;
   ubatch=2048 只读 1 遍。GEMM 调用数 3968 -> 800, 单 pass 449 -> 404 ms。

验证: PPL 逐位不变 6.9318; tg128 不受影响 (28.25 vs 28.29, chunked GDN 只在 n_tokens>=64 生效);
MXFP4 MUL_MAT 60/60; GDN 单测 39/39; 显存无 OOM (32GB R9700)。

**注意: 尚未改代码默认值** (保持 llama.cpp 的 512), 因为 ubatch 是全局参数、直接决定
激活显存占用, 换更小的卡或其他模型可能 OOM。需要时在启动命令加 `-ub 2048`:
```bash
./build/bin/llama-bench      -m <gguf> -dev ROCm0 -p 512,2048 -n 128 -r 5 -fa 1 -ub 2048
./build/bin/llama-perplexity -m <gguf> -f <corpus> -dev ROCm0 -c 4096 -ngl 99 -fa 1 --chunks 8 -ub 2048
./build/bin/llama-cli        -m <gguf> -dev ROCm0 -ngl 99 -fa 1 -st --temp 0 -n 64 -p "..." -ub 2048
```

## 8. GEMM 的架构上限 (切结: 不要重复挖)

`radiance-gemm.cu` 的 atiled kernel 头部带有一份**完整 ablation 记录** (2026-09-01..09-04,
作者在 `~/mxfp4_work/tier7/` 做的)。这是本项目最有价值的先验信息, **动手前必读**, 结论如下。

### 时间分解 (TN=4, M=4096; WMMA 数量固定, 逐项摘除)
| 组成 | 占比 |
|---|---|
| A fragment loads | **28%** (其中 L2->L0 流 ~18%, issue/latency ~10%) |
| W staging | 14% (含 fold VALU 8%) |
| LDS fragment reads | 11% |
| barriers | **1%** |
| WMMA-only | 310 TF/s = 完整时间的 **55%** |

即 **WMMA-only + 各项开销 ≈ 90% -> 内存工作几乎没有与矩阵工作重叠**。

### 作者已实测并否决的方向 (勿重复)
- **LBK 128 @ TN=4**: 246 VGPR, occupancy 5, 慢 12-17% (我 2026-10-01 独立复测: 0.9810 -> 0.9792,
  噪声内, 与"barrier 只占 1%"一致)。launcher 按 TN 选 LBK 是正确设计。
- **A fragments 软件流水 (提前一个 slab)**: TN=4 中性, TN=2 因 occupancy 损失慢 2-5%。
  关键证据: **A 的开销是 L2 流而非延迟** (cache-hot A 的 ablation 仍比 no-A 慢 10%)。
- **预取下一 slab 的 W 到寄存器**: ±1%。
- **W 直接进寄存器 (完全不过 LDS)**: 各形状慢 1.10-1.30x (fold 被全部 WM=4 个 wave 重复做)。
- **TN=8**: 撞累加器墙。**A 走 LDS**: 那就是 folded kernel (更慢)。**TM=3/TN=6**: 净收益 ~0。
- **WPERM (fragment-order 权重布局)**: 与 checkpoint 布局相差在 1% 以内。

### 三处已合入的 codegen 修复 (合计 +1.8..+2.6% 端到端)
1. A fragment 用 SGPR 基址 (`readfirstlane` + 无符号 32 位 lane offset): 16 个 load 编译成
   一条 `s_clause` 的 SADDR `global_load_b64`, 消掉 8 条 `v_add_co` 链与 `s_wait_alu` stall
   (主循环 VALU 63 -> 47 per 64 WMMA)。
2. `kMag` 改从 LDS 取: 它原是全局加载, 依赖 A fragments 之后发射, 而 loadcnt 是顺序的,
   于是 fold 前的 `s_wait_loadcnt 0x0` 会把整批 A 排空。
3. **LDS-only barrier fence**: `__syncthreads()` 是 workgroup 范围的全地址空间 acq/rel
   (含 `global_inv` + loadcnt 排空, 每 slab 两次); 这里 wave 间不共享全局数据,
   所以 fence 收窄到 LDS (`s_wait_dscnt + s_barrier`)。

### 结论
> the remaining ~40% over the WMMA floor is **structural at this tile**

**GEMM 那 51% 的占比已经没有低风险可挖的空间。** 要突破必须换架构 (例如从根上减少 A 的
L2 流量), 不是调参数能解决的。本项目在 GEMM 上新增的有效改动只有 LDS.64 (-2.4%, 见第 4 节),
属于作者未覆盖的小项。

**因此后续投入应转向: (a) 其他占比 5% 级的独立 kernel (swiglu_quant 6.9%, quantize_tokens 5.0%);
(b) P4 / TG 侧 (当前 tg128 28.26, 且有 nextn MTP 层未利用)。**
