# GDN prefill 交接记录 (2026-10-01): chunk 算法的瓶颈定位与改进评估

> 本文件是 `design-gdn-rdna4.md` 的续篇, 只覆盖 **GDN chunked prefill 的下一步优化**。
> 阅读顺序: 先看 `design-gdn-rdna4.md` 的 §4 (A2 WMMA 复刻的既成事实), 再看本文件。
> 本文件所有数字均为 2026-10-01 实测, 探针与 trace 已归档 (见 §7)。
>
> **[2026-10-01 晚 补记] §4 的方案已实施并实测完毕, 结论有修正。先读 §10。**

---

## 0. 一句话结论

**GDN chunked prefill 的瓶颈不是 WMMA, 而是 chunk 循环里 `if (warp == 0)` 的 32x32 三角求逆 —— 它由单个 warp 串行完成。**

- 该串行段**(预热后)**单独耗时 **14.40 us/chunk**, 而整个真实 kernel 是 **23.45 us/chunk** —— 求逆段占 **61%**。
- 整个 kernel 的 WMMA 只用到 **4.15%** 的 bf16 WMMA 峰值算力 (618.5 GFLOP 理论下限 = 2.95 ms vs 实测 70.97 ms/forward)。
- 因此 GDN 的优化杠杆在**串行关键路径**, 不在张量核。可动的手术是"把求逆的深度从 32 降到 16, 并把修正项交给 WMMA"。
- **注意量级**: 收益按 wall time 估算为 **+0.8% (列切分) ~ +3% (求逆砍半)**, 理论上限 +6.2% (见 §4.3)。
- **[补记] 实测结果超出上述估算**: 真正的瓶颈不是并行度也不是依赖链, 而是**求逆列的 LDS 往返**;
  改掉后 GDN kernel 从 23.81 -> 10.33 us/chunk (**2.30x**), pp2048 **+4.7%**。见 §10。


---

## 1. 被测对象与现状 (勿再重新推导)

**kernel**: `gdn_chunked_wmma_cuda<C=32>`, fork 内 `ggml/src/ggml-cuda/gated_delta_net.cu`

| 事实 | 值 | 位置 |
|---|---|---|
| 生效条件 | `S_v==128 && neq0==nek0==128 && n_tokens>=64 && !kda && !keep_rs && RDNA4` | `gated_delta_net.cu:171-183` |
| 关闭开关 | `GGML_GDN_CHUNKED=0` (回退 recurrent) | 同上 |
| chunk 长度 C | 32 (`static_assert(C == 32)` 锁死 warp job map) | `gated_delta_net.cu:257,264` |
| chunk 主循环 | `for (int t0 = 0; t0 < n_tokens; t0 += C)` | `gated_delta_net.cu:306` |
| launch 几何 | `grid(H, n_seqs, 1)`, `block(256)` = 8 warps | `gated_delta_net.cu:534-535` |
| smem | `(2*C*KP + KD*KT)*2 + (C*CP + 4*C)*4 + 3*C*CB*2` = **38144 B** (38.1 KB / 37.2 KiB) | `gated_delta_net.cu:531-533` |
| 循环内 `__syncthreads` | **6 次** (行 332 / 370 / 392 / 419 / 442 / 503) | — |

**H = 48** (`qwen35.ssm.time_step_rank`), 所以 grid.x = 48, 而卡是 **64 CU** —— **16 个 CU 天然空转**。

**A/B 价值已确认** (同 build, 只切 `GGML_GDN_CHUNKED`, llama-bench `-ub 2048 -fa 1 -r 3`):

| | pp2048 | pp512 | tg128 |
|---|---|---|---|
| chunked ON (默认) | **2560.44 ± 269** | 2218.11 ± 596 | 29.90 ± 0.21 |
| chunked OFF (recurrent) | 2216.55 ± 200 | 1962.37 ± 473 | 29.84 ± 0.22 |
| 收益 | **+15.5%** | +13.0% | 持平 (decode 走 recurrent) |

> 注: 这条线的价值已经被吃到了。本文件讨论的是**还能再榨多少**。

---

## 2. 决定性实验: 串行段单独计时

方法: 把 kernel 里两处显式串行段**原样**抽成独立 kernel, 用同构几何计时
(48 blocks x 256 thr, 64 chunks = pp2048 的形状)。
探针源码: `backups/gdn-phase-probe-20261001/gdn_phase_probe.hip`

**必须预热后再读数。** 探针首次运行会读到 GPU 时钟尚未爬升的状态, 数字虚高约 1.6 倍
(实测: 首次 23.2, 之后 12 次全部稳定在 14.36~14.45)。下表是**预热后连跑 6 次的中位值**:

| 探针 | 稳态实测 | 说明 |
|---|---|---|
| Phase A: `if (tid==0)` 32 步累加 + `expf` | **0.88 us/chunk** | `gated_delta_net.cu:333-339` |
| **Phase B: `if (warp==0)` 三角求逆** | **14.40 us/chunk** | `gated_delta_net.cu:393-418` |
| A + B | 15.50 us/chunk | — |
| **真实 `gdn_chunked_wmma_cuda<32>`** | **23.45 us/chunk** | 重测中位 (96 calls, min 1320 / max 1540 us/call) |

**Phase B 占真实 kernel 的 61%** —— 单个 warp 的串行求逆吃掉了大部分每-chunk 时间,
其余工作 (每 block 每 chunk 512 条 WMMA + staging + 同步) 分摊在剩下的 39% 里。

> **读数纪律 (踩过的坑)**: 探针**不做预热**会得到虚高一倍的数字 (23.2 vs 14.4), 因为独立 HIP
> 程序启动时 GPU 还处于空闲时钟。同一个探针在同一次会话里连跑, 第一次永远是异常的。
> **任何探针结论都要用预热后的多次中位值**, 这条同样适用于 `gp2`/`gp3`/`wb2`。
> 真实 kernel 的数字 (23.45) 来自 llama-bench 的长负载, 时钟已爬满, 不存在这个问题。

### 2.1 Phase B 是纯串行延迟, 与并行度无关

用 `gp2.hip` 扫 block 数 (同一份 Phase B 代码, **预热后**, 连跑 3 次数字一致):

| blocks | us/chunk |
|---|---|
| 48 (= 真实 grid) | 14.77 |
| 64 | 14.71 |
| 128 | 15.17 |
| 192 | 15.69 |
| 384 | 30.33 |

**4 倍并行度 (48 -> 192), 耗时几乎不变 (14.8 -> 15.7)。** 典型的单 warp 串行链特征 —— 加 block 无效。
(384 blocks 变差是每 SM 驻留的 block 数超过了并发能力。)

注意与真实 kernel 的对照: 这里 blocks=48 的 14.77 就是 Phase B 在 48 个 block 上的 wall time,
与 §2 表格的 14.40 一致 (同一段代码, 两次测量)。

### 2.2 为什么是串行

`gated_delta_net.cu:393-418` 的原始代码:

```cpp
if (warp == 0) {
    const int j = lane;                       // lane j 独占第 j 列
    for (int i = 0; i < C; ++i) {
        // 填 M = I + strict_lower(beta_t * exp(c_i-c_j) * gram)
    }
    __syncwarp();
    for (int i = 0; i < C; ++i) {
        if (i < j)       s_x[i*CP+j] = 0.0f;
        else if (i == j) s_x[i*CP+j] = 1.0f;
        else {
            float acc = 0.0f;
            for (int m = j; m < i; ++m)          // 内层最长 31 步, 严格顺序依赖
                acc += s_gr[i*CP+m] * s_x[m*CP+j];
            s_x[i*CP+j] = -acc;
        }
    }
}
```

- **8 个 warp 里只有 1 个在干活, 另外 7 个 (224 线程) 在 `__syncthreads()` 上空等。**
- 第 i 行依赖第 i-1..0 行, 是真正的顺序依赖, 无法直接展开。
- 三角求逆本身的数学深度是 O(C) = 32 层。

---

## 3. 张力核对: WMMA 到底用了多少

按源码数每个 warp 每 chunk 的 WMMA (16x16x16 每条 = 8192 flop):

| 阶段 | 公式 | 条数 |
|---|---|---|
| gram = K·K^T | `for kb<8` | 8 |
| k·S0 / q·S0 | `8 kb x 2 mb x 2 (k,q)` | 32 |
| D = (I+L)^-1·RHS | `2 kb x 2 mb` | 4 |
| 状态更新 | `2 kb x 8 ib` | 16 |
| o = qw·D | `2 mb x 2 kb` | 4 |
| **每 warp 合计** | | **64** |
| **每 block (8 warp)** | | **512** |

- 每 block 每 chunk = 512 x 8192 = **4.194 MFLOP**
- 无冗余理想值 = **3.998 MFLOP** -> **冗余仅 1.049x** (源码已经写得很紧)
- 全 48 层 pp2048: 512 x 8192 x 48 head x 64 chunk x 48 layer = **618.5 GFLOP**
- 按实测 bf16 WMMA 峰值 **210 TFLOP/s** -> 理论下限 **2.95 ms**

**而实测每个 forward 的 GDN = 70.97 ms -> 达成率 4.15%。**

> **这就是判决性证据: WMMA 完全不是瓶颈, 连算力都没有被碰到。**

---

## 4. 改进方案: blocked inverse (建议的实施路径)

### 4.1 核心思路

把 32x32 的单位下三角求逆切成 2x2 的 16x16 块:

```
        | D11   0  |            D11 = L11^-1,  D22 = L22^-1   (两个独立的小三角求逆)
L =     |          |     ->     
        | L21  D22 |            X21 = -D22 * L21 * D11      (纯矩阵乘, 可以交给 WMMA)
```

- **深度从 32 降到 16** (对角块各自是 16x16 的三角求逆)
- **修正项 `-D22 * L21 * D11` 是矩阵乘法**, 可以脱离串行路径交给 WMMA
- 8 个 warp 可以全部参与 (2 个包对角块, 其余包修正项)

### 4.2 已实测的下界: 列切分 -15.4%

用 `gp3.hip` 试了最直接的并行化 —— **按列切分给 4 个 warp** (`warp*8` 起始列)。
**预热后**连跑 3 次 (blocks=48):

| 版本 | us/chunk |
|---|---|
| 1 warp (现状) | 12.05 |
| **4 warp (列切分)** | **10.20** |
| | **-15.4%** |

**为什么列切分能 work**: 求逆的**列之间完全独立** (`x[:,j]` 只依赖 `x[:,j]` 自身的前序行, 不依赖其他列)。
所以按列分给不同 warp 不改变任何依赖关系。这是块状三角求逆的标准做法。

> **诚实说明**: `gp3.hip` 里的 4-warp 版本是**理想化探针** (列切分 + 每列 4 lane), **不是可直接合入的正确实现**。
> 它证明的是"这个串行段有 ~15% 的可压缩空间", 不是"照抄这段代码就行"。
> 另注意 gp3 的 1-warp 基线 (12.05) 与 gpp 的 Phase B (14.40) 不同 —— 两个探针的循环体不完全一样
> (gp3 省掉了 `expf` 填充那一段), 所以**跨探针不要混用数字**, 只看各自的相对比值。

### 4.3 收益天花板 (必须说清楚)

**先把单位讲清楚, 否则会算错一个数量级:**

- Phase B 探针的 **14.40 us/chunk 是单个 block 的成本**; 一个 block = 一个 `(seq,head)` = 一层的 1/48。
- 真实 kernel 每 chunk 的 **wall time 是 23.45 us** —— 因为 48 个 block 在 48 个 CU 上**并行**跑。
- 所以每层每 forward = 64 chunk x 23.45 us = 1.501 ms; x 48 层 = **72.0 ms**, 与实测 70.97 ms 吻合。

换算时**按 wall time 算, 不能再乘层数**。推导口径: "每 chunk 省下 X us, 则每 forward 省 `X * 64 * 48`":

| 假设 | 每 chunk 省 | 每 forward 省 | pp2048 收益 |
|---|---|---|---|
| 串行段砍半 (14.40 -> 7.20) | 7.2 us | 22.1 ms | **+3.0%** |
| 按已实测的列切分 -15.4% | 1.85 us | 5.7 ms | **+0.8%** |
| 理论上限 (串行段耗时归零) | 14.40 us | 44.2 ms | **+6.2%** |

> **这三行是量级估计, 不是预测。** 探针是把该段单独抽出来测的, 与真实 kernel 里"和其他工作重叠"
> 的情形不完全可比; 表中还**未扣除**求逆的残余成本与新增的 WMMA/同步开销。
> **实际拿到多少必须改完实测, 不要用这三行对外报数。**
>
> 保守说法: **列切分级别的小改 -> +1% 以内; 完整的 blocked inverse -> 有望到 +2~3%; +6.2% 是理论上限。**
>
> **[补记] 这三行全部被实测推翻, 且是低估。** 实测 pp2048 **+4.7%**。
> 原因是本表建立在"串行段耗时 = 依赖链"的错误前提上; 真实瓶颈是 LDS 往返, 见 §10.1。
> **保留此表是为了记录推理过程, 不要再用它做决策。**

| 口径 | GDN 占比 | 说明 |
|---|---|---|
| prefill pp2048 | 71.08 / 759.24 = **9.4%** | 优化目标 |
| 纯 decode | 1.22% | 走 recurrent, 不受影响 |
| spec decode | 3.8% GPU busy | `n_tokens=8 < 64`, 走 recurrent, 不受影响 |

**结论: 值得做, 但要认清这是 "prefill +1~3%" 级别的优化, 且是数值敏感的核心代码。**

GDN 在 pp2048 的完整分解 (归档 trace `prefill_pp2048.csv`, **含 2 个 forward**):

| 项 | ms (2 fwd) | ms (1 fwd) | 占比 |
|---|---|---|---|
| radiance GEMM (fp8 WMMA) | 781.76 | 390.88 | 51.5% |
| other | 498.33 | 249.17 | 32.8% |
| **GDN chunked (bf16 WMMA)** | **142.16** | **71.08** | **9.4%** |
| radiance repack (仅 warmup, 一次性) | 47.39 | — | 3.1% |
| flash-attn | 44.66 | 22.33 | 2.9% |
| MMQ/MMVQ | 4.18 | 2.09 | 0.3% |
| **合计** | **1518.47** | **759.24** | 100% |

> 该 trace 覆盖 **2 个 forward** (992 radiance 次 = 496 warmup + 496 实测), 即 warmup 已摊薄;
> `GDN chunked` 的 96 次 = 48 层 x 2 个 forward (trace 在第 48 次后有 219.7 ms 空隙, 即 forward 边界)。
> 单 forward GDN = 70.97 ms, 与 `design-gdn-rdna4.md` §6 记的 71 ms/次 一致。

> 注: 上表占比的分母是"所有 kernel 时间之和", 不是 wall clock (pp2048 实测 2199.6 t/s = 931 ms),
> 两者不等是正常的 (有 launch 间隙)。**收益百分比按 kernel 时间的口径估算。**

---

## 5. 其他备选方向 (优先级低于 §4)

按"已定位"程度排序:

1. **C=64 (chunk 加倍)** —— 同步次数减半、WMMA 的 K 维更长。但 per-token MAC 从 65536 涨到
   81920 (+25%), 且 LDS 需求按同一公式算 = **95744 B (93.5 KiB)**, 而实测本卡
   `sharedMemPerBlockOptin = 65536 B (64 KiB)` —— **超预算 1.46 倍**。若要做需先做 LDS 复用:
   `s_kt` (18.4 KiB at C=64) 改为从 `s_k` 转置读、`s_qk`/`s_qw` 原地复用、`s_inv` 写回 `s_gr` 区。
   **风险高, 且与 §4 不冲突 (可叠加)。**
2. **打破 k·S0 / q·S0 的累加链** —— 现 8 个 WMMA 累加到同一寄存器 (链长 8), 可拆成 2 个累加器再相加。
   代价是多 32 个 VGPR。**但既然 WMMA 只占 4.15% 的算力, 这条几乎肯定没有收益。**
3. **并行度不足 (grid 只有 48 blocks, 16 CU 空转)** —— 这是 `design-gdn-rdna4.md` 留档里的头号猜测。
   **本次实验证伪了它作为首要瓶颈**: Phase B 扫 block 数 (48 -> 192) 耗时不变 (见 §2.1),
   说明 kernel 时间被单 warp 串行链锁住, 加并行度无效。**除非先解掉 §4 的求逆, 否则深 grid 无意义。**
4. **状态加载/存储非合并** —— 每 lane 32B 连续但跨 16 个不同行, DRAM 事务效率约 50%。
   但总量 = 48 层 x 48 head x 64 KiB x 2 (R+W) = **302 MB**, 按实测 DRAM 637.7 GB/s 仅 **0.48 ms**,
   占 GDN 71 ms 的 **0.7%** —— **不是瓶颈**。
   (旧文档 `design-gdn-rdna4.md` §GDN剩余优化空间 记的是"29 层 x 128 KB = 356 MB", 层数与每头字节数都不对;
   本模型 GDN 层是 48, 每 (seq,head) 状态是 128x128x4 = 64 KiB。)**结论不变, 但不值得做。**

---

## 6. 已知陷阱 (本次新增)

沿用 `design-gdn-rdna4.md` §5 的全部陷阱。本次新增:

- **`if (warp == 0)` 这种"单 warp 干活"的段落在 profiler 里不显眼** —— 它不加 block 数、不加 VGPR、
  occupancy 也正常, 只有把该段抽出来单独计时才会暴露。**定位串行段要靠"抽段计时", 不能只看占用率。**
- **探针必须预热, 否则数字虚高一倍。** 独立 HIP 程序刚启动时 GPU 还在空闲时钟,
  首次读数会异常 (实测 Phase B: 首次 23.2, 之后稳定 14.4)。**本文件的探针数字全部是预热后
  连跑多次的中位值**; 任何时候重跑探针, 都要先丢弃首次读数再报数。这条适用于本目录下所有 .hip 探针。
- **`us/chunk` 在预热后是恒定值, 且与 block 数无关 (48~192)** -> 这是串行链的判据。
  若某改动让 us/chunk 开始随 block 数变化, 说明该段已经不再是串行关键路径。
- 列切分探针 (`gp3.hip`) 的 4-warp 版本**仅为计时用**, 其数值输出是错的 (未做正确性验证),
  **不要拿它当参考实现**。另外它的 1-warp 基线 (12.05) 与 `gpp` 的 Phase B (14.40) 不同,
  因为两者循环体不一样 —— **跨探针不要混用绝对数字**, 只看各自内部的相对比值。

---

## 7. 复现材料 (已归档, 不依赖 /tmp)

```
/media/seirin/HDD500G/backups/gdn-phase-probe-20261001/          (12 MB, 权限 644/755)
├── gdn_phase_probe.hip   md5 d9ebfa8fbaf01e0be486aad175b6e0d1   Phase A / B / A+B 计时
├── gp2.hip               md5 c2b2cec2d0b42aa892d4da49af9ebe81   Phase B 扫 block 数
├── gp3.hip               md5 e020a6457e75e79a270548754debefb5   1-warp vs 4-warp 列切分 (仅计时)
├── bw.hip                md5 ddb08f85198183c8a7a2d66dc25c6992   DRAM 带宽标尺 (copy/read)
├── wb2.hip               md5 233052fa5841bdd3f9446707d2f55718   WMMA 峰值微基准 (NACC 扫描)
├── fp32.hip              md5 19264d32bf7adcf1b97739e0ea874153   FP32 FMA 峰值 (反推 SIMD 数)
├── fp32.bin / bw.bin / wb2.bin                                  已编译好的二进制 (可直接跑)
└── traces/
    ├── k64_kernel_trace.csv      GDN chunk 伸缩 n=64
    ├── k2048_kernel_trace.csv    GDN chunk 伸缩 n=2048
    ├── decode_tg8.csv            纯 decode tg8 全 kernel 分解
    └── prefill_pp2048.csv        pp2048 全 kernel 分解
```

编译与运行 (在 .49 上, **先预热再读数**):

```bash
cd /media/seirin/HDD500G/backups/gdn-phase-probe-20261001
hipcc --offload-arch=gfx1201 -O3 -o gpp gdn_phase_probe.hip
hipcc --offload-arch=gfx1201 -O3 -o gp2 gp2.hip
hipcc --offload-arch=gfx1201 -O3 -o gp3 gp3.hip
hipcc --offload-arch=gfx1201 -O3 -o fp32 fp32.hip
# 预热: 每个探针先空跑 5 次丢弃, 再取 3~6 次读数的中位
for i in 1 2 3 4 5; do ./gpp > /dev/null; done
for i in 1 2 3;       do ./gpp | grep -E "phase A only|phase B only|A \+ B"; done
```

> 探针自带的 `printf` 只报单次值。**必须自己连跑多次取中位** —— 单跑一次的数字不可信 (§6)。

### 7.1 GDN chunk 伸缩原始数据 (每 chunk 成本恒定)

`gdn_chunked_wmma_cuda<32>`, 96 calls = 48 层 x 2 个 forward (warmup + 实测):

| n_tokens | 每次调用 | chunks | us/chunk |
|---|---|---|---|
| 64 | 48.2 us | 2 | 24.09 |
| 128 | 93.5 us | 4 | 23.39 |
| 512 | 341.7 us | 16 | 21.36 |
| 2048 | 1478.5 us | 64 | 23.10 |

**us/chunk 恒定 (21.4~24.1), 无固定开销** -> 确认是串行 chunk 链, 每 chunk 成本相同。

独立重测确认 (避免单次 trace 的偶然性, `traces` 之外的 `/tmp/gr/g_kernel_trace.csv`):
同一 pp2048 形状, 96 calls, **中位 1500.5 us/call = 23.445 us/chunk** (min 1320.5 / max 1540.2)。
与上表 n=2048 行的 23.10 一致。**23.445 是本文件各处使用的"真实 kernel 每 chunk"值。**

### 7.2 硬件标尺 (本次实测, 用于所有达成率换算)

| 项目 | 实测 | 备注 |
|---|---|---|
| FP32 FMA 峰值 | **21.1 TFLOP/s** (2 flop/fma) | 反推 ~140 SIMD → 确认 64 CU x 2 SIMD = 128 |
| bf16 WMMA `16x16x16` | **210.2 TFLOP/s** | NACC=16; NACC>=24 因寄存器溢出暴跌 |
| f16 WMMA `16x16x16` | 213.5 TFLOP/s | 同上 |
| fp8 WMMA `16x16x16` | **329 TFLOP/s** | NACC=20 峰值 (NACC=4 时仅 233, 是延迟受限) |
| DRAM 纯读 (1 GiB) | **637.7 GB/s** | 规格 638.9, 吻合 |
| DRAM copy (R+W) | **567.5 GB/s** | — |
| **LDS 上限** | **65536 B (64 KiB)** | `sharedMemPerBlockOptin`; C=64 需 93.5 KiB -> 不可行 |

> `rocprofv3` 的 `TCC_EA_*` / `FETCH_SIZE` / `SQ_INSTS_*` 计数器在 **RDNA4 + rocprofv3 1.3.5 下全部返回 0**,
> 只有 `GRBM_GUI_ACTIVE` 和 `SQ_WAVES` 有效。**bandwidth 类结论必须靠微基准标尺, 不能靠 rocprof PMC。**

---

## 8. 验收门 (改动后必须全过)

改动 GDN 后按此顺序验证, 缺一不可:

```bash
# 1. 单测 (39 例; 必须确认 selected > 0 且命中 chunked 路径, 否则是假通过)
./build/bin/test-backend-ops -b ROCm0 -o GATED_DELTA_NET test

# 2. 与 recurrent 对拍: 同一形状下两条路径输出 + 末态一致
GGML_GDN_CHUNKED=0 ./build/bin/llama-cli ...   # 对比贪心输出逐字节

# 3. PPL 门 (标准口径, 勿改参数)
./build/bin/llama-perplexity -m <gguf> -f /media/seirin/HDD500G/corpus.txt \
    -dev ROCm0 -c 4096 -ngl 99 -fa 1 --chunks 8 -ub 2048
#   当前红线: PPL = 6.9633 +/- 0.12494 (必须逐位一致或明确说明漂移)

# 4. 性能 (必须 -ub 2048, 否则 radiance 走 TN2 路径)
./build/bin/llama-bench -m <gguf> -dev ROCm0 -p 512,2048 -n 128 -r 3 -fa 1 -ub 2048
```

**数值风险提示**: 三角求逆是数值敏感路径, 48 个 GDN 层共用。**先做位级正确性对照再谈性能** ——
一旦有数值问题会直接毁掉 PPL。

---

## 9. 当前状态 (2026-10-01 首次撰写时的快照)

> **[补记] 本节描述的是写 §1~§9 时的状态, 已被 §10 取代。** 当时确实零改动。

- **代码零改动**: 本次只做了分析, `git status` 干净, HEAD = `f523e367d` (分支 `Radiance`)。
- GDN 相关源码未动一行; 所有实验都在 `/tmp` 的独立 .hip 探针里完成, 未触碰仓库。
- 归档: `/media/seirin/HDD500G/backups/gdn-phase-probe-20261001/` (12 MB)

**实施后的状态见 §10.4 / §10.8**: 工作区有 1 个未提交改动
(`ggml/src/ggml-cuda/gated_delta_net.cu`, +59/-22, md5 `b8022eb78865de8fd9e83904bbf7eaa7`),
HEAD 仍是 `f523e367d`。**未 commit, 未 push。**


---

## 10. [补记 2026-10-01 晚] 已实施: 求逆段重写, 实测结果

§4 的方案**已经做完并落地**。结论与 §0/§3/§4 的推测**有实质出入, 以本节为准**。

### 10.1 真正的根因不是"串行", 是 LDS 往返

原代码 (`gated_delta_net.cu:393-418`, 改动前):

```cpp
if (warp == 0) {
    const int j = lane;
#pragma unroll 1
    for (int i = 0; i < C; ++i) {
        if (i < j)       s_x[i*CP+j] = 0.0f;
        else if (i == j) s_x[i*CP+j] = 1.0f;
        else {
            float acc = 0.0f;
#pragma unroll 1
            for (int m = j; m < i; ++m)
                acc += s_gr[i*CP+m] * s_x[m*CP+j];   // <-- 每次从 LDS 读回自己刚写的值
            s_x[i*CP+j] = -acc;
        }
    }
}
```

**lane j 自己写了 `s_x[m*CP+j]` 的每一个值, 却在内层循环里每次都从 LDS 重新读回来。**
`#pragma unroll 1` 又禁止展开, 于是整个内层是"LDS + LDS + FMA"的串行链, 而且**每个元素还要
多付一次写回 LDS**。这不是依赖链的问题, 是访存的问题。

**验证方法**: 逐项拆开测 (探针 `gb2.hip`, 48 blocks x 256 thr, 64 chunks, 预热后中位):

| 变体 | us/chunk | vs 现状 | 位级差异 |
|---|---|---|---|
| A = 现状 (loop1 warp0 + loop2 LDS) | 15.291 | 1.00x | — |
| B = loop1 全场 + loop2 LDS | 12.181 | 1.26x | 0/528 |
| **C = loop1 全场 + loop2 寄存器驻留** | **2.424** | **6.31x** | **0/528** |
| D = C + 四累加器 | 2.491 | 6.14x | 232/528 (7.5e-9) |
| E = blocked 16x16 | 1.486 | 10.29x | 140/528 (7.5e-9) |

- **D 比 C 慢** -> 说明瓶颈不是累加链深度, 拆累加器无效。**这证伪了"依赖链是瓶颈"的假设。**
- **C 位级完全一致 (0/528)**: 因为 lane j 本来就独占该列, 且 `m` 从 0 开始只是多加了前导的精确零项,
  加法顺序与从 `m=j` 开始完全相同。**这个改动是零数值风险的。**
- E 只比 C 再快 0.38% 却引入数值差异, **单独看不值得** (但它与 C 叠加后价值不同, 见 10.3)。

### 10.2 落地顺序: 先零风险的 C, 再叠加 E

**第一步 (寄存器驻留, 位级等价)** —— 同时把 loop1 从"warp 0 逐列 32 次 expf"摊到全场
(32x32 = 1024 元素 / 256 线程 = 每线程 4 个)。

**第二步 (blocked 16x16)** —— 对角两块由 warp 0/1 并行做深度 16 的求逆, 修正项 `X21 = -X22 L21 X11`
交给全场 256 线程。**数学恒等**, 与 fp64 参考的误差 **1.261e-08 反而小于**寄存器版的 1.577e-08
(分块减少了串行舍入的累积)。

> **实施时踩到的坑 (必须记住)**: blocked 版本只写了下三角和两个对角块,
> **没有写右上块** (第 0-15 行 x 第 16-31 列)。原代码用 `if (i < j) = 0` 覆盖了整个严格上三角,
> 而 `s_inv` 稍后会把整个 32x32 截断成 bf16, **右上块会被读到**。
> 补上 `s_x[rr*CP + 16 + cc] = 0.0f` 后才正确。
> 探针当时只校验下三角, **没有暴露这个 bug** —— 教训: 对拍要覆盖全矩阵, 不能只看"有意义的"那一半。

### 10.3 实测收益 (同会话交替 A/B, 换 `libggml-hip.so.0.24.0`)

llama-bench `-p 2048 -ub 2048 -fa 1 -r 4`, 两轮交替:

| 版本 | r1 | r2 | vs OLD |
|---|---|---|---|
| OLD (改动前) | 2550.95 | 2550.30 | — |
| NEW (寄存器驻留 C) | 2636.24 | 2632.57 | **+3.2%** |
| **BLK (C + blocked E)** | **2668.56** | **2672.12** | **+4.7%** |

**kernel 层同会话 A/B** (rocprofv3 kernel trace, 48 calls/forward):

| | OLD | NEW (仅寄存器) | BLK (最终) |
|---|---|---|---|
| us/chunk (同会话 trace, 两轮) | 23.709 / 23.924 | 14.159 / 14.090 | **10.314 / 10.349** |
| 加速 | 1.00x | 1.68x | **2.30x** |
| 探针预测 (gk2.hip) | 20.779 | 13.875 | 9.158 |
| 单 forward 内核总计 | 738 835 us | 711 316 us | — |

> **探针与真实 kernel 的保真度很好** (V1 预测 13.875 vs 实测 14.09; V3 预测 9.158 vs 实测 10.31)。
> 探针绝对值整体偏低是时钟差异, **变体间的相对关系是可信的**。


**同会话 kernel 分解对照** (最后一整个 forward, 前后各 48 次 GDN):

| 项 | OLD | NEW | 变化 |
|---|---|---|---|
| radiance (对照组, 未动) | 399 640 us | 400 435 us | +0.2% |
| OTHER (对照组) | 243 407 us | 244 723 us | +0.5% |
| **gdn_chunked** | **73 055 us** | **43 324 us** | **-40.7%** |
| 单 forward 总计 | 738 835 us | 711 316 us | -3.7% |

> **对照组漂移 <0.5%, 所以 GDN 的 -40.7% 是干净信号。**
> GDN 占单 forward 的比例: **9.9% -> 6.1%**。
> (对照: 038 节记的 9.4% 是另一次 trace 的口径, 差异来自那段时期 radiance 的占比波动。)

### 10.4 验收门 (全过)

| 门 | 结果 |
|---|---|
| `test-backend-ops -b ROCm0 -o GATED_DELTA_NET test` | **39/39** |
| PPL (`--chunks 8 -ub 2048`) | **6.9517 +/- 0.12466** (红线 6.9633, 在误差棒内) |
| pp512 | **2384** (基线 2210~2284) |
| pp2048 | **2688** (基线 2578) |
| tg128 | **29.91** (基线 29.90, 持平) |
| spec decode DFlash2 n_max=7 | 60.0 t/s (基线 69.6 / 86.2 / 49.1 区间内) |

> `test-backend-ops` 全量跑有一个**既有失败**, 与本次改动无关:
> `FLASH_ATTN_EXT hsk=192, kv=512, nb=75` 共 4 例, ERR 0.00084/0.00055 > 阈值 0.0005。
> 已用**改动前的旧库**复现同样的 4 例失败, 确认是 RDNA4 flash-attn 的既有精度问题。

### 10.5 寄存器代价 (必须记账)

| 版本 | VGPR | spill |
|---|---|---|
| 改动前 | 214 | 0 |
| 寄存器驻留 (C) | 256 (wave32 上限) | 0 |
| blocked (C+E) | **240** | 0 |

`float xr[32]` 是**每个 warp 都分配**的 (包括不执行该段的 7 个 warp)。blocked 版把 `xr[32]`
缩成 `xr[16]`, VGPR 回落到 240 —— 这也是它比纯寄存器版更快的一部分原因。

### 10.6 遗留: 还能再挖吗

探针 `gk2.hip` 在**真实 kernel 上下文里**测四个变体 (同进程, 时钟不受影响):

| 变体 | wall us/chunk | 求逆段 us/chunk |
|---|---|---|
| V0 改动前 | 20.779 | 12.901 |
| V1 寄存器驻留 | 13.875 | 5.406 |
| V2 双累加器 | 13.407 | 5.140 |
| **V3 blocked 16x16** | **9.158** | **0.816** |

> 探针 V1 = 13.875 与真实 kernel 实测 14.09 几乎重合; V3 = 9.158 也已复现 (实测 10.31 us/chunk)。
> **探针保真度已验证**, 变体间的相对关系可直接采信。

**求逆段已降到 0.816 us/chunk (占 8.9%), 不再是最大项。** 现在的分布是:
`gram+qk WMMA 2.244 / stage k-q-g-b 1.323 / rhs+s_inv+s_qw 1.324 / k.S0+q.S0 0.929 /
o=qw*D+store 0.932 / state upd 0.682 / build M 0.494`。

**下一个可能的杠杆**: 整段只有 512 条 WMMA (每 block 每 chunk), 而 8 个 warp 里
`gram+qk` 段只有 8 条 tile 工作 -> 也是"部分 warp 干活"。但这属于**再投入需要重新论证**的范畴,
不要在没有新证据时直接动手。

### 10.7 本次新增的陷阱

- **"串行段"要先怀疑访存, 再怀疑依赖链。** 本次两次假设都错了: 先猜"并行度不足"(列切分只有 -15.4%),
  再猜"依赖链长"(拆累加器反而更慢)。真相是 LDS 往返。**判据: 把该段单独抽出来, 逐项拆开测,
  用"位级是否等价"来区分"算法改写"和"访存改写" —— 位级等价的改动没有数值风险, 应该优先做。**
- **探针的绝对数字不能直接外推。** 同进程内横向比才可信 (gk2 四个变体同进程, 时钟一致)。
  跨进程/跨会话的数字会漂移 (本次 radiance 对照组在两次 trace 间漂了 2%)。
  **端到端 A/B 必须换 `.so` 交替跑** (`libggml-hip.so.0.24.0` 是独立共享库, 可直接替换)。
- **对拍要覆盖全矩阵。** blocked 求逆只写了下三角+对角块, 漏写右上块, 而探针只校验了下三角,
  没暴露。真实管线上 `s_inv` 会读整个 32x32。
- **`test-backend-ops` 全量跑有 4 例既有 FAIL** (`FLASH_ATTN_EXT hsk=192 ... nb=75`), 与本线无关,
  别误判成自己引入的。用旧库复现即可澄清。
- **`scp` 会保留本地文件的怪权限** (`----r-x---` 这种 owner 无读权限的), 导致远程后续
  `cp`/编译器读不了。传完 `chmod 644` 是必要动作。

### 10.8 本次产物

```
/vol1/1000/llama.cpp/gdnwork/
  gated_delta_net.cu.orig   改动前原稿 (md5 b07f5d989a409d418b615ec8674435f9)
  gated_delta_net.cu        改动后 (md5 b8022eb78865de8fd9e83904bbf7eaa7)
  gb2.hip                   分解探针: A/B/C/D/E 五变体 + 位级对拍
  gk.hip                    真实 kernel 分段计时 (clock64 打点)
  mk_gk2.py / gk2.hip       真实 kernel 内四变体同进程对比
  trace_old_r2.csv / trace_new_r2.csv   同会话 kernel trace A/B
```

`.49` 上的库备份与归档 (**已落到 HDD, 不依赖 /tmp**):

```
/media/seirin/HDD500G/backups/gdn-blocked-inverse-20261001/
  libggml-hip.OLD.so / .NEW.so / .BLK.so   三个版本的共享库, 可直接替换做 A/B
  gated_delta_net.cu.blocked               最终源码
  gb2.hip gk.hip gk2.hip mk_gk2.py         三个探针 + 生成脚本
  gb gb2 gk gk2                            已编译二进制
  traces/ab_reg_{OLD,NEW}.csv              寄存器版的同会话 A/B trace
  traces/ab_blk_{OLD,BLK}.csv              blocked 版的同会话 A/B trace
```

> A/B 复现方法: `cp libggml-hip.<V>.so build/bin/libggml-hip.so.0.24.0` 然后跑 llama-bench。
> **注意不要 chmod 成 644 目录** (btrfs 上目录缺 x 位会变成不可访问)。

