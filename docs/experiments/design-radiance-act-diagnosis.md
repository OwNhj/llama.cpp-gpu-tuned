# Radiance 激活路径诊断 (2026-10-01): 下一步候选的实测筛选

> 承接 `design-gdn-chunk-handoff.md` §10。GDN 优化完后 (kernel 23.8 -> 10.3 us/chunk,
> pp2048 +4.7%), 单 forward 的分布已经改变, 本文件是对**新头部目标**的实测筛选。
> 所有数字均为 2026-10-01 晚实测, 探针已归档。

---

## 0. 结论速览

| 候选 | 实测可省 | 占单 forward | 风险 | 判定 |
|---|---|---|---|---|
| `quantize_tokens_fp8` 去哈希 / 换哈希 | **0** (已贴 DRAM 墙) | 5.0% | - | **否决** (§1) |
| `swiglu_quant_fused` 不写 y | 17.5 ms | 2.4% | **高 (静默错值)** | **否决** (§2) |
| `concat_non_cont` 换 tile 转置 | 11.8 ms | 1.6% | 中 (通用算子, 单测 0 覆盖) | **待定** (§4) |
| **`rms_norm` 小 ncols 的线程浪费** | **9.8 ms** | **1.4%** | 低 (只改派发) | **已实施并验收** (§8) |
| radiance GEMM 突破 252 TFLOP/s | ? | 56.5% | 高 (已深度调优) | **不建议** (§8.2) |
| launch gap | 0 | — | - | **否决** (§8.2) |

**三个原本的头部候选里, 两个被实测否决** (§1 §2)。真正落地的是原先排在最后的 `rms_norm`
—— 实测 **pp2048 +1.08%**, 已提交 `1ebd1a36e`。

**当前状态**: 单 forward 728 ms(kernel 口径), 其中 GEMM 55% 已在 76.6% 峰值、
launch gap 仅 1.4%、其余头部项要么到 DRAM 墙要么有数值风险。**剩下的都是硬骨头。**


---

## 1. `quantize_tokens_fp8`: 已经贴着 DRAM 墙, 没有余量

### 1.1 生产实测

| 项 | 值 |
|---|---|
| 调用数 | **432 / forward** (864 / 2 fwd) |
| 单 forward 总耗时 | **38.3 ms** |
| 单次 | **88.7 us** |
| 单次流量 (K=5120, M=2048) | 读 41.9 MB + 写 10.5 MB = **52.4 MB** |
| 有效带宽 | **591 GB/s = DRAM 峰值 (637.7) 的 92.7%** |

**这个 kernel 没有可优化的空间 —— 它已经在搬它必须搬的数据, 且接近硬件极限。**

### 1.2 哈希跳过机制: 实测 29% 命中率, 但几乎不省时间

用 `GGML_RAD_QSTAT` 插桩实测 (临时插桩, 已回退):

```
[qstat] calls=128 K=5120 hit=41075  miss=98189  hit_rate=29.5%
[qstat] calls=256 K=5120 hit=74956  miss=185140 hit_rate=28.8%
[qstat] calls=384 K=5120 hit=103127 miss=251177 hit_rate=29.1%
```

命中率 **29%**。但反解 `80.6 = 0.71 * miss + 0.29 * hit`:

- 若 miss 就是 DRAM 下限 82.2 us, 则 **hit ≈ 76.6 us** —— 命中几乎和未命中一样贵。

**原因**: 哈希采样是 `xr[(i*37) % K]`, i 从 0 到 127。每个采样点相隔 37 个 float = 148 字节,
**每个采样几乎独占一条 128 字节 cache line**。128 个采样 = 读掉整行 20 KB 里的 **16 KB (80%)**。
所以"跳过"省下的那点工作量, 已经被扫描本身吃掉了。

### 1.3 冷态探针 (`qa.hip`, 192 MB L2 flush)

| 变体 | us/call |
|---|---|
| 真实 (128 点散列) | 115.3 |
| 完全不要散列 | 118.3 |
| 廉价散列 (1 条 cache line) | 125.8 |

**未命中时真实版反而比"完全不做散列"更快** —— 因为那 128 个散点采样顺带给后面的 amax
扫描**预热了 L2**。这不是巧合, 是这个设计实际上的作用。

> **教训**: "跳过重复计算"要先算清**检测本身的成本**。行级哈希在这种
> "检测成本 ≈ 被省工作量" 的场景下是净亏损。
> 真正的冗余是**张量级**的 (同层 `wqkv`/`wqkv_gate`/`beta`/`alpha` 共用同一激活),
> 而 `row_hash[row]` 是按行索引的**全局**表, 所有 K=5120 的 GEMM 共用一张, 会互相覆盖。
> 要吃到张量级冗余必须换机制 (按指针 + 每次 graph 执行失效, 即
> `ggml_rad_fused_acts` 已有的那套), 不是调参能解决的。

---

## 2. `swiglu_quant_fused`: 2.4% 可省, 但**不能做**

### 2.1 实测 (`qa.hip`)

| 变体 | 热态 us | 冷态 us | q 差异 |
|---|---|---|---|
| 真实 (写 y, 再从 L2 读回) | 809.4 | 895.0 | - |
| y 驻留寄存器, 仍写 y | 815.8 | - | 0/35651584 |
| **y 驻留寄存器, 不写 y** | **539.4** | **632.2** | **0/35651584** |

省下的 **274 us/call × 64 call = 17.5 ms/forward (2.4%)**, 全部来自**不写那 143 MB 的 y**。

### 2.2 为什么不做

`y` 是 GLU 的输出张量。radiance 路径下它**确实没人读** (下游 `ffn_down` 直接用融合产出的 q),
但**一旦 radiance 路径降级** (`M < GGML_RAD_PREFILL_MIN_M`, 类型/连续性不满足, 或
`GGML_RAD_DISABLE=1`), MMQ/MMVQ 会去读 `y` —— 读到的就是**上一次执行的残留值**。

这是**静默错值**, 是最坏的一类错误。而且:

- `ggml_rad_fused_acts` 每次 `graph_compute` 开头清空, 但 GLU 的 y **不会**被重新物化;
- decode 时每步输入都变, 残留值必然错;
- 当前没有任何测试能拦住它 (需要一个"radiance 中途降级"的用例)。

**要安全地吃到这 2.4%, 必须让 GLU 在决定不写 y 之前就能确认消费者会走 radiance** —— 
这需要把 consumer 信息传进 `ggml_cuda_try_swiglu_quant_fused` (改签名, 或扫 cgraph)。
属于**需要单独设计**的改动, 不在本次范围。

> **教训**: 融合路径"跳过中间张量"的前提是**降级路径永远不会被走到**。
> 只要存在降级可能, 就必须要么保证物化, 要么显式断言。别用"应该不会"来兜底。

---

## 3. 单 forward 的当前分布 (同会话 trace, `ab_blk_BLK.csv`)

| kernel 族 | ms/forward | 占比 | 次数 | 有效带宽 |
|---|---|---|---|---|
| radiance GEMM (atiled<4>) | 399.5 | 55% | 496 | - |
| OTHER (1500+ 个小 kernel) | 174.4 | 24% | 1530 | - |
| **swiglu_quant_fused** | 52.1 | 7.2% | 64 | 570 GB/s (89%) |
| **quantize_tokens_fp8** | 38.3 | 5.3% | 432 | 591 GB/s (93%) |
| **concat_non_cont** | 32.2 | 4.4% | 48 | 276 GB/s (43%) |
| gdn_chunked (已优化) | 31.5 | 4.4% | 48 | - |

> GDN 从优化前的 9.9% 降到 **4.4%**, 已不再是头部项。

---

## 4. `concat_non_cont`: 1.6% 可省, 且**归档的"已否决"结论需要修正**

### 4.1 真实形状 (探针实测, 不是推导)

```
src0 (conv_states) ne=[3, 10240]      nb=[4, 12]       通道慢变, 3 个时间步连续
src1 (qkv_mixed)   ne=[2048, 10240]   nb=[40960, 4]    通道连续 (真转置视图)
dst                ne=[2051, 10240]   nb=[4, 8204]     时间步连续, 通道慢变
```

`dst(k+t, c) = src1(t, c)` —— 这是一次 **84 MB 的真转置**, 不是行拷贝。
原版 kernel 用 `threadIdx.x` 走 i0, 于是**读按 40960 字节跳、写连续**。

### 4.2 实测 (`cp.hip`, 全形状 diff 0)

| body 行数 | 原版 | tile 转置 | 加速 | 原版带宽 |
|---|---|---|---|---|
| 2 (decode) | 22.8 us | 10.9 us | 2.09x | 29 GB/s |
| 32 | 17.6 us | 9.4 us | 1.87x | 177 GB/s |
| 512 | 262.5 us | 57.3 us | 4.58x | 162 GB/s |
| **2048 (prefill)** | **610.1 us** | **363.9 us** | **1.68x** | **276 GB/s (43%)** |

单 forward 收益: `610 -> 364 us` × 48 层 = **省 11.8 ms (1.6%)**。

### 4.3 与归档记录的矛盾 (必须说明)

`design-gemm-rdna4.md` §5 记的是"tile 方案正确但更慢, 1.0228 ms/call vs 原版 0.660 ms,
慢 55%", 因而关闭这条线。**本次实测与之相反 (1.68x 加速)。**

差异来源已定位——**归档那次记录的形状是 `src1 ne=[2,10240]`, 即 2 行 (decode)**。
但本次在 2 行下测出的也是 **2.09x 加速**, 仍然与归档相反。所以形状不是全部原因,
归档那次 tile 实现的自身缺陷无法从记录中复原。

**两条独立证据的分歧没有解决。** 归档那次是在**真实管线**上测的 (且有 PPL 数据),
本次是**合成探针**。在解释清楚之前, **不要把 1.6% 当成既得收益**。

### 4.4 已知坑 (沿用归档记录, 仍然有效)

- **单测 0 覆盖**: `test-backend-ops` 里**没有**任何用例满足"转置视图 src1"的条件。
  归档那次用 `cudaMemcpyAsync` 误判为行拷贝, **CONCAT 单测 177/177 全绿**, 而 PPL 爆到 424051。
- **`hipMemcpy2DAsync` 处理 3 行头部不是瓶颈** (`mem2d.hip`: 6.5 us vs 标量 kernel 5.3 us),
  所以当年"慢 55%"不能归因于它。
- 探针的 `ne1` 边界处理必须带 `j0 < nbody` 判断, 否则短形状会越界写 (本次踩到, 已修)。

---

## 5. 新发现: `rms_norm` 在 ncols=128 时一半线程空转

从 trace 的 grid 反推 (rocprof 的 `Grid_Size_*` 是 **work-item 数**, 要除以 workgroup):

| 实例 | 真实 grid | ncols | us/call | 次数/fwd | 合计 |
|---|---|---|---|---|---|
| `rms_norm_f32<256,true>` | (48, 2048) | 128 | 337 | 48 | 16.2 ms |
| `rms_norm_f32<256,false>` | (16, 2048) | 128 | 111 | 96 | 10.6 ms |
| `rms_norm_f32<1024,true>` | (2048, 1) | 5120 | 118 | 129 | 15.2 ms |

后两行对照很说明问题:

- `ncols=5120` 用 1024 线程: 读+写 84 MB / 118 us = **712 GB/s (高于 DRAM 峰值, 走 L2)** —— **已到顶**。
- `ncols=128` 用 **256 线程**: 只有 128 个线程有活干, **另外 128 个纯空转**。
  实测 337 us 处理 100 MB = 297 GB/s, **只有前者的 42%**。

**这是一个真实且同质的浪费**: 同一个 kernel 模板, 只因为 `ncols < block_size` 就浪费一半线程。
修法是把 block size 按 `ncols` 选 (128 而非 256), 但 `rms_norm` 是**通用算子**, 改动影响所有模型,
需要单独验证。**上限约 1.0-1.6%, 风险中等, 需要用户决断。**

---

## 6. 本次新增的陷阱

- **"跳过重复计算"必须先算检测成本。** 行级哈希的 128 个散点采样 ≈ 读掉整行 80% 的数据,
  检测成本 ≈ 被省工作量, 净收益接近 0 (§1.2)。
- **融合路径跳过中间张量 = 埋静默错值的雷。** 只要降级路径存在, 就不能省那次物化 (§2.2)。
- **热态探针会骗人。** 同一块输入连跑 30 次, 数据全在 cache 里 —— `no hash` 测出 39.8 us,
  **低于 DRAM 下限 82.2 us**, 这本身就说明测的不是 DRAM 流量。
  **判据: 任何低于理论上限的数字都是探针错误, 不是好消息。** 必须加 L2 flush 复测。
- **归档结论要连形状一起继承。** "tile 更慢" 那条记录的前提形状 (`ne1=2`, decode)
  与 prefill (`ne1=2048`) 不同, 直接沿用会误判 (§4.3)。
- **rocprofv3 的 `Grid_Size_*` 是 work-item 数, 不是 block 数。** 要除以 `Workgroup_Size_*`
  才是真实 grid, 否则会把 `rms_norm` 的 48 个 block 读成 12288 个 (§5)。
- `scp` 保留本地怪权限导致远程编译失败; 传完 `chmod 644`。

---

## 7. 产物

```
/vol1/1000/llama.cpp/gdnwork/
  qa.hip        quantize / swiglu 探针 (热态 + 冷态 L2 flush + 位级对拍)
  cp.hip        concat 转置探针 (可传 body 行数, 覆盖 decode~prefill)
  mem2d.hip     3 行头部的标量 kernel vs hipMemcpy2DAsync

/media/seirin/HDD500G/backups/gdn-blocked-inverse-20261001/
  qa.hip qa  cp.hip cp  mem2d.hip mem2d     源码 + 已编译二进制
  libggml-hip.{OLD,NEW,BLK}.so              三版共享库
  traces/ab_blk_{OLD,BLK}.csv               同会话 A/B trace
```

**仓库状态**: `3a46dce7e` (GDN 改动已提交, 未 push), 工作区干净。
本次诊断**未改动任何仓库文件** —— `radiance-gemm.cu` 的临时插桩已还原,
`build/bin/libggml-hip.so.0.24.0` 重建后与优化验证时的 `libggml-hip.BLK.so`
**md5 完全一致** (`d306e03e9859f48906135a93f9062277`), 证明插桩未留痕。

---

## 8. [补记] OTHER 桶完整分解与 rms_norm 优化 (已实施)

§5 的 `rms_norm` 猜想**已实测确认并落地**。本节记录完整分解与验收。

### 8.1 单 forward 全 kernel 分解 (pp2048, `ab_blk_BLK.csv`, 2276 dispatches)

| kernel | ms | 占比 | n | us/call |
|---|---|---|---|---|
| radiance GEMM atiled<4> | 396.74 | 56.5% | 400 | 991.9 |
| swiglu_quant_fused | 51.99 | 7.4% | 64 | 812.4 |
| quantize_tokens_fp8 | 34.80 | 5.0% | 432 | 80.6 |
| concat_non_cont | 32.35 | 4.6% | 48 | 673.9 |
| gdn_chunked (已优化) | 31.70 | 4.5% | 48 | 660.4 |
| k_bin_bcast add | 25.83 | 3.7% | 176 | 146.8 |
| flash_attn_ext_f16 | 21.81 | 3.1% | 16 | 1363.2 |
| **rms_norm_f32<256,true>** | **20.62** | **2.9%** | 80 | 257.8 |
| ssm_conv_long_token | 15.82 | 2.3% | 48 | 329.5 |
| **rms_norm_f32<1024,true>** | **15.22** | **2.2%** | 129 | 118.0 |
| unary_gated_op silu | 11.97 | 1.7% | 48 | 249.3 |
| **rms_norm_f32<256,false>** | **10.66** | **1.5%** | 96 | 111.0 |
| cpy_scalar | 7.98 | 1.1% | 64 | 124.6 |
| rope_multi | 7.41 | 1.1% | 32 | 231.4 |
| atiled<2> (小 shape) | 4.71 | 0.7% | 96 | 49.1 |
| 其余 11 个小 kernel | ~8.6 | 1.2% | — | — |

### 8.2 两个被排除的假设

**launch gap 不是问题。** 测了 wall time 与 kernel busy time 的差:

| forward | dispatches | span | busy | gaps |
|---|---|---|---|---|
| fwd5 | 2756 | 777.4 ms | 96.3% | 3.7% |
| fwd7 | 2276 | 711.7 ms | **98.6%** | **1.4%** |

gap 中位 **4.4 us**, p99 5.0 us, **无一个超过 20 us** —— 均匀且极小,
CUDA graph 工作正常。**这条线关闭。**

**GEMM 的 252 TFLOP/s 是一堵结构性的墙。** 按 ENGAGE dump 的 8 个真实 shape 逐个算:

| K | N | calls/fwd | ms(2fwd) | TFLOP/s |
|---|---|---|---|---|
| 5120 | 17408 | 128 | 368.40 | 253.7 |
| 6144/17408 | 5120 | 128 | 251.89 | 251.0 |
| 5120 | 10240 | 48 | 81.65 | 252.5 |
| 5120 | 6144 | 48 | 48.97 | 252.6 |
| 5120 | 12288 | 16 | 32.61 | 252.9 |
| 5120 | 1024 | 32 | 6.31 | 217.8 |
| 5120 | 48 | 96 | 9.26 | 20.9 |

**所有大 shape 死死卡在 251-254 TFLOP/s = fp8 WMMA 峰值 (329) 的 76.6%。**

- 算术强度 **1847 FLOP/byte**, 远高于 ridge point (516) -> **纯算力受限**, 不是访存。
- ISA 分析: 全 kernel **只有一个** WMMA 热循环 (`0x2D6FB8..0x2D758C`),
  **198 条指令含 64 条 WMMA, 密度 32.3% (= 3.09 指令/WMMA)**。
  非 WMMA 部分主要是 18 条 `global_load_b64`、16 条 `v_perm_b32`、9 组等待。
- 对照: MMQ MXFP4 是 **7.9 指令/WMMA**, GDN chunked 是 36.9。

**结论: radiance atiled 的发射效率已经很高 (3.09), 再压缩空间有限。**
要突破 76.6% 需要动 WMMA 的排布 (寄存器压力/双缓冲), 属于**上一轮已经深度调优过**的领域,
不建议在没有新思路时重开。

> 注: `RADIANCE_MXFP4_EPIFAST` (号称 prefill 值 7.7-8.0%) **只作用于 `folded` 变体**,
> 而生产走 `atiled`。实测 M=512/1024/2048 三档 A/B **全部无差异** (2291/2292, 2517/2520, 2664/2662),
> 证实了这一点。

### 8.3 rms_norm: 确认的线程浪费

**生产形状是插桩实测的** (临时加 `GGML_NORM_DBG` 打印, 已回退):

| 调用点 | ncols | nrows | block | 次数/fwd | 线程利用 |
|---|---|---|---|---|---|
| `plain` | **128** | 16 | **256** | 96 | **50%** |
| `mul` | **128** | 48 | **256** | 48 | **50%** |
| `mul` | 5120 | 2048 | 1024 | 129 | 100% |
| `mul` | 256 | 4 / 24 | 256 | 6 | 100% |

`for (col = tid; col < ncols; col += block_size)` 在 `ncols=128, block=256` 时,
**tid 128..255 的循环体一次都不执行**, 但仍要参与 `block_reduce` 和每个 `__syncthreads`。

**探针实测** (`rn.hip`, diff 全 0):

| 变体 | us | 加速 | 带宽 |
|---|---|---|---|
| B256 (现状, mult) | 440.8 | 1.00x | 228 GB/s |
| **B128 (mult)** | **273.3** | **1.61x** | 368 GB/s |
| B256 (现状, no mult) | 131.9 | 1.00x | 254 GB/s |
| **B128 (no mult)** | **86.9** | **1.52x** | 386 GB/s |

### 8.4 实施

`norm.cu` 的三处 RMS 派发加一档 `ncols <= 128` 分支:

```cpp
    if (ncols <= 128) {
        const dim3 block_dims(128, 1, 1);
        ... rms_norm_f32<128, ...>
    } else if (ncols < 1024) {
        const dim3 block_dims(256, 1, 1);
        ... rms_norm_f32<256, ...>
    } else {
        ... rms_norm_f32<1024, ...>
    }
```

**改动 +27/-3 行, 只加派发分支, kernel 主体一行未动** (`rms_norm_f32` 本身是模板,
`<128>` 实例化的 `block_reduce` 自动退化为纯 warp 归约)。

### 8.5 验收

**kernel 层同会话 A/B** (rocprofv3, 2 个 forward):

| kernel | OLD us | NEW us | 比值 |
|---|---|---|---|
| `rms_norm_f32<256,true>` | 41 309.8 | (拆成 <128> 20 912.5) | — |
| `rms_norm_f32<256,false>` | 21 407.2 | (拆成 <128> 13 582.3) | — |
| `rms_norm_f32<1024,true>` | 30 716.0 | 30 702.5 | 1.00x (未受影响) |
| **rms_norm 总计** | **93 433.0** | **73 908.9** | **省 9.8 ms/forward (1.4%)** |

**端到端同会话交替 A/B** (llama-bench `-p 2048 -r 8`, 三轮):

| 轮次 | NORM256 | NORM128 | 变化 |
|---|---|---|---|
| 1 | 2731.45 ± 170 | 2757.37 ± 176 | +0.95% |
| 2 | 2720.03 ± 173 | 2755.06 ± 177 | +1.29% |
| 3 | 2728.21 ± 173 | 2755.84 ± 176 | +1.01% |
| | | | **均值 +1.08%** |

三轮方向完全一致 (每轮 NORM128 都更高), 与 kernel 层 1.4% 吻合。

| 门 | 结果 |
|---|---|
| `test-backend-ops -b ROCm0 test` | **315 个 norm 用例全过**; 仅 2 例既有 `FLASH_ATTN_EXT hsk=192` 失败 (与本改动无关, 此前已用旧库复现) |
| PPL (`--chunks 8 -ub 2048`) | **6.9517 +/- 0.12466** —— 与改动前**逐位一致**, 8 个 chunk 中间值也全同 |
| pp512 | **2431.83** (基线 2210~2284) |
| pp2048 | **2755.8** (基线 2578) |
| tg128 | **29.88** (基线 29.90, 持平) |

> PPL 逐位一致是预期的: block 宽度只改变归约的线程数, `<128>` 下 `block_reduce`
> 退化为单个 warp 的 `__shfl_xor` 树, 加法顺序与 `<256>` 的两级树在数学上等价且
> 在 fp32 下逐位相同。

### 8.6 本次新增的陷阱

- **"端到端没差异"不等于"kernel 没优化"。** 第一轮 A/B (`-r 4`, 两轮) 显示 pp2048 差异
  被噪声淹没 (2700.9/2668.6 vs 2701.7/2696.2)。**加大到 `-r 8` 且跑三轮**才让 +1.08% 稳定显现
  (每轮方向一致)。**判据: 看"方向是否每轮一致", 而不是看单轮均值差。**
- **rocprofv3 的 `Grid_Size_*` 是 work-item 数, 要除以 `Workgroup_Size_*` 才是 block 数。**
  把 `rms_norm` 的 48 个 block 误读成 12288 会完全误判占用率。
- **`llvm-objdump` 的 cbranch 立即数是无符号 16 位回绕。** 直接当有符号数读会把
  所有后向分支丢掉 (本次一开始算出"0 个后向分支"), 必须 `imm & 0xFFFF` 再按 >32767 减 65536。
- **ISA 反汇编要先确认取的是哪个实例化。** `atiled<4,false,64,false,false>` 与生产用的
  `<4,false,64,false,true>` 指令构成差别很大 (前者有大量 epilogue 软件编码)。

---

## 9. 非 MXFP4 模型实测: GSQ IQ3_S 对照 (2026-10-01)

### 9.1 问题

"换成低 bit 混合量化模型, 还能这么快吗?"

### 9.2 实测 (同会话, 同 build, `-ub 2048 -fa 1 -r 3`)

| 模型 | 类型 | 大小 | pp512 | pp2048 | tg128 |
|---|---|---|---|---|---|
| `Qwen3.8-27B-Rad-MXFP4-f8out.gguf` | MXFP4 | 14.73 GiB | **2289.6** | **2692.3** | 29.85 |
| `Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf` | IQ3_S | 11.28 GiB | **763.1** | **834.9** | **33.14** |

`/media/seirin/SSD2T_1/gguf/GSQ-Qwen3.8/Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf` (12.1 GB)

**pp2048 = 834.9 vs 2692.3 -> 只有 31%。** pp512 同样只有 33%。

### 9.3 根因 (已用 `GGML_RAD_DEBUG=1` 证实)

**radiance ENGAGE 计数 = 0** —— IQ3_S 一次都没进 radiance 路径。

硬门在 `radiance-gemm.cu:2026`:

```cpp
return amd_wmma_available(cc) && GGML_CUDA_CC_IS_RDNA4(cc) && type == GGML_TYPE_MXFP4 && ...
```

`type == GGML_TYPE_MXFP4` 是不可绕过的。K-quant / I-quant 全部落到通用 MMQ (dp4a),
而这条路径正是 prefill 的 56.5%、跑在 252 TFLOP/s (fp8 WMMA 峰值 76.6%)。

### 9.4 参照: 同一模型关掉 radiance 的代价

`GGML_RAD_DISABLE=1` A/B (权重/参数/ubatch 全同, 唯一变量是路径):

| | pp512 | pp2048 |
|---|---|---|
| radiance ON (MXFP4) | 2205 | 2549 |
| radiance OFF (MXFP4, 走 MMQ **fp8 WMMA**) | 1518 | 1650 |
| 差 | **-31%** | **-35%** |

注意: 这仍是**对非 radiance 最有利**的对照 —— MXFP4 的 MMQ 分支还能吃到 fp8 WMMA
(`quantize_mmq_mxfp8_cuda`, e4m3 W8A8)。IQ3_S 走的是 **dp4a 整数点积**,
在 RDNA4 上通常比 fp8 WMMA 更慢, 所以真实差距可能比 -35% 更大
(**这一句是推断, 未单独实测**)。

### 9.5 哪些改进与类型无关 (换模型仍有效)

| 改动 | 门槛 | pp2048 价值 |
|---|---|---|
| GDN chunked 求逆重写 | 只卡**形状** (`S_v==128 && neq0==nek0==128 && n_tokens>=64 && RDNA4`), 不卡类型 | +4.7% |
| rms_norm block 宽度阶梯 | 纯 `ncols` 判断, 完全通用 | +1.1% |

两项合计约当前总时间的 **6%** —— 保住的是零头, 丢掉的是大头。

### 9.6 decode 反而更快 (预期内)

**tg128 = 33.14 vs 29.85 (+11%)。** decode 是 DRAM 带宽受限而非算力受限,
IQ3_S 的 11.28 GiB 比 MXFP4 的 14.73 GiB 少 23% 的权重字节, 走 MMVQ。
**但这是单个类型的结果, 不能推广到其他 K/I-quant** —— MMVQ 各类型的 kernel 质量不同,
必须逐模型实测。

### 9.7 一句话

**想要 prefill 速度就必须 MXFP4** (或 MXFP8/BF16, 见 README 的支持表)。
低 bit 混合量化能省显存、decode 可能更快, 但 prefill 会掉到 1/3。

---

## 10. K-quant 逐类型排查: 只有 Q2_K 有问题 (2026-10-01)

### 10.1 问题

"是所有 K-quant 都有这个问题吗?" -> **不是, 只有 Q2_K。**

### 10.2 证据一: 配置表 (`mmq-config-rdna4.cuh`, 27 个类型全扫)

| 类型 | 配置数 | 最大 J |
|---|---|---|
| **Q2_K** | **5** | **80** <- 唯一 <128 |
| Q3_K / Q4_K / Q5_K / Q6_K | 8 | 128 |
| IQ1_S/IQ2_S/IQ2_XS/IQ2_XXS/IQ3_XXS/IQ3_S/IQ4_NL/IQ4_XS | 8 | 128 |
| Q1_0/Q2_0/Q4_0/Q4_1/Q5_0/Q5_1/Q8_0 | 8 | 128 |
| MXFP4/MXFP6/MXFP8/NVFP4 | 8 | 128 |
| MXFP4_E4M3 | 10 | 256 |

K-quant 五个里四个完整, **只有 Q2_K 被砍掉 J=96/112/128** (5 条: 16/32/48/64/80)。

### 10.3 证据二: 运行时 (GSQ IQ3_S trace, 8 种实际在跑的类型)

| 类型 | Scratch_Size | VGPR |
|---|---|---|
| **Q2_K** | **2128** | **256 (打满)** |
| Q4_K | 0 | 240 |
| IQ2_S | 0 | 232 |
| IQ3_S / IQ2_XS / IQ4_XS | 0 | 224 |
| IQ3_XXS / IQ2_XXS | 0 | 216 |

只有 Q2_K 溢出寄存器。吞吐: Q4_K 166.9 / IQ2_S 139.4 / IQ4_XS 108.2 TFLOP/s,
**Q2_K 2.9** (差 23-58 倍)。

### 10.4 关键新发现: 连 J=80 自己都在溢出

trace 里 Q2_K 跑的就是它**最大的** J=80 配置 (nt=256, I=128), 而 Scratch=2128 / VGPR=256。
说明**上游给 Q2_K 在 RDNA4 上的整条配置链都失败了**, 不是"缺高 J 配置"那么简单。
Q2_K dequant 需 2-bit 数据 + 4-bit scale-min 双辅助量, 寄存器压力天生高于 Q3_K,
I=128 tile 在 256 VGPR 上限下塞不下。

**修正上一节的初判**: 照搬 Q3_K 补 J=96-128 只会溢出更狠。
可行方向: 给 Q2_K 配 I=64 (或 nt=128) 的高 J 条目, 用 I 维减半换 J 维加宽。

### 10.5 排查中的弯路 (勿重复)

- **mmq.cu.o 的 offload bundle 里只有 type 107 (Q4_0_SYM4) 的 mul_mat_q 符号** ——
  模板实例化分散在多个 TU, 不能靠单个 .o 判断"哪些类型编译进去了"。
  **运行时 trace (Scratch_Size/VGPR 列) 才是权威证据。**
- rocprofv3 的 `Scratch_Size` = local memory (寄存器溢出) 字节数, 0 = 无溢出。
  这是比 VGPR 计数更硬的溢出判据 (VGPR=256 也可能只是恰好用满)。
- Q2_K 异常与本次会话改动无关: OLD 库 (改动前) 同样症状, 且本会话 `git diff` 未碰 MMQ。
- J=80 条目来自上游 commit `6eddde06a` ("CUDA: refactor MMQ kernel configuration #24127"),
  非 fork 引入。

### 10.6 修复方向 (未实施, 待定)

给 `mmq-config-rdna4.cuh` 的 Q2_K 增加低 I 高 J 条目 (仿照其它类型 nt=128/I=64 的行),
预期把 Q2_K 从 2.9 TFLOP/s 拉到 >= IQ2_XS 的 68 -> 单 forward 2352 -> ~1375 ms,
GSQ 模型 pp2048 约 835 -> 1400 (+70%)。**对任何含 Q2_K 的模型通用。**

### 10.7 官方主线验证 (2026-10-01, llama.cpp-ROCm @ a3f241dd0)

问题: "官方主线有这个问题吗?" -> **有, 一模一样。**

- 配置表: 官方与 fork 的 Q2_K 条目**逐条相同** (8 条, 最大 J=80),
  整个文件的 Q-quant 部分仅差 fork 新增的 Q4_0_ROCMI4/Q4_0_SYM4 行,
  Q2_K-Q8_0 全部一致 (diff 证实)。
- 同 GSQ 模型、同 ROCm0、同 `-ub 2048 -fa 1` 实测:

| 指标 | 官方 a3f241dd0 | fork 929337426 |
|---|---|---|
| Q2_K ms (13 calls) | **1053.1** | **1020.1** |
| Q2_K scratch/vgpr | **2128 / 256** | **2128 / 256** |
| Q2_K TFLOP/s | **2.9** | **2.9** |
| IQ3_S / Q4_K TFLOP/s | 90.5 / 163.7 | 92.0 / 166.9 |
| total mmq | 2099.7 ms | 2049.7 ms |
| pp2048 | **771.04** | **834.9 (+8.3%)** |

结论: **Q2_K 的寄存器溢出是上游 #24127 (6eddde06a) 引入的上游缺陷**,
官方与 fork 完全一致; fork 对该模型的 +8.3% 全部来自与类型无关的
GDN 求逆 + rms_norm 改动。修复 Q2_K 配置对两边都适用。

### 10.8 Q2_K 修复已实施并验收 (2026-10-01, commit 4f7f0b5e3)

**改动**: `mmq-config-rdna4.cuh` Q2_K 段 +6/-1 行。
保留原 J=80/I=128 条目作 fallback, 新增 I=64 的高 J 档 (96/112/128, 各 fallback 两态):

```cpp
CASE(GGML_TYPE_Q2_K, 128, 2, 64, 96/112/128, Q2_K, MMQ_ITER_K, false, ...);
```

原理: Q2_K 每个superblock要两份辅助量 (Q3_K 只要一份), 128 深 tile 装不下 ->
寄存器溢出。I 减半后每线程累加器从 40 降到 32 (nt=256/J=80 时), 256 VGPR 装得下。
J 选择循环取 ntiles 最少者, M=2048 时自动选中 J=128。

**验收 (GSQ IQ3_S 模型, 同会话 A/B, 库 md5 c6e7c274...)**:

| 指标 | 修复前 (Q2OLD) | 修复后 (Q2FIX) |
|---|---|---|
| Q2_K kernel 时间 (13 calls/fwd) | 1020.1 ms | **130.2 ms (7.8x)** |
| Q2_K scratch / VGPR | 2128 / 256 | **0** / 256 |
| Q2_K TFLOP/s | 2.9 | **23.1** |
| Q2_K 占单 forward | 49.8% | **11.0%** |
| total MMQ | 2049.7 ms | 1183.4 ms |
| **pp2048** | 820.3 ± 23.9 | **1294.3 ± 57.9 (+58%)** |

| 门 | 结果 |
|---|---|
| `test-backend-ops -b ROCm0 -o MUL_MAT test` | **708/708 OK** (含 Q2_K 全尺寸) |
| 全量 test-backend-ops | 仅既有 2 例 FLASH_ATTN_EXT hsk=192 (与本改动无关) |
| PPL (GSQ 模型, --chunks 4) | **6.0175 ± 0.14727 新旧逐位一致** (tile 宽度不改 K 求和顺序) |
| MXFP4 回归检查 | pp2048 2681.8 vs 2692 ± 265, radiance 路径不受影响 |

**遗留**: Q2_K 现在 23.1 TFLOP/s, 仍低于同模型最慢的 IQ2_XS (68)。
I=64 tile 的工作量翻倍是原因之一。若还要抬, 方向是 nt=128/I=64/J=128 (每线程 64 累加器,
需实测寄存器), 或 Q2_K 专用 LDS 布局。当前收益已拿大头 (+58%), 遗留项待定。

产物归档: backups/gdn-blocked-inverse-20261001/{libggml-hip.Q2OLD,Q2FIX}.so, traces/q2fix.csv。

### 10.9 修复分发到全部三处 (2026-10-01)

| 目标 | 分支 | commit | 状态 |
|---|---|---|---|
| `HDD500G/llama.cpp-gpu-tuned` (fork) | `master` | `35f64e88a` | **已 push origin/master** (cherry-pick) |
| `HDD500G/llama.cpp-gpu-tuned` (fork) | `Radiance` | `4f7f0b5e3` | 本地领先 origin 2 commits (GDN+rms_norm+Q2K), **未 push** |
| `SSD2T_1/llama.cpp-ROCm` (官方主线 checkout) | `local-dev` | `63d37accc` | 本地 commit, patch 干净应用 |
| `SSD2T_1/llama.cpp-W8A8` | `review-improvements` | `d1bb51033` | 本地 commit, patch 干净应用, 其余未提交改动未触碰 |

- 三处的 Q2_K 配置块 (grep CASE 归一化后) **md5 完全一致** (104c9a11...)。
- 各处 md5 不同是因为 fork-Radiance 还带 MXFP8/MXFP4_E4M3 的 fork 特有条目,
  与 Q2_K 修复无关。
- **用户明确要求: 不向 ggml-org/llama.cpp 官方仓库 push/PR。** 官方 checkout
  (origin=ggml-org) 只有本地 commit, 无远端动作。fork 的 push 仅到 OwNhj/llama.cpp-gpu-tuned。
- 遗留: llama.cpp-ROCm / W8A8 两处的 Q2_K 修复**未重新编译验证** (只验了源码一致)。
  两处各自的 build 体系独立, 如需在那些环境跑, 记得先重建。

### 10.10 fork 三个分支全部交付 (2026-10-01)

| 分支 | 本地 = origin | commit | 内容 |
|---|---|---|---|
| `master` | `35f64e88a` = origin, SYNCED | `35f64e88a` | Q2_K 修复 (cherry-pick) |
| `Radiance` | `4f7f0b5e3` = origin, SYNCED | `4f7f0b5e3` | GDN + rms_norm + Q2_K + README, 全部已 push |
| `llama.cpp-Pascal` | `19ce7fcd7` = origin, SYNCED | `19ce7fcd7` | Q2_K 修复 (patch 应用) |

注: Pascal 分支也带着上游合并进来的 `mmq-config-rdna4.cuh` (mmq.cuh 两处引用),
其 Q2_K 原本同样是坏配置, 已一并修复。三个分支均验证: 本地 = origin/Radiance 推送后
fetch 回读一致, 且 `GGML_TYPE_Q2_K, 128, 2, 64, 128` 新配置在三个分支上各出现 2 次。
工作区回到 Radiance, 树干净, token 已销毁。
