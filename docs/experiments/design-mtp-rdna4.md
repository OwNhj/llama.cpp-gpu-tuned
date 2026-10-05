# Qwen3.8-27B RDNA4 MTP 投机解码 (TG 战线)

> **状态 (2026-10-01 最终): tg 28.25 -> 50.7 t/s (1.79x)。**
> 四步叠加: (1) `output.weight`/`token_embd.weight` -> MXFP8 (PPL 6.9318 -> 6.9633);
> (2) MTP 投机解码 (零代码改动); (3) **换 DFlash2 drafter**, 量化到 `f4body-f8head`;
> (4) `MMVQ_RDNA4_MAX_BATCH_SIZE = 4` 补丁 (让 ncols in [5,8] 走 WMMA)。
> greedy 输出逐字节一致; pp2048 无退化; 单测 47/47+13/13+39/39+177/177。
> 上游文档: `design-gdn-rdna4.md` (PP 战线) + `design-gemm-rdna4.md` (GEMM 专项)。
>
> **TG 战线当前最优 = DFlash2 drafter (2026-10-01, 见 §7): tg 28.25 -> 50.7 t/s (1.79x)。**
> 三件齐备 (模型在本地 / fork 有完整实现 / 转换器已支持), 已转换并调优到
> **`f4body-f8head` (1.05 GB, 4.52 BPW)**: body 用 MXFP4, selector 码本保 MXFP8 (用户提出)。
> dflash n_max=7 达 50.5-50.7 t/s, 反超 MTP 最优 (47.5 t/s)。
> 三处贡献: (1) `--spec-type draft-dflash --spec-draft-n-max 7`;
> (2) **`MMVQ_RDNA4_MAX_BATCH_SIZE = 4` 补丁** (§7.5c, 用户提出: 让 ncols in [5,8] 从
> dp4a 改走 WMMA, 对 M=1 的生产路径零副作用); (3) drafter 量化方案。
> 历史红线全部保持 (PPL 6.9633 逐位不变 / 单测 47/47+13/13+39/39+177/177)。
> **⚠ 测量方法学**: 投机解码 A/B 必须加 `--temp 0`, 否则默认采样 (temp 0.80)
> 每次生成不同文本, 方差 +-10%, 会得出完全错误的结论 (本会话踩了两次)。

---

## 1. 结论先行

**MTP (Multi-Token Prediction) 投机解码在这个 fork 里已经完整实现, 之前从未被启用过。**
模型自带 `nextn_predict_layers=1`, 第 65 层 (`blk.64`) 就是一个训练好的 MTP head,
但 `load_mtp` 默认 false, 该层权重**根本不加载** —— 纯浪费。

打开它不需要改一行代码:

```bash
./build/bin/llama-speculative-simple -m <gguf> -dev ROCm0 -ngl 99 -fa 1 -ub 2048 \
    -n 256 --spec-type draft-mtp -p "<prompt>"
```

---

## 2. 实测数据 (2026-10-01)

### 2.1 最终结果 (旧模型 vs 新模型, 各 3 次重复)

| 指标 | OLD (原 MXFP4) | NEW (out/embd -> MXFP8) | 变化 |
|---|---|---|---|
| **tg128 (无投机)** | 28.00 (27.60/28.22/28.19) | **29.85 (29.85/29.85/29.86)** | **+6.6%** |
| pp2048 | 2510.8 (2519/2507/2506) | 2502.1 (2501/2503/2503) | -0.3% (噪声内) |
| **TG + MTP (n_max=2)** | 41.37 (42.22/41.38/40.53) | **46.29 (48.69/45.33/44.83)** | **+11.9%** |
| PPL (8 chunks) | 6.9318 | 6.9633 | +0.45% |
| MTP 接受率 | 58.9 - 66.2% | 56.6 - 69.4% | 持平 |
| 模型大小 | 17448 MiB (5.35 BPW) | **15088 MiB (4.63 BPW)** | **-2.36 GB** |

**相对最初基线 (tg 28.25) 的累计提升: 48.04/28.25 = 1.70x。**

### 2.2 MTP 启用本身 (旧模型, n_max 扫描)

### 2.3 正确性验证 (greedy, `--temp 0`, n=64)
MTP 输出与 baseline **逐字节一致** (剔除加载进度条与末尾性能行后 diff 为空, 两侧均 1457 B)。
-> **投机解码路径没有引入任何精度损失。** 新旧两个模型都验过。

### 2.4 权威统计 (llama-speculative-simple, 旧模型 n_max=3, 生成 257 tokens)
```
n_draft   = 3         每轮 draft 3 个
n_drafted = 297
n_accept  = 158
accept    = 53.199%   <- 接受率
n_predict = 257
decoded 257 tokens in 6.387 seconds, speed: 40.236 t/s
```
-> **平均每轮产出 257/99 = 2.596 个 token**。

### 计时分解 (同一轮测试)
```
prompt eval time  = 4451.56 ms / 422 tokens (94.80 t/s)
sampling time     =  102.68 ms
unaccounted time  = 2195.44 ms / 32.5%     <- draft 侧开销在这里
```

---

## 3. MTP 层结构 (`blk.64`, block_count=65 的最后一层)

```
blk.64.attn_q.weight        [5120, 12288]  MXFP4
blk.64.attn_k.weight        [5120,  1024]  MXFP4
blk.64.attn_v.weight        [5120,  1024]  MXFP4
blk.64.attn_output.weight   [6144,  5120]  MXFP4
blk.64.ffn_gate.weight      [5120, 17408]  MXFP4
blk.64.ffn_up.weight        [5120, 17408]  MXFP4
blk.64.ffn_down.weight      [17408, 5120]  MXFP4
blk.64.nextn.eh_proj.weight [10240, 5120]  MXFP8  <- type 43, 输入 10240 = 2 x 5120
blk.64.nextn.enorm.weight   [5120]         F32
blk.64.nextn.hnorm.weight   [5120]         F32
blk.64.nextn.shared_head_norm.weight [5120] F32
blk.64.attn_norm / post_attention_norm / attn_q_norm / attn_k_norm   F32
```
约 424M 参数 (~225 MB MXFP4)。**所有大权重都是 MXFP4, 可直接复用 radiance 快速路径。**

`eh_proj` 是 **MXFP8** 而不是 MXFP4 —— 因为 `10240 -> 5120` 的融合投影对精度敏感。
好消息: MXFP8 在 RDNA4 上走 MMQ 的 W8A8 fp8 WMMA 路径
(`mmq.cu:442` 注释 "The MXFP8/MXFP6 W8A8 fp8 WMMA path is implemented for RDNA4 only"),
不是慢路径。

---

## 4. 工作原理

MTP 训练目标: 输入 `(h_i, x_{i+1})` 预测 `x_{i+2}`, 其中 `h_i` 是主干在位置 i 的 hidden。
`eh_proj` 的 `10240 = 2 x 5120` 正是 `concat(enorm(embd), hnorm(h))`。

代码路径 (全部已存在):
- `src/models/qwen35.cpp:486 graph_mtp` —— MTP 图 (enorm/hnorm -> concat -> eh_proj -> attn -> FFN -> shared_head)
- `common/speculative.cpp:1330 common_speculative_impl_draft_mtp` —— draft 驱动
- `src/llama-ext.h` —— staging API `llama_set_embeddings_nextn` / `llama_get_embeddings_nextn_ith`
- `src/llama-context.cpp:31` —— `LLAMA_CONTEXT_TYPE_MTP -> LLM_GRAPH_TYPE_DECODER_MTP`
- `common/speculative.cpp:2306` —— 自动检测: 若 `blk.{block_count-1}.nextn.eh_proj.weight` 存在则识别为 MTP

**`process()` 的关键技巧** (speculative.cpp:1530 附近): target decode 之后, 把 h 右移一位
喂给 MTP context —— `memcpy(batch.embd + 1*n_embd, h_tgt, row_bytes * (n_tokens-1))`,
第 0 行用上一轮存下的 `pending_h`。**一次 decode 处理 n_tokens 个位置**,
所以 MTP 能并行产出多个 draft, 每轮数量取决于上一轮被接受的长度。

`is_mem_shared=false` (gemma4 才共享 KV), `chain_heads=false` (单 head, 即 qwen35 模式)。

---

## 5. 已知的坑

- **`--draft-max` / `--draft-min` 已被移除**, 报错提示用新名。正确的是
  `--spec-draft-n-max` / `--spec-draft-n-min` / `--spec-draft-p-min`。
- `llama-bench` **不支持投机解码** (它是纯 decode, 不走 `common_speculative`),
  所以 MTP 的收益在 llama-bench 里**完全看不到**。必须用 `llama-speculative-simple`
  或 `llama-server` / `llama-cli` 测。
- `common_speculative_print_stats` 只在 `examples/speculative-simple` 和
  `tools/server/server-context.cpp:683` 被调用, **llama-cli 不打印统计**。
- 统计走 `SPC_TRC` = `LOG_TRC`, 默认级别看不到。

---

## 6. 优化方向 (按预期收益排序)

### 6.1 `output.weight` 是 BF16, 2.54 GB —— TG 侧最大单项 (已解决)
`rocprofv3 --kernel-trace`, MTP decode, **旧模型** (3262 ms / 160523 dispatches;
绝对速度被 profiler 拖慢, 占比有效):

| 占比 | 调用 | kernel | 说明 |
|---|---|---|---|
| 51.8% | 29677 | `mul_mat_vec_q<MXFP4,4>` | 主干 + MTP 层的 GEMV, 每次 0.057 ms |
| **23.3%** | 182 | `mul_mat_vec_f<bf16,1,256>` | **lm_head, M=1 (target 侧)** |
| **7.3%** | 59 | `mul_mat_f<bf16x2,32,4,8>` | **lm_head, M=3 (draft 侧)** |
| 2.1% | 32004 | `quantize_q8_1` | MMVQ 的激活量化 |
| 0.5% | 180 | `mul_mat_vec_q<type 43>` | MXFP8 = `eh_proj`, 量很小 |

lm_head 合计 **30.6%**。单次 4.18 ms ≈ 2.54 GB / 640 GB/s, 是**纯带宽操作**;
每轮 target 1 次 + draft 1 次 = ~8 ms。

**改动后** (新模型, 1746 ms / 44785 dispatches):
| 占比 | 调用 | kernel |
|---|---|---|
| 30.1% | 6539 | `mul_mat_vec_q<MXFP4,3>` |
| **24.2%** | 59+26 | **`mul_mat_vec_q<MXFP8>` (type 43) 取代全部 BF16** |
| 19.3% | 407 | `mul_mat_q<MXFP4,16>` |
| 5.6% | 8305 | `quantize_q8_1` |
| 4.3% | 407 | `mul_mat_q<MXFP4,48>` |

**`mul_mat_vec_f<bf16>` 与 `mul_mat_f<bf16x2>` 彻底消失**, 全部并入 type 43 的
MXFP8 GEMV。decode 时 M=1, 走 GEMV 而非 WMMA 是正确的 (WMMA 需要 M>=16)。
`token_embd.weight` 只走 `k_get_rows`, 从来不是瓶颈, 但一并量化省了 1.2 GB 显存。

> **用户决策 (2026-10-01)**: `output.weight` 与 `token_embd.weight` 都改到 **MXFP8**。
> 注: 早前文档里"配方刻意保留 embed/lm_head 为 BF16"是笔者的错误推断, 用户从未这样要求。
> 类型选择依据: `GGML_TYPE_F8`(44) 的注释明确写着 "KV-cache only", 不能用于权重;
> 权重该用 `GGML_TYPE_MXFP8`(43) = OCP MX block-scaled FP8 E4M3, 也正是 MTP 的
> `eh_proj` 已经在用的类型。
> 复现命令:
> ```bash
> llama-quantize --output-tensor-type mxfp8 --token-embedding-type mxfp8 \
>     --tensor-type 'nextn\.eh_proj=mxfp8' <bf16源> <输出> MXFP4 48
> ```
> `--dry-run` 已验证: 866 张量 -> 503 mxfp4 + 3 mxfp8 + 360 f32, **BF16 归零**,
> 总量 15088 MiB (4.63 BPW) vs 原 17448 MiB (5.35 BPW), 省 2.36 GB。

`bf16` 源模型在 `/media/seirin/HDD500G/gguf-rad/Qwen3.8-27B-bf16.gguf` (54.6 GB),
HDD 余量 247 GB。原 MXFP4 模型**保留不动**, 新模型是独立文件。

### 6.5 还剩多少空间? 用实测带宽上限算账 (2026-10-01)

**先测硬件天花板** (自写 HIP kernel, 4.3 GB 纯读流式):
```
read-only 4.3 GB in 6.72 ms -> 638.9 GB/s   (3 次完全一致)
```

**当前利用率**:

| 场景 | 实读字节 | 理论下限 (638.9 GB/s) | 实测 | 带宽利用率 |
|---|---|---|---|---|
| tg baseline | 13.20 GB/tok | 20.66 ms -> **48.4 t/s** | 29.85 t/s (33.50 ms) | **62%** |
| tg + MTP (接受 2.5) | 16.05 GB/轮 | 25.12 ms/轮 -> **99.5 t/s** | 46.29 t/s (54.01 ms/轮) | **47%** |

-> **纯带宽口径下 tg 还有 1.6x, MTP 还有 2.1x 的空间。**

**但空间不在"少读字节"上**。GPU 占用率实测 (稳态 decode 窗口):
| 场景 | kernel 数 | GPU 繁忙 | 窗口 | 空闲 |
|---|---|---|---|---|
| 无 MTP | 71,529 | 1332 ms | 3733 ms | **64.3%** |
| 有 MTP | 39,372 | 764 ms | 4647 ms | **83.6%** |

> **更正 (2026-10-01, 原结论两条都错)**: 本节曾据 "CUDA graph warmup complete" 与
> `graphs reused=97` 断言 "CUDA graph 已启用且生效"。实测推翻:
> 1. **`graphs reused` 与 CUDA graph 无关**。它是 llama.cpp 自身的 ggml 图复用计数
>    (`llama-context.cpp:1415` 的 `n_reused`, 由 `can_reuse(gparams)` 决定), 是独立机制。
>    证据: `GGML_CUDA_DISABLE_GRAPHS=1` 时该计数**逐位相同** (125/125, 86/86, 154/154)。
> 2. **CUDA graph 在目标负载上开着更慢**。spec decode 3 prompt 配对 (`--temp 0`, 接受数
>    完全相同 -> 纯时间差): ON 67.62/90.96/49.38 vs OFF 70.00/98.09/49.95,
>    **均值 69.32 vs 72.68 -> OFF 快 4.85%**。
> 3. **该惩罚与 radiance 无关**, 2x2 隔离 (`GGML_RAD_DISABLE` x graph ON/OFF) 两边同幅度:
>    radON pp2048 2518->2648 (+5.1%), radOFF 1622->1692 (+4.3%)。
> 4. **测量陷阱**: graph ON 首次进程 pp512 ~1430, 后续 ~2500 (capture 成本 ~153 ms/进程)。
>    用 `-p 512 -r 3` 会得到 `2153 ± 623` 这种病态误差棒 —— 那是 capture 成本被平均进去,
>    **不是稳态差异**。tg128 稳态其实 ON 略快 (29.89 vs 29.56), 说明差异只在短进程/投机路径。
>
> **推论**: 这 64% / 84% 的空闲**不能**用 "graph 已兜住 launch 开销" 解释。真实原因未定论
> (硬件 `gpu_busy_percent` 在稳态投机窗口读数为 92-100%, 与 kernel trace 的 68.9% 矛盾,
> 差异疑为 profiler 自身开销 —— 同一命令开 profiler 3.835s -> 4.450s, +16%)。

~~CUDA graph **已启用且生效** (产物里有 "CUDA graph warmup complete", bench 报 `graphs reused=97`),
所以这 64% / 84% 的空闲**不是 launch 开销**, 而是：~~ (前提已失效, 下列推理保留备查)
- **M=1 的 GEMV 太短, 无法掩盖 kernel 间依赖延迟** —— 每个 kernel 都要等前一个写回才能开跑,
  小 kernel 的启动/排空时间占比就高。这是 decode 阶段的本质特征, 不是 bug。
- **MTP 的空闲率更高 (83.6%)**, 因为 draft 是**纯串行**的: `while (n_drafting > 0)` 里每轮
  `llama_decode` 只推进一个位置, 无法并行; 且 draft 每步还要做 top-k 采样回读
  (`top_k_radix_select` + `common_sampler_sample` 在 host 侧), 产生 host-device 往返。

**结论: 剩下的空间要靠"合并/放大 kernel"而不是"减少字节"。** 可行方向 (按预期收益):
1. **让 MTP draft 并行化** —— 当前 n_max=2 时 draft 串行 2 次 decode。若能让 MTP 一次
   接受 (h_p, x_{p+1}) 并输出多个位置的 token (需要改 MTP head 用法), 可省一次完整前向。
   这是唯一能显著改变 TG 的方向, 但需要动 `common/speculative.cpp` 的 draft 循环。
2. **`top_k_radix_select` 换 greedy 路径** —— draft sampler 硬编码 `top_k=10`, 而 `p_min=0`
   时其实只取 `data[0]`。改 greedy 可省掉 radix select (实测 1.7ms/32tok, 不大但免费)。
3. **GEMV 融合** —— decode 时 4464 次 `mul_mat_vec_q<MXFP4,1,1>` (每 token 139 次, = 65 层 x2
   加 MTP), 占 37.3%。这些是 MoE-free 的 dense GEMV, 每次只读一个权重矩阵。**无法再省字节**,
   只能靠提高单 kernel 的并发度 (更多 CU 同时喂) —— 但这已经是 bandwidth-bound 的写法了。

### 6.2 n_max 调优 (已完成, 最优是 2 不是默认的 3)
`llama-speculative-simple`, n=256, 同一 prompt:

| n_max | 速度 | 接受率 | n_accept |
|---|---|---|---|
| 1 | 35.6 t/s | - | - |
| 2 | 42.60 t/s | 64.7% | 145 |
| 3 (默认) | 39.72 t/s | 46.9% | 152 |
| 4 | 38.37 t/s | 39.9% | 158 |
| 6 | 26.85 t/s | 24.7% | 153 |

**关键洞察**: `n_accept` 的绝对值几乎不随 n_max 变化 (145-158) —— 模型**每轮真正能多猜对的
token 数固定在 ~1.5 个**, 调大 n_max 只是增加被丢弃的 draft, 把接受率稀释掉。
n_max=6 时浪费太多, 速度跌破 baseline (26.85 < 28.25)。

> **重要更正 (2026-10-01, 4 次重复)**: 上面 n_max=2 优于 3 的差异**是噪声**。
> 严格重复后 (新模型, 同一 prompt):
> | n_max | 4 次实测 | 平均 |
> |---|---|---|
> | 2 | 49.12 / 45.18 / 44.14 / 46.33 | **46.19 t/s** |
> | 3 (默认) | 42.39 / 48.86 / 44.73 / 51.01 | **46.75 t/s** |
>
> 两者**统计上等价**。单次测量方差高达 ±10%, **任何 MTP 结论都必须 4 次以上重复**。
> 选 n_max=2 或 3 都可以; 保持默认 3 也完全没问题。

### 6.2b `p_min` 扫描 (已完成, 全部是负优化, 保持 0)
`--spec-draft-p-min` 提高低置信度 draft 的提前终止阈值:

| p_min | n_max=2 | 接受率 | | p_min | n_max=3 | 接受率 |
|---|---|---|---|---|---|---|
| **0.0** | **41.86 t/s** | 48.9% | | **0.0** | **47.39 t/s** | 51.0% |
| 0.3 | 37.41 | 61.8% | | 0.5 | 32.90 | 64.6% |
| 0.5 | 28.57 | 73.0% | | 0.9 | 36.76 | 94.1% |
| 0.7 | 33.15 | 82.8% | | | | |
| 0.9 | 30.02 | 94.3% | | | | |

**接受率被推到 94%, 速度反而掉 30%** —— 因为提前停止 draft 会让每轮产出的 token 数变少,
而"每轮固定开销"(主干前向) 无法摊薄。**`p_min=0` 是最优, 不要动。**

---

## 7. 复现命令

```bash
# 权威统计 (带接受率)
./build/bin/llama-speculative-simple -m <gguf> -dev ROCm0 -ngl 99 -fa 1 -ub 2048 \
    -n 256 --spec-type draft-mtp -p "<prompt>"

# 快速 A/B
for mode in base mtp; do
  E=""; [ $mode = mtp ] && E="--spec-type draft-mtp"
  ./build/bin/llama-cli -m <gguf> -dev ROCm0 -ngl 99 -fa 1 -ub 2048 -n 256 -st $E -p "<prompt>" \
    2>&1 | grep -oE "Generation: *[0-9.]+ t/s" | tail -1
done
```

---

## 7. 下一步: 换 DFlash2 drafter (三件齐备, 未开始)

用户提示两条, 查证后全部成立, 且指向同一个结论:

### 7.1 硬件: Infinity Cache 64 MB (推翻我此前的带宽口径)
`rocminfo` 实测 R9700: **L1 32 KB / L2 8 MB / L3 (Infinity Cache) 64 MB**, cacheline 256 B。
之前我用 4.3 GB 大缓冲测的 638.9 GB/s 是**真 DRAM 带宽**; 而 `test-backend-ops` 微基准
反复读同一个权重张量时, 47 MB 矩阵**装得进 64 MB IC**, 所以那里读到的更高带宽是 IC 红利。
**含义**: decode 阶段每 token 要流式读 ~14.5 GB 权重, 远超 64 MB IC, 所以 IC 帮不上 decode;
但**同一轮内被重复使用的数据**(如 MTP draft 的多步、verify 的多行) 能吃到 IC。
-> 做"一次 forward 产出多 token"的 drafter, 是唯一能把 IC 从"用不上"变成"用得上"的结构改动。

### 7.2 MTP 是串行, dFlash2 是一次 forward
`common/speculative.cpp:2503` 原文注释:
> `dflash/dspark decode the whole noise block in a single pass and sample every block position on the backend`

对比:
| | MTP (当前) | DFlash2 |
|---|---|---|
| draft 方式 | `while (n_drafting>0)` 每轮 `llama_decode` **只推进 1 个位置** | 输入 `[id_last, <mask> x (block-1)]`, **一次 forward** 产出整块 |
| 每轮草稿数 | n_max=2 (实测 2 和 3 等价) | `block_size - 1 = 7` |
| 运行环境 | 已在跑, 1.70x | 待接入 |

radiance 侧的对应配置 (来自 `radiance-test/DOCKERHUB.md`):
```
--speculative-config '{"method":"mtp","num_speculative_tokens":8,...}'
RADIANCE_DYNAMIC_DRAFT=1    动态草稿深度 (置信度门控), lossless
RADIANCE_DRAFT_TAU=0.28     MTP 下配合 FAST_DRAFT 的阈值
RADIANCE_FAST_DRAFT=1       draft head 降 2bit + 精确 rerank: +16.6% tokens/s
RADIANCE_DRAFT_RERANK=64    dflash 下用 64 (32 会损 5.3% 接受率)
```
-> 用户记忆的"一次 7-15 个草稿"与 `block_size=8`、`num_speculative_tokens:8` 吻合。

### 7.3 三件齐备, 路径已通
1. **模型在本地**: `/media/seirin/HDD500G/radiance-test/models/Qwen3.8-27B-DFlash2-FP8/`
   (2.0 GB safetensors + config.json)。关键配置:
   ```json
   "block_size": 8, "selector_top_k": 16, "mask_token_id": 248070,
   "target_layer_ids": [5,19,33,47,61], "num_hidden_layers": 5
   ```
   只有 **5 层**, 靠抓 target 的 5 个中间层特征 (n_embd_inp_enc = 5 x 5120 = 25600) 工作。
2. **fork 实现完整**: `common/speculative.cpp:910 common_speculative_impl_draft_dflash`
   (`is_dflash2` 分支在 1236 行读 selector lattice), `src/models/dflash.cpp` 完整,
   `LLM_ARCH_DFLASH` 已注册, `--spec-type draft-dflash` 已存在。
3. **转换器已支持**: `conversion/qwen.py:649` 有
   `@ModelBase.register("DFlashDraftModel", "DFlash2DraftModel")` 的 `DFlashModel`。
   **注意 gpu-tuned fork 的 `conversion/qwen.py` 与 SSD2T_1 的版本完全相同 (884 行)**,
   不需要移植任何代码。转换需要 `--target-model-dir` 指向目标模型的 tokenizer。

### 7.4 风险已逐条排查 (2026-10-01)

**(c) 中间层特征: 已解除, 不需要改 qwen35.cpp。**
fork 里有**通用**的中间层输入收集机制, 与模型无关:
- `src/llama-graph.cpp:1376`: 图构建时按 `cparams.embeddings_layer_inp[il]` 自动
  `ggml_set_output(t_layer_inp[il])`, **对所有模型生效**。
- API: `llama_set_embeddings_layer_inp(ctx, lid, true)` / `llama_get_embeddings_layer_inp(ctx, lid)`。
- **dflash 实现自己就会调用它**: `common/speculative.cpp:1047`
  `for k: llama_set_embeddings_layer_inp(ctx_tgt, target_layer_ids[k], true)`,
  读取在 1147 行。所以接 DFlash2 时 target 侧零改动。
- 容量: `llama-context.cpp:126` 已按 `hparams.n_layer() + 1` 预留, 我们 65 层没问题。
- 提取 5 层 x 5120 = 25600 宽的输入, 与 config 的 `target_layer_ids` 长度一致。

**(a) 转换**: 只需 `--target-model-dir` 提供 tokenizer; FP8 -> GGUF 的类型映射
由 `DFlashModel` + `Qwen3Model` 基类处理, 无需新代码。
**(b) selector lattice**: 走 `llama_get_embeddings_nextn(ctx_dft)`
(speculative.cpp:1236 `is_dflash2` 分支), 是 fork 已有契约。

### 7.5 实测结果 (2026-10-01): 转换成功并可运行, 但**比 MTP 慢**

**转换成功** (踩坑见 7.6):
```
dflash2.gguf: 81 tensors, 3.86 GB
general.architecture = dflash
dflash.block_count    = 5          (5 层轻量 drafter)
dflash.block_size     = 8          (一次 forward 产出整块)
dflash.selector_top_k = 16
dflash.target_layers  = [6, 20, 34, 48, 62]   (转换器已做 1-based 修正)
dflash.attention.sliding_window = 2048
```

**速度实测** (同一 prompt, n=256, `-c 131072 -np 1`):
| 配置 | 速度 | 接受率 | n_accept | unaccounted |
|---|---|---|---|---|
| **MTP n_max=2** | **46.3 t/s** | 58-65% | ~145 | 32.5% |
| dflash n_max=3 | 36.26 | 59.2% | 164 | **60.0%** |
| dflash n_max=7 | 27.26 | 26.7% | 169 | 50.1% |

**关键: n_accept 绝对值几乎不随 n_max 变 (164 -> 169)** —— 和 MTP 完全同一个规律:
模型**每轮真正能多猜对的 token 数固定在 ~1.6 个**。
调大 n_max 只让接受率从 59% 崩到 26.7% (分母变大), 速度反而下降。

**dflash 当前更慢的两个原因**:
1. `unaccounted` 高达 50-60% (MTP 只有 32.5%) —— draft 侧开销翻倍。
2. 每轮要抓 target 的 **5 个中间层特征** (`target_layers=[6,20,34,48,62]`),
   经 `batch_inject` 注入 draft 的 K/V。target 侧每 token 多写 100 KB,
   draft 侧每轮多一次 5 层特征 encode。MTP 只需一个 `h_nextn` (1x5120)。

**结论**: dflash 的结构优势 (一次 forward 出 8 个) 被它的特征提取开销吃掉了,
在当前 n_accept ~1.6 的现实下, **MTP 是更优选择**。
要让它赢, 必须先提高每轮接受长度 (那取决于模型, 不是 drafter 实现)。

### 7.5b 修正后的 dflash (2026-10-01) —— 含两次被证伪的结论

> **本节结论已被 §7.5e/7.5f 更正两次, 保留作为方法学教训。**
> ① 最初写"dflash 反超 MTP 53.57 t/s"——那是单次幸运值。
> ② 改为 3 次重复后写"dflash 40.26 仍慢于 MTP 46.18"——**仍然是错的**,
>    因为用的都是默认采样 (`--temp 0.80`), 每次生成不同文本, 方差 +-10%。
> ③ **最终正确结论 (加 `--temp 0` 确定性测量, 见 §7.5e)**:
>    drafter 用 `f4body-f8head` 时 **dflash n_max=7 = 50.5-50.7 t/s, 反超 MTP 的 47.5**。
> **教训: 投机解码的 A/B 必须 `--temp 0`, 否则数据不可用。**
> 教训与 §6.2 完全相同: **MTP/投机类测量的单次方差高达 +-10%, 必须 3 次以上取平均。**
> 下面保留 53.57 那次数据, 但结论以重复测量为准。

**3 次重复的最终数据**:
| 配置 | 3 次实测 t/s | 平均 |
|---|---|---|
| **MTP n_max=3** | 48.43 / 46.76 / 43.35 | **46.18** |
| MTP n_max=2 | 45.34 / 46.05 / 47.64 | 46.34 |
| dflash2 mxfp8 n_max=7 | 38.77 / 43.00 / 39.02 | **40.26** |
| dflash2 mxfp8 n_max=3 | 39.46 (单次) | - |

**单次数据 (仅作趋势参考, 不可作为结论)**:


修正两处后重测, 结论逆转:

| 配置 | 速度 | 接受率 | vs MTP(46.30) |
|---|---|---|---|
| **dflash2 mxfp8, n_max=7** | **53.57 t/s** | 35.1% | **+15.7%** |
| dflash2 mxfp8, n_max=3 | 48.84 | 55.9% | +5.5% |
| dflash2 mxfp4, n_max=7 | 31.66 | 21.2% | -31.6% |
| MTP n_max=2 | 46.30 | 58-65% | 1.00x |

**这直接证明了"接受率低 != 慢"**: mxfp8 n_max=7 的接受率只有 35.1%,
远低于 MTP 的 58-65%, 但速度反而快 15.7%。原因就是 7.4 节算的:
**每轮产出更多 (2.87 vs 2.29) 且每轮成本被压下来了**。
判断 drafter 优劣要看 `token/轮` 与 `ms/轮`, 不能只看接受率。

**两个修正**:
1. **drafter 必须用 mxfp8 而不是 bf16** (用户指出)。
   源权重是 `F8_E4M3` + `weight_scale_inv` (128x128 block),
   config 的 `quantization_config.quant_method = "fp8"`。
   而 `conversion/base.py:448` 的 fp8 分支**无条件 dequant_simple 成浮点**,
   再用 `--outtype` 写出 -> `--outtype bf16` 把 1.73 GB 的 F8 膨胀成 3.46 GB。
   **修正: `--outtype mxfp8`**, 产出 1.99 GB (49 个 MXFP8 权重 + 32 个 F32),
   走的正是 RDNA4 上 MMQ 的 W8A8 fp8 WMMA 路径。
2. **MXFP4 版本反而更慢 (31.66), 且接受率掉到 41.5%**。
   根因: `llama-quantize` 把 **`selector_predecessor/successor` 两个 127 MB 的
   离散码本表也量化成 4bit**, 破坏了 selector 排序。
   源 config 明确把它们列在 `modules_to_not_convert` 里:
   ```
   "modules_to_not_convert": ["candidate_selector.hidden_projection",
                              "candidate_selector.predecessor_codebook",
                              "candidate_selector.successor_codebook", ...]
   ```
   -> **4bit 不是不能做, 但必须先排除 selector 码本与 kernel_projection**。
   注意 `--tensor-type` 对这些张量**无效**: `llama_tensor_get_type()` 第一行
   `if (!tensor_allows_quantization(...)) return tensor->type;` 直接短路了。
   当前 mxfp4-sel 尝试未生效 (量化体积仍是 975.86 MiB), 待用别的方式排除。

### 7.5c `MMVQ_RDNA4_MAX_BATCH_SIZE` 补丁 (已合入, 用户提出)

用户指出: **`MMVQ_MAX_BATCH_SIZE = 8` 让 ncols<=8 永远走 dp4a, 而 dflash 一轮恰好 8 个位置,
所以它一次都吃不到 WMMA**; 而普通 decode (M=1) / MTP (M=1) 本来就到不了这个区间,
所以"锁类型"是多余的, 直接对 RDNA4 放开即可。

**改动** (`ggml/src/ggml-cuda/mmvq.cu` + `.cuh`, `tests/test-backend-ops.cpp`):
```cpp
// mmvq.cuh
#define MMVQ_RDNA4_MAX_BATCH_SIZE 4   // RDNA4 在 16 行就到 WMMA, MMVQ 只在窄 batch 赢
// should_use_mmvq(), RDNA4 分支 (放在 CDNA 分支之后):
if (GGML_CUDA_CC_IS_RDNA4(cc)) {
    return ne11 <= MMVQ_RDNA4_MAX_BATCH_SIZE;
}
// tests: mmq_activation_is_coarse() 的 n > 8 同步改成 n > MMVQ_RDNA4_MAX_BATCH_SIZE
```

**变量分离 (阈值 vs drafter 精度, n_max=7)**:
| drafter | 阈值 | t/s (单次) |
|---|---|---|
| bf16 | 8 | 27.26 |
| bf16 | 4 | 33.87 |
| mxfp8 | 8 | 26.93 |
| mxfp8 | 4 | 37.18 - 53.57 |

-> 两个变量**都有效且必需**: 阈值放宽让 ncols=8 走 WMMA, mxfp8 让 drafter 权重减半。
但即使两者都开, 3 次平均 (40.26) 仍低于 MTP (46.18)。

**测试容差的关键发现 (差点误判)**: 阈值改 4 后出现 5 个 MUL_MAT FAIL:
```
ERR = 0.000513 ~ 0.000755 > 0.000500   (mxfp4: n=5,7,8; mxfp8: n=5,7,8)
```
看起来像 bug, 实际**不是**。`tests/test-backend-ops.cpp:4979` 本来就有针对该情况的容差:
```cpp
// RDNA4 MMQ 把激活量化到 e4m3。均匀网格理论 5.8e-4, 实测最差 6.1e-4, 所以 2e-3 留 3x 余量。
if ((type_a == MXFP4 || MXFP6 || MXFP8 || MXFP4_E4M3) && RDNA4_NATIVE_FP8) return 2e-3;
```
但它的门控 `mmq_activation_is_coarse()` **硬编码 `n > 8`**, 没跟上生产阈值 -> n in [5,8]
走了 MMQ 却仍用 MMVQ 的 5e-4 容差 -> 误报。
实测误差 (5.1e-4 ~ 7.6e-4) 与注释里的 "measured worst case 6.1e-4" 完全吻合。
**同步门控后**: mxfp4 47/47, mxfp8 13/13 全通过, **PPL 6.2193 逐位不变**。

### 7.5d 补丁的最终验收 (2026-10-01)

| 项 | 结果 | 基线 | 判定 |
|---|---|---|---|
| MXFP4 MUL_MAT | 47/47 | 47/47 | OK |
| MXFP8 MUL_MAT | 13/13 | 13/13 | OK |
| GDN | 39/39 | 39/39 | OK |
| CONCAT | 177/177 | 177/177 | OK |
| **PPL 8-chunk** | **6.9633** | 6.9633 | **逐位不变** |
| pp512 / pp2048 | 2279.77 / 2569.17 | 2318.7 / 2577.7 | 噪声内 |
| tg128 | 29.89 | 29.85 | 噪声内 |

-> **补丁对生产路径零副作用**: M=1 的普通 decode 与 MTP draft 都不在 ne11 in [5,8] 区间,
只有 block drafter (dflash) 和并发 serving 才受影响。历史红线全部保持。

### 7.5e TG 战线最终排名 (确定性测量, `--temp 0`, 各 2 次)

> **测量方法学的关键修正**: 之前所有对比都用了 `llama-speculative-simple` 的默认采样
> (`--temp 0.80 --top-k 40`), **每次生成不同文本 -> 接受率不同 -> 方差高达 +-10%**,
> 导致我得出了两个错误结论 (先"dflash 反超 53.57", 后"dflash 更慢 40.26")。
> **必须加 `--temp 0`**: 两次重复的结果差异 < 0.5%, 数据才可用。

| 方案 | t/s | 接受率 | n_accept | 倍数 (vs 28.25) |
|---|---|---|---|---|
| **dflash `f4body-f8head` n_max=7** | **50.49 / 50.72** | 30.4% | 175 | **1.79x** |
| MTP n_max=5 | 47.46 / 47.46 | 41.9% | 174 | 1.68x |
| MTP n_max=3 | 46.34 / 46.20 | 49.2% | 153 | 1.64x |
| MTP n_max=2 | 45.48 / 45.48 | 58.0% | 138 | 1.61x |
| dflash `f4body-f8head` n_max=3 | 44.41 / 44.29 | 45.4% | 148 | 1.57x |
| 纯 MXFP8 lm_head (无投机) | 29.85 | - | - | 1.06x |

**核心洞察**: dflash n_max=7 与 MTP n_max=5 的 `n_accept` 恰好都是 **174-175**,
但 dflash 快 8% —— 差异完全来自**每轮成本**。
再次印证 7.5b 的结论: **判断 drafter 优劣要看每轮成本与每轮产出, 不能看接受率**。

### 7.5f drafter 量化方案对比 (`--temp 0`, 各 2 次, 用户提出 f4body+f8head)

| drafter 量化 | 大小 | t/s | 接受率 | n_accept |
|---|---|---|---|---|
| **F4 body + F8 selector** | **1.05 GB (4.52 BPW)** | **50.82** | 30.4% | 175 |
| 全 MXFP8 | 1.99 GB (8.25 BPW) | 49.35 / 49.60 | 30.2% | 174 |
| 全 MXFP4 | 1.03 GB (4.25 BPW) | 36.93 / 37.00 | 29.8% | 174 |

-> **用户方案胜出**: 比全 F8 快 2.9% 且小 0.94 GB。
**全 MXFP4 慢 27% 而接受率几乎不变 (29.8% vs 30.4%)** -> 慢的原因是
**MXFP4 的 GEMV 核本身更慢**, 不是精度损失。
(之前"MXFP4 接受率掉到 41.5%"是采样噪声造成的假象, 见下方命令行 bug。)

生成命令 (**注意 `--xxx` 必须在位置参数之前**):
```bash
llama-quantize --tensor-type 'selector_predecessor=mxfp8' \
               --tensor-type 'selector_successor=mxfp8' \
               --tensor-type 'selector_hidden=mxfp8' \
               <bf16源> <输出> MXFP4 48
```

### 7.5g 两个命令行 bug (都曾导致错误结论)

1. **`--tensor-type` 位置错了 -> 静默失效**。
   `tools/quantize/quantize.cpp:415` 的参数解析循环:
   ```cpp
   for (; arg_idx < argc && strncmp(argv[arg_idx], "--", 2) == 0; arg_idx++)
   ```
   **遇到第一个非 `--` 参数 (模型路径) 就停止**。我把 `--tensor-type` 放在位置参数之后,
   它根本没被解析, 且**不报错**。这导致我先误判"selector 无法排除"。
2. **`llama-quantize` 对已量化源需要 `--allow-requantize`**。
   从 `dflash2-mxfp8.gguf` 再量化会**静默产出 10.9 MB 的空壳文件** (exit=0!)。
   正确做法是**始终从 bf16 源出发**。

### 7.6 踩坑记录 (都只会在真跑时暴露)

**(a) 转换器会读入 KV 校准文件而失败**
`ValueError: Can not map tensor 'model.layers.0.self_attn.attn.k_scale'`
根因: `model-kvscales.safetensors` (1.3 KB, 15 个 q/k/v scale 标量) 是 KV 激活校准数据,
不是权重, 但转换器的 glob 把它一起吃进去了。
**解法**: 建临时目录只放 `config.json` + `model.safetensors` 再转换。
另: 权重是 FP8 block-wise (`F8_E4M3` + `weight_scale_inv` 128x128), 转换器能正确处理。

**(b) OOM 不是"32GB 放不下", 是 n_ctx 默认用了 262144**
`common.h:450 int32_t n_ctx = 0; // 0 == context the model was trained with`,
而模型 meta 是 `context_length = 262144`。实测 KV 分配:
| ctx | KV buffer | 每 token |
|---|---|---|
| 4096 | 256 MiB | 64 KiB |
| 65536 | 4096 MiB | 64 KiB |
| **262144** | **16384 MiB (16 GB)** | 64 KiB |
严格线性, 64 KiB/token。**这正是"稀疏注意力却吃 16 GB"的答案**:
| | 层数 | 是否占 KV | 与 ctx 关系 |
|---|---|---|---|
| GDN (linear attention) | 48 | 否, 用固定 RS buffer 149.62 MiB | **无关** |
| **full attention** | **16** | **是** | **线性增长** |

`config.json` 的 `layer_types = {linear_attention: 48, full_attention: 16}`,
`full_attention_interval = 4` (每 4 层插 1 个 full attention),
**且没有 sliding window** -> 这 16 层是真 full attention, 16 GB 无法避免。
16 x (2 x 4 kv_heads x 256 head_dim x 2 B) = 64 KiB/token, 与实测精确吻合。

显存账 (实测权重 13597.44 MiB + RS 149.62 MiB):
| ctx | target 总 | +drafter 3.58 GB | |
|---|---|---|---|
| 65536 | 17.92 GB | 21.50 GB | OK |
| 131072 | 21.92 GB | 25.50 GB | OK |
| 200000 | 26.13 GB | 29.71 GB | OK |
| 262144 | 29.92 GB | **33.50 GB** | **OOM** |
-> **同时跑 target + drafter 时 ctx 上限约 200K。**

**(c) `-np` (并行序列) 不影响 KV, 只影响 RS buffer**
实测 c=65536: `np=1` RS=149.62 MiB, `np=4` RS=598.50 MiB (恰好 4x)。
`common.h:455 n_parallel = 1` 本来就是默认值 1。
**所以 `-np` 不是 OOM 的原因** (RS 才 600 MB), 真正原因是 (b) 的 KV。

**(d) 别忘了 `--spec-draft-n-max`**
dflash 的 draft 上限是 `block_size - 1 = 7` (见 speculative.cpp:1001),
但**默认 `n_max=3`** 会把 8 个草稿的能力限制到 3。
要检验"一次 8 个"必须显式传 `--spec-draft-n-max 7`。

### 7.7 执行命令 (已跑通)
```bash
# 转换 (必须先建只含 config.json + model.safetensors 的干净目录)
python3 convert_hf_to_gguf.py <干净目录> \
    --target-model-dir /media/seirin/HDD500G/models/Qwen3.8-27B-bf16 \
    --outfile /media/seirin/HDD500G/gguf-rad/dflash2.gguf --outtype bf16

# 运行 (注意 -c 限制, 别用默认 262144)
llama-speculative-simple -m <目标gguf> -md dflash2.gguf \
    -dev ROCm0 -ngl 99 -ngld 99 -fa 1 -ub 2048 -c 131072 -np 1 \
    -n 256 --spec-type draft-dflash --spec-draft-n-max 7 -p "..."
```

### 7.8 原始结论 (未变)
```bash
# 1. 转换 drafter (需要目标模型的 tokenizer 目录)
python3 convert_hf_to_gguf.py \
    /media/seirin/HDD500G/radiance-test/models/Qwen3.8-27B-DFlash2-FP8 \
    --target-model-dir <Qwen3.8-27B原始HF目录> \
    --outfile /media/seirin/HDD500G/gguf-rad/dflash2.gguf

# 2. 接入 (speculative-simple 用 -md/--model-draft; 校验用 --spec-type draft-dflash)
llama-speculative-simple -m <目标gguf> -md dflash2.gguf \
    --spec-type draft-dflash -n 256 -p "..."
```
**预期**: draft 从"串行 1-2 次 forward"变成"一次 forward 出 7 个", 直接打击那 83.6% 空朇。
**验收标准** (与 MTP 同口径): greedy 输出逐字节一致 + `llama-speculative-simple` 的
接受率/速度对比 + PPL 不受影响 (drafter 不改变 target 输出分布)。

---

## 8. decode 随上下文衰减: radiance 与 llama.cpp 的严格对比 (2026-10-01, 用户提出)

用户观察: radiance 的 decode 稳定在 40-50 tg, 而 llama.cpp 随上下文线性下降。
下面把两边的**口径对齐**后逐一核对。结论: **衰减速率两边几乎完全一样**, 差距不在衰减,
而在每 token 的绝对成本。

### 8.1 先把口径对齐 (这是关键, 之前我把两种分母混在一行比)

radiance 的 35.7/37.2/38.9 是 **ms/update** (一次 forward 产 4.92 token);
llama.cpp llama-bench 的 33.48/39.82 是 **ms/token** (无投机, 一次 forward 产 1 token)。
两者分母不同, 不能直接并列。分开算:

| 口径 | radiance (fp8 KV, R4D) | llama.cpp (f16 KV, FA) |
|---|---|---|
| **每次 forward 的 KV 读取斜率** | 0.1000 ms/update per 1k | 0.0991 ms/token per 1k |
| 每次 forward 产出的 token 数 | 4.92 | 1.00 (无投机) / 3.13 (dflash n7) |
| **每产出 token 的衰减** | **0.0203 ms/token per 1k** | **0.0991 (无投机) / 0.0317 (dflash n7)** |

-> **同一 forward 的斜率两边一样 (0.100 vs 0.099)**; 换算到"每产出 token"后,
radiance 靠 4.92 tok/update 把衰减摊薄了 **1.57x** (相对 dflash n7)。
**这就是 radiance 看起来更平的原因: 它的每轮产出更多, 不是因为它的 attention 更省。**

radiance 数据源: `PERFORMANCE.md` "bench_decode_ctx 35.7/37.2/38.9 ms/step at 0/8k/32k"。
同一套 harness (`radiance-test/paroquant/bench_decode_ctx.py`) 的 docstring 明确写着它存在的理由:
"decode cost grows with context -- the full-attention layers re-read the whole KV cache on every
forward ... On prod FP8 decode falls **217 -> 57.8 tok/s from 6.6k to 97k**"。
(217->57.8 是 prod FP8 口径, 不是 radiance 自身, 但同样说明**这个模型族的 decode 必然随上下文下降**。)

用户看到的 "radiance 稳定 40-50" 应该是**高上下文段的绝对水平** (radiance 97k 还有 57.8 t/s),
而不是衰减率。

### 8.2 带宽账: llama.cpp 已经贴在硬件的墙上, radiance 没有

模型结构: 16 层 full-attention x 4 kv_heads x 256 head_dim x 2 (K,V)。
DRAM 实测 638.9 GB/s。

| | KV 字节/token | 纯带宽下界 | 实测斜率 | 实测/下界 |
|---|---|---|---|---|
| llama.cpp f16 | 64 KiB | 0.1026 ms/1k | 0.0991 | **0.97x (已在下界上)** |
| radiance fp8 | 32 KiB | 0.0513 ms/1k | 0.1000 | **1.95x (差一倍)** |

**这是整件事的核心**: radiance 的 KV 只有一半字节, 理论上衰减应该只有一半,
但它的实测斜率是我们的 2 倍。

原因在 radiance 自己的 decode 内核源码里 (本机缓存, 一手材料):
`/media/seirin/HDD2T/userdata/.cache/radiance-libr4d/b9e42ab-rx9/r4d_attn_decode_h256_gqa6.hip`:

```
//   PF16  split-KV partials as f16 instead of f32. The partial buffer is written once and read
//         back once, and at 16K context that round trip is ~14 MB against ~17 MB of KV -- i.e.
//         nearly half the leg's DRAM traffic.
//   KLDS0 reads K straight from the paged cache, including when that cache is fp8. The widening
//         moves from once-per-workgroup to once-per-warp
```

也就是说 radiance 的 decode attention 是 **split-KV** 的: 32k 上下文时它除了读 64 MiB KV,
还要**写+读一份 partial 缓冲**。这就是它只有 32 KiB/token 却跑出 0.1000 ms/1k 的原因。
再加上它的 KV 是 **paged, block size 16** (`radiance_r4d_attn.py`) 的 gather 访问。

llama.cpp 的 KV 是连续布局, 完美合并, 所以能跑到带宽的 ~100%。

-> **结论: 衰减率我们不可能更好了, 除非减少 KV 字节数。** f16 已经是物理最优。
radiance 的 fp8 KV 并没有换来更平的曲线, 只换来了 2 倍容量。

### 8.3 为什么 llama.cpp 一量化 KV 反而更慢 (用户直觉正确, 定位到代码)

实测 (llama-bench, `-p 0 -n 128 -r 3`):

| KV 类型 | tg128 | @d32768 | @d65536 |
|---|---|---|---|
| f16 | 29.87 | 27.48 | **25.11** |
| q8_0 | 29.58 | 27.02 | 24.07 |
| f8 | 29.57 | 26.52 | **23.53** |

f8 的 KV 只有一半字节, 却慢 6%。rocprofv3 抓到了原因 (**同样的 656 次 launch, 32k 上下文**):

| KV | 实际跑的内核 | 时间 | 每次 |
|---|---|---|---|
| f16 | `flash_attn_tile<256,256,1,2>` | 138 ms | 0.210 ms |
| f8 | `flash_attn_ext_vec<256,1,f8,f8>` | **182 ms** | **0.277 ms** |

反算带宽:
- f16 TILE: 0.210 ms x 638.9 GB/s = 134 MB, 而它需要读 128 MiB -> **贴着带宽上界**。
- f8 VEC: 0.277 ms x 638.9 GB/s = 177 MB, 而它只需要读 64 MiB -> **只跑到 ~36% 带宽**。

代码位置 (`ggml/src/ggml-cuda/fattn.cu:710-721`): 量化 KV 在 `Q->ne[1] <= 2` 时被送进 **VEC**
内核。VEC 确实是原生读 (不 dequant, `fattn-vec.cuh:543` `need_f16_K = type_K == F16` -> false),
**但它是个弱内核** (一 query 行, 并行度/ILP 都差), 在 D=256 长上下文下只发挥 36% 带宽。

而一旦 batch 变大或条件不满足, 落到 **TILE / MMA_F16** 分支时,
`fattn.cu:764-781` 会强制 `need_f16_K = need_f16_V = true` ->
`f16_extra` scratch (`fattn-common.cuh:54-88`) 把整个 KV 反量化成 f16:
**流量从 读64MiB 变成 读32 + 写64 + 读64 = 160 MiB (2.5x)**。

所以**用户的判断是对的**: 量化 KV 今天在 llama.cpp 里纯粹是负收益, 因为
1. M<=2 走弱 VEC 内核 (36% 带宽);
2. 其他情况走 TILE/MMA 但要先 dequant 到 f16 (2.5x 流量);
3. 原生 fp8 WMMA 内核只编了 D=96/128 (`fattn-mma-f8.cuh:20` "Only instantiated for
   DKQ == DV in {96,128}"), **本模型 head_dim=256 用不上**。

### 8.4 那差距到底在哪: 不在衰减, 在每 token 的绝对成本

radiance 自身口径 137.7 t/s, 35.7 ms/update -> 4.92 tok/update (137.7 x 0.0357), 即 **7.26 ms/token**。
llama.cpp dflash n7 是 **19.72 ms/token** (50.7 t/s)。
-> radiance 在 0 上下文就快 **2.7x**。

(注: 32k 处的 ms/token 需要 radiance 的 tok/update 在深度上不变, 我没有这个数据,
所以这里只对比 **0 上下文的绝对水平** 和 **共同的衰减斜率** 两件有证据的事。)

衰减率两边相同 (+9.0% vs +8.7%), 所以**倍数差全部来自绝对成本, 不是被衰减拉开的**。

radiance 自己的文档 (`radiance_r4d_attn.py` 顶部注释) 承认:
"attention is only ~7% of a speculative decode step, which is dominated by the MoE GEMMs"。
-> 它的 decode 是 **GEMM 主导**, 不是 attention 主导。**差距主要在 GEMM, 不在 attention。**

把 2.7x 拆成两个因子 (radiance 137.7 t/s @ 35.7 ms/update -> 4.92 tok/update;
llama.cpp dflash n7 50.7 t/s -> 3.13 tok/update):

| | radiance | llama.cpp | 倍数 |
|---|---|---|---|
| tok/update (drafter 质量) | 4.92 | 3.13 | 1.57x |
| ms/update (每轮成本) | 35.7 | 61.7 | 1.73x |
| **相乘** | | | **2.7x** |

**每轮成本 1.73x 才是主项**。已排除的假设 (实测):
- **不是 attention**: ctx=0 时注意力开销可忽略, 差距照样是 2.7x。
- **不是 M=8 走 dp4a**: 打完 `MMVQ_RDNA4_MAX_BATCH_SIZE=4` 补丁后, 实测 dflash n7 的
  decode GEMM 里 `mul_mat_q<MXFP4,16>` (WMMA) 占 **60.3%**, `mul_mat_vec_q` 只占 8.4%。
  M=8 已经在 tensor core 上了。
- **残差**: 同一条 WMMA 路径, radiance 每轮 35.7ms vs 我们 61.7ms。这是**GEMM 的
  tiling/launch/epilogue 效率差**, 与 attention、与 KV 都无关。

> **口径 caveat**: 上表的 tok/update 不完全同源。radiance 的 4.92 来自 BetterBench 的
> 加权混合 (chat/code/json/... 8 类), llama.cpp 的 3.13 来自我单条技术散文 prompt。
> 两者内容不同, 接受率不可直接比。**可信的是 ms/update 的 1.73x** (两边都是 M=8 同一架构);
> tok/update 的 1.57x 只是量级参考。

### 8.5 结论 (只保留有证据的部分)

1. **衰减率不是问题**: llama.cpp 0.0991 ms/1k 已经贴着 DRAM 下界 (0.97x)。radiance 的
   fp8 KV 斜率反而更差 (0.1000), 因为 split-KV partial 的往返 + paged gather。
   **不要再在 f16 路径上找衰减的优化空间。**
2. **绝对成本是主战场**: 差距 2.7x 在 0 上下文就存在, 而 attention 只占 radiance decode 的 ~7%。
   所以下一步应该看 **GEMM**, 不是 attention。
3. **量化 KV 需要内核才成立**: 能把 KV 减半 (f8) 但没有能用的内核 ->
   M<=2 落到弱 VEC 内核 (36% 带宽), 其余情况 TILE/MMA 要先 dequant 到 f16 (2.5x 流量),
   原生 fp8 WMMA 只编了 D=96/128 而本模型 head_dim=256。
   **补 D=256 的 fp8 FA 内核 = 独立的一条路**, 与衰减率无关, 是拿容量的路。

### 8.6 测量方法学备注 (避免以后误读)

- `rocprofv3` 的 CSV 时间列单位是 **ns**, 不是 us/ms (`/tmp/kan.py` 初版多除了 1000)。
- 用 `llama-bench -d N` 做 kernel trace 时, **填充深度的 prefill 也会被 trace 到**
  (1024 次 `flash_attn_ext_f16<256,256,32,2>` 就是 prefill, 不是 decode)。
  分析 decode 必须按 launch 次数和 M 把 prefill 剔掉。
- `llama-speculative-simple` 的深度 prompt 必须 `-b` 足够大, 否则
  "the prompt exceeds the batch size" 直接退出, 而我用 `-b 16384` 又会 OOM。
  深度扫描建议用 `-b 4096` 配 <=16k token 的 prompt。
- `-ub` 不能随手改。`ssm_alpha/beta` (N_out=48) 在 **M >= 256** 时被 radiance fast path 接管
  (`GGML_RAD_PREFILL_MIN_M` 默认 256), 只有 **M < 256** 才落到 MMQ/MMVQ。所以
  `-ub 512` 和 `-ub 2048` 的 PPL **测的完全是另一条路径**, 拿去验证 MMQ/MMVQ 改动等于测空气。
  要用 PPL 验证这两个路径, 必须 `GGML_RAD_PREFILL_MIN_M=4096` 把门限抬上去。

---

## 9. 欠并发 GEMM 形状: 从并发模型到两条实测路径 (2026-10-01, 用户提出)

### 9.1 问题的准确表述 (用户: "并发不够导致 l3 延迟无法掩盖")

decode 阶段权重流式读取量是固定的 (14.5 GB/token), L3 = 64 MB 装不下, 所以每个权重字节
必然来自 DRAM 一次。要跑满 638.9 GB/s, 需要足够多的**并发访存请求**去掩盖 DRAM 延迟。
问题出在**小 N_out 的形状发不出足够多的 block**。

先量化"并发"到底怎么算。对 `mul_mat_q`:

```
I         = 64        (I = nwarps * rows_per_warp = 4 * 16)
nwarps    = 4         (MXFP4, J=16, nthreads=128)
nty       = ceil(N_out / I)
总 warp 数 = nty * nwarps ~ N_out / 16
```

**关键不变量: 总 warp 数只取决于 N_out, 与 I 无关。** 所以"把 I 切小来增加 block 数"是
无效杠杆 (实测 I=32 中性到负 3-8%), 唯一能真正加并发的轴是 **K**。

机器容量: 128 SIMD x 3 waves = **384 waves**。真实 M=8 verify 各权重类的并发占用:

| 权重类 | N_out | nty | warps | 占机器 | M=8 实测 %DRAM |
|---|---|---|---|---|---|
| **ssm_alpha/beta** | **48** | **1** | **4** | **1%** | **~1%** |
| attn_k/v | 1024 | 16 | 64 | 17% | 20% |
| ffn_down/ssm_out/attn_out | 5120 | 80 | 320 | 83% | 64-76% |
| attn_gate | 6144 | 96 | 384 | 100% | 75% |
| attn_qkv | 10240 | 160 | 640 | 167% | 85% |
| attn_q | 12288 | 192 | 768 | 200% | 73% |
| ffn_gate/up | 17408 | 272 | 1088 | 283% | 88% |
| lm_head (MXFP8) | 248320 | 3880 | - | - | 97% |

`ssm_alpha/beta` 只有 **1 个 block / 4 warp = 1% 机器**, 是最极端的形态。

### 9.2 `ssm_alpha/beta` 的成本: 20 层串行 DRAM 延迟链

profile 直测 (dflash n7, MMA dtype=M=8):

```
nty=1 kernel 自身总时间 : 146.3 ms   (占 8.1% forward)
kernel 前 launch gap    :  28.3 ms   -> 瓶颈在 kernel 内部, 不是 launch
平均每次 kernel          :  21.47 us  (共 6816 次 = 72 轮 x 94.67 次/轮)
```

**为什么是 21.47 us?** 权重总量只有 `48 x 5120 x 4.25/8 = 127.5 KB`, 按 DRAM 638.9 GB/s
理论 0.20 us。实测有效带宽 **6.1 GB/s = DRAM 的 1%**。

原因是 K 循环结构:

```
load_tiles(x, ...)   <- 读权重 tile
__syncthreads()      <- 等全部读完
vec_dot(...)         <- 用
__syncthreads()
```

**每次 K 迭代都同步等一次 DRAM**。`K=5120 / ITER_K=256 = 20` 次迭代:

```
20 x ~1.07 us = 21.4 us   与实测 21.47 us 吻合
```

即 **20 层串行 DRAM 往返**。一个 block 装得下全部权重 (127.5 KB), 缺的不是容量, 是
**同时发出的独立访存数量**。

### 9.3 路径 A: 修好 K 切分 (机制打通, 但收益不成立)

前面两次尝试 (上游 stream-K / 自写 static split-K) 都以内存越界告终
(`HSA_STATUS_ERROR_MEMORY_FAULT`, 正确性 27/44 和 70/86)。**根因找到了**:

```c
// mmq.cuh:1148  设备端
if constexpr (!ggml_cuda_mmq_get_stream_k(type, J, fallback)) {
```

这是 **编译期** 分支, 由 config 表 (`mmq-config-rdna4.cuh`) 的 `stream_k` 标志决定。
之前我在 host 端改 grid 为 `ntiles * split_k`, 但 config 里 `stream_k=false` 让设备端
仍按 tiling 解释 `blockIdx.x` -> 两侧解释不一致 -> 越界。

**正解: 改 config 的 `stream_k` 标志, 不是改 host grid。** 改完正确性 **86/86 通过**。
但性能在真实形状上是**净负**:

| nty | 形状 | base -> stream_k | 结论 |
|---|---|---|---|
| 1 | N=48 | 0.97x | 中性 |
| **16** | **N=1024, K=14336** | **1.46x** | 收益 |
| 32 | N=2048 | 0.92x | 损失 |
| 64+ | N>=4096 | 0.40-0.58x -> 调优后 0.97-0.99x | 基本持平 |

`block_nums_stream_k` 上游无条件用 `nsm=64`, 当 `ntiles >= 64` 时 64 个 block 去做
work-stealing 分片, 丢掉了干净的 tiling 结构。改为 **tile 够多时不切 K**:

```c
const int stream_k_blocks = ntiles_dst < nsm ? nsm : ntiles_dst;
```

大形状恢复到 ~0.97-0.99, 只有 `N=1024,K=14336` 稳定拿到 1.46x。**真实模型形状上净负
(普遍 0.97-0.99), 已回滚。** 唯一保留的价值是: 上游 stream-K 在 RDNA4 上从未启用过,
这次证明了"改 config 标志"才是正确入口, 以后要用可以直接用。

### 9.4 路径 B: 窄权重改走 MMVQ (已合入, 默认开启)

`ssm_alpha/beta` 只有 48 行, **装不满一个 MMQ tile**。但 MMVQ 的 grid 是:

```
grid = (ceil(N_out / rpb), nchannels, ntok),  rpb = calc_rows_per_block(ncols_dst)
ncols_dst=8 -> rpb=2 -> N=48 发出 24 个 block
```

比 MMQ 的 1 个 block 多 **24 倍** 并行度, 且跨 token 也并行。

**改动** (`mmvq.cu`, `ggml_cuda_should_use_mmvq` 增加 `nrows_x` 参数):

```c
if (GGML_CUDA_CC_IS_RDNA4(cc)) {
    static const int small_n = [] {
        const char * e = getenv("GGML_MMVQ_SMALL_N");
        return e ? atoi(e) : 64;
    }();
    if (nrows_x < small_n) {
        return ne11 <= MMVQ_MAX_BATCH_SIZE;
    }
    return ne11 <= MMVQ_RDNA4_MAX_BATCH_SIZE;
}
```

**没有动 `MMVQ_RDNA4_MAX_BATCH_SIZE` (仍是 4)** —— 用户明令禁止改那个阈值。这里是新增
一条**按形状**的路由条件, 且用 `nrows_x < small_n` (严格小于), 64 行以上的权重完全不受影响。

**为什么这是"回归正确路径"而不是"引入新路径":** M=1 时 `should_use_mmvq` 本来就
`return ne11 <= 4` -> true, 普通 decode 一直走 MMVQ。baseline 的不一致在于同一权重
**decode 用 MMVQ、verify 用 MMQ**, 两套数值。改动让 M=2..8 与 M=1 一致。

**效果 (profile 直测, 按 round 归一):**

| 桶 | baseline ms/round | B-on ms/round | 变化 |
|---|---|---|---|
| **MMQ nty=1 (=N48)** | **2.032** | **0.037** | **-98.2%** |
| MMVQ | 0.626 | 1.151 | +0.53 (接收 94 个调用) |
| 其余 MMQ 桶 | 36.63 (合计) | 34.56 (合计) | 持平 |
| **GPU busy 合计** | **45.1** | **43.7** | **-1.44 (-3.2%)** |

**端到端 (6 prompt, `--temp 0`, 确定性):**

| 指标 | off | on | 变化 |
|---|---|---|---|
| ms/round | 66.40 | 64.90 | **-2.26%** |
| tok/round | 3.499 | 3.630 | **+3.75%** |
| t/s | 52.35 | 55.49 | **+5.99%** |

**6/6 prompt 全部 tok/round 上升** (符号检验 p~3%)。单 prompt 首次测到 57.8 -> 60.8 t/s 时
我曾怀疑是"接受率运气", 6 prompt 配对数据否定了这个怀疑: ms/round 确实下降, 且与
profile 的 -1.44 ms/round 吻合 (1.44/66.4 = 2.2%)。

> **口径警告 (这是本轮踩过的坑)**: 投机解码的 t/s 会被接受率放大。`ms/round` 才是
> 每轮成本的真口径。数值变化会改变采样路径进而改变接受率, 所以 t/s 的提升里混了
> "文本路径变好"的成分。**upstream 的 e4m3 W8A8 激活路径是默认行为, 不是 fork 引入的**
> (我一度误判为 fork 降精度, 查 `origin/master` 后已撤回)。

### 9.5 验收 (2026-10-01, B 默认开启)

**正确性 (不得引入新 FAIL):**

| 套件 | 结果 |
|---|---|
| 全量 `MUL_MAT` (所有类型, 不只 mxfp) | **1359/1359 passed** |
| `MUL_MAT` mxfp 子集 | **86/86 passed** |
| `GATED_DELTA_NET` | **39/39 passed** |

**PPL 门 (文档口径 `-c 4096 --chunks 8 -ub 2048`):**

```
Final estimate: PPL = 6.9633 +/- 0.12494
```

与历史红线 **6.9633 完全相同 (逐位一致)**。原因是标准 PPL 的 `M=2048 >= 256` 走 radiance
fast path, `ssm_alpha/beta` 根本不经 MMQ/MMVQ —— 这反过来证明 **B 对 prefill 路径零影响**。

> 附: 用 `GGML_RAD_PREFILL_MIN_M=4096` 强制 `N=48` 落到 MMQ/MMVQ 后实测 `6.9433`
> (B on, 未跑配对 off, 仅记录不下结论)。该配置不是项目使用场景, 不作为验收依据。

**性能红线 (`llama-bench -p 512,2048 -n 128 -r 3 -fa 1 -ub 2048`):**

| 指标 | 历史最优 | B on | 结论 |
|---|---|---|---|
| pp512 | 2283.7 | 2161.41 ± 629.66 | 噪声内 (误差棒极大) |
| pp2048 | 2563-2577 | 2537.99 ± 233.54 | 噪声内 |
| tg128 | 29.85 | 29.89 ± 0.10 | 持平 |

**e2e (dflash n7, `--temp 0`, 2 次):**

| | baseline | B on | 变化 |
|---|---|---|---|
| t/s | 57.78 / 57.78 | **67.98 / 68.32** | **+17.7%** |
| n_accept | 187 | 202 | +8.0% |
| accept | 38.40% | 47.31% | +8.9pt |

> t/s 的增幅大于 ms/round 的 -2.26%, 因为接受率同时上升。**两个效应独立**, 缺一不可:
> 省时间 (确定, profile 直测 2.032 -> 0.037 ms/round) + 数值变化 (接受率, 机制未证)。



### 9.6 decode 线的收尾 (2026-10-01): M 到路径的映射已完备, 无剩余缺口

**先纠正一个我犯过的错误**: 我曾说 "M=8 进不了 WMMA 路径" —— **错的**。
`MMVQ_RDNA4_MAX_BATCH_SIZE = 4` 这个阈值的**全部作用**就是让 M ∈ [5,8] 从 dp4a 升到
MMQ(WMMA)。M=7/8 的大权重**早就在用 WMMA**, 现在 68 t/s 里已含这部分收益。

**M -> 路径的完整映射 (同一次 run 的 trace 实证, 由 kernel 模板参数直接读出):**

| M | 路径 | trace 证据 |
|---|---|---|
| 1 | MMVQ | `M=1 x28080` (纯 decode) |
| 2 | MMVQ | `M=2 x543`, 全宽度 |
| 3 | MMVQ | (阈值 4 以内) |
| 4 | MMVQ | `M=4 x508` |
| **5,6** | **MMQ = WMMA** | 阈值补丁的作用区间 |
| **7** | **MMQ (仅 nrows_x=48 留 MMVQ)** | `M=7: nrows_x={48: 96}` |
| **8** | **MMQ (仅 nrows_x=48 留 MMVQ)** | `M=8: nrows_x={48: 960}` |

两条改动各司其职, 合起来正好完整覆盖:

```
M=7/8 的 MXFP4 权重:
  ├─ nrows_x >= 64  ->  MMQ (WMMA)    <- 阈值补丁(=4) 从 dp4a 升上来
  └─ nrows_x == 48  ->  MMVQ (dp4a)   <- 路径 B (small_n=64) 从 MMQ 降回去
```

**结论: 主题 ("为不同 MXFP4 张量形状定制路径") 在 decode 线上已收口, 无剩余缺口。**

| 形状 | 路径 | 状态 |
|---|---|---|
| N_out >= 64, M ∈ [5,8] | MMQ/WMMA | 已覆盖 (阈值补丁) |
| N_out = 48, M ∈ [5,8] | MMVQ | 已覆盖 (路径 B, 2.032 -> 0.037 ms/rd) |
| N_out >= 64, M <= 4 | MMVQ | 共识 (WMMA 的 16 行 tile 填不满) |
| N_out >= 256 | radiance atiled | 仅 prefill, decode 用不到 |

**radiance 在 M∈[5,8] 不可用的根因 (实测, 非推测):**

强行 `GGML_RAD_PREFILL_MIN_M=8` 打开闸门 -> **OOM 崩溃**:

```
ROCm error: out of memory
  in ggml_cuda_mul_mat_q_radiance (mmq.cu:168)
  hipMalloc(&w.W, wq_bytes)
```

`g_radiance_weights` 是**进程级永久缓存** (按 `src0->data` 索引, 无失效机制), 每个 MXFP4
权重另存一份 repack 副本。显存账 (`-c 131072`):

| 项 | GB |
|---|---|
| target 权重 | 15.83 |
| drafter 权重 | 1.05 |
| KV (131072 tok, 64 KiB/token) | 8.00 |
| **radiance repack 副本** | **13.13** |
| **合计** | **38.01 / 32 GB -> 缺 6 GB** |

**关键: repack 不是"加倍", 是精确定重排** —— 两者逐位同尺寸:

| | B/权重 | 24.72B 参数 |
|---|---|---|
| 原始 MXFP4 (16B qs + 1B E8M0)/32 | 0.53125 | 13.13 GB |
| radiance (W=NK/2 + Ws=NK/32) | 0.53125 | 13.13 GB |

所以问题不是副本太大, 而是**它与原件同时驻留**。W+Ws 恰好等于原 buffer, 理论上可原地
repack 省掉全部 13.13 GB —— 但那要动权重驻留结构, 属结构性改动, 未做。

这解释了为什么 radiance 当初只在 prefill 被验证: prefill 用 `-c 4096`, KV 仅 0.25 GB,
合计 30.26 GB **装得下**。

### 9.7 非 GEMM 账本与一个分类纠正

**纠正**: 我曾把 "非 GEMM" 当整体报为 "下一个方向" (7.67 ms/轮)。**其中 16.2% 是
rms_norm/cpy_scalar/concat 等通用 F32 基础设施, 与本模型的 MXFP4 量化无关**, 不属于
"为 MXFP4 定制路径" 主题。按权重类型重新分类 (同一次 run, 11 轮稳态, GPU busy 43.36 ms/rd):

| ms/轮 | %busy | 类别 |
|---|---|---|
| **31.62** | **72.9%** | **MXFP4 (type39) 权重** |
| 7.04 | 16.2% | 通用 F32 基础设施 (与量化类型无关) |
| 4.07 | 9.4% | MXFP8 (type43) 权重 |
| 0.63 | 1.5% | MXFP4 激活量化 (配套) |

**F32 部分已被上游文档收口** (`design-gemm-rdna4.md` §196-269): rms_norm<1024> 已达
~706 GB/s 无空间; rms_norm<256> 优化 2x 也只 ~2% 端到端; concat 三条路线全否决, 该线关闭。
**勿重复投入。**

### 9.8 MMQ 的 ISA 账 (供以后参考, 本轮未改)

`mul_mat_q<MXFP4, J=16, false>` = **28.66 ms/轮 = 66% GPU busy**。整核 855 条 / 5744 B,
主 K 循环 `0x33ccc..0x34eb0` = 634 条:

| 类别 | 条数 | 占比 |
|---|---|---|
| VALU | 293 | 46.2% |
| **`v_perm_b32` (nibble->fp8 LUT 展开)** | **96** | **15.1%** |
| LDS | 92 | 14.5% |
| WAIT/DELAY | 82 | 12.9% |
| GLOBAL | 30 | 5.4% |
| **`v_wmma_f32_16x16x16_fp8_fp8` (真算力)** | **16** | **2.5%** |

**每 40 条指令才喂出 1 次 WMMA。** 对照 radiance atiled 的作者 ablation: WMMA 占完整
时间 **55%**。这是 MXFP4 特有的结构代价 (4bit nibble 要先展开成 fp8 再折叠 E8M0 scale)。

复现方法 (本轮打通):
```bash
# 1) 解出 gfx1201 ELF (MXFP4 的 MMQ 实例在 template-instances/, 不在 mmq.cu)
cd /tmp && cp <build>/ggml/src/ggml-hip/.../template-instances/mmq-instance-mxfp4.cu.o .
/opt/rocm/lib/llvm/bin/llvm-objdump --offloading mmq-instance-mxfp4.cu.o
# 2) 反汇编目标 kernel (type39 J=16 fallback=false)
F=$(ls *gfx1201*); /opt/rocm/lib/llvm/bin/llvm-nm -S "$F" | grep "_ZL9mul_mat_qI.*L9ggml_type39ELi16ELb0E"
/opt/rocm/lib/llvm/bin/llvm-objdump -d "$F" > mmq.asm
```
注意: **mmq.cu.o 里没有 type39 符号** (只有 type107), 必须找对 .o。
