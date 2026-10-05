# Qwen3.8-27B RDNA4 吞吐改造: llama.cpp-gpu-tuned 对标 vllm-radiance

> 注: 本文件是新任务 (RDNA4 吞吐改造) 的设计文档。与旧的 design.md (MMQ dp4a 审计,
> 已 CONCLUDED) 相互独立。远端仓库: seirin@192.168.31.49:/media/seirin/HDD500G/llama.cpp-gpu-tuned
> 2026-09-30 重写: vLLM radiance TP1 单卡实测落地, 性能归因从 "GDN 主犯" 修正为
> "prefill GEMM 执行 + 融合 + spec decoding"; 纯 MXFP4 重新量化方向经用户否决删除。

目标: 在 seirin-epyc (192.168.31.49) 的 RDNA4 上, 把 fork (OwNhj/llama.cpp-gpu-tuned master)
的 Qwen3.8-27B 吞吐改造成接近 vllm-radiance 的水平。参照数字全部来自 TP1 单卡
R9700 (gfx1201) 实测 (2026-09-30, 本机跑通官方镜像 stilldeadcode/vllm-radiance:0.9.3):

## 参照 (radiance TP1 单卡 R9700 实测, dflash SPEC=7, fp8 KV)

| 指标 | vLLM 实测 | llama.cpp fork 现状 (Mxfp4 混合, ROCm0) |
|---|---|---|
| prefill 30k | 2721 t/s | pp2048 1134 t/s |
| prefill 80k | 3346 t/s | (同上, chunked prefill 差异) |
| TG 短 ctx | 86-88 t/s | tg128 26.2 t/s |
| TG 60k ctx | 44.6 t/s | (待测) |

vLLM 赢在 (日志实据): RADIANCE_MXFP4 304/304 linear 全绑自制 W4A8 内核 +
chunked prefill (2560 token 块) + fuse_norm_quant/fuse_act_quant 编译 pass +
cudagraph 全 capture + R4D attention + dflash 投机采样。
radiance 官方文档 TP1 口径: prefill 3150/3085/3070/2881/2668 @ 2k/8k/16k/32k/64k。

## 基线 (llama.cpp fork, 实测 r3, ngl99, fa on)

| 配置 | pp512 | pp2048 | tg128 |
|---|---|---|---|
| Mxfp4 混合 (18.3G, 5.75bpw) @ ROCm0, f16 KV | 1067.3 | 1134.3 | 26.20 |
| Mxfp4 混合 @ ROCm0, f8 KV | 1027.7 | 1108.3 | 25.90 |
| IQ3_S (GSQ) @ ROCm2, f16 KV | 725.7 | 764.6 | 33.25 |

rocprofv3 剖析 (pp2048, ROCm0): mul_mat_q 75.6% / gated_delta_net 9.5% /
unary_gated_op 3.1% / quantize_mmq_q8_1 1.8%; pp FLOPs 利用率 ~86 TFLOPS (~41% fp8 峰值)。
Mxfp4.gguf 实为混合量化 (Q8_0 31.3% + MXFP4 30.1% + Q6_K 27.7% + Q4_K 10.9%),
字节分布解释 tg128=26.2 (640GB/s 带宽墙: 18.3G/26 t/s)。

已否决方向: 纯 MXFP4 重新量化 (用户否决: 不能重量化的张量会炸, 且更低位数
llama.cpp 也跑不出 vLLM 的性能 -- 实测证实差距在执行层不在格式)。目标模型锁定
现有 Mxfp4 混合 GGUF, 不动权重。

## 改造优先级 (2026-09-30 修正)

### P1: prefill 大 M GEMM 提效 (主攻, mul_mat_q 75.6%)
microbench (2026-09-30, test-backend-ops + qwen3.8 专用形状, R9700, 两次运行 ±1%):
- MXFP4_E4M3 fp8 WMMA: 92-111 TFLOPS (28-34% 峰值) @ n=256-2560 全形状
- f16 WMMA 对照: 69-112 TFLOPS; q8_0 int8 WMMA 对照: 87-96 TFLOPS
- 结论: 瓶颈为 MMQ/MMF 框架共性 (tile/smem/调度), 非 MXFP4 特有;
  vLLM 同卡 ~172 TF (53%), 差距 1.6-1.9x, 改 GEMM kernel 即可对齐。
- fork tile 配置: mmq-config-rdna4.cuh MXFP4_E4M3 最大 I x J = 128 x 128
  (I=src0/M 方向, J=src1/N 方向), stream_k=false。
做法: 先实验更大 J (N 方向 128 -> 256, 摊薄权重读取) 与 nthreads=256 的组合,
参考 radiance A-tiled 思想; smem 占用与 occupancy 用编译期 static_assert 校验。
验收: 34816x512x5120 形状 >= 150 TFLOPS, pp2048 端到端 1134 -> 1800+, PPL 门通过。

### P2: 融合 pass (消 ~15% 杂活)
rms_norm+quant、silu+mul+quant 融合 (对标 vLLM fuse_norm_quant/fuse_act_quant)。
放在 P1 之后: P1 改变 GEMM 布局, 融合接口依赖它。

### P3: GDN prefill chunk 化 (实测占 pp2048 的 18.4%, ~169ms/920ms)
gated_delta_net.cu 逐 token 串行 -> chunk 并行 (f16 WMMA)。

实测诊断 (2026-10-01):
- A0 寄存器预取 t+1 (k/q/v/beta/g): pp2048 2146 ~ 基线 2160 -> 否证, 非全局读延迟。
- A0.5 块级 LDS 双缓冲共享 k/q: pp2048 1955 (-10%) -> 否证; 冗余读在 L1 广播里本来就便宜,
  per-token __syncthreads 把 warp 独立链串行化才是毒药。两者均已 git checkout 回退。
- 定量: 每 token 约 340 clk 串行依赖链 (2 次 warp-reduce + FMA 链) x 2048 token ~ 400us,
  与实测 440us/次 吻合 -> 纯延迟墙, 唯一解是 chunk 化 (链长 T -> T/C)。
- chunk 数学已 numpy 对拍通过 (f32 下 max|o_chunk-o_recur| = 1.19e-7, 状态 5.96e-8):
  D = U - T diag(beta A) K S0, U = T (beta.V), T = (I + tril(diag(beta) rat K K^T))^-1,
  o = scale (diag(A) Q S0 + tril(diag(A) Q K^T diag(1/A)) D),
  S_next = A_C S0 + K^T diag(A_C/A) D。

radiance 真实做法 (决定性证据, 源码 /media/seirin/HDD500G/radiance-test/radiance_gdn.py):
- 头部注释 + 266 行: "The whole layer: conv_prep -> kkt_solve -> chunk_scan on the prefill
  path... Nothing below calls a Triton kernel."
- r4d 的 gdn_chunk_scan_k128_v128_c64_bf16 把 FLA 的 3 个 Triton kernel
  (recompute_w_u_fwd / chunk_gated_delta_rule_fwd_h / chunk_fwd_o) 合成 ONE kernel,
  状态常驻 WMMA 累加器跨整个 chunk 循环; 注释原文: "h, w, u and v_new never reach HBM,
  which is where the win comes from: the same maths reads about a third of the bytes."
- kkt_solve: A = (I + strict_lower(diag(beta) K K^T e^dg))^-1 per chunk, **fp32 gram 留在 LDS**,
  只有 A (T,H,CHUNK) bf16 落 HBM (442 行)。
- 几何 k128/v128/c64 与本模型 GDN 完全一致 (head_k=head_v=128);
  serve-mxfp4.sh CHUNK=8192 (prefill chunk = max-num-batched-tokens)。
- 对照: 朴素 FLA 三 kernel 版要落 h(50MB) + w/u(50MB) + A_cumsum(25MB) + A(12.6MB) ~ 138MB
  (T=2048, H=48, K=V=128, 32 chunks, bf16) —— 这正是 radiance 用融合消掉的东西。

结论 (按用户判据 "150MB 常驻在 vllm 有没有"): **vllm 没有**。150MB 三 kernel 方案作废。
vllm 有的是融合版 (kkt_solve + chunk_scan, 中间量全不进 HBM, 只落 A ~12.6MB 瞬时),
但那是闭源 libr4d 手写 HIP, 自研 = 2 个新 kernel (LDS 内 gram+三角求逆 + WMMA 累加器跨
chunk 融合 WY/scan/output) 的多日工程, 无参考源码可抄。

### A1 首版移植实测 (2026-10-01): 功能正确, 性能负 (2.9-4x 慢), 已改 opt-in
实现: gdn_chunked_prefill_cuda<C=16, VS=32> (f32, 全 LDS FMA), 每 workgroup 一个
(v-head, seq, 32 列) 切片, 状态常驻 LDS 跨 chunk; 数学公式 numpy 已对拍 (1.19e-7)。
正确性: 当时以为 test-backend-ops GDN 36/36 通过 —— **实为假通过**, 见第 3 节与 A2 结果第 5 条。
性能 A/B (同形状, test-backend-ops perf, μs/run):
  | 形状 | recurrent(OFF) | chunked(ON) | 比 |
  | 4 heads x128, 512 tok  | 202.51 | 809.91 | 4.0x 慢 |
  | 32 heads x128, 512 tok | 614.36 | 1803.08 | 2.9x 慢 |
  | 32 heads x128, 1024 tok| 1230.95 | - | - |
慢的四个根因 (可证):
  1) 无寄存器分块: 每个 MAC 两次 LDS 读 -> LDS 带宽墙; recurrent 版状态列常驻寄存器,
     每次读入的 k/q 被 4 行复用。LDS ~19.7 TB/s vs FMA ~19.7 T/s, 2 读/MAC 直接腰斩。
  2) workgroup 太少: grid = H x (S_v/VS) = 32x4=128 (4-head 用例仅 16 个) x 128 线程;
     4-head 用例 = 1 warp/CU, GPU 饿死; 36.5KB LDS/workgroup 进一步限占用。
  3) 每 chunk ~10 次 __syncthreads x 32 chunk, 而 workgroup 只 4 warp 无法隐藏
     (与 A0.5 per-token barrier 同款毒药)。
  4) 结构性 FLOP 税: chunk 化多出 Gram(K K^T) 与 Q K^T, 合计 ~57K MAC/token-head
     vs recurrent 的 32K, 约 2x。
radiance 之所以快: 其 kernel 是 **bf16 WMMA 张量核** + 状态常驻 **累加器寄存器** 跨整个
chunk 循环 (gdn_chunk_scan_k128_v128_c64_bf16, C=64), 手调闭源库。chunk 化数学本身不赢,
赢的是张量核形态。当前 f32+LDS FMA 移植是另一台机器。
后续三选一 (待用户定): (a) 张量核 bf16 WMMA 重写 (radiance 真形态, 多日, 数值转 bf16);
(b) 现有 f32 版做寄存器分块+拆 V 提并行 (约 1 天, 天花板 ~1.5x); (c) 收口转 TG。
当前默认: chunked 为 opt-in (GGML_GDN_CHUNKED=1), 模型回到基线 (pp2048 2110 vs 2160, 噪声内)。

### A1 续: v2 寄存器分块 / v3 内层展开 (2026-10-01, 用户指示"继续做到底")
| 版本 | 形态 | 32heads/512tok | vs recurrent(614.36) |
| v1 | f32 全 LDS FMA, C=16/VS=32 | 1803.08 us | 2.9x 慢 |
| v2 | + 4x4 寄存器分块, K 维跨 lane 切分, float4 LDS | 1283.59 us | 2.1x 慢 |
| v3 | + 内层循环模板化+全展开 (消除运行时边界) | 1456.78 us | 2.4x 慢 |
正确性: 三代当时都报 36/36 —— **实为假通过** (head_size=128 的用例 n_tokens 只有 1/4,
  低于 chunked 的 64 门槛, kernel 从未执行), 且其原地求逆是错的。详见 A2 结果第 3-5 条。
诊断证据:
- VGPR 95, SGPR/VGPR spill 均为 0 (clang -Rpass-analysis=kernel-resource-usage)
  -> 不是寄存器压力。
- 实测吞吐 0.70 T MAC/s = 峰值(19.7 T)的 3.5%; 而 LDS @2B/MAC 的理论界是 26 ms/pp2048
  -> 差理论界 12-14x, 故也不是 LDS 带宽墙, 是**延迟/占用墙**。
- 结构: workgroup 数 = H x (S_v/VS) = 32x4 = 128 (4-head 用例仅 16), 每 workgroup 4 warp,
  LDS 47.4KB -> 每 CU 仅 2 个 workgroup = 8 warp/CU -> 无法隐藏 LDS ~30-40 cycle 延迟;
  每 chunk 8 个 phase / ~9 次 barrier x 32 chunk 进一步串行化。
- 相位 tile 数与线程数错配: kS0/qS0 恰好 128 job 用满 128 线程, 但 D/out 仅 64, gram/qk 26;
  拆小 VS 会让每相位 tile 更少 (线程闲置更严重) 且 Gram/QK 冗余翻倍 -> 无解的死结。
结论: FMA 形态与本题形状结构性错配 — 一个 warp 用 4x4 tile 只能算 16 个输出, 而 16x32 的输出面
根本喂不饱 128 线程; radiance 用 WMMA (一个 warp 一个 16x16 tile) 正是绕开这一点。
翻盘只剩 WMMA bf16 (多日工程, 且数值转 bf16 需重过 PPL 门)。

备份 (用户要求): /media/seirin/HDD500G/backups/llamacpp-fast-20260930-215341/
  tracked-changes.patch + radiance-gemm.{cu,cuh} + gdn-versions/{orig,v1,v3} + README.md (含还原步骤
  与全部实测数字)。当前工作树 = 该备份状态 (chunked opt-in 默认关, 模型速度不受影响)。

### P4: TG 侧
单卡带宽墙 ~34 t/s (物理)。改进只能来自: MTP/draft-mtp 端到端调优 (已有基础),
长上下文 tg 衰减归因 (f8 KV 假设待 bs=1 长上下文复核)。

## 数值正确性门
- 每次 kernel 改动: test-backend-ops 的 GDN/FFN/attention + MMQ 用例。
- 端到端: llama-perplexity wiki.test.raw, PPL 相对 master 基线漂移 < 1% 才算过。
- 基准: -dev ROCm0 (Mxfp4) / ROCm2 (IQ3_S), -p 2048 -n 128 -r 3, 对比本文档基线表。
- vLLM 参照服务: 容器 vllmmxfp4074 (TP1 GPUS=0 dflash), 不测时停掉释放 R9700。

## 风险
- gfx1200 (ROCm2) 与 gfx1201 (ROCm0) WMMA 同族, Mxfp4 模型以 ROCm0 实测为准。
- radiance 的 libr4d GDN 细节闭源, GDN 部分只能按 DeltaNet 论文 + FLA 推。
- MMQ tile 改动影响所有量化类型 -> 用 int8 路径回归测试兜底。

## P1-2 实验记录
改动点 (2026-10-01):
1. mmq-config-rdna4.cuh: MXFP4_E4M3 fallback=false 段追加 J=192(J=80..192 nthreads=256)
   与 J=256(nthreads=512) 的 CASE 行。
2. mmq.cuh:1587: switch_J 遍历上限 128 -> 256。
3. mmq.cuh:1651 附近 switch(J_best): 补 case 192 / case 256。
smem 预算 (smpbo 64KB): J=192/nt256 = 45.8KB; J=256/nt512 = 49.0KB; 均 fit。
加载/写回覆盖验证: nwarps=8(256线程) data-load 16 步、scale-load 4 步覆盖 I=128;
nwarps=16(512线程) 减半步数; vec_dot/write_back 的 j0 循环均为 constexpr 步进, J=192/256 合法。
回滚: git checkout -- 两文件 + 重编。

结果 (2026-10-01): 正确性 OK (34816 形状 n=8..512 逐元素 vs CPU, 4/4 backends)。
[更正] 上面的正确性验证无效: 34816 用例只注册在 make_test_cases_perf(), test 模式
过滤器选中 0 个用例, "4/4 backends passed" 是空跑。test-backend-ops 的 -p 过滤
选中 0 用例时仍报 OK, 不能作为通过证据; 任何验证必须先确认选中用例数 > 0。
有效正确性门 (PPL, corpus.txt 8 chunks, n_ctx 4096, fa on, ROCm0):
  基线 (stash 改动后重编): [8] = 6.4652
  改动版 (J=192/256):      [8] = 6.4652  (逐 chunk 完全一致)
perf: 全部 15 个 prefill 形状 +0.5~1.0%, 最大形状 110.8 -> 111.7 TF。
结论: N 方向 tile 加宽只带来 ~+1%, 与权重读带宽受限假设不符。
瓶颈重定位: J=128 时 x-tile 已经被 128 个 y-column 摊薄 (x-tile读 = K/(2*J) 次),
继续加宽摊薄空间很小; 剩余差距 (111 vs vLLM ~172 TF) 更可能来自:
(a) y (q8_1 activation) 全局读与 scale 处理路径,
(b) 调度/occupancy (5 CUs/GC, 每 GC 1 个 workgroup),
(c) K 循环内的 mma 发射密度。
下一步 P1-2b: rocprof 对比 J=128 选中的 kernel 与 vLLM 的 GEMM 指标 (valu/VALUUtilization,
LDS 带宽, wave 占用), 按证据选 (a)/(b)/(c) 之一再动手。
注: J 加宽改动保留 (一致 +1%, 无回退, 仅选中更少 tile 时生效)。

## P1-2b rocprof 定位结论 (2026-10-01)
对象: mul_mat_q<MXFP4_E4M3,128,false> @ 34816x512x5120, ROCm0 (gfx1201)。
采集: rocprofv3 --pmc (gfx1201 上 SQ_INSTS_VALU 恒 0, VALU_USEFUL/LDS_CONFLICT/VSIMD
等 counter 缺失, 只有 SQ_WAVES/GRBM 可用; duration 用 Start/End_Timestamp)。
实测: VGPR=216, SGPR=128, LDS=0(!fp8 路径不用 LDS), wg=256(8 waves), grid=278528,
kernel dur=1.577ms -> 115.7 TFLOPS (与 microbench 一致)。
分析:
- 每 wave 全程仅 10 条 wmma (81920 FLOP/wave / 8192), 278528 块 x 8 waves;
- 每 wave 串行 ~2102 周期, wmma 发射占比 <0.5% -> 数学发射完全不是瓶颈;
- y (q8_1) 侧全局重读 ~3TB/s 走 L2 (x 373GB/s), f16 对照排除 VRAM 墙;
- 216 VGPR -> 每 SIMD 只驻留 1 wave -> 全局/L2 读延迟 (~400-600cyc) 无法被
  其他 wave 遮盖, 每条指令平均 ~7-14cyc 有效吞吐 -> 延迟受限。
对照: f16 的 mul_mat_f (同 wmma, 双缓冲+ntA/ntB 寄存器分块, 112TF) 说明同卡
f16 WMMA 能到 112TF; vLLM radiance ~172TF 同理是高 occupancy 结构。
结论: P1-2 的正确方向 = 降 VGPR / 提高 occupancy / y 读预取双缓冲, 而非加宽 tile。
候选改动 (按风险从低到高):
(a) load_tiles 里 e2m1->e4m3 LUT 展开改算术 (v_perm 依赖 64KB LUT 常驻, 占 VGPR/SGPR);
(b) y tile 全局读软流水 (提前 1 个 K 块发出 read, 类似 MMF 的 gather_tile 双缓冲);
(c) 削 sum[]/C 累加器寄存器 (J=128 时每 wave 32 float 累加, 降 I/ntx 重新平衡)。

## radiance vLLM 源码对照 (2026-10-01, radiance_mxfp4_fp8.hip 1846 行已逐段读)
注意: 之前记录的 "~172 TF" 偏保守, hip 源码注释实测: folded 190TF, atiled 215-220TF,
WMMA-only 理论上限 310TF (发射效率 55%), 端到端 PP 5141 t/s @2k。
结构对比 (radiance atiled vs 本 fork MMQ fp8):
| 维度 | radiance atiled | MMQ fp8 (现状) |
|---|---|---|
| tile | BMF=256 x BNF=128 (TM=4,TN=4,WN=2), NTHREADS=256 | I=128 x J=128, nt=256 |
| A(激活) 进数 | fragment 直读 global->寄存器(0 LDS!), 256B/frag 连续 | x_tile 38.9KB LDS + LUT 展开 |
| W(权重) 进数 | LDS 折叠 e2m1->e4m3 (9-18KB), 每 K-slab 一次 | 每 K-iter 展开进 LDS(Q8_1 布局) |
| acc | TM*TN=16x floatx8=128 float/wave | 32 float/wave (J*I/256) |
| 每 wave WMMA | 16/slab x K/64 slab (密集) | 10/全程 (0.2%!) |
| occupancy | 3 block/CU (LDS 9-18KB) | 1 wave/SIMD (216 VGPR + 48KB smem) |
关键手法 (每条都标了实测数字, 全部可抄):
1. clamp never predicate: 边界 clamp 不用 exec-mask 分支, 单 s_wait_loadcnt 覆盖整段
   staging load, 值 5-20% (prefill)。
2. SGPR base + SADDR: wave-uniform 64bit base 提到寄存器, lane 只算 32bit offset,
   省掉 v_add_co/ci 链 (matmul 块 565 条指令里 71 条是地址算术)。
3. __builtin_amdgcn_sched_barrier(0) 每 k-step: 禁编译器把后续 step 的 fragment load
   上提, 保寄存器 (否则 346 条指令 + 140 v_dual_mov_b32 悬在 WMMA 前)。
4. LDS-only fence: __syncthreads 是全地址空间 acq/rel, 改 s_wait_dscnt+s_barrier
   (radiance_lds_barrier), 去掉两次 global_inv+loadcnt 排空。
5. kMag LUT 放 LDS (ds_load 有独立计数器, 不挂 loadcnt 链)。
6. EPIFAST epilogue: 全块在界内时零谓词零 64bit 地址算术 (省 15% 指令流)。
7. folded scale: W 折叠进 e4m3 时把 E8M0 指数折进 byte (kMag 查表), 内循环无 scale 乘法
   (99.998% 块无损, 极端 flush 10-282 块); 内循环变纯 WMMA。
8. A-tiled 布局 (最大头, +13%): fused rms+quant kernel 直接输出 fragment-tiled
   fp8 (每 16m x 16k = 256B 连续), GEMM 端 sA staging 整个消失 (原占 24-32%)。
   代价: activation 布局耦合, 需要 producer 配合改 -> 我们 P2 融合 pass 正好能做。
9. 已被 radiance 实测否证的路 (别再试): W 直接进寄存器不经过 LDS (1.1-1.3x 慢),
   A fragment 软流水下一 slab (230VGPR 时中性/负), 下一 slab W 寄存器预取 (+-1%),
   TN=8 (acc 寄存器墙), TM=3/TN=6 (net 0)。
行动映射 (到本 fork):
- P1-2c 短平快: 在 MMQ 现框架内抄 #1 #2 #3 #4 (codegen 级, 不动算法) -> 目标
  降 VGPR/提高发射效率; 预期 +10-20% 而非 2x。
- P1-2d 结构级 (若 P1-2c 不够): MMQ 端加 fp8 快路径 = x_tile 常驻不重展开 (q8_1 y
  已是 fp8, x LUT 展开每 K-iter 重复做是最大浪费) 或按 radiance 把 x 也 fragment-tiled。
- P2 融合 pass (原计划): 正好对齐 #8, rms+quant 输出 fragment-tiled, 一步到位。

## P1-2c 实验记录 (2026-09-30)
1. y tile 双缓冲 (两半 K 块同时 staging, 中间去 barrier 对): PPL 位精确一致
   (6.4652, 语义正确), 但 perf 0.77-0.86x 全面回退 (均值 0.81x)。
   => 去掉中途 barrier 不能省: occupancy 1 wave/SIMD 时第二个 vec_dot 反正要等
   x 展开, 重叠无从发生。改动保留在代码里但 y_prefetch 门控 return false。
2. occupancy 提示 (launch_bounds minBlocks 2->4, J=128 与 J=256 两行): 无变化
   (108.1 -> 108.8 TF, 噪声) => occupancy 由 216 VGPR 硬约束决定, 提示无效。
3. IF cache 假设 (用户提出 RDNA4 靠大 IF cache): 证伪。
   mul_mat_q<MXFP4_E4M3,128,false> ISA = 8052 B (llvm-nm -S / offload bundle 解出),
   32KB L1 指令缓存占 25%, 单 kernel 体放得下, 无取指墙。radiance 565 指令/块
   同样远小于 32KB。IF cache 优势体现为"整体驻留后发射无抖动", 不是我们的差距来源。
4. 关键新认知: gfx12 wmma 有独立 Accum VGPR 文件 (256B/lane), radiance TM*TN=16
   个 floatx8 累加器走 Accum VGPR, 主 VGPR 只剩 93 个装地址/索引 -> 3 block/CU。
   MMQ 的 216 主 VGPR 被什么占着需 ISA 级确认 (疑似: load_tiles 的 LUT 展开中间值
   + 手写地址算术 + sum[] 部分不走 Accum)。
下一步 (P1-2d, 按 radiance 证据重排):
- 优先: ISA 反汇编 mul_mat_q<128,false> 确认 216 VGPR 的构成, 找可削减项。
- 然后: 评估在 MMQ 内用 Accum VGPR 显式放置 acc (gfx12 compiler 自动分文件,
  wmma 链上的 acc 天然在 Accum 文件; 检查 sum[] 是否被合并进 wmma 链)。
- 结构级: A-tiled 路线与 P2 融合合并 (上面 #8), 这是 radiance +13% 的来源。

## P1-2d ISA 级账本 (2026-09-30, 完成定性定位)
方法: 从 mmq-instance-mxfp4-e4m3.cu.o 的 offload bundle 解出 gfx1201 ELF,
llvm-objdump -d 反汇编 mul_mat_q<MXFP4_E4M3,128,false> (8052B, 1173 条)。
指令分布: 32 wmma (2.7%) | 96 v_perm (LUT 展开) | 195 v_dual_* |
50 v_add_co/ci (64bit 地址) | 160 wait/delay | 53 global_load + 54 ds_store。
VGPR 216 构成: 64 = sum 累加器 (v[106:170), J*I/256), ~40 = A/B 片段,
~110 = 地址/索引/LUT 中间值。
时间账 (34816x512x5120, dur 1.577ms): 每 tile 每 iter(K=256) 79us = 229k cycles,
而循环体仅 1173 条 => 发射利用率 0.5%; 其余 99.5% 全部是等内存延迟
(2 wave/SIMD x L2 600cyc 延迟, 无足够 wave 填充)。
radiance 对照: 565 条/iter, 93 VGPR -> 5 wave/SIMD, 发射利用率 ~50%。
理论: occupancy 2.5x x 发射率 100x = 效率差全部可归因到 occupancy 链。
IF cache 已排除 (kernel 8KB << 32KB)。
P1-2e 行动 (唯一杠杆 = 削 VGPR 提高 occupancy):
1. 砍 sum[]: 每 wave 64 float 累加器过大 -> 降 J 到 64 (sum 32) 或改每波
   负责行数 (ntx), 优先试 J=64/I=128 with nthreads=256 (smem 26KB, 4+ block/CU)。
2. 地址算术: 仿 radiance #2, wave-uniform base 放 SGPR (readfirstlane), lane 只算
   32bit 偏移, 消 50 条 v_add_co/ci 与相关寄存器。
3. LUT 展开值复用: load_tiles 的 perm 结果直接进 ds_store, 不留长生命寄存器
   (检查编译器是否已做; 若没做, 拆 load_tiles 内联块)。
顺序: 1 最便宜 (纯配置改动 + 编译), 先试 J=64; 若 occupancy 上去且 TF 逼近
150+ 再做 2/3; 若 J=64 仍 2 wave/SIMD 则必须走 ISA 精修。

## P1-2e 结果 (2026-09-30): 配置级实验全部到顶
- I=64/J=64/nt=128/occ=4 强制选中: 94.5 TF (< J=128 的 108)。
  小 tile 的 staging/perm 开销按波数放大, 吞掉 occupancy 收益。
- 结论: MMQ 框架内纯配置调整已到顶 (~110 TF)。
  VGPR 216 的大头 (64 acc + ~110 地址/LUT 中间值) 是框架结构决定的:
  radiance 的 93 VGPR 靠的是 (a) acc 走 Accum VGPR 且 TM/TN 寄存器分块,
  (b) SGPR wave-uniform base + SADDR, (c) kMag 在 LDS, (d) 无 ids/expert
  的复杂寻址。这些在 MMQ 框架内都动不了 => P1 (MMQ 线) 收官。
判定: P1 线理论天花板 ~110 TF (发射利用率 0.5%), 与 vLLM 215TF 的差距
只能靠 P2 融合 pass (A-tiled, 上游 rms/silu+quant 直接产 fragment-tiled fp8)
+ 自写 GEMM (radiance 结构) 解决。P2 之前先把 radiance_mxfp4_fp8.hip 的
atiled kernel + fused quant 移植路径写清 (P2 计划见下)。

## P2-3 atiled 状态 (2026-09-30, 已收口)
- atiled 谜题真相: 隔离台 iso_at_main 读 AT 只读 M*K, fragment buffer 实际
  Mt*Kt*256 -> 后半未初始化 -> 全 NaN/同点假失败。修尺寸后真实集 maxabs=0.000000 一次通过。
  教训: 隔离台自身要有 folded 对照先证明台子活着。
- 集成完成: quant kernel 输出 fragment-tiled (L0 公式), gemm 入口 launch_at_f32
  (M>=2048&&N>=128 -> TN4/LBK64 否则 TN2/LBK128); pp2048 2136, pp512 1930。

## P2-4 B1 swiglu+quant 融合 (2026-09-30, 完成)
- 落点: ggml-cuda.cu GGML_OP_GLU case 里先试 ggml_cuda_try_swiglu_quant_fused
  (条件: src0/src1 皆 MUL_MAT 且 weight type39, N%64==0, M>=min_m, 行连续);
  融合 kernel = silu(gate)*up 写 y (表达式顺序与 unary_gated_op_kernel<op_silu> 一致,
  保位精确) + per-token e4m3 直写 fragment-tiled q (行哈希 memoization)。
- 注册表: y 指针 -> per-K 持久 q 缓冲; graph_compute 开头清空 (单图内 glu 必在
  其 down 前执行, 依赖链覆盖旁路, key 复用最坏退化为 hash 跳过, 无错值路径)。
- mmq.cu down fast path: 先查注册表命中则跳过 quantize pass, 否则原路 hash-quantize。
- 收益实测: pp2048 2136 -> 2160.8 (+1.2%), pp512 1958。比预估小, 账: fused 只消掉
  quantize 回读 y 的 f32 流 (~9GB/轮); gate/up 读与 y 写是 f32 中间物固有成本,
  要整链消掉需 bf16 进 bf16 出的 down 直读 gate/up (radiance silu_mul_quant 同级融合),
  需 graph 级侵入, 暂不做。
- 红线全绿: PPL 8-chunk 逐位一致 6.9328; MUL_MAT 1359/1359; SWIGLU 24/24;
  ENGAGE 3968 (down 全部走 fused 查表路径)。GGML_RAD_DISABLE 同时关 gemm+glu hook。

## radiance 方案全量 MXFP4 GGUF (完成, 见 PP 战线总账)
路线: modelscope 下载 Qwen/Qwen3.8-27B bf16 (55.6G, 已完) -> llama.cpp-quantize
convert --outtype bf16 (无损基底, 866 tensors, text-only) -> fork llama-quantize
--ftype 位置参数 MXFP4 + --tensor-type-file 正则豁免:
  ^token_embd\.weight$=bf16
  ^output\.weight$=bf16
  ^blk\.64\.nextn\..*\.weight$=mxfp8   (radiance MTP 头是 fp8_e4m3; fork 有 MXFP8)
radiance 全表: 所有 linear 投影(gate/up/down/qkv/o/in_proj_a/b/z/out_proj/k/v)=MXFP4,
norm/conv/A_log/dt_bias=BF16 原样, embed/lm_head=BF16。fork 的 quantize 对这些的
自动行为: 1D(norm/bias) 与 4D(conv1d) 不量化; ssm_a(1D) 跳过。与 radiance 一致。
产物: /media/seirin/HDD500G/gguf-rad/Qwen3.8-27B-Rad-MXFP4.gguf (预计 ~19-20G)
注意: 混合基线 GGUF 里 qkv 拆 q/k/v 且 attn_qkv 为 type 12; 新 GGUF 中
linear_attn 层的 in_proj_qkv 会映射成 attn_qkv -> type 39 -> fast path 覆盖全部注意力。
验证链: llama-bench pp/tg + PPL(corpus.txt 8 chunks, 新基线=自己 f16 全量 PPL) +
ENGAGE 日志计数。

## P2 移植可行性核查 (2026-09-30, 结论: 完全可行, 直接移植)
模型结构修正: Qwen3.8-27B 是 DENSE 模型 (qwen3_5, 65 层, 无 MoE!)
  之前"Mixed 126 张量含 MoE"理解有误 -- 全部是标准 mul_mat, 无 mul_mat_id 障碍。
张量语义核查 (类型 39 = GGML_TYPE_MXFP4, E8M0 scale):
  llama dequant: kvalues_fp4[nib] * ue? -> 值 = (2*e2m1[nib]) * 0.5 * 2^E(e byte)
  radiance:      kMag 折叠 e4m3 * 2^(Ws-Wref) * 2^Wref = e2m1[nib] * 2^Ws
  => 完全等价: Ws = E(e byte), Wref = 行 max(E), qs 平面直拷 nibble 零转换!
  注意: 仅类型 39 (MXFP4/E8M0) 兼容; 类型 46 (MXFP4_E4M3, UE4M3 scale) 不兼容
  (UE4M3 有 3bit 尾数不是纯 2 的幂, kMag 指数相减折叠失效) -- 模型里没有 46。
repack 成本: llama [N,K/32]x17B 交错 -> radiance W[N,K/2] + Ws[K/32,N] + Wref[N]
  字节不变 + ~2.6MB Wref (65 层), 加载时设备 kernel 一次完成, 无显存翻倍。
激活量化: radiance 需 per-token fp8 + scale; 写 per-token amax 量化 kernel
  (对应 radiance_add_rms_quant 的量化部分, P2 融合本来就要写)。
epilogue: radiance 写 bf16 -> 改 f32 (一行)。
分阶段验收:
  阶段1: standalone harness 编译 radiance folded kernel (WPERM=false 起),
    repack 单张量, 与 MMQ 输出逐元素对比 (容差 fp8 级) + 34816 形状测 TF,
    目标复现 ~190TF。
  阶段2: 接入 llama.cpp mul_mat dispatch (类型 39 prefill 大 M 路径),
    repack 挂加载流程, PPL 8-chunk 门 (<= 基线+1%), pp2048 目标 1500+。
  阶段3: A-tiled 升级 (+13%) + fused rms/silu+quant, 目标 pp2048 1800+。
  阶段4 (可选): decode kernel (TG 侧), TG 现为带宽墙, 优先级最低。
风险: dispatch 拦截条件要保守 (仅 type39 + RDNA4 + M>=513 + 非视图张量),
  fallback 回 MMQ 保证兼容; 语料 PPL 门兜量化误差。

## PP 战线总账 (2026-09-30, ROCm0 R9700, fa on, ngl99, -p 2048 -n 0/-p 512 -r 3)
| 阶段 | pp512 | pp2048 | 说明 |
|---|---|---|---|
| 基线 (混合 GGUF, MMQ fp8) | 1067.3 | 1134.3 | |
| P2-1 radiance folded 移植 (混合 GGUF, 部分 ENGAGE) | 1132.8 | 1213.6 | J 加宽/双缓冲/occupancy 均否证后的真解 |
| P2-2 全量 radiance 方案 GGUF (503 mxfp4 全覆盖) | 1663-1703 | 1813-1863 | +56~64% |
| P2-2b quantize v2 (行哈希记忆+float4 向量化) | 1703 | 1863.6 | gate/up 共享激活 skip 第二次量化 |
| P2-3 atiled (fragment-tiled A, 0 LDS staging) | 1929.8 | 2136.0 | +88% vs 基线 |
| P2-4 B1 swiglu+quant 融合 | 1938.4 | 2128.5 | 与 P2-3 同水位 (融合收益被噪声覆盖) |
| **A2 chunked GDN bf16 WMMA (状态常驻累加器)** | **2165.4** | **2388.8** | **+12.2%; 相对基线 +125%** |
tg128 = 28.26 (4.5bpw 带宽红利, +8%)。数值门: PPL 8-chunk **6.9318** (A2 后; A2 前 6.9328,
漂移 0.014%, 门限 1%); fast path ON/OFF 逐位一致; 混合基线 6.4652 的差是全量 mxfp4 的量化代价,
与 radiance checkpoint 同源; MUL_MAT 1359/1359; atiled 红线复测在 bash-120。
A2 后 llama-cli 贪心输出 ON/OFF **逐字节一致** (1569 B, 唯一差异是性能统计行)。
对照 vLLM radiance: prefill 2721 t/s @30k ctx (不同负载口径, 不可直接比) —— 我们 pp 数字已同量级。
A2 后的 rocprof 分解 (pp2048 单次 ≈ 863ms): radiance GEMM 53%, **GDN 8.2%**, swiglu_quant 6.4%,
quantize_tokens 5.0%, flash_attn 2.8%。GDN 已从改造前的 ~18% 降到 8.2%, 不再是主要矛盾。

### atiled 破案记录 (教训级)
- 之前"atiled iso 布局 4 假设全败"的结论被 harness 自身 bug 污染:
  iso_at_main 读 AT 只读 M*K=40960B, 而 fragment buffer 是 Mt*Kt*256=81920B ->
  后半未初始化 -> 全 NaN / 同点假失败。修尺寸后 atiled 真实集 maxabs=0.000000 完美通过。
  教训: 隔离台自己也要 golden 校验 (先跑 folded 对照确认台子本身活着)。
- fork 集成: quant kernel 输出 fragment-tiled (L0 公式, u32 store 在 8B group 内),
  gemm 入口换 launch_at_f32 (M>=2048&&N>=128 -> TN4/LBK64, 否则 TN2/LBK128, 对齐 radiance);
  缓冲按 (Mcap+15)&~15 行分配 (padding 行读垃圾但 epilogue 丢弃, WMMA 不跨 m 行污染)。

## A2: 完整复刻 radiance GDN (WMMA 张量核, 用户 2026-10-01 指示"做到底, 完整复刻")

### 目标形态 (radiance gdn_chunk_scan_k128_v128_c64_bf16 的等价物)
- 几何: KD=128, V=128, C=64 (radiance 同款), 操作数 bf16, 累加器 f32
- 两 kernel 架构 (关键: 把"难看的"小输出相位移出串行核):
  * **K1 kkt_solve**: grid = (n_chunks, H, n_seqs) —— **完全并行, 无序列依赖**。
    载入 k[C][KD] -> bf16 LDS; 用 WMMA 算 Gram KK^T (C=64 -> 4x4 个 16x16 tile, 4 warp 各 4 tile,
    归约 8 步); LDS 内构造 M = I + strict_lower(beta_t (A_t/A_s) KK^T) 并求逆 (分块: 4 个 16x16
    对角块 + 块回代, 避免 64 步串行); 输出 A=(I+L)^-1 为 bf16 到 HBM (T x H x 64, ~12.6MB@T=2048)。
    gram 用 fp32 LDS 累加 (radiance 注释: "the fp32 gram stays in LDS")。
  * **K2 chunk_scan**: grid = (H, n_seqs) 或多切片; block 256 线程 (8 warp) ——
    **状态 S0 (128x128 f32) 常驻 WMMA 累加器**: 64 个 16x16 tile / 8 warp = 每 warp 8 tile
    = 每 lane 64 f32 累加器; h/w/u/v_new 一律不进 HBM (radiance 原话: 赢就赢在这)。
    每 chunk: 
      - 载入 k,q (bf16 LDS), 由 A_t/beta 构造 RHS 与 WY 表示
      - k.S0 / q.S0: 需把 S0 累加器转成 B 操作数 -> 每 chunk 经 LDS 往返一次 (bf16, 32KB),
        这是"状态常驻累加器"的固有代价, 但远小于 HBM 往返
      - D = A @ RHS (WMMA), o = scale*(A_t q.S0 + tril(QK^T ratios) D) (WMMA)
      - 状态更新: acc(S0) = A_C*acc(S0) + K^T @ (diag(A_C/A_s) D) —— 直接累加进累加器
  * LDS 预算 (C=64, KD=V=128, bf16 操作数): k 16KB + q 16KB + S0 往返 32KB + D 32KB(f32)
    + A 8KB -> 需精打细算, 可能要 C=32 或 V 切片 (待定, 先按 C=64 试)。

### 数值门
- bf16 操作数会改变舍入: PPL 门从"逐位一致"放宽为**漂移 < 1%**, 且必须与 GGML_GDN_CHUNKED=0
  的 f32 recurrent 路径对拍 (GDN 单测已有 tolerance 机制; 最终 39/39)。
- 逐步验证: K1 的 A 矩阵先与 f32 numpy 参考对拍; K2 再与 recurrent 输出对拍。

### 已就绪的基建 (fork 内)
- mma.cuh: `__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12` (及 f16 变体)
- radiance-gemm.cu: gfx12 WMMA builtin 的实拍用法 (fragment 布局/launch_bounds 参考)
- 现有 chunked f32 版 (opt-in) 提供: 完整索引/布局语义 + 已验证的数学 + 测试脚手架

### 落地顺序 (每步都可独立验证)
1. K1 kkt_solve (含 WMMA Gram + 分块求逆), 与 numpy 对拍 A 矩阵
2. K2 chunk_scan (状态常驻累加器), 与 recurrent 对拍输出+末态
3. 接入 dispatch, 39/39 单测 (原 36 例 + 补 3 个 head_size=128 且 n_tokens>=64 的真命中例)
4. 端到端: PPL 漂移门 + pp2048/tg128 bench + llama-cli 真实输出
5. 若超越 recurrent: 默认开启; 否则保留 opt-in 并记录结论

### A2 进展: RDNA4 bf16 WMMA fragment 布局已实测确认 (2026-10-01)
复刻的第一个未知量已解 —— 探针在 .49:/tmp/wmma_probe.hip, wmma_probe2.hip, wmma_verify.hip:
- 布局 (含非对称判据打破 A/C 转置二义性):
    A 操作数: lane L, slot j -> A(row = L%16,        col = 8*(L/16)+j)
    B 操作数: lane L, slot j -> B(row = 8*(L/16)+j,  col = L%16)
    C 累加器: lane L, slot j -> C(row = 8*(L/16)+j,  col = L%16)
- 端到端验证: 16x16x16 bf16 WMMA vs CPU 参考 maxabs = 0.0000, 256/256 元素正确
  -> "LAYOUTS CONFIRMED"。A 与 B/C 的编码不同 (B 为转置约定), 这点如果猜错整个 kernel 全废。
- builtin: __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(bf16x8 a, bf16x8 b, f32x8 c)
  (fork mma.cuh:1327 同款; 每 lane 8 bf16 A + 8 bf16 B + 8 f32 C)

### A2 结果 (2026-10-01 完成): 单 kernel 形态胜出, pp2048 +12.2%

落地形态与上面的"两 kernel 架构"预案**不同**: 最终是**单 kernel**, K1/K2 没有拆开。原因见下。

**实测 (R9700, fa on, ngl99, r3)**

| 指标 | A2 前 (recurrent) | A2 后 (WMMA chunked) | 变化 |
|---|---|---|---|
| pp512 | 1938.4 | **2165.4** | +11.7% |
| pp2048 | 2128.5 | **2388.8** | **+12.2%** |
| tg128 | 28.26 | 28.26 | 0 (n_tokens<64 自动回退) |
| PPL 8-chunk | 6.9328 | 6.9318 | 0.014% |
| GDN 单测 | 39/39 | 39/39 | 容差 5e-5 (bf16) |
| CLI 贪心输出 | - | - | 与 recurrent 逐字节一致 |

**最终设计 (gdn_chunked_wmma_cuda\<C=32\>, 一 workgroup = 一个 (seq, head), 8 warp)**

- **状态常驻累加器, 零 LDS 往返**: warp w 拥有 v 列块 [16w,16w+16) 的全部 128 行, 即 8 个
  16x16 tile = 每 lane 8 个 f32x8。这正是原预案里"每 chunk 把状态经 LDS 往返转 B 操作数"
  —— **最终发现根本不需要**: gfx12 的 **C 累加器布局与 B 操作数布局逐 slot 相同**
  (都是 row=8*(L/16)+j, col=L%16), 所以状态转 B 只需一次 bf16 convert, 既不过 LDS 也不过 shuffle。
  这是整个 kernel 成立的关键, 也是它比 f32 版快一个数量级的根因。
- **K1/K2 不拆**: kkt_solve 的输出 A=(I+L)^-1 只有 C*C 个数, 拆成独立 kernel 要把它写回 HBM
  (~12.6MB@T=2048) 再读回, 而单 kernel 内 s_gr/s_x 两块 32x36 f32 (4.6KB) 就装下了, 且省掉
  一次 kernel 启动与一轮 k 重载。radiance 拆 K1/K2 是因为它要 C=64 且用独立 grid 换并行度,
  我们 C=32 单 kernel 的 grid 已经够 (每层 48 workgroup)。
- 每 chunk 阶段 (每阶段一次 __syncthreads, 共 6 次):
  `载入 k/q/g/b -> gram(K K^T)+qk(Q K^T) -> k.S0/q.S0(状态直接转 B) + M 构造与求逆
   -> RHS=beta(v-A_t k.S0) + s_inv/s_qw -> D=(I+L)^-1 RHS -> o 与状态累加`
- **数值稳定**: A_t/A_s 一律写成 `expf(c_t - c_s)` (c = gate 累积**和**, f32), 绝不写成
  `A_t/A_s` 的商。gate 强负时 A_t 下溢到 0, 商形式会得到 0/0 = NaN 让整个 chunk 报废。
  测试的 gate 是 [-20,-1e-4] 均匀分布, 这个修复是它能在单测里通过的前提。
- LDS 预算 38.1KB (C=32, KD=V=128): s_k 8448 + s_q 8448 + s_kt 9216 + s_gr 4608(f32, gram->M)
  + s_inv/s_qk/s_qw 各 2304 + gate 4x128。`s_q` 在 q.S0 之后死亡, 被复用为求逆的输出 scratch
  `s_x` (见下方"原地求逆"陷阱)。
- 实测效率: 每 chunk 每 workgroup 512 个 WMMA (=2.1M MAC), 每 workgroup 约 1880 cycle,
  即 WMMA 吞吐的 ~85%。**继续压 GDN 的空间已经不大**。

**踩坑记录 (全部是"不跑就发现不了"的类型)**

1. **`RDNA4` 宏在 host pass 不可见**。它是 `vendors/hip.h` 里由 `__gfx1201__` 定义的
   **device pass** 宏; 多 arch 编译时 host TU 只编一次, `#if defined(RDNA4)` 恒假 ->
   dispatch 直接走 `GGML_ABORT("fatal error")`。正确做法: **kernel 在所有 arch 都定义**
   (wmma builtin 用 `#if defined(RDNA4)` 单独包住, 非 RDNA4 返回 c), host 侧只做
   `#if defined(GGML_USE_HIP)` + **运行时 `GGML_CUDA_CC_IS_RDNA4(cc)`** 门控。
2. **`if (tid < C*C)` 在 C=32 时只覆盖 1/4**。workgroup 只有 256 线程而 C*C=1024,
   于是 s_inv/s_qw 只写了前 8 行, 其余是 LDS 垃圾 -> D 出现 1e31 量级的爆值。
   旧 f32 版 C=16 时 C*C=256 恰好等于线程数, 把这个坑盖住了。改成 `for (idx=tid; idx<C*C; idx+=NTH)`。
3. **原地三角求逆是错的**。`X[i][j] = -sum_{m=j}^{i-1} M[i][m] X[m][j]` 需要读 **M**,
   但 in-place 时 `M[i][m]` 已被 lane m 覆写成 X[i][m]。必须双 buffer。旧 f32 版一直是错的
   (它同样 in-place), 只是从没被执行过。修法: 求逆输出到 s_x (=死掉的 s_q)。
4. **上三角不清零**。求逆 lane j 的 M 构造循环若写成 `for (i = j; ...)`, 上三角会留着 gram
   的残值, D 里把上三角垃圾算进去 -> d[1][0] = 2.7e31。改成 `for (i = 0; ...)`, lane j
   写满整列 (i<j 写 0, i==j 写 1)。
5. **"36/36 通过"曾是空跑**。原测试集里 head_size==128 的用例 n_seq_tokens 只有 1 和 4,
   **全部低于 chunked 的 64 门槛**, 所以 A1 三代 f32 kernel 与 A2 早期版本从来没被真正执行过,
   报 OK 是测试框架的假通过。已补 3 个用例 (head_size=128, n_seq_tokens = 64/100/512),
   现在 39 个。**教训与 atiled 破案同源: 先确认测试真的选中并执行了目标路径。**
6. **容差要按路径分**: bf16 chunked 实测漂移稳定 ~7e-6 且**不随 token 数增长**
   (说明是单次量化误差, 无累积), 但 f32 的 1e-7 绝对门不可能达到。override
   `max_nmse_err()` 仅在 `head_size==128 && n_seq_tokens>=64 && GGML_GDN_CHUNKED!=0`
   时放宽到 5e-5 —— 精确对齐 chunked 的真实启用条件, 不掩盖 recurrent 路径。
7. **llama-cli 必须加 `-st`**。否则生成完会进交互模式空等 stdin, 看起来像"卡死/巨慢"
   (实测差 100 倍以上)。这不是性能问题, 是参数问题。

**A1 (f32 chunked) 的最终处置**: 整个删除。它比 recurrent 慢 2-3x, 且求逆有上述 bug,
WMMA 版全面取代它。备份在 `/media/seirin/HDD500G/backups/llamacpp-fast-20260930-215341/gdn-versions/`。

### 落地顺序 (原始预案, 实际执行见上方 "A2 结果")
预案是 K1/K2 双 kernel + 状态经 LDS 往返; 实测证明单 kernel + 状态零往返更好。以下三步是
原计划, **保留供理解设计意图, 不要照抄** (真正的实现形态见 "A2 结果" 一节):
1. ~~K1 kkt_solve: grid (n_chunks, H, n_seqs) 全并行; 载 k[C][KD] -> bf16 LDS;
   WMMA 算 Gram (C=64: 4x4 个 16x16 tile, 4 warp 各 4 tile, KD/16=8 步归约);
   LDS 内 M = I + strict_lower(beta_t (A_t/A_s) KK^T), 分块求逆(4 个 16x16 对角块+回代);
   输出 A=(I+L)^-1 bf16 -> HBM。~~
2. ~~K2 chunk_scan: 状态 (128x128 f32) 常驻累加器 (8 warp = 每 lane 64 f32);
   每 chunk 需把状态经 LDS 往返一次转成 B 操作数 (k.S0 / q.S0), 用上面的 B 编码;~~
   状态更新直接累加进累加器; h/w/u/v_new 不进 HBM。**这部分成立并已实现。**
3. 逐步对拍: K1 的 A vs numpy; K2 输出+末态 vs recurrent; 然后 39 单测 + PPL 漂移门 + bench。

## GDN 剩余优化空间 (留档 2026-10-01; 用户指示先转 GEMM 专项, 后续有机会再榨)

A2 后 GDN 占 pp2048 的 8.2% (71ms/863ms), 每 chunk 已达 WMMA 吞吐 ~85%。已知但**尚未做**的方向:

1. **并行度不足 (最可能的真因)**: 每层只启动 48 个 workgroup (每 (seq,head) 一个), 32 CU 上
   仅 1.5 波, 波内没有第二个 workgroup 填延迟空洞。加深 grid 需要跨 workgroup 的状态依赖
   (scan 结构) 或让一个 workgroup 处理多个 head (状态 LDS 装不下)。
2. **occupancy 未实测**: LDS 38.1KB; 若 RDNA4 每 CU 是 128KB 则可驻留 3 个 workgroup,
   否则只有 1-2 个。`-Rpass-analysis=kernel-resource-usage` 报 VGPR=26 明显失真 (st[8] 就占
   64 个), 该工具对本 kernel 不可信, 需用 rocprof 的 occupancy 计数器实测。
3. **状态加载/存储非合并**: 每 lane 32B 连续但跨 16 个不同行, DRAM 事务效率约 50%
   (有效 64KB -> 实际 128KB)。总量 29 层 x 48 head x 128KB x 2 = 356MB ≈ 0.6ms —— **不是瓶颈**,
   不值得为它加 LDS staging。
4. **C=64**: chunk 数减半、同步次数减半、WMMA 的 K 维更长, 但 per-token MAC 从 65536 涨到
   81920 (+25%), 且 LDS 需求约 113KB 超预算。若要试先做 LDS 复用: s_kt (17.4KB) 改为从 s_k
   转置读 (每 fragment 8 次标量 LDS), s_qk/s_qw 原地复用, s_inv 写回 s_gr 区。
5. **打破依赖链**: k.S0 / q.S0 的 8 个 WMMA 累加到同一寄存器 (链长 8), 可拆成 2 个累加器再相加。
   代价是多 32 个 VGPR, 有溢出风险, 需先确认 VGPR 实际占用。
6. **求逆并行化**: 现在 warp 0 的 32 个 lane 各扫一列 (最慢的 lane 496 次 FMA 串行), 其余 7 个
   warp 空等。可改分块求逆 (4 个 16x16 对角块 + 块回代), 块乘用 WMMA。

**预期上限**: 即使把 GDN 压到 30ms (从 71ms), 也只省 41ms -> pp2048 +5.5%。
**优先级明确低于 GEMM (53%)**, 故先转 GEMM 专项 (见 `design-gemm-rdna4.md`)。

## 进度
- [x] 环境勘察 / clone / 编译
- [x] 基线表 + rocprof 剖析 + 混合量化字节分布
- [x] vLLM radiance TP1 单卡实测 (镜像 + checkpoint + 服务, 数字见上)
- [x] P1-1/P1-2/P1-2b~e 全部收官 (MMQ ~110TF 到顶)
- [x] P2-1 radiance folded kernel 移植 + dispatch + 数值修复 (iso 台裁决)
- [x] P2-2 radiance 方案全量 MXFP4 GGUF (503 张量 type39, 17G, 5.35bpw)
- [x] P2-2b quantize v2 (行哈希记忆 + float4)
- [x] P2-3 atiled 集成 (pp2048 2136, +88%; 谜题真相=隔离台自身尺寸 bug)
- [x] P2-4 B1 swiglu+quant 融合 (pp2048 2160.8)
- [x] A1 GDN chunked f32 (v1/v2/v3 全败, 数学正确性能负, 已整个删除由 A2 取代)
- [x] **A2 GDN WMMA 复刻 (完成: pp2048 2388.8, +12.2%; PPL 漂移 0.014%; 已设为默认)**
- [ ] P4 TG 侧 (MTP/draft; 当前 28.26)

---

# 交接记录 (2026-10-01, 上下文压缩后续作必读)

## 1. 环境与命令 (全部实测可用)
- 远端: `ssh seirin@192.168.31.49`; 仓库 `/media/seirin/HDD500G/llama.cpp-gpu-tuned`
  (HEAD 7a5408676 + 未提交改动; 个人 fork, **不 commit/push/PR**)
- 模型: `/media/seirin/HDD500G/gguf-rad/Qwen3.8-27B-Rad-MXFP4.gguf` (17G, 503 张量 type39)
- 编译: `cmake --build build --target llama-bench llama-perplexity llama-cli test-backend-ops -j 48`
- GDN 单测: `./build/bin/test-backend-ops -b ROCm0 -o GATED_DELTA_NET test` (36 例)
- GDN 性能: `./build/bin/test-backend-ops -b ROCm0 -o GATED_DELTA_NET perf`
  (关注形状 32 heads/head_size=128/n_seq_tokens=512 —— 接近真机 H=48)
- 端到端: `./build/bin/llama-bench -m <gguf> -dev ROCm0 -p 512,2048 -n 128 -r 3 -fa 1`
- PPL 门: `./build/bin/llama-perplexity -m <gguf> -f /media/seirin/HDD500G/corpus.txt -dev ROCm0 -c 4096 -ngl 99 -fa 1 --chunks 8`
- 开关: `GGML_GDN_CHUNKED=0` (**关闭** chunked GDN, 回退到 recurrent; 默认开), `GGML_RAD_DISABLE=1` (关 radiance GEMM),
  `GGML_RAD_PREFILL_MIN_M` (默认 256), `GGML_RAD_DEBUG=1` (ENGAGE 日志)
- CLI 对比**必须加 `-st`**, 否则生成完进交互模式空等 stdin (看起来像卡死):
  `./build/bin/llama-cli -m <gguf> -dev ROCm0 -ngl 99 -fa 1 -st --temp 0 -n 64 -p "..."`

## 2. 当前最优状态 (2026-10-01 最新; 必须保持不退化)

**注: 本节数字已含 GEMM 专项的成果 (见 `design-gemm-rdna4.md`)。GDN 单独交付时是 2165/2388.8。**

| 指标 | 值 | 备注 |
|---|---|---|
| pp512 / pp2048 | **2279.8 / 2569.2** (MXFP8 版) | 基线 1067/1134, **+127%**。**须带 `-ub 2048`** |
| tg128 | **29.89 (MXFP8 版) / 28.26 (原版)** | 带宽红利; chunked 只作用于 n_tokens>=64, 不碰 decode |
| TG + 投机解码 (DFlash2) | **50.7 t/s** (确定性测量) | 累计 **1.79x**; 见 `design-mtp-rdna4.md` |
| PPL 8-chunk (MXFP8 版) | **6.9633** | 原版 6.9318 (+0.45%, 来自 lm_head 量化) |
| GDN 单测 | 39/39 (默认路径) + 39/39 (`GGML_GDN_CHUNKED=0`) | chunked 容差 5e-5 |
| MXFP4 MUL_MAT / CONCAT | 60/60 / 177/177 | |
| 真实 CLI | chunked vs recurrent 贪心输出逐字节一致 | 1569 B, 仅性能统计行不同 |
**备份**: `/media/seirin/HDD500G/backups/llamacpp-wmma-gdn-final-20260930-224947/`
(gated_delta_net.cu + test-backend-ops.cpp, md5 b07f5d989a409d418b615ec8674435f9)。
A1 之前的整体备份仍在 `llamacpp-fast-20260930-215341/`。

## 3. GDN chunked f32 三代实测 (已废弃: 结论是"必须上 WMMA", 不是"chunked 不行")
**注意: 本节的三代 f32 kernel 已整个删除, 由 A2 的 WMMA 版取代。留档是为了记住"形态错了, 不是
方向错了" —— 同样的问题 chunked 化 + WMMA 后 pp2048 涨 12.2%。**
形状 32heads×128×512tok: recurrent(默认,最快) **614.36 μs**; v1 1803 / v2 1283 / v3 1457 μs。
三代号称 36/36 单测通过 —— 但事后查明 head_size=128 的用例 n_tokens 只有 1 和 4, **全部低于
chunked 的 64 门槛, 即那三代内核从未真正被执行过**, 且它们的原地求逆本身是错的。三条当时的证据:
- VGPR 95, spill 0 (`clang -Rpass-analysis=kernel-resource-usage`) -> 非寄存器压力
- 0.70 T MAC/s = FP32 峰值(19.7T) 3.5%; LDS @2B/MAC 理论界 26ms/pp2048 -> 非带宽墙
- 真因: workgroup 数 = H×(S_v/VS) = 128 个 (4-head 用例仅 16), 每 workgroup 4 warp,
  LDS 47KB -> 每 CU 2 个 = 8 warp/CU, 藏不住 LDS 30-40 cycle 延迟;
  **形状错配**: 一个 warp 4x4 tile 只算 16 输出, 16x32 输出面喂不饱 128 线程;
  拆小 VS -> tile 更少 + Gram/QK 冗余翻倍 = 死结。**必须 WMMA (radiance 同款)。**

## 4. A2 WMMA 复刻: 已确认的关键事实 (勿再重新推导)
### 4.1 RDNA4 bf16 WMMA fragment 布局 (探针实测 + CPU 对拍确认)
```
A 操作数: lane L, slot j -> A(row = L%16,       col = 8*(L/16)+j)
B 操作数: lane L, slot j -> B(row = 8*(L/16)+j, col = L%16)     <- 转置约定, 与 A 不同!
C 累加器: lane L, slot j -> C(row = 8*(L/16)+j, col = L%16)
```
- builtin: `__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(bf16x8 a, bf16x8 b, f32x8 c)`
  (fork `mma.cuh:1327` 同款封装; 每 lane 8 bf16 A + 8 bf16 B + 8 f32 C)
- 验证: 16x16x16 vs CPU maxabs=0.0000, 256/256 正确; 探针文件 `.49:/tmp/wmma_probe{,2}.hip`, `/tmp/wmma_verify.hip`
- **踩坑记录**: A/C 编码互换会产生完全相同的观测 (转置二义性), 必须用非对称判据
  (A 放 lane16 slot0 + B 放 lane0 slot0, 看乘积是否为零) 才能打破。布局猜错则 kernel 全废。

### 4.2 数学 (numpy 已对拍 f32 1.19e-7; 与 radiance kkt_solve 定义逐项一致)
C = chunk 长度, A_t = Π_{r<=t} exp(g_r):
```
L[t][s] = beta_t (A_t/A_s) (k_t . k_s),  s < t        # (I+L) 严格下三角
RHS[t]  = beta_t (v_t - A_t (k_t . S0))
D       = (I+L)^-1 RHS
o_t     = scale (A_t (q_t . S0) + Σ_{s<=t} (A_t/A_s)(q_t . k_s) D_s)
S0     <- A_{C-1} S0 + Σ_s (A_{C-1}/A_s) k_s ⊗ D_s
```
radiance 的 kkt_solve 定义 (radiance_gdn.py:440): `A = (I + strict_lower(diag(beta) K K^T e^dg))^-1`,
fp32 gram 留 LDS, 只输出 A (T,H,CHUNK) bf16。GDN 几何: head_k=head_v=128, C=64。

### 4.3 K2 设计要点 (状态常驻累加器) —— **最终实现与原预案的差异**
原预案 (下面 4 条) 有两处被实测推翻, 以实际代码为准:
- ~~每 chunk 需把状态转成 B 操作数 → 经 LDS 往返按 k-step 切片~~ -> **根本不需要 LDS 往返**。
  gfx12 的 C 累加器布局与 B 操作数布局逐 slot 相同, 状态转 B 只是一次 bf16 convert (见 A2 结果)。
- ~~LDS 预算 58KB~~ -> 实测 **38.1KB**, 因为省掉了状态切片往返 (8KB) 且 s_q 死后复用为求逆 scratch。
- 保留成立的部分: 一 workgroup = 一个 (seq, head), 8 warp; warp w 拥有列块 16w..16w+15 的全部
  KD 行 (= 8 个 16x16 tile = 每 lane 64 f32); 状态更新用 A = K^T 直接累加进累加器;
  h/w/u/v_new 一律不进 HBM (radiance 原话: "the win comes from" 这一点)。

### 4.4 验证链 (已全部执行通过)
1. Gram/inverse 单元: 与 numpy 的 `(I+L)^-1` 对拍
2. K2 整体: 与 recurrent kernel 输出+末态对拍 (`GGML_GDN_CHUNKED=0` vs 默认, 同形状)
3. `test-backend-ops -o GATED_DELTA_NET test` **39/39** (已补 head_size=128 且 n_tokens>=64 的用例,
   否则 chunked 路径是 0 覆盖的假通过)
4. 端到端: PPL 6.9328 -> 6.9318 (漂移 0.014%) + pp2048 2388.8 + llama-cli 贪心输出逐字节一致
5. 已超越 recurrent (pp2048 +12.2%), 故 chunked 为**默认开**; `GGML_GDN_CHUNKED=0` 回退

## 5. 已知陷阱 (避免重复踩)
- **`RDNA4` 宏只在 device pass 可见**, host 侧用它做 `#if` 会静默走错分支 -> host 用运行时 cc
- **`if (tid < N)` 当 N > 线程数时静默只覆盖一部分** -> 必须写成 `for (idx=tid; idx<N; idx+=NTH)`
- **三角求逆不能 in-place** (需要读 M 而写 X), 且上三角必须显式清零
- **`A_t/A_s` 必须写成 `expf(c_t - c_s)`** (c 为 gate 累积和), 否则强负 gate 下 0/0 = NaN
- `test-backend-ops -p <filter>` 的正则是 regex, `*` 是量词, 含 `(`/`)` 的模式要写对
- `test-backend-ops` 可能 0/0 空跑仍报 OK -> **必须确认 selected 数 > 0 且形状真的命中目标路径**
- `llama-cli` 不加 `-st` 会进交互模式空等 stdin (像卡死)
- `pkill -f "bin/llama-cli"` 会匹配到 ssh 命令行自身把自己的会话杀掉 -> 用 `pgrep -x llama-cli`
- `hipcc ... | head` 会因 SIGPIPE 杀掉编译 → 用 `grep -E "error"` 并保留退出码
- llama-bench 数值有 ±100-170 抖动 → 判断退化要看多次或换微基准 (test-backend-ops perf)
- 改完 .cu 必须重编再测 (曾因 stash/pop 静默回退二进制浪费数轮)
- `__shfl_xor_sync` 在 HIP 是 4 参宏 (mask, var, laneMask, width)
- 隔离台自身要有 golden 对照 (atiled 谜题就是被 harness 自己的尺寸 bug 污染了数轮)

## 6b. `keep_rs=true` 变体的 ISA 账 (2026-10-01, 供参考, 未改代码)

投机解码走的是 `<S_v=128, KDA=false, **keep_rs=true**>` (状态快照), 与纯 decode 的
`keep_rs=false` 是**不同实例**。两者 ISA:

| 实例 | 指令数 | 字节 | ds_bpermute | wait_dscnt | global_ld | fp |
|---|---|---|---|---|---|---|
| `<128,false,false>` (纯 decode) | 283 | 1552 | 10 | 11 | 15 | 13 |
| `<128,false,true>` (spec decode) | **291** | 1596 | 10 | 12 | 15 | 13 |

**`keep_rs` 只多 8 条 (+2.8%)** —— 快照写回几乎免费。token 循环体 `0x161b0..0x16538` = **153 条**。

循环体构成 (149~153 条):

| 类别 | 条数 | 占比 |
|---|---|---|
| **wait/delay** | **51** | **34%** |
| 浮点运算 | 27 | 18% |
| 访存 (global+LDS) | 26 | 17% |
| 地址算术 | 16 | 11% |

**两个关键观察:**

1. **10 条 `ds_bpermute_b32`, 每条紧跟 `s_wait_dscnt 0x0`** —— 源码两次 `warp_reduce_sum`
   被降级成 LDS 往返 (`__shfl_xor_sync` -> `ds_bpermute`), 5 步全串行, 且**在关键路径上**
   (`kv` 归约 -> `delta` -> 状态更新 -> 下一 token 依赖它)。每 token 2 条串行归约链。
2. **token 维度是内核内串行循环** (`for (int t = 0; t < n_tokens; t++)`), 无并行机会 ——
   同序列的 token 间有真依赖。已有的 `gdn_chunked` 路径要求 `n_tokens >= 64`,
   spec decode 的 n_tokens=8 永远够不到。

**实测缩放** (同变体, 只改 verify batch):

| n_tokens | 每次耗时 | 每 token |
|---|---|---|
| 1 (实为 false 变体, 283 条) | 7.41 us | 7.41 |
| 8 | **34.14 us** | **4.27** |

-> **非纯串行**: 8 倍 token 只花 4.61 倍时间, 拟合 `t ≈ 3.59 us (固定) + 3.82 us/token`。

**占用不是问题**: 32 VGPR / 0 LDS, 1536 blocks 已溢满 12 waves (`SQ_WAVES=6144` = 1536 x 4 warp)。
所以瓶颈是**延迟受限** (5.6~58.7 cyc/instr), 不是算力或占用。

**结论: 不改。** 依据 §6: GDN 在 prefill 侧已从 ~18% 降到 8.2%; 在 spec decode 侧实测
**1.64 ms/轮 = 3.8% GPU busy** (48 层 x 34.14 us)。要动就得改归约方式或 token 并行,
收益上限 ~3.8%, 风险与收益不成比例。

复现:
```bash
cd /tmp && cp <build>/.../gated_delta_net.cu.o .
/opt/rocm/lib/llvm/bin/llvm-objdump --offloading gated_delta_net.cu.o   # 解出 gfx1201 ELF
F=$(ls *gfx1201*)
/opt/rocm/lib/llvm/bin/llvm-nm -S "$F" | grep gated_delta_net           # 找 ILi128ELb0ELb1EE
/opt/rocm/lib/llvm/bin/llvm-objdump -d "$F" > gdn.asm
```
注意: 反汇编注释里的地址才是真地址 (`// 000000015F00:`), 行首没有地址。

---

## 6. 下一步 (A2 已完成, 这里是新起点)

A2 收官后 rocprof 分解 (pp2048 单次 ≈863ms): **radiance GEMM 53%** (459ms) > GDN 8.2% (71ms)
> swiglu_quant 6.4% > quantize_tokens 5.0% > flash_attn 2.8%。
GDN 已从改造前的 ~18% 降到 8.2%, 且实测每 chunk 已达 WMMA 吞吐的 ~85% —— **继续压 GDN 上限只有
8%, 不值得**。真正的下一个战场是 radiance GEMM 那 53%, 以及 P4 (TG/MTP, 当前 28.26)。
