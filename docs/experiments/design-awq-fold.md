# design-awq-fold.md - AWQ 通道均衡折叠（免 kernel 改动的量化前校正）

目标（用户批准 2026-10-02）：在不改 radiance/MMQ/MMVQ 任何 kernel、不改 graph 的前提下，把激活离群通道摊平，降低 MXFP4(W) + fp8-e4m3-per-row(A) 双端量化误差。手段 = AWQ/SmoothQuant 对角缩放折叠：
```
x' = x / s   (通过 norm 权重吸收: gamma' = gamma / s)
W' = W * s   (bf16 权重列缩放)
```
数学恒等（f32 舍入误差内），运行时完全无感知。

## 1. 折叠分组（源码确证，qwen35.cpp）

- **G1 pre-attn 组**（norm = `blk.i.attn_norm.weight`，f32 [5120]）：
  - GDN 层（48 个，判据=有 ssm_alpha）：消费者 `attn_qkv` [5120,10240] + `attn_gate` [5120,6144] + `ssm_alpha` [5120,48] + `ssm_beta` [5120,48]
  - full-attn 层（16 个）+ nextn blk.64：仅 `attn_qkv`（GDN 尺寸 10240 / full 尺寸 14336）
  - 证据：`build_qkvz(cur)`、`ssm_beta(cur)`、`ssm_alpha(cur)` 全部吃 attn_norm 输出；`ssm_dt.bias` 在 GEMM 后相加不受影响；beta 的 sigmoid、alpha 的 softplus 作用在 GEMM 输出上（输出恒等）→ 安全。
- **G2 post-attn 组**（norm = `blk.i.post_attention_norm.weight`）：消费者 `ffn_gate` + `ffn_up`（`build_ffn(cur, ffn_up, ffn_gate, ...)` 两路同输入）。`ffn_down` 输入是 FFN 中间激活，**不折**。
- **不碰**：`attn_output`/`ssm_out`（输入是注意力输出非 norm 输出）、`eh_proj`（输入是 enorm/hnorm concat）、`q_norm/k_norm/ssm_norm`（head 内维度）、`conv1d`、embed/out。
- 层范围 blk.0..64 全部 65 个块（nextn 有自己的 attn_norm/post_attention_norm）。

## 2. scale 计算与 α 网格

- 每通道能量：imatrix-f16src.gguf 的 `<layer>.in_sum2`（Σ x² per channel，350 chunks 校准）→ rms_k = sqrt(in_sum2_k / n_tokens)，用 G1 的 `attn_qkv` 条目、G2 的 `ffn_gate` 条目。
- `s = rms^α`，几何均值归一（geomean(s)=1），clip 到 [1/8, 8]。
- α 网格：{0.5（AWQ 经典值）, 1.0（完全均衡）}。选择判据 = **tt-errors 的 im_rmse J**（对 fold 语义不变：im_new = im_old/s²、w'=w·s 使 Σim·e² 与 Σim'·e'² 逐项相等，两模型可直接比）。
- imatrix 配套重写：`in_sum2' = in_sum2 / s²`（fold 后激活真实变化）。

## 3. 实施（.49）

1. f16 源重导出（清理时删了）：convert --fuse-qkv --outtype bf16（后台进行中）。
2. fold 脚本：shutil.copy 输出文件 → 按 gguf tensor offset 原地重写（类型/尺寸不变：bf16 权重列缩放、f32 norm 权重除法）→ 每 α 一个文件 + 一个重写后的 imatrix。
3. 每 α：llama-quantize --imatrix im_awq --tensor-type-file types-B145.txt --tt-errors → J(α) vs J(baseline=final-B145-newim.csv)。
4. 胜者：PPL 8-chunk 单卡（基线 6.7461）+ TP 1/1 bench（基线 pp512/pp2048/tg 见 rebuild 文档 §8.6/8.8）。
5. 若 J 无改善 → 如实记录负结果（fold 与 per-row A-fp8 的收益假说被否）。

## 4. 风险

- s 的 f32 舍入给 norm/weight 引入 ~1e-7 相对扰动：MXFP4 块 scale 重吸收时个别块翻转 1 格（fp32 vs 数学恒等的微小偏离），PPL 层面预期 <0.01%。
- 折叠后 W 动态范围 ×8：bf16 存储不会溢出（bf16 指数域同 f32），但 ffn_down 输出的 fp8 silu 融合路径不变（折叠不触及中间激活）。
- radiance repack 只读 MXFP4 块字节，格式不变 → fast path 覆盖完全不变（B145 配方原样复用）。

## 5. 进度
- [x] 分组映射（§1，源码证据齐全）
- [x] f16 重导出完成（`Qwen3.8-27B-f16.gguf` 54657731968 B）
- [x] fold α=0.5/1.0 + imatrix 重写（469 tensors folded，identity verify rel err 2.84e-07）
- [x] J 对比（**负结果**，见 §6）
- [x] 清理 AWQ 产物（f16 源 + 量化模型，释放 ~140 GB；保留 errs CSV 证据）
（随做随写）

## 6. 结果（2026-10-03，决定性负结果）

量化 + tt-errors 对比（同 types-B145 配方、同 ima-awq05/10 imatrix，J=Σ im_rmse²，n=470 有 imatrix 张量）：

| 变体 | J=Σim_rmse² | Δ vs baseline |
|---|---|---|
| baseline B145 | **3.705061** | — |
| AWQ α=0.5 | 3.747055 | **+1.13%（更差）** |
| AWQ α=1.0 | 3.733620 | **+0.77%（更差）** |

**结论：AWQ/SmoothQuant 通道均衡折叠在本场景是负优化，否决。**

逐层模式（α=1.0 vs baseline，比值 a10/base）：
- **改善**：`ssm_alpha`（0.56-0.88x）、`ssm_beta`（0.70-0.86x）—— G1 折叠组的小矩阵，激活摊平直接降误差。
- **恶化**：`attn_gate`（1.05-1.25x，blk.33 最高 1.246）、`attn_qkv`（部分 1.04-1.07x）、`ffn_gate`（1.02-1.05x）—— 大矩阵因 fold 把 W 列缩放 ×s，权重动态范围扩大，per-32 block 的 MXFP4 量化 scale 效率反降。

**根因**：radiance fast path 的 A 端已是 per-row fp8-e4m3（非 per-tensor），激活通道不均衡对 A 端量化本来就不敏感——AWQ 摊平激活的核心收益场景（A per-tensor 时才显著）不成立。fold 只剩 W 端动态范围 ×s 的负面作用，大矩阵净恶化盖过小矩阵净改善。

证据文件（保留）：
- `/media/seirin/SSD2T_1/hf/awq/errs-awq05.csv`、`errs-awq10.csv`（tt-errors 输出）
- baseline：`/media/seirin/SSD2T_1/gguf/recipe-main/final-B145-newim.csv`（J=3.705061）

已删除（释放 ~140 GB）：`f16-awq05.gguf`、`f16-awq10.gguf`（各 54.6 GB）、`awq05.gguf`、`awq10.gguf`（各 15.6 GB）、`ima-awq05.gguf`、`ima-awq10.gguf`。

B145 baseline 模型（`Qwen3.8-27B-Rad-MX-mix145.gguf`）保持最优，无需替换。
