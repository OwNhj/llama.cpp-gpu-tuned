# design-mxfp4-rebuild.md - Qwen3.8-27B 主模型重建（BF16 源 -> fuse-qkv 导出 -> imatrix -> MXFP4+MXFP8/MXFP6/MXFP4）

发起：用户 2026-10-02 指令。三件事：
1. 主模型需要重新量化一个 MXFP4+MXFP8 的（旧 gguf-rad 已不在 HDD500G，bf16 源 gguf 也丢了）。
2. imatrix 重新生成，用 `/media/seirin/SSD2T_1/llama.cpp-quantize`（它支持 MTP/nextn 权重的 imatrix 校准，改造后或可给 drafter 用）。
3. 源文件从 ModelScope 下 safetensors BF16 原版。
4. （用户提出）探索导出时融合 gate/up 是否可行，已知 QKV 可融合。
5. （用户提出）量化时 output.weight 与 token_embd.weight 按重要性分配 MXFP6 / MXFP4。

## 1. 源与下载

- ModelScope `Qwen/Qwen3.8-27B`：18 shards，55.56 GB，**全 BF16**（已用 Range 读 safetensors header 验证：392 tensors，dtype 全 BF16）。
- 结构：`model.language_model.layers.0..*`（848 tensors，GDN `linear_attn.*` + 每 4 层 full attention）、`model.visual.*`（333，多模态塔）、`mtp.layers.0.*`（11）、`lm_head.weight`、`model.language_model.embed_tokens.weight`。
- 注意：`tie_word_embeddings: false` 且 lm_head 独立存在 -> embed/out 是两个不同矩阵，可分别定档。
- 下载目录 `/media/seirin/SSD2T_1/hf/Qwen3.8-27B-BF16/`，脚本 `/home/seirin/dl_qwen.sh`（resume-safe，重试 3 次），日志 `dl.log`/`dl.out`。
- 实测速度约 65 MB/s，全程约 15 分钟。
- **踩坑记录**：本 harness 的 `scp` 保留本地属主/权限导致 root 属主文件远端 seirin 不可读——脚本一律 base64 管道传。非交互 shell PATH 无 `~/.local/bin`，modelscope CLI 用绝对路径。

## 2. gate/up 融合可行性（调查结论）

现状（llama.cpp-quantize 导出侧 + tuned C++ 加载侧双向核查）：

| 融合 | Python 导出 | C++ 模型侧 | 结论 |
|---|---|---|---|
| QKV (dense attn) | `--fuse-qkv` 完整 | qwen35 已注册 `ATTN_QKV`（`qwen35.cpp:77`，`TENSOR_NOT_REQUIRED`） | **已通，直接用** |
| gate/up MoE | `fuse_gate_up_exps` -> `ffn_gate_up_exps` | `LLM_TENSOR_FFN_GATE_UP_EXPS` + `llama-graph.cpp:2169`（mul_mat_id 一次 + view 切半） | 已通，但 qwen35 非 MoE，用不上 |
| gate/up dense | **无** `MODEL_TENSOR.FFN_GATE_UP`（dense） | **无**：`build_ffn(up, gate, down)` 两路分开；`llama-model.cpp:415` 的正则 `ffn_gate_up(_exps)?` 只服务 checkpoint 拆分加载，不建图 | **需要两侧都改** |

dense 融合要动的地方（评估后）：
- quantize 仓库 `gguf-py`：新增 `MODEL_TENSOR.FFN_GATE_UP` + mapping（gate_N/up_N 合成 `blk.%d.ffn_gate_up`，`torch.cat([gate,up], dim=0)`），改 `set_gguf_parameters`/长度断言。qwen.py 的 modify_tensors 借道 base.py 已有的 `fuse_gate_up_exps` 模式复制一份 dense 版。
- tuned C++：`llama-arch` 枚举 + `qwen35.cpp` `create_tensor(FFN_GATE_UP)` + `build_ffn` 增加 gate_up 入参（view 切两半，与 MoE 版同构：`ggml_view_2d(gu, n_ff, ...)` 前半 gate 后半 up，取决于拼接方向）。
- radiance fast path：repack 天然可行——融合张量就是 `[2N, K]` 的普通 2D 权重，repack 把它当作 N 翻倍的单块处理，甚至省一次 repack-cache 查找。**MXFP4 的 E8M0 scale 是 per-32 元素独立**，不存在 FP8/NVFP4 那种"gate/up 共享 scale"问题，融合无数值副作用。
- 收益：少一次 GEMM launch、radiance tile 填充率提升（N 从 17408 到 34816，对 TN 分块更饱满）；`swiglu` 融合 act kernel 已按两个独立输入设计，改 view 即可。
- 风险：`ggml_view` 后 silu_mul kernel 的内存访问从两个连续 buffer 变一个的大偏移——带宽中性；graph 层改动会牵动 lora adapter 路径（`pattern_ffn_gate_up_weight` 的 lora 检查已有，反而说明有人铺过路）。
- 决策：**一期不做 dense 融合**（改动横跨两个仓库 + 图构建，与本次"重建模型"主线正交）。先用已通的 `--fuse-qkv` 导出；融合 gate_up 列为后续独立实验，有 radiance 收益假说可以单测。

## 3. 导出计划

用 llama.cpp-quantize（有 fuse-qkv 且 bf16 直读，之前已确认 fork supports Qwen3_5TextModel）：

```
python3 convert_hf_to_gguf.py --model Qwen3.8-27B-BF16 --outtype bf16 \
    --fuse-qkv --outfile Qwen3.8-27B-f16src.gguf
```
- 视觉塔：`--model` 全量会带 visual。text-only 需要 `--text-only`（查 CLI 是否存在）或量后剔除。**先验证主模型只跑文本**：serve 时 mmproj 是独立 gguf，text gguf 里的 visual 塔会被 loader 按 arch 跳过吗——不跳就会白占 2-3 GB。决定：导出带 visual，量化时用 `--exclude-weights` 让 visual 走 COPY 路径或后续确认 llama.cpp 对 language_model_only 的处理。
- MTP：fork convert 之前记录支持 `Qwen3_5TextModel`，mtp.* 应自动映射为 nextn.*。

## 4. imatrix 计划

`llama.cpp-quantize/tools/imatrix`：
- 校准语料：之前用过 wiki + 代码混合（`imatrix_unsloth.gguf` 是下载的对照，自己做的叫 `imatrix-qkv.gguf`，940 entries，说明 fused qkv 训练集存在且已配好名字映射——**imatrix 的 tensor 名必须匹配导出布局**：这次导出同样 `--fuse-qkv`，名字对得上）。
- MTP 支持：`set_mtp_prefix` + 第二 context `LLAMA_CONTEXT_TYPE_MTP` 做单 token 链式 decode 校准（`imatrix.cpp:858+`）。
- 待确认：CLI 怎么开启 mtp 校准（`--mtp`? 前缀参数?）——查 common/arg.cpp。
- 改造给 drafter 用（用户提的，记为后续项）：DFlash2 drafter 的激活分布 ≈ 目标模型 hidden state + drafter 自身前向，用目标模型校准集对 drafter 权重算 imatrix 是合理的——但 drafter 是独立 checkpoint（enc.input/embed 等结构不同名），imatrix 收集要能加载 drafter 模型跑。llama-imatrix 本来就对任意 gguf 收集，所以**今天就能用 llama.cpp-quantize 给 DFlash2-mix 收集 imatrix**（如果 DFlash2 bf16 gguf 能加载执行）。列为 DFlash2 后续迭代。

## 5. 量化计划

类型族（radiance fast path 偏好 MXFP4/E8M0）：
- 线性投影主体：MXFP4（type 39，radiance 唯一类型）。
- attn_qkv（融合后单块）：MXFP4 或按扫描降 MXFP6；注意 fused 后 qkv 单张量 [n_embd, q+2k_dim]，radiance gate 要求 K%64==0、N%16==0——n_embd=5120 OK，行数 12288+2048+2048=16384 OK。
- **token_embd 与 output.weight：用户指令按重要性分别 MXFP6/MXFP4**：
  - 带宽视角：decode 每步 lm_head 全量读，embed 只 gather 一行——**embed 降精度几乎不伤 tg，out 降精度省 tg 带宽但伤 logits**。
  - 质量视角：lm_head 直接产出 logits，PPL 更敏感。
  - 结论：output.weight=MXFP6，token_embd=MXFP4。用 tt-errors + 最终 PPL/接受率验证（DFlash2 的接受率测试正好用新主模型）。
- MTP/nextn.*：MXFP8（E4M3+E8M0，decode 关键路径，带宽收益已实测 1.70x 组合的一半来自这里）。
- norm/conv/A_log/dt_bias：F32/F16 保持。
- 决策方式：先导出 f16 源 gguf，用打了 `--tt-errors` 的 tuned llama-quantize 在**主模型**上跑 MXFP4/6/8 三档扫描（主模型 866+ tensors，扫描一次约 20-40 分钟），再出混合配方。这是本次重建与 tt-errors patch 的闭环。

## 7. MXFP4/6/8 与 imatrix（用户询问，已源码确证）

三个 MX 类型**全部真支持 imatrix 加权量化**，非跳过：
- `quantize_mxfp4/6/8()`（ggml-quants.c）：`if (quant_weights) -> *_impl else -> *_ref`。
- `quantize_row_mxfp4_impl`（:406）：对每 32 块的 E8M0 指数做 grid search（候选 `e0-6..e0+1`），目标函数 `sse += qw[j]*(x-dq)^2`，加权最小者胜。语义：牺牲低重要性元素的表示换高精度给高重要性元素。
- `quantize_row_mxfp8_impl`（:626）/ `quantize_row_mxfp6_impl`（:811）：同款加权搜索，per-sub-block 粒度（候选 ±6）。
- `tensor_requires_imatrix`（llama-quant.cpp:982）：IQ1/IQ2 系列与 IQ3_XXS 必需，Q2_K_S 例外必需；**MXFP 系列不在必需表里** = 可选增强，无 imatrix 时走纯 amax 路径。
- 推论 1：imatrix 必须在类型扫描之前生成，否则 tt-errors 测的是"无 imatrix"的误差，与最终带 imatrix 量化的配方不匹配。
- 推论 2：MX 类型的误差对 imatrix 质量的敏感度高于 K-quant（因为 scale 直接被加权优化），新 imatrix 对 MXFP4 主类型的 PPL 增益应可观。

## 8. 进度与决策记录

### 8.1 执行记录（续）
- 12:04 BF16 下载完成：55,563,006,776 字节（vs index 声明 55,562,855,904，多出的为 safetensors header，合理），0 张量缺失。
- 12:07 导出完成：`Qwen3.8-27B-f16.gguf` 54.6G，832 tensors，`--fuse-qkv` 生效（`attn_qkv [5120, 10240]` GDN 层 / `14336` full-attn 层——full-attn 层含 attn_gate 所以行数不同），`blk.64.nextn.*` MTP 块在，embed/out 独立 bf16 [5120, 248320]。
- **旧 imatrix-qkv.gguf 覆盖度实测**：470 entries 对 832 tensors 的新模型 **direct match 470/470**，包括 `blk.64.nextn.eh_proj.weight`！bf16 可量化张量里只缺 `token_embd.weight` 与 `output.weight`（imatrix 本来就不给 embedding 类收集）。也就是说旧 imatrix 与本次导出布局**天然兼容**（同一 arch、同一 fuse-qkv 命名）。
- 用户指令"重新生成 imatrix"仍执行（quantize 仓库 --calibrate-mtp --calibrate-mtp-steps 3，语料不变 350 chunks，-dev ROCm0,ROCm1,ROCm2 --tensor-split 2,1,2）：旧 imatrix 无 dataset 元数据可信度低，且 nextn 校准方式未知；新 imatrix 与扫描配对更干净。
- 12:20 q8_0 源完成（29.0G，38s）供 imatrix 加载：`Qwen3.8-27B-q8_0src.gguf`。
- imatrix 输出目标：`imatrix-f16src.gguf`。完成后跑 `scan_main.sh`：9 个候选类型（mxfp4/mxfp6/mxfp8/iq4_nl/iq4_xs/q4_k/q5_k/q6_k/q8_0），带 `--imatrix`，`--tt-errors` 出逐张量误差，真实模式（非 dry-run）写 /dev/shm 后即删。
- embed/out 无 imatrix：llama-quantize 会 warn "did not find weights"，量化照走无权重路径（ref quantizer）；对 MXFP4/6 这类 amax+scale 的 embedding 影响有限，最终用 PPL 验收。

### 执行计划（原）
1. 导出：`python3 convert_hf_to_gguf.py --model /media/seirin/SSD2T_1/hf/Qwen3.8-27B-BF16 --outtype bf16 --fuse-qkv --outfile /media/seirin/SSD2T_1/gguf/Qwen3.8-27B/Qwen3.8-27B-f16.gguf`（TEXT 模式自动跳过 visual）。
2. imatrix：quantize 仓库 llama-imatrix，f16 源 + 三卡 + calibrate-mtp（对齐历史成功命令）。
3. 扫描（tuned 仓库 llama-quantize + --tt-errors，--dry-run 不可用因为要真量化）：类型矩阵 `mxfp4 mxfp6 mxfp8 iq4_xs q4_k q6_k q8_0`，约 7 轮。
4. 配方：线性层按 tt-errors 贪心；embed/out 按用户指令 MXFP4/MXFP6（用扫描数据定谁吃哪个档：预期 embed=MXFP4、out=MXFP6，理由是 decode 带宽不对称）；nextn 保底 MXFP8。
5. 验收：PPL（8-chunk 对齐 6.93 基线）、llama-bench pp512/pp2048/tg128、MTP+DFlash2 接受率（`--temp 0` 方法学教训）。

- [x] BF16 源确认（ModelScope Qwen/Qwen3.8-27B，全 BF16，55.56 GB）
- [x] 下载完成（55.56G，0 缺失）
- [x] 导出 bf16 GGUF（--fuse-qkv，832 tensors）
- [ ] imatrix 生成（--calibrate-mtp，进行中；GPU 三卡，ETA 数十分钟）
- [ ] 九档类型扫描（tt-errors + imatrix，进行中；CPU-bound，单类型实测 3.5 min）
- [ ] 配方生成 + 量化
- [ ] 验收（PPL/bench/接受率；DFlash2 mix 两档也并入此步用新主模型测）

### 8.2a 主模型扫描结果与配方（13:16 扫描 9/9 完成）

Uniform totals (472 quantizable tensors, imatrix-weighted where available):
| type | GiB | J (sum err^2) |
|---|---|---|
| mxfp4 | 13.52 | 6.035 |
| iq4_xs | 13.52 | 1.965 |
| q4_K | 14.31 | 1.470 |
| iq4_nl | 14.31 | 1.903 |
| mxfp6 | 19.88 | 0.406 |
| mxfp8 | 26.24 | 0.335 |
| q8_0 | 27.69 | 0.031 |

Per-class sensitivity at mxfp4 (weighted rmse mean): ssm_alpha 0.126 > ffn_up 0.114 > ssm_out 0.112 (max 0.139) > attn_output/ffn_down ~0.112 > attn_gate 0.108. GDN 侧支（alpha/beta/out）确实比 FFN 主干更敏感。mxfp6 over mxfp4: median err ratio 0.25（4 倍改善）for +1.75 bpw。

**B 族配方（radiance 覆盖优先，贪心从 all-mxfp4 升级）**:
- B145: 14.49 GiB, J=3.72, 79% bytes on radiance fast path; 升级 186 张量（全部是 ssm_alpha/beta/out + attn_gate + attn_output 这类敏感侧支，FFN/qkv 主干保持 mxfp4）
- B155: 15.49 GiB, J=2.76, 60% 覆盖
- A 族纯 J 贪心（15.5G 时 J=2.73 但覆盖掉到 ~36%）不推荐——radiance pp 收益（实测 +60%）按全 MXFP4 权重占比折算，36% 覆盖的 pp 收益大约只剩零头。

决策：**默认 B145**。理由：与旧 15.1G f4body-f8head 模型同量级尺寸；J 从 all-mxfp4 的 6.04 降到 3.72（-39%）且几乎不付覆盖代价；embed=mxfp4 / out=mxfp6 / nextn=mxfp8 固定规则按用户指令执行。
验收线：PPL 对比旧 6.9517（Mxfp4 gguf 口径）；pp/tg bench；DFlash2 接受率 A/B（target 换重建模型）。

### 8.2 sanity 与 imatrix 身份核查
- 单类型 sanity（mxfp4 + 旧 imatrix-qkv.gguf）：exit=0，472 行 CSV，`im_rmse` 有效列产出（如 ffn_down 0.110 vs 未加权 0.114），仅 embed/out 无 imatrix（`im_rmse=-1`，预期，走 ref 量化器）。量化尺寸 13850 MiB / 4.25 BPW。=> scan_main.sh 机制、命名匹配、CSV 格式全部验证通过。
- 历史命令显示旧 imatrix 生成源是 `Xing-q8_0.gguf`（该文件现已不在磁盘）。但 470/470 张量名 + fused-qkv 布局 + `blk.64.nextn.*` 齐全，证明它确为 Qwen3.8 本体所用。扫描仍带它（加权误差列有效）。
- 新 imatrix `imatrix-f16src.gguf`（本次 --calibrate-mtp 从 q8_0 源生成）用于**最终量化**，与扫描数据分离——扫描只需相对排序，最终一次用最佳 imatrix。

### 8.3 主模型配方算法（main_recipe.py）
- 固定规则（用户指令 + 敏感度）：`token_embd=mxfp4`、`output=mxfp6`（lm_head 直出 logits，重要性高者吃更高精度档，且 embed 为 gather-only 降精度几乎不伤 tg）、`blk.64.nextn.*=mxfp8`（decode 关键路径）。
- 自由层贪心：per-tensor pareto 阶梯 + radiance 偏向（非 mxfp4 点若被 mxfp4 点在 err 和 bytes 双向 5% 内支配则丢弃），目标 `J = sum im_rmse_i^2`，预算 13/14/15/16 GiB 多档。
- 尺寸实测锚点：全 mxfp4 = 13.5 GiB（4.25 BPW）；q8_0 参考在扫描表内。

### 8.4 多卡 radiance 崩溃（真实 bug，已修复）
新模型跑多卡 PPL 立即崩在 `quantize_tokens_fp8`（GPU index 1, illegal memory access）。**旧 MXFP4 模型同配置同样崩** => 与本配方无关，是 radiance 集成的历史缺陷，此前从未暴露（过往 PPL/bench 都是单卡口径）。

根因：`ggml_rad_act_bufs` 激活 scratch 缓存**只按 K 建键**，不含 device。tensor-split 下 GPU0 先为某 K 分配 buffer，GPU1 的 `ggml_rad_act_get` 命中同一 K 直接复用那块显存，而 kernel 跑在 device 1 上下文 => 跨设备指针非法访问。

修复（`radiance-gemm.cu`）：缓存键改 `std::pair<int,int64_t>`（`ggml_cuda_get_device()`, K）；buffer 增长时先释放旧三件套再重建（顺带修掉一个增长路径的泄漏）。

验证：
- 修复前：多卡必崩（新旧模型皆然）；单卡正常。
- 修复后：多卡 `Final estimate: PPL = 6.7461 +/- 0.12137`，与单卡同参数结果**逐位一致**。

### 8.5 B145 成品验收（新 imatrix `imatrix-f16src.gguf`，470 entries 含 6 项 nextn 校准）
- 量化：`Qwen3.8-27B-Rad-MX-mix145.gguf` 14851.83 MiB（4.56 BPW），227 s；override 精确生效 187 条。
- 成品 vs 扫描预测：472/472 张量 rmse+bytes 完全匹配（diff 0）；470 行带有效 im_rmse。
- PPL（c=4096 -fa 1 chunks=8 ub=2048，corpus.txt）：**6.7461 +/- 0.12137**（多卡=单卡）。

### 8.6 不对等多卡分配验证（用户指令：tensor-split 2/1，跳过慢卡 9060XT）
> 方法学修正：llama-bench 的 `-dev`/`-ts` 逗号是笛卡尔积独立测试点（每个单卡），真多卡切分要用斜杠 `-dev ROCm0/ROCm2 -ts 2/1`。

单卡 ROCm0（旧混合 18.28G vs 新 B145 14.50G，graph on）：pp512 1223.84 -> 1928.39 (+58%)；pp2048 1401.13 -> 2309.02 (+65%)；tg128 26.29 -> 28.23。

**真双卡 2/1（`-dev ROCm0/ROCm2 -ts 2/1`，radiance 修复后 graph 正常）**：
| 指标 | 旧混合 | 新 B145 |
|---|---|---|
| pp512 | 1300.97 | **2235.78 (+72%)** |
| pp2048 | 1401.13 | **3171.25 (+126%)** |
| tg128 | 26.29 | 27.86 |

pp2048 3171（方差 ±4.9，极稳）创历史最好（此前最好记录为三卡 2692）。2/1 把大卡 R9700 权重份额加倍，radiance 在 32G 卡上 tile 填充更满，pp 增益比单卡更高。tg128 两模型接近（都走 MMVQ，与 radiance/切分无关）。

**layer 切分 vs tensor 并行（B145，2/1，graph on）**：
| 指标 | layer（默认） | tensor（并行） |
|---|---|---|
| pp512 | 2235.78 | **2587.64 (+16%)** |
| pp2048 | **3171.25** | 2434.77 (-23%) |
| tg128 | 27.86 | **34.34 (+23%)** |

解读（含 ubatch 对照实验，pp2048 -r 3）：
| pp2048 | layer | tensor |
|---|---|---|
| ub=512 (bench 默认) | **3171** | 2435 |
| ub=2048 | 2524 | **2877** |

- decode：TP 每层两卡各读一半权重并行，带宽近似翻倍 -> tg +23%（34.34 为无投机历史最高，此前 29.85）。
- prefill 取决于 ubatch：小 ubatch 下 layer 是事实上的数据并行（两卡各异步算不同 512-token 批次，零通信），TP 则是每层付规约 + 主卡激活量化串行化，亏 23%；ub=2048 时 radiance 进 TN4 区、单层计算量翻倍摊薄通信，且 layer 的大 ubatch 模式反而退化，TP 赢 14%。
- 生产 server 口径（`--split-mode tensor -ub 2048`）下 TP = 2877，优于旧混合模型同配置，且 tg 更高：维持 TP 配置正确。

PPL 对比说明（避免误读）：盘上 `Qwen3.8-27B-Mxfp4.gguf`(18.28 GiB) 是 Q8_0/Q4_K + 少量 MXFP4 的**混合**模型（PPL 6.4754），并非全 MXFP4；真正的全 MXFP4 基线（17.4 GiB, out=bf16, PPL 6.9517）已在之前的清理中删除，无法复测。
- B145(14.5G, 6.7461) vs 全MXFP4(17.4G, 6.9517)：**尺寸 -17%，PPL 更低（更好）**——升级敏感层 + 新 imatrix 的收益。
- B145 真双卡 PPL = 6.7461，与单卡/三卡逐位一致（切分不改数值）。

**TP 份额语义修正（用户指出：2/1 是 9070 等 9700，正确）**：TP 下 `-ts` 切权重列份额 = 计算份额，与显存无关。两卡同为 gfx1201 64CU 算力同级，2/1 让 R9700 每层扛 2/3 计算、9070XT 算完 1/3 空闲等同步点，有效层时间锚在 2/3T。实测（ub=2048, -r 3）：单 R9700 pp2048 2438 / tg 28.25；TP 2/1 2877 / 34.34；**TP 1/1 3363 / 41.04**。1/1 对 2/1 的理论增益 1.33x、实测 1.17x（同步/规约占比两者接近，差异纯是均衡性）。生产配置应改 `--split-mode tensor --tensor-split 1,1`，R9700 富余显存留给 KV（--cache-ram），不要用 ts 表达。历史 2,1,2 / 2,1,1 配置是 layer 模式直觉的遗留。

### 8.7 radiance 跨模型加载 OOM 崩溃（真实 bug，已修复）
现象：`llama-bench -dev ROCm0 -ts 2,1`（同卡加载模型两次）在第二个测试点崩；栈指向 `ggml_cuda_mul_mat_q_radiance` 的 `hipMalloc`。隔离：`GGML_RAD_DISABLE=1` 不崩、旧混合模型不崩、PPL 不崩。

根因（插桩 + 错误消息落盘确证）：`g_radiance_weights` 以权重设备指针为 key，但模型 buffer 释放时**从不清理这些缓存条目**。第一次加载的 repack（B145 全 MXFP4 覆盖约 +9G 显存）永久滞留；第二次加载新 buffer 指针不同 -> cache miss -> 再分配 9G -> 单卡 OOM（`hipMalloc: out of memory`）。旧混合模型只泄漏 ~4G 所以侥幸不崩——这就是"全 MXFP4 才暴露"的原因。与 graph capture 无关（capcheck 全是 status=0，崩溃发生在 capture 前的 eager warmup 阶段）。

修复（owner = repack 缓存生命周期）：
1. `mmq.cu` 新增 `ggml_cuda_invalidate_weight_caches(base, size)`：按地址区间清除落在被释放 buffer 内的 radiance + decode 缓存条目（顺带消除 stale-pointer 复用 = 读到旧 repack 权重的正确性隐患）。
2. `ggml-cuda.cu` 的 `ggml_backend_cuda_buffer_free_buffer` 在 cudaFree 前调用它（对 COMPUTE buffer 同样安全：其 key 不落在该区间）。
3. `vendors/hip.h` 补 `cudaStreamIsCapturing/cudaStreamCaptureStatus/cudaStreamCaptureStatusNone` 三个宏映射（HIP 侧原本缺失）。
4. 保留 capture-guard（miss 且流正在 capture -> 回退 MMQ）：独立正确的防御，argsort.cu 有同类先例。

验证：原崩溃命令 exit=0（两测试点 ENGAGE 1120 次）；真双卡完整 bench 全过；PPL 6.7461 逐位一致（修复零数值影响）。

### 8.8 gfx1200 (9060 XT) 跑 radiance：HSA_OVERRIDE_GFX_VERSION=12.0.1（用户方案，验证通过）
现象：9700+9060 TP 崩 `invalid device function`（device 1）——radiance/WMMA kernel 的 code object 没有 gfx1200 变体（编译期 per-arch 宏/覆盖问题），运行时在 gfx1200 上查不到函数。`GGML_CUDA_ALLREDUCE=internal` 可绕过 RCCL 跑通但 9060 退 MMQ dp4a（pp512 1916）。

用户方案：`HSA_OVERRIDE_GFX_VERSION=12.0.1` 让驱动把 gfx1200 报告为 gfx1201，运行时按 gfx1201 选 code object（同 RDNA4 ISA 家族）。

验证（默认 RCCL，无 internal env）：
- 崩溃消失，exit=0，radiance 在 9060 上 ENGAGE。
- **数值安全（决定性证据）**：9700+9060(TP 2/1, HSA) PPL = **6.7401 ± 0.12132**；9700+9070(TP 2/1, 原生 gfx1201) PPL = **6.7401 ± 0.12132**——逐位一致。欺骗 ISA 不改变任何计算结果。
- 注意 TP 与 layer 的 PPL 有微小差异（6.7401 vs 6.7461）：TP 每层 bf16 allreduce 的舍入路径不同，与 9060 无关（对照组同样 6.7401）。

**份额平衡验证（用户直觉正确）**：同为 2/1 份额，9060(32CU, 恰好 9700 的一半) 组合 pp512 = **2391.71** > 9070(64CU, 与 9700 同级) 组合 2235.78——2/1 对 9060 是计算平衡的（两卡同到同步点），对 9070 则是慢卡空等。但天花板不同：
| TP 2/1 组合 | pp512 | pp2048(ub2048) | tg128 | PPL |
|---|---|---|---|---|
| 9700+9060 (HSA 12.0.1) | **2391.71** | 2513.79 | 32.29 | 6.7401 |
| 9700+9070 | 2235.78 | **2877.20** | **34.34** | 6.7401 |

2/1 平衡红利在延迟敏感的小 M（pp512）兑现；大 M 与 decode 看总算力/总带宽，9060 半 CU + 128-bit 位宽封顶（用户"正好一半"的判断在平衡性维度成立，在吞吐上限维度 9070 仍占优）。最优仍是 9700+9070 TP 1/1（3363/41.04）。

使用限制：HSA override 属 ISA 伪装，gfx1200 若有未文档化差异会静默出错——本次 472 张量 PPL 逐位对照是强验证样本，但长期方案应修编译覆盖（radiance kernel 补 gfx1200 target 或运行时探测降级）。
