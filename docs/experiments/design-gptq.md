# design-gptq.md - GPTQ/OBS 完整 Hessian 量化（路线A）+ 校准集优化（路线B）

背景：AWQ fold 已否决（J +1.13%/0.77%，根因= radiance A 端 per-row fp8 对激活分布不敏感）。
分析结论：imatrix（diag H）下 MXFP4 量化已是逐块逐元素最优（块间/元素间解耦），
唯一改进空间 = 引入 H 非对角元素做跨通道误差补偿（GPTQ/OBS）。
llama.cpp 主线与 fork 均无 GPTQ 支持（已查证），需自己实现。

目标：对 Qwen3.8-27B B145 配方实现 OBS 量化，tt-errors J 对比 baseline 3.705061，
J 降 >2% 则 PPL 8-chunk 终验（单卡基线 6.7461）。kernel/类型不变。

## 路线A：GPTQ/OBS

### A1. H 收集（tools/imatrix/imatrix.cpp 扩展）
- 新选项 `--hessian-dir <dir>`：激活时除 diag（in_sum2）外，累积完整 H = Σ x x^T。
- 对称，只存上三角 f32。内存峰值 ~30GB（125GB RAM OK）。
- blocked SYRK + OpenMP（56 核 EPYC，~1 TFLOP/s，总 3.9e15 FLOP @ 256 chunks ≈ 1h）。
- 输出：`<dir>/<tensor_name>.bin`（f32 上三角行优先，K*(K+1)/2）+ meta（K, n_tokens）。
- 只对普通 MUL_MAT 收集（MUL_MAT_ID MoE 路径跳过，本模型无 MoE）。
- calib 运行：BF16 f16 模型 + 3 卡 layer split（-ts 1/1/1，layer 模式每层完整，H 不分裂）。
  256 chunks，数据集 imatrix-calibration-part-cleaned.txt。

### A2. OBS 量化（ggml/src/ggml-quants.c + src/llama-quant.cpp）
- 新函数 `quantize_row_mxfp4_gptq(src, dst, nrow, ne0, imatrix, h_ut, percdamp)`：
  1. 预处理：Hd = H + percdamp*diag(H)；Cholesky(Hd)=L；Hinv = chol(inv(Hd)) 上三角。
     （标准 GPTQ 算法；CPU OpenMP，K=17408 时 ~5s/tensor，65 层 ~5min 可接受。）
  2. 每输出行 n：w = src[n]；沿 K 逐 32-block：
     - 用 diag(H) 近似 grid search 选 E8M0 scale s；
     - block 内逐元素：选 E2M1 码本（im 加权），量化后立即 OBS 补偿未量化列
       w[j:] -= err * Hinv[j,j:] / Hinv[j,j]；
     - block 结束后误差自然传播到后续 block（无边界截断）。
  3. MXFP4 适配说明：OBS 逐列补偿与 32-block 缩放兼容（补偿方向沿 K 维 = block 排列方向）。
     scale 选择用 diag 近似，元素级码本选择+补偿是精确 OBS。
- llama-quantize 新选项 `--gptq-hessian <dir>`：量化每个 2D 权重时按名读 H 走 GPTQ 路径，
  其余 tensor 走原路径；tt-errors 照常输出（J 直接可比）。
- percdamp 默认 0.01（GPTQ 论文值）。

### A3. 验证流程
1. calib 收集 H（~1h）。
2. llama-quantize --gptq-hessian + types-B145 + --tt-errors → J 对比 3.705061。
3. J 降 >2% → 胜者 PPL 8-chunk 单卡（基线 6.7461）+ TP 1/1 bench。
4. 清理：H 文件（~30GB）验证后删除。

## 路线B：校准集优化
- 检查现有 imatrix-calibration-part-cleaned.txt 构成。
- 准备补充集（代码 + 中文对话，按用户实际用途），重新生成 imatrix，
  同 B145 配方量化对比 J。
- 与路线A串行共用 GPU（A calib 优先）。

## 5. 关键发现：J 指标对 GPTQ 是系统性误导（2026-10-03，实测确证）

真实数据（baseline = mix145 gguf，GPTQ = test-gptq.gguf，同一份真实 H；
自研 mmap GGUF 解析器经 baseline 相对 rmse 逐位 MATCH 验证，0.114461/0.112633）：

| 张量 | J（对角代理） | trEH = trace(EᵀHE)（真实输出误差） |
|---|---|---|
| blk.0.attn_qkv | 1.1299e7 → 2.1410e7（**1.89x 恶化**） | 1.2207e7 → 1.1834e7（**改善 3.1%**） |
| blk.0.ffn_down | 411.7 → 707.4（恶化） | 420.2 → 388.8（**改善 7.5%**） |

C++ 端 tt-errors 的 im_rmse 比值 1.376² = 1.89，与 python J 比值精确一致。

结论：
- GPTQ 按定义**故意抬高对角误差**换取交叉项补偿，故 `--tt-errors` 的 J 会系统性
  否决 GPTQ。用 J 选 AWQ 是合适的（AWQ 不改目标函数），**但不能用 J 选 GPTQ**，
  唯一有效裁判是端到端 PPL。
- 秩亏校准（M<K）会虚报收益（合成实验 38%）：误差被藏进校准未覆盖的方向。
  故正式校准用 128 chunks（ntok=65536 >> K=17408，充分满秩）。

## 5.1 收益预算：误差天花板定位（2026-10-03）

新增实测基线：
| 配置 | PPL | 说明 |
|---|---|---|
| bf16 源（同配方同 corpus，3卡 layer split, ngl=65） | **6.2413** | 理论下限 |
| B145 量化 + radiance 快路径 | 6.7461 | 当前生产模型 |
| B145 量化 + GGML_RAD_DISABLE=1（A 端不量化） | 6.7375 | 单卡 |

- 量化总损失 = 6.7461/6.2413 - 1 = **8.1%**。
- 关掉 A 端 fp8 量化，PPL 只动 **0.13%** → **"A 端激活量化是瓶颈"假说被否**。
  这也解释了 AWQ 零收益（它只优化 A 端分布）。
- 误差按类型分解（J 占比）：mxfp4 **95.7%**、mxfp6 4.3%、mxfp8 0.0%。
  mxfp4 内部均匀：ffn_up 23.0% / ffn_down 22.1% / ffn_gate 21.8% / attn_qkv 21.7%
  / attn_gate 6.2%。B145 的类型分配已接近最优，无重分配空间。

GPTQ 收益预算：覆盖 280/285 个 mxfp4 张量（blk.64 nextn 不在主 forward 图内，
无 H，自动走原路径）。按实测 trEH 改善 3-7.5%（均值约 5%）× mxfp4 占比 95.7%
× 量化损失 8.1% ≈ **PPL 改善 0.38%**，略低于 0.5% 验收线 → 需实测终结争论。

## 6. 工程实现记录

- `tools/imatrix/imatrix.cpp`：新增 `--hessian-dir`，`accumulate_hessian()`
  用 OpenMP 分块 SYRK 累积上三角 packed H；main 里在 common_params_parse 前
  剥离该自定义参数（否则被 arg parser 拒绝）；CMakeLists 需显式
  `find_package(OpenMP)` + link（llama-imatrix target 默认不继承 -fopenmp，
  漏掉时 SYRK 退化为单线程，2 chunks 从 4 分钟变 10 分钟 CPU）。
- `ggml/src/ggml-quants.c`：`quantize_row_mxfp4_gptq()`，逐列量化 + 立即用
  U（inv(H) 的上三角 Cholesky 因子）补偿后续列；scale 仍在补偿后的值上 grid search。
- `src/llama-quant.cpp`：`--gptq-u-dir`，量化 MXFP4 时按张量名加载 U 走 GPTQ 分支，
  行间独立故可线程并行；本地声明结构 `gptq_block_mxfp4`（17B）避免依赖内部头。
- `include/llama.h` + `tools/quantize/quantize.cpp`：新增 `gptq_u_dir` 参数。
- `gptq_preprocess.py`：packed H → dampen → chol(inv(H)) 上三角 U，numpy OpenBLAS。

校准命令（128 chunks，3 卡 layer split，embed/out 压 CPU 以塞进 65GB）：
```
llama-imatrix -m Qwen3.8-27B-f16.gguf -f imatrix-calibration-part-cleaned.txt \
  --hessian-dir <out> -c 512 --chunks 128 \
  -dev ROCm0,ROCm1,ROCm2 -ts 2/1/1 --split-mode layer -ngl 65 --load-mode none \
  --override-tensor "token_embd.weight=CPU,output.weight=CPU"
```
产出 464 个 H（58GB，blk.64 nextn 层不在主 forward 图内故无 H）。
耗时：128 chunks ≈ 168 分钟。

## 10. J -> PPL 映射定律与格式极限（2026-10-03，核心结论）

四个实测点（同 corpus 8-chunk，同 bench 配置 -ub 2048 -fa 1 -dev ROCm0）：

| 配置 | J | PPL | pp2048 | 说明 |
|---|---|---|---|---|
| B145 (E8M0+radiance) | 3.7051 | 6.7461 | ~2130 | 当前生产 |
| mix60 (60 张量切 e4m3) | 3.2652 (-11.9%) | 6.7188 (-0.40%) | 2021 (-5%) | 不划算 |
| e4m3 全量 (MMQ) | 2.5109 (-32.2%) | **6.6048 (-2.10%)** | 1355-1500 (-30%) | 质量最好，速度代价大 |
| bf16 源 | ~0 | 6.2413 (-7.48%) | - | 理论下限 |

幂律拟合（mix60 vs e4m3all）：`dPPL% ~ (dJ%)^1.66`（超线性）。
推论：
- 小改进收益极差。GPTQ 实测 trEH 改善 3-7.5% -> 预测 PPL 仅 ~0.1%，
  **低于 0.5% 验收线**，不值得承担速度/复杂度风险。
- 只有"整格式换代"级别（E8M0 -> UE4M3）才能拿到 ~2% 量级。

radiance 无法无损支持 UE4M3（结构性障碍，非工作量问题）：
fast path 的 folded-scale 设计前提就是"块 scale 是 2 的幂"——
`kMag[d][...]` 常量表把 E8M0 块指数当 binade 位移折进 e4m3 权重字节，
从而"从内层循环彻底移除 per-32-block rescale"（radiance-gemm.cu:20-27 注释）。
UE4M3 带 3 位尾数，`e2m1_code x ue4m3_scale` 乘积需要 5-6 位尾数，
折进 e4m3（3 位尾数）会引入 ~3-6% 权重舍入误差，直接抵消 32% 的 J 收益。
要支持就得放弃 folded scale 回到内层 per-block rescale = 退化成 MMQ 速度。

结论：在 gfx1201 radiance fast path 约束下（type==MXFP4 + E8M0），
**B145 已接近该格式的误差下限**（grid search 在 E8M0 网格内已逐块最优，
AWQ/GPTQ 分别因 A 端非瓶颈/超线性衰减而无收益）。

## 11. GPTQ 全量终验：决定性负结果，路线关闭（2026-10-03）

128-chunk H（ntok=65536 满秩）+ 280 U + OBS 全量量化（1827s，280/280 命中 GPTQ 路径）：

- J = 5.1627（vs B145 3.7051，+39%；0/470 张量对角改善）——与"J 失效"预言一致，
  对角误差被故意转移到交叉项。
- **PPL 8-chunk = 6.9203 vs 基线 6.7461：恶化 2.59%**。真实 trEH 的单张量改善
  （3-7.5%）没有兑现：OBS 把权重改离原函数，补偿误差沿 65 层残差链累积放大，
  逐层局部最优在全局意义下是净伤害。GPTQ 论文环境（浅层/激活不量化）与
  MXFP4(W)+fp8(A) 深链组合根本不同。
- 产物 gptq-all.gguf 已删；实验数据（errs-gptq-all.csv）保留。
- 幂律 §10 预测 GPTQ 收益 ~0.1%，实测为 -2.59%：J/trEH 都不可作为代理，
  **只有端到端 PPL 可信**——这条方法论教训是整个实验最重要的产出。

## 11.1 radiance 约束下的终局（用户决策线）

| 路线 | 端到端 | 判定 |
|---|---|---|
| AWQ fold | J +1.13% | 否决（A 端 per-row 对激活分布不敏感） |
| GPTQ/OBS | **PPL -2.59%（更差）** | 否决（跨层累积伤害） |
| e4m3 全量 | PPL +2.10%，但 radiance 不支持 e4m3（folded-scale 前提=2 的幂） | 格式冲突 |
| mix60 | PPL +0.40%，pp2048 -5% | 不划算 |
| A 端 fp8 隔离 | PPL 仅差 0.13% | 激活侧无空间 |
| imatrix grid search（B145） | **6.7461，radiance 99.8% 覆盖** | **终态** |

结论：**在 type39+E8M0+radiance 约束下，B145 已处于可达的帕累托前沿**。
唯一理论上行方向是换格式（e4m3 或 INT K-quant），都要求放弃 radiance fast path；
GSQ-RCO（IST-DASLab 新作，见 design 附注）输出 IQ2/IQ3 系列格式，同样不走 radiance，
且其增益场景是 ≤3.5bpw——4.56bpw 的 MXFP4 上预期收益有限。若将来要再进一步，
可评估把 GSQ 的 Gumbel-Softmax 逐元素搜索移植进 mxfp4 量化器（格式不变、
radiance 不变，只换舍入决策），但工程量大、收益不确定，暂不投入。

## 12. 交付物清单（本次 GPTQ 工程留下的可复用能力）
- `llama-imatrix --hessian-dir <dir>`：完整 Hessian 采集（OpenMP 分块 SYRK）
- `llama-quantize --gptq-u-dir <dir>`：MXFP4 OBS 误差补偿量化
- `gptq_preprocess.py`：H -> U（dampen + chol(inv)）预处理
- `--tt-errors` J 指标的适用边界已查清：同目标函数才可比（AWQ 可，GPTQ 不可）
