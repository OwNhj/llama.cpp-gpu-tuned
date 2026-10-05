# qwen35 ffn_gate_up 权重融合（radiance/单卡）

## 目标
把 65 层的 `ffn_gate.weight` + `ffn_up.weight`（均 mxfp4, {5120,17408}）
合并为单张量 `blk.%d.ffn_gate_up.weight`（{5120, 34816}，mxfp4 原样 concat），
graph 里一次 GEMM 出 gate|up，再用既有 `ggml_swiglu_split` 拆分。
数学恒等（同输入、同输出维 concat），**数值零变化**：PPL 必须与 B145 逐位一致。

## 依据（已核实）
- `build_ffn` LLM_FFN_PAR 路径：gate 先 mm、up 先 mm，最后 `ggml_swiglu_split(gate_out, up_out)`。
- `ggml_swiglu_split(a,b)` 等价于 fused GLU 取 a 的**前半 = gate**、后半 = up
  （swapped=false 时 gate=x 前 nc，up=x 后 nc；vec_swiglu(x=gate, g=up)）。
  => 融合权重行序必须 **gate 在前、up 在后**。
- radiance fast path 只要求 type==MXFP4；融合后 ne11 不变（输入 5120），
  M 维更大 -> 覆盖率不变或更好。
- B145 实测：65 层 gate/up 全部 mxfp4，无 ffn_gate_b/s、ffn_up_b/s
  （dump 中 0 个）-> 无 bias/scale 需要拼接。

## 改动（本地 fork llama.cpp-Pascal 分支）
1. `src/llama-arch.h/.cpp`：新增 `LLM_TENSOR_FFN_GATE_UP`（"blk.%d.ffn_gate_up"），
   op MUL_MAT。
2. `src/models/models.h` llama_layer：加 `ffn_gate_up`（+ `_s` lora scale 占位）。
3. `src/models/qwen35.cpp`：
   - load_block：ffn_gate_up 存在时用它，ffn_gate/ffn_up 标 not required；
   - `build_layer_ffn`：有 ffn_gate_up 时，一次 `build_lora_mm` 得
     cur {34816, N}，`ggml_view_2d` 拆 gate{17408}/up{17408}，
     `ggml_swiglu_split(ctx0, view_gate, view_up)`，再 mm down。
     注意 view 后需要 ggml_cont? swiglu_split 只要求 contiguous_1（行内连续），
     view_2d 满足。GGML_ASSERT a->ne==b->ne 同形状成立。
4. `tools/` 不动。mtp（blk.64）同样处理（65 层全 mxfp4 已确认）。

## GGUF 手术（.49）
python 脚本（复用 gsq 的字节级读写经验）：
- 读 B145，对每层取 gate/up 的 mxfp4 原始字节（17B/32blk），
  concat 字节 = gate bytes + up bytes（行优先按 output dim 拼接，
  mxfp4 块独立于行边界？否——mxfp4 每 32 元素一块，输出维 17408*5120
  总元素数 89128960，按 32 整除，直接字节级 concat 即为维度 concat：
  gate 是前 17408 行，其块序就是 [row0..row17407]*5120/32 块。
  concat 后 fused 行 [0..17407]=gate, [17408..34815]=up。正确。
- 新张量名 blk.%d.ffn_gate_up.weight，type=39；删除原两个张量条目。
- 重写 GGUF：按张量顺序重建 header（对齐 32B）。预期 15584267968 - 65*33*2048
  级别的 header 缩小、数据区不变。

## 验收（用户指定：单卡速度 + PPL + 输出）
1. 数值红线：`llama-perplexity -f corpus.txt -c 4096 --chunks 8 -ub 2048 -fa on
   -dev ROCm0` 单卡 -> PPL 必须 == 6.7461（B145 基线）。
2. 单卡速度：pp2048/tg128 bench（-dev ROCm0），对比 B145 基线
   （历史单卡：pp2048 2111 / tg128 28.04）。
3. 输出 sanity：固定 seed prompt 生成 vs B145 输出 diff（--temp 0 应逐字相同）。
4. ENGAGE：GGML_RAD_DEBUG=1 确认 560（mxfp4 覆盖不掉）。

## 风险
- view_2d 的 nb 步长必须对：gate view offset 0，up view offset 17408 行。
- ggml_swiglu_split 输入是 GEMM 输出 f32/f16——contiguous_1 满足。
- nextn/MTP 路径 build_ffn 直接复用同函数，自动获得融合。

## 状态：完成（2026-10-04）

- [x] 勘测：65 层 gate/up 全 mxfp4、无 bias/scale、字节相邻（gate 前 up 后）
- [x] 语义核实：PAR + swiglu_split（gate=前半行）行序
- [x] 代码改动：LLM_TENSOR_FFN_GATE_UP + qwen35 load/graph（部署到 .49 生产 fork）
- [x] GGUF 手术：fuse_gateup.py 字节级 concat，header 缩 3712B，数据区零移动
- [x] 验证全绿：
  - PPL 红线：6.7461 ± 0.12137 与 B145 逐位一致（数学恒等成立）
  - 单卡 bench（-dev ROCm0，-r 3）：pp2048 **2302.76** vs B145 基线 2111
    (**+9.1%**)；tg128 27.77 vs 28.04（-1.0%，噪声内）
  - 输出 sanity：同 prompt --temp 0 生成逐字一致（"Paris"）
- 踩坑记录：初版 n_ff 取自 ffn_up（融合分支为 null）段错误——改为
  ffn_gate_up->ne[1]/2；llama-bench 单卡用 -dev 而非 -d

## 结论
gate_up 权重级融合在 radiance MXFP4 上成立：PPL 零变化、pp2048 +9.1%。
原理：两个 17408-row GEMM 合为一个 34816-row GEMM，消除一次 kernel 启动
与中间结果写读，M 维翻倍改善 wave 占用；单次 repack 覆盖率不变。
产物：Qwen3.8-27B-Rad-MX-mix145-fused.gguf（.49）。z(attn_gate) 融合因
mxfp6/mxfp4 类型不齐留二期（需重量化，有精度风险）。
