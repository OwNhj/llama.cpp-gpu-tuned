# MXFP4-RAD 存储类型（路径 B）：radiance 平面布局直落盘

## 目标
新增 GGUF 类型 MXFP4_RAD：数据区直接是 radiance repack 后的平面布局，
load 后 repack 变零拷贝（指针直指权重 buffer），显存从 28GB 降到 15.6GB，
并消除 load 时 repack 时间。

## 布局定义（与 radiance_repack_kernel 输出严格一致）
设张量 [N 行, K 列]，nb = K/32 块：
- **qs 平面**：W[n * (nb*16) + b*16 + m]，16B/块，nibble 为 interleaved 序
  （byte m: lo=elem 2m, hi=elem 2m+1；源 llama split-half 的 lo/hi 各自相邻化）
- **scale 平面**：Ws[b * N + n]，1B/块，e8m0 byte 原样
- **Wref 平面**：每行 max e，[N] bytes
存储顺序：[qs 平面][scale 平面][Wref 平面]，总字节 = N*K/2 + N*K/32 + N
vs 原 mxfp4 = N*K/32*17 = N*K/2 + N*K/32 → **净增 N 字节（Wref），可忽略**

GGML 侧：**fork 上 47 空闲（46=E4M3 已占，105-108 TURBO/ROCMI 系占用，
COUNT=109）→ type id 47**。，type_size 不能表达三个平面 →
**GGML 只当"不透明 blob"处理**：type_size 记 1B？不行，ggml 需 row 结构。
方案：type_size = 17（与 mxfp4 同）只为通过 nbytes 计算？
不行——nbytes = ne[0]/blck_size * type_size * ne[1]...
正确做法：**type_size 记 17、blck_size 32 与 mxfp4 相同**，
但数据排列是每"块行"连续 16B 码 + 单独 scale 平面——ggml 的 row 抽象
只认 [block0][block1]... 连续排列，平面布局不满足。
=> **GGML 层声明为 non-quantized opaque**：to_float 提供一个
dequantize_rad_ref（按平面布局解），type_size = 17, blck_size = 32,
nbytes 计算一致（总字节同为 N*K/32*17，Wref 多出的 N 字节放在
tensor 末尾 padding——nbytes 会差 N/row？N*K/32*17 vs 平面总和 N*K/2+
N*K/32+N = 差 N。把 Wref 并进最后一个 scale 平面尾部空隙不可行）。
简化：**qs 平面按块行混排**：每行先 [16B码 x nb] 再 [nb scale]... 不行，
kernel 要求 Ws 是 [nb,N] 列主序。
=> 最终布局妥协：**平面顺序 qs / Ws / Wref，type_size = 17**，
   nbytes 差额 N 字节通过 blck_size 校正不行。
**定案**：type_size = 17, blck_size = 32（nbytes 与 mxfp4 完全一致，
   差 N 字节）；Wref 不落盘——kernel 端 Wref 改为从 Ws 在线推导
   （row max = max over b of Ws[b*N+n]，GEMM 前 M>=513 一次性算，
   或干脆 Wref 平面存进文件但 type_size 用 17 且最后 N 字节当
   "额外"——ggml nbytes 是硬算的，多 N 字节读不进来）。
   => **Wref 由 CUDA 侧 lazy 计算**（每 tensor 一次，N 字节，开销极小，
   缓存在 g_radiance_weights，已有机制）。文件只存 qs + Ws 平面，
   总字节 = N*K/2 + N*K/32 = 正好 mxfp4 的 nbytes。完美。

## 改动清单
1. ggml.h: GGML_TYPE_MXFP4_RAD = 48; QK 不变
2. ggml.c: 类型表项 {name "mxfp4_rad", type_size 17, blck_size 32,
   to_float = dequantize_row_mxfp4_rad_ref, from_float = quantize...(落
   平面布局)}——from_float 用于 quantize 工具直出
3. ggml-quants.h/.c: quantize_row_mxfp4_rad_ref（f32→平面 interleave）
   + dequantize_row_mxfp4_rad_ref（CPU 路径/校验用，慢无妨）
4. llama-arch / llama-model-loader：无（类型透传）
5. quantize 工具：支持 -t mxfp4_rad
6. **mmq.cu radiance 分派**：type==MXFP4_RAD 时跳过 repack kernel，
   W/Ws 直接 = src0->data + 平面偏移；Wref lazy 算（新小 kernel 或复用）
   ；gate 条件原样
7. **MMVQ（decode）**：src0 现在是平面布局——MMVQ 不认识。两个选择：
   a. MXFP4_RAD 强制全走 radiance 路径（decode 也用 GEMM kernel，M=1
      性能存疑）；
   b. MMVQ 加平面布局 reader（改动大）；
   c. **load 时反变换回 mxfp4？** 违背初衷。
   => 定案 a：先试 decode 直接走 radiance GEMM（M=1 时 tile 利用率低，
      实测 tg128；若掉太多再补 MMVQ reader）
8. graph capture：指针直指 buffer，天然 capture-safe

## 转换器
fuse_gateup.py 同款字节手术：读 B145，对每个 mxfp4 张量做
split-half→interleave 重排 + scale 抽出到平面 → 写 mxfp4_rad 类型。
CPU 单线程 ~12GB 重排，python 太慢 → numpy 向量化或写 C 工具；
numpy: 每 17B 块 reshape 处理，nibble 查表，可行。

## 验收
- 文件大小 ≈ B145（每张量 +0 字节；总差 <1KB）
- 显存驻留：nvidia-smi/rocm-smi 模型部分 ≈ 15.6GB（vs 28GB）
- PPL：与 B145 逐位一致（纯重排零数值变化）
- pp2048：与 radiance 版持平（~2300）
- tg128：decode 走 radiance GEMM 的实测值；>20 t/s 可接受，否则补 b
- ENGAGE=560 不变

## 阶段一结果（2026-10-04，全平面 rad2 布局）

### 关键布局修正：交错 -> 全平面
第一版 to_radi.py 用**行交错**布局（每行 [nb*16 码][nb scale]）——错的：radiance GEMM
要求码是连续 W 平面。后果是 prefill 又要拷一份 W 平面（零拷贝失效），decode 的
unrad 又一份 raw（stride 单位还写错成字节，MMVQ 按 block 索引 -> 越界 -> GPU 队列
hang 污染，反复重启）。改成**全平面** rad2：文件 = [码平面 N*nb*16][scale 平面 N*nb]，
与 radiance 内部 W/Ws 布局逐字节一致。

### 功能验证全绿（rad2，单卡 ROCm0）
- c512 warmup（decode graph 路径，rad 版正是这里 hang）：**1.0731 逐位匹配，不崩**
- c4096 --chunks 8 完整 PPL 红线：**6.7461 ± 0.12137 == B145 逐位一致**
- ENGAGE=1120（560x2pass 覆盖不变），gather 零拷贝生效，token_embd 保持 mxfp4 正确
- 修复三处真 bug：全平面 alias、invalidate_weight_caches 对 alias W 的 double-free、
  MMVQ stride 单位（block 非字节）

### 显存账（诚实结论：阶段一持平，非省）
rad2 让 prefill 的 W 零拷贝（省 11.6GB repack），但 decode 走 MMVQ 仍需 unrad 重建
raw 副本（又 +11.6GB，g_rad_raw 缓存常驻）。净效果 = 把转换从 prefill 挪到 decode，
总量不变：
  B145  = raw(15.58) + W_repack(11.6) + Ws(0.77)
  rad2  = raw(15.58) + decode_raw(11.6) + Ws(0.77)
**真正省 11.6GB 必须阶段二**：decode 也原生读全平面（radiance decode kernel，现被
RADIANCE_MXFP4_DECODE_MAX_M=0 关闭），一个布局通吃 prefill+decode，只需 raw 平面+Ws。
rad2 的文件布局本就是为阶段二准备的——码/ scale 平面即 radiance 期望布局，零拷贝天然成立。

### 速度
rad2 vs B145 单卡 bench：见 bench-rad2.log（prefill 零拷贝应持平或略快，decode unrad
首帧多一次转换、后续走缓存）。

## 状态
- [x] ggml 类型注册（local fork id=47 + .49 生产仓库）
- [x] quant/dequant/unrad/gather ref + CUDA 实现
- [x] radiance 分派零拷贝（全平面 alias）
- [x] 转换器（to_radi2.py，全平面）
- [x] 阶段一验收：PPL 逐位一致 + decode graph 不崩 + ENGAGE 覆盖不变
- [x] 阶段二实测：见下节（负结论）

## 阶段二实测（2026-10-04）：显存承诺不成立，路径 B 关闭

给 `should_use_mmvq` 加 GGML_RAD_DECODE 门控让 radiance decode kernel（M=1..8，
split-K partial）接管 RAD decode，不再走 MMVQ-unrad。三档 peak VRAM（32GB gfx1201，
hip ROCm0 -> smi dev2，llama-cli -c 2048）：

| 配置 | peak VRAM |
|---|---|
| B145（raw + prefill repack） | 30975 MB |
| rad2 + decode 走 MMVQ（W alias，decode unrad） | 31501 MB |
| rad2 + decode 走 radiance kernel（全零拷贝 + split-K scratch） | 31514 MB |

三档都 ~31GB，rad2 **不降反微增**。根因：radiance 用 interleave 码平面、MMVQ 用
split-half raw，两者布局天然不同——单卡 prefill+decode 都要时，运行时总得凑齐两份
（要么文件 raw + prefill 转 interleave，要么文件 interleave + decode 转 raw），文件格式
怎么排都省不掉第二份。rad2+decodeON 本想两份合一（都读 interleave），但 radiance decode
kernel 的 split-K partial scratch 抵消了 unrad 的节省。

**判定：路径 B（MXFP4_RAD 存储类型）在本硬件配置下无法降低峰值显存，作为独立优化路线关闭。**
功能上 rad2 完全正确（PPL 逐位匹配、ENGAGE 覆盖不变、decode 不崩），代码与转换器保留备查：
若将来出现"只 prefill 不 decode"或"显存充裕但想省 repack 时间"的负载，rad2 的加载期零拷贝
可省 ~数秒 repack。但作为省显存手段，负结论，不推广。

**生产维持 B145**（raw 存储 + 运行时 lazy repack），显存 31GB/32GB，无改动。

## 状态（终）
- [x] 全部实现 + 验证
- [x] 阶段一：功能正确，显存持平
- [x] 阶段二：显存实测负结论，路线关闭
