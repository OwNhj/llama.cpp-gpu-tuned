# Calibration pipeline and quant-layout options (fork features)

Availability:
- Calibration tooling (sections 1-3): all fork branches (`master`, `llama.cpp-Pascal`, `Radiance`)
- Fused gate|up (section 4) and MXFP4_RAD storage type (section 5): `Radiance` branch only

All features are **default-off**. Stock GGUFs (type 39/45/etc.) load and run
bit-identically with or without these changes: paired PPL 6.7461 on
Qwen3.8-27B verified in the same build, same GPU.

## 1. Per-tensor quantization error report (`--tt-errors`)

```sh
llama-quantize src-f16.gguf out.gguf MXFP4 --tensor-type <recipe> --tt-errors errors.csv
```

One CSV row per quantized weight tensor: name, type, im_rmse / im_mae
(importance-weighted when `--imatrix` is supplied). Use it to score
mixed-precision recipes without full eval runs.

Caveat measured on Qwen3.8-27B: the J aggregate over this CSV does NOT predict
end-to-end PPL on calibration-driven routes (GPTQ: J -31.9% yet PPL +2.59%;
GSQ: J +0.7..+1.4% with PPL -2.6..-0.5%). Always gate a recipe on paired
`llama-perplexity` runs (same build, `--temp 0` where sampling is involved).

## 2. Hessian export (`--hessian-dir`)

```sh
llama-imatrix -m model-f16.gguf -f corpus.txt --hessian-dir hess/ \
  -c 512 --chunks 24 -dev ROCm0,ROCm1,ROCm2 -ts 2/1/1 --split-mode layer -ngl 65 \
  --load-mode none --override-tensor "token_embd.weight=CPU,output.weight=CPU"
```

Writes per-tensor upper-triangle Hessians (f64) into `hess/` (~58 GB for
Qwen3.8-27B at 24 chunks; single-GPU use `-ngl 99` with an f16-fitting model).
Consumers: the GPTQ path below and offline coordinate-descent / OBS solvers
the GSQ solver side is in `scripts/gsq/` (see scripts/gsq/README.md).

Corpus-shape rule learned the hard way: the effective calibration window is the
**first `--chunks x 512` tokens** of `-f`. For multi-domain corpora, interleave
at small (~4 KB) sample granularity inside that head; concatenating whole source
files trains only the first one.

## 3. GPTQ-style requantization (`--gptq-u-dir`)

Preprocess: Cholesky-factor each collected Hessian into U factors (scripts in
`/media/seirin/SSD2T_1/hf/gsq/` on the tuning rig), then

```sh
llama-quantize src-f16.gguf out.gguf MXFP4 --imatrix imatrix.dat --gptq-u-dir U/
```

Result on Qwen3.8-27B: **rejected** (PPL 2.59% worse than plain grid-search
MXFP4). Kept for per-domain experiments only: the mxfp4 E2M1 grid + E8M0 scale
pair leaves the solver far less freedom than int4 affine grids, and
compensation is distribution-bound (trades calibration-window error against
off-window error, which usually loses end-to-end).

## 4. Fused gate|up weights (Radiance branch)

`blk.N.ffn_gate_up.weight` (output rows = gate half first, up half second,
out-dim = 2 x ffn) replaces the separate ffn_gate/ffn_up pair when present.
The graph builds one GEMM and feeds row views into the existing swiglu split.
Same-input weight concat is a mathematical identity: PPL unchanged (6.7461
bit-exact), measured pp2048 +9.1% single-GPU on Qwen3.8-27B MXFP4 (2302 vs
2111), decode unchanged. Models with either layout load unchanged - old GGUFs
are untouched.

Rewriter: `fuse_gateup.py` (byte-exact concat for same-type adjacent blocks;
mixed types or per-block scales need a proper merger).

## 5. MXFP4_RAD storage type (type 47, Radiance branch)

mxfp4 tensors pre-arranged in the radiance kernel plane layout:

```
per tensor: [ N x K/2 interleaved-qs plane ][ N x K/32 row-major e8m0 scale plane ]
```

Same byte budget as stock mxfp4 (17 B per 32 elements). Convert with
`to_radi2.py`, then load normally - no flags needed:

- Prefill fast path: the W plane aliases the weight buffer (zero-copy repack);
  scales + row max gathered once per tensor. ENGAGE coverage identical.
- Decode: MMVQ rebuilds a cached split-half raw copy per tensor on first use;
  or set `GGML_RAD_DECODE=1` plus `RADIANCE_MXFP4_DECODE_MAX_M=8` to route
  decode through the radiance decode kernel directly (no raw copy; K/N shape
  constraints apply).
- Measured on Qwen3.8-27B (single 32 GB RDNA4 card): PPL bit-exact vs mxfp4,
  tg128 within noise, **peak VRAM unchanged** - the two runtime layouts still
  coexist (prefill wants interleaved planes, MMVQ decode wants split-half raw).
  Treat type 47 as load-time repack savings, not a memory optimization.

## Environment switches (radiance, Radiance branch)

| variable | effect |
|---|---|
| `GGML_RAD_DISABLE=1` | turn off the prefill fast path (any mxfp4 type) |
| `GGML_RAD_PREFILL_MIN_M` | prefill threshold, default 256 |
| `GGML_RAD_DEBUG=1` | per-call dispatch log (ENGAGE / decline + reason) |
| `GGML_RAD_DECODE=1` | keep MXFP4_RAD decode off MMVQ (radiance decode kernel) |
| `RADIANCE_MXFP4_DECODE_MAX_M` | radiance decode batch ceiling, default 0 (off) |
