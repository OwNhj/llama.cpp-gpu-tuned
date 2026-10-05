# GSQ: discrete-code refinement for MXFP4 (offline pipeline)

GSQ-style (arXiv 2604.18556, reinterpreted) exact greedy **coordinate descent**
over the E2M1 code assignment of a production MXFP4 GGUF, minimizing
`tr(E^T H E) + lam*||E||F^2` per tensor. Only the 4-bit codes are rewritten -
E8M0 exponents, tensor types, file size and the radiance fast path are all
byte-layout preserved (verified: 285 exponent tensors identical, ENGAGE=560).

Requires: python3 + numpy + torch (CUDA/ROCm build) for the refiner; the
verification tools use numpy only (`gguf` python package for the reader).

## Pipeline (5 steps)

```sh
# 0) env for all steps (model files live outside this repo):
export GSQ_SRC=/path/Qwen3.8-27B-f16.gguf            # bf16/f16 source (for H + tt)
export GSQ_PROD=/path/Qwen3.8-27B-Rad-MX-mix145.gguf # production MXFP4 to refine
export GSQ_IMATRIX=/path/imatrix.dat
export GSQ_HD=$PWD/hess                               # Hessian dir (58 GB for 24 chunks)

# 1) build the calibration corpus (multi-source, anti-leak split):
python3 build_d3.py            # -> calib-d3.txt + eval-{gen,zh,code,wiki}.txt

# 2) collect Hessians (C++ side; see ../../CALIBRATION.md section 2):
llama-imatrix -m $GSQ_SRC -f calib-d3.txt --hessian-dir $GSQ_HD \
  -c 512 --chunks 24 ...

# 3) refine codes on GPU (~30-70 min for the full model; resumable via
#    <out>.progress.json):
GSQ_OUT=gsq-domain.gguf python3 gsq_gpu_final.py 100 2 all280.txt
#    args: <lam> <rounds> [only_file]   (lam=100 default; rounds=2)

# 4) verify byte-preservation: exponents untouched, size identical:
GSQ_REF=gsq-domain.gguf python3 verify_bytes.py

# 5) score per-tensor error (proxy only!) and gate on end-to-end PPL:
GSQ_REF=gsq-domain.gguf GSQ_TTOUT=tt.json python3 gsq_tt_par.py
llama-perplexity -m gsq-domain.gguf -f eval-set.txt -c 4096 --chunks 8 -ub 2048 -fa on ...
```

`all280.txt` is the newline-separated list of type-39 linear tensors to refine
(280 in Qwen3.8-27B; rebuild it for other models with a GGUF scan).

## Decision rules learned the hard way (Qwen3.8-27B, ~12 paired runs)

- **J (per-tensor error) does NOT select winners on this route.** Measured
  pairs: d2 J +1.44% with PPL **-2.64%**; d3/d3b J +1.0% with PPL **worse**.
  Only end-to-end PPL on a corpus from the deployment distribution gates a
  recipe.
- **Calibration distribution == deployment distribution is the whole game.**
  Single-domain calibration won: prose corpus -> -2.64% PPL in-domain,
  -1.33% transferred. Multi-source mixed-window calibration lost: d3b (fixed
  4KB-granularity interleave, leak-free, same-distribution eval) was **+5.8%
  on its own calibration distribution** - mixed-source Hessians are ill-conditioned
  and the conservative rotation trades error into other domains' high-curvature
  directions. Per-domain models, per-domain calibration.
- The effective calibration window is the FIRST `--chunks x 512` tokens of the
  `-f` corpus. `build_d3.py` interleaves sources at 4 KB granularity and
  asserts the 80 KB head-window covers all sources; copy that design for any
  new corpus.
- The refiner keeps ||E||F nearly constant (`fro ~1.0005`) while tr(E^T H E)
  drops ~27%: it *rotates* error rather than removing it. That is fine within
  a matched domain and harmful across domains - hence the rule above.

## Measured results (Qwen3.8-27B, B145 baseline PPL 6.7461)

| variant | calibration | PPL | verdict |
|---|---|---|---|
| r1/r2 | code corpus (B145's own) | -0.48% | below 0.5% bar |
| d2 | same-book non-eval prose | **-2.64%** | winner, deployable |
| lit-calib | eval text itself | -2.73% | leak ceiling only |
| d3 | mixed corpus, 64 KB window | +7.1% | window-granularity bug |
| d3b | mixed corpus, fixed 4 KB window | +5.8% (same-dist) | mixed-source H is harmful, closed |

## GGUF rewriter tools (same byte-level machinery)

- `fuse_gateup.py`  (`GGUF_IN`/`GGUF_OUT`): concat ffn_gate+ffn_up rows into
  one mxfp4 `ffn_gate_up` tensor (data region byte-identical, gate rows first).
  Runtime: Radiance branch loads either layout (see CALIBRATION.md section 4).
- `to_radi2.py` (`GGUF_IN`/`GGUF_OUT`): mxfp4 (type 39) -> MXFP4_RAD (type 47)
  radiance-plane storage layout; zero-copy prefill. See CALIBRATION.md 5.
- `gsq_pack.py selftest` (`GGUF_IN=...`): nibble interleave roundtrip checks.

Full experiment histories: `docs/experiments/design-gsq.md` (this route),
`design-gptq.md`, `design-awq-fold.md` (rejected siblings),
`design-gateup-fuse.md`, `design-mxfp4-rad.md`.

Optional overrides: `GSQ_CORPUS` (source corpus dir, default `./corpus`),
`GSQ_DIR` (output dir, default script dir), `GSQ_PROSE_EN` (single prose file
for build_d3), `GSQ_CPU=1` (force CPU numpy path in verify tools).
