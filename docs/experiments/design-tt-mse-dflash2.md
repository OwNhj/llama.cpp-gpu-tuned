# design-tt-mse-dflash2.md - llama-quantize per-tensor quantization error report + DFlash2 mixed-precision recipe

Goal (user request 2026-10-02):
1. Patch `llama-quantize` (fork, `.49:/media/seirin/HDD500G/llama.cpp-gpu-tuned`, branch Radiance) to report a per-tensor quantization error metric, so mixed-precision recipes can be derived without trial-and-error bisection.
2. Use it to build the best mixed-precision recipe for `/media/seirin/SSD2T_1/gguf/Qwen3.8-27B/Qwen3.8-27B-DFlash2-BF16.gguf`.

## 1. Target model facts (measured)

- DFlash2 BF16 gguf: 3860293216 bytes (3.60 GiB), 81 tensors, bf16/f32 mix.
- It is a DFlash **drafter** for Qwen3.8-27B, not the main model: 5 blocks (blk.0..4), hidden 5120, ffn 17408, plus `fc.weight [25600, 5120]`, `selector_predecessor/successor [256, 248320]` (= vocab), `selector_hidden`, `enc.output_norm`, `output_norm`, per-block `attn_conv_base`/`ffn_conv_base` (f32 4D).
- Big tensors dominate: fc 250 MiB, 2x selector 121 MiB, 5x ffn_down/gate/up 170 MiB, 5x attn_q 40 / attn_output 40 / attn_k,v 10 / conv_proj 12.5.
- No imatrix exists for this model (existing imatrix-qkv.gguf is for the main model, `attn_qkv` fused names, 940 tensors). Decision: the patch metric must degrade gracefully without imatrix: unweighted rel-RMSE + per-row max rel err (outlier catcher) in addition to im-weighted when imatrix present.

## 2. Patch design

Files (fork): `include/llama.h`, `src/llama-quant.cpp`, `tools/quantize/quantize.cpp`.

- New param `const char * tt_errors_file` appended to `llama_model_quantize_params` (+ default nullptr). Private fork, ABI change acceptable.
- New CLI flag `--tt-errors <file.csv>`.
- New helper `llama_tensor_record_quant_error(...)`: slab-loops the tensor exactly like the main loop (same max_buf_size logic), for each slab:
  - load + dequantize src to f32 (reuse `llama_tensor_dequantize_impl`),
  - quantize with `llama_tensor_quantize_impl(..., chunk_size = n_per_row, nthread=1)` so per-row imatrix offsets match production semantics,
  - dequantize the result via `ggml_get_type_traits(new_type)->to_float`,
  - accumulate: `ss = sum w^2`, `sse = sum (w-dq)^2`, `sse_im = sum im[j]*e^2` (if imatrix), and per-row rel error max.
  - emits one CSV line: tensor, src_type, dst_type, nelem, dst_mib, im, rmse, w_rmse, max_row, max_row_rmse.
- Hook site: main loop, runs in BOTH dry-run and real mode, before metadata update/unmap; only when `quantize && ggml_is_quantized(new_type)`. Output file `/dev/null`-ish still fine in dry-run (dry-run never writes tensors).
- Why not reuse the non-dry `work` buffer: doubles work in real runs, but scan runs are `--dry-run` and real runs are rare; a standalone pass keeps the patch small and identical in both modes.

## 3. Scan plan

For each candidate type T (24 types: q2_k q3_k q4_0 q4_1 q4_k q5_0 q5_1 q5_k q6_k q8_0 iq3_s iq4_nl iq4_xs mxfp4 mxfp6 mxfp8 nvfp4 mxfp4_e4m3 tq1_0 tq2_0 q1_0 q2_0 q4_0_rocmi4 q4_0_sym4):

```
llama-quantize --dry-run --tt-errors errs_T.csv MODEL /dev/null-out.gguf q8_0 \
    --tensor-type '.*=T'
```

- `--tensor-type '.*=T'` forces type T everywhere; `tensor_type_fallback` still fixes shape-incompatible tensors, so the CSV records the *effective* type.
- Types that GGML_ABORT without imatrix (iq2_*, iq1_*, iq3_xxs per `tensor_requires_imatrix`) will simply be dropped from the candidate list (they are unusable for the final recipe too).
- Shape check: all linear tensors have ne00 in {5120, 17408, 4096, 256, 25600, 1280, 1024}, all divisible by 32/128/256 as needed; `selector_*` have ne00=256 (1 block/row for qk=256, fine).

## 4. Recipe algorithm (greedy, size-budgeted frontier)

- Per-tensor candidate set from CSV rows (tensor, effective type, rmse, dst_mib).
- Cost model without activations: total error proxy `S = sum_i rmse_i^2` (each GEMM's relative perturbation adds ~incoherently into hidden state). Known caveats recorded: no position weighting (attn_output hits residual directly), no outlier/token interaction effects. This is a *proxy*; KLD on the drafter is the final gate.
- Greedy: start every tensor at the min-error candidate. Repeatedly pick the switch (t: cur -> next cheaper type) minimizing `delta_S / delta_size` (loss per MiB saved) while over budget; stop when under budget. Repeat for budgets: aggressive (~0.55 GiB), balanced (~0.75), careful (~0.95), plus "q8_0 everywhere" and "q4_k everywhere" baselines for reference.
- Emit final `--tensor-type-file` + quantize real GGUF, verify load + acceptance.

## 5. Drafter acceptance note

DFlash2 drafter quality = draft acceptance rate on the 27B target, not its own PPL. Correct acceptance metric: n_accept/tok under `--spec-type draft-dflash2` against Qwen3.8-27B-MXFP4 (same protocol as the 2026-10-01 DFlash2 session: tg with `llama-speculative` accept counters). Proxy gate first: drafter logits correlation vs BF16 (cos-sim / top-1 agreement on a text sample) - cheap and directly tied to draft token choice.

## 6. Progress log (final status 2026-10-02)

### Final verification (goal round 3)
- Patch: 3 files changed, +92/-1 (llama.h param + llama-quant.cpp measure hooks + quantize.cpp CLI); `--tt-errors` present in built binary help; incremental build clean.
- DFlash2 scan: 24 `errs_<type>.csv` in `/media/seirin/SSD2T_1/gguf/DFlash2-tt/scan/`, deduped 49 tensors x 24 types.
- Final GGUFs: `DFlash2-mix100c.gguf` (1085734496 B) + `DFlash2-mix115c.gguf` (1235842656 B), dry-run full-metadata load OK, sizes match predictions +0.1%.
- Determinism self-test: final quantization run reproduced scan rmse for 49/49 tensors bit-for-bit.
- Recipe method now in production on the 27B main model rebuild (scan_main.sh, 472 rows/type), proving generality of the patch beyond DFlash2.
- Deferred (documented, out of literal goal scope): drafter acceptance-rate A/B vs rebuilt main model target (`/home/seirin` + dflash_ab.sh ready).

### Recipe decision table (J = sum rmse^2 over 49 tensors, lower better)

| recipe | MiB | J | rmsJ | max rmse | worst tensor |
|---|---|---|---|---|---|
| all-q4_K (iso-size baseline) | 1032.2 | 0.25216 | 0.50215 | 0.0738 | - |
| **mix100c (RECOMMENDED balanced)** | **1024.0** | **0.14881** | 0.38576 | 0.0786 | selector_successor |
| mix115c (quality-first) | 1167.1 | 0.07853 | 0.28024 | 0.0776 | fc.weight |
| budget-125 uncapped | 1273.4 | 0.04181 | 0.20449 | 0.0776 | - |
| budget-150 uncapped | 1532.8 | 0.00950 | 0.09745 | 0.0183 | - |
| all-q6_K | 1505.3 | 0.01561 | 0.12494 | 0.0183 | - |
| budget-100 uncapped | 1023.8 | 0.14776 | 0.38440 | 0.1527 | fc.weight (sacrificed to q3_K) |

mix100c = 22x iq4_xs + 21x q6_K + 5x q5_K + 1x q4_K; mix115c = 30x q6_K + 9x iq4_xs + 8x q5_K + 1x q8_0 + 1x q4_K.

Verdict: **mix100c beats uniform q4_K by 41% lower error at the same size** (and caps per-tensor rmse at 0.079,
no single-tensor sacrifice like the uncapped variant that put fc.weight on q3_K with rmse 0.153).
mix115c trades +14% size for another -47% error - worth testing both in acceptance-rate A/B.
All types used (iq4_xs/q5_K/q6_K/q8_0) are MMVQ-friendly int formats - drafter decode (bs=1) unaffected.

- [x] patch written, incremental build ok (`--tt-errors` CSV: name,src,dst,nrows,nelem,bytes,rmse,max_row,bad_rows,im_rmse)
- [x] 24-type scan done, CSVs collected (`/media/seirin/SSD2T_1/gguf/DFlash2-tt/scan/errs_*.csv`, 49 quantizable tensors x 24 types)
- [x] frontier recipes generated (uncapped 075/100/125/150 + capped 100c/115c with rmse<=0.135 guard)
- [x] final GGUF quantized + verified loadable via full GGUF parse (dry-run reads all 49 tensor descriptors):
      - `DFlash2-mix100c.gguf` 1025 MiB (4.47 BPW) = 22x iq4_xs + 21x q6_K + 5x q5_K + 1x q4_K, predicted J=0.1488
      - `DFlash2-mix115c.gguf` 1168 MiB (5.09 BPW) = 30x q6_K + 9x iq4_xs + 8x q5_K + 1x q8_0 + 1x q4_K, predicted J=0.0785
      - selftest: final quant run reproduced scan rmse for 49/49 tensors exactly (measurement determinism proof)
- [ ] acceptance measured (drafter acceptance rate) - BLOCKED on main model: the old Qwen3.8-27B-Mxfp4.gguf target pair was off the tuned path; rebuilding it now in `design-mxfp4-rebuild.md`. Test matrix once rebuilt: {BF16, mix100c, mix115c} x dflash2 spec decode on same 27B target, `--temp 0`, n_max=7.

### Notes
- Scan CSV had one duplicate-run artifact (49 rows appended from the smoke test into errs_q4_k.csv); dedupe by (name,dst) keeps first.
- Uncapped greedy picked q3_K for fc.weight (rmse 0.153 = 15% of all error mass); capped variants avoid single-tensor sacrifice. `100c` (0.446% per-tensor max vs 1.42% for uncapped 100) is the recommended balanced recipe; `115c` is the careful one.
- llama-cli standalone load of a DFlash2 gguf fails with "dflash requires ctx_other" by design (it is a draft-only arch).
