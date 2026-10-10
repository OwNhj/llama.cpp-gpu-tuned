#pragma once

#include "common.cuh"

// radiance MXFP4 x fp8 W8A8 WMMA GEMM fast path (RDNA4 prefill).
// all entries assume ggml_cuda_radiance_supported() was true and buffers are device-side.
void ggml_cuda_radiance_gather_scales(const unsigned char * src_rad, int N, int K,
                                      unsigned char * Ws, unsigned char * Wref,
                                      cudaStream_t stream);

// MXFP6 packed: gather the per-row reference exponent from the checkpoint's own block e[] tails.
// No Ws: the fold reads those tails in situ.
void ggml_cuda_radiance_gather_scales_ptq1(const unsigned char * src, int N, int K,
                                           unsigned char * Wref, cudaStream_t stream);

void ggml_cuda_radiance_gather_scales_mxfp6(const unsigned char * src, int N, int K,
                                            unsigned char * Wref, cudaStream_t stream);

void ggml_cuda_radiance_repack(const void * src_llama, int64_t s01, int N, int K,
                               void * W, void * Ws, void * Wref, cudaStream_t stream);

void ggml_cuda_radiance_quantize_tokens(const float * x, int64_t sx, int64_t K, int64_t M,
                                        unsigned char ** q, float ** scale, cudaStream_t stream);

// int8 activation quantizer for the PTQ1_0 path (separate scratch from the fp8 one).
void ggml_cuda_radiance_quantize_tokens_i8(const float * x, int64_t sx, int64_t K, int64_t M,
                                           signed char ** q, float ** scale, cudaStream_t stream);

void ggml_cuda_radiance_gemm_f32(const void * a_q, const void * w, const void * ws, const void * wref,
                                 const float * as, float * c, int M, int N, int K, cudaStream_t stream,
                                 bool radsc = false, bool e6pack = false, bool ptq1 = false);

bool ggml_cuda_radiance_supported(int cc, ggml_type type, int64_t ne00, int64_t ne01,
                                  int64_t ne11, int64_t ne10, bool contiguous_dst);

// fused GLU fast path (B1): produce down-GEMM fp8 alongside y. lookup/registry span one graph.
bool ggml_cuda_try_swiglu_quant_fused(ggml_backend_cuda_context & ctx, const ggml_tensor * dst);

// fused add + rms norm + weight mul + per-token fp8 (A2). act_key is the tensor whose rows were
// quantized (the norm output mul->data); the following radiance GEMM finds it by that pointer.
bool ggml_cuda_radiance_add_rms_norm_quant(const float * a, const float * b, const float * weight,
                                           float * residual, float * norm, const void * act_key,
                                           int64_t ncols, int64_t nrows, float eps,
                                           cudaStream_t stream);
// transform width the fused PTQ1_0 Hadamard + int8 quantizer is instantiated for
#define FWHT_I8_N 1024

// fused Hadamard transform + int8 activation quantize for the PTQ1_0 prefill path. Writes the
// int8 codes over the transform's own f32 buffer and returns the row scales; null keeps the split.
float * ggml_cuda_radiance_fwht_quant_i8(const void * x, bool x_f32, const float * signs, int n_blk,
                                         void * q, int64_t K, int64_t M, cudaStream_t stream);
bool ggml_cuda_radiance_fwht_i8_reserve(int64_t M, int64_t K, cudaStream_t stream);
// keyed by the consuming matmul's output tensor; a hit is erased, so each consumer is served once
bool ggml_rad_fwht_i8_act_lookup(const void * consumer, int64_t K, int64_t M,
                                 signed char ** q, float ** scale);
void ggml_rad_fwht_i8_act_register(const void * consumer, signed char * q, float * scale,
                                   int64_t K, int64_t M);
void ggml_rad_fwht_i8_acts_reset(void);

// true when the radiance weight caches already hold this weight on the current device.
bool ggml_cuda_radiance_weight_ready(const void * w);

bool ggml_rad_fused_act_lookup(const void * act, int64_t K,
                               const unsigned char ** q, const float ** scale);
void ggml_rad_fused_acts_reset(void);
