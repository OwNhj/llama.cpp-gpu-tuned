#include "common.cuh"
#include "mmq.cuh"
#include "quantize.cuh"
#include "mmid.cuh"
#include "radiance-gemm.cuh"

#include <cstdint>
#include <cstdlib>
#include <unordered_map>


// --- MXFP4_RAD decode fallback helpers (definitions live in the radiance section below) ---
struct ggml_radiance_weight;


// forward decls from radiance-gemm.cu (gfx1200 single TU build)
__global__ void radiance_unrad_kernel(const unsigned char * __restrict__ src_rad, int N, int nb,
                                      unsigned char * __restrict__ dst_raw);
void ggml_cuda_radiance_gather_scales(const unsigned char * src_rad, int N, int K,
                                      unsigned char * Ws, unsigned char * Wref,
                                      cudaStream_t stream);

static void ggml_cuda_mul_mat_q_switch_type(ggml_backend_cuda_context & ctx, const mmq_args & args, cudaStream_t stream, const ggml_prec prec_src1) {
    switch (args.type_x) {
        case GGML_TYPE_Q1_0:
            mul_mat_q_case<GGML_TYPE_Q1_0>(ctx, args, stream);
            break;
        case GGML_TYPE_Q2_0:
            mul_mat_q_case<GGML_TYPE_Q2_0>(ctx, args, stream);
            break;
        case GGML_TYPE_PQ2_0:
            mul_mat_q_case<GGML_TYPE_PQ2_0>(ctx, args, stream);
            break;
        case GGML_TYPE_PTQ1_0:
            mul_mat_q_case<GGML_TYPE_PTQ1_0>(ctx, args, stream);
            break;
        case GGML_TYPE_Q4_0:
            mul_mat_q_case<GGML_TYPE_Q4_0>(ctx, args, stream);
            break;
        case GGML_TYPE_Q4_1:
            mul_mat_q_case<GGML_TYPE_Q4_1>(ctx, args, stream);
            break;
        case GGML_TYPE_Q5_0:
            mul_mat_q_case<GGML_TYPE_Q5_0>(ctx, args, stream);
            break;
        case GGML_TYPE_Q5_1:
            mul_mat_q_case<GGML_TYPE_Q5_1>(ctx, args, stream);
            break;
        case GGML_TYPE_Q8_0:
            mul_mat_q_case<GGML_TYPE_Q8_0>(ctx, args, stream);
            break;
// -----------------------------------------------------------------------
        case GGML_TYPE_Q2_K:
            mul_mat_q_case<GGML_TYPE_Q2_K>(ctx, args, stream);
            break;
        case GGML_TYPE_Q3_K:
            mul_mat_q_case<GGML_TYPE_Q3_K>(ctx, args, stream);
            break;
        case GGML_TYPE_Q4_K:
            mul_mat_q_case<GGML_TYPE_Q4_K>(ctx, args, stream);
            break;
        case GGML_TYPE_Q5_K:
            mul_mat_q_case<GGML_TYPE_Q5_K>(ctx, args, stream);
            break;
        case GGML_TYPE_Q6_K:
            mul_mat_q_case<GGML_TYPE_Q6_K>(ctx, args, stream);
            break;
// -----------------------------------------------------------------------
        case GGML_TYPE_IQ1_S:
            mul_mat_q_case<GGML_TYPE_IQ1_S>(ctx, args, stream);
            break;
        case GGML_TYPE_IQ2_XXS:
            mul_mat_q_case<GGML_TYPE_IQ2_XXS>(ctx, args, stream);
            break;
        case GGML_TYPE_IQ2_XS:
            mul_mat_q_case<GGML_TYPE_IQ2_XS>(ctx, args, stream);
            break;
        case GGML_TYPE_IQ2_S:
            mul_mat_q_case<GGML_TYPE_IQ2_S>(ctx, args, stream);
            break;
        case GGML_TYPE_IQ3_XXS:
            mul_mat_q_case<GGML_TYPE_IQ3_XXS>(ctx, args, stream);
            break;
        case GGML_TYPE_IQ3_S:
            mul_mat_q_case<GGML_TYPE_IQ3_S>(ctx, args, stream);
            break;
        case GGML_TYPE_IQ4_XS:
            mul_mat_q_case<GGML_TYPE_IQ4_XS>(ctx, args, stream);
            break;
        case GGML_TYPE_IQ4_NL:
            mul_mat_q_case<GGML_TYPE_IQ4_NL>(ctx, args, stream);
            break;
// -----------------------------------------------------------------------
// -----------------------------------------------------------------------
        case GGML_TYPE_Q4_0_ROCMI4:
            mul_mat_q_case<GGML_TYPE_Q4_0_ROCMI4>(ctx, args, stream);
            break;
        case GGML_TYPE_Q4_0_SYM4:
            mul_mat_q_case<GGML_TYPE_Q4_0_SYM4>(ctx, args, stream);
            break;
        case GGML_TYPE_MXFP4:
            // src1 at Q4 uses the native FP4 instructions, which are Blackwell-only
            if (prec_src1 == GGML_PREC_Q4) {
                mul_mat_q_case<GGML_TYPE_MXFP4, GGML_PREC_Q4>(ctx, args, stream);
                break;
            }
            mul_mat_q_case<GGML_TYPE_MXFP4>(ctx, args, stream);
            break;
        case GGML_TYPE_MXFP4_RAD:
            // zero-copy: the RAD tile loader reads the interleaved code plane and the row-major
            // scale plane straight from args.x. the scale-plane offset assumes one contiguous
            // [N, nb*16] code plane (2D tensor); MXFP4_RAD has no 3D/MoE producer, so abort
            // instead of reading out of bounds.
            if (args.nchannels_x != 1) {
                GGML_ABORT("MXFP4_RAD MMQ only supports 2D tensors");
            }
            mul_mat_q_case<GGML_TYPE_MXFP4_RAD>(ctx, args, stream);
            break;
        case GGML_TYPE_NVFP4:
            if (prec_src1 == GGML_PREC_Q4) {
                mul_mat_q_case<GGML_TYPE_NVFP4, GGML_PREC_Q4>(ctx, args, stream);
                break;
            }
            mul_mat_q_case<GGML_TYPE_NVFP4>(ctx, args, stream);
            break;
        case GGML_TYPE_MXFP8:
            mul_mat_q_case<GGML_TYPE_MXFP8>(ctx, args, stream);
            break;
        case GGML_TYPE_MXFP6:
            mul_mat_q_case<GGML_TYPE_MXFP6>(ctx, args, stream);
            break;
        case GGML_TYPE_MXFP4_E4M3:
            mul_mat_q_case<GGML_TYPE_MXFP4_E4M3>(ctx, args, stream);
            break;
        default:
            GGML_ABORT("fatal error");
            break;
    }
}

struct ggml_radiance_weight {
    void * W;
    void * Ws;
    void * Wref;
    int    device;
    bool   own_w = true;   // false when W aliases the weight buffer (MXFP4_RAD)
    bool   e6pack = false; // MXFP6 packed: read the checkpoint bytes in place, fold to e4m3
};

static std::unordered_map<const void *, ggml_radiance_weight> g_radiance_weights;

static void ggml_cuda_free_radiance_weights() {
    // W may point into the model weight buffer (MXFP4_RAD zero-copy); only owned
    // allocations are freed. Track ownership via a flag inside the entry.
    for (auto & kv : g_radiance_weights) {
        if (kv.second.own_w) cudaFree(kv.second.W);
        cudaFree(kv.second.Ws);
        cudaFree(kv.second.Wref);
    }
    g_radiance_weights.clear();
}

// true when the radiance weight caches already hold this weight on the current device, so a
// matmul reading it cannot lose the radiance path to a capture-time repack miss.
bool ggml_cuda_radiance_weight_ready(const void * w) {
    auto it = g_radiance_weights.find(w);
    return it != g_radiance_weights.end() && it->second.device == ggml_cuda_get_device();
}

// the repack cache is keyed by the weight tensor's device pointer. when a model's CUDA buffer
// is freed the keys become dangling: the next load may reuse the same address (stale repacked
// weights) or a different one (leaked repacks -> OOM across reloads). drop every cache entry
// whose key falls inside the freed range.
void ggml_cuda_invalidate_weight_caches(const void * base, size_t size) {
    const char * lo = (const char *) base;
    const char * hi = lo + size;
    for (auto it = g_radiance_weights.begin(); it != g_radiance_weights.end();) {
        const char * k = (const char *) it->first;
        if (k >= lo && k < hi) {
            if (it->second.own_w) cudaFree(it->second.W);   // RAD aliases the weight buffer: never free it
            cudaFree(it->second.Ws);
            cudaFree(it->second.Wref);
            it = g_radiance_weights.erase(it);
        } else {
            ++it;
        }
    }
}

// radiance W8A8 fp8 WMMA prefill fast path for MXFP4 weights. returns true when handled.
static bool ggml_cuda_mul_mat_q_radiance(ggml_backend_cuda_context & ctx, const ggml_tensor * src0,
                                         const ggml_tensor * src1, const ggml_tensor * ids, ggml_tensor * dst) {
    if (getenv("GGML_RAD_DISABLE")) {
        return false;
    }
    if (ids) {
        return false;
    }
    if (src0->ne[2] != 1 || src1->ne[2] != 1 || src0->ne[3] != 1 || src1->ne[3] != 1) {
        return false;
    }

    const int cc     = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    const int device = ggml_cuda_get_device();

    const int64_t K    = src0->ne[0];             // also src1->ne[0]
    const int64_t N    = src0->ne[1];             // dst rows
    const int64_t M    = src1->ne[1];             // tokens

    const bool contig_w  = src0->nb[1] == ggml_row_size(src0->type, K);
    const bool contig_a  = src1->nb[1] == K * sizeof(float);
    const bool contig_dst = dst->nb[1] == N * sizeof(float) && dst->nb[2] == dst->nb[1] * M;
    const bool dbg = getenv("GGML_RAD_DEBUG") != nullptr;
    if (!contig_w || !contig_a || !contig_dst) {
        if (dbg) fprintf(stderr, "[rad] decline contig w=%d a=%d dst=%d K=%ld N=%ld M=%ld\n",
                          contig_w, contig_a, contig_dst, (long)K, (long)N, (long)M);
        return false;
    }
    if (!ggml_cuda_radiance_supported(cc, src0->type, K, N, M, src1->ne[0], contig_dst)) {
        if (dbg) fprintf(stderr, "[rad] decline policy K=%ld N=%ld M=%ld type=%d\n",
                         (long)K, (long)N, (long)M, (int)src0->type);
        return false;
    }
    if (dbg) fprintf(stderr, "[rad] ENGAGE K=%ld N=%ld M=%ld\n", (long)K, (long)N, (long)M);

    cudaStream_t stream = ctx.stream();

    // one-time repack of the weight tensors (weights are static across calls)
    const void * key = src0->data;
    auto it = g_radiance_weights.find(key);
    if (it == g_radiance_weights.end() || it->second.device != device) {
        // capture-safe contract: repack allocates; a miss while the stream is capturing must
        // fall through to MMQ instead of the illegal cudaMalloc inside the capture.
        cudaStreamCaptureStatus cs = cudaStreamCaptureStatusNone;
        if (cudaStreamIsCapturing(stream, &cs) == cudaSuccess && cs != cudaStreamCaptureStatusNone) {
            if (dbg) fprintf(stderr, "[rad] decline capture-miss K=%ld N=%ld M=%ld\n", (long)K, (long)N, (long)M);
            return false;
        }
        ggml_radiance_weight w;
        w.device = device;
        const size_t wq_bytes  = (size_t)N * K / 2;
        const size_t ws_bytes  = (size_t)N * (K / 32);
        // PTQ1_0 keeps the row normaliser as fp16; every other type uses one e8m0 byte.
        const size_t wref_bytes = src0->type == GGML_TYPE_PTQ1_0 ? (size_t) 2 * N : (size_t) N;
        if (src0->type == GGML_TYPE_MXFP6) {
            // MXFP6 zero-copy: the atiled E6PACK path reads the checkpoint's packed 6-bit bytes
            // in place and folds them to e4m3, so W aliases the weight buffer and there is no
            // scale plane to stage (each source block carries its own e[] tail, read in situ).
            // Wref (row max exponent, N bytes) comes from a gather over those tails.
            w.own_w  = false;
            w.e6pack = true;
            w.W      = (unsigned char *) src0->data;
            w.Ws     = nullptr;
            CUDA_CHECK(cudaMalloc(&w.Wref, wref_bytes));
            ggml_cuda_radiance_gather_scales_mxfp6((const unsigned char *) src0->data, N, K,
                                                   (unsigned char *) w.Wref, stream);
        } else if (src0->type == GGML_TYPE_PTQ1_0) {
            // PTQ1_0 zero-copy: the atiled PTQ1 staging reads the checkpoint's 28-byte
            // blocks in place and normalises each block's fp16 d against the row max, so
            // W aliases the weight buffer and there is no scale plane. Only Wref (the row
            // max exponent, one byte per row) is materialised.
            w.own_w = false;
            w.W     = (unsigned char *) src0->data;
            w.Ws    = nullptr;
            CUDA_CHECK(cudaMalloc(&w.Wref, wref_bytes));
            ggml_cuda_radiance_gather_scales_ptq1((const unsigned char *) src0->data, (int) N, (int) K,
                                                  (unsigned char *) w.Wref, stream);
        } else if (src0->type == GGML_TYPE_MXFP4_RAD) {
            w.own_w = false;   // W aliases the weight buffer (full-plane rad2 layout)
            // zero-copy: the atiled RADSC path reads the row-major scale plane straight from
            // the weight buffer (W + N*K/2), so no transposed Ws is allocated. Wref (row max
            // exponent, N bytes) still comes from the gather pass with Ws == nullptr.
            w.W = (unsigned char *) src0->data;   // code plane aliases the weight buffer
            w.Ws = nullptr;
            CUDA_CHECK(cudaMalloc(&w.Wref, wref_bytes));
            ggml_cuda_radiance_gather_scales((const unsigned char *) src0->data, N, K, nullptr, (unsigned char *) w.Wref, stream);
        } else {
            CUDA_CHECK(cudaMalloc(&w.W,    wq_bytes));
            CUDA_CHECK(cudaMalloc(&w.Ws,   ws_bytes));
            CUDA_CHECK(cudaMalloc(&w.Wref, wref_bytes));
            ggml_cuda_radiance_repack(src0->data, K / 32, N, K, w.W, w.Ws, w.Wref, stream);
        }
        CUDA_CHECK(cudaGetLastError());
        it = g_radiance_weights.emplace(key, w).first;
    }
    const ggml_radiance_weight & w = it->second;

    // activation fp8: prefer the q the fused GLU already produced; else quantize from f32
    const bool ptq1 = src0->type == GGML_TYPE_PTQ1_0;
    unsigned char * q8 = nullptr;
    float * as = nullptr;
    // PTQ1_0 needs int8 activations, so it must not consume the e4m3 buffer the fused
    // producers (add_rms/swiglu) register, nor the fp8 quantizer's row-fingerprint memo.
    if (!ptq1 && ggml_rad_fused_act_lookup(src1->data, K, (const unsigned char **) &q8, (const float **) &as)) {
        // reuse the fused producer's e4m3
    } else if (ptq1) {
        signed char * q8s = nullptr;
        // A fused Hadamard producer may already have written this activation's int8 codes and row
        // scales; only PTQ1_0 activations ever land in that registry (see ggml_cuda_try_fwht_quant_i8).
        if (!ggml_rad_fwht_i8_act_lookup(dst->data, K, M, &q8s, &as)) {
            ggml_cuda_radiance_quantize_tokens_i8((const float *) src1->data, src1->nb[1] / sizeof(float),
                                                  K, M, &q8s, &as, stream);
            if (q8s == nullptr) {
                return false;   // act scratch realloc declined under capture, fall back to MMQ
            }
        }
        q8 = (unsigned char *) q8s;
    } else {
        ggml_cuda_radiance_quantize_tokens((const float *) src1->data, src1->nb[1] / sizeof(float),
                                           K, M, &q8, &as, stream);
        if (q8 == nullptr) {
            return false;   // act scratch realloc declined under capture, fall back to MMQ
        }
    }

    ggml_cuda_radiance_gemm_f32(q8, w.W, w.Ws, w.Wref, as,
                                (float *) dst->data, (int) M, (int) N, (int) K, stream,
                                src0->type == GGML_TYPE_MXFP4_RAD, w.e6pack, ptq1);
    CUDA_CHECK(cudaGetLastError());
    return true;
}

// overrides the src1 precision requested by the graph, "auto" keeps the requested one
static ggml_prec ggml_cuda_mmq_get_prec_env() {
    const char * env_c = getenv("GGML_CUDA_MMQ_PREC");
    if (env_c == nullptr) {
        return GGML_PREC_UNDEFINED;
    }
    std::string env_cpp = env_c;
    for (char & c : env_cpp) {
        c = std::tolower(c);
    }
    if (env_cpp == "q4") {
        return GGML_PREC_Q4;
    }
    if (env_cpp == "q8") {
        return GGML_PREC_Q8;
    }
    if (env_cpp != "auto") {
        GGML_LOG_WARN("%s: Unknown value for GGML_CUDA_MMQ_PREC: '%s'. Available: 'q4', 'q8', 'auto'.\n", __func__, env_cpp.c_str());
    }
    return GGML_PREC_UNDEFINED;
}

// src1 is quantized to Q8_1 unless the FP4 types can use 4-bit activations, in which case they
// default to the native W4A4 instructions on Blackwell.
static ggml_prec ggml_cuda_mmq_get_prec_src1(const ggml_tensor * src0, const ggml_tensor * dst, const int cc) {
    static const ggml_prec prec_env = ggml_cuda_mmq_get_prec_env();

    ggml_prec prec = prec_env;
    if (prec == GGML_PREC_UNDEFINED) {
        prec = (ggml_prec) ggml_get_op_params_i32(dst, 3);
    }

    // Q4 only for the FP4 types on Blackwell
    GGML_ASSERT(prec == GGML_PREC_UNDEFINED || prec == GGML_PREC_Q8 || prec == GGML_PREC_Q4);
    const bool can_use_q4 = (src0->type == GGML_TYPE_NVFP4 || src0->type == GGML_TYPE_MXFP4 || src0->type == GGML_TYPE_MXFP4_RAD) && blackwell_mma_available(cc);
    if (prec == GGML_PREC_Q8 || !can_use_q4) {
        return GGML_PREC_Q8;
    }
    return GGML_PREC_Q4;}

static size_t ggml_cuda_mmq_q8_buffer_size(
        ggml_type type, bool fallback, int cc,
        int64_t ne10_padded, int64_t ne11, int64_t ne12, int64_t ne13);

void ggml_cuda_mul_mat_q(
        ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids,
        ggml_tensor * dst, const ggml_tensor * gate,
        const ggml_tensor * norm_weight, const ggml_tensor * norm_scale,
        void * external_q8, bool quantize_external, const float * external_norm_scale) {
    GGML_ASSERT(        src1->type == GGML_TYPE_F32);
    GGML_ASSERT(        dst->type  == GGML_TYPE_F32);
    GGML_ASSERT(!ids || ids->type  == GGML_TYPE_I32); // Optional, used for batched GGML_MUL_MAT_ID.
    GGML_ASSERT(!gate || (!ids && gate->type == GGML_TYPE_F32));
    GGML_ASSERT((norm_weight == nullptr) == (norm_scale == nullptr && external_norm_scale == nullptr));
    GGML_ASSERT(!norm_weight || (!ids && !gate && norm_weight->type == GGML_TYPE_F32 &&
                ((norm_scale && norm_scale->type == GGML_TYPE_F32) || external_norm_scale)));

    GGML_TENSOR_BINARY_OP_LOCALS;

    if (ggml_cuda_mul_mat_q_radiance(ctx, src0, src1, ids, dst)) {
        return;
    }

    cudaStream_t stream = ctx.stream();
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;

    const size_t ts_src0 = ggml_type_size(src0->type);
    const size_t ts_src1 = ggml_type_size(src1->type);
    const size_t ts_dst  = ggml_type_size(dst->type);

    GGML_ASSERT(        nb00       == ts_src0);
    GGML_ASSERT(        nb10       == ts_src1);
    GGML_ASSERT(        nb0        == ts_dst);
    GGML_ASSERT(!ids || ids->nb[0] == ggml_type_size(ids->type));

    const char  * src0_d = (const char  *) src0->data;
    const float * src1_d = (const float *) src1->data;
    float       *  dst_d = (float       *)  dst->data;

    // If src0 is a temporary compute buffer, clear any potential padding.
    if (ggml_backend_buffer_get_usage(src0->buffer) == GGML_BACKEND_BUFFER_USAGE_COMPUTE) {
        const size_t size_data  = ggml_nbytes(src0);
        const size_t size_alloc = ggml_backend_buffer_get_alloc_size(src0->buffer, src0);
        if (size_alloc > size_data) {
            GGML_ASSERT(ggml_is_contiguously_allocated(src0));
            GGML_ASSERT(!src0->view_src);
            CUDA_CHECK(cudaMemsetAsync((char *) src0->data + size_data, 0, size_alloc - size_data, stream));
        }
    }

    const int64_t ne10_padded = GGML_PAD(ne10, MATRIX_ROW_PADDING);

    const int64_t s01 = src0->nb[1] / ts_src0;
    const int64_t s1  =  dst->nb[1] / ts_dst;
    const int64_t s02 = src0->nb[2] / ts_src0;
    const int64_t s2  =  dst->nb[2] / ts_dst;
    const int64_t s03 = src0->nb[3] / ts_src0;
    const int64_t s3  =  dst->nb[3] / ts_dst;

    const bool fallback = ne01 % 128 != 0;

    const ggml_prec prec_src1 = ggml_cuda_mmq_get_prec_src1(src0, dst, cc);

    const bool use_native_fp4 = prec_src1 == GGML_PREC_Q4;
    const size_t y_block_size       = use_native_fp4 ? sizeof(block_fp4_mmq) : sizeof(block_q8_1_mmq);
    const size_t y_values_per_block = use_native_fp4 ? QK_FP4_MMQ            : QK8_1_MMQ;

    if (!ids) {
        const size_t nbytes_src1_q8_1 = use_native_fp4 ?
            ne13*ne12*ne11*ne10_padded*y_block_size/y_values_per_block +
                ggml_cuda_mmq_get_J_max(src0->type, fallback, cc, ne11)*sizeof(block_q8_1_mmq) :
            ggml_cuda_mmq_q8_buffer_size(src0->type, fallback, cc, ne10_padded, ne11, ne12, ne13);
        GGML_ASSERT(!external_q8 || (!gate && !use_native_fp4));
        ggml_cuda_pool_alloc<char> src1_q8_1(ctx.pool());
        char * src1_q8_ptr = external_q8 ? (char *) external_q8 : src1_q8_1.alloc(nbytes_src1_q8_1);
        ggml_cuda_pool_alloc<float> src1_scale(ctx.pool());
        if (src0->type == GGML_TYPE_NVFP4 && use_native_fp4) {
            src1_scale.alloc(ne13*ne12*ne11);
        }

        if (!external_q8 || quantize_external) {
            const int64_t s11 = src1->nb[1] / ts_src1;
            const int64_t s12 = src1->nb[2] / ts_src1;
            const int64_t s13 = src1->nb[3] / ts_src1;
            if (use_native_fp4) {
                static constexpr size_t align_float8 = 32;
                const bool use_aligned_float8 = ggml_cuda_is_aligned(src1, align_float8);
                static_assert(sizeof(block_fp4_mmq) == 4 * sizeof(block_q8_1));
                    quantize_mmq_fp4_cuda(src1_d, nullptr, src1_q8_ptr, src1_scale.ptr, src0->type, use_aligned_float8, ne10, s11, s12, s13, ne10_padded,
                                        ne11, ne12, ne13, stream);

            } else if ((src0->type == GGML_TYPE_MXFP8 || src0->type == GGML_TYPE_MXFP6 || src0->type == GGML_TYPE_MXFP4 || src0->type == GGML_TYPE_MXFP4_RAD || src0->type == GGML_TYPE_MXFP4_E4M3)
                       && GGML_CUDA_CC_IS_RDNA4(cc)) {
                // e4m3 y tiles for the W8A8 fp8 WMMA path (RDNA4 only). NVFP4 is deliberately
                // NOT here: its MMQ vec_dot is the mainline int8 one (q8_0_16), which consumes
                // q8_1 int8 y. Producing e4m3 y while running an int8 vec_dot feeds int8 code
                // e4m3 bytes and corrupts the result, so the two choices have to move together.
                quantize_mmq_mxfp8_cuda(src1_d, nullptr, src1_q8_1.get(), src0->type, ne10, s11, s12, s13, ne10_padded,
                                       ne11, ne12, ne13, stream);
            } else {
                if (norm_weight) {
                    GGML_ASSERT(norm_weight->ne[0] == src1->ne[0] && ggml_nrows(norm_weight) == 1);
                    GGML_ASSERT(ggml_is_contiguous(norm_weight));
                    GGML_ASSERT(external_norm_scale || (ggml_is_contiguous(norm_scale) &&
                                ggml_nelements(norm_scale) >= ggml_nrows(src1)));
                    quantize_mmq_q8_1_rms_cuda(src1_d, (const float *) norm_weight->data,
                                               external_norm_scale ? external_norm_scale : (const float *) norm_scale->data,
                                               src1_q8_ptr, src0->type,
                                               ne10, s11, s12, s13, ne10_padded, ne11, ne12, ne13, stream);
                } else if (gate) {
                    GGML_ASSERT(ggml_are_same_shape(src1, gate));
                    GGML_ASSERT(ggml_is_contiguous(gate));
                    quantize_mmq_q8_1_swiglu_cuda(src1_d, (const float *) gate->data, src1_q8_ptr, src0->type,
                                                  ne10, s11, s12, s13, ne10_padded, ne11, ne12, ne13, stream);
                } else {
                    quantize_mmq_q8_1_cuda(src1_d, nullptr, src1_q8_ptr, src0->type, ne10, s11, s12, s13, ne10_padded,
                                           ne11, ne12, ne13, stream);
                }
            }
            CUDA_CHECK(cudaGetLastError());
        }

        // Stride depends on quantization format
        const int64_t s12 = use_native_fp4 ?
                                ne11 * ne10_padded * sizeof(block_fp4_mmq) / (QK_FP4_MMQ * sizeof(int)) :
                                ne11 * ne10_padded * sizeof(block_q8_1) / (QK8_1 * sizeof(int));
        const int64_t s13 = ne12*s12;

        const mmq_args args = {
            src0_d, src0->type, (const int *) src1_q8_ptr, nullptr, nullptr, dst_d,
            src0->type == GGML_TYPE_NVFP4 && use_native_fp4 ? src1_scale.ptr : nullptr,
            ne00, ne01, ne1, s01, ne11, s1,
            ne02, ne12, s02, s12, s2,
            ne03, ne13, s03, s13, s3,
            ne1, ne1};
        ggml_cuda_mul_mat_q_switch_type(ctx, args, stream, prec_src1);
        return;
    }

    GGML_ASSERT(ne13 == 1);
    GGML_ASSERT(nb12 % nb11 == 0);
    GGML_ASSERT(nb2  % nb1  == 0);

    const int64_t n_expert_used = ids->ne[0];
    const int64_t ne_get_rows = ne12 * n_expert_used;
    GGML_ASSERT(ne1 == n_expert_used);

    ggml_cuda_pool_alloc<int32_t> ids_src1(ctx.pool(), ne_get_rows);
    ggml_cuda_pool_alloc<int32_t> ids_dst(ctx.pool(), ne_get_rows);
    ggml_cuda_pool_alloc<int32_t> expert_bounds(ctx.pool(), ne02 + 1);

    // gate/up activations are broadcast across experts (ne11 == 1): quantize each token once and
    // scatter to its slots. ids_src1 then holds the inverse map (token slot -> compact row).
    const bool dedup_bcast = ne11 == 1 && n_expert_used > 1;

    {
        GGML_ASSERT(ids->nb[0] == ggml_element_size(ids));
        const int si1  = ids->nb[1] / ggml_element_size(ids);
        const int sis1 = nb12 / nb11;

        ggml_cuda_launch_mm_ids_helper((const int32_t *) ids->data, ids_src1.get(), ids_dst.get(), expert_bounds.get(),
            ne02, ne12, n_expert_used, ne11, si1, sis1, /*write_inverse =*/ dedup_bcast, stream);
        CUDA_CHECK(cudaGetLastError());
    }

    const size_t nbytes_src1_q8_1 = ne12*n_expert_used*ne10_padded * y_block_size/y_values_per_block +
        ggml_cuda_mmq_get_J_max(src0->type, fallback, cc, ne12) * sizeof(block_q8_1_mmq);
    ggml_cuda_pool_alloc<char> src1_q8_1(ctx.pool(), nbytes_src1_q8_1);
    ggml_cuda_pool_alloc<float> src1_scale(ctx.pool());
    if (src0->type == GGML_TYPE_NVFP4 && use_native_fp4) {
        src1_scale.alloc(ne12*n_expert_used);
    }

    const int64_t ne11_flat = ne12*n_expert_used;
    const int64_t ne12_flat = 1;
    const int64_t ne13_flat = 1;

    {
        const int64_t s11 = src1->nb[1] / ts_src1;
        const int64_t s12 = src1->nb[2] / ts_src1;
        const int64_t s13 = src1->nb[3] / ts_src1;

        if (use_native_fp4) {
            static constexpr size_t align_float8 = 32;
            const bool use_aligned_float8 = ggml_cuda_is_aligned(src1, align_float8);
            if (dedup_bcast) {
                quantize_scatter_mmq_fp4_cuda(src1_d, ids_src1.get(), src1_q8_1.get(), src1_scale.ptr, src0->type, use_aligned_float8, ne10,
                                        /*stride_token=*/s12, ne10_padded, ne12, ne11_flat, n_expert_used, stream);
            } else {
                quantize_mmq_fp4_cuda(src1_d, ids_src1.get(), src1_q8_1.get(), src1_scale.ptr, src0->type, use_aligned_float8, ne10, s11, s12, s13,
                                        ne10_padded, ne11_flat, ne12_flat, ne13_flat, stream);
            }
        } else if ((src0->type == GGML_TYPE_MXFP8 || src0->type == GGML_TYPE_MXFP6 || src0->type == GGML_TYPE_MXFP4 || src0->type == GGML_TYPE_MXFP4_RAD || src0->type == GGML_TYPE_MXFP4_E4M3)
                   && GGML_CUDA_CC_IS_RDNA4(cc)) {
            // e4m3 y tiles for the W8A8 fp8 WMMA path (RDNA4 only); other archs fall through
            // to the q8_1 int8 y consumed by the mainline int8 vec_dots. NVFP4 is excluded here
            // for the same reason as in ggml_cuda_mul_mat_q: its MMQ vec_dot is the int8 one
            // (q8_0_16), so e4m3 y would be read as int8 code and corrupt the result.
            if (dedup_bcast) {
                quantize_scatter_mmq_mxfp8_cuda(src1_d, ids_src1.get(), src1_q8_1.get(), src0->type, ne10,
                                        /*stride_token=*/s12, ne10_padded, ne12, ne11_flat, n_expert_used, stream);
            } else {
                quantize_mmq_mxfp8_cuda(src1_d, ids_src1.get(), src1_q8_1.get(), src0->type, ne10, s11, s12, s13,
                                       ne10_padded, ne11_flat, ne12_flat, ne13_flat, stream);
            }
        } else if (dedup_bcast) {
            quantize_scatter_mmq_q8_1_cuda(src1_d, ids_src1.get(), src1_q8_1.get(), src0->type, ne10,
                                    /*stride_token=*/s12, ne10_padded, ne12, ne11_flat, n_expert_used, stream);
        } else {
            quantize_mmq_q8_1_cuda(src1_d, ids_src1.get(), src1_q8_1.get(), src0->type, ne10, s11, s12, s13,
                                   ne10_padded, ne11_flat, ne12_flat, ne13_flat, stream);
        }
        CUDA_CHECK(cudaGetLastError());
    }

    static_assert(QK_FP4_MMQ == 8 * QK_MXFP4, "QK_FP4_MMQ needs to be 8 * QK_MXFP4");
    const int64_t s12 = use_native_fp4 ? ne11 * ne10_padded * sizeof(block_fp4_mmq) / (QK_FP4_MMQ * sizeof(int)) :
                                         ne11 * ne10_padded * sizeof(block_q8_1) / (QK8_1 * sizeof(int));
    const int64_t s13 = ne12*s12;

    // Note that ne02 is used instead of ne12 because the number of y channels determines the z dimension of the CUDA grid.
    const mmq_args args = {
        src0_d, src0->type, (const int *) src1_q8_1.get(), ids_dst.get(), expert_bounds.get(), dst_d,
        src1_scale.ptr,
        ne00, ne01, ne_get_rows, s01, ne_get_rows, s1,
        ne02, ne02, s02, s12, s2,
        ne03, ne13, s03, s13, s3,
        ne12};

    ggml_cuda_mul_mat_q_switch_type(ctx, args, stream, prec_src1);
}

static size_t ggml_cuda_mmq_q8_buffer_size(
        ggml_type type, bool fallback, int cc,
        int64_t ne10_padded, int64_t ne11, int64_t ne12, int64_t ne13) {
    return ne13*ne12*ne11*ne10_padded*sizeof(block_q8_1_mmq)/QK8_1_MMQ +
        ggml_cuda_mmq_get_J_max(type, fallback, cc, ne11)*sizeof(block_q8_1_mmq);
}

size_t ggml_cuda_mul_mat_q_q8_size(const ggml_tensor * src0, const ggml_tensor * src1) {
    GGML_ASSERT(src0 && src1 && src1->type == GGML_TYPE_F32);
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    return ggml_cuda_mmq_q8_buffer_size(src0->type, src0->ne[1] % 128 != 0, cc,
        GGML_PAD(src1->ne[0], MATRIX_ROW_PADDING), src1->ne[1], src1->ne[2], src1->ne[3]);
}

void ggml_cuda_mul_mat_q_fused_two(
        ggml_backend_cuda_context & ctx,
        const ggml_tensor * src0_a, const ggml_tensor * src0_b, const ggml_tensor * src1,
        ggml_tensor * dst_a, ggml_tensor * dst_b,
        const ggml_tensor * norm_weight, const float * norm_scale) {
    GGML_ASSERT(src0_a && src0_b && src1 && dst_a && dst_b && norm_weight && norm_scale);
    GGML_ASSERT(src0_a->type == src0_b->type && src1->type == GGML_TYPE_F32 &&
                dst_a->type == GGML_TYPE_F32 && dst_b->type == GGML_TYPE_F32);
    GGML_ASSERT(src0_a->ne[0] == src1->ne[0] && src0_b->ne[0] == src1->ne[0]);
    GGML_ASSERT(src0_a->ne[2] == src0_b->ne[2] && src0_a->ne[3] == src0_b->ne[3]);
    GGML_ASSERT(ggml_is_contiguous(src1) && ggml_is_contiguous(norm_weight));
    GGML_ASSERT(norm_weight->type == GGML_TYPE_F32 && norm_weight->ne[0] == src1->ne[0] &&
                ggml_nrows(norm_weight) == 1);

    cudaStream_t stream = ctx.stream();
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    const int64_t ne10 = src1->ne[0];
    const int64_t ne11 = src1->ne[1];
    const int64_t ne12 = src1->ne[2];
    const int64_t ne13 = src1->ne[3];
    const int64_t ne10_padded = GGML_PAD(ne10, MATRIX_ROW_PADDING);
    const bool fallback = src0_a->ne[1] % 128 != 0;
    GGML_ASSERT(fallback == (src0_b->ne[1] % 128 != 0));

    const size_t nbytes_src1_q8_1 = ggml_cuda_mmq_q8_buffer_size(
        src0_a->type, fallback, cc, ne10_padded, ne11, ne12, ne13);
    ggml_cuda_pool_alloc<char> src1_q8_1(ctx.pool(), nbytes_src1_q8_1);
    const int64_t s11 = src1->nb[1] / sizeof(float);
    const int64_t s12_src = src1->nb[2] / sizeof(float);
    const int64_t s13_src = src1->nb[3] / sizeof(float);
    quantize_mmq_q8_1_rms_cuda((const float *) src1->data, (const float *) norm_weight->data,
                               norm_scale, src1_q8_1.get(), src0_a->type,
                               ne10, s11, s12_src, s13_src, ne10_padded, ne11, ne12, ne13, stream);
    CUDA_CHECK(cudaGetLastError());

    const int64_t stride_q_channel = ne11*ne10_padded*sizeof(block_q8_1_mmq)/(QK8_1_MMQ*sizeof(int));
    const int64_t stride_q_sample = ne12*stride_q_channel;
    auto launch_one = [&](const ggml_tensor * src0, ggml_tensor * dst) {
        GGML_ASSERT(src0->type == src0_a->type && src0->ne[0] == ne10);
        const size_t ts0 = ggml_type_size(src0->type);
        const mmq_args args = {
            (const char *) src0->data, src0->type, (const int *) src1_q8_1.ptr, nullptr, nullptr, (float *) dst->data,
            nullptr,
            src0->ne[0], src0->ne[1], dst->ne[1], (int64_t)(src0->nb[1]/ts0), ne11, (int64_t)(dst->nb[1]/sizeof(float)),
            src0->ne[2], ne12, (int64_t)(src0->nb[2]/ts0), stride_q_channel, (int64_t)(dst->nb[2]/sizeof(float)),
            src0->ne[3], ne13, (int64_t)(src0->nb[3]/ts0), stride_q_sample, (int64_t)(dst->nb[3]/sizeof(float)),
            dst->ne[1]};
        ggml_cuda_mul_mat_q_switch_type(ctx, args, stream, GGML_PREC_Q8);
    };
    launch_one(src0_a, dst_a);
    launch_one(src0_b, dst_b);
}

bool ggml_cuda_should_use_mmq(enum ggml_type type, int cc, int64_t ne11, int64_t n_experts) {
#ifdef GGML_CUDA_FORCE_CUBLAS
    return false;
#endif // GGML_CUDA_FORCE_CUBLAS

    bool mmq_supported;

    switch (type) {
        case GGML_TYPE_PTQ1_0:
            mmq_supported = true; // HIP: RDNA4 WMMA (I8/I4) paths in mmq.cuh
            break;
        case GGML_TYPE_Q1_0:
        case GGML_TYPE_Q2_0:
        case GGML_TYPE_PQ2_0:
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q4_1:
        case GGML_TYPE_Q5_0:
        case GGML_TYPE_Q5_1:
        case GGML_TYPE_Q8_0:
// -------------------------------------------------
        case GGML_TYPE_Q2_K:
        case GGML_TYPE_Q3_K:
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q5_K:
        case GGML_TYPE_Q6_K:
// -------------------------------------------------
        case GGML_TYPE_IQ1_S:
        case GGML_TYPE_IQ2_XXS:
        case GGML_TYPE_IQ2_XS:
        case GGML_TYPE_IQ2_S:
        case GGML_TYPE_IQ3_XXS:
        case GGML_TYPE_IQ3_S:
        case GGML_TYPE_IQ4_XS:
        case GGML_TYPE_IQ4_NL:
// -------------------------------------------------
        case GGML_TYPE_MXFP4:
        case GGML_TYPE_MXFP4_RAD:
        case GGML_TYPE_NVFP4:
        case GGML_TYPE_MXFP8:
        case GGML_TYPE_MXFP6:
        case GGML_TYPE_MXFP4_E4M3:
        case GGML_TYPE_Q4_0_ROCMI4:
        case GGML_TYPE_Q4_0_SYM4:
            mmq_supported = true;
            break;
        default:
            mmq_supported = false;
            break;
    }

    // The MXFP8/MXFP6 W8A8 fp8 WMMA path is implemented for RDNA4 only.
    if ((type == GGML_TYPE_MXFP8 || type == GGML_TYPE_MXFP6 || type == GGML_TYPE_MXFP4_E4M3) && !GGML_CUDA_CC_IS_RDNA4(cc)) {
        return false;
    }

    if (!mmq_supported) {
        return false;
    }

    // MMQ tiles require at least 48 KiB per-block shared memory; fall back to BLAS otherwise.
    {
        const int    id    = ggml_cuda_get_device();
        const size_t smpbo = ggml_cuda_info().devices[id].smpbo;
        if (smpbo < 48 * 1024) {
            return false;
        }
    }

    if (type == GGML_TYPE_PTQ1_0) {
        // the fp16 dequantize + cuBLAS fallback is the source of PTQ1_0's extra error on CUDA, so
        // the MMQ tile path runs at every batch by default; the env var is the A/B knob for
        // deployments that prefer cuBLAS's ~7% at pp512 over the accuracy
        static const int64_t max_batch = [] {
            const char * s = getenv("GGML_CUDA_PTQ1_0_MMQ_MAX_BATCH");
            return s ? (int64_t) atoll(s) : (int64_t) MMQ_PTQ1_0_MAX_BATCH_SIZE;
        }();
        return ne11 <= max_batch;
    }

    if (turing_mma_available(cc)) {
        return true;
    }

    if (ggml_cuda_highest_compiled_arch(cc) < GGML_CUDA_CC_DP4A) {
        return false;
    }

#ifdef GGML_CUDA_FORCE_MMQ
    return true;
#endif //GGML_CUDA_FORCE_MMQ

    if (GGML_CUDA_CC_IS_NVIDIA(cc)) {
        return !fp16_mma_hardware_available(cc) || ne11 < MMQ_DP4A_MAX_BATCH_SIZE;
    }

    if (amd_mfma_available(cc)) {
        // As of ROCM 7.0 rocblas/tensile performs very poorly on CDNA3 and hipblaslt (via ROCBLAS_USE_HIPBLASLT)
        // performs better but is currently suffering from a crash on this architecture.
        // TODO: Revisit when hipblaslt is fixed on CDNA3
        if (GGML_CUDA_CC_IS_CDNA3(cc)) {
            return true;
        }
        if (n_experts > 64 || ne11 <= 128) {
            return true;
        }
        if (type == GGML_TYPE_Q4_0 || type == GGML_TYPE_Q4_1 || type == GGML_TYPE_Q5_0 || type == GGML_TYPE_Q5_1) {
            return true;
        }
        if (ne11 <= 256 && (type == GGML_TYPE_Q4_K || type == GGML_TYPE_Q5_K)) {
            return true;
        }
        return false;
    }

    if (amd_wmma_available(cc)) {
        if (GGML_CUDA_CC_IS_RDNA3(cc)) {
            // High expert counts are almost always better on MMQ due to
            //     the synchronization overhead in the cuBLAS/hipBLAS path:
            // https://github.com/ggml-org/llama.cpp/pull/18202
            if (n_experts >= 64) {
                return true;
            }

            // For some quantization types MMQ can have lower peak TOPS than hipBLAS
            //     so it's only faster for sufficiently small batch sizes:
            switch (type) {
                case GGML_TYPE_Q2_K:
                    return ne11 <= 128;
                case GGML_TYPE_Q6_K:
                    return ne11 <= (GGML_CUDA_CC_IS_RDNA3_0(cc) ? 128 : 256);
                case GGML_TYPE_IQ2_XS:
                case GGML_TYPE_IQ2_S:
                    return GGML_CUDA_CC_IS_RDNA3_5(cc) || ne11 <= 128;
                default:
                    return true;
            }
        }

        // For RDNA4 MMQ is consistently faster than dequantization + hipBLAS:
        // https://github.com/ggml-org/llama.cpp/pull/18537#issuecomment-3706422301
        return true;
    }

    // gfx900 (Vega 10) lacks native dp4a, loses to dequant + hipBLAS
    // for dense matrices; keep MMQ only for MoE, where the
    // hipBLAS path is much slower.
    if (cc == GGML_CUDA_CC_VEGA) {
        return n_experts > 0;
    }

    return (!GGML_CUDA_CC_IS_CDNA(cc)) || ne11 < MMQ_DP4A_MAX_BATCH_SIZE;
}

// free cached radiance weight buffers at process exit (fork-local simplification)
static struct ggml_radiance_weight_dtor {
    ~ggml_radiance_weight_dtor() { ggml_cuda_free_radiance_weights(); }
} ggml_radiance_weight_dtor_instance;
