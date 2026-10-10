#include "common.cuh"
#include <hip/hip_runtime.h>
#include <map>
#include <hip/hip_bf16.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <unordered_map>
#include <stdexcept>
typedef float floatx8 __attribute__((ext_vector_type(8)));
typedef int int32x8_t __attribute__((ext_vector_type(8)));
typedef int int2_t __attribute__((ext_vector_type(2)));
typedef unsigned int uint4_t __attribute__((ext_vector_type(4)));
typedef unsigned int uint2_t __attribute__((ext_vector_type(2)));
__device__ __constant__ unsigned char kLUT[16] = {
    0x00, 0x30, 0x38, 0x3C, 0x40, 0x44, 0x48, 0x4C,
    0x80, 0xB0, 0xB8, 0xBC, 0xC0, 0xC4, 0xC8, 0xCC};

// kMag[d]: the 8 e2m1 magnitudes, each already shifted DOWN by d binades and encoded as an e4m3
// byte, packed 4 per word so `v_perm_b32` can resolve four lookups per instruction. Folding the MX
// block exponent into the WEIGHT here removes the per-32-block rescale from the inner loop
// entirely -- the scale becomes one per-row factor applied in the epilogue. The sign is separable
// (a negative code is the positive byte with bit 7 set), which is why only magnitudes are tabulated.
//
// The table runs into e4m3 SUBNORMALS, and that is deliberate. Stopping at the smallest e4m3 NORMAL
// (2^-6) makes the fold exact only while d <= 5, and this checkpoint exceeds that: mlp.down_proj
// reaches d=10 and holds 11,354 of the model's 14,433 out-of-range blocks. On layer 58 that zeroed
// 6.7% of the weights on 36 output channels, for per-channel errors up to 18.7% and a 1.18%
// whole-layer norm error -- squarely inside the range CHECKALL flags. Extending into subnormals
// (down to 2^-9) makes every d <= 8 exact and leaves d=9..12 rounding rather than flushing:
// measured, every affected layer with max d <= 8 goes to exactly zero error, and layer 58 (the only
// d=10 one) drops 4x to 0.31%.
//
// This is only sound because the gfx12 fp8 WMMA HONOURS e4m3 subnormals rather than flushing them
// -- verified on hardware, exact for every m*2^-9, m=1..7 (~/mxfp4_work/tier6/subnorm.hip).
__device__ __constant__ unsigned int kMag[16][2] = {
  {0x3c383000u, 0x4c484440u},
  {0x34302800u, 0x44403c38u},
  {0x2c282000u, 0x3c383430u},
  {0x24201800u, 0x34302c28u},
  {0x1c181000u, 0x2c282420u},
  {0x14100800u, 0x24201c18u},
  {0x0c080400u, 0x1c181410u},
  {0x06040200u, 0x14100c08u},
  {0x03020100u, 0x0c080604u},
  {0x01010000u, 0x06040302u},
  {0x01000000u, 0x03020101u},
  {0x00000000u, 0x01010100u},
  {0x00000000u, 0x01000000u},
  {0x00000000u, 0x00000000u},
  {0x00000000u, 0x00000000u},
  {0x00000000u, 0x00000000u},
};

// E2M3 (MXFP6) staging. The 6-bit code is split across two planes -- MXFP4's nibble plane carries
// bits 0-3, a second plane 2 bits per weight -- and rejoins here. Every E2M3 magnitude is k/8, so
// every code is an exact e4m3 byte and the block exponent still folds into it, but only 6 binades
// below the row's largest (the builder clamps the spread and the loader rechecks it), hence 7 rows
// of d. Per d: t0/t1 are the 8 subnormal bytes (exponent field 0, magnitude k/8) for one v_perm,
// and b4 is the bias the normal half takes as ONE bytewise add, since its e4m3 byte is affine in
// the code. The fourth word pads the row to a uint4 for the A-tiled kernel's LDS copy.
__device__ __constant__ unsigned int kE2M3[7][4] = {
  {0x2c282000u, 0x36343230u, 0x30303030u, 0u},
  {0x24201800u, 0x2e2c2a28u, 0x28282828u, 0u},
  {0x1c181000u, 0x26242220u, 0x20202020u, 0u},
  {0x14100800u, 0x1e1c1a18u, 0x18181818u, 0u},
  {0x0c080400u, 0x16141210u, 0x10101010u, 0u},
  {0x06040200u, 0x0e0c0a08u, 0x08080808u, 0u},
  {0x03020100u, 0x07060504u, 0x00000000u, 0u},
};

static __device__ __forceinline__ unsigned int e2m3_spread2(unsigned int x) {
  return (__umul24(x & 0x3003u, 0x00010010u) | __umul24(x & 0x0330u, 0x00001100u)) & 0x30303030u;
}
static __device__ __forceinline__ unsigned int e2m3_fold4(unsigned int c6, unsigned int t0,
                                                          unsigned int t1, unsigned int b4) {
  const unsigned int sub = __builtin_amdgcn_perm(t1, t0, c6 & 0x07070707u);
  const unsigned int arit = (c6 & 0x1F1F1F1Fu) + b4;
  const unsigned int nz = (((c6 >> 3) & 0x03030303u) + 0x03030303u) & 0x04040404u;
  return __builtin_amdgcn_perm(arit, sub, 0x03020100u | nz) | ((c6 & 0x20202020u) << 2);
}
static __device__ __forceinline__ uint2_t e2m3_unpack8(unsigned int wv, unsigned int h2,
                                                       unsigned int t0, unsigned int t1,
                                                       unsigned int b4) {
  const unsigned int be = e2m3_fold4((wv & 0x0F0F0F0Fu) | e2m3_spread2(h2), t0, t1, b4);
  const unsigned int bo = e2m3_fold4(((wv >> 4) & 0x0F0F0F0Fu) | e2m3_spread2(h2 >> 2), t0, t1, b4);
  return uint2_t{__builtin_amdgcn_perm(bo, be, 0x05010400u),
                 __builtin_amdgcn_perm(bo, be, 0x07030602u)};
}
// MXFP6 checkpoint order -> the two e2m3_unpack8 inputs. Codes are packed little-endian at six
// bits each (code j at bit 6*j), so a 24-bit half holds exactly four of them. The split is the
// point: 6 = 4 + 2 lands on e2m3_unpack8's wv (nibble j = elem j & 0xF, two elements per byte)
// and h2 (2-bit field j = elem j >> 4) with no cross-lane exchange, which is why MXFP6 needs no
// repack where MXFP4's split-half nibbles do.
static __device__ __forceinline__ void e2m3_split8(unsigned long long w, unsigned int & wv, unsigned int & h2) {
  const unsigned int t0 = (unsigned int) (w & 0xFFFFFFu);
  const unsigned int t1 = (unsigned int) ((w >> 24) & 0xFFFFFFu);
  const unsigned int l0 = (t0 & 0x00000Fu) | ((t0 >>  2) & 0x0000F0u) | ((t0 >>  4) & 0x000F00u) | ((t0 >>  6) & 0x00F000u);
  const unsigned int l1 = (t1 & 0x00000Fu) | ((t1 >>  2) & 0x0000F0u) | ((t1 >>  4) & 0x000F00u) | ((t1 >>  6) & 0x00F000u);
  const unsigned int u0 = ((t0 >>  4) & 0x000003u) | ((t0 >>  8) & 0x00000Cu) | ((t0 >> 12) & 0x000030u) | ((t0 >> 16) & 0x0000C0u);
  const unsigned int u1 = ((t1 >>  4) & 0x000003u) | ((t1 >>  8) & 0x00000Cu) | ((t1 >> 12) & 0x000030u) | ((t1 >> 16) & 0x0000C0u);
  wv = l0 | (l1 << 16);
  h2 = u0 | (u1 << 8);
}

#define BM 128
#define BN 64
#define BK 64
#define PAD 8
#define ASTR (BK + PAD)
#define WSTR (BK + PAD)
#define NWAVE 8

__global__ __launch_bounds__(NWAVE * 32) void radiance_mxfp4_fp8_gemm(
    const unsigned char *__restrict__ A, const unsigned char *__restrict__ W,
    const unsigned char *__restrict__ Ws, const float *__restrict__ As,
    __bf16 *__restrict__ C, int M, int N, int K) {
  __shared__ unsigned char sA[BM * ASTR];
  __shared__ unsigned char sW[BN * WSTR];
  __shared__ unsigned char sS[BN * (BK / 32)];

  const int tid = threadIdx.x, lane = tid & 31, wave = tid >> 5;
  const int wm = wave >> 1, wn = wave & 1;
  const int col = lane & 15, kb8 = (lane >> 4) * 8;
  const int m0 = blockIdx.y * BM, n0 = blockIdx.x * BN;
  const int nblk = K / 32;

  floatx8 acc[2][2];
#pragma unroll
  for (int i = 0; i < 2; ++i)
#pragma unroll
    for (int j = 0; j < 2; ++j)
#pragma unroll
      for (int e = 0; e < 8; ++e) acc[i][j][e] = 0.f;

  for (int k0 = 0; k0 < K; k0 += BK) {
    {   // A tile: 32 B per thread. CLAMP, NEVER PREDICATE: a bounds-predicated staging load
        // lands in its own s_and_saveexec region and the compiler cannot carry a counted
        // s_wait_loadcnt across an exec-mask merge, so it emits s_wait_loadcnt 0x0 after every
        // load -- one full memory round trip each where one per slab would do. Clamped, all the
        // slab's loads issue back to back under a single wait. Safe because the epilogue already
        // drops rows past M and columns past N, and a clamped read returns real data. Measured on
        // the int4 twin (autoround-tests/RESULTS.md, 2026-08-26): worth 7-26% at decode and
        // 5-20% at prefill. Applies to every staging load below.
      const int r = tid >> 1, off = (tid & 1) * 32;
      const int rc = r < M - 1 - m0 ? r : M - 1 - m0;
      const unsigned char *src = A + (size_t)(m0 + rc) * K + k0 + off;
      const uint4_t v0 = *(const uint4_t *)(src);
      const uint4_t v1 = *(const uint4_t *)(src + 16);
      *(uint4_t *)(&sA[r * ASTR + off]) = v0;
      *(uint4_t *)(&sA[r * ASTR + off + 16]) = v1;
    }
    {   // W tile: 8 packed bytes per thread -> 16 e4m3 bytes, guarded on the N tail
      const int r = tid >> 2, off = (tid & 3) * 8;
      const int rc = r < N - 1 - n0 ? r : N - 1 - n0;              // clamp, never predicate
      const unsigned long long packed =
          *(const unsigned long long *)(W + ((size_t)(n0 + rc) * K + k0) / 2 + off);
      unsigned char out[16];
#pragma unroll
      for (int t = 0; t < 8; ++t) {
        const unsigned char byte = (packed >> (8 * t)) & 0xFF;
        out[2 * t] = kLUT[byte & 0xF];
        out[2 * t + 1] = kLUT[(byte >> 4) & 0xF];
      }
      *(uint4_t *)(&sW[r * WSTR + off * 2]) = *(const uint4_t *)out;
    }
    if (tid < BN * (BK / 32)) {   // scales are [K/32, N]: coalesced across n
      const int r = tid / (BK / 32), b = tid % (BK / 32);
      const int gn = n0 + r;
      const int gc = gn < N - 1 ? gn : N - 1;                      // clamp, never predicate
      sS[tid] = Ws[(size_t)(k0 / 32 + b) * N + gc];
    }
    __syncthreads();

#pragma unroll
    for (int blk = 0; blk < BK / 32; ++blk) {
      floatx8 tmp[2][2];
#pragma unroll
      for (int i = 0; i < 2; ++i)
#pragma unroll
        for (int j = 0; j < 2; ++j)
#pragma unroll
          for (int e = 0; e < 8; ++e) tmp[i][j][e] = 0.f;
#pragma unroll
      for (int step = 0; step < 2; ++step) {     // two WMMA steps span one 32-wide MX block
        const int kk = blk * 32 + step * 16 + kb8;
        int2_t af[2], wf[2];
#pragma unroll
        for (int i = 0; i < 2; ++i) {
          const unsigned char *p = &sA[(wm * 32 + i * 16 + col) * ASTR + kk];
          af[i][0] = *(const int *)p; af[i][1] = *(const int *)(p + 4);
        }
#pragma unroll
        for (int j = 0; j < 2; ++j) {
          const unsigned char *p = &sW[(wn * 32 + j * 16 + col) * WSTR + kk];
          wf[j][0] = *(const int *)p; wf[j][1] = *(const int *)(p + 4);
        }
#pragma unroll
        for (int i = 0; i < 2; ++i)
#pragma unroll
          for (int j = 0; j < 2; ++j)
            tmp[i][j] = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(af[i], wf[j], tmp[i][j]);
      }
#pragma unroll
      for (int j = 0; j < 2; ++j) {
        // E8M0 is a power of two by definition, so 2^(e-127) is exactly the IEEE-754 float
        // whose exponent field is e: bits = e << 23. One integer shift instead of a
        // transcendental, worth +4% (exp2f ran 4x per lane per K-slab). e=0 would mean 2^-127
        // (denormal) and yields 0.0f here; it only arises for all-zero blocks, where the
        // product is 0 either way.
        const float s = __int_as_float(
            (int)sS[(wn * 32 + j * 16 + col) * (BK / 32) + blk] << 23);
#pragma unroll
        for (int i = 0; i < 2; ++i)
#pragma unroll
          for (int e = 0; e < 8; ++e) acc[i][j][e] += tmp[i][j][e] * s;
      }
    }
    __syncthreads();
  }

#pragma unroll
  for (int i = 0; i < 2; ++i)
#pragma unroll
    for (int j = 0; j < 2; ++j)
#pragma unroll
      for (int e = 0; e < 8; ++e) {
        const int m = m0 + wm * 32 + i * 16 + kb8 + e;
        const int n = n0 + wn * 32 + j * 16 + col;
        if (m < M && n < N) C[(size_t)m * N + n] = (__bf16)(acc[i][j][e] * As[m]);
      }
}


// Autotuned folded-scale kernel. The MX block exponent is pushed into the weight byte during the
// LDS upconvert (kMag), so the inner loop is pure WMMA -- no temp accumulator, no per-32-block
// rescale, no LDS scale reads. Lossless while the shifted value stays an e4m3 normal (d <= 5):
// 99.998% of the checkpoint's 761M blocks qualify, worst case 10, 282 blocks flush.
//
// Tile config from a correctness-gated sweep of 28 shapes (BM/BN/BK/TM/TN/wave grid). BM=256 via
// TM=4 won at EVERY production shape and batch size -- a taller tile buys more A reuse per weight
// read. It also removed the large-M cliff the earlier hand-written BM=128 version had.
#define TM 4
#define WM 4
#define WN 2
#undef NWAVE
#define NWAVE (WM * WN)
#define NTHREADS (NWAVE * 32)
#define BMF (WM * TM * 16)
#define BNF_OF(tn) (WN * (tn) * 16)
#define BNF BNF_OF(2)

// TN is templated because the winner depends on M. Widening the N tile (TN=4 -> BNF=128) amortises
// the A-tile staging, which an ablation prices at 24% of runtime -- A is BMF*BK = 16 KB per K-slab
// against W's 2 KB, and with BNF=64 it is re-staged across twice as many column blocks. Measured on
// gate_up 17408x5120, tight A/B, 10 alternations with no overlap between the distributions:
//   M=8192 +10.0%  M=6144 +9.7%  M=4096 +8.5%  M=3072 +6.9%
//   M=2048 +1.3%   M=1536 -1.6%  M=1024 -4.2%  M=512 -8.8%
// Below the crossover the wide tile cannot fill: it halves the block count in N while BMF still
// masks most of a short M. Hence TN=4 only from M >= 2048; RADIANCE_MXFP4_TN4_MIN_M overrides.
// WPERM selects the weight layout. false is the checkpoint's own [N, K/2]; true is fragment
// order, one 32-lane uint32 slot per (n-tile, k-step), which is what libr4d's
// r4d_gemm_mxfp4a8_nt_m64 reads and what makes a wave's weight read 128 contiguous bytes instead
// of sixteen rows K/2 bytes apart. Both paths produce the SAME sW tile, so everything downstream
// of the staging loop is untouched.
template <int TN, bool WPERM, bool EPIFAST = true, bool E6 = false, bool F32OUT = false>
__global__ __launch_bounds__(NTHREADS) void radiance_mxfp4_fp8_gemm_folded(
    const unsigned char *__restrict__ A, const unsigned char *__restrict__ W,
    const unsigned char *__restrict__ Ws, const unsigned char *__restrict__ Wref,
    const float *__restrict__ As, __bf16 *__restrict__ C, int M, int N, int K,
    int pb1, int pb2, long astride, const unsigned char *__restrict__ WH = nullptr,
    float *__restrict__ Cf = nullptr) {
  __shared__ unsigned char sA[BMF * ASTR];
  __shared__ unsigned char sW[WN * TN * 16 * WSTR];

  const int tid = threadIdx.x, lane = tid & 31, wave = tid >> 5;
  const int wm = wave / WN, wn = wave % WN;
  const int col = lane & 15, kb8 = (lane >> 4) * 8;
  constexpr int BNF_T = WN * TN * 16;
  const int m0 = blockIdx.y * BMF, n0 = blockIdx.x * BNF_T;
  // Merged-linear partition select (paroquant: one launch over the whole N, partition p's n-blocks
  // read the p-th rotated copy of A and its scales). Boundaries are multiples of every n-tile
  // (launcher-checked), so a block never straddles one. pb1 = pb2 = 1<<30 (the default) is p = 0.
  {
    const int p_ = n0 >= pb2 ? 2 : (n0 >= pb1 ? 1 : 0);
    A += (size_t)p_ * astride;
    As += (size_t)p_ * M;
  }

  floatx8 acc[TM][TN];
#pragma unroll
  for (int i = 0; i < TM; ++i)
#pragma unroll
    for (int j = 0; j < TN; ++j)
#pragma unroll
      for (int e = 0; e < 8; ++e) acc[i][j][e] = 0.f;

  for (int k0 = 0; k0 < K; k0 += BK) {
    // Wave-uniform 64-bit bases, hoisted out of the per-thread address arithmetic. Written as
    // `base + int_offset`, the compiler keeps the base in SGPRs and emits the SADDR form of
    // global_load, so a lane pays one 32-bit add instead of the v_add_co_u32 / v_add_co_ci_u32
    // pair a full 64-bit address costs. The matmul block was spending ~71 of its 565 instructions
    // on that arithmetic while using only 29 of the ~102 available SGPRs.
    // r*K fits an int by a wide margin: r < BMF = 256 and K <= 17408, so r*K < 4.5e6.
    const unsigned char *__restrict__ Ab = A + (size_t)m0 * K + k0;
    const unsigned char *__restrict__ Wb = W + ((size_t)n0 * K + k0) / 2;
    // s_prefetch_data on the next slab's weight runs: measured and REJECTED 2026-08-28 --
    // -0.3 to -1% on every production shape (tier7 spfd mode). With branchless staging at
    // occupancy 10 the loads are already fully overlapped; early L2 arrival buys nothing.
    const unsigned char *__restrict__ Wsb = Ws + (size_t)(k0 / 32) * N + n0;
    const unsigned char *__restrict__ Wrefb = Wref + n0;
    // ---- A tile: BM*BK bytes, 16 B per thread per iteration ----
#pragma unroll
    for (int off = 0; off < BMF * BK; off += NTHREADS * 16) {
      const int idx = off + tid * 16;
      const int r = idx / BK, c = idx % BK;
      const int rc = r < M - 1 - m0 ? r : M - 1 - m0;              // clamp, never predicate
      *(uint4_t *)(&sA[r * ASTR + c]) = *(const uint4_t *)(Ab + (rc * K + c));
    }
    // ---- W tile: BN*BK/2 packed bytes, 8 B per thread per iteration -> 16 e4m3 bytes ----
    if constexpr (WPERM) {
      // Fragment order: slot s of the tile is one lane's four packed bytes (eight elements) for
      // one (n-tile, k-step), so consecutive threads read consecutive uint32 and a wave takes 128
      // contiguous bytes. WSLOTS adjacent lane slots per thread; adjacent slots differ only in
      // the row (lane index is half*16 + row), so one vector read covers WSLOTS adjacent rows at
      // the same k and the wave still reads its bytes contiguously.
      //
      // WSLOTS is 4 (a 16 B request) whenever the tile has the slots for it, else 2 (8 B). The
      // original 2-slot trade was priced against PREDICATED staging: one slot (4 B) cost 14-25%
      // of prefill, two recovered most of it, and at TN=4 the remaining narrow-request penalty
      // was ~11% against checkpoint order (tier7, 2026-08-28). A 4-slot group never straddles
      // the k-half boundary because 16 % 4 == 0.
      constexpr int KSTEPS_T = BK / 16, NTILES_T = BNF_T / 16;
      constexpr int TOT_SLOTS = NTILES_T * KSTEPS_T * 32;
      constexpr int WSLOTS = (TOT_SLOTS >= NTHREADS * 4) ? 4 : 2;
      const unsigned int *__restrict__ Wp = (const unsigned int *)W;
      const int ksteps_g = K / 16, kstep0 = k0 / 16;
#pragma unroll
      for (int off = 0; off < TOT_SLOTS; off += NTHREADS * WSLOTS) {
        const int sl = off + tid * WSLOTS;              // first of the group; WSLOTS-aligned
        const int lane_ = sl & 31, rest = sl >> 5;
        const int kst = rest % KSTEPS_T, ntl = rest / KSTEPS_T;
        const int r = ntl * 16 + (lane_ & 15), kloc = kst * 16 + (lane_ >> 4) * 8;
        const int gn = n0 + r;
        // Clamp, never predicate (see the A-tile note in the first kernel). gn is always
        // WSLOTS-aligned here and N % 16 == 0 under WPERM, so clamping to the last WSLOTS-aligned
        // row keeps the vector read aligned and every gc + q in bounds; the fragment address is
        // recomputed from the clamped row.
        const int gc = gn < N - WSLOTS ? gn : N - WSLOTS;
        const int lanec = (lane_ & 16) | (gc & 15);
        const unsigned int *src = &Wp[((size_t)(gc >> 4) * ksteps_g + kstep0 + kst) * 32 + lanec];
        unsigned int wq[WSLOTS];
        if constexpr (WSLOTS == 4) {
          const uint4_t v = *(const uint4_t *)src;
          wq[0] = v[0]; wq[1] = v[1]; wq[2] = v[2]; wq[3] = v[3];
        } else {
          const uint2_t v = *(const uint2_t *)src;
          wq[0] = v[0]; wq[1] = v[1];
        }
        unsigned int hq[WSLOTS];
        if constexpr (E6) {
          const unsigned char *srch = &WH[(((size_t)(gc >> 4) * ksteps_g + kstep0 + kst) * 32 + lanec) * 2];
          if constexpr (WSLOTS == 4) {
            const uint2_t h = *(const uint2_t *)srch;
            hq[0] = h[0] & 0xFFFFu; hq[1] = h[0] >> 16; hq[2] = h[1] & 0xFFFFu; hq[3] = h[1] >> 16;
          } else {
            const unsigned int h = *(const unsigned int *)srch;
            hq[0] = h & 0xFFFFu; hq[1] = h >> 16;
          }
        }
        const int blk = (k0 + kst * 16) / 32;
        // One dword read each for the WSLOTS reference and block exponents instead of 2*WSLOTS
        // byte reads: gc is WSLOTS-aligned and N % 16 == 0, so both reads are aligned.
        unsigned int refw, wsw;
        if constexpr (WSLOTS == 4) {
          refw = *(const unsigned int *)(Wref + gc);
          wsw = *(const unsigned int *)(Ws + (size_t)blk * N + gc);
        } else {
          refw = *(const unsigned short *)(Wref + gc);
          wsw = *(const unsigned short *)(Ws + (size_t)blk * N + gc);
        }
        if constexpr (E6) {
#pragma unroll
          for (int q = 0; q < WSLOTS; ++q) {
            int d = (int)((refw >> (8 * q)) & 0xFF) - (int)((wsw >> (8 * q)) & 0xFF);
            d = d < 0 ? 0 : (d > 6 ? 6 : d);
            const unsigned int t0 = kE2M3[d][0], t1 = kE2M3[d][1], b4 = kE2M3[d][2];
            *(uint2_t *)(&sW[(r + q) * WSTR + kloc]) = e2m3_unpack8(wq[q], hq[q], t0, t1, b4);
          }
        } else
#pragma unroll
        for (int q = 0; q < WSLOTS; ++q) {
          int d = (int)((refw >> (8 * q)) & 0xFF) - (int)((wsw >> (8 * q)) & 0xFF);
          d = d < 0 ? 0 : (d > 15 ? 15 : d);
          const unsigned int t0 = kMag[d][0], t1 = kMag[d][1];
          const unsigned int ev = wq[q] & 0x0F0F0F0Fu;
          const unsigned int od = (wq[q] >> 4) & 0x0F0F0F0Fu;
          const unsigned int be = __builtin_amdgcn_perm(t1, t0, ev & 0x07070707u)
                                | ((ev & 0x08080808u) << 4);
          const unsigned int bo = __builtin_amdgcn_perm(t1, t0, od & 0x07070707u)
                                | ((od & 0x08080808u) << 4);
          *(uint2_t *)(&sW[(r + q) * WSTR + kloc]) =
              uint2_t{__builtin_amdgcn_perm(bo, be, 0x05010400u),
                      __builtin_amdgcn_perm(bo, be, 0x07030602u)};
        }
      }
    } else
#pragma unroll
    for (int off = 0; off < BNF_T * BK / 2; off += NTHREADS * 8) {
      const int idx = off + tid * 8;
      const int r = idx / (BK / 2), c = idx % (BK / 2);
      const int rc = r < N - 1 - n0 ? r : N - 1 - n0;          // clamp, never predicate
      // Same hoist as A. rc*(K/2) < 1.2e6 and the scale offset < 3*N, both comfortably int.
      const unsigned long long packed = *(const unsigned long long *)(Wb + (rc * (K / 2) + c));
      const int blk = (c * 2) / 32;                            // k0's share is in Wsb
      int dsh = (int)Wrefb[rc] - (int)Wsb[blk * N + rc];       // scales are [K/32, N]
      dsh = dsh < 0 ? 0 : (dsh > 15 ? 15 : dsh);
      // v_perm_b32 resolves 4 table lookups per instruction from the 8-byte magnitude table;
      // the sign is separable because entries 8-15 are entries 0-7 with bit 7 set. The upconvert
      // measured 21% of runtime with per-byte constant loads (156 -> 188 TF with it removed
      // entirely), and this recovers +9-12% of that.
      const unsigned int t0 = kMag[dsh][0], t1 = kMag[dsh][1];
      unsigned int outw[4];
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        const unsigned int w = (unsigned int)(packed >> (32 * h));
        const unsigned int ev = w & 0x0F0F0F0Fu;
        const unsigned int od = (w >> 4) & 0x0F0F0F0Fu;
        const unsigned int be = __builtin_amdgcn_perm(t1, t0, ev & 0x07070707u)
                              | ((ev & 0x08080808u) << 4);
        const unsigned int bo = __builtin_amdgcn_perm(t1, t0, od & 0x07070707u)
                              | ((od & 0x08080808u) << 4);
        outw[2 * h]     = __builtin_amdgcn_perm(bo, be, 0x05010400u);
        outw[2 * h + 1] = __builtin_amdgcn_perm(bo, be, 0x07030602u);
      }
      *(uint4_t *)(&sW[r * WSTR + c * 2]) = uint4_t{outw[0], outw[1], outw[2], outw[3]};
    }
    __syncthreads();

#pragma unroll
    for (int step = 0; step < BK / 16; ++step) {
      const int kk = step * 16 + kb8;
      int2_t af[TM], wf[TN];
#pragma unroll
      for (int i = 0; i < TM; ++i) {
        const unsigned char *p = &sA[(wm * TM * 16 + i * 16 + col) * ASTR + kk];
        af[i][0] = *(const int *)p; af[i][1] = *(const int *)(p + 4);
      }
#pragma unroll
      for (int j = 0; j < TN; ++j) {
        const unsigned char *p = &sW[(wn * TN * 16 + j * 16 + col) * WSTR + kk];
        wf[j][0] = *(const int *)p; wf[j][1] = *(const int *)(p + 4);
      }
      // Hard scheduling barrier per k-step. Without it the compiler hoists EVERY step's fragment
      // loads above the first WMMA -- the ISA shows 346 instructions and 140 v_dual_mov_b32 ahead
      // of a 64-WMMA run that itself contains zero moves. Those moves are the register shuffling
      // that holds four k-steps of fragments live at once. Fencing each step keeps one step's
      // fragments live instead of four. Same idea as the attention kernel's SGB bit, which notes
      // that without the barrier its prefetch is inert.
      __builtin_amdgcn_sched_barrier(0);
#pragma unroll
      for (int i = 0; i < TM; ++i)
#pragma unroll
        for (int j = 0; j < TN; ++j)
          acc[i][j] = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(af[i], wf[j], acc[i][j]);
    }
    __syncthreads();
  }

  float rf[TN];
#pragma unroll
  for (int j = 0; j < TN; ++j) {
    const int n = n0 + wn * TN * 16 + j * 16 + col;
    rf[j] = (n < N) ? __int_as_float((int)Wref[n] << 23) : 0.f;
  }
  // Epilogue addressing and predication. `C[(size_t)m * N + n]` is a full 64-bit address per
  // element and there are TM*TN*8 of them, each in its own s_and_saveexec region from the bounds
  // test. Measured in the ISA: 145 v_add_co_u32/v_add_co_ci pairs, 103 s_and_saveexec and 130
  // 64-bit shifts in a kernel with 32 WMMA -- about 15% of the instruction stream spent getting
  // to memory rather than computing.
  //
  // Hoisting a wave-uniform base and indexing with a 32-bit offset gets the SADDR form, so a lane
  // pays one 32-bit add instead of a 64-bit add pair -- the same reasoning as the staging bases
  // above. And a block lying entirely inside M and N needs no bounds test at all; on a real
  // prefill only the last row-block and column-block are ragged. The offset cannot overflow an
  // int: at most (kb8 + (TM-1)*16 + 7) * N + N <= 64 * 34816.
  //
  // BIT-IDENTICAL to the path below by construction -- same operands, same order, same rounding.
  // It is gated that way rather than on a tolerance, because nothing here may change numerics.
  if constexpr (EPIFAST) {
    __bf16 *__restrict__ Cb = C + (size_t)(m0 + wm * TM * 16) * N;
    float *__restrict__ Cfb = Cf + (size_t)(m0 + wm * TM * 16) * N;
    const float *__restrict__ Asb = As + m0 + wm * TM * 16;
    const bool full = (m0 + wm * TM * 16 + (TM - 1) * 16 + kb8 + 7 < M) &&
                      (n0 + wn * TN * 16 + (TN - 1) * 16 + col < N);
    if (full) {
#pragma unroll
      for (int i = 0; i < TM; ++i)
#pragma unroll
        for (int j = 0; j < TN; ++j)
#pragma unroll
          for (int e = 0; e < 8; ++e) {
            const int r = i * 16 + kb8 + e;
            const int n = n0 + wn * TN * 16 + j * 16 + col;
            if constexpr (F32OUT) { Cfb[r * N + n] = acc[i][j][e] * rf[j] * Asb[r]; }
            else { Cb[r * N + n] = (__bf16)(acc[i][j][e] * rf[j] * Asb[r]); }
          }
      return;
    }
  }
#pragma unroll
  for (int i = 0; i < TM; ++i)
#pragma unroll
    for (int j = 0; j < TN; ++j)
#pragma unroll
      for (int e = 0; e < 8; ++e) {
        const int m = m0 + wm * TM * 16 + i * 16 + kb8 + e;
        const int n = n0 + wn * TN * 16 + j * 16 + col;
        if (m < M && n < N) {
          if constexpr (F32OUT) { if (Cf) Cf[(size_t)m * N + n] = acc[i][j][e] * rf[j] * As[m]; }
          else { C[(size_t)m * N + n] = (__bf16)(acc[i][j][e] * rf[j] * As[m]); }
        }
      }
}


// ---------------------------------------------------------------- A-tiled prefill kernel
// Same tile, same W staging, same epilogue as the folded kernel above, but the activation
// arrives in WMMA-FRAGMENT-TILED layout: each 16m x 16k fp8 fragment is 256 contiguous bytes in
// lane order (lane l owns bytes 8l..8l+7 = A[mt*16 + l%16][ks*16 + (l/16)*8 .. +7]), the layout
// radiance_add_rms_quant / radiance_silu_mul_quant emit under RADIANCE_MXFP4_A_TILED_MIN_M. One
// coalesced global_load_b64 per fragment per wave lands straight in the af[] register the WMMA
// reads, so sA and its staging (the largest single cost of the folded kernel: ablated at 24-32%)
// are gone; only W goes through LDS (9-18 KB), so 3 blocks fit a CU instead of 2.
//
// Measured 2026-09-01 (~/mxfp4_work/tier7/lf.hip, DRAM-fed, prod server up, bit-identical on all
// six shapes x M tails x TN): vs the shipped folded kernel on the checkpoint layout, 0.84-0.87x
// at M=2048 (TN=4), 0.87-0.91x at M=1024 (TN=2), 0.878x at out M=8192 and 0.868x at down M=4096
// -- 215-220 TF/s against 190. The fragment-order (WPERM) layout is within 1% of that, so this
// kernel also removes the old "wp1 costs prefill" trade. Rejected on the way, all measured:
// W direct-to-register with an in-register fold (no LDS at all) 1.10-1.30x SLOWER at every
// shape -- the fold is then done by all WM=4 waves and the 4 B/lane W loads add instructions;
// a register-pipelined A prefetch (next slab in flight) 1.05-1.10x slower than this plain
// form at 250 VGPRs, and it spills at TN=4 with a 64-k slab (93 VGPRs, 2.2-4.9x).
// LBK: K per slab. With no A tile in LDS a 128-deep W slab is affordable (17 KB at TN=4, 3
// blocks/CU) and halves the barriers per K. Measured 2026-09-02 (tier7 lf.hip, bit-identical):
// at TN=2 (M < 2048) LBK=128 is 2-4% faster than 64 on every shape (gate48 -22%); at TN=4 it
// costs registers (246 VGPRs, occupancy 5) and loses 12-17%, so the launcher keys LBK on TN.
//
// ROUND 2 (2026-09-03, ~/mxfp4_work/tier7/at2.{h,hip}, abl.hip). An ablation of this kernel
// (WMMA count held, one piece removed at a time, TN=4 M=4096) priced its non-WMMA time: A
// fragment loads 28% (of which ~18% is the L2->L0 stream and ~10% issue/latency), W staging
// 14% (8% of it the fold VALU), LDS fragment reads 11%, barriers 1%; WMMA-only runs at 310
// TF/s = 55% of the full time, and WMMA-only + everything-else sum to ~90%, i.e. the memory
// work barely overlaps the matrix work. Three codegen fixes, each bit-identical, together
// 0.94-0.96x paired vs the 09-02 kernel on every shape at BOTH bands and both W layouts:
//   * SGPR bases for the A fragments (readfirstlane on the tile offset) with an UNSIGNED 32-bit
//     lane offset, so the 16 loads compile to one s_clause of SADDR global_load_b64 with
//     immediate offsets instead of eight 64-bit v_add_co/v_add_co_ci chains and their
//     s_wait_alu stalls (main loop VALU 63 -> ~47 per 64 WMMA).
//   * kMag served from LDS. As a global (constant) load it is a DEPENDENT load issued after the
//     A fragments, and loadcnt is in order, so `s_wait_loadcnt 0x0` before the fold drained the
//     whole A batch every slab. A ds_load has its own counter.
//   * LDS-only barrier fences. __syncthreads() is a workgroup-scope acq/rel on ALL address
//     spaces: `global_inv` plus a full loadcnt drain, twice per slab. Nothing in global memory
//     is shared between waves here, so the fence is scoped to LDS (s_wait_dscnt + s_barrier).
// Priced and REJECTED in the same round (all gated bit-identical): A fragments software-
// pipelined one slab ahead (loop unrolled by two, 230 VGPRs at TN=4): neutral on top of the
// three fixes at TN=4 and -2..-5% at TN=2 from the occupancy loss (the A cost is the L2 stream,
// not latency -- a cache-hot A ablation is still 10% off no-A); prefetching next slab's W into
// registers: +-1%. Every rebalancing of the tile trades one memory stream for another (TN=8 is
// the accumulator wall; A shared via LDS is the folded kernel; TM=3/TN=6 nets ~0), so the
// remaining ~40% over the WMMA floor is structural at this tile.
// Served 2026-09-04 (serve-mxfp4.sh defaults, BetterBench quick --prefill, same-day paired vs
// the 09-02 kernel): PP t/s 5141->5274 @2k, 5283->5385 @8k, 5109->5203 @16k, 4937->5046 @32k,
// 4686->4794 @64k (+1.8..+2.6%); decode untouched (this kernel only runs at M >= 513).
static __device__ __forceinline__ void radiance_lds_barrier() {
  __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup", "local");
  __builtin_amdgcn_s_barrier();
  __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup", "local");
}

// ---- PTQ1_0 (type 143) staging helpers ----
//
// A 28-byte block holds 128 trits in base-3 packing plus one fp16 scale:
//   qs[0..15]  5 trits/byte -> elements 0..79   (one trit index per 16 elements)
//   qs[16..23] 5 trits/byte -> elements 80..119 (one trit index per 8 elements)
//   qh[0..1]   4 trits/byte -> elements 120..127 (one trit index per 2 elements)
//   d (fp16) at offset 26
// Element order matches dequantize_row_ptq1_0 in ggml-quants.c. The value is
// trit * d with trit in {-1,0,+1}.
//
// Decode the 16 trits starting at `base` (a multiple of 16) into four dwords of
// four sign-extended int8 each (0xFF / 0x00 / 0x01).
static __device__ __forceinline__ void ptq1_0_decode16(const unsigned char * __restrict__ blk,
                                                       int base, unsigned int trit[4]) {
    constexpr unsigned int kP3[5] = {1u, 3u, 9u, 27u, 81u};
    const unsigned int * qs = (const unsigned int *) blk;   // 24 B, 4-byte aligned
    const unsigned char * qh = blk + 24;
    if (base < 80) {
        const unsigned int pw = kP3[base >> 4];
        trit[0] = (unsigned int) ptq1_0_decode4_int8_same_trit((int) qs[0], pw);
        trit[1] = (unsigned int) ptq1_0_decode4_int8_same_trit((int) qs[1], pw);
        trit[2] = (unsigned int) ptq1_0_decode4_int8_same_trit((int) qs[2], pw);
        trit[3] = (unsigned int) ptq1_0_decode4_int8_same_trit((int) qs[3], pw);
    } else if (base < 112) {
        // two 8-element groups: trit index n then n+1
        const unsigned int pw = kP3[(base - 80) >> 3];
        trit[0] = (unsigned int) ptq1_0_decode4_int8_same_trit((int) qs[4], pw);
        trit[1] = (unsigned int) ptq1_0_decode4_int8_same_trit((int) qs[5], pw);
        trit[2] = (unsigned int) ptq1_0_decode4_int8_same_trit((int) qs[4], pw * 3u);
        trit[3] = (unsigned int) ptq1_0_decode4_int8_same_trit((int) qs[5], pw * 3u);
    } else {
        // base == 112: elements 112..119 come from qs, 120..127 from qh
        const unsigned int pw = kP3[4];
        trit[0] = (unsigned int) ptq1_0_decode4_int8_same_trit((int) qs[4], pw);
        trit[1] = (unsigned int) ptq1_0_decode4_int8_same_trit((int) qs[5], pw);
        // elements 120..127: the byte alternates qh[0]/qh[1], trit index (e-120)>>1
        unsigned int lo = 0, hi = 0;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const unsigned int a = (unsigned int) ptq1_0_trit_pow3(qh[0], kP3[j]) & 0xFFu;
            const unsigned int b = (unsigned int) ptq1_0_trit_pow3(qh[1], kP3[j]) & 0xFFu;
            if (j < 2) {
                lo |= a << (8 * (2 * j));
                lo |= b << (8 * (2 * j + 1));
            } else {
                hi |= a << (8 * (2 * (j - 2)));
                hi |= b << (8 * (2 * (j - 2) + 1));
            }
        }
        trit[2] = lo;
        trit[3] = hi;
    }
}

// Four packed sign-extended trits -> four int8 bytes: 0 -> 0, +1 -> M, -1 -> -M.
// nz is one per non-zero lane so nz*M places M in exactly those bytes; the sign
// comes from the trit's own high bit, and (0x100 - M) is -M for the negative ones.
static __device__ __forceinline__ unsigned int ptq1_0_trit_to_i8(unsigned int t, unsigned int M) {
    const unsigned int nz   = (t | (t >> 7)) & 0x01010101u;   // 1 where trit != 0
    const unsigned int neg  = (t >> 7) & 0x01010101u;         // 1 where trit == -1
    const unsigned int mag  = nz * M;
    const unsigned int nmag = nz * ((0x100u - M) & 0xFFu);
    return (mag & ~(neg * 0xFFu)) | (nmag & (neg * 0xFFu));
}

// Row-normalised int8 magnitude for one block. The row scale D_row = 2^(Wref-127)
// is applied once in the epilogue, so the weight carries round(d/D_row * 127).
static __device__ __forceinline__ unsigned int ptq1_0_i8_mag(const unsigned char * __restrict__ blk,
                                                            unsigned int wref_byte) {
    const unsigned int h = (unsigned int) blk[26] | ((unsigned int) blk[27] << 8);
    const float d = (1.0f + (float) (h & 0x3FFu) * (1.0f / 1024.0f))
                  * exp2f((float) ((int) ((h >> 10) & 0x1Fu) - 15));
    const float drow = exp2f((float) ((int) wref_byte - 127));
    int q = (int) rintf(d / drow * 127.0f);
    if (q > 127) q = 127;
    if (q < 0)   q = 0;
    return (unsigned int) q;
}

template <int TN, bool WPERM, int LBK = BK, bool E6 = false, bool F32OUT = false, bool RADSC = false,
          bool E6PACK = false, int TWM = WM, int TWN = WN, bool PTQ1 = false>
__global__ __launch_bounds__(TWM * TWN * 32) void radiance_mxfp4_fp8_gemm_atiled(
    const unsigned char *__restrict__ AT, const unsigned char *__restrict__ W,
    const unsigned char *__restrict__ Ws, const unsigned char *__restrict__ Wref,
    const float *__restrict__ As, __bf16 *__restrict__ C, int M, int N, int K,
    int pb1, int pb2, long astride, const unsigned char *__restrict__ WH = nullptr,
    float *__restrict__ Cf = nullptr) {
  constexpr int NTHREADS_T = TWM * TWN * 32;
  constexpr int BMF_T = TWM * TM * 16;
  constexpr int NS = LBK / 16;
  constexpr int LWSTR = LBK + PAD;
  constexpr int BNF_T = TWN * TN * 16;
  constexpr int NIT = BNF_T * LBK / 2 / (NTHREADS_T * 8);   // row-major W staging iterations per slab
  static_assert(NIT >= 1 && NIT * NTHREADS_T * 8 == BNF_T * LBK / 2, "W tile must be 8 B/thread multiples");
  __shared__ unsigned char sW[BNF_T * LWSTR];
  __shared__ unsigned int sMag[32];   // kMag, 128 B: ds_load keeps the lookup off the loadcnt chain
  __shared__ unsigned int sTab[7 * 4];


  const int tid = threadIdx.x, lane = tid & 31, wave = tid >> 5;
  const int wm = wave / TWN, wn = wave % TWN;
  const int col = lane & 15, kb8 = (lane >> 4) * 8;
  // Dispatch m fastest, so the resident blocks cover every m-block at once and the whole A
  // matrix stays cached while the W slabs stream past it. A linear id keeps coverage exact:
  // every (m, n) pair is still visited once. Worth -3.4% on the prefill GEMM (858 -> 831 ms,
  // same interleaved A/B as above).
  const int mb_count = (M + BMF_T - 1) / BMF_T;
  const int bid_ = blockIdx.y * gridDim.x + blockIdx.x;
  const int m0 = (bid_ % mb_count) * BMF_T, n0 = (bid_ / mb_count) * BNF_T;
  // Merged-linear partition select (paroquant: one launch over the whole N, partition p's n-blocks
  // read the p-th rotated copy of A and its scales). Boundaries are multiples of every n-tile
  // (launcher-checked), so a block never straddles one. pb1 = pb2 = 1<<30 (the default) is p = 0.
  {
    const int p_ = n0 >= pb2 ? 2 : (n0 >= pb1 ? 1 : 0);
    AT += (size_t)p_ * astride;
    As += (size_t)p_ * M;
  }
  const int ksteps_g = K / 16;
  const int Mt = (M + 15) >> 4;

  if constexpr (E6 || E6PACK) { if (tid < 28) sTab[tid] = ((const unsigned int *)kE2M3)[tid]; }
  else if (tid < 32) sMag[tid] = ((const unsigned int *)kMag)[tid];
  radiance_lds_barrier();

  // A tile bases, one per M-fragment: wave-uniform, forced into SGPRs; the lane part is an
  // unsigned 32-bit offset so the loads take the SADDR form (max offset < 2^31 for all prod).
  // Tile-granular clamp, never predicate: rows of a clamped tile only reach accumulators the
  // epilogue drops.
  const unsigned char *abase[TM];
#pragma unroll
  for (int i = 0; i < TM; ++i) {
    int mt = (m0 >> 4) + wm * TM + i; mt = mt < Mt - 1 ? mt : Mt - 1;
    abase[i] = AT + (size_t)__builtin_amdgcn_readfirstlane(mt * ksteps_g * 256);
  }
  const unsigned int aoff = lane * 8;

  // W staging geometry, hoisted out of the K loop.
  //   Row-major (checkpoint) layout: row r, packed-byte column c, clamped row rc; 8 B/thread.
  //   Fragment order (WPERM): slot s of the tile is one lane's four packed bytes (eight
  //   elements) for one (n-tile, k-step), so consecutive threads read consecutive uint32 and a
  //   wave takes 128 contiguous bytes. WSLOTS adjacent lane slots per thread; adjacent slots
  //   differ only in the row (lane index is half*16 + row), so one vector read covers WSLOTS
  //   adjacent rows at the same k and the wave still reads its bytes contiguously. WSLOTS is 4
  //   (16 B) whenever the tile has the slots for it, else 2 (8 B); a 4-slot group never
  //   straddles the k-half boundary because 16 % 4 == 0. Clamp to the last WSLOTS-aligned row
  //   (N % 16 == 0 under WPERM) so the vector read stays aligned and every gc + q is in bounds.
  constexpr int KSTEPS_T = LBK / 16, NTILES_T = BNF_T / 16;
  constexpr int TOT_SLOTS = NTILES_T * KSTEPS_T * 32;
  constexpr int WSLOTS = (TOT_SLOTS >= NTHREADS_T * 4) ? 4 : 2;
  constexpr int NITW = TOT_SLOTS / (NTHREADS_T * WSLOTS);
  static_assert(!WPERM || NITW * NTHREADS_T * WSLOTS == TOT_SLOTS, "WPERM slot groups must tile");
  constexpr int NW = WPERM ? NITW : NIT;
  int wrow[NW], wcol[NW], wrc[NW], wkst[NW], wlanec[NW];
#pragma unroll
  for (int it = 0; it < NW; ++it) {
    if constexpr (WPERM) {
      const int sl = it * NTHREADS_T * WSLOTS + tid * WSLOTS;
      const int lane_ = sl & 31, rest = sl >> 5;
      const int kst = rest % KSTEPS_T, ntl = rest / KSTEPS_T;
      wrow[it] = ntl * 16 + (lane_ & 15);
      wcol[it] = kst * 16 + (lane_ >> 4) * 8;          // kloc
      const int gn = n0 + wrow[it];
      wrc[it] = gn < N - WSLOTS ? gn : N - WSLOTS;      // global clamped row
      wlanec[it] = (lane_ & 16) | (wrc[it] & 15);
      wkst[it] = kst;
    } else if constexpr (E6PACK) {
      // MXFP6 packed: one iteration is a 16-element group = 12 packed bytes, so wcol is the
      // group's ELEMENT base in [0, LBK) rather than a 4-bit byte column. The group count
      // BNF_T*LBK/16 equals NIT*NTHREADS_T, so the iteration count is unchanged.
      const int g = it * NTHREADS_T + tid;
      wrow[it] = g / (LBK / 16);
      wcol[it] = (g % (LBK / 16)) * 16;
      wrc[it] = wrow[it] < N - 1 - n0 ? wrow[it] : N - 1 - n0;
      wkst[it] = 0; wlanec[it] = 0;
    } else {
      const int idx = it * NTHREADS_T * 8 + tid * 8;
      wrow[it] = idx / (LBK / 2); wcol[it] = idx % (LBK / 2);
      wrc[it] = wrow[it] < N - 1 - n0 ? wrow[it] : N - 1 - n0;
      wkst[it] = 0; wlanec[it] = 0;
    }
  }
  unsigned int wref[NW];   // per-row max exponent(s): one byte, or WSLOTS packed under WPERM
#pragma unroll
  for (int it = 0; it < NW; ++it) {
    if constexpr (WPERM) {
      if constexpr (WSLOTS == 4) wref[it] = *(const unsigned int *)(Wref + wrc[it]);
      else                       wref[it] = *(const unsigned short *)(Wref + wrc[it]);
    } else wref[it] = Wref[n0 + wrc[it]];
  }

  // PTQ1 accumulates in int32 (the i8 WMMA's native form); every other path in float.
  using acc_t = typename std::conditional<PTQ1, int32x8_t, floatx8>::type;
  acc_t acc[TM][TN];
#pragma unroll
  for (int i = 0; i < TM; ++i)
#pragma unroll
    for (int j = 0; j < TN; ++j)
#pragma unroll
      for (int e = 0; e < 8; ++e) acc[i][j][e] = 0;

  for (int k0 = 0; k0 < K; k0 += LBK) {
    // A fragments straight from global: one SADDR clause, consumed only at the WMMAs.
    int2_t af[TM][NS];
#pragma unroll
    for (int i = 0; i < TM; ++i)
#pragma unroll
      for (int st = 0; st < NS; ++st)
        af[i][st] = *(const int2_t *)(abase[i] + (aoff + (unsigned int)(k0 * 16) + (unsigned int)(st * 256)));

    // ---- W tile -> LDS, folded to e4m3 (kMag[d] via v_perm, see the folded kernel) ----
    uint4_t pk[NW]; unsigned int wsv[NW];
    uint2_t hp[NW];
    if constexpr (WPERM) {
      const unsigned int *__restrict__ Wp = (const unsigned int *)W;
      const int kstep0 = k0 / 16;
#pragma unroll
      for (int it = 0; it < NW; ++it) {
        const unsigned int *src = &Wp[((size_t)(wrc[it] >> 4) * ksteps_g + kstep0 + wkst[it]) * 32 + wlanec[it]];
        if constexpr (WSLOTS == 4) pk[it] = *(const uint4_t *)src;
        else { const uint2_t v = *(const uint2_t *)src; pk[it] = uint4_t{v[0], v[1], 0u, 0u}; }
        if constexpr (E6) {
          const unsigned char *srch = &WH[(((size_t)(wrc[it] >> 4) * ksteps_g + kstep0 + wkst[it]) * 32 + wlanec[it]) * 2];
          if constexpr (WSLOTS == 4) hp[it] = *(const uint2_t *)srch;
          else                       hp[it] = uint2_t{*(const unsigned int *)srch, 0u};
        }
        const int blk = (k0 + wkst[it] * 16) / 32;
        if constexpr (WSLOTS == 4) wsv[it] = *(const unsigned int *)(Ws + (size_t)blk * N + wrc[it]);
        else                       wsv[it] = *(const unsigned short *)(Ws + (size_t)blk * N + wrc[it]);
      }
    } else if constexpr (E6PACK) {
      // MXFP6 checkpoint order, no repack. One 16-element group is 12 packed bytes; the 96
      // bits split into two 48-bit halves, each eight 6-bit codes at bit 6*j. The scale comes
      // from the source block's own e[] tail, past the 192-byte qs region of that block.
      // A group never straddles a block (256/16 == 16 groups per block), so one group sees
      // exactly one sub-block scale.
      //
      // Measured alternative (2026-10-09): pairing two adjacent groups per thread so the 24 B
      // read is 8-byte aligned and shares one scale and one kE2M3 row. Net zero end to end at
      // pp2048: TN=4 shapes gained ~1.1 ms (684.7 vs 690.9/693.0 us) while the TN=2 N=64 shape
      // lost ~1.2 ms (53.8 vs 47.6 us), because that shape is launch-bound, not load-bound.
      // The 29% gap to RADSC is the format itself (6.25 vs 4.53 bpw); E6PACK already streams
      // more bytes per second (283.7 vs 248.7 GB/s), so the width of the load is not the limit.
      const size_t blk_row_bytes = (size_t) (K / QK_MXFP6) * sizeof(block_mxfp6);
      const unsigned char *__restrict__ Wb =
          W + (size_t) __builtin_amdgcn_readfirstlane((int) ((size_t) n0 * blk_row_bytes));
#pragma unroll
      for (int it = 0; it < NW; ++it) {
        const unsigned int ksub = (unsigned int) (k0 + wcol[it]);
        const size_t rowoff = (size_t) wrc[it] * blk_row_bytes
                            + (size_t) (ksub / QK_MXFP6) * sizeof(block_mxfp6);
        const unsigned char *src = Wb + rowoff + (size_t) (ksub % QK_MXFP6) / 16 * 12;
        // 12 bytes, 4-byte aligned (block 200 and group offset 12 are both multiples of 4).
        const unsigned int a0 = *(const unsigned int *) (src + 0);
        const unsigned int a1 = *(const unsigned int *) (src + 4);
        const unsigned int a2 = *(const unsigned int *) (src + 8);
        unsigned int wv0, h20, wv1, h21;
        e2m3_split8((unsigned long long) a0 | ((unsigned long long) (a1 & 0xFFFFu) << 32), wv0, h20);
        e2m3_split8((unsigned long long) (a1 >> 16) | ((unsigned long long) a2 << 16), wv1, h21);
        pk[it] = uint4_t{wv0, wv1, 0u, 0u};
        hp[it] = uint2_t{h20, h21};
        wsv[it] = Wb[rowoff + (size_t) 6 * QK_MXFP6 / 8 + (size_t) (ksub % QK_MXFP6) / QK_MXFP6_SUB];
      }
    } else if constexpr (PTQ1) {
      // PTQ1_0 zero-copy: read the checkpoint's 28-byte blocks in place. LBK == 128 ==
      // QK_PTQ1_0, so one K-slab is exactly one block per row and the 16-element group
      // a slot handles never straddles a block. wcol[it] is the element base in [0,128);
      // /16 gives the 16-element group index within the block.
      const int kblk = k0 / 128;
      const int nbk  = K / 128;
#pragma unroll
      for (int it = 0; it < NW; ++it) {
        const unsigned char * __restrict__ blk = W + ((size_t) (n0 + wrc[it]) * nbk + kblk) * 28;
        const unsigned int M = ptq1_0_i8_mag(blk, wref[it]);
        unsigned int tr[4];
        // wcol[it] is a PACKED BYTE column in [0, LBK/2): byte column c covers elements
        // 2c and 2c+1. decode16 wants an element base that is a multiple of 16, and the
        // eight bytes a slot writes are a contiguous 16-element group, so the base is
        // (wcol*2) rounded down to the group start.
        ptq1_0_decode16(blk, (wcol[it] * 2) & ~15, tr);
        unsigned int outw[4];
#pragma unroll
        for (int h = 0; h < 4; ++h) outw[h] = ptq1_0_trit_to_i8(tr[h], M);
        *(uint4_t *) (&sW[wrow[it] * LWSTR + wcol[it] * 2]) = uint4_t{outw[0], outw[1], outw[2], outw[3]};
      }
    } else {
      const unsigned char *__restrict__ Wb = W + (size_t)__builtin_amdgcn_readfirstlane((int)(((size_t)n0 * K + k0) / 2));
      if constexpr (RADSC) {
        // Read each fragment's scale straight from the global scale plane instead of staging
        // the slab through sS, which removes the barrier that staging needed. The row sS would
        // have used for row r is min(n0 + r, N - 1), the same clamp the W loads already apply.
        const unsigned char *__restrict__ Sc = W + (size_t)N * (K / 2) + (size_t)k0 / 32;
#pragma unroll
        for (int it = 0; it < NW; ++it) {
          const uint2_t v = *(const uint2_t *)(Wb + (unsigned int)(wrc[it] * (K / 2) + wcol[it]));
          pk[it] = uint4_t{v[0], v[1], 0u, 0u};
          const int grow = n0 + wrow[it] < N ? n0 + wrow[it] : N - 1;
          wsv[it] = Sc[(size_t)grow * (K / 32) + (unsigned int)((wcol[it] * 2) / 32)];
        }
      } else {
        const unsigned char *__restrict__ Wsb = Ws + (size_t)__builtin_amdgcn_readfirstlane((k0 / 32) * N + n0);
#pragma unroll
        for (int it = 0; it < NW; ++it) {
          const uint2_t v = *(const uint2_t *)(Wb + (unsigned int)(wrc[it] * (K / 2) + wcol[it]));
          pk[it] = uint4_t{v[0], v[1], 0u, 0u};
          wsv[it] = Wsb[(unsigned int)(((wcol[it] * 2) / 32) * N + wrc[it])];
        }
      }
    }
    if constexpr (WPERM) {
      if constexpr (E6) {
#pragma unroll
        for (int it = 0; it < NW; ++it)
#pragma unroll
          for (int q = 0; q < WSLOTS; ++q) {
            int d = (int)((wref[it] >> (8 * q)) & 0xFF) - (int)((wsv[it] >> (8 * q)) & 0xFF);
            d = d < 0 ? 0 : (d > 6 ? 6 : d);
            const uint4_t t = *(const uint4_t *)&sTab[d * 4];
            *(uint2_t *)(&sW[(wrow[it] + q) * LWSTR + wcol[it]]) =
                e2m3_unpack8(pk[it][q], (hp[it][q >> 1] >> (16 * (q & 1))) & 0xFFFFu, t[0], t[1], t[2]);
          }
      } else
#pragma unroll
      for (int it = 0; it < NW; ++it)
#pragma unroll
        for (int q = 0; q < WSLOTS; ++q) {
          int d = (int)((wref[it] >> (8 * q)) & 0xFF) - (int)((wsv[it] >> (8 * q)) & 0xFF);
          d = d < 0 ? 0 : (d > 15 ? 15 : d);
          const uint2_t t = *(const uint2_t *)&sMag[d * 2];
          const unsigned int wq = pk[it][q];
          const unsigned int ev = wq & 0x0F0F0F0Fu;
          const unsigned int od = (wq >> 4) & 0x0F0F0F0Fu;
          const unsigned int be = __builtin_amdgcn_perm(t[1], t[0], ev & 0x07070707u)
                                | ((ev & 0x08080808u) << 4);
          const unsigned int bo = __builtin_amdgcn_perm(t[1], t[0], od & 0x07070707u)
                                | ((od & 0x08080808u) << 4);
          *(uint2_t *)(&sW[(wrow[it] + q) * LWSTR + wcol[it]]) =
              uint2_t{__builtin_amdgcn_perm(bo, be, 0x05010400u),
                      __builtin_amdgcn_perm(bo, be, 0x07030602u)};
        }
    } else if constexpr (E6PACK) {
      // MXFP6 packed: one group is 16 elements in two 8-element halves, both inside one
      // 32-element sub-block, so a single d covers the pair. Same e2m3_unpack8 as the
      // fragment-order E6 path, fed the split of the checkpoint's packed codes.
#pragma unroll
      for (int it = 0; it < NW; ++it) {
        int d = (int) wref[it] - (int) wsv[it];
        d = d < 0 ? 0 : (d > 6 ? 6 : d);
        const uint4_t t = *(const uint4_t *) &sTab[d * 4];
        *(uint2_t *) &sW[wrow[it] * LWSTR + wcol[it]] =
            e2m3_unpack8(pk[it][0], hp[it][0], t[0], t[1], t[2]);
        *(uint2_t *) &sW[wrow[it] * LWSTR + wcol[it] + 8] =
            e2m3_unpack8(pk[it][1], hp[it][1], t[0], t[1], t[2]);
      }
    } else if constexpr (PTQ1) {
      // Already int8 in the staging branch above (trit * row-normalised magnitude).
    } else {
#pragma unroll
      for (int it = 0; it < NW; ++it) {
        int dsh = (int)wref[it] - (int)wsv[it];
        dsh = dsh < 0 ? 0 : (dsh > 15 ? 15 : dsh);
        const uint2_t t = *(const uint2_t *)&sMag[dsh * 2];
        unsigned int outw[4];
#pragma unroll
        for (int h = 0; h < 2; ++h) {
          const unsigned int w = pk[it][h];
          const unsigned int ev = w & 0x0F0F0F0Fu;
          const unsigned int od = (w >> 4) & 0x0F0F0F0Fu;
          const unsigned int be = __builtin_amdgcn_perm(t[1], t[0], ev & 0x07070707u)
                                | ((ev & 0x08080808u) << 4);
          const unsigned int bo = __builtin_amdgcn_perm(t[1], t[0], od & 0x07070707u)
                                | ((od & 0x08080808u) << 4);
          outw[2 * h]     = __builtin_amdgcn_perm(bo, be, 0x05010400u);
          outw[2 * h + 1] = __builtin_amdgcn_perm(bo, be, 0x07030602u);
        }
        *(uint4_t *)(&sW[wrow[it] * LWSTR + wcol[it] * 2]) = uint4_t{outw[0], outw[1], outw[2], outw[3]};
      }
    }
    radiance_lds_barrier();

#pragma unroll
    for (int step = 0; step < NS; ++step) {
      const int kk = step * 16 + kb8;
      int2_t wf[TN];
#pragma unroll
      for (int j = 0; j < TN; ++j) {
        const unsigned char *p = &sW[(wn * TN * 16 + j * 16 + col) * LWSTR + kk];
        // p is 8-byte aligned: row stride is LBK + PAD (72, a multiple of 8) and kk = step*16 + kb8
        // with kb8 in {0, 8}. One LDS.64 instead of two LDS.32 per fragment.
        *(int2_t *) &wf[j] = *(const int2_t *) p;
      }
      __builtin_amdgcn_sched_barrier(0);
#pragma unroll
      for (int i = 0; i < TM; ++i)
#pragma unroll
        for (int j = 0; j < TN; ++j) {
          if constexpr (PTQ1) {
            // Same lane convention as the fp8 WMMA, i32 accumulator (no rounding).
            const int2_t av = *(const int2_t *) &af[i][step];
            const int2_t bv = *(const int2_t *) &wf[j];
            acc[i][j] = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(true, av, true, bv, acc[i][j], true);
          } else {
            acc[i][j] = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(af[i][step], wf[j], acc[i][j]);
          }
        }
    }
    radiance_lds_barrier();
  }

  // Epilogue: identical to the folded kernel's EPIFAST path (bit-identical by construction).
  float rf[TN];
#pragma unroll
  for (int j = 0; j < TN; ++j) {
    const int n = n0 + wn * TN * 16 + j * 16 + col;
    // PTQ1 stores the weight as trit * round(d/D_row * 127), so the row factor carries
    // a 1/127 the MXFP4 paths do not have. Everything else is identical.
    const float q = PTQ1 ? (1.0f / 127.0f) : 1.0f;
    rf[j] = (n < N) ? __int_as_float((int)Wref[n] << 23) * q : 0.f;
  }
  {
    __bf16 *__restrict__ Cb = C + (size_t)(m0 + wm * TM * 16) * N;
    float *__restrict__ Cfb = Cf ? Cf + (size_t)(m0 + wm * TM * 16) * N : nullptr;
    const float *__restrict__ Asb = As + m0 + wm * TM * 16;
    const bool full = (m0 + wm * TM * 16 + (TM - 1) * 16 + kb8 + 7 < M) &&
                      (n0 + wn * TN * 16 + (TN - 1) * 16 + col < N);
    if (full) {
#pragma unroll
      for (int i = 0; i < TM; ++i)
#pragma unroll
        for (int j = 0; j < TN; ++j)
#pragma unroll
          for (int e = 0; e < 8; ++e) {
            const int r = i * 16 + kb8 + e;
            const int n = n0 + wn * TN * 16 + j * 16 + col;
            if constexpr (F32OUT) { Cfb[r * N + n] = acc[i][j][e] * rf[j] * Asb[r]; }
            else { Cb[r * N + n] = (__bf16)(acc[i][j][e] * rf[j] * Asb[r]); }
          }
      return;
    }
  }
#pragma unroll
  for (int i = 0; i < TM; ++i)
#pragma unroll
    for (int j = 0; j < TN; ++j)
#pragma unroll
      for (int e = 0; e < 8; ++e) {
        const int m = m0 + wm * TM * 16 + i * 16 + kb8 + e;
        const int n = n0 + wn * TN * 16 + j * 16 + col;
        if (m < M && n < N) {
          if constexpr (F32OUT) { if (Cf) Cf[(size_t)m * N + n] = acc[i][j][e] * rf[j] * As[m]; }
          else { C[(size_t)m * N + n] = (__bf16)(acc[i][j][e] * rf[j] * As[m]); }
        }
      }
}

// ---------------------------------------------------------------- decode path (M <= 16)
// The tiles above are sized for prefill: BMF=256 via TM=4. At decode M is 5 (batch 1 x SPEC+1), so
// per wave the folded kernel issues K/DBK * DBK/16 * TM * TN = 4352 WMMA against 5 real rows -- 51x
// more MACs than useful, a ~70 us matrix-issue floor on a shape whose DRAM floor is 41 us. TM=1
// cuts issued MACs 16x and shrinks the A tile from 256xBK to 16xBK at the same time.
//
// Split-K is not optional here: with TM=1 the independent N-fragments number N/16, so N=5120 gives
// only 320 waves = 2.5 per SIMD no matter how the N tile is chosen. Measured, DKS=1 is 2.4x worse
// than DKS=4 on mlp.down. Partials are fp32 [DKS][M][N]; the folded scale composes for free because
// Wref[n] is a whole-row max over k, so both per-row factors are applied once in the reduce.
//
// DBK=128 here, which REVERSES the prefill answer (where it measured -34%). That loss was pure LDS
// occupancy: 22.5 -> 43.5 KB took resident blocks 2 -> 1. With a 16-row A tile, LDS is
// (16+BND)*(DBK+PAD) = 10.88 KB, so 5-6 blocks still fit. Measured 1.87x on gate_up. See
// ~/mxfp4_work/tier5/RESULTS.md.
#define DEC_MTILE 16
#define DEC_PAD 8
static __device__ __forceinline__ unsigned int wq_dec(uint2_t v, int q) { return q ? v[1] : v[0]; }
// Nontemporal (streaming) weight loads, `global_load ... th:TH_LOAD_NT` on gfx1201. The decode
// weight stream is read exactly once per step, so a streaming hint keeps it from displacing the
// split-K partials and A in L2/MALL. Applied to the W loads ONLY: Ws/Wref are re-read per slab and
// the partials must stay cached for the fused reduce. RADIANCE_MXFP4_DECODE_NT=1 selects it.
template <bool NT, typename T>
static __device__ __forceinline__ T ld_w(const T *p) {
  if constexpr (NT) return __builtin_nontemporal_load(p);
  else return *p;
}

// DTM = number of 16-row M-fragments; the launcher picks ceil(M/16). vLLM captures decode at
// num_seqs*(SPEC+1): 5/10/20/25/35/40 under mtp at SPEC=4, and batch x 8 under a dflash drafter
// at SPEC=7, which reaches 64 at max_num_seqs=8. DTM 1..4 covers every reachable batch of both.
// Use the SMALLEST DTM that covers M: measured at M=20, DTM=2 beats DTM=3 by 12% (89.1 vs 99.9 us
// on gate_up) purely because the wider tile computes rows nobody asked for.
template <int DWN, int DBK, int DKS, int DTM, bool WPERM, bool DIRECT = true, bool NT = false,
          bool E6 = false>
__global__ __launch_bounds__(DWN * 32) void radiance_mxfp4_fp8_gemm_decode(
    const unsigned char *__restrict__ A, const unsigned char *__restrict__ W,
    const unsigned char *__restrict__ Ws, const unsigned char *__restrict__ Wref,
    const float *__restrict__ As, float *__restrict__ P, int *__restrict__ cnt,
    __bf16 *__restrict__ C, int M, int N, int K, int pb1, int pb2, long astride,
    const unsigned char *__restrict__ WH = nullptr) {
  constexpr int BND = DWN * 16;
  constexpr int DASTR = DBK + DEC_PAD, DWSTR = DBK + DEC_PAD;
  constexpr int DNTHREADS = DWN * 32;
  __shared__ unsigned char sA[DEC_MTILE * DTM * DASTR];
  __shared__ unsigned char sW[BND * DWSTR];
  __shared__ int s_last;

  const int tid = threadIdx.x, lane = tid & 31, wave = tid >> 5;
  const int col = lane & 15, kb8 = (lane >> 4) * 8;
  const int n0 = blockIdx.x * BND;
  const int ks = blockIdx.z;
  // Merged-linear partition select (paroquant: one launch over the whole N, partition p's n-blocks
  // read the p-th rotated copy of A and its scales). Boundaries are multiples of every n-tile
  // (launcher-checked), so a block never straddles one. pb1 = pb2 = 1<<30 (the default) is p = 0.
  {
    const int p_ = n0 >= pb2 ? 2 : (n0 >= pb1 ? 1 : 0);
    A += (size_t)p_ * astride;
    As += (size_t)p_ * M;
  }

  const int slabs = (K + DBK - 1) / DBK;
  const int spb = (slabs + DKS - 1) / DKS;
  const int s_lo = ks * spb, s_hi = min(slabs, s_lo + spb);

  floatx8 acc[DTM];
#pragma unroll
  for (int i = 0; i < DTM; ++i)
#pragma unroll
    for (int e = 0; e < 8; ++e) acc[i][e] = 0.f;

  // ---- per-slab pieces, shared by the three slab loops below ----
  auto stage_a = [&](int k0) {
#pragma unroll
    for (int off = 0; off < DEC_MTILE * DTM * DBK; off += DNTHREADS * 16) {
      const int idx = off + tid * 16;
      if (idx < DEC_MTILE * DTM * DBK) {
        const int r = idx / DBK, c = idx % DBK;
        const int rc = r < M - 1 ? r : M - 1;                  // clamp, never predicate
        *(uint4_t *)(&sA[r * DASTR + c]) = *(const uint4_t *)(A + (size_t)rc * K + k0 + c);
      }
    }
  };
  // Row-order (checkpoint layout) W staging from a packed 8-byte word per (iteration, thread).
  // getw(it, gc, c) supplies the word so the plain and line-group loops share the fold.
  constexpr int NITW = BND * DBK / 2 / (DNTHREADS * 8);
  auto stage_w_row = [&](int k0, auto &&getw) {
#pragma unroll
    for (int it = 0; it < NITW; ++it) {
      const int idx = it * DNTHREADS * 8 + tid * 8;
      const int r = idx / (DBK / 2), c = idx % (DBK / 2), gn = n0 + r;
      const int gc = gn < N - 1 ? gn : N - 1;                  // clamp, never predicate
      const unsigned long long packed = getw(it, gc, c);
      const int blk = (k0 + c * 2) / 32;
      int dsh = (int)Wref[gc] - (int)Ws[(size_t)blk * N + gc];
      dsh = dsh < 0 ? 0 : (dsh > 15 ? 15 : dsh);
      const unsigned int t0 = kMag[dsh][0], t1 = kMag[dsh][1];
      unsigned int outw[4];
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        const unsigned int wv = (unsigned int)(packed >> (32 * h));
        const unsigned int ev = wv & 0x0F0F0F0Fu;
        const unsigned int od = (wv >> 4) & 0x0F0F0F0Fu;
        const unsigned int be = __builtin_amdgcn_perm(t1, t0, ev & 0x07070707u)
                              | ((ev & 0x08080808u) << 4);
        const unsigned int bo = __builtin_amdgcn_perm(t1, t0, od & 0x07070707u)
                              | ((od & 0x08080808u) << 4);
        outw[2 * h]     = __builtin_amdgcn_perm(bo, be, 0x05010400u);
        outw[2 * h + 1] = __builtin_amdgcn_perm(bo, be, 0x07030602u);
      }
      *(uint4_t *)(&sW[r * DWSTR + c * 2]) = uint4_t{outw[0], outw[1], outw[2], outw[3]};
    }
  };
  auto stage_w_wperm = [&](int k0) {
    if constexpr (WPERM) {
      // Fragment order, as in the folded kernel: consecutive threads take consecutive uint32, so
      // a wave's weight read is 128 contiguous bytes instead of sixteen rows K/2 bytes apart.
      // TWO lane slots per thread (an 8 B request), the width the int4 twin measured optimal
      // (4 B was 11% slower there; 16 B identical). Adjacent slots are adjacent rows at the same
      // k, exactly as in the folded kernel above.
      constexpr int KSTEPS_T = DBK / 16, NTILES_T = BND / 16;
      const unsigned int *__restrict__ Wp = (const unsigned int *)W;
      const int ksteps_g = K / 16, kstep0 = k0 / 16;
#pragma unroll
      for (int off = 0; off < NTILES_T * KSTEPS_T * 32; off += DNTHREADS * 2) {
        const int sl = off + tid * 2;                   // first of the pair; always even
        const int lane_ = sl & 31, rest = sl >> 5;
        const int kst = rest % KSTEPS_T, ntl = rest / KSTEPS_T;
        const int r = ntl * 16 + (lane_ & 15), kloc = kst * 16 + (lane_ >> 4) * 8;
        const int gn = n0 + r;
        // Clamp, never predicate; gn is always even here and N % 16 == 0 under WPERM, so
        // clamping to the last even row keeps the uint2 read 8-byte aligned and gc + 1 in
        // bounds; the fragment address is recomputed from the clamped row.
        const int gc = gn < N - 2 ? gn : N - 2;
        const int lanec = (lane_ & 16) | (gc & 15);
        const uint2_t wv = ld_w<NT>(
            (const uint2_t *)&Wp[((size_t)(gc >> 4) * ksteps_g + kstep0 + kst) * 32 + lanec]);
        unsigned int hv = 0;
        if constexpr (E6)
          hv = ld_w<NT>((const unsigned int *)&WH[(((size_t)(gc >> 4) * ksteps_g + kstep0 + kst) * 32 + lanec) * 2]);
        const int blk = (k0 + kst * 16) / 32;
        if constexpr (E6) {
#pragma unroll
          for (int q = 0; q < 2; ++q) {
            int d = (int)Wref[gc + q] - (int)Ws[(size_t)blk * N + gc + q];
            d = d < 0 ? 0 : (d > 6 ? 6 : d);
            const unsigned int t0 = kE2M3[d][0], t1 = kE2M3[d][1], b4 = kE2M3[d][2];
            *(uint2_t *)(&sW[(r + q) * DWSTR + kloc]) =
                e2m3_unpack8(wq_dec(wv, q), (hv >> (16 * q)) & 0xFFFFu, t0, t1, b4);
          }
        } else
#pragma unroll
        for (int q = 0; q < 2; ++q) {
          int d = (int)Wref[gc + q] - (int)Ws[(size_t)blk * N + gc + q];
          d = d < 0 ? 0 : (d > 15 ? 15 : d);
          const unsigned int t0 = kMag[d][0], t1 = kMag[d][1];
          const unsigned int ev = wq_dec(wv, q) & 0x0F0F0F0Fu;
          const unsigned int od = (wq_dec(wv, q) >> 4) & 0x0F0F0F0Fu;
          const unsigned int be = __builtin_amdgcn_perm(t1, t0, ev & 0x07070707u)
                                | ((ev & 0x08080808u) << 4);
          const unsigned int bo = __builtin_amdgcn_perm(t1, t0, od & 0x07070707u)
                                | ((od & 0x08080808u) << 4);
          *(uint2_t *)(&sW[(r + q) * DWSTR + kloc]) =
              uint2_t{__builtin_amdgcn_perm(bo, be, 0x05010400u),
                      __builtin_amdgcn_perm(bo, be, 0x07030602u)};
        }
      }
    }
  };
  auto mma_slab = [&]() {
    __syncthreads();
#pragma unroll
    for (int step = 0; step < DBK / 16; ++step) {
      const int kk = step * 16 + kb8;
      int2_t af[DTM], wf;
#pragma unroll
      for (int i = 0; i < DTM; ++i) {
        const unsigned char *pa = &sA[(i * 16 + col) * DASTR + kk];
        af[i][0] = *(const int *)pa; af[i][1] = *(const int *)(pa + 4);
      }
      const unsigned char *pw = &sW[(wave * 16 + col) * DWSTR + kk];
      wf[0] = *(const int *)pw; wf[1] = *(const int *)(pw + 4);
#pragma unroll
      for (int i = 0; i < DTM; ++i)
        acc[i] = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(af[i], wf, acc[i]);
    }
    __syncthreads();
  };

  if constexpr (WPERM) {
    for (int s = s_lo; s < s_hi; ++s) {
      const int k0 = s * DBK;
      stage_a(k0);
      stage_w_wperm(k0);
      mma_slab();
    }
  } else {
    for (int s = s_lo; s < s_hi; ++s) {
      const int k0 = s * DBK;
      stage_a(k0);
      stage_w_row(k0, [&](int, int gc, int c) {
        return ld_w<NT>((const unsigned long long *)(W + ((size_t)gc * K + k0) / 2 + c));
      });
      mma_slab();
    }
  }

  const int n = n0 + wave * 16 + col;

  // At DKS==1 the accumulator already holds the whole K range, so the partial buffer, the
  // threadfence, the atomic and the reduction pass are all pure overhead -- and that overhead is
  // not small. Partial traffic is DKS*M*N floats: constant in the weight stream but LINEAR IN M,
  // 11.1 MB against gate_up's 44.6 MB of weights at M=40 and 17.8 MB at M=64. Picking DKS by shape
  // and by M is the launcher's job (split_k_for below); this is the path that makes DKS==1 free.
  // Bit-identical to the DKS==1 reduction path, which summed a single term -- the DIRECT=false
  // instantiation exists so that equivalence can be tested rather than asserted.
  if constexpr (DKS == 1 && DIRECT) {
    if (n < N) {
      const float rf = __int_as_float((int)Wref[n] << 23);
#pragma unroll
      for (int i = 0; i < DTM; ++i)
#pragma unroll
        for (int e = 0; e < 8; ++e) {
          const int m = i * 16 + kb8 + e;
          if (m < M) C[(size_t)m * N + n] = (__bf16)(acc[i][e] * rf * As[m]);
        }
    }
    return;
  }

  if (n < N) {
#pragma unroll
    for (int i = 0; i < DTM; ++i)
#pragma unroll
      for (int e = 0; e < 8; ++e) {
        const int m = i * 16 + kb8 + e;
        if (m < M) P[((size_t)ks * M + m) * N + n] = acc[i][e];
      }
  }

  // Fused reduction. The DKS blocks covering one n-range race on an atomic counter and whoever
  // arrives last reduces in place, so there is no second kernel launch -- 13,376 of them per
  // decode run in the profile, at a 4.5 us mean host gap. It is also faster in pure kernel time
  // (-5.5% on gate_up to -29% on gdn.out) because the partials are still in cache when the
  // reduction reads them. The last block resets the counter, leaving it zeroed for the next
  // launch; launches on one stream are serialised, so no separate clearing pass is needed.
  __syncthreads();
  if (tid == 0) {
    __threadfence();
    s_last = (atomicAdd(&cnt[blockIdx.x], 1) == DKS - 1);
  }
  __syncthreads();
  if (!s_last) return;
  if (tid == 0) cnt[blockIdx.x] = 0;

  const int nhi = min(n0 + BND, N);
  for (int nn = n0 + tid; nn < nhi; nn += DNTHREADS) {
    const float rf = __int_as_float((int)Wref[nn] << 23);
    for (int m = 0; m < M; ++m) {
      float sum = 0.f;
      for (int k = 0; k < DKS; ++k) sum += P[((size_t)k * M + m) * N + nn];
      C[(size_t)m * N + nn] = (__bf16)(sum * rf * As[m]);
    }
  }
}

// Split-K partial slab. PyTorch owns the memory; we only hold the pointer.
//
// This used to hipMalloc here and that was wrong twice over. Allocating lazily from launch() puts
// the allocation inside CUDA-graph capture whenever the torch.compile cache is warm (vLLM then
// skips the eager profile run), where it is illegal. And allocating from this .so at all produced
// a completely misleading error: quark's TileLang registers a pybind11 exception translator in the
// same ABI domain, so ANY C++ exception escaping our module is reported as
// "HIP runtime library (libamdhip64.so) not found ... TileLang's ROCm backend", pointing at our own
// pybind frame. Letting torch allocate it removes both problems and makes the buffer visible from
// Python, where its lifetime is obvious.
//
// Sized DEC_KS(4) x 64 rows x N(<=32768) x 4 B = 32 MiB, allocated in radiance_mxfp4.py. 64 rows
// because a dflash drafter at SPEC=7 reaches M = max_num_seqs x 8 = 64. Single-stream decode per
// rank is assumed. split_k_for() may choose a narrower split, which only shrinks the requirement.
// 36864, not 32768 (2026-09-16): TP=1 gate_up is N=34816, and at 32768 the gate below sent every
// decode call of the widest linear to the folded prefill tile. TP=2's widest shape is 17408, so
// for it this is a comparison against a bigger constant and nothing else; the scratch that backs
// DKS>1 is sized in radiance_mxfp4.py from the rank's own width, and nblk >= 110 picks DKS=1
// (no scratch) for the 34816 shape anyway.
#define DEC_MAX_N 36864
#define DEC_KS 4
#define DEC_DWN 8           // waves per decode block; the N tile is DEC_DWN*16 = 128 wide

// Split-K width: the smallest split whose grid still fills the machine.
//
// The partial buffer costs DKS*M*N floats of write-then-read traffic -- invisible next to the
// weight stream at M=5, dominant at M=64 (on gate_up, 17.8 MB against 44.6 MB of weights) -- and
// the in-place reduction serialises onto whichever of the DKS blocks finishes last while the other
// DKS-1 have exited. The benefit is grid fill: with DTM=1 the independent N-fragments number N/16,
// so a narrow shape cannot fill the GPU without splitting K. Cost rises with M, benefit does not,
// so the fill target the split has to clear FALLS as M rises. That is the whole rule.
//
// Measured cell by cell in ~/mxfp4_work/tier6 (mxks.hip, NCOPY=6 so every shape is DRAM-fed -- at
// NCOPY=3 the 22.3 MB down weight sits in the 64 MB Infinity Cache, reads 847 GB/s, and flatters
// whichever split reloads it least). Best DKS, us, this box:
//
//   shape     N      nblk   M=5  M=8  M=16 M=24 M=32 M=40 M=48 M=56 M=64
//   gate_up   17408   136    4    4    1    1    1    1    1    1    1
//   n8192      8192    64    4    2    2    2    2    2    2    2    2
//   n7168      7168    56    4    2    2    2    2    2    2    2    2
//   down       5120    40    4    4    4    4    4    4    4    2    2
//   out        5120    40    4    4    4    4    4    4    4    2    2
//   gate48       48     1    4    4    4    4    4    4    4    4    4
//
// Those six ARE the production shape list, read off a running server with RADIANCE_MXFP4_MHIST
// rather than derived from the model config -- the config-derived guess had the fused qkv at
// N=4096 and missed 7168 and 8192 completely, which is exactly the band where DKS=2 lives. Deriving
// this shape list by hand got it wrong; do not do that again.
//
// Two further things the table says that a coarser rule gets wrong:
//   * At M<=8 EVERY shape wants the widest split, gate_up included (96.3 us at DKS=4 against 98.0
//     at DKS=1). That is the single-stream shape -- dflash SPEC=7 makes M = num_seqs*8 -- so a rule
//     that reaches for DKS=1 on width alone would trade latency on the commonest call there is.
//   * nblk=40 crosses over at M=56, but nblk=56 and nblk=64 cross over at M=8. The crossover is not
//     a property of M alone, which is why the threshold is on nblk*ks and not on either separately.
//
// M in production is NOT just num_seqs*(SPEC+1): the same probe shows 1, 2, 8, 9, 13, 16, 21, 24,
// 32, 47, 48 and 64 all reaching this kernel, plus 72 and 99 from mixed prefill+decode batches that
// exceed DEC_MAX_TM*16 and fall through to the folded tile.
// BK for the decode tile, keyed to DTM.
//
// tier5 picked BK=128 for the whole decode band, but measured it only at M<=16 (DTM=1) and against
// a SINGLE weight buffer -- 44.6 MB of gate_up sits inside the 64 MB last-level cache, and tier5's
// own probe read 1251 GB/s there against a 635 GB/s cold-DRAM peak. Re-measured DRAM-fed
// (~/mxfp4_work/tier6/grid.hip), BK=128 still wins on geomean, but it loses hard at DTM=4 because
// LDS/block is (16*DTM + DWN*16)*(DBK+8): it grows 19.6 -> 26.1 KB across DTM while BK=64 goes only
// 10.4 -> 13.8, so at DTM=4 it is 2 resident blocks per CU against 4, occupancy 8 against 12. That
// is the same LDS cliff that set the PREFILL tile to BK=64; it reaches decode once the batch is
// deep enough. gate_up at M=56/64 measures 0.80x / 0.79x, reproduced across three separate stages.
//
// Deliberately narrow. At DTM=4 the fill rule below yields ks==1 only for gate_up, so this touches
// exactly the one shape where the win is decisive. down and out also measure ~0.955x at BK=64, and
// n8192/n7168/gate48 measure 1.00-1.04x -- all inside the 1-4% run-to-run spread seen between
// sweep stages, so they keep BK=128 until that is re-measured rather than widening a rule fitted to
// six points. Note the two axes are coupled BOTH ways: at BK=64, down and out want KS=4 where at
// BK=128 they want KS=2, so split_k_for's high-M band is a BK=128 property, not a shape property.
//
// RADIANCE_MXFP4_DECODE_BK=128 pins the old behaviour; that is the A/B control.
static bool decode_bk64(int tm, int ks) {
  static const bool pin128 = [] {
    const char *e = getenv("RADIANCE_MXFP4_DECODE_BK");
    return e && atoi(e) == 128;
  }();
  return !pin128 && tm == 4 && ks == 1;
}

static int split_k_for(int nblk, int M) {
  // Cached, not a getenv per call: this runs once per decode GEMM and there are 304 of them per
  // forward, against a 4.5 us launch gap. Same pattern as the other knobs in launch().
  static const int forced = [] {
    const char *e = getenv("RADIANCE_MXFP4_DECODE_KS");
    return e ? atoi(e) : 0;
  }();
  if (forced) return forced;
  // Fill target in blocks. Re-measured with branchless staging (~/mxfp4_work/tier7, 2026-08-28):
  // the old M<=8 special case (widest split always) INVERTED -- the serialised staging round
  // trips were what a wider split's extra fill was paying for, and with them gone the smallest
  // filling split wins at M<=8 on every shape (gate_up dks4->dks1 -2.7/-5.3%, n8192/n7168
  // dks4->dks2 -4.3/-8.4%). The down/out crossover to dks2 also moved down, M>=56 -> M>=32
  // (down M=40 61.7->59.3, out M=48 32.6->30.4). fill=110 at M<=24 and 78 above reproduces the
  // best measured cell on all six shapes within the 1-4% run spread; out M=16 is why the
  // boundary cannot drop below 24 (dks2 there is +6%).
  const int fill = (M <= 24) ? 110 : 78;
  for (int ks = 1; ks < DEC_KS; ks <<= 1)
    if (nblk * ks >= fill) return ks;
  return DEC_KS;
}
#define DEC_MAX_TM 8        // covers M<=128: a dflash drafter at SPEC=7 and max_num_seqs=16
                            // reaches batch x 8 = 128. Extended from 4 on 2026-08-29 for the
                            // 16-concurrent serve; the launcher's decode_max_m env still gates
                            // WHICH M take this kernel (64 at max_num_seqs=8, unchanged), so
                            // the extension is inert until the serve opts in. The split-K
                            // scratch must cover DEC_KS*decode_max_m rows -- radiance_mxfp4.py
                            // sizes it from RADIANCE_MXFP4_DECODE_MAX_M.
static float *g_decode_partial = nullptr;
static size_t g_decode_partial_bytes = 0;

static int *g_decode_cnt = nullptr;

static void set_decode_scratch(uintptr_t ptr, size_t bytes, uintptr_t cnt) {
  g_decode_partial = (float *)ptr;
  g_decode_partial_bytes = bytes;
  g_decode_cnt = (int *)cnt;        // must be ZEROED by the caller, once
}

static void launch_impl(uintptr_t a, uintptr_t w, uintptr_t ws, uintptr_t wref, uintptr_t as,
                        uintptr_t c, int M, int N, int K, int pb1, int pb2, long astride,
                        uintptr_t stream, uintptr_t wh = 0) {
  if (K % BK) throw std::runtime_error("radiance_mxfp4_fp8: K must be a multiple of 64");
  // partition boundaries must be n-tile aligned for every kernel below (decode 128, folded 64/128)
  if ((pb1 < N && pb1 % 128) || (pb2 < N && pb2 % 128))
    throw std::runtime_error("radiance_mxfp4_fp8: partition boundaries must be multiples of 128");
  if (!wref && pb1 < N) throw std::runtime_error("radiance_mxfp4_fp8: partitions need the folded path");
  dim3 block(NWAVE * 32);
  dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
  // the tuned folded kernel wins at every shape measured, so it is the default whenever the
  // per-row reference exponent is available
  if (wref) {
    static const int tn4_min_m = [] {
      const char *e = getenv("RADIANCE_MXFP4_TN4_MIN_M");
      return e ? atoi(e) : 2048;
    }();
    // Decode band. Default 0 = dark; the branch must come first because 16 < tn4_min_m.
    static const int decode_max_m = [] {
      const char *e = getenv("RADIANCE_MXFP4_DECODE_MAX_M");
      return e ? atoi(e) : 0;
    }();
    // Weight layout, set once for the process by the loader: it decides whether the tensor handed
    // to us is the checkpoint's [N, K/2] or fragment order. Both kernels below read it, and it
    // MUST agree with radiance_mxfp4.py's WPERM -- disagreeing reads the weight as garbage.
    static const bool dec_wperm = [] {
      const char *e = getenv("RADIANCE_MXFP4_WPERM");
      return e && atoi(e) != 0;
    }();
    // Epilogue fast path. Default ON (it is bit-identical and worth 7.7-8.0% at prefill shapes),
    // but reachable so the batched-throughput A/B can be run against the SAME binary and server
    // config -- comparing against an older BetterBench run would confound it with every other
    // launcher knob that has changed since.
    static const bool epifast = [] {
      const char *e = getenv("RADIANCE_MXFP4_EPIFAST");
      return !e || atoi(e) != 0;
    }();
    // Streaming weight loads in the decode kernel (see ld_w). Default off until measured.
    static const bool dec_nt = [] {
      const char *e = getenv("RADIANCE_MXFP4_DECODE_NT");
      return e && atoi(e) != 0;
    }();
    dim3 fblock(NTHREADS);
    constexpr int DWN = DEC_DWN;                 // tuned: ~/mxfp4_work/tier5 + tier6/grid.hip
    constexpr int DBND = DWN * 16;
    const int dec_nblk = (N + DBND - 1) / DBND;
    // M <= 64 keeps the shipped policy verbatim (split_k_for + decode_bk64) -- the 16-concurrent
    // extension must not move a single cell of the 8-and-under band. M in (64, 128] is a NEW
    // band, measured cell-by-cell in tier7 (2026-08-29, DRAM-fed, all six shapes to M=128):
    //   nblk >= 110 (gate_up)  -> dks1/bk64   (bk128 is +38% at M=128: 251 vs 181 us)
    //   nblk >= 48  (n8192/7168)-> dks1/bk128 (the old fill=78 rule picked dks2, -5%)
    //   nblk <= 8   (gate48)   -> dks4/bk128  (widest split, as everywhere)
    //   nblk ~40, K >= 6144 (down: 68 k-slabs) -> M <= 80: dks2/bk128; else dks4/bk64
    //     (down goes non-monotonic: at M >= 96 the wide split + BK=64's occupancy wins,
    //      d4/b64 111.8 vs d2/b128 153.4 at M=128)
    //   nblk ~40, K <  6144 (out: 24 k-slabs)  -> dks1/bk128 (few slabs, splits never pay)
    int dks, bkx;
    if (M > 64) {
      if (dec_nblk >= 110)      { dks = 1; bkx = 64; }
      else if (dec_nblk >= 48)  { dks = 1; bkx = 128; }
      else if (dec_nblk <= 8)   { dks = DEC_KS; bkx = 128; }
      else if (K >= 6144)       { if (M <= 80) { dks = 2; bkx = 128; }
                                  else         { dks = DEC_KS; bkx = 64; } }
      else                      { dks = 1; bkx = 128; }
    } else {
      dks = split_k_for(dec_nblk, M);
      bkx = 0;   // resolved below from decode_bk64, exactly as shipped
    }
    // The decode kernel stages whole DBK-byte K slabs with no k bound (stage_a / stage_w read
    // A + row*K + k0 + c across the slab), so K must be a multiple of the slab. Every stock shape
    // is a multiple of 128 and never met this; the TP=3 dummy-head padding (radiance_tp3pad)
    // makes down_proj K = 17472 / 5824 (TP=1 / TP=3): a multiple of 64 but not 128, and at
    // BK=128 the last slab read the NEXT row's bytes -- a serve that emitted nothing but token 0
    // (2026-09-06). Take the BK=64 instantiation for such a K; it exists for split 1 and 4 only,
    // so the split-2 cell widens to 4. Inert when K % 128 == 0 (all stock shapes).
    if (K % 128) {
      bkx = 64;
      if (dks == 2) dks = DEC_KS;
    }
    // DKS==1 writes C directly and needs neither the partial buffer nor the block counter;
    // anything wider needs both. Sizing the guard by the CHOSEN ks (not the widest) also means a
    // shape that would have overflowed the scratch at DKS=4 can still take the kernel at DKS=1.
    const bool have_scratch = g_decode_partial && g_decode_cnt &&
        (size_t)dks * M * N * sizeof(float) <= g_decode_partial_bytes;
    if (M > 0 && M <= decode_max_m && M <= DEC_MTILE * DEC_MAX_TM && N <= DEC_MAX_N &&
        (dks == 1 || have_scratch)) {
      float *P = g_decode_partial;
      const dim3 dblock(DWN * 32);
      const int tm = (M + DEC_MTILE - 1) / DEC_MTILE;   // smallest tile that covers M
      if (bkx == 0) bkx = decode_bk64(tm, dks) ? 64 : 128;
#define RAD_DEC_LAUNCH(BK_, KS_, TM_)                                                         \
      hipLaunchKernelGGL((radiance_mxfp4_fp8_gemm_decode<DWN, BK_, KS_, TM_, RAD_WPERM, true,  \
                                                          RAD_NT, RAD_E6>),                   \
                         dim3(dec_nblk, 1, KS_), dblock,                                       \
                         0, (hipStream_t)stream, (const unsigned char *)a,                     \
                         (const unsigned char *)w, (const unsigned char *)ws,                  \
                         (const unsigned char *)wref, (const float *)as, P, g_decode_cnt,      \
                         (__bf16 *)c, M, N, K, pb1, pb2, astride, (const unsigned char *)wh)
#define RAD_DEC_BY_TM(BK_, KS_)                                                               \
      if (tm == 1)      RAD_DEC_LAUNCH(BK_, KS_, 1);                                           \
      else if (tm == 2) RAD_DEC_LAUNCH(BK_, KS_, 2);                                           \
      else if (tm == 3) RAD_DEC_LAUNCH(BK_, KS_, 3);                                           \
      else if (tm == 4) RAD_DEC_LAUNCH(BK_, KS_, 4);                                           \
      else if (tm == 5) RAD_DEC_LAUNCH(BK_, KS_, 5);                                           \
      else if (tm == 6) RAD_DEC_LAUNCH(BK_, KS_, 6);                                           \
      else if (tm == 7) RAD_DEC_LAUNCH(BK_, KS_, 7);                                           \
      else              RAD_DEC_LAUNCH(BK_, KS_, 8)
#define RAD_DEC_BY_KS                                                                         \
      if (dks <= 1)      { if (bkx == 64) { RAD_DEC_BY_TM(64, 1); }                            \
                           else           { RAD_DEC_BY_TM(128, 1); } }                         \
      else if (dks == 2) { RAD_DEC_BY_TM(128, 2); }                                            \
      else               { if (bkx == 64) { RAD_DEC_BY_TM(64, 4); }                            \
                           else           { RAD_DEC_BY_TM(128, 4); } }
      // Streaming (NT) weight loads pay ONLY under WPERM, where a wave's weight read is one
      // whole 128 B line: measured 2026-09-01 (tier7 ntab, DRAM-fed) -5..-8% on gate_up /
      // n8192 / n7168 / down at M<=48, ~0 at M=64, i.e. 97% of the 635 GB/s stream. On the
      // checkpoint layout a row's slab is half (DBK=128) or a quarter (64) of a line and the
      // rest is only wanted by the NEXT slab, so NT re-fetches every line: 2.0-3.6x SLOWER.
      // Two restagings that batch a thread's loads for the whole line (8 B x G, then one 16 B
      // per lane / 4 rows per wave) were STILL 1.4-1.8x under NT and -5..-19% without it, so
      // the row layout keeps plain loads and dec_nt is ignored there.
      if (wh) {
#define RAD_E6 true
#define RAD_WPERM true
        if (dec_nt) {
#define RAD_NT true
          RAD_DEC_BY_KS;
#undef RAD_NT
        } else {
#define RAD_NT false
          RAD_DEC_BY_KS;
#undef RAD_NT
        }
#undef RAD_WPERM
#undef RAD_E6
        return;
      }
#define RAD_E6 false
      if (dec_wperm) {
#define RAD_WPERM true
        if (dec_nt) {
#define RAD_NT true
          RAD_DEC_BY_KS;
#undef RAD_NT
        } else {
#define RAD_NT false
          RAD_DEC_BY_KS;
#undef RAD_NT
        }
#undef RAD_WPERM
      } else {
#define RAD_WPERM false
#define RAD_NT false
        RAD_DEC_BY_KS;
#undef RAD_NT
#undef RAD_WPERM
      }
#undef RAD_E6
#undef RAD_DEC_BY_KS
#undef RAD_DEC_BY_TM
#undef RAD_DEC_LAUNCH
      return;
      // no scratch (alloc failed) -- fall through to the folded kernel, never produce nothing
    }
#define RAD_FOLDED_E(TN_, WP_, EF_, E6_)                                                        \
    do {                                                                                        \
      constexpr int B_ = BNF_OF(TN_);                                                           \
      dim3 fgrid((N + B_ - 1) / B_, (M + BMF - 1) / BMF);                                       \
      hipLaunchKernelGGL((radiance_mxfp4_fp8_gemm_folded<TN_, WP_, EF_, E6_>), fgrid, fblock, 0, \
                         (hipStream_t)stream, (const unsigned char *)a,                         \
                         (const unsigned char *)w, (const unsigned char *)ws,                   \
                         (const unsigned char *)wref, (const float *)as, (__bf16 *)c, M, N, K, \
                         pb1, pb2, astride, (const unsigned char *)wh);                         \
    } while (0)
#define RAD_FOLDED(TN_, WP_, E6_)                                                               \
    do { if (epifast) RAD_FOLDED_E(TN_, WP_, true, E6_); else RAD_FOLDED_E(TN_, WP_, false, E6_); } while (0)
    // TN=4 only when the shape is wide enough to use the 128-wide tile: at N=48 (gdn gate)
    // TN=4 computes 128 columns against 48 real ones and measures 1.8-2x SLOWER than TN=2 at
    // every M >= 2048 (56 vs 101-115 us, ~/mxfp4_work/tier7 tnx). The M crossover itself
    // re-measured within noise of 2048 on all large-N shapes with branchless staging.
    if (M >= tn4_min_m && N >= BNF_OF(4)) {
      if (wh) RAD_FOLDED(4, true, true);
      else if (dec_wperm) RAD_FOLDED(4, true, false); else RAD_FOLDED(4, false, false);
    } else {
      if (wh) RAD_FOLDED(2, true, true);
      else if (dec_wperm) RAD_FOLDED(2, true, false); else RAD_FOLDED(2, false, false);
    }
#undef RAD_FOLDED
#undef RAD_FOLDED_E
  } else
    hipLaunchKernelGGL(radiance_mxfp4_fp8_gemm, grid, block, 0, (hipStream_t)stream,
                       (const unsigned char *)a, (const unsigned char *)w,
                       (const unsigned char *)ws, (const float *)as, (__bf16 *)c, M, N, K);
}

// Stock entry: one activation, one partition (bit-identical to before the partition select).
static void launch(uintptr_t a, uintptr_t w, uintptr_t ws, uintptr_t wref, uintptr_t as,
                   uintptr_t c, int M, int N, int K, uintptr_t stream) {
  launch_impl(a, w, ws, wref, as, c, M, N, K, 1 << 30, 1 << 30, 0, stream);
}
// Merged-linear entry (paroquant): a = [P, M, K] rotated copies astride bytes apart, as = [P, M];
// n-blocks in [0, pb1) read copy 0, [pb1, pb2) copy 1, [pb2, N) copy 2.
static void launch_p(uintptr_t a, uintptr_t w, uintptr_t ws, uintptr_t wref, uintptr_t as,
                     uintptr_t c, int M, int N, int K, int pb1, int pb2, long astride,
                     uintptr_t stream) {
  launch_impl(a, w, ws, wref, as, c, M, N, K, pb1, pb2, astride, stream);
}
static void launch6_p(uintptr_t a, uintptr_t w, uintptr_t wh, uintptr_t ws, uintptr_t wref,
                      uintptr_t as, uintptr_t c, int M, int N, int K, int pb1, int pb2,
                      long astride, uintptr_t stream) {
  launch_impl(a, w, ws, wref, as, c, M, N, K, pb1, pb2, astride, stream, wh);
}


// ---- fused residual-add + Gemma RMSNorm + per-token fp8 quant -------------------------------
//
// The decode-step epilogue of every RowParallel linear: after the all-reduce, the traced graph
// runs (add+rms_norm) and the traced per-token quant -- 2-3 inductor kernels per site whose
// tiling inductor chooses for the M=8192 compile hint and replays at M=8 (XBLOCK=1, one row per
// workgroup, ~4.4 us of kernels + 2 dispatch gaps per site). This kernel does the whole epilogue
// in ONE launch. Semantics mirror vllm.ir.ops.fused_add_rms_norm with the GemmaRMSNorm weight
// convention plus the radiance traced quant, in exactly its precision:
//   s      = f32(y) + f32(residual)         (fp32 sum feeds BOTH residual_out and the variance)
//   res_o  = bf16(s)
//   n      = bf16( s * rsqrt(mean(s^2)+eps) * (f32(w)+1) )   <- Gemma: weight is (1+w), fp32
//   scale  = max(amax|f32(n)|/448, 1/(448*512))               <- amax over the BF16-ROUNDED n
//   q      = e4m3_rne(clamp(f32(n)/scale, +-448))
// One workgroup per row; latency-bound at decode M<=8 like everything else in the step, and at
// prefill M=chunk there are thousands of rows so occupancy takes care of itself.
#define ARNQ_T 256
typedef unsigned int uint32_tt;
union ArnqVec8 { uint4 u4; __bf16 h[8]; };
template <int ART, bool TILED = false>
__global__ __launch_bounds__(ART) void radiance_add_rms_quant(
    const __bf16 *__restrict__ y, const __bf16 *__restrict__ res, const __bf16 *__restrict__ w,
    unsigned char *__restrict__ q, float *__restrict__ scale, __bf16 *__restrict__ res_out,
    int K, float eps) {
  const int row = blockIdx.x;
  const int tid = threadIdx.x;
  y += (long)row * K; res += (long)row * K; res_out += (long)row * K;
  if (!TILED) q += (long)row * K;
  // TILED: q is the fragment-tiled layout the atiled GEMM reads (see there): 8-group g of this
  // row lands in fragment (row/16, g/2) at lane (row%16) + 16*(g%2). Same single uint2 store;
  // the scatter across fragments measured +0.3..+3.8% on this kernel (tier7/aq.hip).
  const int mt = row >> 4, lrow = row & 15, kst = K >> 4;
  // 16-byte vectorized: thread t owns 8-element groups t, t+T, t+2T, ... The scalar version of
  // this kernel measured 12.5 us at M=8 under graph replay -- 20 independent 2-byte loads per
  // thread per tensor is pure latency serialization at 8 workgroups. Vectorized it is one
  // uint4 load per tensor per group. K must be a multiple of 8 (5120/8704/2048 all are).
  constexpr int MAXG = 5;                       // 8704 / (256*8) rounds up to 5
  const int KV = K >> 3;                        // groups of 8
  float sv[MAXG][8];
  int ng = 0;
  float ssq = 0.f;
  for (int g = tid; g < KV; g += ART, ++ng) {
    ArnqVec8 vy, vr;
    vy.u4 = ((const uint4 *)y)[g];
    vr.u4 = ((const uint4 *)res)[g];
    ArnqVec8 vo;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      float v = (float)vy.h[j] + (float)vr.h[j];
      sv[ng][j] = v;
      vo.h[j] = (__bf16)v;
      ssq += v * v;
    }
    ((uint4 *)res_out)[g] = vo.u4;
  }
  __shared__ float lds[ART / 32];
  for (int o = 16; o; o >>= 1) ssq += __shfl_down(ssq, o);
  if ((tid & 31) == 0) lds[tid >> 5] = ssq;
  __syncthreads();
  if (tid < ART / 32) {
    float v = lds[tid];
    for (int o = ART / 64; o; o >>= 1) v += __shfl_down(v, o);
    if (tid == 0) lds[0] = v;
  }
  __syncthreads();
  const float inv = rsqrtf(lds[0] / (float)K + eps);
  __syncthreads();                       // lds[0] is reused for the amax reduction below
  float amax = 0.f;
  int mg = 0;
  for (int g = tid; g < KV; g += ART, ++mg) {
    ArnqVec8 vw;
    vw.u4 = ((const uint4 *)w)[g];
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      float nv = (float)(__bf16)(sv[mg][j] * inv * ((float)vw.h[j] + 1.f));
      sv[mg][j] = nv;
      amax = fmaxf(amax, fabsf(nv));
    }
  }
  for (int o = 16; o; o >>= 1) amax = fmaxf(amax, __shfl_down(amax, o));
  if ((tid & 31) == 0) lds[tid >> 5] = amax;
  __syncthreads();
  if (tid < ART / 32) {
    float v = lds[tid];
    for (int o = ART / 64; o; o >>= 1) v = fmaxf(v, __shfl_down(v, o));
    if (tid == 0) lds[0] = v;
  }
  __syncthreads();
  const float sc = fmaxf(lds[0] * (1.f / 448.f), 1.f / (448.f * 512.f));
  const float rs = 1.f / sc;
  if (tid == 0) scale[row] = sc;
  mg = 0;
  for (int g = tid; g < KV; g += ART, ++mg) {
    uint32_tt lo = 0, hi = 0;
#pragma unroll
    for (int j = 0; j < 4; j += 2) {
      float a = fminf(fmaxf(sv[mg][j] * rs, -448.f), 448.f);
      float b = fminf(fmaxf(sv[mg][j + 1] * rs, -448.f), 448.f);
      lo |= (__builtin_amdgcn_cvt_pk_fp8_f32(a, b, 0u, false) & 0xffffu) << (j * 8);
    }
#pragma unroll
    for (int j = 4; j < 8; j += 2) {
      float a = fminf(fmaxf(sv[mg][j] * rs, -448.f), 448.f);
      float b = fminf(fmaxf(sv[mg][j + 1] * rs, -448.f), 448.f);
      hi |= (__builtin_amdgcn_cvt_pk_fp8_f32(a, b, 0u, false) & 0xffffu) << ((j - 4) * 8);
    }
    if (TILED) ((uint2 *)q)[(mt * kst + (g >> 1)) * 32 + lrow + 16 * (g & 1)] = make_uint2(lo, hi);
    else ((uint2 *)q)[g] = make_uint2(lo, hi);
  }
}

// ---- fused SiLU-mul + per-token fp8 quant ----------------------------------------------------
// The MLP's activation epilogue: gate_up output [M, 2N] -> silu(gate) * up -> per-token e4m3.
// Same skeleton and precision contract as radiance_add_rms_quant: every intermediate is rounded
// exactly where the traced path rounds (silu to bf16, product to bf16, amax over the rounded
// product). One workgroup per row.
template <int ART, bool TILED = false>
__global__ __launch_bounds__(ART) void radiance_silu_mul_quant(
    const __bf16 *__restrict__ gu, unsigned char *__restrict__ q, float *__restrict__ scale,
    int N) {
  const int row = blockIdx.x;
  const int tid = threadIdx.x;
  gu += (long)row * 2 * N;
  if (!TILED) q += (long)row * N;
  const int mt = row >> 4, lrow = row & 15, kst = N >> 4;   // TILED: as in add_rms_quant
  constexpr int MAXG = 5;                  // N <= 10240 (8704 in this model)
  const int NV = N >> 3;
  float pv[MAXG][8];
  int ng = 0;
  float amax = 0.f;
  for (int g = tid; g < NV; g += ART, ++ng) {
    ArnqVec8 vg, vu;
    vg.u4 = ((const uint4 *)gu)[g];
    vu.u4 = ((const uint4 *)(gu + N))[g];
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      float x = (float)vg.h[j];
      float t = (float)(__bf16)(x / (1.f + expf(-x)));       // silu, rounded to bf16
      float p = (float)(__bf16)(t * (float)vu.h[j]);         // product, rounded to bf16
      pv[ng][j] = p;
      amax = fmaxf(amax, fabsf(p));
    }
  }
  __shared__ float lds[ART / 32];
  for (int o = 16; o; o >>= 1) amax = fmaxf(amax, __shfl_down(amax, o));
  if ((tid & 31) == 0) lds[tid >> 5] = amax;
  __syncthreads();
  if (tid < ART / 32) {
    float v = lds[tid];
    for (int o = ART / 64; o; o >>= 1) v = fmaxf(v, __shfl_down(v, o));
    if (tid == 0) lds[0] = v;
  }
  __syncthreads();
  const float sc = fmaxf(lds[0] * (1.f / 448.f), 1.f / (448.f * 512.f));
  const float rs = 1.f / sc;
  if (tid == 0) scale[row] = sc;
  int mg = 0;
  for (int g = tid; g < NV; g += ART, ++mg) {
    uint32_tt lo = 0, hi = 0;
#pragma unroll
    for (int j = 0; j < 4; j += 2) {
      float a = fminf(fmaxf(pv[mg][j] * rs, -448.f), 448.f);
      float b = fminf(fmaxf(pv[mg][j + 1] * rs, -448.f), 448.f);
      lo |= (__builtin_amdgcn_cvt_pk_fp8_f32(a, b, 0u, false) & 0xffffu) << (j * 8);
    }
#pragma unroll
    for (int j = 4; j < 8; j += 2) {
      float a = fminf(fmaxf(pv[mg][j] * rs, -448.f), 448.f);
      float b = fminf(fmaxf(pv[mg][j + 1] * rs, -448.f), 448.f);
      hi |= (__builtin_amdgcn_cvt_pk_fp8_f32(a, b, 0u, false) & 0xffffu) << ((j - 4) * 8);
    }
    if (TILED) ((uint2 *)q)[(mt * kst + (g >> 1)) * 32 + lrow + 16 * (g & 1)] = make_uint2(lo, hi);
    else ((uint2 *)q)[g] = make_uint2(lo, hi);
  }
}

static void launch_silu_mul_quant(uintptr_t gu, uintptr_t q, uintptr_t scale, int M, int N,
                                  uintptr_t stream, int tiled) {
  if (N > 40 * 512 || (N & 7))
    throw std::runtime_error("radiance_silu_mul_quant: N must be 8-aligned and <= 20480");
  if (tiled && (N & 15)) throw std::runtime_error("radiance_silu_mul_quant: tiled needs N % 16 == 0");
  // Above the 256-thread block's register budget (MAXG=5 groups of 8 -> N <= 10240, which is
  // every TP>=2 shape) the 512-thread instantiation covers N <= 20480: TP=1's intermediate is
  // 17408 (2026-09-16). Shapes at or below 10240 launch exactly as before.
  if (N > 40 * ARNQ_T) {
    if (tiled)
      hipLaunchKernelGGL((radiance_silu_mul_quant<512, true>), dim3(M), dim3(512), 0,
                         (hipStream_t)stream, (const __bf16 *)gu, (unsigned char *)q, (float *)scale, N);
    else
      hipLaunchKernelGGL((radiance_silu_mul_quant<512>), dim3(M), dim3(512), 0, (hipStream_t)stream,
                         (const __bf16 *)gu, (unsigned char *)q, (float *)scale, N);
    return;
  }
  if (tiled) {
    if (M >= 2048)
      hipLaunchKernelGGL((radiance_silu_mul_quant<512, true>), dim3(M), dim3(512), 0,
                         (hipStream_t)stream, (const __bf16 *)gu, (unsigned char *)q, (float *)scale, N);
    else
      hipLaunchKernelGGL((radiance_silu_mul_quant<ARNQ_T, true>), dim3(M), dim3(ARNQ_T), 0,
                         (hipStream_t)stream, (const __bf16 *)gu, (unsigned char *)q, (float *)scale, N);
    return;
  }
  if (M >= 2048)
    hipLaunchKernelGGL((radiance_silu_mul_quant<512>), dim3(M), dim3(512), 0, (hipStream_t)stream,
                       (const __bf16 *)gu, (unsigned char *)q, (float *)scale, N);
  else
    hipLaunchKernelGGL((radiance_silu_mul_quant<ARNQ_T>), dim3(M), dim3(ARNQ_T), 0,
                       (hipStream_t)stream, (const __bf16 *)gu, (unsigned char *)q,
                       (float *)scale, N);
}

// ---- fused per-head gated RMSNorm + per-token fp8 quant (GDN out_proj input) ----------------
// Replaces RMSNormGated(norm_before_gate=True, group per 128-wide head) + the traced per-token
// quant in _output_projection, which inductor emits as two kernels per GDN layer (a variance
// reduction and a normalize+gate+quant). Precision contract mirrors radiance_add_rms_quant and
// the traced reference: fp32 throughout, ((x * rsqrt(mean(x^2) + eps)) * w) * silu(z) rounded
// to bf16 once, amax over the rounded values, e4m3 with the vLLM min-scale clamp. The per-head
// sum of squares is a 16-lane xor-shuffle: 16 consecutive 8-groups make one head and, with ART a
// multiple of 16 and N a multiple of 128, every 16-lane segment is either fully active or fully
// idle in each pass. x: [M, N] contiguous bf16; z: [M, N] bf16 with row stride zs (a column
// slice of the fused qkvz projection, no copy); w: [128] bf16 shared by all heads.
template <int ART>
__global__ __launch_bounds__(ART) void radiance_gdn_norm_quant(
    const __bf16 *__restrict__ x, const __bf16 *__restrict__ z, long zs,
    const __bf16 *__restrict__ w, unsigned char *__restrict__ q, float *__restrict__ scale,
    int N, float eps) {
  const int row = blockIdx.x;
  const int tid = threadIdx.x;
  x += (long)row * N; z += (long)row * zs; q += (long)row * N;
  constexpr int MAXG = 5;                       // N <= ART*8*5 = 10240 (3072 in this model)
  const int NV = N >> 3;                        // 8-element groups; 16 per head
  float sv[MAXG][8];
  int ng = 0;
  float amax = 0.f;
  for (int g = tid; g < NV; g += ART, ++ng) {
    ArnqVec8 vx, vz, vw;
    vx.u4 = ((const uint4 *)x)[g];
    float ssq = 0.f;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      const float v = (float)vx.h[j];
      sv[ng][j] = v;
      ssq += v * v;
    }
    for (int o = 8; o; o >>= 1) ssq += __shfl_xor(ssq, o, 16);
    const float inv = rsqrtf(ssq * (1.f / 128.f) + eps);
    vz.u4 = ((const uint4 *)z)[g];
    vw.u4 = ((const uint4 *)w)[g & 15];
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      const float zz = (float)vz.h[j];
      const float sg = zz / (1.f + expf(-zz));                       // silu, fp32
      const float nv = (float)(__bf16)(((sv[ng][j] * inv) * (float)vw.h[j]) * sg);
      sv[ng][j] = nv;
      amax = fmaxf(amax, fabsf(nv));
    }
  }
  __shared__ float lds[ART / 32];
  for (int o = 16; o; o >>= 1) amax = fmaxf(amax, __shfl_down(amax, o));
  if ((tid & 31) == 0) lds[tid >> 5] = amax;
  __syncthreads();
  if (tid < ART / 32) {
    float v = lds[tid];
    for (int o = ART / 64; o; o >>= 1) v = fmaxf(v, __shfl_down(v, o));
    if (tid == 0) lds[0] = v;
  }
  __syncthreads();
  const float sc = fmaxf(lds[0] * (1.f / 448.f), 1.f / (448.f * 512.f));
  const float rs = 1.f / sc;
  if (tid == 0) scale[row] = sc;
  int mg = 0;
  for (int g = tid; g < NV; g += ART, ++mg) {
    uint32_tt lo = 0, hi = 0;
#pragma unroll
    for (int j = 0; j < 4; j += 2) {
      float a = fminf(fmaxf(sv[mg][j] * rs, -448.f), 448.f);
      float b = fminf(fmaxf(sv[mg][j + 1] * rs, -448.f), 448.f);
      lo |= (__builtin_amdgcn_cvt_pk_fp8_f32(a, b, 0u, false) & 0xffffu) << (j * 8);
    }
#pragma unroll
    for (int j = 4; j < 8; j += 2) {
      float a = fminf(fmaxf(sv[mg][j] * rs, -448.f), 448.f);
      float b = fminf(fmaxf(sv[mg][j + 1] * rs, -448.f), 448.f);
      hi |= (__builtin_amdgcn_cvt_pk_fp8_f32(a, b, 0u, false) & 0xffffu) << ((j - 4) * 8);
    }
    ((uint2 *)q)[g] = make_uint2(lo, hi);
  }
}

static void launch_gdn_norm_quant(uintptr_t x, uintptr_t z, long zs, uintptr_t w, uintptr_t q,
                                  uintptr_t scale, int M, int N, double eps, uintptr_t stream) {
  if (N > 40 * ARNQ_T || (N & 127))
    throw std::runtime_error("radiance_gdn_norm_quant: N must be a multiple of 128 and <= 10240");
  if (zs & 7) throw std::runtime_error("radiance_gdn_norm_quant: z row stride must be 8-aligned");
  hipLaunchKernelGGL((radiance_gdn_norm_quant<ARNQ_T>), dim3(M), dim3(ARNQ_T), 0,
                     (hipStream_t)stream, (const __bf16 *)x, (const __bf16 *)z, zs,
                     (const __bf16 *)w, (unsigned char *)q, (float *)scale, N, (float)eps);
}

static void launch_add_rms_quant(uintptr_t y, uintptr_t res, uintptr_t w, uintptr_t q,
                                 uintptr_t scale, uintptr_t res_out, int M, int K, double eps,
                                 uintptr_t stream, int tiled) {
  if (K > 40 * ARNQ_T || (K & 7)) throw std::runtime_error("radiance_add_rms_quant: K must be 8-aligned and <= 10240");
  if (tiled) {
    if (K & 15) throw std::runtime_error("radiance_add_rms_quant: tiled needs K % 16 == 0");
    if (M >= 2048)
      hipLaunchKernelGGL((radiance_add_rms_quant<512, true>), dim3(M), dim3(512), 0, (hipStream_t)stream,
                         (const __bf16 *)y, (const __bf16 *)res, (const __bf16 *)w,
                         (unsigned char *)q, (float *)scale, (__bf16 *)res_out, K, (float)eps);
    else
      hipLaunchKernelGGL((radiance_add_rms_quant<ARNQ_T, true>), dim3(M), dim3(ARNQ_T), 0,
                         (hipStream_t)stream, (const __bf16 *)y, (const __bf16 *)res,
                         (const __bf16 *)w, (unsigned char *)q, (float *)scale,
                         (__bf16 *)res_out, K, (float)eps);
    return;
  }
  // T=512 at prefill-class M: measured 726 -> 685 us (405 -> 429 GB/s) at M=8192 on a clean
  // GPU. Changes the per-row summation order (512-way vs 256-way striding) => gated by GSM8K,
  // not byte-compare. Decode-class M keeps T=256 (T=512 measured worse at small M in the nq
  // sweep). The PREFETCH variant this replaces measured 828 us at M=8192 -- 14% WORSE than the
  // original -- its +40 VGPRs/thread costs occupancy exactly when 8192 blocks compete; the
  // M=2048 idle-GPU microbench that justified it was the fill-the-GPU-first trap.
  if (M >= 2048)
    hipLaunchKernelGGL((radiance_add_rms_quant<512>), dim3(M), dim3(512), 0, (hipStream_t)stream,
                       (const __bf16 *)y, (const __bf16 *)res, (const __bf16 *)w,
                       (unsigned char *)q, (float *)scale, (__bf16 *)res_out, K, (float)eps);
  else
    hipLaunchKernelGGL((radiance_add_rms_quant<ARNQ_T>), dim3(M), dim3(ARNQ_T), 0,
                       (hipStream_t)stream, (const __bf16 *)y, (const __bf16 *)res,
                       (const __bf16 *)w, (unsigned char *)q, (float *)scale,
                       (__bf16 *)res_out, K, (float)eps);
}

// GEMM on a fragment-tiled activation (see radiance_mxfp4_fp8_gemm_atiled). The caller decides
// the layout per tensor (radiance_mxfp4.py a_tiled_*); this only dispatches. Prefill-class M
// only: the decode kernel reads row-major A, so the Python side never tiles below its band.
static void launch_at_impl(uintptr_t a, uintptr_t w, uintptr_t ws, uintptr_t wref, uintptr_t as,
                           uintptr_t c, int M, int N, int K, int pb1, int pb2, long astride,
                           uintptr_t stream, uintptr_t wh = 0) {
  if (K % 128) throw std::runtime_error("radiance_mxfp4_fp8: tiled A needs K % 128 == 0");
  if ((pb1 < N && pb1 % 128) || (pb2 < N && pb2 % 128))
    throw std::runtime_error("radiance_mxfp4_fp8: partition boundaries must be multiples of 128");
  if (!wref) throw std::runtime_error("radiance_mxfp4_fp8: tiled A needs the folded path (wref)");
  static const int tn4_min_m = [] {
    const char *e = getenv("RADIANCE_MXFP4_TN4_MIN_M");
    return e ? atoi(e) : 2048;
  }();
  static const bool wperm = [] {
    const char *e = getenv("RADIANCE_MXFP4_WPERM");
    return e && atoi(e) != 0;
  }();
  dim3 fblock(NTHREADS);
#define RAD_AT(TN_, WP_, E6_)                                                                   \
  do {                                                                                          \
    constexpr int B_ = BNF_OF(TN_);                                                             \
    dim3 fgrid((N + B_ - 1) / B_, (M + BMF - 1) / BMF);                                         \
    hipLaunchKernelGGL((radiance_mxfp4_fp8_gemm_atiled<TN_, WP_, (TN_ == 2 ? 128 : (E6_ ? 64 : 128)), E6_>), \
                       fgrid, fblock, 0,                                                        \
                       (hipStream_t)stream, (const unsigned char *)a, (const unsigned char *)w, \
                       (const unsigned char *)ws, (const unsigned char *)wref, (const float *)as, \
                       (__bf16 *)c, M, N, K, pb1, pb2, astride, (const unsigned char *)wh);    \
  } while (0)
  if (M >= tn4_min_m && N >= BNF_OF(4)) {
    if (wh) RAD_AT(4, true, true); else if (wperm) RAD_AT(4, true, false); else RAD_AT(4, false, false);
  } else {
    if (wh) RAD_AT(2, true, true); else if (wperm) RAD_AT(2, true, false); else RAD_AT(2, false, false);
  }
#undef RAD_AT
}
static void launch_at(uintptr_t a, uintptr_t w, uintptr_t ws, uintptr_t wref, uintptr_t as,
                      uintptr_t c, int M, int N, int K, uintptr_t stream) {
  launch_at_impl(a, w, ws, wref, as, c, M, N, K, 1 << 30, 1 << 30, 0, stream);
}
static void launch_at_p(uintptr_t a, uintptr_t w, uintptr_t ws, uintptr_t wref, uintptr_t as,
                        uintptr_t c, int M, int N, int K, int pb1, int pb2, long astride,
                        uintptr_t stream) {
  launch_at_impl(a, w, ws, wref, as, c, M, N, K, pb1, pb2, astride, stream);
}
static void launch6_at_p(uintptr_t a, uintptr_t w, uintptr_t wh, uintptr_t ws, uintptr_t wref,
                         uintptr_t as, uintptr_t c, int M, int N, int K, int pb1, int pb2,
                         long astride, uintptr_t stream) {
  launch_at_impl(a, w, ws, wref, as, c, M, N, K, pb1, pb2, astride, stream, wh);
}


// ---- f32-out launcher for llama.cpp integration (TN policy mirrors launch_impl) ----
static void launch_f32(uintptr_t a, uintptr_t w, uintptr_t ws, uintptr_t wref, uintptr_t as,
                       uintptr_t cf, int M, int N, int K, uintptr_t stream) {
    dim3 fblock(NTHREADS);
    if (M >= 2048 && N >= BNF_OF(4)) {
        constexpr int B_ = BNF_OF(4);
        dim3 fgrid((N + B_ - 1) / B_, (M + BMF - 1) / BMF);
        hipLaunchKernelGGL((radiance_mxfp4_fp8_gemm_folded<4, false, true, false, true>),
                           fgrid, fblock, 0, (hipStream_t)stream,
                           (const unsigned char *)a, (const unsigned char *)w,
                           (const unsigned char *)ws, (const unsigned char *)wref,
                           (const float *)as, (__bf16 *)nullptr, M, N, K, 1 << 30, 1 << 30, 0,
                           (const unsigned char *)nullptr, (float *)cf);
    } else {
        constexpr int B_ = BNF_OF(2);
        dim3 fgrid((N + B_ - 1) / B_, (M + BMF - 1) / BMF);
        hipLaunchKernelGGL((radiance_mxfp4_fp8_gemm_folded<2, false, true, false, true>),
                           fgrid, fblock, 0, (hipStream_t)stream,
                           (const unsigned char *)a, (const unsigned char *)w,
                           (const unsigned char *)ws, (const unsigned char *)wref,
                           (const float *)as, (__bf16 *)nullptr, M, N, K, 1 << 30, 1 << 30, 0,
                           (const unsigned char *)nullptr, (float *)cf);
    }
}

// ===================== llama.cpp integration =====================

// repack: llama block_mxfp4 rows [N, nb] x 17B (row stride s01 blocks) ->
//        W [N, K/2] qs bytes; Ws [nb, N] scale bytes (e8m0 byte as-is); Wref [N] row max e.
__global__ void radiance_repack_kernel(const char * __restrict__ src, int64_t s01, int nb,
                                        unsigned char * __restrict__ W,
                                        unsigned char * __restrict__ Ws,
                                        unsigned char * __restrict__ Wref) {
    const int n = blockIdx.x;               // one row per block
    const int tid = threadIdx.x;
    const int stride = blockDim.x;
    const uint8_t * row = (const uint8_t *) src + (int64_t)n * s01 * 17;
    for (int b = tid; b < nb; b += stride) {
        const uint8_t * blk = row + (int64_t)b * 17;
        const uint8_t e = blk[0];
        Ws[(int64_t)b * gridDim.x + n] = e;   // e8m0 byte = radiance scale byte
        // llama nibble order is split-half (qs byte j: lo=element j, hi=element j+16);
        // the radiance kernel expects interleaved (byte m: lo=element 2m, hi=element 2m+1).
        // Read byte-by-byte: &blk[1] is only 1-byte aligned (17-byte stride), a vector load here is UB.
        unsigned char lo[16], hi[16], outb[16];
#pragma unroll
        for (int j = 0; j < 16; ++j) { lo[j] = blk[1 + j] & 0x0F; hi[j] = blk[1 + j] >> 4; }
#pragma unroll
        for (int m = 0; m < 8; ++m) {
            outb[m]     = lo[2*m] | (lo[2*m + 1] << 4);
            outb[8 + m] = hi[2*m] | (hi[2*m + 1] << 4);
        }
        for (int m = 0; m < 16; ++m) W[(size_t)n * (nb * 16) + b * 16 + m] = outb[m];
    }
    // row max e for Wref via shared reduce
    __shared__ uint8_t shmax[256];
    uint8_t local = 0;
    for (int b = tid; b < nb; b += stride) {
        const uint8_t e = row[(int64_t)b * 17];
        if (e > local) local = e;
    }
    shmax[tid] = local;
    __syncthreads();
    for (int off = 128; off > 0; off >>= 1) {
        if (tid < off && shmax[tid + off] > shmax[tid]) shmax[tid] = shmax[tid + off];
        __syncthreads();
    }
    if (tid == 0) Wref[n] = shmax[0];
}

// per-token e4m3 quantize with row-hash memoization:
//   x [M, K] (row stride sx) f32 -> q [M, K] e4m3 + scale [M].
//   A row is requantized only when its 128-sample 64-bit hash changed since the last call for the
//   same persistent q buffer. ffn_gate and ffn_up read one and the same activation, so the second
//   GEMM of the pair skips the pass. Device-side check only: graph-capture safe, worst case one
//   redundant requantize, and a hash collision can only reuse a stale-but-equal row.
//
// NIT > 0 holds the row in registers so the amax scan and the quantize pass share one global
// read; it needs K <= blockDim.x * 4 * NIT. NIT == 0 keeps the two-pass form for larger K.
template <int NIT>
__global__ void quantize_tokens_fp8(const float * __restrict__ x, int64_t sx, int64_t K,
                                    unsigned char * __restrict__ q, float * __restrict__ scale,
                                    uint2 * __restrict__ row_hash) {
    const int64_t row = blockIdx.y;
    const int tid = threadIdx.x;
    const int nthreads = blockDim.x;
    const float * xr = x + row * sx;
    __shared__ uint2 h;
    __shared__ int skip;
    if (tid < 64) {
        uint32_t h0 = 0x9e3779b9u, h1 = 0x27d4eb2fu;
        for (int i = tid; i < 128; i += 64) {
            const uint32_t b = __float_as_uint(xr[(int64_t)(i * 37) % K]);
            h0 = (h0 ^ b) * 0x85ebca6bu; const uint32_t t1 = h1 ^ (b + 0x9e3779b9u + (h0 << 6) + (h0 >> 2)); h1 = ((t1 << 13) | (t1 >> 19)) * 0xc2b2ae35u;
        }
        h.x = h0; h.y = h1;
    }
    __syncthreads();
    if (tid == 0) {
        const uint2 old = row_hash[row];
        skip = (old.x == h.x && old.y == h.y);
    }
    __syncthreads();
    if (skip) {
        return;
    }
    if (tid == 0) {
        row_hash[row] = h;
    }
    float4 v[NIT > 0 ? NIT : 1];
    float local = 0.0f;
    if (NIT > 0) {
#pragma unroll
        for (int i = 0; i < NIT; ++i) {
            const int64_t k = (int64_t)(tid + i * nthreads) * 4;
            if (k < K) {
                v[i] = *(const float4 *)(xr + k);
                local = fmaxf(local, fmaxf(fmaxf(fabsf(v[i].x), fabsf(v[i].y)), fmaxf(fabsf(v[i].z), fabsf(v[i].w))));
            }
        }
    } else {
        for (int64_t k = tid * 4; k < K; k += (int64_t)nthreads * 4) {
            const float4 t = *(const float4 *)(xr + k);
            local = fmaxf(local, fmaxf(fmaxf(fabsf(t.x), fabsf(t.y)), fmaxf(fabsf(t.z), fabsf(t.w))));
        }
    }
    __shared__ float sh[32];
    for (int off = 16; off > 0; off >>= 1)
        local = fmaxf(local, __shfl_xor(local, off));
    if ((tid & 31) == 0) sh[tid >> 5] = local;
    __syncthreads();
    if (tid == 0) {
        float amax = 0.0f;
        for (int i = 0; i < nthreads / 32; ++i) amax = fmaxf(amax, sh[i]);
        scale[row] = amax > 0.0f ? amax / 448.0f : 0.0f;
    }
    __syncthreads();
    const float d = scale[row];
    const float dinv = d > 0.0f ? 1.0f / d : 0.0f;
    // fragment-tiled output for the atiled GEMM: each 16m x 16k fp8 fragment is 256 contiguous
    // bytes; k..k+3 never straddles an 8-byte group (K % 8 == 0, tid*4 aligned), one u32 store fits.
    const int64_t Kt = K >> 4;
    const int64_t frag_row = (row >> 4) * Kt;
    const int lrow = (int)(row & 15);
    if (NIT > 0) {
#pragma unroll
        for (int i = 0; i < NIT; ++i) {
            const int64_t k = (int64_t)(tid + i * nthreads) * 4;
            if (k < K) {
                const uint32_t p0 = ggml_cuda_fp32x2_to_e4m3x2(v[i].x * dinv, v[i].y * dinv);
                const uint32_t p1 = ggml_cuda_fp32x2_to_e4m3x2(v[i].z * dinv, v[i].w * dinv);
                const int64_t pos = (frag_row + (k >> 4)) * 256 + (lrow + 16 * ((int)((k >> 3) & 1))) * 8 + (k & 7);
                *(uint32_t *)(q + pos) = (p0 & 0xFFFFu) | ((p1 & 0xFFFFu) << 16);
            }
        }
    } else {
        for (int64_t k = tid * 4; k < K; k += (int64_t)nthreads * 4) {
            const float4 t = *(const float4 *)(xr + k);
            const uint32_t p0 = ggml_cuda_fp32x2_to_e4m3x2(t.x * dinv, t.y * dinv);
            const uint32_t p1 = ggml_cuda_fp32x2_to_e4m3x2(t.z * dinv, t.w * dinv);
            const int64_t pos = (frag_row + (k >> 4)) * 256 + (lrow + 16 * ((int)((k >> 3) & 1))) * 8 + (k & 7);
            *(uint32_t *)(q + pos) = (p0 & 0xFFFFu) | ((p1 & 0xFFFFu) << 16);
        }
    }
}

// public entries (declared in radiance-gemm.cuh)
// MXFP4_RAD zero-copy: buffer holds per tensor row [nb*16 interleaved codes][nb e8m0 scales]
// (row-major scales). radiance wants Ws[b*N + n]; gather once per tensor.
__global__ void radiance_gather_scales_kernel(const unsigned char * __restrict__ src_rad, int N, int nb,
                                              unsigned char * __restrict__ Ws,
                                              unsigned char * __restrict__ Wref) {
    // rad2 layout: code plane [N][nb*16] at offset 0, scale plane [N][nb] at offset N*nb*16
    const int n = blockIdx.x;               // one row per block
    const int tid = threadIdx.x;
    const int stride = blockDim.x;
    const unsigned char * sc = src_rad + (int64_t)N * nb * 16 + (int64_t)n * nb;
    uint8_t lmax = 0;
    for (int b = tid; b < nb; b += stride) {
        const uint8_t e = sc[b];
        if (Ws != nullptr) Ws[(int64_t)b * gridDim.x + n] = e;   // Wref-only build: null Ws ok
        if (e > lmax) lmax = e;
    }
    __shared__ uint8_t shmax[256];
    shmax[tid] = lmax;
    __syncthreads();
    for (int off = 128; off > 0; off >>= 1) {
        if (tid < off && shmax[tid + off] > shmax[tid]) shmax[tid] = shmax[tid + off];
        __syncthreads();
    }
    if (tid == 0) Wref[n] = shmax[0];
}

// MXFP4_RAD -> standard mxfp4 (17B blocks: e8m0 + 16B split-half codes), inverse of the
// plane transform. Used once per tensor when decode (MMQ/MMVQ) needs the raw layout.
__global__ void radiance_unrad_kernel(const unsigned char * __restrict__ src_rad, int N, int nb,
                                      unsigned char * __restrict__ dst_raw) {
    // rad2 layout: code plane at 0, scale plane at N*nb*16
    const int n = blockIdx.x;
    const int tid = threadIdx.x;
    const int stride = blockDim.x;
    const unsigned char * qs = src_rad + (int64_t)n * (nb * 16);
    const unsigned char * sc = src_rad + (int64_t)N * nb * 16 + (int64_t)n * nb;
    unsigned char * out = dst_raw + (int64_t)n * nb * 17;
    for (int b = tid; b < nb; b += stride) {
        unsigned char * blk = out + (int64_t)b * 17;
        blk[0] = sc[b];
        const unsigned char * p = qs + b * 16;
        // radi byte m (m<8):   lo nibble = elem 2m,   hi nibble = elem 2m+1
        // radi byte 8+m:       lo nibble = elem 16+2m, hi nibble = elem 16+2m+1
        // llama byte j:        lo nibble = elem j,   hi nibble = elem 16+j
        unsigned char elem[32];
        for (int m = 0; m < 8; ++m) {
            elem[2*m]     = p[m] & 0x0F;
            elem[2*m+1]   = p[m] >> 4;
            elem[16+2*m]  = p[8+m] & 0x0F;
            elem[16+2*m+1]= p[8+m] >> 4;
        }
        for (int j = 0; j < 16; ++j) {
            blk[1 + j] = elem[j] | (elem[16 + j] << 4);
        }
    }
}

void ggml_cuda_radiance_gather_scales(const unsigned char * src_rad, int N, int K,
                                      unsigned char * Ws, unsigned char * Wref,
                                      cudaStream_t stream) {
    const int nb = (int)(K / 32);
    radiance_gather_scales_kernel<<<N, 256, 0, stream>>>(src_rad, N, nb, Ws, Wref);
}

// MXFP6 packed: the per-row reference exponent is the max over the row's block e[] tails. There
// is no separate scale plane (each 200-byte block carries its own eight), so unlike the RAD path
// this reads the checkpoint bytes directly. Ws is unused; the fold reads the same tails in situ.
__global__ void radiance_gather_scales_mxfp6_kernel(const unsigned char * __restrict__ src, int N,
                                                    int nb, unsigned char * __restrict__ Wref) {
    const int n = blockIdx.x;               // one row per block
    const int tid = threadIdx.x;
    const int stride = blockDim.x;
    const unsigned char * row = src + (size_t) n * nb * sizeof(block_mxfp6);
    uint8_t local = 0;
    for (int b = tid; b < nb; b += stride) {
        const unsigned char * e = row + (size_t) b * sizeof(block_mxfp6) + 6 * QK_MXFP6 / 8;
        for (int s = 0; s < QK_MXFP6 / QK_MXFP6_SUB; ++s) {
            if (e[s] > local) local = e[s];
        }
    }
    __shared__ uint8_t shmax[256];
    shmax[tid] = local;
    __syncthreads();
    for (int off = 128; off > 0; off >>= 1) {
        if (tid < off && shmax[tid + off] > shmax[tid]) shmax[tid] = shmax[tid + off];
        __syncthreads();
    }
    if (tid == 0) Wref[n] = shmax[0];
}

// PTQ1_0 zero-copy: W aliases the checkpoint's 28-byte blocks, so there is no scale
// plane to gather. Only Wref is needed: the row's largest fp16 exponent field, mapped to
// the e8m0 convention the epilogue uses (rf = 2^(Wref[n]-127)). The staged weight is
// round(d / 2^(Wref-127) * 127), so Wref is the row normaliser, not a lossless bound.
// Measured on the real checkpoint a row's d spans at most ~2 binades below its max.
__global__ void radiance_gather_scales_ptq1_kernel(const unsigned char * __restrict__ src,
                                                   int N, int K, unsigned char * __restrict__ Wref) {
    const int n = blockIdx.x;
    const int tid = threadIdx.x;
    const int stride = blockDim.x;
    const int nb = K / 128;
    const unsigned char * __restrict__ row = src + (size_t) n * nb * 28;
    unsigned int lmax = 0;
    for (int b = tid; b < nb; b += stride) {
        const unsigned int h = (unsigned int) row[b * 28 + 26] | ((unsigned int) row[b * 28 + 27] << 8);
        const unsigned int e = (h >> 10) & 0x1Fu;      // fp16 biased exponent field
        if (e > lmax) lmax = e;
    }
    __shared__ unsigned int sh[256];
    sh[tid] = lmax;
    __syncthreads();
    for (int off = 128; off > 0; off >>= 1) {
        if (tid < off && sh[tid + off] > sh[tid]) sh[tid] = sh[tid + off];
        __syncthreads();
    }
    // The row normaliser must be >= every d in the row, and a block with exponent e holds
    // d in [2^e, 2^(e+1)). Taking the next binade above the row max keeps round(d/drow*127)
    // inside [0,127] for every block; using e_max itself would push the largest block to
    // 254 and clamp half its magnitude away. fp16 biased exponent -> unbiased e_max ->
    // e8m0 biased (e_max + 1) + 127 == exp_field + 113.
    if (tid == 0) Wref[n] = (unsigned char) (sh[0] + 113u);
}

void ggml_cuda_radiance_gather_scales_ptq1(const unsigned char * src, int N, int K,
                                           unsigned char * Wref, cudaStream_t stream) {
    radiance_gather_scales_ptq1_kernel<<<dim3((unsigned) N), 256, 0, stream>>>(src, N, K, Wref);
}

void ggml_cuda_radiance_gather_scales_mxfp6(const unsigned char * src, int N, int K,
                                            unsigned char * Wref, cudaStream_t stream) {
    const int nb = (int)(K / QK_MXFP6);
    radiance_gather_scales_mxfp6_kernel<<<N, 256, 0, stream>>>(src, N, nb, Wref);
}

void ggml_cuda_radiance_repack(const void * src_llama, int64_t s01, int N, int K,
                               void * W, void * Ws, void * Wref, cudaStream_t stream) {
    const int nb = K / 32;
    radiance_repack_kernel<<<N, 256, 0, stream>>>((const char *) src_llama, s01, nb,
        (unsigned char *) W, (unsigned char *) Ws, (unsigned char *) Wref);
}

// persistent activation buffers so a skipped (hash-equal) row still sees its previous q bytes.
struct ggml_rad_act_buf {
    unsigned char * q;
    float * scale;
    uint2 * hashes;
    int64_t M, K;
};
// one buffer set per (device, K): hashes must never leak across tensors with different row
// widths, and on multi-GPU (tensor split) a K-keyed cache hands device 0's allocation to the
// kernel running on device 1 -> illegal memory access. current device is set by the caller.
static std::map<std::pair<int, int64_t>, ggml_rad_act_buf> ggml_rad_act_bufs;
static std::map<std::pair<int, int64_t>, ggml_rad_act_buf> ggml_rad_act_bufs_i8;
static ggml_rad_act_buf ggml_rad_act_get(int64_t M, int64_t K, cudaStream_t stream) {
    const int dev = ggml_cuda_get_device();
    auto key = std::make_pair(dev, K);
    auto it = ggml_rad_act_bufs.find(key);
    if (it == ggml_rad_act_bufs.end() || it->second.M < M) {
        // capture-safe: growing the scratch issues cudaMalloc/MemsetAsync, illegal in a capture.
        // callers must treat the empty set as "decline" and fall back to the non-radiance path.
        cudaStreamCaptureStatus cs = cudaStreamCaptureStatusNone;
        if (cudaStreamIsCapturing(stream, &cs) == cudaSuccess && cs != cudaStreamCaptureStatusNone) {
            return { nullptr, nullptr, nullptr, 0, 0 };
        }
        if (it != ggml_rad_act_bufs.end()) {  // growing: release the old set on its own device
            cudaFree(it->second.q);
            cudaFree(it->second.scale);
            cudaFree(it->second.hashes);
            ggml_rad_act_bufs.erase(it);
        }
        ggml_rad_act_buf b = {nullptr, nullptr, nullptr, 0, 0};
        const int64_t Mcap = std::max<int64_t>(M, 4096);
        cudaMalloc((void **)&b.q,   (size_t)((Mcap + 15) & ~15) * K);
        cudaMalloc((void **)&b.scale, (size_t)Mcap * 4);
        cudaMalloc((void **)&b.hashes, (size_t)Mcap * sizeof(uint2));
        cudaMemsetAsync(b.hashes, 0xFF, (size_t)Mcap * sizeof(uint2), stream); // force first requant
        b.M = Mcap; b.K = K;
        it = ggml_rad_act_bufs.emplace(key, b).first;
    }
    return it->second;
}

// int8 activation quantizer for the PTQ1_0 path. Same shape as quantize_tokens_fp8
// (row-fingerprint memo, fragment-tiled output) but saturates to [-127,127] and stores
// the scale as amax/127 so the epilogue's As[m] keeps the same meaning.
//
// The fingerprint is seeded differently from the fp8 kernel AND the buffer set is
// separate, so an fp8-quantised row can never be mistaken for an int8 one.
template <int NIT>
__global__ void quantize_tokens_i8(const float * __restrict__ x, int64_t sx, int64_t K,
                                   signed char * __restrict__ q, float * __restrict__ scale,
                                   uint2 * __restrict__ row_hash) {
    const int64_t row = blockIdx.y;
    const int tid = threadIdx.x;
    const int nthreads = blockDim.x;
    const float * xr = x + row * sx;
    __shared__ uint2 h;
    __shared__ int skip;
    if (tid < 64) {
        uint32_t h0 = 0x85ebca6bu, h1 = 0xc2b2ae35u;   // different seed than the fp8 kernel
        for (int i = tid; i < 128; i += 64) {
            const uint32_t b = __float_as_uint(xr[(int64_t)(i * 37) % K]);
            h0 = (h0 ^ b) * 0x85ebca6bu; const uint32_t t1 = h1 ^ (b + 0x9e3779b9u + (h0 << 6) + (h0 >> 2)); h1 = ((t1 << 13) | (t1 >> 19)) * 0xc2b2ae35u;
        }
        h.x = h0; h.y = h1;
    }
    __syncthreads();
    if (tid == 0) {
        const uint2 old = row_hash[row];
        skip = (old.x == h.x && old.y == h.y);
    }
    __syncthreads();
    if (skip) {
        return;
    }
    if (tid == 0) {
        row_hash[row] = h;
    }
    float4 v[NIT > 0 ? NIT : 1];
    float local = 0.0f;
    if (NIT > 0) {
#pragma unroll
        for (int i = 0; i < NIT; ++i) {
            const int64_t k = (int64_t)(tid + i * nthreads) * 4;
            if (k < K) {
                v[i] = *(const float4 *)(xr + k);
                local = fmaxf(local, fmaxf(fmaxf(fabsf(v[i].x), fabsf(v[i].y)), fmaxf(fabsf(v[i].z), fabsf(v[i].w))));
            }
        }
    } else {
        for (int64_t k = tid * 4; k < K; k += (int64_t)nthreads * 4) {
            const float4 t = *(const float4 *)(xr + k);
            local = fmaxf(local, fmaxf(fmaxf(fabsf(t.x), fabsf(t.y)), fmaxf(fabsf(t.z), fabsf(t.w))));
        }
    }
    __shared__ float sh[32];
    for (int off = 16; off > 0; off >>= 1)
        local = fmaxf(local, __shfl_xor(local, off));
    if ((tid & 31) == 0) sh[tid >> 5] = local;
    __syncthreads();
    if (tid == 0) {
        float amax = 0.0f;
        for (int i = 0; i < nthreads / 32; ++i) amax = fmaxf(amax, sh[i]);
        scale[row] = amax > 0.0f ? amax / 127.0f : 0.0f;
    }
    __syncthreads();
    const float d = scale[row];
    const float dinv = d > 0.0f ? 1.0f / d : 0.0f;
    const int64_t Kt = K >> 4;
    const int64_t frag_row = (row >> 4) * Kt;
    const int lrow = (int) (row & 15);
    // Four int8 at a time, same fragment-tiled position formula as the fp8 kernel.
#define PTQ1_Q4(a, b, c, e_) do {                                                  \
        const int qa = __float2int_rn((a) * dinv);                                 \
        const int qb = __float2int_rn((b) * dinv);                                 \
        const int qc = __float2int_rn((c) * dinv);                                 \
        const int qe = __float2int_rn((e_) * dinv);                                \
        const int ca = qa < -127 ? -127 : (qa > 127 ? 127 : qa);                   \
        const int cb = qb < -127 ? -127 : (qb > 127 ? 127 : qb);                   \
        const int cc = qc < -127 ? -127 : (qc > 127 ? 127 : qc);                   \
        const int ce = qe < -127 ? -127 : (qe > 127 ? 127 : qe);                   \
        *(uint32_t *) (q + pos) = (uint32_t)(ca & 0xFF) | ((uint32_t)(cb & 0xFF) << 8) \
                                | ((uint32_t)(cc & 0xFF) << 16) | ((uint32_t)(ce & 0xFF) << 24); \
    } while (0)
    if (NIT > 0) {
#pragma unroll
        for (int i = 0; i < NIT; ++i) {
            const int64_t k = (int64_t)(tid + i * nthreads) * 4;
            if (k < K) {
                const int64_t pos = (frag_row + (k >> 4)) * 256 + (lrow + 16 * ((int) ((k >> 3) & 1))) * 8 + (k & 7);
                PTQ1_Q4(v[i].x, v[i].y, v[i].z, v[i].w);
            }
        }
    } else {
        for (int64_t k = tid * 4; k < K; k += (int64_t)nthreads * 4) {
            const float4 t = *(const float4 *)(xr + k);
            const int64_t pos = (frag_row + (k >> 4)) * 256 + (lrow + 16 * ((int) ((k >> 3) & 1))) * 8 + (k & 7);
            PTQ1_Q4(t.x, t.y, t.z, t.w);
        }
    }
#undef PTQ1_Q4
}

void ggml_cuda_radiance_quantize_tokens_i8(const float * x, int64_t sx, int64_t K, int64_t M,
                                           signed char ** q, float ** scale, cudaStream_t stream) {
    const int dev = ggml_cuda_get_device();
    auto key = std::make_pair(dev, K);
    auto it = ggml_rad_act_bufs_i8.find(key);
    if (it == ggml_rad_act_bufs_i8.end() || it->second.M < M) {
        cudaStreamCaptureStatus cs = cudaStreamCaptureStatusNone;
        if (cudaStreamIsCapturing(stream, &cs) == cudaSuccess && cs != cudaStreamCaptureStatusNone) {
            *q = nullptr;
            return;
        }
        if (it != ggml_rad_act_bufs_i8.end()) {
            cudaFree(it->second.q);
            cudaFree(it->second.scale);
            cudaFree(it->second.hashes);
            ggml_rad_act_bufs_i8.erase(it);
        }
        ggml_rad_act_buf b = {nullptr, nullptr, nullptr, 0, 0};
        const int64_t Mcap = std::max<int64_t>(M, 4096);
        cudaMalloc((void **) &b.q, (size_t) ((Mcap + 15) & ~15) * K);
        cudaMalloc((void **) &b.scale, (size_t) Mcap * 4);
        cudaMalloc((void **) &b.hashes, (size_t) Mcap * sizeof(uint2));
        cudaMemsetAsync(b.hashes, 0xFF, (size_t) Mcap * sizeof(uint2), stream);
        b.M = Mcap; b.K = K;
        it = ggml_rad_act_bufs_i8.emplace(key, b).first;
    }
    ggml_rad_act_buf & b = it->second;
    const dim3 grid(1, (unsigned) M);
    constexpr int QNTH = 256, QNIT = 5;
    if (K <= (int64_t) QNTH * 4 * QNIT) {
        quantize_tokens_i8<QNIT><<<grid, QNTH, 0, stream>>>(x, sx, K, (signed char *) b.q, b.scale, b.hashes);
    } else {
        quantize_tokens_i8<0><<<grid, QNTH, 0, stream>>>(x, sx, K, (signed char *) b.q, b.scale, b.hashes);
    }
    *q = (signed char *) b.q;
    *scale = b.scale;
}

void ggml_cuda_radiance_quantize_tokens(const float * x, int64_t sx, int64_t K, int64_t M,
                                        unsigned char ** q, float ** scale, cudaStream_t stream) {
    ggml_rad_act_buf b = ggml_rad_act_get(M, K, stream);
    if (b.q == nullptr) {
        *q = nullptr;   // declined under capture; caller falls back to MMQ
        return;
    }
    const dim3 grid(1, (unsigned) M);
    constexpr int QNTH = 256, QNIT = 5;
    if (K <= (int64_t) QNTH * 4 * QNIT) {
        quantize_tokens_fp8<QNIT><<<grid, QNTH, 0, stream>>>(x, sx, K, b.q, b.scale, b.hashes);
    } else {
        quantize_tokens_fp8<0><<<grid, QNTH, 0, stream>>>(x, sx, K, b.q, b.scale, b.hashes);
    }
    *q = b.q; *scale = b.scale;
}

static void launch_at_f32(uintptr_t at, uintptr_t w, uintptr_t ws, uintptr_t wref, uintptr_t as,
                          uintptr_t cf, int M, int N, int K, uintptr_t stream, bool radsc,
                          bool e6pack = false, bool ptq1 = false) {
    // TN=4 uses a 8x1 wave tile: 512 wide in M, 64 wide in N. This trades A rereads for W
    // rereads, which pays here because A is small and stays in cache while W streams from
    // DRAM: at M=2048/N=6144/K=5120 the A stream grows 503 -> 1007 MB (still well inside the
    // 64 MB MALL, and it was already re-read 48 times) while the W stream halves 126 -> 63 MB.
    // Measured over a pp2048 pass, interleaved A/B, 3 rounds, non-overlapping: 817.1/818.5/
    // 820.7 -> 783.7/787.3/788.8 ms (-4.0%), and +2.0% end to end on pp2048.
    // TN=2 keeps the 4x2 default: it has few enough blocks that the wider M tile starves it.
    // E6PACK keeps LBK=64: its MXFP6 group load regressed +6% at 128.
    constexpr int TWM4 = 8, TWN4 = 1;
    constexpr int B4_ = TWN4 * 4 * 16;
    constexpr int BMF4_ = TWM4 * TM * 16;
    static_assert(TWM4 * TWN4 * 32 == NTHREADS, "fblock below assumes the TN=4 tile keeps NWAVE waves");
    dim3 fblock(NTHREADS);
    if (M >= 2048 && N >= B4_) {
        dim3 fgrid((N + B4_ - 1) / B4_, (M + BMF4_ - 1) / BMF4_);
#define RAD_LAUNCH_AT4(RS_, E6P_, P1_)   hipLaunchKernelGGL((radiance_mxfp4_fp8_gemm_atiled<4, false, (E6P_ ? 64 : 128), false, true, RS_, E6P_, TWM4, TWN4, P1_>), fgrid, fblock, 0, (hipStream_t)stream, (const unsigned char *)at, (const unsigned char *)w, (const unsigned char *)ws, (const unsigned char *)wref, (const float *)as, (__bf16 *)nullptr, M, N, K, 1 << 30, 1 << 30, 0, (const unsigned char *)nullptr, (float *)cf)
        if (ptq1) RAD_LAUNCH_AT4(false, false, true); else if (e6pack) RAD_LAUNCH_AT4(false, true, false); else if (radsc) RAD_LAUNCH_AT4(true, false, false); else RAD_LAUNCH_AT4(false, false, false);
#undef RAD_LAUNCH_AT4
    } else {
        constexpr int B_ = BNF_OF(2);
        dim3 fgrid((N + B_ - 1) / B_, (M + BMF - 1) / BMF);
#define RAD_LAUNCH_AT2(RS_, E6P_, P1_)   hipLaunchKernelGGL((radiance_mxfp4_fp8_gemm_atiled<2, false, 128, false, true, RS_, E6P_, WM, WN, P1_>), fgrid, fblock, 0, (hipStream_t)stream, (const unsigned char *)at, (const unsigned char *)w, (const unsigned char *)ws, (const unsigned char *)wref, (const float *)as, (__bf16 *)nullptr, M, N, K, 1 << 30, 1 << 30, 0, (const unsigned char *)nullptr, (float *)cf)
        if (ptq1) RAD_LAUNCH_AT2(false, false, true); else if (e6pack) RAD_LAUNCH_AT2(false, true, false); else if (radsc) RAD_LAUNCH_AT2(true, false, false); else RAD_LAUNCH_AT2(false, false, false);
#undef RAD_LAUNCH_AT2
    }
}

void ggml_cuda_radiance_gemm_f32(const void * a_q, const void * w, const void * ws, const void * wref,
                                 const float * as, float * c, int M, int N, int K, cudaStream_t stream,
                                 bool radsc, bool e6pack, bool ptq1) {
    launch_at_f32((uintptr_t) a_q, (uintptr_t) w, (uintptr_t) ws, (uintptr_t) wref,
                  (uintptr_t) as, (uintptr_t) c, M, N, K, (uintptr_t) stream, radsc, e6pack, ptq1);
}

bool ggml_cuda_radiance_supported(int cc, ggml_type type, int64_t ne00, int64_t ne01,
                                  int64_t ne11, int64_t ne10, bool contiguous_dst) {
    // RDNA4-only W8A8 fp8 fast path for the MXFP4 weight prefill GEMM, aligned to radiance kernel limits:
    //   M(tokens) >= 256 (decode stays on MMQ/MMVQ), K % 64 == 0 (BK=64), N % 16 == 0 (n-tile).
    //   Env GGML_RAD_PREFILL_MIN_M overrides the threshold for tuning.
    // Both MXFP4 types are zero-copy below this threshold: M <= 4 on the MMVQ plane vec_dot,
    // M in [5,255] on the MMQ RAD tile loader, so neither pays an unrad rebuild.
    const int64_t min_m = getenv("GGML_RAD_PREFILL_MIN_M") ? atoll(getenv("GGML_RAD_PREFILL_MIN_M")) : 256;
    // MXFP6 (type 45) joins here under E6PACK: its packed checkpoint bytes are read in place, so
    // it is zero-copy exactly like the two MXFP4 types and needs no new GGUF type or repack.
    // GGML_RAD_MXFP6_DISABLE isolates the MXFP6 contribution in an A/B (GGML_RAD_DISABLE would
    // also drop MXFP4, so the two effects could not be told apart).
    const bool mxfp6_ok = type != GGML_TYPE_MXFP6 || getenv("GGML_RAD_MXFP6_DISABLE") == nullptr;
    const bool type_ok = (type == GGML_TYPE_MXFP4 || type == GGML_TYPE_MXFP4_RAD || type == GGML_TYPE_MXFP6) && mxfp6_ok;
    // PTQ1_0 rides the same WMMA shape with an int8 datapath: its trits become row-
    // normalised int8 weights and the activations are int8. Needs K % 128 == 0 because
    // one 128-element block is the unit of both the trit packing and the scale.
    const bool ptq1_ok = type == GGML_TYPE_PTQ1_0 && ne00 % 128 == 0;
    return amd_wmma_available(cc) && GGML_CUDA_CC_IS_RDNA4(cc) && (type_ok || ptq1_ok) &&
        ne11 >= min_m && ne00 % 64 == 0 && ne10 == ne00 && (ne01 % 16 == 0) && contiguous_dst;
}


// ===================== fused swiglu + per-token fp8 quant (B1) =====================

// The GLU node's y feeds one type-39 down GEMM whose quantize pass would re-read y. Producing
// q here saves that read plus one kernel launch. y is ALWAYS written (the plain down path and
// any non-fast consumer still need it); only the e4m3 pass is memoized: the (gate, up) row
// signature lets a repeated call skip requantizing byte-identical rows.
__global__ void swiglu_quant_fused_kernel(
        const float * __restrict__ gate, const float * __restrict__ up,
        float * __restrict__ y, unsigned char * __restrict__ q,
        float * __restrict__ scale, uint2 * __restrict__ row_hash,
        int64_t N, int64_t M, int64_t gs, int64_t us) {
    const int64_t row = blockIdx.y;
    const int tid = threadIdx.x;
    const int nthreads = blockDim.x;
    // gs/us are the gate/up row strides in floats. A fused gate_up GEMM hands both out as views
    // of one [2N, M] result, so the stride is 2N and up starts N floats into the row. Two separate
    // gate/up GEMMs give stride N each. y stays the GLU output, always N wide.
    const float * gr = gate + row * gs;
    const float * ur = up + row * us;
    float * yr = y + row * N;

    // (gate, up) row signature: 64 strided samples each, one thread per sample pair.
    __shared__ uint2 h;
    __shared__ int skip;
    __shared__ float sh[8];
    if (tid < 64) {
        uint32_t h0 = 0x9e3779b9u ^ (uint32_t)N, h1 = 0x27d4eb2fu ^ (uint32_t)(N >> 16);
        for (int i = tid; i < 64; i += 64) {
            const uint32_t bg = __float_as_uint(gr[i]);
            const uint32_t bu = __float_as_uint(ur[(int64_t)(i + 32) * 37 % N]);
            h0 = (h0 ^ bg) * 0x85ebca6bu;
            const uint32_t t1 = h1 ^ (bu + 0x9e3779b9u + (h0 << 6) + (h0 >> 2));
            h1 = ((t1 << 13) | (t1 >> 19)) * 0xc2b2ae35u;
        }
        h.x = h0; h.y = h1;
    }
    __syncthreads();
    if (tid == 0) {
        const uint2 old = row_hash[row];
        skip = (old.x == h.x && old.y == h.y);
        row_hash[row] = h;
    }
    __syncthreads();

    // pass 1: y = silu(gate) * up, always; amax over y for the quant pass.
    // same expression order as unary_gated_op_kernel<op_silu>: (g / (1 + expf(-g))) * u
    float local = 0.0f;
    for (int64_t k = tid * 4; k < N; k += (int64_t)nthreads * 4) {
        const float4 gv = *(const float4 *)(gr + k);
        const float4 uv = *(const float4 *)(ur + k);
        float4 yv;
        yv.x = (gv.x / (1.0f + expf(-gv.x))) * uv.x;
        yv.y = (gv.y / (1.0f + expf(-gv.y))) * uv.y;
        yv.z = (gv.z / (1.0f + expf(-gv.z))) * uv.z;
        yv.w = (gv.w / (1.0f + expf(-gv.w))) * uv.w;
        *(float4 *)(yr + k) = yv;
        local = fmaxf(local, fmaxf(fmaxf(fabsf(yv.x), fabsf(yv.y)), fmaxf(fabsf(yv.z), fabsf(yv.w))));
    }
    if (skip) {
        return;
    }
    for (int off = 16; off > 0; off >>= 1)
        local = fmaxf(local, __shfl_xor(local, off));
    if ((tid & 31) == 0) sh[tid >> 5] = local;
    __syncthreads();
    if (tid == 0) {
        float amax = 0.0f;
        for (int i = 0; i < nthreads / 32; ++i) amax = fmaxf(amax, sh[i]);
        scale[row] = amax > 0.0f ? amax / 448.0f : 0.0f;
    }
    __syncthreads();
    const float d = scale[row];
    const float dinv = d > 0.0f ? 1.0f / d : 0.0f;
    const int64_t Kt = N >> 4;
    const int64_t frag_row = (row >> 4) * Kt;
    const int lrow = (int)(row & 15);
    for (int64_t k = tid * 4; k < N; k += (int64_t)nthreads * 4) {
        const float4 yv = *(const float4 *)(yr + k);
        const uint32_t p0 = ggml_cuda_fp32x2_to_e4m3x2(yv.x * dinv, yv.y * dinv);
        const uint32_t p1 = ggml_cuda_fp32x2_to_e4m3x2(yv.z * dinv, yv.w * dinv);
        const int64_t pos = (frag_row + (k >> 4)) * 256 + (lrow + 16 * ((int)((k >> 3) & 1))) * 8 + (k & 7);
        *(uint32_t *)(q + pos) = (p0 & 0xFFFFu) | ((p1 & 0xFFFFu) << 16);
    }
}

// fused-activation registry: maps a GLU output buffer to its ready-made fragment-tiled q.
// q/scale live in the persistent per-K buffers; entries only ever span one graph execution
// (cleared at graph_compute start), so a stale pointer cannot be consumed.
struct ggml_rad_fused_act {
    unsigned char * q;
    float * scale;
    int64_t K;
    // A one-shot entry is erased by the first lookup. The map is keyed by a data pointer and the
    // graph allocator reuses addresses across layers, so a producer whose activation width repeats
    // every layer (the norm output, K = n_embd) could otherwise hand a later layer the previous
    // layer's q. Producers with exactly one consumer set this.
    bool once;
};
static std::unordered_map<const void *, ggml_rad_fused_act> ggml_rad_fused_acts;

void ggml_rad_fused_acts_reset(void) {
    ggml_rad_fused_acts.clear();
}

bool ggml_rad_fused_act_lookup(const void * act, int64_t K,
                               const unsigned char ** q, const float ** scale) {
    auto it = ggml_rad_fused_acts.find(act);
    if (it == ggml_rad_fused_acts.end() || it->second.K != K) {
        return false;
    }
    *q = it->second.q;
    *scale = it->second.scale;
    if (it->second.once) {
        ggml_rad_fused_acts.erase(it);
    }
    return true;
}

// ===================== fused add + RMS norm + weight mul + per-token fp8 (A2) ===================
//
// The post-attention sequence ADD -> RMS_NORM -> MUL, plus the per-token e4m3 that the following
// radiance GEMM would otherwise build in its own pass. Saves the quantize kernel's read of the f32
// norm and one launch. The residual (add->data) and the f32 norm (mul->data) are still written:
// other consumers read them.
//
// The reduction runs exactly like norm.cu's add_rms_norm_f32<block_size>: one strided scalar
// accumulation per thread, then block_reduce<SUM>. The norm values therefore stay bit-identical to
// the unfused kernel, and the amax below sees the same f32 values the standalone quantize pass
// would read back, in the same expression order (rscale * residual * weight).
template <int NTH, int NIT>
__global__ void add_rms_quant_fused_kernel(
        const float * __restrict__ a, const float * __restrict__ b,
        const float * __restrict__ weight, float * __restrict__ residual,
        float * __restrict__ dst, unsigned char * __restrict__ q,
        float * __restrict__ scale, uint2 * __restrict__ hashes, int ncols, float eps) {
    const int64_t row = blockIdx.x;
    const int tid = threadIdx.x;
    const float * ar = a + row * ncols;
    const float * br = b + row * ncols;
    float * rr = residual + row * ncols;
    float * dr = dst + row * ncols;

    __shared__ float lds[NTH / 32];
    __shared__ float lds2[NTH / 32];

    float tmp = 0.0f;
#pragma unroll
    for (int i = 0; i < NIT; ++i) {
        const int col = tid + i * NTH;
        if (col < ncols) {
            const float xi = __fadd_rn(ar[col], br[col]);
            rr[col] = xi;
            tmp += xi * xi;
        }
    }
    tmp = block_reduce<block_reduce_method::SUM, NTH>(tmp, lds);
    const float rscale = rsqrtf(tmp / ncols + eps);

    float amax = 0.0f;
#pragma unroll
    for (int i = 0; i < NIT; ++i) {
        const int col = tid + i * NTH;
        if (col < ncols) {
            const float v = rscale * rr[col] * weight[col];
            dr[col] = v;
            amax = fmaxf(amax, fabsf(v));
        }
    }
    amax = block_reduce<block_reduce_method::MAX, NTH>(amax, lds2);
    const float d = amax > 0.0f ? amax / 448.0f : 0.0f;
    if (tid == 0) {
        scale[row] = d;
        // q is written unconditionally, so retire this row's memo signature. A later
        // quantize_tokens_fp8 call shares this (device, K) buffer; if it saw the previous
        // execution's signature it would take the skip branch and leave these bytes in place for a
        // different tensor. ~0 can never equal a real signature, so it always recomputes.
        hashes[row] = make_uint2(0xFFFFFFFFu, 0xFFFFFFFFu);
    }
    const float dinv = d > 0.0f ? 1.0f / d : 0.0f;
    // Fragment-tiled layout, same position formula as quantize_tokens_fp8: a 16m x 16k fragment is
    // 256 contiguous bytes and k..k+3 never straddles an 8-byte group, so one u32 store fits.
    const int64_t Kt = ncols >> 4;
    const int64_t frag_row = (row >> 4) * Kt;
    const int lrow = (int) (row & 15);
    for (int64_t k = tid * 4; k < ncols; k += (int64_t) NTH * 4) {
        const float4 nv = *(const float4 *) (dr + k);
        const uint32_t p0 = ggml_cuda_fp32x2_to_e4m3x2(nv.x * dinv, nv.y * dinv);
        const uint32_t p1 = ggml_cuda_fp32x2_to_e4m3x2(nv.z * dinv, nv.w * dinv);
        const int64_t pos = (frag_row + (k >> 4)) * 256 + (lrow + 16 * ((int) ((k >> 3) & 1))) * 8 + (k & 7);
        *(uint32_t *) (q + pos) = (p0 & 0xFFFFu) | ((p1 & 0xFFFFu) << 16);
    }
}

#define RAD_A2_DISPATCH(NTH_)                                                                     \
    do {                                                                                          \
        const int nit = (int) ((ncols + NTH_ - 1) / NTH_);                                         \
        switch (nit) {                                                                             \
            case 1: add_rms_quant_fused_kernel<NTH_, 1><<<grid, NTH_, 0, stream>>>(                \
                        a, b, weight, residual, norm, bb.q, bb.scale, bb.hashes, (int) ncols, eps); break; \
            case 2: add_rms_quant_fused_kernel<NTH_, 2><<<grid, NTH_, 0, stream>>>(                \
                        a, b, weight, residual, norm, bb.q, bb.scale, bb.hashes, (int) ncols, eps); break; \
            case 3: add_rms_quant_fused_kernel<NTH_, 3><<<grid, NTH_, 0, stream>>>(                \
                        a, b, weight, residual, norm, bb.q, bb.scale, bb.hashes, (int) ncols, eps); break; \
            case 4: add_rms_quant_fused_kernel<NTH_, 4><<<grid, NTH_, 0, stream>>>(                \
                        a, b, weight, residual, norm, bb.q, bb.scale, bb.hashes, (int) ncols, eps); break; \
            case 5: add_rms_quant_fused_kernel<NTH_, 5><<<grid, NTH_, 0, stream>>>(                \
                        a, b, weight, residual, norm, bb.q, bb.scale, bb.hashes, (int) ncols, eps); break; \
            default: return false;                                                                 \
        }                                                                                          \
    } while (0)

// Runs the fused kernel and registers q under act_key. Returns false when the shape is outside the
// instantiated set or the scratch is unavailable, in which case the caller keeps the unfused path.
bool ggml_cuda_radiance_add_rms_norm_quant(const float * a, const float * b, const float * weight,
                                           float * residual, float * norm, const void * act_key,
                                           int64_t ncols, int64_t nrows, float eps,
                                           cudaStream_t stream) {
    if (getenv("GGML_RAD_DISABLE") || getenv("GGML_RAD_FUSE_OFF") || getenv("GGML_RAD_NORM_QUANT_OFF")) {
        return false;
    }
    if (ncols <= 0 || nrows <= 0 || (ncols % 16) != 0 || ncols > 5120) {
        return false;
    }
    ggml_rad_act_buf bb = ggml_rad_act_get(nrows, ncols, stream);
    if (bb.q == nullptr) {
        return false;   // declined under capture
    }
    const dim3 grid((unsigned) nrows);
    if (ncols < 1024) {
        RAD_A2_DISPATCH(256);
    } else {
        RAD_A2_DISPATCH(1024);
    }
#undef RAD_A2_DISPATCH
    ggml_rad_fused_acts[act_key] = { bb.q, bb.scale, ncols, true };
    return true;
}

// true when dst was handled (y written, q produced and registered).
bool ggml_cuda_try_swiglu_quant_fused(ggml_backend_cuda_context & ctx, const ggml_tensor * dst) {
    if (getenv("GGML_RAD_DISABLE") || getenv("GGML_RAD_FUSE_OFF")) {
        return false;
    }
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    if (!GGML_CUDA_CC_IS_RDNA4(cc) || !amd_wmma_available(cc)) {
        return false;
    }
    const ggml_tensor * gate = dst->src[0];
    const ggml_tensor * up   = dst->src[1];
    if (!gate || !up || dst->type != GGML_TYPE_F32 ||
        gate->type != GGML_TYPE_F32 || up->type != GGML_TYPE_F32) {
        return false;
    }
    // gate/up are either two separate GEMM results or two views of one fused gate_up GEMM result
    // (Qwen3.5 FFN). Walk through the view so a fused gate_up is accepted too. The kernel only
    // needs each operand's row stride, since a view's data already carries its own offset.
    const auto gemm_src = [](const ggml_tensor * t) -> const ggml_tensor * {
        if (t->op == GGML_OP_MUL_MAT) {
            return t;
        }
        if (t->op == GGML_OP_VIEW && t->view_src != nullptr && t->view_src->op == GGML_OP_MUL_MAT) {
            return t->view_src;
        }
        return nullptr;
    };
    const ggml_tensor * gsrc = gemm_src(gate);
    const ggml_tensor * usrc = gemm_src(up);
    if (!gsrc || !usrc || !gsrc->src[0] || !usrc->src[0]) {
        return false;
    }
    // the weights must be a radiance type, else the downstream down-GEMM cannot consume q.
    const auto rad_type = [](ggml_type t) {
        return t == GGML_TYPE_MXFP4 || t == GGML_TYPE_MXFP4_RAD;
    };
    if (!rad_type(gsrc->src[0]->type) || !rad_type(usrc->src[0]->type)) {
        return false;
    }
    const int64_t N = dst->ne[0];
    const int64_t M = dst->ne[1];
    const int64_t min_m = getenv("GGML_RAD_PREFILL_MIN_M") ? atoll(getenv("GGML_RAD_PREFILL_MIN_M")) : 256;
    if (N % 64 != 0 || M < min_m || dst->ne[2] != 1) {
        return false;
    }
    if (gate->ne[0] != N || up->ne[0] != N || gate->ne[1] != M || up->ne[1] != M) {
        return false;
    }
    if (dst->nb[1] != N * (int64_t) sizeof(float)) {
        return false;
    }
    const int64_t gs = gate->nb[1] / (int64_t) sizeof(float);
    const int64_t us = up->nb[1] / (int64_t) sizeof(float);
    if (gs < N || us < N) {
        return false;
    }
    ggml_rad_act_buf b = ggml_rad_act_get(M, N, ctx.stream());
    if (b.q == nullptr) {
        return false;
    }
    const dim3 grid(1, (unsigned) M);
    swiglu_quant_fused_kernel<<<grid, 256, 0, ctx.stream()>>>(
        (const float *) gate->data, (const float *) up->data, (float *) dst->data,
        b.q, b.scale, b.hashes, N, M, gs, us);
    ggml_rad_fused_acts[dst->data] = { b.q, b.scale, N, false };
    return true;
}
