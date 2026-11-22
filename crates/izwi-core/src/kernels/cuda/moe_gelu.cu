#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <math.h>
#include <stdint.h>

// Softmax-routed MoE with BF16 experts and GELU-tanh gating
// (`kernels::moe_gelu`; Gemma 4's experts). Activations and accumulation are
// F32. One warp per output channel; weights are read 8 BF16 per lane (one
// 16-byte load), sweeps of 256 values, then a butterfly. Every pair's dot
// products are summed in the same order whichever pairs share a call, so a
// row's output does not depend on the other rows of the call.
//
// Routing is F32 [tokens, 2 * slots]: `slots` expert ids (exact small
// integers), then their weights.

__device__ __forceinline__ float mgl_warp_sum(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  return v;
}

__device__ __forceinline__ bool mgl_valid(float id, int experts) {
  return id >= 0.0f && (int)id < experts;
}

__device__ __forceinline__ float mgl_bf16(unsigned short bits) {
  return __uint_as_float(((unsigned)bits) << 16);
}

// Eight BF16 weights (one uint4) times eight F32 activations.
__device__ __forceinline__ float mgl_dot8(uint4 q, const float* __restrict__ x) {
  const float4 a = *reinterpret_cast<const float4*>(x);
  const float4 b = *reinterpret_cast<const float4*>(x + 4);
  float acc = mgl_bf16(q.x & 0xFFFFu) * a.x;
  acc = fmaf(mgl_bf16(q.x >> 16), a.y, acc);
  acc = fmaf(mgl_bf16(q.y & 0xFFFFu), a.z, acc);
  acc = fmaf(mgl_bf16(q.y >> 16), a.w, acc);
  acc = fmaf(mgl_bf16(q.z & 0xFFFFu), b.x, acc);
  acc = fmaf(mgl_bf16(q.z >> 16), b.y, acc);
  acc = fmaf(mgl_bf16(q.w & 0xFFFFu), b.z, acc);
  return fmaf(mgl_bf16(q.w >> 16), b.w, acc);
}

// A lane's share of row . x over `len` values (len % 8 == 0).
__device__ __forceinline__ float mgl_row_dot(const __nv_bfloat16* __restrict__ row,
                                             const float* __restrict__ x,
                                             int len, int lane) {
  float acc = 0.0f;
  for (int k0 = lane * 8; k0 < len; k0 += 256) {
    acc += mgl_dot8(*reinterpret_cast<const uint4*>(row + k0), x + k0);
  }
  return acc;
}

// mgl_row_dot of two rows over the same x, one load of x for both: each
// row's sum is accumulated in exactly mgl_row_dot's order.
__device__ __forceinline__ float2 mgl_row_dot2(const __nv_bfloat16* __restrict__ a,
                                               const __nv_bfloat16* __restrict__ b,
                                               const float* __restrict__ x, int len, int lane) {
  float acc_a = 0.0f, acc_b = 0.0f;
  for (int k0 = lane * 8; k0 < len; k0 += 256) {
    acc_a += mgl_dot8(*reinterpret_cast<const uint4*>(a + k0), x + k0);
    acc_b += mgl_dot8(*reinterpret_cast<const uint4*>(b + k0), x + k0);
  }
  return make_float2(acc_a, acc_b);
}

__device__ __forceinline__ float mgl_gelu_tanh(float v) {
  const float inner = 0.7978845608028654f * (v + 0.044715f * v * v * v);
  return 0.5f * v * (1.0f + tanhf(inner));
}

// One warp per token: softmax over every expert in F32, the top `slots`
// probabilities (ties to the lower expert), renormalized over the kept ones
// and multiplied by the expert's scale. Grid (tokens), 32 threads;
// experts <= 512, slots <= 16.
extern "C" __global__ void izwi_mgl_route(const float* __restrict__ logits,
                                          const float* __restrict__ scale,
                                          float* __restrict__ routing,
                                          int experts, int slots) {
  const int lane = threadIdx.x;
  const int token = blockIdx.x;
  const float* row = logits + (int64_t)token * experts;
  float v[16];
  float best = -INFINITY;
#pragma unroll
  for (int i = 0; i < 16; ++i) {
    const int e = lane + i * 32;
    v[i] = e < experts ? row[e] : -INFINITY;
    best = fmaxf(best, v[i]);
  }
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) {
    best = fmaxf(best, __shfl_xor_sync(0xffffffffu, best, o));
  }
  float sum = 0.0f;
#pragma unroll
  for (int i = 0; i < 16; ++i) {
    const int e = lane + i * 32;
    v[i] = e < experts ? expf(v[i] - best) : -1.0f;
    if (e < experts) sum += v[i];
  }
  sum = mgl_warp_sum(sum);
  float kept_prob[16];
  int kept_id[16];
  float kept = 0.0f;
  for (int s = 0; s < slots; ++s) {
    // The lane's best remaining expert, then the warp's (lower id on ties).
    float pick = -1.0f;
    int pick_id = 0x7fffffff;
#pragma unroll
    for (int i = 0; i < 16; ++i) {
      const int e = lane + i * 32;
      if (e < experts && (v[i] > pick || (v[i] == pick && e < pick_id))) {
        pick = v[i];
        pick_id = e;
      }
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
      const float other = __shfl_xor_sync(0xffffffffu, pick, o);
      const int other_id = __shfl_xor_sync(0xffffffffu, pick_id, o);
      if (other > pick || (other == pick && other_id < pick_id)) {
        pick = other;
        pick_id = other_id;
      }
    }
    // The owner retires it.
#pragma unroll
    for (int i = 0; i < 16; ++i) {
      if (lane + i * 32 == pick_id) v[i] = -2.0f;
    }
    kept_prob[s] = pick / sum;
    kept_id[s] = pick_id;
    kept += kept_prob[s];
  }
  if (lane == 0) {
    float* out = routing + (int64_t)token * 2 * slots;
    for (int s = 0; s < slots; ++s) {
      out[s] = (float)kept_id[s];
      out[slots + s] = kept_prob[s] / kept * scale[kept_id[s]];
    }
  }
}

// Pairs grouped by expert (offsets[e]..offsets[e + 1] index sorted_pairs);
// one block of 1024 threads, experts <= 512. The order within a group does
// not matter: each pair's results are its own.
extern "C" __global__ void izwi_mgl_group_pairs(
    const float* __restrict__ routing, int* __restrict__ offsets,
    int* __restrict__ sorted_pairs, int tokens, int slots, int experts) {
  __shared__ int counts[512];
  __shared__ int fill[512];
  __shared__ int starts[512];
  const int tid = threadIdx.x;
  const int threads = blockDim.x;
  for (int e = tid; e < experts; e += threads) {
    counts[e] = 0;
    fill[e] = 0;
  }
  __syncthreads();
  const int pairs = tokens * slots;
  for (int p = tid; p < pairs; p += threads) {
    const int token = p / slots;
    const float id = routing[(int64_t)token * 2 * slots + (p - token * slots)];
    if (mgl_valid(id, experts)) atomicAdd(&counts[(int)id], 1);
  }
  __syncthreads();
  if (tid == 0) {
    int running = 0;
    for (int e = 0; e < experts; ++e) {
      starts[e] = running;
      offsets[e] = running;
      running += counts[e];
    }
    offsets[experts] = running;
  }
  __syncthreads();
  for (int p = tid; p < pairs; p += threads) {
    const int token = p / slots;
    const float id = routing[(int64_t)token * 2 * slots + (p - token * slots)];
    if (mgl_valid(id, experts)) {
      const int e = (int)id;
      sorted_pairs[starts[e] + atomicAdd(&fill[e], 1)] = p;
    }
  }
}

// act[pair, n] = gelu_tanh(gate_up[e, n] . x) * (gate_up[e, inter + n] . x)
// for every pair routed to expert e; each weight row is read once per call
// and stays in L1 across the expert's pairs. Grid (experts, inter / 8), 256
// threads.
extern "C" __global__ void izwi_mgl_up_grouped(
    const float* __restrict__ x, const int* __restrict__ offsets,
    const int* __restrict__ sorted_pairs,
    const __nv_bfloat16* __restrict__ gate_up, float* __restrict__ act,
    int slots, int hidden, int inter) {
  const int lane = threadIdx.x & 31;
  const int e = blockIdx.x;
  const int begin = offsets[e];
  const int end = offsets[e + 1];
  const int n = blockIdx.y * 8 + (threadIdx.x >> 5);
  if (begin == end || n >= inter) return;
  const __nv_bfloat16* gate = gate_up + ((int64_t)e * 2 * inter + n) * hidden;
  const __nv_bfloat16* up = gate_up + ((int64_t)e * 2 * inter + inter + n) * hidden;
  for (int i = begin; i < end; ++i) {
    const int pair = sorted_pairs[i];
    const float* xt = x + (int64_t)(pair / slots) * hidden;
    const float2 gu = mgl_row_dot2(gate, up, xt, hidden, lane);
    const float g = mgl_warp_sum(gu.x);
    const float u = mgl_warp_sum(gu.y);
    if (lane == 0) act[(int64_t)pair * inter + n] = mgl_gelu_tanh(g) * u;
  }
}

// partial[pair, h] = down[e, h] . act[pair] for every pair routed to e.
// Each warp takes rows h and h + 8 of its block's 16, one load of act for
// both. Grid (experts, hidden / 16), 256 threads.
extern "C" __global__ void izwi_mgl_down_grouped(
    const float* __restrict__ act, const int* __restrict__ offsets,
    const int* __restrict__ sorted_pairs,
    const __nv_bfloat16* __restrict__ down, float* __restrict__ partial,
    int hidden, int inter) {
  const int lane = threadIdx.x & 31;
  const int e = blockIdx.x;
  const int begin = offsets[e];
  const int end = offsets[e + 1];
  const int h = blockIdx.y * 16 + (threadIdx.x >> 5);
  if (begin == end || h >= hidden) return;
  const bool second = h + 8 < hidden;
  const __nv_bfloat16* row = down + ((int64_t)e * hidden + h) * inter;
  const __nv_bfloat16* row2 = second ? row + 8 * (int64_t)inter : row;
  for (int i = begin; i < end; ++i) {
    const int pair = sorted_pairs[i];
    const float2 p = mgl_row_dot2(row, row2, act + (int64_t)pair * inter, inter, lane);
    const float p0 = mgl_warp_sum(p.x), p1 = mgl_warp_sum(p.y);
    if (lane == 0) {
      partial[(int64_t)pair * hidden + h] = p0;
      if (second) partial[(int64_t)pair * hidden + h + 8] = p1;
    }
  }
}

// y[t, h] = sum_s weight_s * partial[pair_s, h] in slot order;
// grid (hidden / 256, tokens), 256 threads.
extern "C" __global__ void izwi_mgl_combine(
    const float* __restrict__ partial, const float* __restrict__ routing,
    float* __restrict__ y, int slots, int hidden, int experts) {
  const int h = blockIdx.x * blockDim.x + threadIdx.x;
  const int token = blockIdx.y;
  if (h >= hidden) return;
  const float* row = routing + (int64_t)token * 2 * slots;
  float total = 0.0f;
  for (int s = 0; s < slots; ++s) {
    if (!mgl_valid(row[s], experts)) continue;
    total = fmaf(row[slots + s],
                 partial[((int64_t)token * slots + s) * hidden + h], total);
  }
  y[(int64_t)token * hidden + h] = total;
}

// ---------------------------------------------------------------------------
// FP8 experts: E4M3 weights with one F32 scale per output row
// (`kernels::moe_gelu::Fp8Weights`). The same grouping, routing and combine;
// the up and down kernels read half the bytes. Each lane loads 16 weights
// (16 bytes) a sweep of 512 and decodes them by moving bits into FP16; a
// row's dot product is scaled once at the end. Widths must be multiples of
// 16.



// E4M3 -> FP16 by moving bits: seee.emmm becomes the FP16 s0eeee.mmm0000000,
// exactly the E4M3 value times 2^-8 for every finite code (subnormals
// included), so the 2^8 rides in the row scale (`MGL_E4M3_UNSCALE`). Bytes 0
// and 2 of `word` land in one half2, bytes 1 and 3 in the other.
#define MGL_E4M3_UNSCALE 256.0f

__device__ __forceinline__ float2 mgl_e4m3x2_even(unsigned int word) {
  const unsigned int bits = ((word & 0x00800080u) << 8) | ((word & 0x007F007Fu) << 7);
  return __half22float2(*reinterpret_cast<const __half2*>(&bits));
}

__device__ __forceinline__ float2 mgl_e4m3x2_odd(unsigned int word) {
  return mgl_e4m3x2_even(word >> 8);
}

// Four E4M3 bytes of `word` times four F32 activations (unscaled by 2^-8).
__device__ __forceinline__ float mgl_dot4_fp8(unsigned int word, float4 a, float acc) {
  const float2 even = mgl_e4m3x2_even(word), odd = mgl_e4m3x2_odd(word);
  acc = fmaf(even.x, a.x, acc);
  acc = fmaf(odd.x, a.y, acc);
  acc = fmaf(even.y, a.z, acc);
  return fmaf(odd.y, a.w, acc);
}

// A lane's share of an E4M3 row . x over `len` values (len % 16 == 0), in
// units of 2^-8 (callers multiply by scale * MGL_E4M3_UNSCALE).
__device__ __forceinline__ float mgl_row_dot_fp8(const unsigned char* __restrict__ row,
                                                 const float* __restrict__ x, int len,
                                                 int lane) {
  float acc = 0.0f;
  for (int k0 = lane * 16; k0 < len; k0 += 512) {
    const uint4 q = *reinterpret_cast<const uint4*>(row + k0);
    const float4* xv = reinterpret_cast<const float4*>(x + k0);
    acc = mgl_dot4_fp8(q.x, xv[0], acc);
    acc = mgl_dot4_fp8(q.y, xv[1], acc);
    acc = mgl_dot4_fp8(q.z, xv[2], acc);
    acc = mgl_dot4_fp8(q.w, xv[3], acc);
  }
  return acc;
}

// mgl_row_dot_fp8 of two rows over the same x, one load of x for both, each
// row's sum in mgl_row_dot_fp8's order.
__device__ __forceinline__ float2 mgl_row_dot2_fp8(const unsigned char* __restrict__ a,
                                                   const unsigned char* __restrict__ b,
                                                   const float* __restrict__ x, int len,
                                                   int lane) {
  float acc_a = 0.0f, acc_b = 0.0f;
  for (int k0 = lane * 16; k0 < len; k0 += 512) {
    const uint4 qa = *reinterpret_cast<const uint4*>(a + k0);
    const uint4 qb = *reinterpret_cast<const uint4*>(b + k0);
    const float4* xv = reinterpret_cast<const float4*>(x + k0);
    const float4 x0 = xv[0], x1 = xv[1], x2 = xv[2], x3 = xv[3];
    acc_a = mgl_dot4_fp8(qa.x, x0, acc_a);
    acc_a = mgl_dot4_fp8(qa.y, x1, acc_a);
    acc_a = mgl_dot4_fp8(qa.z, x2, acc_a);
    acc_a = mgl_dot4_fp8(qa.w, x3, acc_a);
    acc_b = mgl_dot4_fp8(qb.x, x0, acc_b);
    acc_b = mgl_dot4_fp8(qb.y, x1, acc_b);
    acc_b = mgl_dot4_fp8(qb.z, x2, acc_b);
    acc_b = mgl_dot4_fp8(qb.w, x3, acc_b);
  }
  return make_float2(acc_a, acc_b);
}

// izwi_mgl_up_grouped with E4M3 gate_up [experts, 2 * inter, hidden] and
// scales [experts, 2 * inter].
extern "C" __global__ void izwi_mgl_up_grouped_fp8(
    const float* __restrict__ x, const int* __restrict__ offsets,
    const int* __restrict__ sorted_pairs, const unsigned char* __restrict__ gate_up,
    const float* __restrict__ scale, float* __restrict__ act, int slots, int hidden,
    int inter) {
  const int lane = threadIdx.x & 31;
  const int e = blockIdx.x;
  const int begin = offsets[e];
  const int end = offsets[e + 1];
  const int n = blockIdx.y * 8 + (threadIdx.x >> 5);
  if (begin == end || n >= inter) return;
  const int64_t gate_row = (int64_t)e * 2 * inter + n;
  const int64_t up_row = gate_row + inter;
  const unsigned char* gate = gate_up + gate_row * hidden;
  const unsigned char* up = gate_up + up_row * hidden;
  const float gate_scale = scale[gate_row] * MGL_E4M3_UNSCALE;
  const float up_scale = scale[up_row] * MGL_E4M3_UNSCALE;
  for (int i = begin; i < end; ++i) {
    const int pair = sorted_pairs[i];
    const float* xt = x + (int64_t)(pair / slots) * hidden;
    const float2 gu = mgl_row_dot2_fp8(gate, up, xt, hidden, lane);
    const float g = mgl_warp_sum(gu.x) * gate_scale;
    const float u = mgl_warp_sum(gu.y) * up_scale;
    if (lane == 0) act[(int64_t)pair * inter + n] = mgl_gelu_tanh(g) * u;
  }
}

// izwi_mgl_down_grouped with E4M3 down [experts, hidden, inter] and scales
// [experts, hidden].
extern "C" __global__ void izwi_mgl_down_grouped_fp8(
    const float* __restrict__ act, const int* __restrict__ offsets,
    const int* __restrict__ sorted_pairs, const unsigned char* __restrict__ down,
    const float* __restrict__ scale, float* __restrict__ partial, int hidden, int inter) {
  const int lane = threadIdx.x & 31;
  const int e = blockIdx.x;
  const int begin = offsets[e];
  const int end = offsets[e + 1];
  const int h = blockIdx.y * 16 + (threadIdx.x >> 5);
  if (begin == end || h >= hidden) return;
  const bool second = h + 8 < hidden;
  const int64_t r = (int64_t)e * hidden + h;
  const int64_t r2 = second ? r + 8 : r;
  const float s = scale[r] * MGL_E4M3_UNSCALE, s2 = scale[r2] * MGL_E4M3_UNSCALE;
  for (int i = begin; i < end; ++i) {
    const int pair = sorted_pairs[i];
    const float2 p = mgl_row_dot2_fp8(down + r * inter, down + r2 * inter,
                                      act + (int64_t)pair * inter, inter, lane);
    const float p0 = mgl_warp_sum(p.x) * s, p1 = mgl_warp_sum(p.y) * s2;
    if (lane == 0) {
      partial[(int64_t)pair * hidden + h] = p0;
      if (second) partial[(int64_t)pair * hidden + h + 8] = p1;
    }
  }
}

// Round-to-nearest-even, saturating F32 -> E4M3FN (the inverse of mgl_e4m3
// on its finite values; |v| <= 448 after scaling).
__device__ __forceinline__ unsigned char mgl_to_e4m3(float v) {
  const unsigned char sign = v < 0.0f ? 128 : 0;
  float a = fminf(fabsf(v), 448.0f);
  if (!(a > 0.0f)) return 0;  // zero and NaN encode as +0
  int e;
  frexpf(a, &e);              // a = f * 2^e, f in [0.5, 1)
  int exp = e - 1 + 7;        // biased exponent of 1.m * 2^(e - 1)
  unsigned int code;
  if (exp <= 0) {
    // Subnormal: value = m * 2^-9, m in 0..7.
    const float m = rintf(a * 512.0f);
    code = (unsigned int)m;   // m == 8 rounds up to the smallest normal
  } else {
    const float m = rintf((ldexpf(a, -(e - 1)) - 1.0f) * 8.0f);
    code = ((unsigned int)exp << 3) + (unsigned int)m;  // m == 8 carries
    if (code > 126) code = 126;                          // 448, largest finite
  }
  return sign | (unsigned char)code;
}

// One block per row of w (BF16, `cols` wide): scale[row] = max|w| / 448 (1
// for an all-zero row) and q[row] = E4M3(w / scale).
extern "C" __global__ void izwi_mgl_fp8_quantize_rows(const __nv_bfloat16* __restrict__ w,
                                                      unsigned char* __restrict__ q,
                                                      float* __restrict__ scale, int cols) {
  __shared__ float peaks[32];
  const int64_t row = blockIdx.x;
  const __nv_bfloat16* src = w + row * cols;
  float peak = 0.0f;
  for (int c = threadIdx.x; c < cols; c += blockDim.x)
    peak = fmaxf(peak, fabsf(__bfloat162float(src[c])));
  for (int o = 16; o > 0; o >>= 1) peak = fmaxf(peak, __shfl_xor_sync(0xffffffffu, peak, o));
  if ((threadIdx.x & 31) == 0) peaks[threadIdx.x >> 5] = peak;
  __syncthreads();
  if (threadIdx.x < 32) {
    peak = threadIdx.x < (blockDim.x >> 5) ? peaks[threadIdx.x] : 0.0f;
    for (int o = 16; o > 0; o >>= 1) peak = fmaxf(peak, __shfl_xor_sync(0xffffffffu, peak, o));
    if (threadIdx.x == 0) peaks[0] = peak > 0.0f ? peak / 448.0f : 1.0f;
  }
  __syncthreads();
  const float s = peaks[0];
  if (threadIdx.x == 0) scale[row] = s;
  for (int c = threadIdx.x; c < cols; c += blockDim.x)
    q[row * cols + c] = mgl_to_e4m3(__bfloat162float(src[c]) / s);
}

// out[g] = BF16(E4M3(q[experts[g]]) * scale) for `count` experts of `rows` x
// `cols` (prefill's grouped GEMM reads the used experts in BF16).
extern "C" __global__ void izwi_mgl_fp8_dequant_experts(
    const unsigned char* __restrict__ q, const float* __restrict__ scale,
    const int* __restrict__ experts, __nv_bfloat16* __restrict__ out, int count, int rows,
    int cols) {
  const int64_t per = (int64_t)rows * cols;
  const int64_t total = per * count;
  for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; i < total;
       i += (int64_t)blockDim.x * gridDim.x) {
    const int64_t g = i / per;
    const int64_t within = i - g * per;
    const int64_t r = within / cols;
    const int64_t e = experts[g];
    const unsigned int b = q[e * per + within];
    const unsigned short bits = (unsigned short)(((b & 0x80u) << 8) | ((b & 0x7Fu) << 7));
    out[i] = __float2bfloat16(__half2float(*reinterpret_cast<const __half*>(&bits)) *
                              (scale[e * rows + r] * MGL_E4M3_UNSCALE));
  }
}
