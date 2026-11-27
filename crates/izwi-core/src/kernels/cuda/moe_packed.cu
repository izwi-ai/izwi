#include <math.h>
#include <stdint.h>

// Sigmoid-routed sparse experts whose weights stay in a packed quantized
// form (`kernels::moe_packed`):
//
//   NVFP4: 4-bit E2M1 codes, two per byte (even element in the low nibble),
//          one FP8 E4M3 scale per 16 values along a row, one F32 factor per
//          expert. value = code * scale * factor.
//   MXFP8: FP8 E4M3 values, one scale byte s per 32 values along a row.
//          value = e4m3 * 2^(s - 127).
//
// Activations and every sum are F32. One thread per output value: a routed
// (token, slot) pair's dot products are summed in the same order whichever
// pairs share a call, so a row's output does not depend on the batch.
//
// Routing is F32 [tokens, 2 * slots]: `slots` expert ids ascending (exact
// small integers), then their weights.

#define MPK_MAX_EXPERTS 512
#define MPK_MAX_SLOTS 16

__device__ __forceinline__ float mpk_e4m3(unsigned b) {
  const unsigned e = (b >> 3) & 15u;
  const float m = (float)(b & 7u);
  const float v = e == 0u ? m * 0.001953125f : (1.0f + m * 0.125f) * scalbnf(1.0f, (int)e - 7);
  return (b & 128u) ? -v : v;
}

// The value of a 4-bit E2M1 code: sign, two exponent bits, one mantissa bit.
__device__ __forceinline__ float mpk_e2m1(unsigned c) {
  const unsigned e = (c >> 1) & 3u;
  const float m = (float)(c & 1u);
  const float v = e == 0u ? 0.5f * m : (1.0f + 0.5f * m) * (float)(1u << (e - 1u));
  return (c & 8u) ? -v : v;
}

__device__ __forceinline__ float mpk_softcap(float v, float cap) { return tanhf(v / cap) * cap; }

// row . x for an NVFP4 row of `cols` values (cols % 16 == 0): per 16-value
// block, the code sum times the block scale; the caller applies the factor.
__device__ __forceinline__ float mpk_nvfp4_dot(const unsigned char* __restrict__ codes,
                                               const unsigned char* __restrict__ scales,
                                               const float* __restrict__ x, int cols) {
  float total = 0.0f;
  for (int block = 0; block < cols / 16; ++block) {
    const unsigned char* q = codes + block * 8;
    const float* xb = x + block * 16;
    float dot = 0.0f;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const unsigned byte = q[i];
      dot = fmaf(mpk_e2m1(byte & 15u), xb[2 * i], dot);
      dot = fmaf(mpk_e2m1(byte >> 4), xb[2 * i + 1], dot);
    }
    total = fmaf(dot, mpk_e4m3(scales[block]), total);
  }
  return total;
}

// row . x for an MXFP8 row of `cols` values (cols % 32 == 0).
__device__ __forceinline__ float mpk_mxfp8_dot(const unsigned char* __restrict__ values,
                                               const unsigned char* __restrict__ scales,
                                               const float* __restrict__ x, int cols) {
  float total = 0.0f;
  for (int block = 0; block < cols / 32; ++block) {
    const unsigned char* q = values + block * 32;
    const float* xb = x + block * 32;
    float dot = 0.0f;
#pragma unroll
    for (int i = 0; i < 32; ++i) dot = fmaf(mpk_e4m3(q[i]), xb[i], dot);
    total = fmaf(dot, scalbnf(1.0f, (int)scales[block] - 127), total);
  }
  return total;
}

// One thread per token: sigmoid scores; the `slots` experts with the highest
// score + bias (ties to the lower expert); their raw scores normalised to
// sum to one and multiplied by `route_scale`. Ids are written ascending.
// Grid (ceil(tokens / 128)), 128 threads; experts <= 512, slots <= 16.
extern "C" __global__ void izwi_mpk_route(const float* __restrict__ logits,
                                          const float* __restrict__ bias,
                                          float* __restrict__ routing,
                                          int tokens, int experts, int slots, float route_scale) {
  const int token = blockIdx.x * blockDim.x + threadIdx.x;
  if (token >= tokens) return;
  const float* row = logits + (int64_t)token * experts;
  int ids[MPK_MAX_SLOTS];
  float picked[MPK_MAX_SLOTS];
  for (int s = 0; s < slots; ++s) {
    float best = -INFINITY;
    int best_id = -1;
    float best_score = 0.0f;
    for (int e = 0; e < experts; ++e) {
      bool taken = false;
      for (int p = 0; p < s; ++p) taken = taken || ids[p] == e;
      if (taken) continue;
      const float score = 1.0f / (1.0f + expf(-row[e]));
      const float key = score + bias[e];
      // Ascending scan with a strict comparison: the lower expert wins a tie.
      if (best_id < 0 || key > best) {
        best = key;
        best_id = e;
        best_score = score;
      }
    }
    // Insert ascending by expert id.
    int at = s;
    while (at > 0 && ids[at - 1] > best_id) {
      ids[at] = ids[at - 1];
      picked[at] = picked[at - 1];
      --at;
    }
    ids[at] = best_id;
    picked[at] = best_score;
  }
  float total = 0.0f;
  for (int s = 0; s < slots; ++s) total += picked[s];
  total += 1e-20f;
  float* out = routing + (int64_t)token * 2 * slots;
  for (int s = 0; s < slots; ++s) {
    out[s] = (float)ids[s];
    out[slots + s] = picked[s] / total * route_scale;
  }
}

// act[pair, i] = softcap(gate, gate_cap) * sigmoid(gate) * softcap(up, up_cap)
// with gate = row 2i . x and up = row 2i + 1 . x (stored rows interleave gate
// and up). Grid (pairs, ceil(inter / 128)), 128 threads.
#define MPK_UP(NAME, DOT, CODE_BYTES, BLOCK)                                                    \
  extern "C" __global__ void NAME(const float* __restrict__ x,                                  \
                                  const float* __restrict__ routing,                            \
                                  const unsigned char* __restrict__ codes,                      \
                                  const unsigned char* __restrict__ scales,                     \
                                  const float* __restrict__ factors,                            \
                                  float* __restrict__ act,                                      \
                                  int slots, int hidden, int inter, int experts,                \
                                  float gate_cap, float up_cap) {                               \
    const int i = blockIdx.y * blockDim.x + threadIdx.x;                                        \
    if (i >= inter) return;                                                                     \
    const int pair = blockIdx.x;                                                                \
    const int token = pair / slots;                                                             \
    const int expert = (int)routing[(int64_t)token * 2 * slots + pair % slots];                 \
    float* out = act + (int64_t)pair * inter + i;                                               \
    if (expert < 0 || expert >= experts) {                                                      \
      *out = 0.0f;                                                                              \
      return;                                                                                   \
    }                                                                                           \
    const float* xt = x + (int64_t)token * hidden;                                              \
    const int64_t code_row = (int64_t)(CODE_BYTES);                                             \
    const int64_t scale_row = (int64_t)(hidden / (BLOCK));                                      \
    const int64_t first = ((int64_t)expert * 2 * inter + 2 * i);                                \
    const float factor = factors[expert];                                                       \
    const float gate = DOT(codes + first * code_row, scales + first * scale_row, xt, hidden) *  \
                       factor;                                                                  \
    const float up = DOT(codes + (first + 1) * code_row, scales + (first + 1) * scale_row, xt,  \
                         hidden) *                                                              \
                     factor;                                                                    \
    *out = mpk_softcap(gate, gate_cap) * (1.0f / (1.0f + expf(-gate))) * mpk_softcap(up, up_cap); \
  }

MPK_UP(izwi_mpk_up_nvfp4, mpk_nvfp4_dot, hidden / 2, 16)
MPK_UP(izwi_mpk_up_mxfp8, mpk_mxfp8_dot, hidden, 32)

// partial[pair, h] = down row h . act[pair]. Grid (pairs, ceil(hidden / 128)),
// 128 threads.
#define MPK_DOWN(NAME, DOT, CODE_BYTES, BLOCK)                                                  \
  extern "C" __global__ void NAME(const float* __restrict__ act,                                \
                                  const float* __restrict__ routing,                            \
                                  const unsigned char* __restrict__ codes,                      \
                                  const unsigned char* __restrict__ scales,                     \
                                  const float* __restrict__ factors,                            \
                                  float* __restrict__ partial,                                  \
                                  int slots, int hidden, int inter, int experts) {              \
    const int h = blockIdx.y * blockDim.x + threadIdx.x;                                        \
    if (h >= hidden) return;                                                                    \
    const int pair = blockIdx.x;                                                                \
    const int token = pair / slots;                                                             \
    const int expert = (int)routing[(int64_t)token * 2 * slots + pair % slots];                 \
    float* out = partial + (int64_t)pair * hidden + h;                                          \
    if (expert < 0 || expert >= experts) {                                                      \
      *out = 0.0f;                                                                              \
      return;                                                                                   \
    }                                                                                           \
    const int64_t row = (int64_t)expert * hidden + h;                                           \
    *out = DOT(codes + row * (int64_t)(CODE_BYTES), scales + row * (int64_t)(inter / (BLOCK)),  \
               act + (int64_t)pair * inter, inter) *                                            \
           factors[expert];                                                                     \
  }

MPK_DOWN(izwi_mpk_down_nvfp4, mpk_nvfp4_dot, inter / 2, 16)
MPK_DOWN(izwi_mpk_down_mxfp8, mpk_mxfp8_dot, inter, 32)

// y[token, h] = sum over slots, in slot order, of weight * partial.
// Grid (tokens, ceil(hidden / 128)), 128 threads.
extern "C" __global__ void izwi_mpk_combine(const float* __restrict__ partial,
                                            const float* __restrict__ routing,
                                            float* __restrict__ y, int slots, int hidden) {
  const int h = blockIdx.y * blockDim.x + threadIdx.x;
  if (h >= hidden) return;
  const int token = blockIdx.x;
  const float* route = routing + (int64_t)token * 2 * slots;
  float acc = 0.0f;
  for (int s = 0; s < slots; ++s) {
    acc = fmaf(route[slots + s], partial[((int64_t)token * slots + s) * hidden + h], acc);
  }
  y[(int64_t)token * hidden + h] = acc;
}
