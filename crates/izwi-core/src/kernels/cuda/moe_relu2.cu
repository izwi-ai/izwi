#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <math.h>
#include <stdint.h>

// Sigmoid-routed relu² MoE with one FP8 scale per expert projection
// (`kernels::moe_relu2`; the Metal kernels there are the same arithmetic).
// One warp per output channel; weights are E4M3FN bytes read 16 per lane.
// The per-pair and expert-major forms sum each dot product in the same
// order (lane sweeps of 512 values, then a butterfly), so they agree bit for
// bit and a row's output does not depend on the other rows of the call.

#define MR2_NONE 0xFFFFFFFFu

// E4M3FN byte -> value * 2^-8 through an F16 lane (exact for finite values).
__device__ __forceinline__ float mr2_e4m3(unsigned b) {
  const unsigned short h =
      (unsigned short)(((b & 0x80u) << 8) | ((b & 0x7Fu) << 7));
  return __half2float(__ushort_as_half(h));
}

__device__ __forceinline__ float mr2_dot4(unsigned w, const float* x) {
  float acc = mr2_e4m3(w & 0xFFu) * x[0];
  acc = fmaf(mr2_e4m3((w >> 8) & 0xFFu), x[1], acc);
  acc = fmaf(mr2_e4m3((w >> 16) & 0xFFu), x[2], acc);
  return fmaf(mr2_e4m3(w >> 24), x[3], acc);
}

__device__ __forceinline__ float mr2_dot16(uint4 q, const float* __restrict__ x) {
  float v[16];
#pragma unroll
  for (int i = 0; i < 16; ++i) v[i] = x[i];
  float acc = mr2_dot4(q.x, v);
  acc += mr2_dot4(q.y, v + 4);
  acc += mr2_dot4(q.z, v + 8);
  return acc + mr2_dot4(q.w, v + 12);
}

__device__ __forceinline__ float mr2_warp_sum(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  return v;
}

__device__ __forceinline__ bool mr2_valid(float id, int experts) {
  return id >= 0.0f && (int)id < experts;
}

// logits[t, e] = x[t] . gate[e] in F32; grid (experts / 8, tokens), 256 threads.
extern "C" __global__ void izwi_mr2_router_logits(
    const float* __restrict__ x, const float* __restrict__ gate,
    float* __restrict__ logits, int tokens, int experts, int hidden) {
  const int lane = threadIdx.x & 31;
  const int e = blockIdx.x * 8 + (threadIdx.x >> 5);
  const int t = blockIdx.y;
  if (e >= experts || t >= tokens) return;
  const float* xr = x + (int64_t)t * hidden;
  const float* gr = gate + (int64_t)e * hidden;
  float acc = 0.0f;
  for (int k = lane; k < hidden; k += 32) acc = fmaf(xr[k], gr[k], acc);
  acc = mr2_warp_sum(acc);
  if (lane == 0) logits[(int64_t)t * experts + e] = acc;
}

// One warp per token, four per block. Selection on sigmoid(logit) + bias
// (F32, ties to the lower expert); weights sigmoid / sum(picked) * scale.
extern "C" __global__ void izwi_mr2_route(
    const float* __restrict__ logits, const float* __restrict__ bias,
    float* __restrict__ out, int tokens, int experts, int top_k, float scale) {
  const int lane = threadIdx.x & 31;
  const int token = blockIdx.x * 4 + (threadIdx.x >> 5);
  if (token >= tokens) return;
  const float* row = logits + (int64_t)token * experts;
  float s[16];
  float b[16];
#pragma unroll
  for (int i = 0; i < 16; ++i) {
    const int e = lane + 32 * i;
    if (e < experts) {
      s[i] = 1.0f / (1.0f + expf(-row[e]));
      b[i] = s[i] + bias[e];
    } else {
      s[i] = 0.0f;
      b[i] = -INFINITY;
    }
  }
  float* dst = out + (int64_t)token * 2 * top_k;
  unsigned taken = 0;
  float picked_sum = 0.0f;
  for (int k = 0; k < top_k; ++k) {
    float best = -INFINITY;
    float best_s = 0.0f;
    unsigned best_e = MR2_NONE;
#pragma unroll
    for (int i = 0; i < 16; ++i) {
      const unsigned e = lane + 32 * i;
      const bool open = (int)e < experts && ((taken >> i) & 1u) == 0u;
      if (open && (best_e == MR2_NONE || b[i] > best ||
                   (b[i] == best && e < best_e))) {
        best = b[i];
        best_s = s[i];
        best_e = e;
      }
    }
    for (int o = 16; o > 0; o >>= 1) {
      const float other = __shfl_xor_sync(0xffffffffu, best, o);
      const float other_s = __shfl_xor_sync(0xffffffffu, best_s, o);
      const unsigned other_e = __shfl_xor_sync(0xffffffffu, best_e, o);
      if (other_e != MR2_NONE &&
          (best_e == MR2_NONE || other > best ||
           (other == best && other_e < best_e))) {
        best = other;
        best_s = other_s;
        best_e = other_e;
      }
    }
#pragma unroll
    for (int i = 0; i < 16; ++i) {
      if ((unsigned)(lane + 32 * i) == best_e) taken |= 1u << i;
    }
    picked_sum += best_s;
    if (lane == 0) {
      dst[k] = (float)best_e;
      dst[top_k + k] = best_s;
    }
  }
  if (lane == 0) {
    for (int k = 0; k < top_k; ++k) {
      dst[top_k + k] = dst[top_k + k] / picked_sum * scale;
    }
  }
}

__device__ __forceinline__ float mr2_up_dot(const uint8_t* __restrict__ row,
                                            const float* __restrict__ xt,
                                            int hidden, int lane) {
  float acc = 0.0f;
  for (int k0 = lane * 16; k0 < hidden; k0 += 512) {
    acc += mr2_dot16(*reinterpret_cast<const uint4*>(row + k0), xt + k0);
  }
  return acc;
}

// act[pair, n] = relu(scale * up[e, n] . x)^2; grid (pairs, inter / 8), 256 threads.
extern "C" __global__ void izwi_mr2_up(
    const float* __restrict__ x, const float* __restrict__ routing,
    const uint8_t* __restrict__ w, const float* __restrict__ ws,
    float* __restrict__ act, int slots, int hidden, int inter, int experts) {
  const int lane = threadIdx.x & 31;
  const int pair = blockIdx.x;
  const int token = pair / slots;
  const int slot = pair - token * slots;
  const int n = blockIdx.y * 8 + (threadIdx.x >> 5);
  if (n >= inter) return;
  const float id = routing[(int64_t)token * 2 * slots + slot];
  if (!mr2_valid(id, experts)) {
    if (lane == 0) act[(int64_t)pair * inter + n] = 0.0f;
    return;
  }
  const int e = (int)id;
  const uint8_t* row = w + ((int64_t)e * inter + n) * hidden;
  const float v =
      mr2_warp_sum(mr2_up_dot(row, x + (int64_t)token * hidden, hidden, lane)) *
      256.0f * ws[e];
  if (lane == 0) {
    const float r = fmaxf(v, 0.0f);
    act[(int64_t)pair * inter + n] = r * r;
  }
}

// y[t, h] = sum_s weight_s * (scale * down[e_s, h] . act[pair_s]), slot order;
// grid (tokens, hidden / 8), 256 threads.
extern "C" __global__ void izwi_mr2_down(
    const float* __restrict__ act, const float* __restrict__ routing,
    const uint8_t* __restrict__ w, const float* __restrict__ ws,
    float* __restrict__ y, int slots, int hidden, int inter, int experts) {
  const int lane = threadIdx.x & 31;
  const int token = blockIdx.x;
  const int h = blockIdx.y * 8 + (threadIdx.x >> 5);
  if (h >= hidden) return;
  float total = 0.0f;
  for (int s = 0; s < slots; ++s) {
    const float id = routing[(int64_t)token * 2 * slots + s];
    if (!mr2_valid(id, experts)) continue;
    const int e = (int)id;
    const uint8_t* row = w + ((int64_t)e * hidden + h) * inter;
    const float* a = act + ((int64_t)token * slots + s) * inter;
    float acc = 0.0f;
    for (int k0 = lane * 16; k0 < inter; k0 += 512) {
      acc += mr2_dot16(*reinterpret_cast<const uint4*>(row + k0), a + k0);
    }
    const float p = mr2_warp_sum(acc) * 256.0f * ws[e];
    total = fmaf(routing[(int64_t)token * 2 * slots + slots + s], p, total);
  }
  if (lane == 0) y[(int64_t)token * hidden + h] = total;
}

// Pairs grouped by expert (offsets[e]..offsets[e + 1] index sorted_pairs);
// one block of 1024 threads, experts <= 512.
extern "C" __global__ void izwi_mr2_group_pairs(
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
    if (mr2_valid(id, experts)) atomicAdd(&counts[(int)id], 1);
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
    if (mr2_valid(id, experts)) {
      const int e = (int)id;
      sorted_pairs[starts[e] + atomicAdd(&fill[e], 1)] = p;
    }
  }
}

// One expert's up channel for all its pairs, the row in registers
// (hidden <= 4096); grid (experts, inter / 8), 256 threads.
extern "C" __global__ void izwi_mr2_up_grouped(
    const float* __restrict__ x, const int* __restrict__ offsets,
    const int* __restrict__ sorted_pairs, const uint8_t* __restrict__ w,
    const float* __restrict__ ws, float* __restrict__ act, int slots,
    int hidden, int inter) {
  const int lane = threadIdx.x & 31;
  const int e = blockIdx.x;
  const int begin = offsets[e];
  const int end = offsets[e + 1];
  const int n = blockIdx.y * 8 + (threadIdx.x >> 5);
  if (begin == end || n >= inter) return;
  const uint8_t* row = w + ((int64_t)e * inter + n) * hidden;
  const int sweeps = (hidden + 511) / 512;
  uint4 q[8];
#pragma unroll
  for (int c = 0; c < 8; ++c) {
    const int k0 = lane * 16 + c * 512;
    q[c] = (c < sweeps && k0 < hidden)
               ? *reinterpret_cast<const uint4*>(row + k0)
               : make_uint4(0, 0, 0, 0);
  }
  const float scale = ws[e];
  for (int i = begin; i < end; ++i) {
    const int pair = sorted_pairs[i];
    const float* xt = x + (int64_t)(pair / slots) * hidden;
    float acc = 0.0f;
#pragma unroll
    for (int c = 0; c < 8; ++c) {
      const int k0 = lane * 16 + c * 512;
      if (c < sweeps && k0 < hidden) acc += mr2_dot16(q[c], xt + k0);
    }
    const float v = mr2_warp_sum(acc) * 256.0f * scale;
    if (lane == 0) {
      const float r = fmaxf(v, 0.0f);
      act[(int64_t)pair * inter + n] = r * r;
    }
  }
}

// One expert's down channel for all its pairs, the row in registers
// (inter <= 2048); partial[pair, h] = scale * dot. Grid (experts, hidden / 8).
extern "C" __global__ void izwi_mr2_down_grouped(
    const float* __restrict__ act, const int* __restrict__ offsets,
    const int* __restrict__ sorted_pairs, const uint8_t* __restrict__ w,
    const float* __restrict__ ws, float* __restrict__ partial, int hidden,
    int inter) {
  const int lane = threadIdx.x & 31;
  const int e = blockIdx.x;
  const int begin = offsets[e];
  const int end = offsets[e + 1];
  const int h = blockIdx.y * 8 + (threadIdx.x >> 5);
  if (begin == end || h >= hidden) return;
  const uint8_t* row = w + ((int64_t)e * hidden + h) * inter;
  const int sweeps = (inter + 511) / 512;
  uint4 q[4];
#pragma unroll
  for (int c = 0; c < 4; ++c) {
    const int k0 = lane * 16 + c * 512;
    q[c] = (c < sweeps && k0 < inter)
               ? *reinterpret_cast<const uint4*>(row + k0)
               : make_uint4(0, 0, 0, 0);
  }
  const float scale = ws[e];
  for (int i = begin; i < end; ++i) {
    const int pair = sorted_pairs[i];
    const float* a = act + (int64_t)pair * inter;
    float acc = 0.0f;
#pragma unroll
    for (int c = 0; c < 4; ++c) {
      const int k0 = lane * 16 + c * 512;
      if (c < sweeps && k0 < inter) acc += mr2_dot16(q[c], a + k0);
    }
    const float p = mr2_warp_sum(acc) * 256.0f * scale;
    if (lane == 0) partial[(int64_t)pair * hidden + h] = p;
  }
}

// y[t, h] = sum_s weight_s * partial[pair_s, h] in slot order, as
// izwi_mr2_down sums; grid (hidden / 256, tokens), 256 threads.
extern "C" __global__ void izwi_mr2_combine(
    const float* __restrict__ partial, const float* __restrict__ routing,
    float* __restrict__ y, int slots, int hidden, int experts) {
  const int h = blockIdx.x * blockDim.x + threadIdx.x;
  const int token = blockIdx.y;
  if (h >= hidden) return;
  const float* row = routing + (int64_t)token * 2 * slots;
  float total = 0.0f;
  for (int s = 0; s < slots; ++s) {
    if (!mr2_valid(row[s], experts)) continue;
    total = fmaf(row[slots + s],
                 partial[((int64_t)token * slots + s) * hidden + h], total);
  }
  y[(int64_t)token * hidden + h] = total;
}

// out[i] = decode(w[expert][i]) * scale[expert], the long-span GEMM path's
// transient dense expert; grid ceil(count / 256), 256 threads.
extern "C" __global__ void izwi_mr2_dequant_f32(
    const uint8_t* __restrict__ w, const float* __restrict__ ws,
    float* __restrict__ out, int expert, int64_t count) {
  const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= count) return;
  out[i] = mr2_e4m3(w[(int64_t)expert * count + i]) * 256.0f * ws[expert];
}

extern "C" __global__ void izwi_mr2_dequant_bf16(
    const uint8_t* __restrict__ w, const float* __restrict__ ws,
    __nv_bfloat16* __restrict__ out, int expert, int64_t count) {
  const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= count) return;
  out[i] = __float2bfloat16(
      mr2_e4m3(w[(int64_t)expert * count + i]) * 256.0f * ws[expert]);
}
