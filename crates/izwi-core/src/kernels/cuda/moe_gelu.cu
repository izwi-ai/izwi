#include <cuda_bf16.h>
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
    const float g = mgl_warp_sum(mgl_row_dot(gate, xt, hidden, lane));
    const float u = mgl_warp_sum(mgl_row_dot(up, xt, hidden, lane));
    if (lane == 0) act[(int64_t)pair * inter + n] = mgl_gelu_tanh(g) * u;
  }
}

// partial[pair, h] = down[e, h] . act[pair] for every pair routed to e.
// Grid (experts, hidden / 8), 256 threads.
extern "C" __global__ void izwi_mgl_down_grouped(
    const float* __restrict__ act, const int* __restrict__ offsets,
    const int* __restrict__ sorted_pairs,
    const __nv_bfloat16* __restrict__ down, float* __restrict__ partial,
    int hidden, int inter) {
  const int lane = threadIdx.x & 31;
  const int e = blockIdx.x;
  const int begin = offsets[e];
  const int end = offsets[e + 1];
  const int h = blockIdx.y * 8 + (threadIdx.x >> 5);
  if (begin == end || h >= hidden) return;
  const __nv_bfloat16* row = down + ((int64_t)e * hidden + h) * inter;
  for (int i = begin; i < end; ++i) {
    const int pair = sorted_pairs[i];
    const float p =
        mgl_warp_sum(mgl_row_dot(row, act + (int64_t)pair * inter, inter, lane));
    if (lane == 0) partial[(int64_t)pair * hidden + h] = p;
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
