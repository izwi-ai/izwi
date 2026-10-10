#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <math.h>

// Qwen3.6-MoE device-routed sparse-expert kernels over 128x128 block-scaled
// E4M3FN weights. Three launches replace the host-routed per-expert loop:
// router (softmax + top-k + renormalize), grouped gate+up with the SwiGLU
// epilogue, and grouped down with the weighted combine. Expert ids stay on the
// device, so a decode step has no host synchronization inside the MoE block.
//
// Routing tensor contract (F32, [tokens, 2 * slots]): the first `slots` values
// of a row are expert ids (small integers, exact in F32), the next `slots`
// values are the combine weights. slots = top_k (+1 when the shared expert is
// folded in as a stacked slot).

#define Q36_WARPS 8
#define Q36_MAX_SLOTS 32
#define Q36_MAX_LANE_EXPERTS 16

extern __shared__ float q36_shared[];

__device__ __forceinline__ float q36_warp_sum(float v) {
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  return v;
}

__device__ __forceinline__ float q36_warp_max(float v) {
  for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
  return v;
}

// Four E4M3FN bytes dotted with four activations. Each byte is moved into the
// high byte of an F16 lane and its exponent/mantissa shifted down one bit, which
// reproduces the E4M3 value exactly as an F16 scaled by 2^-8 (bias 7 vs 15);
// callers multiply the final sum by 256. Exact for every finite E4M3 value,
// subnormals included. 0x7F/0xFF (NaN) decode as finite values; published
// checkpoints contain none.
__device__ __forceinline__ float q36_dot4(unsigned w, const float* x) {
  unsigned lo = __byte_perm(w, 0u, 0x1404);
  unsigned hi = __byte_perm(w, 0u, 0x3424);
  lo = (lo & 0x80008000u) | ((lo & 0x7F007F00u) >> 1);
  hi = (hi & 0x80008000u) | ((hi & 0x7F007F00u) >> 1);
  const float2 a = __half22float2(*reinterpret_cast<const __half2*>(&lo));
  const float2 b = __half22float2(*reinterpret_cast<const __half2*>(&hi));
  float acc = a.x * x[0];
  acc = fmaf(a.y, x[1], acc);
  acc = fmaf(b.x, x[2], acc);
  return fmaf(b.y, x[3], acc);
}

__device__ __forceinline__ float q36_dot16(uint4 q, const float* x) {
  float acc = q36_dot4(q.x, x);
  acc += q36_dot4(q.y, x + 4);
  acc += q36_dot4(q.z, x + 8);
  return acc + q36_dot4(q.w, x + 12);
}

// One warp per token. shared_mode: 0 none, 1 ungated shared slot (weight 1),
// 2 shared slot gated by sigmoid(logits[token, num_experts]).
extern "C" __global__ void qwen36moe_route_f32(
    const float* __restrict__ logits,
    float* __restrict__ out,
    int tokens,
    int num_experts,
    int ldl,
    int top_k,
    int shared_mode,
    int shared_slot_id,
    int norm_topk) {
  const int lane = threadIdx.x & 31;
  const int token = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  if (token >= tokens) {
    return;
  }
  const float* row = logits + (size_t)token * ldl;
  float v[Q36_MAX_LANE_EXPERTS];
  float local_max = -INFINITY;
#pragma unroll
  for (int i = 0; i < Q36_MAX_LANE_EXPERTS; ++i) {
    const int e = lane + 32 * i;
    v[i] = e < num_experts ? row[e] : -INFINITY;
    local_max = fmaxf(local_max, v[i]);
  }
  const float row_max = q36_warp_max(local_max);
  float local_sum = 0.f;
#pragma unroll
  for (int i = 0; i < Q36_MAX_LANE_EXPERTS; ++i) {
    if (lane + 32 * i < num_experts) {
      local_sum += expf(v[i] - row_max);
    }
  }
  const float total = q36_warp_sum(local_sum);

  const int slots = top_k + (shared_mode != 0 ? 1 : 0);
  float* dst = out + (size_t)token * 2 * slots;
  unsigned taken = 0u;
  float picked_sum = 0.f;
  for (int k = 0; k < top_k; ++k) {
    float best = -INFINITY;
    int best_e = 0x7fffffff;
#pragma unroll
    for (int i = 0; i < Q36_MAX_LANE_EXPERTS; ++i) {
      const int e = lane + 32 * i;
      const bool open = e < num_experts && ((taken >> i) & 1u) == 0u;
      if (open && (best_e == 0x7fffffff || v[i] > best || (v[i] == best && e < best_e))) {
        best = v[i];
        best_e = e;
      }
    }
    for (int o = 16; o > 0; o >>= 1) {
      const float other = __shfl_xor_sync(0xffffffffu, best, o);
      const int other_e = __shfl_xor_sync(0xffffffffu, best_e, o);
      if (other_e != 0x7fffffff &&
          (best_e == 0x7fffffff || other > best || (other == best && other_e < best_e))) {
        best = other;
        best_e = other_e;
      }
    }
#pragma unroll
    for (int i = 0; i < Q36_MAX_LANE_EXPERTS; ++i) {
      if (lane + 32 * i == best_e) {
        taken |= 1u << i;
      }
    }
    const float p = expf(best - row_max);
    picked_sum += p;
    if (lane == 0) {
      dst[k] = (float)best_e;
      dst[slots + k] = p;
    }
  }
  if (lane == 0) {
    const float denom = norm_topk != 0 ? picked_sum : total;
    for (int k = 0; k < top_k; ++k) {
      dst[slots + k] = dst[slots + k] / denom;
    }
    if (shared_mode != 0) {
      dst[top_k] = (float)shared_slot_id;
      dst[slots + top_k] =
          shared_mode == 2 ? 1.f / (1.f + expf(-row[num_experts])) : 1.f;
    }
  }
}

// act[pair, n] = silu(gate_e[n] . x_t) * (up_e[n] . x_t), pair = token * slots + slot.
// w13: [experts_total, 2 * inter, hidden] bytes, gate rows then up rows.
// s13: [experts_total, 2 * inter / 128, hidden / 128] F32 block scales.
// Grid: (tokens * slots, ceil(inter / Q36_WARPS)); one warp per output channel.
template <class T>
__device__ void q36_gate_up(
    const T* __restrict__ x,
    const float* __restrict__ routing,
    const unsigned char* __restrict__ w13,
    const float* __restrict__ s13,
    T* __restrict__ act,
    int slots,
    int hidden,
    int inter,
    int experts_total) {
  const int pair = blockIdx.x;
  const int token = pair / slots;
  const int slot = pair - token * slots;
  const int expert = (int)routing[(size_t)token * 2 * slots + slot];
  for (int k = threadIdx.x; k < hidden; k += blockDim.x) {
    q36_shared[k] = float(x[(size_t)token * hidden + k]);
  }
  __syncthreads();
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int n = blockIdx.y * Q36_WARPS + warp;
  if (n >= inter) {
    return;
  }
  if (expert < 0 || expert >= experts_total) {
    if (lane == 0) {
      act[(size_t)pair * inter + n] = T(0.f);
    }
    return;
  }
  const int kblocks = hidden >> 7;
  const size_t gate_row = (size_t)expert * 2 * inter + n;
  const unsigned char* wg = w13 + gate_row * hidden;
  const unsigned char* wu = wg + (size_t)inter * hidden;
  const float* expert_scales = s13 + (size_t)expert * ((2 * inter) >> 7) * kblocks;
  const float* sg = expert_scales + (size_t)(n >> 7) * kblocks;
  const float* su = expert_scales + (size_t)((inter + n) >> 7) * kblocks;
  float accg = 0.f;
  float accu = 0.f;
  for (int k0 = lane * 16; k0 < hidden; k0 += 32 * 16) {
    const uint4 qg = *reinterpret_cast<const uint4*>(wg + k0);
    const uint4 qu = *reinterpret_cast<const uint4*>(wu + k0);
    const float* xv = q36_shared + k0;
    const int kb = k0 >> 7;
    accg = fmaf(q36_dot16(qg, xv), sg[kb], accg);
    accu = fmaf(q36_dot16(qu, xv), su[kb], accu);
  }
  accg = q36_warp_sum(accg) * 256.f;
  accu = q36_warp_sum(accu) * 256.f;
  if (lane == 0) {
    const float silu = accg / (1.f + expf(-accg));
    act[(size_t)pair * inter + n] = T(silu * accu);
  }
}

// y[token, h] = sum_slot weight[slot] * (down_e[h] . act[token * slots + slot]).
// w2: [experts_total, hidden, inter] bytes; s2: [experts_total, hidden / 128, inter / 128].
// Grid: (tokens, ceil(hidden / Q36_WARPS)); the combine accumulates in F32 and
// rounds once, replacing per-expert index_add in the activation dtype.
template <class T>
__device__ void q36_down(
    const T* __restrict__ act,
    const float* __restrict__ routing,
    const unsigned char* __restrict__ w2,
    const float* __restrict__ s2,
    T* __restrict__ y,
    int slots,
    int hidden,
    int inter,
    int experts_total) {
  __shared__ int ids[Q36_MAX_SLOTS];
  __shared__ float weights[Q36_MAX_SLOTS];
  const int token = blockIdx.x;
  const size_t base = (size_t)token * slots * inter;
  for (int i = threadIdx.x; i < slots * inter; i += blockDim.x) {
    q36_shared[i] = float(act[base + i]);
  }
  if (threadIdx.x < slots) {
    ids[threadIdx.x] = (int)routing[(size_t)token * 2 * slots + threadIdx.x];
    weights[threadIdx.x] = routing[(size_t)token * 2 * slots + slots + threadIdx.x];
  }
  __syncthreads();
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int h = blockIdx.y * Q36_WARPS + warp;
  if (h >= hidden) {
    return;
  }
  const int kblocks = inter >> 7;
  float total = 0.f;
  for (int s = 0; s < slots; ++s) {
    const int e = ids[s];
    if (e < 0 || e >= experts_total) {
      continue;
    }
    const unsigned char* w = w2 + ((size_t)e * hidden + h) * inter;
    const float* sc = s2 + ((size_t)e * (hidden >> 7) + (h >> 7)) * kblocks;
    const float* av = q36_shared + s * inter;
    float acc = 0.f;
    for (int k0 = lane * 16; k0 < inter; k0 += 32 * 16) {
      const uint4 q = *reinterpret_cast<const uint4*>(w + k0);
      acc = fmaf(q36_dot16(q, av + k0), sc[k0 >> 7], acc);
    }
    total = fmaf(weights[s], acc, total);
  }
  total = q36_warp_sum(total) * 256.f;
  if (lane == 0) {
    y[(size_t)token * hidden + h] = T(total);
  }
}

#define Q36_EXPORT(T, S)                                                                      \
  extern "C" __global__ void __launch_bounds__(256) qwen36moe_gate_up_##S(                   \
      const T* x, const float* r, const unsigned char* w, const float* s, T* a, int slots,    \
      int hidden, int inter, int experts) {                                                   \
    q36_gate_up<T>(x, r, w, s, a, slots, hidden, inter, experts);                             \
  }                                                                                           \
  extern "C" __global__ void __launch_bounds__(256) qwen36moe_down_##S(                      \
      const T* a, const float* r, const unsigned char* w, const float* s, T* y, int slots,    \
      int hidden, int inter, int experts) {                                                   \
    q36_down<T>(a, r, w, s, y, slots, hidden, inter, experts);                                \
  }

Q36_EXPORT(__half, f16)
Q36_EXPORT(__nv_bfloat16, bf16)
