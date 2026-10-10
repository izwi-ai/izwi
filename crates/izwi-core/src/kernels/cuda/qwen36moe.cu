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

// ---------------------------------------------------------------------------
// Gated DeltaNet single-token decode (Qwen3.6 linear-attention layers).
//
// Two launches replace the ~49-op Candle chain per layer: a causal-conv step
// over the 3-slot history ring, then one block per value head that runs the
// softplus/sigmoid gating, q/k L2 norms, the delta-rule recurrence (state held
// in registers) and the gated RMSNorm. Key heads are mapped by index (grouped:
// v / repeats, tiled: v % key_heads), so no expanded q/k copies are made. The
// old state is read-only; the next state goes to a fresh allocation, keeping
// state publication transactional.

// out = silu(h0*w0 + h1*w1 + h2*w2 + x*w3) per channel (history oldest first);
// cur = float(x), the ring's next slot. w: [conv_dim, 4] F32.
template <class T>
__device__ void q36_gdn_conv(
    const T* __restrict__ x,
    const float* __restrict__ w,
    const float* __restrict__ h0,
    const float* __restrict__ h1,
    const float* __restrict__ h2,
    float* __restrict__ out,
    float* __restrict__ cur,
    int conv_dim) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= conv_dim) {
    return;
  }
  const float xc = float(x[c]);
  const float* wc = w + (size_t)c * 4;
  float v = xc * wc[3];
  v = v + h0[c] * wc[0];
  v = v + h1[c] * wc[1];
  v = v + h2[c] * wc[2];
  out[c] = v / (1.f + expf(-v));
  cur[c] = xc;
}

// conv: [conv_dim] F32 = q (key_heads*128) | k (key_heads*128) | v (value_heads*128)
// after conv+silu. z: [value_heads*128]; beta_raw, alpha: [value_heads]
// (projection outputs, pre-activation); dt_bias, a: [value_heads] F32 with
// a = -exp(A_log); norm_w: [128] F32. state_in/state_out: [value_heads, 128, 128]
// F32 (key rows, value columns). y: [value_heads*128] = rmsnorm(o) * w * silu(z).
// Block: 512 threads; warp w owns value columns (w & 3) * 32 + lane over key
// rows (w >> 2) * 32 .. +32. Grid: value_heads.
template <class T>
__device__ void q36_gdn_decode(
    const float* __restrict__ conv,
    const T* __restrict__ z,
    const T* __restrict__ beta_raw,
    const T* __restrict__ alpha,
    const float* __restrict__ dt_bias,
    const float* __restrict__ a,
    const float* __restrict__ norm_w,
    const float* __restrict__ state_in,
    float* __restrict__ state_out,
    T* __restrict__ y,
    int key_heads,
    int value_heads,
    int grouped,
    float norm_eps) {
  __shared__ float qs[128];
  __shared__ float ks[128];
  __shared__ float red[4][128];
  __shared__ float part[8];
  __shared__ float stats[2];
  const int h = blockIdx.x;
  const int repeats = value_heads / key_heads;
  const int kh = grouped != 0 ? h / repeats : h % key_heads;
  const int tid = threadIdx.x;
  const int warp = tid >> 5;
  const int lane = tid & 31;
  const int col = (warp & 3) * 32 + lane;
  const int rg = warp >> 2;
  const int key_width = key_heads * 128;
  if (tid < 128) {
    qs[tid] = conv[kh * 128 + tid];
    ks[tid] = conv[key_width + kh * 128 + tid];
  }
  __syncthreads();
  if (warp < 8) {
    const float val = warp < 4 ? qs[warp * 32 + lane] : ks[(warp - 4) * 32 + lane];
    const float sq = q36_warp_sum(val * val);
    if (lane == 0) {
      part[warp] = sq;
    }
  }
  __syncthreads();
  if (tid == 0) {
    const float qsum = part[0] + part[1] + part[2] + part[3];
    const float ksum = part[4] + part[5] + part[6] + part[7];
    // l2norm(x) = x / sqrt(sum(x^2) + 1e-6); queries also take 1/sqrt(Dk).
    stats[0] = 1.f / (sqrtf(qsum + 1e-6f) * sqrtf(128.f));
    stats[1] = 1.f / sqrtf(ksum + 1e-6f);
  }
  __syncthreads();
  const float qscale = stats[0];
  const float knorm = stats[1];
  const float gate_in = float(alpha[h]) + dt_bias[h];
  const float softplus = fmaxf(gate_in, 0.f) + log1pf(expf(-fabsf(gate_in)));
  const float decay = expf(softplus * a[h]);
  const float beta = 1.f / (1.f + expf(-float(beta_raw[h])));

  const size_t row0 = (size_t)h * 128 + (size_t)rg * 32;
  const float* sin = state_in + row0 * 128 + col;
  float s[32];
  float recalled = 0.f;
#pragma unroll
  for (int i = 0; i < 32; ++i) {
    s[i] = sin[(size_t)i * 128] * decay;
    recalled = fmaf(ks[rg * 32 + i] * knorm, s[i], recalled);
  }
  red[rg][col] = recalled;
  __syncthreads();
  const float kv = red[0][col] + red[1][col] + red[2][col] + red[3][col];
  const float delta = (conv[2 * key_width + h * 128 + col] - kv) * beta;
  float* sout = state_out + row0 * 128 + col;
  float o = 0.f;
#pragma unroll
  for (int i = 0; i < 32; ++i) {
    const float updated = fmaf(ks[rg * 32 + i] * knorm, delta, s[i]);
    sout[(size_t)i * 128] = updated;
    o = fmaf(qs[rg * 32 + i] * qscale, updated, o);
  }
  __syncthreads();
  red[rg][col] = o;
  __syncthreads();
  float out = 0.f;
  if (rg == 0) {
    out = red[0][col] + red[1][col] + red[2][col] + red[3][col];
    const float sq = q36_warp_sum(out * out);
    if (lane == 0) {
      part[warp] = sq;
    }
  }
  __syncthreads();
  if (rg == 0) {
    const float mean_sq = (part[0] + part[1] + part[2] + part[3]) / 128.f;
    const float zz = float(z[h * 128 + col]);
    const float gate = zz / (1.f + expf(-zz));
    y[h * 128 + col] = T(out / sqrtf(mean_sq + norm_eps) * norm_w[col] * gate);
  }
}

#define Q36_GDN_EXPORT(T, S)                                                                   \
  extern "C" __global__ void qwen36moe_gdn_conv_##S(                                           \
      const T* x, const float* w, const float* h0, const float* h1, const float* h2,          \
      float* out, float* cur, int conv_dim) {                                                  \
    q36_gdn_conv<T>(x, w, h0, h1, h2, out, cur, conv_dim);                                     \
  }                                                                                            \
  extern "C" __global__ void __launch_bounds__(512) qwen36moe_gdn_decode_##S(                 \
      const float* conv, const T* z, const T* beta_raw, const T* alpha, const float* dt_bias, \
      const float* a, const float* norm_w, const float* state_in, float* state_out, T* y,     \
      int key_heads, int value_heads, int grouped, float norm_eps) {                           \
    q36_gdn_decode<T>(conv, z, beta_raw, alpha, dt_bias, a, norm_w, state_in, state_out, y,   \
                      key_heads, value_heads, grouped, norm_eps);                              \
  }

Q36_GDN_EXPORT(__half, f16)
Q36_GDN_EXPORT(__nv_bfloat16, bf16)
Q36_GDN_EXPORT(float, f32)

// ---------------------------------------------------------------------------
// RMSNorm with an F32 gain over 16-bit activations, optionally fused with the
// preceding residual add. Replaces cast -> rms_norm -> cast (and the add) with
// one launch. The residual sum is rounded to T before normalization, matching
// a T-dtype add followed by the norm. One block per row.
template <class T>
__device__ void q36_rms_norm(
    const T* __restrict__ x,
    const T* __restrict__ residual,
    const float* __restrict__ w,
    T* __restrict__ sum_out,
    T* __restrict__ out,
    int hidden,
    float eps) {
  __shared__ float part[32];
  __shared__ float inv_shared;
  const size_t base = (size_t)blockIdx.x * hidden;
  float ss = 0.f;
  for (int i = threadIdx.x; i < hidden; i += blockDim.x) {
    float v = float(x[base + i]);
    if (residual != nullptr) {
      const T sum = T(float(residual[base + i]) + v);
      if (sum_out != nullptr) {
        sum_out[base + i] = sum;
      }
      v = float(sum);
    }
    ss = fmaf(v, v, ss);
  }
  ss = q36_warp_sum(ss);
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  if (lane == 0) {
    part[warp] = ss;
  }
  __syncthreads();
  if (warp == 0) {
    const int warps = (blockDim.x + 31) >> 5;
    float total = lane < warps ? part[lane] : 0.f;
    total = q36_warp_sum(total);
    if (lane == 0) {
      inv_shared = 1.f / sqrtf(total / (float)hidden + eps);
    }
  }
  __syncthreads();
  const float inv = inv_shared;
  for (int i = threadIdx.x; i < hidden; i += blockDim.x) {
    float v = float(x[base + i]);
    if (residual != nullptr) {
      v = float(T(float(residual[base + i]) + v));
    }
    out[base + i] = T(v * inv * w[i]);
  }
}

#define Q36_NORM_EXPORT(T, S)                                                                   \
  extern "C" __global__ void __launch_bounds__(256) qwen36moe_rms_norm_##S(                    \
      const T* x, const float* w, T* out, int hidden, float eps) {                              \
    q36_rms_norm<T>(x, nullptr, w, nullptr, out, hidden, eps);                                  \
  }                                                                                             \
  extern "C" __global__ void __launch_bounds__(256) qwen36moe_add_rms_norm_##S(                \
      const T* x, const T* residual, const float* w, T* sum_out, T* out, int hidden,           \
      float eps) {                                                                              \
    q36_rms_norm<T>(x, residual, w, sum_out, out, hidden, eps);                                 \
  }

Q36_NORM_EXPORT(__half, f16)
Q36_NORM_EXPORT(__nv_bfloat16, bf16)

// ---------------------------------------------------------------------------
// Full-attention q/k head norm + partial rotate-half RoPE for one token.
//
// Replaces the per-layer chain of q/k contiguous copies, cast -> rms_norm ->
// cast, the host-built cos/sin upload and cos/sin/cast ops, the rotary op and
// the pass-through concatenation with one launch. Block b < num_heads handles
// query head b; the rest handle key heads. Query heads are read from the
// gated q_proj layout [num_heads, 2 * head_dim] (query half first).
// Rounding mirrors the reference chain: normalized values, cos and sin are
// rounded to T before the rotation, and the result is rounded to T. Angles are
// position * inv_freq in F32 with the interleaved M-RoPE sections
// (dims 1,4,7,.. < 3*sec_h use the height position, 2,5,8,.. < 3*sec_w the
// width position) when the three positions differ.
template <class T>
__device__ void q36_qk_norm_rope(
    const T* __restrict__ q_src,
    const T* __restrict__ k_src,
    const float* __restrict__ q_gain,
    const float* __restrict__ k_gain,
    const float* __restrict__ inv_freq,
    T* __restrict__ q_out,
    T* __restrict__ k_out,
    int num_heads,
    int head_dim,
    int rope_dim,
    int pos_t,
    int pos_h,
    int pos_w,
    int sec_h,
    int sec_w,
    float eps) {
  __shared__ float part[32];
  __shared__ float inv_shared;
  const int head = blockIdx.x;
  const bool is_query = head < num_heads;
  const T* src = is_query ? q_src + (size_t)head * 2 * head_dim
                          : k_src + (size_t)(head - num_heads) * head_dim;
  const float* gain = is_query ? q_gain : k_gain;
  T* dst = is_query ? q_out + (size_t)head * head_dim
                    : k_out + (size_t)(head - num_heads) * head_dim;
  float ss = 0.f;
  for (int i = threadIdx.x; i < head_dim; i += blockDim.x) {
    const float v = float(src[i]);
    ss = fmaf(v, v, ss);
  }
  ss = q36_warp_sum(ss);
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  if (lane == 0) {
    part[warp] = ss;
  }
  __syncthreads();
  if (warp == 0) {
    const int warps = (blockDim.x + 31) >> 5;
    float total = lane < warps ? part[lane] : 0.f;
    total = q36_warp_sum(total);
    if (lane == 0) {
      inv_shared = 1.f / sqrtf(total / (float)head_dim + eps);
    }
  }
  __syncthreads();
  const float inv = inv_shared;
  for (int i = threadIdx.x; i < head_dim; i += blockDim.x) {
    q36_shared[i] = float(T(float(src[i]) * inv * gain[i]));
  }
  __syncthreads();
  const int half = rope_dim >> 1;
  const bool sectioned = pos_t != pos_h || pos_t != pos_w;
  for (int i = threadIdx.x; i < head_dim; i += blockDim.x) {
    float out = q36_shared[i];
    if (i < rope_dim) {
      const int j = i < half ? i : i - half;
      int pos = pos_t;
      if (sectioned) {
        if (j % 3 == 1 && j < 3 * sec_h) {
          pos = pos_h;
        } else if (j % 3 == 2 && j < 3 * sec_w) {
          pos = pos_w;
        }
      }
      const float angle = (float)pos * inv_freq[j];
      const float c = float(T(cosf(angle)));
      const float s = float(T(sinf(angle)));
      out = i < half ? q36_shared[i] * c - q36_shared[i + half] * s
                     : q36_shared[i - half] * s + q36_shared[i] * c;
    }
    dst[i] = T(out);
  }
}

#define Q36_QK_EXPORT(T, S)                                                                      \
  extern "C" __global__ void __launch_bounds__(256) qwen36moe_qk_norm_rope_##S(                 \
      const T* q_src, const T* k_src, const float* q_gain, const float* k_gain,                 \
      const float* inv_freq, T* q_out, T* k_out, int num_heads, int head_dim, int rope_dim,     \
      int pos_t, int pos_h, int pos_w, int sec_h, int sec_w, float eps) {                        \
    q36_qk_norm_rope<T>(q_src, k_src, q_gain, k_gain, inv_freq, q_out, k_out, num_heads,        \
                        head_dim, rope_dim, pos_t, pos_h, pos_w, sec_h, sec_w, eps);             \
  }

Q36_QK_EXPORT(__half, f16)
Q36_QK_EXPORT(__nv_bfloat16, bf16)

// ---------------------------------------------------------------------------
// Router GEMV: logits[t, e] = round_T(x[t] . w[e]) as F32. Reads the 16-bit
// activations and F32 router rows directly, replacing cast -> F32 GEMM ->
// cast -> cast before routing. Logits are rounded through T so routing sees
// the same values as a T-dtype router projection. One warp per router row,
// eight rows per block; grid (ceil(rows / 8), tokens). Requires hidden % 4 == 0.
template <class T>
__device__ void q36_router_logits(
    const T* __restrict__ x,
    const float* __restrict__ w,
    float* __restrict__ out,
    int rows,
    int hidden) {
  const int lane = threadIdx.x & 31;
  const int row = blockIdx.x * Q36_WARPS + (threadIdx.x >> 5);
  const int token = blockIdx.y;
  if (row >= rows) {
    return;
  }
  const float* wr = w + (size_t)row * hidden;
  const T* xt = x + (size_t)token * hidden;
  float acc = 0.f;
  for (int k = lane * 4; k < hidden; k += 32 * 4) {
    const float4 wv = *reinterpret_cast<const float4*>(wr + k);
    acc = fmaf(float(xt[k]), wv.x, acc);
    acc = fmaf(float(xt[k + 1]), wv.y, acc);
    acc = fmaf(float(xt[k + 2]), wv.z, acc);
    acc = fmaf(float(xt[k + 3]), wv.w, acc);
  }
  acc = q36_warp_sum(acc);
  if (lane == 0) {
    out[(size_t)token * rows + row] = float(T(acc));
  }
}

#define Q36_ROUTER_EXPORT(T, S)                                                                \
  extern "C" __global__ void __launch_bounds__(256) qwen36moe_router_logits_##S(              \
      const T* x, const float* w, float* out, int rows, int hidden) {                          \
    q36_router_logits<T>(x, w, out, rows, hidden);                                             \
  }

Q36_ROUTER_EXPORT(__half, f16)
Q36_ROUTER_EXPORT(__nv_bfloat16, bf16)
