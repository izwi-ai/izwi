#include <cuda_bf16.h>
#include <math.h>

// Relative-position self-attention in one launch (`kernels::relpos_attention`):
// Transformer-XL scores (Parakeet, Nemotron Streaming) and Shaw block-local
// scores (Granite Speech), the softmax and the value product, for every head.
//
// One warp owns one query (b, i, h); a block holds 8 warps of consecutive
// queries of one (b, h). Lane l owns dims [l*n, (l+1)*n) of Dh = 32*n. The
// query sees the keys [lo, hi) of its band cut at its row's key length; a
// query that sees none writes zeros. For each key, in order:
//
//   content  = butterfly(lane FMA chain of qa . k_j)      qa = rnd(q + u) or q
//   position = butterfly(lane FMA chain of qb . pos[r])    qb = rnd(q + v) or q
//   XL:   s = rnd(rnd(rnd(content) + rnd(position)) * scale), r = Tq - 1 - i + j
//   Shaw: s = rnd(rnd(rnd(content) * scale) + rnd(position * position_scale)),
//         r = clamp(i - j, -C, C) + center
//
// where rnd rounds to BF16 when round_bf16 is set (eager's roundings in the
// BF16 encoders) and is the identity otherwise. Lane (j - lo) % 32 keeps s in
// shared memory. The softmax is exact two-pass: lane maxima over the lane's
// scores and an xor butterfly of maxima, exp(s - max) through rp_exp, lane
// sums and a butterfly, then p = rnd(e / sum). The output is each lane's FMA
// chain of p_t * v over the keys in order. Every operation is an explicit
// _rn intrinsic (nothing contracts), so `relpos_attention::reference` on the
// host computes the same bits.
//
// Arguments are element strides; dims are contiguous. pos is [R, H, Dh]
// (XL; p_sh is its head stride) or [R, Dh] (Shaw; p_sh = 0). key_lens is
// [B] or null (uniform_key_len then serves every row). out is contiguous
// [B, Tq, H, Dh]. Grid (ceil(Tq / 8), H, B), 256 threads, 8 * max_span
// floats of dynamic shared memory; max_span bounds every query's key count.

#define RP_WARPS 8
#define RP_MAX_LANE_DIMS 8
#define RP_FULL_MASK 0xffffffffu

__device__ __forceinline__ float rp_load(const float* p, long long i) { return p[i]; }
__device__ __forceinline__ float rp_load(const __nv_bfloat16* p, long long i) {
  return __bfloat162float(p[i]);
}

__device__ __forceinline__ void rp_store(float* p, long long i, float v) { p[i] = v; }
__device__ __forceinline__ void rp_store(__nv_bfloat16* p, long long i, float v) {
  p[i] = __float2bfloat16_rn(v);
}

__device__ __forceinline__ float rp_round(float x, int round_bf16) {
  return round_bf16 ? __bfloat162float(__float2bfloat16_rn(x)) : x;
}

// y * 2^n for y in [0.5, 2) and n in [-151, 127]: one multiply in the normal
// range, two below it (the second rounds), as `scale_by_power_of_two`.
__device__ __forceinline__ float rp_scale2(float y, int n) {
  if (n >= -126) return __fmul_rn(y, __int_as_float((n + 127) << 23));
  return __fmul_rn(__fmul_rn(y, __int_as_float((n + 127 + 64) << 23)),
                   __int_as_float((127 - 64) << 23));
}

// `relpos_attention::det_exp`: Cody-Waite reduction and Cephes' expf
// polynomial, the same literals and operations.
__device__ __forceinline__ float rp_exp(float x) {
  if (isnan(x)) return x;
  if (x <= -104.0f) return 0.0f;
  const float n = rintf(__fmul_rn(x, 1.442695f));
  float r = __fmaf_rn(-n, 0.6933594f, x);
  r = __fmaf_rn(-n, -0.00021219444f, r);
  float p = 0.00019875691f;
  p = __fmaf_rn(p, r, 0.0013981999f);
  p = __fmaf_rn(p, r, 0.008333452f);
  p = __fmaf_rn(p, r, 0.041665796f);
  p = __fmaf_rn(p, r, 0.16666666f);
  p = __fmaf_rn(p, r, 0.5f);
  const float y = __fadd_rn(__fmaf_rn(p, __fmul_rn(r, r), r), 1.0f);
  return rp_scale2(y, (int)n);
}

__device__ __forceinline__ long long rp_clip(long long value, long long len) {
  return value < 0 ? 0 : (value > len ? len : value);
}

template <class S, class O>
__device__ void rp_attend(
    const S* __restrict__ q, const S* __restrict__ k, const S* __restrict__ v,
    const S* __restrict__ pos, const float* __restrict__ bias_u,
    const float* __restrict__ bias_v, const unsigned int* __restrict__ key_lens,
    O* __restrict__ out, int q_len, int k_len, int heads, int head_dim, long long q_sb,
    long long q_st, long long q_sh, long long k_sb, long long k_st, long long k_sh,
    long long v_sb, long long v_st, long long v_sh, long long p_sr, long long p_sh, int shaw,
    int shaw_clamp, int shaw_center, int band, int band_size, int left_chunks,
    long long query_origin, long long key_origin, int uniform_key_len, float scale,
    float position_scale, int round_bf16, int max_span) {
  extern __shared__ float rp_scores[];
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const long long i = (long long)blockIdx.x * RP_WARPS + warp;
  const int h = blockIdx.y;
  const int b = blockIdx.z;
  if (i >= q_len) return;  // the whole warp
  float* scores = rp_scores + (long long)warp * max_span;
  const int per_lane = head_dim >> 5;
  const int d0 = lane * per_lane;

  long long key_len = k_len;
  if (key_lens != nullptr) {
    if (key_lens[b] < (unsigned int)k_len) key_len = key_lens[b];
  } else if (uniform_key_len < k_len) {
    key_len = uniform_key_len;
  }
  long long lo = 0, hi = key_len;
  if (band == 1) {
    const long long chunk = band_size;
    const long long position = query_origin + i;
    long long query_chunk = position / chunk;
    if (position % chunk != 0 && position < 0) query_chunk -= 1;  // floor
    lo = rp_clip(chunk * (query_chunk - left_chunks) - key_origin, key_len);
    hi = rp_clip(chunk * (query_chunk + 1) - key_origin, key_len);
  } else if (band == 2) {
    const long long start = i / band_size * band_size;
    lo = rp_clip(start, key_len);
    hi = rp_clip(start + band_size, key_len);
  }

  O* out_row = out + (((long long)b * q_len + i) * heads + h) * head_dim + d0;
  if (hi <= lo) {
#pragma unroll
    for (int d = 0; d < RP_MAX_LANE_DIMS; ++d)
      if (d < per_lane) rp_store(out_row, d, 0.0f);
    return;
  }
  const int span = (int)(hi - lo);

  float qa[RP_MAX_LANE_DIMS], qb[RP_MAX_LANE_DIMS];
  const S* q_row = q + b * q_sb + i * q_st + h * q_sh + d0;
#pragma unroll
  for (int d = 0; d < RP_MAX_LANE_DIMS; ++d) {
    if (d < per_lane) {
      const float x = rp_load(q_row, d);
      if (shaw) {
        qa[d] = x;
        qb[d] = x;
      } else {
        qa[d] = rp_round(__fadd_rn(x, bias_u[h * head_dim + d0 + d]), round_bf16);
        qb[d] = rp_round(__fadd_rn(x, bias_v[h * head_dim + d0 + d]), round_bf16);
      }
    }
  }

  for (long long j = lo; j < hi; ++j) {
    long long r;
    if (shaw) {
      long long distance = i - j;
      if (distance < -shaw_clamp) distance = -shaw_clamp;
      if (distance > shaw_clamp) distance = shaw_clamp;
      r = shaw_center + distance;
    } else {
      r = (long long)q_len - 1 - i + j;
    }
    const S* k_row = k + b * k_sb + j * k_st + h * k_sh + d0;
    const S* p_row = pos + r * p_sr + h * p_sh + d0;
    float content = __fmul_rn(qa[0], rp_load(k_row, 0));
    float position = __fmul_rn(qb[0], rp_load(p_row, 0));
#pragma unroll
    for (int d = 1; d < RP_MAX_LANE_DIMS; ++d) {
      if (d < per_lane) {
        content = __fmaf_rn(qa[d], rp_load(k_row, d), content);
        position = __fmaf_rn(qb[d], rp_load(p_row, d), position);
      }
    }
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
      content = __fadd_rn(content, __shfl_xor_sync(RP_FULL_MASK, content, offset));
      position = __fadd_rn(position, __shfl_xor_sync(RP_FULL_MASK, position, offset));
    }
    float s;
    if (shaw) {
      const float scaled = rp_round(__fmul_rn(rp_round(content, round_bf16), scale), round_bf16);
      s = rp_round(__fadd_rn(scaled, rp_round(__fmul_rn(position, position_scale), round_bf16)),
                   round_bf16);
    } else {
      const float sum = rp_round(
          __fadd_rn(rp_round(content, round_bf16), rp_round(position, round_bf16)), round_bf16);
      s = rp_round(__fmul_rn(sum, scale), round_bf16);
    }
    if (lane == (int)((j - lo) & 31)) scores[j - lo] = s;
  }

  // Each lane reads back only the scores it wrote until the barrier below.
  float max = -INFINITY;
  for (int t = lane; t < span; t += 32) {
    const float s = scores[t];
    if (s > max) max = s;
  }
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    const float other = __shfl_xor_sync(RP_FULL_MASK, max, offset);
    if (other > max) max = other;
  }
  float sum = 0.0f;
  for (int t = lane; t < span; t += 32) {
    const float e = rp_exp(__fsub_rn(scores[t], max));
    scores[t] = e;
    sum = __fadd_rn(sum, e);
  }
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1)
    sum = __fadd_rn(sum, __shfl_xor_sync(RP_FULL_MASK, sum, offset));
  for (int t = lane; t < span; t += 32)
    scores[t] = rp_round(__fdiv_rn(scores[t], sum), round_bf16);
  __syncwarp();

  float acc[RP_MAX_LANE_DIMS];
#pragma unroll
  for (int d = 0; d < RP_MAX_LANE_DIMS; ++d) acc[d] = 0.0f;
  for (int t = 0; t < span; ++t) {
    const float p = scores[t];
    const S* v_row = v + b * v_sb + (lo + t) * v_st + h * v_sh + d0;
#pragma unroll
    for (int d = 0; d < RP_MAX_LANE_DIMS; ++d)
      if (d < per_lane) acc[d] = __fmaf_rn(p, rp_load(v_row, d), acc[d]);
  }
#pragma unroll
  for (int d = 0; d < RP_MAX_LANE_DIMS; ++d)
    if (d < per_lane) rp_store(out_row, d, acc[d]);
}

#define RP_EXPORT(NAME, S, O)                                                                \
  extern "C" __global__ void __launch_bounds__(256) izwi_relpos_attention_##NAME(           \
      const S* q, const S* k, const S* v, const S* pos, const float* bias_u,                \
      const float* bias_v, const unsigned int* key_lens, O* out, int q_len, int k_len,      \
      int heads, int head_dim, long long q_sb, long long q_st, long long q_sh, long long k_sb, \
      long long k_st, long long k_sh, long long v_sb, long long v_st, long long v_sh,        \
      long long p_sr, long long p_sh, int shaw, int shaw_clamp, int shaw_center, int band,  \
      int band_size, int left_chunks, long long query_origin, long long key_origin,         \
      int uniform_key_len, float scale, float position_scale, int round_bf16,              \
      int max_span) {                                                                        \
    rp_attend<S, O>(q, k, v, pos, bias_u, bias_v, key_lens, out, q_len, k_len, heads,       \
                    head_dim, q_sb, q_st, q_sh, k_sb, k_st, k_sh, v_sb, v_st, v_sh, p_sr,   \
                    p_sh, shaw, shaw_clamp, shaw_center, band, band_size, left_chunks,      \
                    query_origin, key_origin, uniform_key_len, scale, position_scale,       \
                    round_bf16, max_span);                                                   \
  }

// storage_output
RP_EXPORT(f32_f32, float, float)
RP_EXPORT(bf16_bf16, __nv_bfloat16, __nv_bfloat16)
RP_EXPORT(bf16_f32, __nv_bfloat16, float)
RP_EXPORT(f32_bf16, float, __nv_bfloat16)
