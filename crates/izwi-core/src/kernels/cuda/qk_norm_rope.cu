#include <cuda_bf16.h>
#include <cuda_fp16.h>

// Per-head q/k RMSNorm and partial RoPE from host-built tables, read straight
// from a packed projection (`kernels::qk_norm_rope`). One block per (row,
// head): heads [0, H) are queries, [H, H + KV) keys; a key block also copies
// its value head. With d the head width and r the rotated dims:
//
//   n_i   = A(x_i * rsqrt(mean(x * x) + eps) * gain_i)     A: the activation dtype
//   out_a = A(n_a * c - n_b * s),  out_b = A(n_a * s + n_b * c)   for each pair (a, b) < r
//   out_i = n_i                                                   for i >= r
//
// The pairs are (p, p + r / 2) (NeoX half split) or (2p, 2p + 1)
// (interleaved), and (c, s) come from row `position` of the cos/sin tables.

#define QNR_THREADS 128

__device__ __forceinline__ float qnr_load(const float* p, long long i) { return p[i]; }
__device__ __forceinline__ float qnr_load(const __half* p, long long i) { return __half2float(p[i]); }
__device__ __forceinline__ float qnr_load(const __nv_bfloat16* p, long long i) {
  return __bfloat162float(p[i]);
}
__device__ __forceinline__ void qnr_store(float* p, long long i, float v) { p[i] = v; }
__device__ __forceinline__ void qnr_store(__half* p, long long i, float v) { p[i] = __float2half(v); }
__device__ __forceinline__ void qnr_store(__nv_bfloat16* p, long long i, float v) {
  p[i] = __float2bfloat16(v);
}
__device__ __forceinline__ float qnr_round(float v, const float*) { return v; }
__device__ __forceinline__ float qnr_round(float v, const __half*) {
  return __half2float(__float2half(v));
}
__device__ __forceinline__ float qnr_round(float v, const __nv_bfloat16*) {
  return __bfloat162float(__float2bfloat16(v));
}

struct QnrDims {
  int width, heads, kv_heads, d, rotary, interleaved, q_offset, q_stride, k_offset,
      v_offset, half_r;
};

template <class T>
__device__ void qnr_head(const T* __restrict__ packed, const float* __restrict__ q_gain,
                         const float* __restrict__ k_gain, const float* __restrict__ cos_t,
                         const float* __restrict__ sin_t, const unsigned* __restrict__ positions,
                         T* __restrict__ q, T* __restrict__ k, T* __restrict__ v,
                         const QnrDims dims, float eps) {
  __shared__ float partial[QNR_THREADS];
  __shared__ float normed[512];
  const int width = dims.width, heads = dims.heads, kv_heads = dims.kv_heads, d = dims.d;
  const int rotary = dims.rotary, interleaved = dims.interleaved;
  const int row = blockIdx.x, head = blockIdx.y;
  const bool is_q = head < heads;
  const int kv = head - heads;
  const long long src = (long long)row * width +
                        (is_q ? dims.q_offset + (long long)head * dims.q_stride
                              : dims.k_offset + (long long)kv * d);
  const float* gain = is_q ? q_gain : k_gain;
  T* dst = is_q ? q + ((long long)row * heads + head) * d : k + ((long long)row * kv_heads + kv) * d;
  const T* tag = nullptr;
  float acc = 0.0f;
  for (int i = threadIdx.x; i < d; i += QNR_THREADS) {
    const float x = qnr_load(packed, src + i);
    acc += x * x;
  }
  partial[threadIdx.x] = acc;
  __syncthreads();
  for (int stride = QNR_THREADS / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
    __syncthreads();
  }
  const float scale = rsqrtf(partial[0] / (float)d + eps);
  for (int i = threadIdx.x; i < d; i += QNR_THREADS) {
    normed[i] = qnr_round(qnr_load(packed, src + i) * scale * gain[i], tag);
  }
  __syncthreads();
  const int half_r = dims.half_r;
  const long long table = (long long)positions[row] * half_r;
  for (int i = threadIdx.x; i < d; i += QNR_THREADS) {
    float out = normed[i];
    if (i < rotary) {
      const int p = interleaved ? i / 2 : (i < half_r ? i : i - half_r);
      const int a = interleaved ? 2 * p : p;
      const int b = interleaved ? 2 * p + 1 : p + half_r;
      const float c = cos_t[table + p], s = sin_t[table + p];
      out = (i == a) ? normed[a] * c - normed[b] * s : normed[a] * s + normed[b] * c;
    }
    qnr_store(dst, i, out);
  }
  if (!is_q) {
    const long long vsrc = (long long)row * width + dims.v_offset + (long long)kv * d;
    T* vdst = v + ((long long)row * kv_heads + kv) * d;
    for (int i = threadIdx.x; i < d; i += QNR_THREADS) vdst[i] = packed[vsrc + i];
  }
}

#define QNR_EXPORT(NAME, T)                                                           \
  extern "C" __global__ void izwi_qk_norm_rope_##NAME(                               \
      const T* packed, const float* q_gain, const float* k_gain, const float* cos_t, \
      const float* sin_t, const unsigned* positions, T* q, T* k, T* v, int width,    \
      int heads, int kv_heads, int d, int rotary, int interleaved, int q_offset,     \
      int q_stride, int k_offset, int v_offset, float eps) {                         \
    const QnrDims dims{width, heads, kv_heads, d, rotary, interleaved, q_offset,     \
                       q_stride, k_offset, v_offset, rotary / 2};                    \
    qnr_head<T>(packed, q_gain, k_gain, cos_t, sin_t, positions, q, k, v, dims, eps); \
  }

QNR_EXPORT(f32, float)
QNR_EXPORT(f16, __half)
QNR_EXPORT(bf16, __nv_bfloat16)
