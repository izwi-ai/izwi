#include <cuda_bf16.h>
#include <cuda_fp16.h>

// Bidirectional attention over a few positions (`kernels::cuda::small_attention`),
// straight from a fused projection: qkv is [batch, positions, 3, heads, dim]
// (each position's q, then k, then v), cos/sin [positions, dim / 2]. q and k
// take the interleaved rotation (pairs (2i, 2i + 1), as Candle's rope_i),
// scores and softmax run in F32, and out is [batch, positions, heads, dim].
// One block per (batch row, head), one thread per dimension; a row's values
// never depend on another row's.

#define SA_MAX_POSITIONS 4
#define SA_MAX_WARPS 8

__device__ __forceinline__ float sa_load(const float* p, long long i) { return p[i]; }
__device__ __forceinline__ float sa_load(const __half* p, long long i) { return __half2float(p[i]); }
__device__ __forceinline__ float sa_load(const __nv_bfloat16* p, long long i) {
  return __bfloat162float(p[i]);
}
__device__ __forceinline__ void sa_store(float* p, long long i, float v) { p[i] = v; }
__device__ __forceinline__ void sa_store(__half* p, long long i, float v) { p[i] = __float2half(v); }
__device__ __forceinline__ void sa_store(__nv_bfloat16* p, long long i, float v) {
  p[i] = __float2bfloat16(v);
}

template <class T>
__device__ void sa_attend(const T* __restrict__ qkv, const T* __restrict__ cos,
                          const T* __restrict__ sin, T* __restrict__ out, int positions,
                          int heads, int dim, float scale) {
  __shared__ float partial[SA_MAX_POSITIONS * SA_MAX_POSITIONS][SA_MAX_WARPS];
  __shared__ float scores[SA_MAX_POSITIONS * SA_MAX_POSITIONS];
  const long long row = blockIdx.x / heads;
  const int head = blockIdx.x % heads;
  const int t = threadIdx.x;
  const int warp = t >> 5, lane = t & 31, warps = (dim + 31) >> 5;
  const bool live = t < dim;
  const int half = dim >> 1, pair = t >> 1, mate = t ^ 1;
  const long long stride = 3LL * heads * dim;
  float q[SA_MAX_POSITIONS], k[SA_MAX_POSITIONS], v[SA_MAX_POSITIONS];
  for (int p = 0; p < positions; ++p) {
    q[p] = k[p] = v[p] = 0.0f;
    if (!live) continue;
    const long long base = (row * positions + p) * stride + (long long)head * dim;
    const long long kb = base + (long long)heads * dim, vb = kb + (long long)heads * dim;
    const float c = sa_load(cos, (long long)p * half + pair);
    const float s = sa_load(sin, (long long)p * half + pair);
    const float qx = sa_load(qkv, base + t), qm = sa_load(qkv, base + mate);
    const float kx = sa_load(qkv, kb + t), km = sa_load(qkv, kb + mate);
    // Even t: x[2i] c - x[2i+1] s; odd t: x[2i] s + x[2i+1] c.
    q[p] = (t & 1) ? qm * s + qx * c : qx * c - qm * s;
    k[p] = (t & 1) ? km * s + kx * c : kx * c - km * s;
    v[p] = sa_load(qkv, vb + t);
  }
  for (int i = 0; i < positions; ++i) {
    for (int j = 0; j < positions; ++j) {
      float sum = q[i] * k[j];
      for (int offset = 16; offset > 0; offset >>= 1)
        sum += __shfl_xor_sync(0xffffffffu, sum, offset);
      if (lane == 0) partial[i * positions + j][warp] = sum;
    }
  }
  __syncthreads();
  if (t < positions * positions) {
    float sum = 0.0f;
    for (int w = 0; w < warps; ++w) sum += partial[t][w];
    scores[t] = sum * scale;
  }
  __syncthreads();
  if (!live) return;
  for (int i = 0; i < positions; ++i) {
    float peak = -INFINITY;
    for (int j = 0; j < positions; ++j) peak = fmaxf(peak, scores[i * positions + j]);
    float total = 0.0f, acc = 0.0f;
    for (int j = 0; j < positions; ++j) {
      const float e = expf(scores[i * positions + j] - peak);
      total += e;
      acc += e * v[j];
    }
    sa_store(out, ((row * positions + i) * heads + head) * (long long)dim + t, acc / total);
  }
}

#define SA_EXPORT(NAME, T)                                                                 \
  extern "C" __global__ void izwi_small_rope_attention_##NAME(                              \
      const T* qkv, const T* cos, const T* sin, T* out, int positions, int heads, int dim, \
      float scale) {                                                                       \
    sa_attend<T>(qkv, cos, sin, out, positions, heads, dim, scale);                        \
  }

SA_EXPORT(f32, float)
SA_EXPORT(f16, __half)
SA_EXPORT(bf16, __nv_bfloat16)
