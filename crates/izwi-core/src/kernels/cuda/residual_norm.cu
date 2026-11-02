#include <cuda_bf16.h>
#include <cuda_fp16.h>

// Residual add fused with RMSNorm (`kernels::residual_norm`), one block per
// row, any mix of F32/F16/BF16 residual, delta and output:
//
//   sum    = R(float(residual) + float(delta))        R: the residual's dtype
//   n      = O(float(sum))                            O: the output's dtype
//   normed = O(float(n) * rsqrt(mean(n * n) + eps) * gain)
//
// Without a delta, `sum` is the input itself and is not written. Each thread
// sums its strided columns in order and the block reduces in a fixed tree, so
// a row's bits depend on that row alone.

#define RN_THREADS 256

__device__ __forceinline__ float rn_load(const float* p, int i) { return p[i]; }
__device__ __forceinline__ float rn_load(const __half* p, int i) { return __half2float(p[i]); }
__device__ __forceinline__ float rn_load(const __nv_bfloat16* p, int i) {
  return __bfloat162float(p[i]);
}

__device__ __forceinline__ void rn_store(float* p, int i, float v) { p[i] = v; }
__device__ __forceinline__ void rn_store(__half* p, int i, float v) { p[i] = __float2half(v); }
__device__ __forceinline__ void rn_store(__nv_bfloat16* p, int i, float v) {
  p[i] = __float2bfloat16(v);
}

__device__ __forceinline__ float rn_round(float v, const float*) { return v; }
__device__ __forceinline__ float rn_round(float v, const __half*) {
  return __half2float(__float2half(v));
}
__device__ __forceinline__ float rn_round(float v, const __nv_bfloat16*) {
  return __bfloat162float(__float2bfloat16(v));
}

template <class R, class D, class O>
__device__ void rn_row(const R* __restrict__ residual, const D* __restrict__ delta,
                       const float* __restrict__ gain, R* __restrict__ sum,
                       O* __restrict__ out, int hidden, float eps) {
  __shared__ float partial[RN_THREADS];
  const long long base = (long long)blockIdx.x * hidden;
  const R* r = residual + base;
  const O* tag = nullptr;
  float acc = 0.0f;
  for (int i = threadIdx.x; i < hidden; i += RN_THREADS) {
    float s = rn_load(r, i);
    if (delta != nullptr) {
      s = rn_round(s + rn_load(delta + base, i), r);
      rn_store(sum + base, i, s);
    }
    const float n = rn_round(s, tag);
    acc += n * n;
  }
  partial[threadIdx.x] = acc;
  __syncthreads();
  for (int stride = RN_THREADS / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
    __syncthreads();
  }
  const float scale = rsqrtf(partial[0] / (float)hidden + eps);
  // Each thread rereads the columns it wrote itself.
  const R* s_row = delta != nullptr ? sum + base : r;
  for (int i = threadIdx.x; i < hidden; i += RN_THREADS) {
    const float n = rn_round(rn_load(s_row, i), tag);
    rn_store(out + base, i, n * scale * gain[i]);
  }
}

#define RN_EXPORT(NAME, R, D, O)                                                  \
  extern "C" __global__ void izwi_residual_norm_##NAME(                          \
      const R* residual, const D* delta, const float* gain, R* sum, O* out,      \
      int hidden, float eps) {                                                   \
    rn_row<R, D, O>(residual, delta, gain, sum, out, hidden, eps);               \
  }

// residual_delta_output
RN_EXPORT(f32_f32_f32, float, float, float)
RN_EXPORT(f32_f32_f16, float, float, __half)
RN_EXPORT(f32_f32_bf16, float, float, __nv_bfloat16)
RN_EXPORT(bf16_f32_bf16, __nv_bfloat16, float, __nv_bfloat16)
RN_EXPORT(bf16_bf16_bf16, __nv_bfloat16, __nv_bfloat16, __nv_bfloat16)
RN_EXPORT(f16_f32_f16, __half, float, __half)
RN_EXPORT(f16_f16_f16, __half, __half, __half)
