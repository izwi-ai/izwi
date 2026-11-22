#include <cuda_bf16.h>
#include <cuda_fp16.h>

// In place over a contiguous [rows, cols] output (`kernels::cuda::bias_rows`):
// y[r, c] = O(float(y[r, c]) + float(bias[c])), then, with gelu, Candle's
// exact GELU x * normcdf(x), all in F32. The geometry rides as arguments, so
// a launch uploads nothing (a Candle broadcast add uploads its strides).

__device__ __forceinline__ float br_load(const float* p, long long i) { return p[i]; }
__device__ __forceinline__ float br_load(const __half* p, long long i) { return __half2float(p[i]); }
__device__ __forceinline__ float br_load(const __nv_bfloat16* p, long long i) {
  return __bfloat162float(p[i]);
}
__device__ __forceinline__ void br_store(float* p, long long i, float v) { p[i] = v; }
__device__ __forceinline__ void br_store(__half* p, long long i, float v) { p[i] = __float2half(v); }
__device__ __forceinline__ void br_store(__nv_bfloat16* p, long long i, float v) {
  p[i] = __float2bfloat16(v);
}

template <class T>
__device__ void br_rows(T* __restrict__ y, const T* __restrict__ bias, long long rows,
                        long long cols, int gelu) {
  const long long total = rows * cols;
  for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < total;
       i += (long long)blockDim.x * gridDim.x) {
    // Round the sum to the output's dtype first, as the eager add did.
    br_store(y, i, br_load(y, i) + br_load(bias, i % cols));
    if (gelu) {
      const float v = br_load(y, i);
      br_store(y, i, v * normcdff(v));
    }
  }
}

#define BR_EXPORT(NAME, T)                                                              \
  extern "C" __global__ void izwi_bias_rows_##NAME(T* y, const T* bias, long long rows, \
                                                   long long cols, int gelu) {         \
    br_rows<T>(y, bias, rows, cols, gelu);                                              \
  }

BR_EXPORT(f32, float)
BR_EXPORT(f16, __half)
BR_EXPORT(bf16, __nv_bfloat16)
