#include <cuda_bf16.h>
#include <cuda_fp16.h>

// SwiGLU over a packed [gate | up] projection (`kernels::swiglu`):
// out[r, i] = O(silu(float(gate)) * float(up)), silu(v) = v / (1 + exp(-v)),
// as the eager chain computes it in F32. Grid (ceil(width / 256), rows).

__device__ __forceinline__ float sg_load(const float* p, long long i) { return p[i]; }
__device__ __forceinline__ float sg_load(const __half* p, long long i) { return __half2float(p[i]); }
__device__ __forceinline__ float sg_load(const __nv_bfloat16* p, long long i) {
  return __bfloat162float(p[i]);
}
__device__ __forceinline__ void sg_store(float* p, long long i, float v) { p[i] = v; }
__device__ __forceinline__ void sg_store(__half* p, long long i, float v) { p[i] = __float2half(v); }
__device__ __forceinline__ void sg_store(__nv_bfloat16* p, long long i, float v) {
  p[i] = __float2bfloat16(v);
}

template <class I, class O>
__device__ void sg_row(const I* __restrict__ packed, O* __restrict__ out, int width) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= width) return;
  const long long row = blockIdx.y;
  const float g = sg_load(packed, row * 2 * width + i);
  const float u = sg_load(packed, row * 2 * width + width + i);
  sg_store(out, row * width + i, g / (1.0f + expf(-g)) * u);
}

#define SG_EXPORT(NAME, I, O)                                                      \
  extern "C" __global__ void izwi_swiglu_##NAME(const I* packed, O* out, int width) { \
    sg_row<I, O>(packed, out, width);                                              \
  }

SG_EXPORT(f32_f32, float, float)
SG_EXPORT(f16_f16, __half, __half)
SG_EXPORT(bf16_bf16, __nv_bfloat16, __nv_bfloat16)
SG_EXPORT(f32_f16, float, __half)
SG_EXPORT(f32_bf16, float, __nv_bfloat16)
