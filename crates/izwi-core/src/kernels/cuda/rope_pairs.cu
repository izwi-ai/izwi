#include <cuda_bf16.h>
#include <cuda_fp16.h>

// Interleaved-pair rotary embedding over [batch, sequence, heads, dim]
// (`kernels::cuda::rope_pairs`): pairs (2i, 2i + 1) of position t rotate by
// cos/sin[t, i] (F32), in F32, each output rounded once to the input's dtype:
//   out[2i]     = x[2i] * c - x[2i+1] * s
//   out[2i + 1] = x[2i] * s + x[2i+1] * c
// One thread per pair; no layout change, so no transposes around it.

__device__ __forceinline__ float rp_load(const float* p, long long i) { return p[i]; }
__device__ __forceinline__ float rp_load(const __half* p, long long i) { return __half2float(p[i]); }
__device__ __forceinline__ float rp_load(const __nv_bfloat16* p, long long i) {
  return __bfloat162float(p[i]);
}
__device__ __forceinline__ void rp_store(float* p, long long i, float v) { p[i] = v; }
__device__ __forceinline__ void rp_store(__half* p, long long i, float v) { p[i] = __float2half(v); }
__device__ __forceinline__ void rp_store(__nv_bfloat16* p, long long i, float v) {
  p[i] = __float2bfloat16(v);
}

template <class T>
__device__ void rp_rotate(const T* __restrict__ x, const float* __restrict__ cos,
                          const float* __restrict__ sin, T* __restrict__ out,
                          long long tokens, int sequence, int heads, int half) {
  const long long total = tokens * heads * half;
  for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < total;
       i += (long long)blockDim.x * gridDim.x) {
    const int pair = (int)(i % half);
    const long long rest = i / half;
    const long long token = rest / heads;
    const int t = (int)(token % sequence);
    const long long at = 2 * i;
    const float x0 = rp_load(x, at), x1 = rp_load(x, at + 1);
    const float c = cos[(long long)t * half + pair], s = sin[(long long)t * half + pair];
    rp_store(out, at, x0 * c - x1 * s);
    rp_store(out, at + 1, x0 * s + x1 * c);
  }
}

#define RP_EXPORT(NAME, T)                                                                   \
  extern "C" __global__ void izwi_rope_pairs_##NAME(const T* x, const float* cos,             \
                                                    const float* sin, T* out,                 \
                                                    long long tokens, int sequence, int heads, \
                                                    int half) {                               \
    rp_rotate<T>(x, cos, sin, out, tokens, sequence, heads, half);                            \
  }

RP_EXPORT(f32, float)
RP_EXPORT(f16, __half)
RP_EXPORT(bf16, __nv_bfloat16)
