#include <cuda_bf16.h>
#include <cuda_fp16.h>

// Depthwise (groups == channels) stride-1 1-D convolution in one launch
// (`kernels::depthwise_conv`), one thread per output element:
//
//   y[b, c, t] = O(sum_k float(x[b, c, t + k * dilation - pad_left]) * float(w[c, k])
//                  + float(bias[c]))
//
// with x read as zero outside [0, in_len). The sum runs in F32 in tap order
// with every product and every partial sum rounded on its own (__fmul_rn and
// __fadd_rn are never contracted into an FMA), the first tap's product is the
// start value and the bias comes last: the operations, and the order, of the
// shift-and-accumulate composition's broadcast_mul / add / broadcast_add, so
// the two agree bit for bit. Padded taps multiply a zero as the composition
// does, so signed zeros and non-finite weights propagate the same way.
//
// x is addressed through element strides (any view, channels-first or
// channels-last); out is contiguous in that layout. bias may be null.
// 1-D grid of ceil(total / 256) blocks of 256 threads.

__device__ __forceinline__ float dw_load(const float* p, long long i) { return p[i]; }
__device__ __forceinline__ float dw_load(const __half* p, long long i) { return __half2float(p[i]); }
__device__ __forceinline__ float dw_load(const __nv_bfloat16* p, long long i) {
  return __bfloat162float(p[i]);
}

__device__ __forceinline__ void dw_store(float* p, long long i, float v) { p[i] = v; }
__device__ __forceinline__ void dw_store(__half* p, long long i, float v) { p[i] = __float2half(v); }
__device__ __forceinline__ void dw_store(__nv_bfloat16* p, long long i, float v) {
  p[i] = __float2bfloat16_rn(v);
}

template <class X, class W>
__device__ void dw_conv(const X* __restrict__ x, const W* __restrict__ weight,
                        const W* __restrict__ bias, X* __restrict__ out, int channels,
                        int in_len, int out_len, int taps, int dilation, int pad_left,
                        long long stride_b, long long stride_c, long long stride_t,
                        int channels_last, long long total) {
  const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= total) return;
  long long b, c, t;
  if (channels_last) {
    c = i % channels;
    const long long bt = i / channels;
    t = bt % out_len;
    b = bt / out_len;
  } else {
    t = i % out_len;
    const long long bc = i / out_len;
    c = bc % channels;
    b = bc / channels;
  }
  const X* row = x + b * stride_b + c * stride_c;
  const W* w = weight + c * taps;
  float acc = 0.0f;
  for (int k = 0; k < taps; ++k) {
    const long long s = t + (long long)k * dilation - pad_left;
    const float v = (s >= 0 && s < in_len) ? dw_load(row, s * stride_t) : 0.0f;
    const float term = __fmul_rn(v, dw_load(w, k));
    acc = k == 0 ? term : __fadd_rn(acc, term);
  }
  if (bias != nullptr) acc = __fadd_rn(acc, dw_load(bias, c));
  dw_store(out, i, acc);
}

#define DW_EXPORT(NAME, X, W)                                                          \
  extern "C" __global__ void izwi_depthwise_conv1d_##NAME(                             \
      const X* x, const W* weight, const W* bias, X* out, int channels, int in_len,   \
      int out_len, int taps, int dilation, int pad_left, long long stride_b,          \
      long long stride_c, long long stride_t, int channels_last, long long total) {    \
    dw_conv<X, W>(x, weight, bias, out, channels, in_len, out_len, taps, dilation,     \
                  pad_left, stride_b, stride_c, stride_t, channels_last, total);       \
  }

// activation_weight (the output takes the activation's dtype)
DW_EXPORT(f32_f32, float, float)
DW_EXPORT(f16_f16, __half, __half)
DW_EXPORT(bf16_bf16, __nv_bfloat16, __nv_bfloat16)
DW_EXPORT(f16_f32, __half, float)
DW_EXPORT(bf16_f32, __nv_bfloat16, float)
