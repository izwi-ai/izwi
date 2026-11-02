#include <cuda_bf16.h>
#include <cuda_fp16.h>

// Row-invariant multi-row GEMV: y[r, c] = sum_k x[r, k] * W[c, k] for up to
// IZWI_ROW_GEMV_MAX_ROWS rows of x in one launch, with W dense (the
// activations' 16-bit type or F32) or E4M3FN with one F32 scale per 128x128
// block.
//
// One warp owns one output channel and reads its weight row exactly once per
// call: each lane loads IZWI_ROW_GEMV_CHUNK consecutive weights at a time,
// lane-strided over K, and folds them into every row's accumulator before
// moving on. A row's arithmetic never involves another row or the row count:
//
//   part = fma chain over the chunk's 8 products, in k order
//   acc  = fma(part, block scale (1 for dense weights), acc), chunk by chunk
//   y    = the warp's xor butterfly over the 32 lane sums, lane 0's result
//
// so row r's bits are the same whatever the number of rows beside it. Every
// multiply-add is an explicit fmaf, so FP contraction cannot reorder it.
// Two accumulator capacities (8 and 64 rows) bound the register file for
// small calls; the per-row arithmetic is identical in both.
//
// Contract (checked on the host): K a multiple of 8 (of 128 for E4M3),
// 16-byte aligned rows of x and W, F32 scales [ceil(N / 128), K / 128].

#define IZWI_ROW_GEMV_MAX_ROWS 64
#define IZWI_ROW_GEMV_WARPS 8
#define IZWI_ROW_GEMV_CHUNK 8
#define IZWI_ROW_GEMV_BLOCK 128

__device__ __forceinline__ float izwi_row_gemv_float(float value) { return value; }
__device__ __forceinline__ float izwi_row_gemv_float(__half value) {
  return __half2float(value);
}
__device__ __forceinline__ float izwi_row_gemv_float(__nv_bfloat16 value) {
  return __bfloat162float(value);
}

template <typename T>
__device__ __forceinline__ T izwi_row_gemv_store(float value);
template <>
__device__ __forceinline__ float izwi_row_gemv_store<float>(float value) {
  return value;
}
template <>
__device__ __forceinline__ __half izwi_row_gemv_store<__half>(float value) {
  return __float2half_rn(value);
}
template <>
__device__ __forceinline__ __nv_bfloat16 izwi_row_gemv_store<__nv_bfloat16>(float value) {
  return __float2bfloat16_rn(value);
}

// Exact E4M3FN decode: sign, 4 exponent bits (bias 7), 3 mantissa bits;
// exponent 0 is subnormal (m * 2^-9) and S.1111.111 is NaN.
__device__ __forceinline__ float izwi_row_gemv_e4m3(unsigned int byte) {
  const unsigned int exponent = (byte >> 3) & 15u;
  const unsigned int mantissa = byte & 7u;
  float value = exponent == 0u
                    ? static_cast<float>(mantissa) * 0.001953125f
                    : __uint_as_float(((exponent + 120u) << 23) | (mantissa << 20));
  if (exponent == 15u && mantissa == 7u) {
    value = __uint_as_float(0x7fc00000u);
  }
  return (byte & 128u) ? -value : value;
}

// Eight consecutive 16-bit values as floats: one 16-byte load.
template <typename T>
__device__ __forceinline__ void izwi_row_gemv_load8(const T* source, float* out) {
  const uint4 raw = *reinterpret_cast<const uint4*>(source);
  const T* lanes = reinterpret_cast<const T*>(&raw);
#pragma unroll
  for (int i = 0; i < IZWI_ROW_GEMV_CHUNK; ++i) {
    out[i] = izwi_row_gemv_float(lanes[i]);
  }
}

// Eight consecutive F32 weights: two 16-byte loads.
template <>
__device__ __forceinline__ void izwi_row_gemv_load8<float>(const float* source, float* out) {
  const uint4 low = *reinterpret_cast<const uint4*>(source);
  const uint4 high = *reinterpret_cast<const uint4*>(source + 4);
  out[0] = __uint_as_float(low.x);
  out[1] = __uint_as_float(low.y);
  out[2] = __uint_as_float(low.z);
  out[3] = __uint_as_float(low.w);
  out[4] = __uint_as_float(high.x);
  out[5] = __uint_as_float(high.y);
  out[6] = __uint_as_float(high.z);
  out[7] = __uint_as_float(high.w);
}

// Eight consecutive E4M3FN weights, decoded exactly: one 8-byte load.
template <>
__device__ __forceinline__ void izwi_row_gemv_load8<unsigned char>(const unsigned char* source,
                                                                   float* out) {
  const uint2 raw = *reinterpret_cast<const uint2*>(source);
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    out[i] = izwi_row_gemv_e4m3((raw.x >> (8 * i)) & 0xffu);
    out[4 + i] = izwi_row_gemv_e4m3((raw.y >> (8 * i)) & 0xffu);
  }
}

template <typename TX, typename TW, typename TY, int ROWS>
__device__ void izwi_row_gemv(const TX* __restrict__ x,
                              const TW* __restrict__ weights,
                              const float* __restrict__ scales,
                              TY* __restrict__ y,
                              int rows,
                              int channels,
                              int width) {
  const int lane = threadIdx.x & 31;
  const int channel = blockIdx.x * IZWI_ROW_GEMV_WARPS + (threadIdx.x >> 5);
  // Uniform across the warp: a warp past the last channel has nothing to do.
  if (channel >= channels || rows <= 0 || rows > ROWS) {
    return;
  }
  const TW* row = weights + static_cast<size_t>(channel) * width;
  const float* row_scales =
      scales == nullptr
          ? nullptr
          : scales + static_cast<size_t>(channel / IZWI_ROW_GEMV_BLOCK) *
                         (width / IZWI_ROW_GEMV_BLOCK);
  float acc[ROWS];
#pragma unroll
  for (int r = 0; r < ROWS; ++r) {
    acc[r] = 0.0f;
  }
  for (int k0 = lane * IZWI_ROW_GEMV_CHUNK; k0 < width; k0 += 32 * IZWI_ROW_GEMV_CHUNK) {
    float w[IZWI_ROW_GEMV_CHUNK];
    izwi_row_gemv_load8<TW>(row + k0, w);
    const float scale = row_scales == nullptr ? 1.0f : row_scales[k0 / IZWI_ROW_GEMV_BLOCK];
#pragma unroll
    for (int r = 0; r < ROWS; ++r) {
      if (r < rows) {
        float a[IZWI_ROW_GEMV_CHUNK];
        izwi_row_gemv_load8<TX>(x + static_cast<size_t>(r) * width + k0, a);
        float part = 0.0f;
#pragma unroll
        for (int i = 0; i < IZWI_ROW_GEMV_CHUNK; ++i) {
          part = fmaf(a[i], w[i], part);
        }
        acc[r] = fmaf(part, scale, acc[r]);
      }
    }
  }
#pragma unroll
  for (int r = 0; r < ROWS; ++r) {
    if (r < rows) {
      float total = acc[r];
#pragma unroll
      for (int offset = 16; offset > 0; offset >>= 1) {
        total += __shfl_xor_sync(0xffffffffu, total, offset);
      }
      if (lane == 0) {
        y[static_cast<size_t>(r) * channels + channel] = izwi_row_gemv_store<TY>(total);
      }
    }
  }
}

#define IZWI_DEFINE_ROW_GEMV(XS, TX, WS, TW, YS, TY, ROWS)                       \
  extern "C" __global__ void __launch_bounds__(IZWI_ROW_GEMV_WARPS * 32)       \
      izwi_row_gemv_##XS##_##WS##_##YS##_r##ROWS(                              \
          const TX* x, const TW* weights, const float* scales, TY* y,          \
          int rows, int channels, int width) {                                 \
    izwi_row_gemv<TX, TW, TY, ROWS>(x, weights, scales, y, rows, channels,     \
                                    width);                                    \
  }

#define IZWI_DEFINE_ROW_GEMV_ROWS(XS, TX, WS, TW, YS, TY) \
  IZWI_DEFINE_ROW_GEMV(XS, TX, WS, TW, YS, TY, 8)          \
  IZWI_DEFINE_ROW_GEMV(XS, TX, WS, TW, YS, TY, 64)

#define IZWI_DEFINE_ROW_GEMV_OUTPUTS(XS, TX, WS, TW) \
  IZWI_DEFINE_ROW_GEMV_ROWS(XS, TX, WS, TW, XS, TX)   \
  IZWI_DEFINE_ROW_GEMV_ROWS(XS, TX, WS, TW, f32, float)

IZWI_DEFINE_ROW_GEMV_OUTPUTS(f16, __half, f16, __half)
IZWI_DEFINE_ROW_GEMV_OUTPUTS(f16, __half, f32, float)
IZWI_DEFINE_ROW_GEMV_OUTPUTS(f16, __half, e4m3, unsigned char)
IZWI_DEFINE_ROW_GEMV_OUTPUTS(bf16, __nv_bfloat16, bf16, __nv_bfloat16)
IZWI_DEFINE_ROW_GEMV_OUTPUTS(bf16, __nv_bfloat16, f32, float)
IZWI_DEFINE_ROW_GEMV_OUTPUTS(bf16, __nv_bfloat16, e4m3, unsigned char)
