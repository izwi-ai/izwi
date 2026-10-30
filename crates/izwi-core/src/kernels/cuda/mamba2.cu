#include <math.h>

// Mamba2 selective scan (Nemotron-H Mamba2 mixers), F32. One block per
// (head, row); thread = p * lanes + lane holds COLS state columns of row p in
// registers, so the state never leaves the block between tokens. Each token:
// B and C of the head's group are staged in shared memory, then every thread
// updates its columns and the lanes of one p reduce y with warp shuffles.
//
//   dt = softplus(dt_raw + dt_bias)    (v > 20 gives v; no clamp)
//   S  = S * exp(A * dt) + (B * dt) * x
//   y  = sum_n S * C + D * x
//
// The input state is read-only and the output state a fresh buffer, so the
// caller's previous state stays a valid rollback checkpoint.

__device__ __forceinline__ float izwi_mamba2_softplus(float v) {
  if (v > 20.0f) return v;
  return log1pf(expf(v));
}

template <int COLS>
__device__ void izwi_mamba2_scan(const float* __restrict__ xbc,
                                 const float* __restrict__ dt,
                                 const float* __restrict__ params,
                                 const float* __restrict__ state_in,
                                 float* __restrict__ y,
                                 float* __restrict__ state_out,
                                 int tokens, int heads, int head_dim,
                                 int groups, int width, int lanes,
                                 int channels) {
  __shared__ float bc[512];
  const int head = blockIdx.x;
  const int row = blockIdx.y;
  const int tid = threadIdx.x;
  const int threads = blockDim.x;
  const int p = tid / lanes;
  const int lane = tid % lanes;
  const int g = head / (heads / groups);
  const int inner = heads * head_dim;
  const int n0 = lane * COLS;
  const long long state_base =
      ((static_cast<long long>(row) * heads + head) * head_dim + p) * width + n0;
  float s[COLS];
#pragma unroll
  for (int i = 0; i < COLS; ++i) s[i] = state_in[state_base + i];
  const float bias = params[head];
  const float a = params[heads + head];
  const float d = params[2 * heads + head];
  for (int t = 0; t < tokens; ++t) {
    const long long token = static_cast<long long>(row) * tokens + t;
    const long long base = token * channels;
    for (int i = tid; i < 2 * width; i += threads) {
      bc[i] = i < width ? xbc[base + inner + g * width + i]
                        : xbc[base + inner + (groups + g) * width + (i - width)];
    }
    __syncthreads();
    const float step = izwi_mamba2_softplus(dt[token * heads + head] + bias);
    const float decay = expf(a * step);
    const float x = xbc[base + head * head_dim + p];
    float acc = 0.0f;
#pragma unroll
    for (int i = 0; i < COLS; ++i) {
      s[i] = s[i] * decay + (bc[n0 + i] * step) * x;
      acc += s[i] * bc[width + n0 + i];
    }
    // The lanes of one p are consecutive and aligned inside a warp.
    for (int offset = lanes >> 1; offset > 0; offset >>= 1) {
      acc += __shfl_xor_sync(0xffffffffu, acc, offset);
    }
    if (lane == 0) y[token * inner + head * head_dim + p] = acc + d * x;
    __syncthreads();
  }
#pragma unroll
  for (int i = 0; i < COLS; ++i) state_out[state_base + i] = s[i];
}

#define IZWI_MAMBA2_SCAN(COLS)                                                   \
  extern "C" __global__ void izwi_mamba2_scan_c##COLS(                           \
      const float* __restrict__ xbc, const float* __restrict__ dt,               \
      const float* __restrict__ params, const float* __restrict__ state_in,      \
      float* __restrict__ y, float* __restrict__ state_out, int tokens,          \
      int heads, int head_dim, int groups, int width, int lanes,                 \
      int channels) {                                                            \
    izwi_mamba2_scan<COLS>(xbc, dt, params, state_in, y, state_out, tokens,      \
                           heads, head_dim, groups, width, lanes, channels);     \
  }

IZWI_MAMBA2_SCAN(8)
IZWI_MAMBA2_SCAN(16)
IZWI_MAMBA2_SCAN(32)
