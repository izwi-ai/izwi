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

// One decode step of the causal depthwise convolution with bias and SiLU
// (`kernels::mamba2::conv_step_silu`), read straight from the in_proj output
// `projected` [rows, width]: xBC at `xbc_offset`, dt at `dt_offset`. Per
// channel, in tap order as the eager chain adds them:
//   v = (((bias + h0 * w0) + h1 * w1) + ...) + x * w[K-1];  u = v / (1 + exp(-v))
// The next history is the last K-1 pre-convolution rows; dt is copied out
// contiguous for the scan. The products and sums round separately (no FMA
// contraction), as the composed ops do. Grid (ceil(C / 256), rows).
extern "C" __global__ void izwi_mamba2_conv_step_silu(
    const float* __restrict__ projected, const float* __restrict__ history,
    const float* __restrict__ weight, const float* __restrict__ bias,
    float* __restrict__ u, float* __restrict__ dt, float* __restrict__ next,
    int width, int xbc_offset, int dt_offset, int channels, int taps, int heads) {
  const int row = blockIdx.y;
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  const long long in = (long long)row * width;
  const long long hist = (long long)row * (taps - 1) * channels;
  if (c < channels) {
    float v = bias[c];
    for (int tap = 0; tap < taps - 1; ++tap) {
      const float h = history[hist + (long long)tap * channels + c];
      v = __fadd_rn(v, __fmul_rn(h, weight[c * taps + tap]));
      if (tap > 0) next[hist + (long long)(tap - 1) * channels + c] = h;
    }
    const float x = projected[in + xbc_offset + c];
    v = __fadd_rn(v, __fmul_rn(x, weight[c * taps + taps - 1]));
    next[hist + (long long)(taps - 2) * channels + c] = x;
    u[(long long)row * channels + c] = v / (1.0f + expf(-v));
  }
  if (c < heads) dt[(long long)row * heads + c] = projected[in + dt_offset + c];
}

// Mamba2's gated grouped RMSNorm for decode rows
// (`kernels::mamba2::gated_group_rms_norm_step`): g = y * silu(z) with z read
// from `projected` at `z_offset`, then per group of `group` channels
// out = g / sqrt(mean(g * g) + eps) * weight. One block of 256 per
// (row, group), group <= 1024.
extern "C" __global__ void izwi_mamba2_gated_norm_step(
    const float* __restrict__ y, const float* __restrict__ projected,
    const float* __restrict__ weight, float* __restrict__ out, int width,
    int z_offset, int inner, int group, float eps) {
  __shared__ float gated[1024];
  __shared__ float partial[256];
  const int row = blockIdx.y, g0 = blockIdx.x * group;
  const long long yrow = (long long)row * inner;
  const long long zrow = (long long)row * width + z_offset;
  float acc = 0.0f;
  for (int i = threadIdx.x; i < group; i += 256) {
    const float z = projected[zrow + g0 + i];
    const float v = y[yrow + g0 + i] * (z / (1.0f + expf(-z)));
    gated[i] = v;
    acc += v * v;
  }
  partial[threadIdx.x] = acc;
  __syncthreads();
  for (int stride = 128; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
    __syncthreads();
  }
  const float rms = sqrtf(partial[0] / (float)group + eps);
  for (int i = threadIdx.x; i < group; i += 256) {
    out[yrow + g0 + i] = gated[i] / rms * weight[g0 + i];
  }
}
