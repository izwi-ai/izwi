#include <math.h>

// Qwen3.8 single-token depthwise convolution. The packed result contains the
// activated output followed by the next three-slot history. Keeping both in one
// allocation lets the model stage the new transactional state without stacking
// three Candle tensors after every token.
extern "C" __global__ void qwen38_causal_conv_decode_f32(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ history,
    float* __restrict__ packed_output,
    int conv_dim) {
  const int gid = blockIdx.x * blockDim.x + threadIdx.x;
  const int total_elements = conv_dim * 4;
  if (gid >= total_elements) return;

  if (gid < conv_dim) {
    const int channel = gid;
    const int history_base = channel * 3;
    const int weight_base = channel * 4;
    float value = history[history_base] * weight[weight_base]
                + history[history_base + 1] * weight[weight_base + 1]
                + history[history_base + 2] * weight[weight_base + 2]
                + input[channel] * weight[weight_base + 3];
    packed_output[channel] = value / (1.0f + expf(-value));
    return;
  }

  const int state_index = gid - conv_dim;
  const int channel = state_index / 3;
  const int slot = state_index - channel * 3;
  const int history_base = channel * 3;
  packed_output[conv_dim + state_index] =
      slot < 2 ? history[history_base + slot + 1] : input[channel];
}
