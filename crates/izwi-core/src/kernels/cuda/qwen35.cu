#include <math.h>

extern "C" __global__ void qwen35_causal_conv_sequence_f32(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ history,
    float* __restrict__ packed_output,
    int conv_dim,
    int sequence,
    int kernel_size,
    int output_elements,
    int total_elements) {
  const int gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid >= total_elements) return;

  const int history_len = kernel_size - 1;
  if (gid < output_elements) {
    const int token_idx = gid / conv_dim;
    const int channel_idx = gid - token_idx * conv_dim;
    float value = 0.0f;
    for (int tap = 0; tap < kernel_size; ++tap) {
      const int source_idx = token_idx + tap;
      const float source = source_idx < history_len
          ? history[channel_idx * history_len + source_idx]
          : input[(source_idx - history_len) * conv_dim + channel_idx];
      value += source * weight[channel_idx * kernel_size + tap];
    }
    packed_output[gid] = value / (1.0f + expf(-value));
    return;
  }

  const int state_idx = gid - output_elements;
  const int channel_idx = state_idx / history_len;
  const int history_idx = state_idx - channel_idx * history_len;
  const int source_idx = sequence + history_idx;
  packed_output[gid] = source_idx < history_len
      ? history[channel_idx * history_len + source_idx]
      : input[(source_idx - history_len) * conv_dim + channel_idx];
}

// Qwen3.5 Gated DeltaNet sequence recurrence for F32 tensors.
//
// One CUDA block owns one (batch, head) pair. Each lane owns one or more
// value columns, so the complete recurrent state remains on-device for the
// whole sequence and no synchronization is needed between value columns.
extern "C" __global__ void qwen35_gated_delta_sequence_f32(
    const float* __restrict__ qkv,
    const float* __restrict__ gates,
    const float* __restrict__ initial_state,
    float* __restrict__ packed_output,
    int batch,
    int sequence,
    int heads,
    int key_dim,
    int value_dim) {
  const int batch_head = blockIdx.x;
  if (batch_head >= batch * heads) {
    return;
  }
  const int batch_idx = batch_head / heads;
  const int head_idx = batch_head - batch_idx * heads;
  const int output_elements = batch * sequence * heads * value_dim;
  float* state = packed_output + output_elements;
  const int state_base = batch_head * key_dim * value_dim;

  for (int index = threadIdx.x; index < key_dim * value_dim;
       index += blockDim.x) {
    state[state_base + index] = initial_state[state_base + index];
  }
  __syncthreads();

  const float query_scale = rsqrtf((float)key_dim);
  const int qkv_width = key_dim * 2 + value_dim;
  for (int value_idx = threadIdx.x; value_idx < value_dim;
       value_idx += blockDim.x) {
    for (int token_idx = 0; token_idx < sequence; ++token_idx) {
      const int token_head = (batch_idx * sequence + token_idx) * heads + head_idx;
      const int qkv_base = token_head * qkv_width;
      const int gate_base = token_head * 2;
      const float decay = expf(gates[gate_base]);
      const float beta = gates[gate_base + 1];

      float recalled_value = 0.0f;
      for (int key_idx = 0; key_idx < key_dim; ++key_idx) {
        const int state_idx = state_base + key_idx * value_dim + value_idx;
        recalled_value += qkv[qkv_base + key_dim + key_idx]
                            * (decay * state[state_idx]);
      }
      const float delta =
          (qkv[qkv_base + key_dim * 2 + value_idx] - recalled_value) * beta;

      float result = 0.0f;
      for (int key_idx = 0; key_idx < key_dim; ++key_idx) {
        const int state_idx = state_base + key_idx * value_dim + value_idx;
        const float next_state = decay * state[state_idx]
                                 + qkv[qkv_base + key_dim + key_idx] * delta;
        state[state_idx] = next_state;
        result += qkv[qkv_base + key_idx] * query_scale * next_state;
      }
      packed_output[token_head * value_dim + value_idx] = result;
    }
  }
}
