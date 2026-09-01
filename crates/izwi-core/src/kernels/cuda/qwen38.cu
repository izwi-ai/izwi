#include <math.h>

// Qwen3.8 decode-only elementwise epilogues. These symbols intentionally live
// in the model-family source so they can evolve independently of Qwen3.5.
extern "C" __global__ void qwen38_silu_mul_decode_f32(
    const float* __restrict__ gate,
    const float* __restrict__ up,
    float* __restrict__ output,
    int elements) {
  const int gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid >= elements) return;
  const float value = gate[gid];
  output[gid] = (value / (1.0f + expf(-value))) * up[gid];
}

extern "C" __global__ void qwen38_l2_norm_decode_f32(
    const float* __restrict__ input,
    float* __restrict__ output,
    int hidden_dim,
    float eps) {
  const int row = blockIdx.x;
  const int base = row * hidden_dim;
  extern __shared__ float reduction[];
  float squares = 0.0f;
  for (int column = threadIdx.x; column < hidden_dim; column += blockDim.x) {
    const float value = input[base + column];
    squares += value * value;
  }
  reduction[threadIdx.x] = squares;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      reduction[threadIdx.x] += reduction[threadIdx.x + stride];
    }
    __syncthreads();
  }
  const float inverse_norm = rsqrtf(reduction[0] + eps);
  for (int column = threadIdx.x; column < hidden_dim; column += blockDim.x) {
    output[base + column] = input[base + column] * inverse_norm;
  }
}

extern "C" __global__ void qwen38_gated_rms_norm_decode_f32(
    const float* __restrict__ hidden,
    const float* __restrict__ gate,
    const float* __restrict__ weight,
    float* __restrict__ output,
    int hidden_dim,
    float eps) {
  const int row = blockIdx.x;
  const int base = row * hidden_dim;
  extern __shared__ float reduction[];
  float squares = 0.0f;
  for (int column = threadIdx.x; column < hidden_dim; column += blockDim.x) {
    const float value = hidden[base + column];
    squares += value * value;
  }
  reduction[threadIdx.x] = squares;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      reduction[threadIdx.x] += reduction[threadIdx.x + stride];
    }
    __syncthreads();
  }
  const float inverse_rms = rsqrtf(reduction[0] / (float)hidden_dim + eps);
  for (int column = threadIdx.x; column < hidden_dim; column += blockDim.x) {
    const int index = base + column;
    const float gate_value = gate[index];
    const float silu_gate = gate_value / (1.0f + expf(-gate_value));
    output[index] = hidden[index] * inverse_rms * weight[column] * silu_gate;
  }
}

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

// Qwen3.8 single-token Gated DeltaNet recurrence for F32 tensors.
//
// Unlike the shared sequence kernel, this decode-specialized ABI consumes the
// native mixed-QKV layout. A value head maps directly to its repeated key head,
// so decode does not materialize expanded query/key tensors or a concatenated
// QKV tensor. The old state remains read-only and the next state is written to
// a fresh packed allocation, preserving transactional state publication.
extern "C" __global__ void qwen38_deltanet_decode_f32(
    const float* __restrict__ mixed_qkv,
    const float* __restrict__ gates,
    const float* __restrict__ initial_state,
    float* __restrict__ packed_output,
    int key_heads,
    int value_heads,
    int key_dim,
    int value_dim) {
  const int value_head = blockIdx.x;
  if (value_head >= value_heads) return;

  const int repeats = value_heads / key_heads;
  const int key_head = value_head / repeats;
  const int key_width = key_heads * key_dim;
  const int value_width = value_heads * value_dim;
  const int query_base = key_head * key_dim;
  const int key_base = key_width + key_head * key_dim;
  const int value_base = key_width * 2 + value_head * value_dim;
  const int state_base = value_head * key_dim * value_dim;
  float* next_state = packed_output + value_width;

  __shared__ float query_scale_shared;
  __shared__ float key_norm_shared;
  if (threadIdx.x == 0) {
    float query_squares = 0.0f;
    float key_squares = 0.0f;
    for (int key_idx = 0; key_idx < key_dim; ++key_idx) {
      const float query_value = mixed_qkv[query_base + key_idx];
      const float key_value = mixed_qkv[key_base + key_idx];
      query_squares += query_value * query_value;
      key_squares += key_value * key_value;
    }
    query_scale_shared = rsqrtf(query_squares + 1.0e-6f)
                         * rsqrtf((float)key_dim);
    key_norm_shared = rsqrtf(key_squares + 1.0e-6f);
  }
  __syncthreads();
  const float query_scale = query_scale_shared;
  const float key_norm = key_norm_shared;
  const float decay = expf(gates[value_head * 2]);
  const float beta = gates[value_head * 2 + 1];

  for (int value_idx = threadIdx.x; value_idx < value_dim;
       value_idx += blockDim.x) {
    float recalled_value = 0.0f;
    for (int key_idx = 0; key_idx < key_dim; ++key_idx) {
      const int state_idx = state_base + key_idx * value_dim + value_idx;
      recalled_value += mixed_qkv[key_base + key_idx] * key_norm
                        * (decay * initial_state[state_idx]);
    }
    const float delta =
        (mixed_qkv[value_base + value_idx] - recalled_value) * beta;

    float result = 0.0f;
    for (int key_idx = 0; key_idx < key_dim; ++key_idx) {
      const int state_idx = state_base + key_idx * value_dim + value_idx;
      const float normalized_key = mixed_qkv[key_base + key_idx] * key_norm;
      const float updated = decay * initial_state[state_idx]
                            + normalized_key * delta;
      next_state[state_idx] = updated;
      result += mixed_qkv[query_base + key_idx] * query_scale * updated;
    }
    packed_output[value_head * value_dim + value_idx] = result;
  }
}
