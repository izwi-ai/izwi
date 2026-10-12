#include <math.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

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
  output[gid] = (value / (1.0f + expf(-value))) * float(up[gid]);
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
    output[base + column] = float(input[base + column]) * inverse_norm;
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
    output[index] = float(hidden[index]) * inverse_rms * float(weight[column]) * silu_gate;
  }
}


// Qwen3.8 decode-only elementwise epilogues. These symbols intentionally live
// in the model-family source so they can evolve independently of Qwen3.5.
extern "C" __global__ void qwen38_silu_mul_decode_f16(
    const __half* __restrict__ gate,
    const __half* __restrict__ up,
    __half* __restrict__ output,
    int elements) {
  const int gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid >= elements) return;
  const float value = gate[gid];
  output[gid] = (value / (1.0f + expf(-value))) * float(up[gid]);
}

extern "C" __global__ void qwen38_l2_norm_decode_f16(
    const __half* __restrict__ input,
    __half* __restrict__ output,
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
    output[base + column] = float(input[base + column]) * inverse_norm;
  }
}

extern "C" __global__ void qwen38_gated_rms_norm_decode_f16(
    const __half* __restrict__ hidden,
    const __half* __restrict__ gate,
    const __half* __restrict__ weight,
    __half* __restrict__ output,
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
    output[index] = float(hidden[index]) * inverse_rms * float(weight[column]) * silu_gate;
  }
}


// Qwen3.8 decode-only elementwise epilogues. These symbols intentionally live
// in the model-family source so they can evolve independently of Qwen3.5.
extern "C" __global__ void qwen38_silu_mul_decode_bf16(
    const __nv_bfloat16* __restrict__ gate,
    const __nv_bfloat16* __restrict__ up,
    __nv_bfloat16* __restrict__ output,
    int elements) {
  const int gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid >= elements) return;
  const float value = gate[gid];
  output[gid] = (value / (1.0f + expf(-value))) * float(up[gid]);
}

extern "C" __global__ void qwen38_l2_norm_decode_bf16(
    const __nv_bfloat16* __restrict__ input,
    __nv_bfloat16* __restrict__ output,
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
    output[base + column] = float(input[base + column]) * inverse_norm;
  }
}

extern "C" __global__ void qwen38_gated_rms_norm_decode_bf16(
    const __nv_bfloat16* __restrict__ hidden,
    const __nv_bfloat16* __restrict__ gate,
    const __nv_bfloat16* __restrict__ weight,
    __nv_bfloat16* __restrict__ output,
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
    output[index] = float(hidden[index]) * inverse_rms * float(weight[column]) * silu_gate;
  }
}


// Qwen3.8 single-token depthwise convolution. The packed result contains the
// activated output followed by the next three-slot history. Keeping both in one
// allocation lets the model stage the new transactional state without stacking
// three Candle tensors after every token. The input is the QKV projection row in
// its own dtype (F32, or BF16/F16 straight from the projection, saving the cast);
// the weight, history and packed output stay F32. A 16-bit input converts
// exactly to F32, so every entry computes the same values as the F32 entry on
// the cast input.
// One thread per output element: the first `conv_dim` threads write the
// activated output, the rest the next three-slot history. The direct entries
// pass one packed buffer for both; the table-addressed entries below pass the
// history addresses from a device state table. Both run this exact body.
template <class T>
__device__ void qwen38_causal_conv_decode_into(
    const T* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ history,
    float* __restrict__ output,
    float* __restrict__ next_history,
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
                + float(input[channel]) * weight[weight_base + 3];
    output[channel] = value / (1.0f + expf(-value));
    return;
  }

  const int state_index = gid - conv_dim;
  const int channel = state_index / 3;
  const int slot = state_index - channel * 3;
  const int history_base = channel * 3;
  next_history[state_index] =
      slot < 2 ? history[history_base + slot + 1] : float(input[channel]);
}

template <class T>
__device__ void qwen38_causal_conv_decode(
    const T* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ history,
    float* __restrict__ packed_output,
    int conv_dim) {
  qwen38_causal_conv_decode_into<T>(
      input, weight, history, packed_output, packed_output + conv_dim, conv_dim);
}

extern "C" __global__ void qwen38_causal_conv_decode_f32(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ history,
    float* __restrict__ packed_output,
    int conv_dim) {
  qwen38_causal_conv_decode<float>(input, weight, history, packed_output, conv_dim);
}

extern "C" __global__ void qwen38_causal_conv_decode_bf16(
    const __nv_bfloat16* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ history,
    float* __restrict__ packed_output,
    int conv_dim) {
  qwen38_causal_conv_decode<__nv_bfloat16>(input, weight, history, packed_output, conv_dim);
}

extern "C" __global__ void qwen38_causal_conv_decode_f16(
    const __half* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ history,
    float* __restrict__ packed_output,
    int conv_dim) {
  qwen38_causal_conv_decode<__half>(input, weight, history, packed_output, conv_dim);
}

// Table-addressed entries for CUDA graph replay. A captured graph bakes kernel
// arguments, but each decode step reads the previous history and writes a
// fresh one at new addresses. These entries read both from the per-layer
// device state table the host refreshes before each replay (the
// `kernels/cuda/gdn.rs` StateTable layout, six entries per layer):
//   table[slot + 0] : history [C, 3] F32, oldest first (read)
//   table[slot + 3] : next history [C, 3] F32 (written)
// Entries 1 and 2 are unused here; 4 and 5 address the recurrent state for
// the table-addressed recurrence. The activated output goes to `output`.
#define QWEN38_CONV_TABLE_EXPORT(T, S)                                                   \
  extern "C" __global__ void qwen38_causal_conv_decode_table_##S(                        \
      const T* __restrict__ input, const float* __restrict__ weight,                     \
      const unsigned long long* __restrict__ table, int slot, float* __restrict__ output, \
      int conv_dim) {                                                                    \
    const unsigned long long* entries = table + slot;                                    \
    qwen38_causal_conv_decode_into<T>(input, weight, (const float*)entries[0], output,   \
                                      (float*)entries[3], conv_dim);                     \
  }

QWEN38_CONV_TABLE_EXPORT(float, f32)
QWEN38_CONV_TABLE_EXPORT(__nv_bfloat16, bf16)
QWEN38_CONV_TABLE_EXPORT(__half, f16)

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

// Qwen3.8 Gated DeltaNet prefill: one launch runs a chunk's whole token loop
// on the conv output's native grouped layout.
//
// Register-resident (the design of llama.cpp's and MLX's GDN kernels): eight
// lanes own one value column of one value head, each holding 16 of the 128
// key rows of that column's state in registers for the whole token loop; the
// two dot products per token are 8-lane shuffle reductions, so the state is
// read once and written once per launch. Value head v reads key head
// v / (value_heads / key_heads) (HF grouped order): no expanded q/k exists.
// q and k are L2-normalized in the kernel, x * rsqrt(sum x^2 + 1e-6), and q is
// scaled by 1/sqrt(key_dim), as the composition does.
//   mixed_qkv     [sequence, conv_dim] F32 conv output, q | k | v, rows
//                 16-byte aligned (conv_dim % 4 == 0)
//   gates         [sequence, value_heads, 2] F32 (log decay g, beta)
//   initial_state [value_heads, 128, value_dim] F32
//   packed_output [sequence, value_heads, value_dim] outputs, then the final
//                 [value_heads, 128, value_dim] state
// grid (value_dim / 16, value_heads), 128 threads: four columns per warp.
#define QWEN38_GDN_SEQ_KEY_DIM 128
#define QWEN38_GDN_SEQ_LANES 8
#define QWEN38_GDN_SEQ_KEYS (QWEN38_GDN_SEQ_KEY_DIM / QWEN38_GDN_SEQ_LANES)
#define QWEN38_GDN_SEQ_COLUMNS 16
__device__ __forceinline__ void qwen38_gdn_load16(const float* source, float* target) {
#pragma unroll
  for (int i = 0; i < QWEN38_GDN_SEQ_KEYS; i += 4) {
    const float4 v = *reinterpret_cast<const float4*>(source + i);
    target[i] = v.x;
    target[i + 1] = v.y;
    target[i + 2] = v.z;
    target[i + 3] = v.w;
  }
}
extern "C" __global__ void __launch_bounds__(128) qwen38_deltanet_sequence_f32(
    const float* __restrict__ mixed_qkv,
    const float* __restrict__ gates,
    const float* __restrict__ initial_state,
    float* __restrict__ packed_output,
    int sequence,
    int key_heads,
    int value_heads,
    int value_dim) {
  const int lane = threadIdx.x & 31;
  const int part = lane & (QWEN38_GDN_SEQ_LANES - 1);
  const int column = blockIdx.x * QWEN38_GDN_SEQ_COLUMNS + (threadIdx.x >> 3);
  const int value_head = blockIdx.y;
  const int key_head = value_head / (value_heads / key_heads);
  const int key_width = key_heads * QWEN38_GDN_SEQ_KEY_DIM;
  const size_t conv_dim = (size_t)key_width * 2 + (size_t)value_heads * value_dim;
  const int key0 = part * QWEN38_GDN_SEQ_KEYS;
  const size_t state_base = (size_t)value_head * QWEN38_GDN_SEQ_KEY_DIM * value_dim;
  const float* query_row = mixed_qkv + key_head * QWEN38_GDN_SEQ_KEY_DIM + key0;
  const float* key_row = query_row + key_width;
  const float* value_row = mixed_qkv + (size_t)key_width * 2 + (size_t)value_head * value_dim + column;
  const float* gate_row = gates + (size_t)value_head * 2;
  const size_t gate_stride = (size_t)value_heads * 2;
  const float query_scale = rsqrtf((float)QWEN38_GDN_SEQ_KEY_DIM);

  float state[QWEN38_GDN_SEQ_KEYS];
#pragma unroll
  for (int i = 0; i < QWEN38_GDN_SEQ_KEYS; ++i) {
    state[i] = initial_state[state_base + (size_t)(key0 + i) * value_dim + column];
  }
  // Token t's inputs are loaded during token t - 1: the loads do not depend
  // on the state, so they overlap the serial reductions.
  float query[QWEN38_GDN_SEQ_KEYS], key[QWEN38_GDN_SEQ_KEYS];
  qwen38_gdn_load16(query_row, query);
  qwen38_gdn_load16(key_row, key);
  float value = value_row[0], g = gate_row[0], beta = gate_row[1];
  for (int t = 0; t < sequence; ++t) {
    float next_query[QWEN38_GDN_SEQ_KEYS], next_key[QWEN38_GDN_SEQ_KEYS];
    float next_value = 0.0f, next_g = 0.0f, next_beta = 0.0f;
    if (t + 1 < sequence) {
      const size_t row = (size_t)(t + 1) * conv_dim;
      qwen38_gdn_load16(query_row + row, next_query);
      qwen38_gdn_load16(key_row + row, next_key);
      next_value = value_row[row];
      next_g = gate_row[(size_t)(t + 1) * gate_stride];
      next_beta = gate_row[(size_t)(t + 1) * gate_stride + 1];
    } else {
#pragma unroll
      for (int i = 0; i < QWEN38_GDN_SEQ_KEYS; ++i) {
        next_query[i] = 0.0f;
        next_key[i] = 0.0f;
      }
    }
    const float decay = expf(g);
    float recalled = 0.0f, key_squares = 0.0f;
#pragma unroll
    for (int i = 0; i < QWEN38_GDN_SEQ_KEYS; ++i) {
      state[i] *= decay;
      recalled = fmaf(key[i], state[i], recalled);
      key_squares = fmaf(key[i], key[i], key_squares);
    }
#pragma unroll
    for (int offset = QWEN38_GDN_SEQ_LANES / 2; offset > 0; offset >>= 1) {
      recalled += __shfl_xor_sync(0xffffffffu, recalled, offset);
      key_squares += __shfl_xor_sync(0xffffffffu, key_squares, offset);
    }
    const float key_norm = rsqrtf(key_squares + 1.0e-6f);
    const float update = (value - recalled * key_norm) * beta * key_norm;
    float output = 0.0f, query_squares = 0.0f;
#pragma unroll
    for (int i = 0; i < QWEN38_GDN_SEQ_KEYS; ++i) {
      state[i] = fmaf(key[i], update, state[i]);
      output = fmaf(query[i], state[i], output);
      query_squares = fmaf(query[i], query[i], query_squares);
    }
#pragma unroll
    for (int offset = QWEN38_GDN_SEQ_LANES / 2; offset > 0; offset >>= 1) {
      output += __shfl_xor_sync(0xffffffffu, output, offset);
      query_squares += __shfl_xor_sync(0xffffffffu, query_squares, offset);
    }
    if (part == 0) {
      packed_output[((size_t)t * value_heads + value_head) * value_dim + column] =
          output * rsqrtf(query_squares + 1.0e-6f) * query_scale;
    }
#pragma unroll
    for (int i = 0; i < QWEN38_GDN_SEQ_KEYS; ++i) {
      query[i] = next_query[i];
      key[i] = next_key[i];
    }
    value = next_value;
    g = next_g;
    beta = next_beta;
  }
  float* final_state = packed_output + (size_t)sequence * value_heads * value_dim;
#pragma unroll
  for (int i = 0; i < QWEN38_GDN_SEQ_KEYS; ++i) {
    final_state[state_base + (size_t)(key0 + i) * value_dim + column] = state[i];
  }
}
