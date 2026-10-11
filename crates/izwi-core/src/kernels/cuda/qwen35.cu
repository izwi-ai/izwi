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

// Chunked Gated DeltaNet prefill (the FLA chunk_gated_delta_rule algorithm).
//
// Within a chunk of C tokens starting from state S0, with gamma_t the running
// sum of the log decays g and the same q/k/v/beta as the sequential kernel:
//   delta_t = beta_t (v_t - e^{gamma_t} S0^T k_t)
//             - sum_{s<t} beta_t e^{gamma_t - gamma_s} (k_t . k_s) delta_s
//   o_t     = e^{gamma_t} S0^T q~_t + sum_{s<=t} e^{gamma_t - gamma_s} (q~_t . k_s) delta_s
//   S_end   = e^{gamma_C} S0 + sum_s e^{gamma_C - gamma_s} k_s delta_s^T
// (q~ = q / sqrt(key_dim)), which unrolls the per-token recurrence exactly.
// The prepare kernel builds each chunk's gamma, A (the strictly lower
// k.k term) and P (the lower q.k term) for every chunk at once; the scan
// kernel walks the chunks in order, one block per (head, 16 value columns):
// the update is independent per value column, so the heads' state is split
// across 8 blocks instead of serialized through one.
#define QWEN35_GDN_CHUNK 64
#define QWEN35_GDN_COLUMNS 16
#define QWEN35_GDN_DTILE 32
#define QWEN35_GDN_MAX_KEY_DIM 256

// grid (chunks, heads), 256 threads. gamma: [chunks, heads, CHUNK];
// a, p: [chunks, heads, CHUNK, CHUNK], zero outside the triangle and past
// the sequence. key_dim is a multiple of 32, at most 256.
extern "C" __global__ void qwen35_gated_delta_chunk_prepare_f32(
    const float* __restrict__ qkv,
    const float* __restrict__ gates,
    float* __restrict__ gamma_out,
    float* __restrict__ a_out,
    float* __restrict__ p_out,
    int sequence,
    int heads,
    int key_dim,
    int value_dim) {
  __shared__ float gamma[QWEN35_GDN_CHUNK];
  __shared__ float beta[QWEN35_GDN_CHUNK];
  const int chunk = blockIdx.x;
  const int head = blockIdx.y;
  const int start = chunk * QWEN35_GDN_CHUNK;
  const int len = min(QWEN35_GDN_CHUNK, sequence - start);
  const int qkv_width = key_dim * 2 + value_dim;
  if (threadIdx.x == 0) {
    float running = 0.0f;
    for (int t = 0; t < QWEN35_GDN_CHUNK; ++t) {
      if (t < len) {
        const float* gate = gates + ((size_t)(start + t) * heads + head) * 2;
        running += gate[0];
        gamma[t] = running;
        beta[t] = gate[1];
      } else {
        gamma[t] = 0.0f;
        beta[t] = 0.0f;
      }
    }
  }
  __syncthreads();

  const size_t chunk_head = (size_t)chunk * heads + head;
  for (int t = threadIdx.x; t < QWEN35_GDN_CHUNK; t += blockDim.x) {
    gamma_out[chunk_head * QWEN35_GDN_CHUNK + t] = gamma[t];
  }
  float* a = a_out + chunk_head * QWEN35_GDN_CHUNK * QWEN35_GDN_CHUNK;
  float* p = p_out + chunk_head * QWEN35_GDN_CHUNK * QWEN35_GDN_CHUNK;
  const float query_scale = rsqrtf((float)key_dim);
  const int lane = threadIdx.x & 31;
  const int warps = blockDim.x >> 5;
  const int per_lane = key_dim >> 5;
  // One warp per row t: q_t and k_t stay in registers, each lane holding
  // every 32nd element, and each (t, s) dot product is a warp reduction.
  for (int t = threadIdx.x >> 5; t < QWEN35_GDN_CHUNK; t += warps) {
    float key_t[QWEN35_GDN_MAX_KEY_DIM / 32];
    float query_t[QWEN35_GDN_MAX_KEY_DIM / 32];
    if (t < len) {
      const float* row = qkv + ((size_t)(start + t) * heads + head) * qkv_width;
#pragma unroll
      for (int i = 0; i < QWEN35_GDN_MAX_KEY_DIM / 32; ++i) {
        if (i < per_lane) {
          query_t[i] = row[lane + 32 * i] * query_scale;
          key_t[i] = row[key_dim + lane + 32 * i];
        }
      }
    }
    for (int s = 0; s < QWEN35_GDN_CHUNK; ++s) {
      float key_dot = 0.0f;
      float query_dot = 0.0f;
      const bool inside = t < len && s <= t;
      if (inside) {
        const float* key_s =
            qkv + ((size_t)(start + s) * heads + head) * qkv_width + key_dim;
#pragma unroll
        for (int i = 0; i < QWEN35_GDN_MAX_KEY_DIM / 32; ++i) {
          if (i < per_lane) {
            const float k = key_s[lane + 32 * i];
            key_dot += key_t[i] * k;
            query_dot += query_t[i] * k;
          }
        }
      }
      for (int offset = 16; offset > 0; offset >>= 1) {
        key_dot += __shfl_xor_sync(0xffffffff, key_dot, offset);
        query_dot += __shfl_xor_sync(0xffffffff, query_dot, offset);
      }
      if (lane == 0) {
        float a_value = 0.0f;
        float p_value = 0.0f;
        if (inside) {
          const float decay = expf(gamma[t] - gamma[s]);
          if (s < t) {
            a_value = beta[t] * decay * key_dot;
          }
          p_value = decay * query_dot;
        }
        a[t * QWEN35_GDN_CHUNK + s] = a_value;
        p[t * QWEN35_GDN_CHUNK + s] = p_value;
      }
    }
  }
}

// grid (heads, value_dim / 16), 256 threads. Same packed output as
// qwen35_gated_delta_sequence_f32: [sequence, heads, value_dim] outputs,
// then the final [heads, key_dim, value_dim] state. value_dim is a multiple
// of 16.
extern "C" __global__ void qwen35_gated_delta_chunk_scan_f32(
    const float* __restrict__ qkv,
    const float* __restrict__ gates,
    const float* __restrict__ initial_state,
    const float* __restrict__ gamma_in,
    const float* __restrict__ a_in,
    const float* __restrict__ p_in,
    float* __restrict__ packed_output,
    int sequence,
    int heads,
    int key_dim,
    int value_dim) {
  __shared__ float state[QWEN35_GDN_MAX_KEY_DIM][QWEN35_GDN_COLUMNS];
  __shared__ float key_tile[QWEN35_GDN_CHUNK][QWEN35_GDN_DTILE + 1];
  __shared__ float query_tile[QWEN35_GDN_CHUNK][QWEN35_GDN_DTILE + 1];
  __shared__ float delta[QWEN35_GDN_CHUNK][QWEN35_GDN_COLUMNS];
  __shared__ float gamma[QWEN35_GDN_CHUNK];
  __shared__ float beta[QWEN35_GDN_CHUNK];

  const int head = blockIdx.x;
  const int column0 = blockIdx.y * QWEN35_GDN_COLUMNS;
  const int tid = threadIdx.x;
  const int qkv_width = key_dim * 2 + value_dim;
  const float query_scale = rsqrtf((float)key_dim);
  const size_t head_state = (size_t)head * key_dim * value_dim;
  // Matmul ownership: row t = tid / 4, columns c4 .. c4 + 3.
  const int row = tid >> 2;
  const int c4 = (tid & 3) * 4;

  for (int index = tid; index < key_dim * QWEN35_GDN_COLUMNS; index += blockDim.x) {
    const int d = index / QWEN35_GDN_COLUMNS;
    const int c = index - d * QWEN35_GDN_COLUMNS;
    state[d][c] = initial_state[head_state + (size_t)d * value_dim + column0 + c];
  }

  const int chunks = (sequence + QWEN35_GDN_CHUNK - 1) / QWEN35_GDN_CHUNK;
  for (int chunk = 0; chunk < chunks; ++chunk) {
    const int start = chunk * QWEN35_GDN_CHUNK;
    const int len = min(QWEN35_GDN_CHUNK, sequence - start);
    const size_t chunk_head = (size_t)chunk * heads + head;
    const float* a = a_in + chunk_head * QWEN35_GDN_CHUNK * QWEN35_GDN_CHUNK;
    const float* p = p_in + chunk_head * QWEN35_GDN_CHUNK * QWEN35_GDN_CHUNK;
    for (int t = tid; t < QWEN35_GDN_CHUNK; t += blockDim.x) {
      gamma[t] = gamma_in[chunk_head * QWEN35_GDN_CHUNK + t];
      beta[t] = t < len ? gates[((size_t)(start + t) * heads + head) * 2 + 1] : 0.0f;
    }

    // K S0 and q~ S0 for this chunk, over key tiles of 32.
    float key_acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float query_acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (int d0 = 0; d0 < key_dim; d0 += QWEN35_GDN_DTILE) {
      __syncthreads();
      for (int index = tid; index < QWEN35_GDN_CHUNK * QWEN35_GDN_DTILE; index += blockDim.x) {
        const int t = index / QWEN35_GDN_DTILE;
        const int j = index - t * QWEN35_GDN_DTILE;
        float k = 0.0f;
        float q = 0.0f;
        if (t < len) {
          const float* source = qkv + ((size_t)(start + t) * heads + head) * qkv_width;
          q = source[d0 + j] * query_scale;
          k = source[key_dim + d0 + j];
        }
        key_tile[t][j] = k;
        query_tile[t][j] = q;
      }
      __syncthreads();
      for (int j = 0; j < QWEN35_GDN_DTILE; ++j) {
        const float k = key_tile[row][j];
        const float q = query_tile[row][j];
        for (int i = 0; i < 4; ++i) {
          const float s = state[d0 + j][c4 + i];
          key_acc[i] += k * s;
          query_acc[i] += q * s;
        }
      }
    }
    // The right-hand side beta_t (v_t - e^{gamma_t} (K S0)_t); padded rows 0.
    // (q~ S0) stays in this thread's registers for its own output rows.
    for (int i = 0; i < 4; ++i) {
      float rhs = 0.0f;
      if (row < len) {
        const float v = qkv[((size_t)(start + row) * heads + head) * qkv_width + key_dim * 2
                            + column0 + c4 + i];
        rhs = beta[row] * (v - expf(gamma[row]) * key_acc[i]);
      }
      delta[row][c4 + i] = rhs;
    }
    __syncthreads();

    // Forward substitution: delta_t -= sum_{s<t} A[t][s] delta_s. Each warp
    // owns two columns, 16 lanes per column splitting the sum over s, so a
    // column's rows only need warp-level ordering.
    {
      const int lane = tid & 31;
      const int column = (tid >> 5) * 2 + (lane >> 4);
      const int part = lane & 15;
      for (int t = 1; t < len; ++t) {
        float sum = 0.0f;
        for (int s = part; s < t; s += 16) {
          sum += a[t * QWEN35_GDN_CHUNK + s] * delta[s][column];
        }
        for (int offset = 8; offset > 0; offset >>= 1) {
          sum += __shfl_xor_sync(0xffffffff, sum, offset);
        }
        if (part == 0) {
          delta[t][column] -= sum;
        }
        __syncwarp();
      }
    }
    __syncthreads();

    // o_t = e^{gamma_t} (q~ S0)_t + sum_{s<=t} P[t][s] delta_s.
    if (row < len) {
      float out[4];
      const float decay = expf(gamma[row]);
      for (int i = 0; i < 4; ++i) {
        out[i] = decay * query_acc[i];
      }
      for (int s = 0; s <= row; ++s) {
        const float weight = p[row * QWEN35_GDN_CHUNK + s];
        for (int i = 0; i < 4; ++i) {
          out[i] += weight * delta[s][c4 + i];
        }
      }
      float* destination = packed_output
          + ((size_t)(start + row) * heads + head) * value_dim + column0 + c4;
      for (int i = 0; i < 4; ++i) {
        destination[i] = out[i];
      }
    }

    // S <- e^{gamma_end} S + sum_s e^{gamma_end - gamma_s} k_s delta_s^T, by
    // key tiles; each thread owns two columns of one key row in the tile.
    const float gamma_end = gamma[len - 1];
    const float state_decay = expf(gamma_end);
    const int tile_row = tid >> 3;
    const int c2 = (tid & 7) * 2;
    for (int d0 = 0; d0 < key_dim; d0 += QWEN35_GDN_DTILE) {
      __syncthreads();
      for (int index = tid; index < QWEN35_GDN_CHUNK * QWEN35_GDN_DTILE; index += blockDim.x) {
        const int s = index / QWEN35_GDN_DTILE;
        const int j = index - s * QWEN35_GDN_DTILE;
        float k = 0.0f;
        if (s < len) {
          k = qkv[((size_t)(start + s) * heads + head) * qkv_width + key_dim + d0 + j]
              * expf(gamma_end - gamma[s]);
        }
        key_tile[s][j] = k;
      }
      __syncthreads();
      float acc0 = 0.0f;
      float acc1 = 0.0f;
      for (int s = 0; s < len; ++s) {
        const float k = key_tile[s][tile_row];
        acc0 += k * delta[s][c2];
        acc1 += k * delta[s][c2 + 1];
      }
      state[d0 + tile_row][c2] = state_decay * state[d0 + tile_row][c2] + acc0;
      state[d0 + tile_row][c2 + 1] = state_decay * state[d0 + tile_row][c2 + 1] + acc1;
    }
    __syncthreads();
  }

  float* final_state = packed_output + (size_t)sequence * heads * value_dim + head_state;
  for (int index = tid; index < key_dim * QWEN35_GDN_COLUMNS; index += blockDim.x) {
    const int d = index / QWEN35_GDN_COLUMNS;
    const int c = index - d * QWEN35_GDN_COLUMNS;
    final_state[(size_t)d * value_dim + column0 + c] = state[d][c];
  }
}
