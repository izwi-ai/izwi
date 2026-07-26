#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <math.h>

template <typename T>
__device__ inline float izwi_to_float(T value);

template <>
__device__ inline float izwi_to_float<float>(float value) {
  return value;
}

template <>
__device__ inline float izwi_to_float<__half>(__half value) {
  return __half2float(value);
}

template <>
__device__ inline float izwi_to_float<__nv_bfloat16>(
    __nv_bfloat16 value) {
  return __bfloat162float(value);
}

template <typename T>
__device__ inline T izwi_from_float(float value);

template <>
__device__ inline float izwi_from_float<float>(float value) {
  return value;
}

template <>
__device__ inline __half izwi_from_float<__half>(float value) {
  return __float2half_rn(value);
}

template <>
__device__ inline __nv_bfloat16 izwi_from_float<__nv_bfloat16>(
    float value) {
  return __float2bfloat16_rn(value);
}

// Direct single-query paged attention.
//
// One CUDA block owns one (sequence, query-head) pair. K/V remain in the
// physical [page, token, kv-head, dim] arena; the block table and first-page
// offset select the visible logical sequence. Online softmax keeps memory
// bounded independently of context length and GQA maps each query head to its
// source KV head without expanding KV tensors.
template <typename T>
__device__ void izwi_paged_decode_attention(
    const T* __restrict__ queries,
    const T* __restrict__ keys,
    const T* __restrict__ values,
    const unsigned int* __restrict__ metadata,
    T* __restrict__ output,
    int batch,
    int query_heads,
    int kv_heads,
    int page_tokens,
    int max_blocks,
    int key_dim,
    int value_dim,
    float softmax_scale) {
  const int row_head = blockIdx.x;
  if (row_head >= batch * query_heads) {
    return;
  }
  const int row = row_head / query_heads;
  const int query_head = row_head - row * query_heads;
  const int queries_per_kv = query_heads / kv_heads;
  const int kv_head = query_head / queries_per_kv;
  const unsigned int context_len = metadata[row];
  const unsigned int first_page_offset = metadata[batch + row];
  const unsigned int* block_table = metadata + batch * 2;
  const int query_base = row_head * key_dim;

  extern __shared__ float reduction[];
  float output_0 = 0.0f;
  float output_1 = 0.0f;
  float running_max = -INFINITY;
  float running_sum = 0.0f;

  for (unsigned int logical_token = 0; logical_token < context_len;
       ++logical_token) {
    const unsigned int physical_token = logical_token + first_page_offset;
    const unsigned int logical_page = physical_token / page_tokens;
    const unsigned int page_offset = physical_token -
                                     logical_page * page_tokens;
    const unsigned int physical_page =
        block_table[row * max_blocks + logical_page];
    const int key_base =
        ((physical_page * page_tokens + page_offset) * kv_heads + kv_head) *
        key_dim;
    const int value_base =
        ((physical_page * page_tokens + page_offset) * kv_heads + kv_head) *
        value_dim;

    float partial = 0.0f;
    for (int dim = threadIdx.x; dim < key_dim; dim += blockDim.x) {
      partial += izwi_to_float(queries[query_base + dim]) *
                 izwi_to_float(keys[key_base + dim]);
    }
    reduction[threadIdx.x] = partial;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
      if (threadIdx.x < stride) {
        reduction[threadIdx.x] += reduction[threadIdx.x + stride];
      }
      __syncthreads();
    }

    if (threadIdx.x == 0) {
      const float score = reduction[0] * softmax_scale;
      const float next_max = fmaxf(running_max, score);
      const float previous_weight = expf(running_max - next_max);
      const float token_weight = expf(score - next_max);
      running_sum = running_sum * previous_weight + token_weight;
      running_max = next_max;
      reduction[0] = previous_weight;
      reduction[1] = token_weight;
      reduction[2] = running_sum;
      reduction[3] = running_max;
    }
    __syncthreads();

    const float previous_weight = reduction[0];
    const float token_weight = reduction[1];
    if (threadIdx.x < value_dim) {
      output_0 = output_0 * previous_weight +
                 izwi_to_float(values[value_base + threadIdx.x]) *
                     token_weight;
    }
    const int second_dim = threadIdx.x + blockDim.x;
    if (second_dim < value_dim) {
      output_1 = output_1 * previous_weight +
                 izwi_to_float(values[value_base + second_dim]) *
                     token_weight;
    }
    __syncthreads();
    running_sum = reduction[2];
    running_max = reduction[3];
  }

  const int output_base = row_head * value_dim;
  if (threadIdx.x < value_dim) {
    output[output_base + threadIdx.x] =
        izwi_from_float<T>(output_0 / running_sum);
  }
  const int second_dim = threadIdx.x + blockDim.x;
  if (second_dim < value_dim) {
    output[output_base + second_dim] =
        izwi_from_float<T>(output_1 / running_sum);
  }
}

extern "C" __global__ void physical_paged_decode_f32(
    const float* queries,
    const float* keys,
    const float* values,
    const unsigned int* metadata,
    float* output,
    int batch,
    int query_heads,
    int kv_heads,
    int page_tokens,
    int max_blocks,
    int key_dim,
    int value_dim,
    float softmax_scale) {
  izwi_paged_decode_attention(
      queries, keys, values, metadata, output, batch, query_heads, kv_heads,
      page_tokens, max_blocks, key_dim, value_dim, softmax_scale);
}

extern "C" __global__ void physical_paged_decode_f16(
    const __half* queries,
    const __half* keys,
    const __half* values,
    const unsigned int* metadata,
    __half* output,
    int batch,
    int query_heads,
    int kv_heads,
    int page_tokens,
    int max_blocks,
    int key_dim,
    int value_dim,
    float softmax_scale) {
  izwi_paged_decode_attention(
      queries, keys, values, metadata, output, batch, query_heads, kv_heads,
      page_tokens, max_blocks, key_dim, value_dim, softmax_scale);
}

extern "C" __global__ void physical_paged_decode_bf16(
    const __nv_bfloat16* queries,
    const __nv_bfloat16* keys,
    const __nv_bfloat16* values,
    const unsigned int* metadata,
    __nv_bfloat16* output,
    int batch,
    int query_heads,
    int kv_heads,
    int page_tokens,
    int max_blocks,
    int key_dim,
    int value_dim,
    float softmax_scale) {
  izwi_paged_decode_attention(
      queries, keys, values, metadata, output, batch, query_heads, kv_heads,
      page_tokens, max_blocks, key_dim, value_dim, softmax_scale);
}

// Consume a physical circular ShortConv ring directly. The ring layout is
// [capacity, batch, hidden], while input/output use [batch, hidden, steps].
extern "C" __global__ void physical_ring_shortconv_f32(
    const float* __restrict__ ring,
    const float* __restrict__ input,
    const float* __restrict__ weight,
    float* __restrict__ output,
    int batch,
    int hidden,
    int steps,
    int capacity,
    unsigned long long expected_cursor,
    unsigned long long valid_length,
    int output_elements) {
  const int gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid >= output_elements) {
    return;
  }
  const int step = gid % steps;
  const int hidden_idx = (gid / steps) % hidden;
  const int batch_idx = gid / (steps * hidden);
  const long long window_start =
      static_cast<long long>(expected_cursor) + step + 1 - capacity;
  const unsigned long long oldest = expected_cursor - valid_length;
  float value = 0.0f;
  for (int tap = 0; tap < capacity; ++tap) {
    const long long source = window_start + tap;
    if (source < 0) {
      continue;
    }
    float source_value;
    const unsigned long long absolute_source =
        static_cast<unsigned long long>(source);
    if (absolute_source < expected_cursor) {
      if (absolute_source < oldest) {
        continue;
      }
      const unsigned long long physical = absolute_source % capacity;
      source_value =
          ring[(physical * batch + batch_idx) * hidden + hidden_idx];
    } else {
      const unsigned long long input_step =
          absolute_source - expected_cursor;
      if (input_step > static_cast<unsigned long long>(step)) {
        continue;
      }
      source_value =
          input[(batch_idx * hidden + hidden_idx) * steps + input_step];
    }
    value += source_value * weight[hidden_idx * capacity + tap];
  }
  output[gid] = value;
}
