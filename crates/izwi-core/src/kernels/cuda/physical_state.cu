#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <math.h>

#if __has_include(<cuda_fp8.h>)
#include <cuda_fp8.h>
#define IZWI_CUDA_HAS_FP8 1
#endif

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

#if defined(IZWI_CUDA_HAS_FP8)
template <>
__device__ inline float izwi_to_float<__nv_fp8_e4m3>(
    __nv_fp8_e4m3 value) {
  return static_cast<float>(value);
}
#endif

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
template <typename T, typename KV = T>
__device__ void izwi_paged_decode_attention(
    const T* __restrict__ queries,
    const KV* __restrict__ keys,
    const KV* __restrict__ values,
    const unsigned int* __restrict__ metadata,
    T* __restrict__ output,
    int batch,
    int query_heads,
    int kv_heads,
    int page_tokens,
    int max_blocks,
    int key_dim,
    int value_dim,
    int capacity_pages,
    float softmax_scale,
    float softcap) {
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
  const int output_base = row_head * value_dim;

  // Leave a deterministic result if a future caller bypasses host-side plan
  // validation. Every condition below is uniform across the thread block.
  bool metadata_valid = page_tokens > 0 && max_blocks > 0 &&
                        capacity_pages > 0 && context_len > 0;
  unsigned long long required_pages = 0;
  if (metadata_valid) {
    metadata_valid =
        first_page_offset < static_cast<unsigned int>(page_tokens);
    const unsigned long long physical_tokens =
        static_cast<unsigned long long>(context_len) + first_page_offset;
    required_pages =
        (physical_tokens + static_cast<unsigned long long>(page_tokens) - 1) /
        static_cast<unsigned long long>(page_tokens);
    metadata_valid = metadata_valid && required_pages > 0 &&
                     required_pages <=
                         static_cast<unsigned long long>(max_blocks);
  }
  if (metadata_valid) {
    for (unsigned long long logical_page = 0; logical_page < required_pages;
         ++logical_page) {
      if (block_table[row * max_blocks + logical_page] >=
          static_cast<unsigned int>(capacity_pages)) {
        metadata_valid = false;
        break;
      }
    }
  }
  if (!metadata_valid) {
    if (threadIdx.x < value_dim) {
      output[output_base + threadIdx.x] = izwi_from_float<T>(0.0f);
    }
    const int second_output_dim = threadIdx.x + blockDim.x;
    if (second_output_dim < value_dim) {
      output[output_base + second_output_dim] = izwi_from_float<T>(0.0f);
    }
    return;
  }

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
      float score = reduction[0] * softmax_scale;
      if (softcap > 0.0f) {
        score = softcap * tanhf(score / softcap);
      }
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
    running_sum = reduction[2];
    running_max = reduction[3];
    // Complete every shared read before the next iteration reuses reduction.
    __syncthreads();
  }

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

// Direct ragged causal paged prefill. One block owns one
// (query-token, query-head) pair. Compact metadata stores four row vectors
// (query start/length, final context length, first-page offset) followed by a
// rectangular physical block table. This avoids expanding prefill into one
// host-authored decode row per query token when FlashAttention is ineligible.
template <typename T, typename KV = T>
__device__ void izwi_paged_prefill_attention(
    const T* __restrict__ queries,
    const KV* __restrict__ keys,
    const KV* __restrict__ values,
    const unsigned int* __restrict__ metadata,
    T* __restrict__ output,
    int sequences,
    int total_queries,
    int query_heads,
    int kv_heads,
    int page_tokens,
    int max_blocks,
    int key_dim,
    int value_dim,
    int capacity_pages,
    int window_tokens,
    float softmax_scale,
    float softcap) {
  const int query_head_index = blockIdx.x;
  if (query_head_index >= total_queries * query_heads) {
    return;
  }
  const int query_index = query_head_index / query_heads;
  const int query_head = query_head_index - query_index * query_heads;
  const int queries_per_kv = query_heads / kv_heads;
  const int kv_head = query_head / queries_per_kv;
  const unsigned int* query_starts = metadata;
  const unsigned int* query_lengths = metadata + sequences;
  const unsigned int* context_lengths = metadata + sequences * 2;
  const unsigned int* first_page_offsets = metadata + sequences * 3;
  const unsigned int* block_table = metadata + sequences * 4;

  int row = -1;
  unsigned int local_query = 0;
  for (int candidate = 0; candidate < sequences; ++candidate) {
    const unsigned int start = query_starts[candidate];
    const unsigned int length = query_lengths[candidate];
    if (query_index >= static_cast<int>(start) &&
        query_index < static_cast<int>(start + length)) {
      row = candidate;
      local_query = static_cast<unsigned int>(query_index) - start;
      break;
    }
  }

  bool metadata_valid = row >= 0 && page_tokens > 0 && max_blocks > 0 &&
                        capacity_pages > 0;
  unsigned int visible_context = 0;
  unsigned int physical_start = 0;
  unsigned long long required_pages = 0;
  if (metadata_valid) {
    const unsigned int query_len = query_lengths[row];
    const unsigned int context_len = context_lengths[row];
    const unsigned int first_page_offset = first_page_offsets[row];
    metadata_valid = query_len > 0 && query_len <= context_len &&
                     local_query < query_len &&
                     first_page_offset < static_cast<unsigned int>(page_tokens);
    if (metadata_valid) {
      const unsigned int causal_context =
          context_len - query_len + local_query + 1;
      visible_context =
          window_tokens > 0
              ? min(causal_context, static_cast<unsigned int>(window_tokens))
              : causal_context;
      const unsigned int dropped = causal_context - visible_context;
      physical_start = first_page_offset + dropped;
      const unsigned long long physical_tokens =
          static_cast<unsigned long long>(context_len) + first_page_offset;
      required_pages =
          (physical_tokens + static_cast<unsigned long long>(page_tokens) - 1) /
          static_cast<unsigned long long>(page_tokens);
      metadata_valid = visible_context > 0 && required_pages > 0 &&
                       required_pages <=
                           static_cast<unsigned long long>(max_blocks);
    }
  }
  if (metadata_valid) {
    for (unsigned long long logical_page = 0; logical_page < required_pages;
         ++logical_page) {
      if (block_table[row * max_blocks + logical_page] >=
          static_cast<unsigned int>(capacity_pages)) {
        metadata_valid = false;
        break;
      }
    }
  }

  const int output_base = query_head_index * value_dim;
  if (!metadata_valid) {
    if (threadIdx.x < value_dim) {
      output[output_base + threadIdx.x] = izwi_from_float<T>(0.0f);
    }
    const int second_output_dim = threadIdx.x + blockDim.x;
    if (second_output_dim < value_dim) {
      output[output_base + second_output_dim] = izwi_from_float<T>(0.0f);
    }
    return;
  }

  const int query_base = query_head_index * key_dim;
  extern __shared__ float reduction[];
  float output_0 = 0.0f;
  float output_1 = 0.0f;
  float running_max = -INFINITY;
  float running_sum = 0.0f;

  for (unsigned int logical_token = 0; logical_token < visible_context;
       ++logical_token) {
    const unsigned int physical_token = physical_start + logical_token;
    const unsigned int logical_page = physical_token / page_tokens;
    const unsigned int page_offset =
        physical_token - logical_page * page_tokens;
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
      float score = reduction[0] * softmax_scale;
      if (softcap > 0.0f) {
        score = softcap * tanhf(score / softcap);
      }
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
    running_sum = reduction[2];
    running_max = reduction[3];
    // Complete every shared read before the next iteration reuses reduction.
    __syncthreads();
  }

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

// Partitioned decode for long contexts. The first pass computes an independent
// online-softmax state for each token partition. The second pass combines those
// states with the same max-rescaling identity, so no materialized score tensor
// is required and arbitrary first-page offsets retain the one-pass semantics.
template <typename T, typename KV = T>
__device__ void izwi_paged_decode_attention_partition(
    const T* __restrict__ queries,
    const KV* __restrict__ keys,
    const KV* __restrict__ values,
    const unsigned int* __restrict__ metadata,
    float* __restrict__ partials,
    int batch,
    int query_heads,
    int kv_heads,
    int page_tokens,
    int max_blocks,
    int key_dim,
    int value_dim,
    int capacity_pages,
    int partition_tokens,
    int num_partitions,
    float softmax_scale,
    float softcap) {
  const int row_head = blockIdx.x;
  const int partition = blockIdx.y;
  if (row_head >= batch * query_heads || partition >= num_partitions) {
    return;
  }
  const int row = row_head / query_heads;
  const int query_head = row_head - row * query_heads;
  const unsigned int context_len = metadata[row];
  const unsigned int first_page_offset = metadata[batch + row];
  const unsigned int* block_table = metadata + batch * 2;
  const unsigned int token_start =
      static_cast<unsigned int>(partition) * partition_tokens;
  const unsigned int token_end =
      min(context_len, token_start + static_cast<unsigned int>(partition_tokens));
  const int query_base = row_head * key_dim;
  const unsigned long long partial_stride =
      static_cast<unsigned long long>(value_dim) + 2;
  const unsigned long long partial_base =
      (static_cast<unsigned long long>(row_head) * num_partitions + partition) *
      partial_stride;

  bool metadata_valid = query_heads > 0 && page_tokens > 0 && max_blocks > 0 &&
                        kv_heads > 0 && query_heads >= kv_heads &&
                        query_heads % kv_heads == 0 && context_len > 0 &&
                        first_page_offset < static_cast<unsigned int>(page_tokens) &&
                        partition_tokens > 0 && num_partitions > 0;
  if (metadata_valid) {
    const unsigned long long physical_tokens =
        static_cast<unsigned long long>(context_len) + first_page_offset;
    const unsigned long long required_pages =
        (physical_tokens + static_cast<unsigned long long>(page_tokens) - 1) /
        static_cast<unsigned long long>(page_tokens);
    metadata_valid = required_pages > 0 &&
                     required_pages <= static_cast<unsigned long long>(max_blocks);
  }
  if (!metadata_valid) {
    if (threadIdx.x == 0) {
      partials[partial_base] = -INFINITY;
      partials[partial_base + 1] = 0.0f;
    }
    if (threadIdx.x < value_dim) {
      partials[partial_base + 2 + threadIdx.x] = 0.0f;
    }
    const int second_dim = threadIdx.x + blockDim.x;
    if (second_dim < value_dim) {
      partials[partial_base + 2 + second_dim] = 0.0f;
    }
    return;
  }
  const int kv_head = query_head / (query_heads / kv_heads);

  extern __shared__ float reduction[];
  float output_0 = 0.0f;
  float output_1 = 0.0f;
  float running_max = -INFINITY;
  float running_sum = 0.0f;

  for (unsigned int logical_token = token_start; logical_token < token_end;
       ++logical_token) {
    const unsigned int physical_token = logical_token + first_page_offset;
    const unsigned int logical_page = physical_token / page_tokens;
    const unsigned int page_offset = physical_token - logical_page * page_tokens;
    const unsigned int physical_page =
        block_table[row * max_blocks + logical_page];
    // Host validation proves every referenced page is in the arena. Keep a
    // defensive guard because this kernel is also an exported PTX symbol.
    if (physical_page >= static_cast<unsigned int>(capacity_pages)) {
      if (threadIdx.x == 0) {
        partials[partial_base] = -INFINITY;
        partials[partial_base + 1] = 0.0f;
      }
      if (threadIdx.x < value_dim) {
        partials[partial_base + 2 + threadIdx.x] = 0.0f;
      }
      const int second_dim = threadIdx.x + blockDim.x;
      if (second_dim < value_dim) {
        partials[partial_base + 2 + second_dim] = 0.0f;
      }
      return;
    }
    const int key_base =
        ((physical_page * page_tokens + page_offset) * kv_heads + kv_head) *
        key_dim;
    const int value_base =
        ((physical_page * page_tokens + page_offset) * kv_heads + kv_head) *
        value_dim;

    float dot = 0.0f;
    for (int dim = threadIdx.x; dim < key_dim; dim += blockDim.x) {
      dot += izwi_to_float(queries[query_base + dim]) *
             izwi_to_float(keys[key_base + dim]);
    }
    reduction[threadIdx.x] = dot;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
      if (threadIdx.x < stride) {
        reduction[threadIdx.x] += reduction[threadIdx.x + stride];
      }
      __syncthreads();
    }
    if (threadIdx.x == 0) {
      float score = reduction[0] * softmax_scale;
      if (softcap > 0.0f) {
        score = softcap * tanhf(score / softcap);
      }
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
                 izwi_to_float(values[value_base + threadIdx.x]) * token_weight;
    }
    const int second_dim = threadIdx.x + blockDim.x;
    if (second_dim < value_dim) {
      output_1 = output_1 * previous_weight +
                 izwi_to_float(values[value_base + second_dim]) * token_weight;
    }
    running_sum = reduction[2];
    running_max = reduction[3];
    // Complete every shared read before the next iteration reuses reduction.
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    partials[partial_base] = running_max;
    partials[partial_base + 1] = running_sum;
  }
  if (threadIdx.x < value_dim) {
    partials[partial_base + 2 + threadIdx.x] = output_0;
  }
  const int second_dim = threadIdx.x + blockDim.x;
  if (second_dim < value_dim) {
    partials[partial_base + 2 + second_dim] = output_1;
  }
}

template <typename T>
__device__ void izwi_paged_decode_attention_reduce(
    const float* __restrict__ partials,
    T* __restrict__ output,
    int row_heads,
    int value_dim,
    int num_partitions) {
  const int row_head = blockIdx.x;
  if (row_head >= row_heads) {
    return;
  }
  const unsigned long long partial_stride =
      static_cast<unsigned long long>(value_dim) + 2;
  const unsigned long long row_base =
      static_cast<unsigned long long>(row_head) * num_partitions * partial_stride;
  const int output_base = row_head * value_dim;
  extern __shared__ float reduction[];
  float output_0 = 0.0f;
  float output_1 = 0.0f;
  float running_max = -INFINITY;
  float running_sum = 0.0f;

  for (int partition = 0; partition < num_partitions; ++partition) {
    const unsigned long long partial_base = row_base + partition * partial_stride;
    if (threadIdx.x == 0) {
      const float partial_max = partials[partial_base];
      const float partial_sum = partials[partial_base + 1];
      if (partial_sum > 0.0f) {
        const float next_max = fmaxf(running_max, partial_max);
        reduction[0] = expf(running_max - next_max);
        reduction[1] = expf(partial_max - next_max);
        running_sum = running_sum * reduction[0] + partial_sum * reduction[1];
        running_max = next_max;
      } else {
        reduction[0] = 1.0f;
        reduction[1] = 0.0f;
      }
      reduction[2] = running_sum;
      reduction[3] = running_max;
    }
    __syncthreads();
    if (threadIdx.x < value_dim) {
      output_0 = output_0 * reduction[0] +
                 partials[partial_base + 2 + threadIdx.x] * reduction[1];
    }
    const int second_dim = threadIdx.x + blockDim.x;
    if (second_dim < value_dim) {
      output_1 = output_1 * reduction[0] +
                 partials[partial_base + 2 + second_dim] * reduction[1];
    }
    running_sum = reduction[2];
    running_max = reduction[3];
    // Complete every shared read before the next iteration reuses reduction.
    __syncthreads();
  }

  if (threadIdx.x < value_dim) {
    output[output_base + threadIdx.x] =
        izwi_from_float<T>(running_sum > 0.0f ? output_0 / running_sum : 0.0f);
  }
  const int second_dim = threadIdx.x + blockDim.x;
  if (second_dim < value_dim) {
    output[output_base + second_dim] =
        izwi_from_float<T>(running_sum > 0.0f ? output_1 / running_sum : 0.0f);
  }
}

#define IZWI_DEFINE_PAGED_DECODE_PARTITION(SUFFIX, TYPE)                       \
  extern "C" __global__ void physical_paged_decode_partition_##SUFFIX(       \
      const TYPE* queries, const TYPE* keys, const TYPE* values,               \
      const unsigned int* metadata, float* partials, int batch,                \
      int query_heads, int kv_heads, int page_tokens, int max_blocks,          \
      int key_dim, int value_dim, int capacity_pages, int partition_tokens,    \
      int num_partitions, float softmax_scale, float softcap) {                \
    izwi_paged_decode_attention_partition(                                     \
        queries, keys, values, metadata, partials, batch, query_heads,         \
        kv_heads, page_tokens, max_blocks, key_dim, value_dim, capacity_pages, \
        partition_tokens, num_partitions, softmax_scale, softcap);             \
  }                                                                            \
  extern "C" __global__ void physical_paged_decode_reduce_##SUFFIX(          \
      const float* partials, TYPE* output, int row_heads, int value_dim,        \
      int num_partitions) {                                                     \
    izwi_paged_decode_attention_reduce(partials, output, row_heads, value_dim,  \
                                        num_partitions);                        \
  }

IZWI_DEFINE_PAGED_DECODE_PARTITION(f32, float)
IZWI_DEFINE_PAGED_DECODE_PARTITION(f16, __half)
IZWI_DEFINE_PAGED_DECODE_PARTITION(bf16, __nv_bfloat16)

#if defined(IZWI_CUDA_HAS_FP8)
#define IZWI_DEFINE_PAGED_DECODE_PARTITION_FP8(SUFFIX, TYPE)                   \
  extern "C" __global__ void physical_paged_decode_partition_##SUFFIX##_fp8( \
      const TYPE* queries, const __nv_fp8_e4m3* keys,                          \
      const __nv_fp8_e4m3* values, const unsigned int* metadata,               \
      float* partials, int batch, int query_heads, int kv_heads,               \
      int page_tokens, int max_blocks, int key_dim, int value_dim,             \
      int capacity_pages, int partition_tokens, int num_partitions,            \
      float softmax_scale, float softcap) {                                    \
    izwi_paged_decode_attention_partition<TYPE, __nv_fp8_e4m3>(                \
        queries, keys, values, metadata, partials, batch, query_heads,         \
        kv_heads, page_tokens, max_blocks, key_dim, value_dim, capacity_pages, \
        partition_tokens, num_partitions, softmax_scale, softcap);             \
  }

IZWI_DEFINE_PAGED_DECODE_PARTITION_FP8(f16, __half)
IZWI_DEFINE_PAGED_DECODE_PARTITION_FP8(bf16, __nv_bfloat16)
#endif

// Grouped-query split decode. One block owns (row, KV head, partition) and
// reads each K/V row of its token range once for every query head sharing that
// KV head; the kernels above (and FlashAttention-2 without splits) run one
// block per query head over the whole context, so B = 1 decode occupies only
// Hq SMs and re-reads each KV row group-size times. A warp scores 32 keys at a
// time, one key per lane, so scores need no cross-lane reduction, then folds
// those 32 value rows into accumulators that give each lane a slice of the
// head dimension. Partitions write the [max, sum, unnormalized output]
// partials of the partitioned kernel; izwi_paged_decode_split_reduce merges
// them. Token ranges derive from the device context length, so the host only
// chooses the partition count.
#define IZWI_SPLIT_DECODE_WARPS 4
#define IZWI_SPLIT_DECODE_MAX_GROUP 8
#define IZWI_SPLIT_DECODE_MAX_PARTITIONS 1024

// Loads N contiguous elements as floats. The host requires base pointers
// aligned to 16 bytes, and callers index in multiples of N.
template <typename T, int N>
__device__ __forceinline__ void izwi_load_floats(const T* __restrict__ source,
                                                 float* target) {
  if constexpr (N * sizeof(T) == 16) {
    const uint4 raw = *reinterpret_cast<const uint4*>(source);
    const T* lanes = reinterpret_cast<const T*>(&raw);
#pragma unroll
    for (int i = 0; i < N; ++i) {
      target[i] = izwi_to_float(lanes[i]);
    }
  } else if constexpr (N * sizeof(T) == 8) {
    const uint2 raw = *reinterpret_cast<const uint2*>(source);
    const T* lanes = reinterpret_cast<const T*>(&raw);
#pragma unroll
    for (int i = 0; i < N; ++i) {
      target[i] = izwi_to_float(lanes[i]);
    }
  } else {
#pragma unroll
    for (int i = 0; i < N; ++i) {
      target[i] = izwi_to_float(source[i]);
    }
  }
}

template <typename T, int HEAD_DIM>
__device__ void izwi_paged_decode_gqa_partition(
    const T* __restrict__ queries,
    const T* __restrict__ keys,
    const T* __restrict__ values,
    const unsigned int* __restrict__ metadata,
    float* __restrict__ partials,
    int batch,
    int query_heads,
    int kv_heads,
    int page_tokens,
    int max_blocks,
    int capacity_pages,
    int num_partitions,
    float softmax_scale,
    float softcap) {
  constexpr int MAX_GROUP = IZWI_SPLIT_DECODE_MAX_GROUP;
  constexpr int LANE_DIMS = HEAD_DIM / 32;
  constexpr int KEY_CHUNK = 16 / sizeof(T);
  static_assert(HEAD_DIM % (32 * 4) == 0, "head dim must give each lane 4n dims");
  static_assert(HEAD_DIM % KEY_CHUNK == 0, "key rows load in 16-byte chunks");
  __shared__ float shared_queries[MAX_GROUP * HEAD_DIM];
  __shared__ float shared_output[MAX_GROUP * HEAD_DIM];
  __shared__ float shared_max[IZWI_SPLIT_DECODE_WARPS][MAX_GROUP];
  __shared__ float shared_sum[IZWI_SPLIT_DECODE_WARPS][MAX_GROUP];
  __shared__ int shared_corrupt;

  const int row_kv = blockIdx.x;
  const int partition = blockIdx.y;
  if (row_kv >= batch * kv_heads || partition >= num_partitions) {
    return;
  }
  const int row = row_kv / kv_heads;
  const int kv_head = row_kv - row * kv_heads;
  const int group = query_heads / kv_heads;
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const unsigned int context_len = metadata[row];
  const unsigned int first_page_offset = metadata[batch + row];
  const unsigned int* block_table = metadata + batch * 2 + row * max_blocks;
  const unsigned long long partial_stride = HEAD_DIM + 2;

  // Every condition is uniform across the block. An invalid row writes empty
  // partials, which the reduction turns into a zero output.
  bool valid = group >= 1 && group <= MAX_GROUP &&
               group * kv_heads == query_heads && page_tokens > 0 &&
               max_blocks > 0 && capacity_pages > 0 && num_partitions > 0 &&
               context_len > 0 &&
               first_page_offset < static_cast<unsigned int>(page_tokens);
  if (valid) {
    const unsigned long long physical_tokens =
        static_cast<unsigned long long>(context_len) + first_page_offset;
    valid = (physical_tokens + page_tokens - 1) / page_tokens <=
            static_cast<unsigned long long>(max_blocks);
  }
  unsigned int token_start = 0;
  unsigned int token_end = 0;
  if (valid) {
    const unsigned int span =
        (context_len + num_partitions - 1) / num_partitions;
    token_start = min(context_len, partition * span);
    token_end = min(context_len, token_start + span);
  }

  for (int i = threadIdx.x; i < group * HEAD_DIM && valid; i += blockDim.x) {
    shared_queries[i] =
        izwi_to_float(queries[(row * query_heads + kv_head * group) * HEAD_DIM + i]);
  }
  if (threadIdx.x == 0) {
    shared_corrupt = 0;
  }
  __syncthreads();

  float running_max[MAX_GROUP];
  float running_sum[MAX_GROUP];  // this lane's keys only, reduced at the end
  float accumulator[MAX_GROUP][LANE_DIMS];
#pragma unroll
  for (int g = 0; g < MAX_GROUP; ++g) {
    running_max[g] = -INFINITY;
    running_sum[g] = 0.0f;
#pragma unroll
    for (int j = 0; j < LANE_DIMS; ++j) {
      accumulator[g][j] = 0.0f;
    }
  }

  for (unsigned int tile = token_start + warp * 32; tile < token_end;
       tile += IZWI_SPLIT_DECODE_WARPS * 32) {
    const unsigned int token = tile + lane;
    bool live = token < token_end;
    int row_offset = 0;
    float score[MAX_GROUP];
#pragma unroll
    for (int g = 0; g < MAX_GROUP; ++g) {
      score[g] = 0.0f;
    }
    if (live) {
      const unsigned int physical_token = token + first_page_offset;
      const unsigned int logical_page = physical_token / page_tokens;
      const unsigned int physical_page = block_table[logical_page];
      if (physical_page >= static_cast<unsigned int>(capacity_pages)) {
        shared_corrupt = 1;
        live = false;
      } else {
        row_offset = ((physical_page * page_tokens +
                       (physical_token - logical_page * page_tokens)) *
                          kv_heads +
                      kv_head) *
                     HEAD_DIM;
        const T* key = keys + row_offset;
#pragma unroll 4
        for (int d = 0; d < HEAD_DIM; d += KEY_CHUNK) {
          float key_chunk[KEY_CHUNK];
          izwi_load_floats<T, KEY_CHUNK>(key + d, key_chunk);
#pragma unroll
          for (int g = 0; g < MAX_GROUP; ++g) {
            if (g < group) {
              const float* query = shared_queries + g * HEAD_DIM + d;
#pragma unroll
              for (int j = 0; j < KEY_CHUNK; ++j) {
                score[g] += query[j] * key_chunk[j];
              }
            }
          }
        }
      }
    }

    // Lane 0 always holds a live token, so each tile maximum is finite.
#pragma unroll
    for (int g = 0; g < MAX_GROUP; ++g) {
      if (g < group) {
        float scaled = score[g] * softmax_scale;
        if (softcap > 0.0f) {
          scaled = softcap * tanhf(scaled / softcap);
        }
        float tile_max = live ? scaled : -INFINITY;
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
          tile_max = fmaxf(tile_max, __shfl_xor_sync(0xffffffffu, tile_max, offset));
        }
        const float next_max = fmaxf(running_max[g], tile_max);
        const float rescale = expf(running_max[g] - next_max);
        running_max[g] = next_max;
        score[g] = live ? expf(scaled - next_max) : 0.0f;
        running_sum[g] = running_sum[g] * rescale + score[g];
#pragma unroll
        for (int j = 0; j < LANE_DIMS; ++j) {
          accumulator[g][j] *= rescale;
        }
      }
    }

    const int tile_keys = static_cast<int>(min(32u, token_end - tile));
    for (int key = 0; key < tile_keys; ++key) {
      const int value_row = __shfl_sync(0xffffffffu, row_offset, key);
      float value[LANE_DIMS];
      izwi_load_floats<T, LANE_DIMS>(values + value_row + lane * LANE_DIMS, value);
#pragma unroll
      for (int g = 0; g < MAX_GROUP; ++g) {
        if (g < group) {
          const float weight = __shfl_sync(0xffffffffu, score[g], key);
#pragma unroll
          for (int j = 0; j < LANE_DIMS; ++j) {
            accumulator[g][j] += weight * value[j];
          }
        }
      }
    }
  }

#pragma unroll
  for (int g = 0; g < MAX_GROUP; ++g) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
      running_sum[g] += __shfl_xor_sync(0xffffffffu, running_sum[g], offset);
    }
  }
  if (lane == 0) {
#pragma unroll
    for (int g = 0; g < MAX_GROUP; ++g) {
      shared_max[warp][g] = running_max[g];
      shared_sum[warp][g] = running_sum[g];
    }
  }
  __syncthreads();

  // Merge the warps in a fixed order so the result is deterministic.
  float block_max[MAX_GROUP];
  float block_sum[MAX_GROUP];
  float warp_scale[MAX_GROUP];
  const bool corrupt = shared_corrupt != 0;
#pragma unroll
  for (int g = 0; g < MAX_GROUP; ++g) {
    block_max[g] = -INFINITY;
    for (int w = 0; w < IZWI_SPLIT_DECODE_WARPS; ++w) {
      block_max[g] = fmaxf(block_max[g], shared_max[w][g]);
    }
    block_sum[g] = 0.0f;
    warp_scale[g] = 0.0f;
    if (block_max[g] != -INFINITY && !corrupt) {
      for (int w = 0; w < IZWI_SPLIT_DECODE_WARPS; ++w) {
        if (shared_max[w][g] != -INFINITY) {
          block_sum[g] += shared_sum[w][g] * expf(shared_max[w][g] - block_max[g]);
        }
      }
      if (running_max[g] != -INFINITY) {
        warp_scale[g] = expf(running_max[g] - block_max[g]);
      }
    } else {
      block_max[g] = -INFINITY;
    }
  }
  for (int w = 0; w < IZWI_SPLIT_DECODE_WARPS; ++w) {
    if (warp == w) {
#pragma unroll
      for (int g = 0; g < MAX_GROUP; ++g) {
        if (g < group) {
#pragma unroll
          for (int j = 0; j < LANE_DIMS; ++j) {
            const int index = g * HEAD_DIM + lane * LANE_DIMS + j;
            const float scaled = accumulator[g][j] * warp_scale[g];
            shared_output[index] = w == 0 ? scaled : shared_output[index] + scaled;
          }
        }
      }
    }
    __syncthreads();
  }

  for (int i = threadIdx.x; i < group * HEAD_DIM; i += blockDim.x) {
    const int g = i / HEAD_DIM;
    const unsigned long long base =
        (static_cast<unsigned long long>(row * query_heads + kv_head * group + g) *
             num_partitions +
         partition) *
        partial_stride;
    partials[base + 2 + (i - g * HEAD_DIM)] =
        valid && !corrupt ? shared_output[i] : 0.0f;
  }
  if (threadIdx.x < group) {
    const int g = threadIdx.x;
    const unsigned long long base =
        (static_cast<unsigned long long>(row * query_heads + kv_head * group + g) *
             num_partitions +
         partition) *
        partial_stride;
    // Indexing by the thread's own head keeps the register arrays static.
    float head_max = -INFINITY;
    float head_sum = 0.0f;
#pragma unroll
    for (int h = 0; h < MAX_GROUP; ++h) {
      if (h == g) {
        head_max = block_max[h];
        head_sum = block_sum[h];
      }
    }
    partials[base] = valid ? head_max : -INFINITY;
    partials[base + 1] = valid ? head_sum : 0.0f;
  }
}

// Merges split-decode partials: one block per (row, query head). Unlike the
// serial partition reduction above, the partition weights are computed in one
// parallel pass, so each output dimension is an independent weighted sum.
template <typename T>
__device__ void izwi_paged_decode_split_reduce(
    const float* __restrict__ partials,
    T* __restrict__ output,
    int row_heads,
    int value_dim,
    int num_partitions) {
  __shared__ float weights[IZWI_SPLIT_DECODE_MAX_PARTITIONS];
  __shared__ float scratch[32];
  const int row_head = blockIdx.x;
  if (row_head >= row_heads || num_partitions > IZWI_SPLIT_DECODE_MAX_PARTITIONS) {
    return;
  }
  const unsigned long long stride = static_cast<unsigned long long>(value_dim) + 2;
  const float* row = partials + static_cast<unsigned long long>(row_head) * num_partitions * stride;
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int warps = (blockDim.x + 31) >> 5;

  float local_max = -INFINITY;
  for (int p = threadIdx.x; p < num_partitions; p += blockDim.x) {
    if (row[p * stride + 1] > 0.0f) {
      local_max = fmaxf(local_max, row[p * stride]);
    }
  }
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    local_max = fmaxf(local_max, __shfl_xor_sync(0xffffffffu, local_max, offset));
  }
  if (lane == 0) {
    scratch[warp] = local_max;
  }
  __syncthreads();
  float global_max = -INFINITY;
  for (int w = 0; w < warps; ++w) {
    global_max = fmaxf(global_max, scratch[w]);
  }
  __syncthreads();

  float local_sum = 0.0f;
  for (int p = threadIdx.x; p < num_partitions; p += blockDim.x) {
    const float sum = row[p * stride + 1];
    const float weight = sum > 0.0f ? expf(row[p * stride] - global_max) : 0.0f;
    weights[p] = weight;
    local_sum += sum * weight;
  }
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    local_sum += __shfl_xor_sync(0xffffffffu, local_sum, offset);
  }
  if (lane == 0) {
    scratch[warp] = local_sum;
  }
  __syncthreads();
  float total = 0.0f;
  for (int w = 0; w < warps; ++w) {
    total += scratch[w];
  }

  for (int d = threadIdx.x; d < value_dim; d += blockDim.x) {
    float value = 0.0f;
    for (int p = 0; p < num_partitions; ++p) {
      value += weights[p] * row[p * stride + 2 + d];
    }
    output[static_cast<unsigned long long>(row_head) * value_dim + d] =
        izwi_from_float<T>(total > 0.0f ? value / total : 0.0f);
  }
}

#define IZWI_DEFINE_PAGED_DECODE_SPLIT(SUFFIX, TYPE, DIM)                      \
  extern "C" __global__ void __launch_bounds__(IZWI_SPLIT_DECODE_WARPS * 32)  \
      physical_paged_decode_split_##SUFFIX##_d##DIM(                           \
          const TYPE* queries, const TYPE* keys, const TYPE* values,           \
          const unsigned int* metadata, float* partials, int batch,            \
          int query_heads, int kv_heads, int page_tokens, int max_blocks,      \
          int capacity_pages, int num_partitions, float softmax_scale,         \
          float softcap) {                                                     \
    izwi_paged_decode_gqa_partition<TYPE, DIM>(                                \
        queries, keys, values, metadata, partials, batch, query_heads,         \
        kv_heads, page_tokens, max_blocks, capacity_pages, num_partitions,     \
        softmax_scale, softcap);                                               \
  }

IZWI_DEFINE_PAGED_DECODE_SPLIT(f16, __half, 128)
IZWI_DEFINE_PAGED_DECODE_SPLIT(f16, __half, 256)
IZWI_DEFINE_PAGED_DECODE_SPLIT(bf16, __nv_bfloat16, 128)
IZWI_DEFINE_PAGED_DECODE_SPLIT(bf16, __nv_bfloat16, 256)

#define IZWI_DEFINE_PAGED_DECODE_SPLIT_REDUCE(SUFFIX, TYPE)                    \
  extern "C" __global__ void physical_paged_decode_split_reduce_##SUFFIX(     \
      const float* partials, TYPE* output, int row_heads, int value_dim,       \
      int num_partitions) {                                                    \
    izwi_paged_decode_split_reduce<TYPE>(partials, output, row_heads,          \
                                         value_dim, num_partitions);           \
  }

IZWI_DEFINE_PAGED_DECODE_SPLIT_REDUCE(f16, __half)
IZWI_DEFINE_PAGED_DECODE_SPLIT_REDUCE(bf16, __nv_bfloat16)

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
    int capacity_pages,
    float softmax_scale,
    float softcap) {
  izwi_paged_decode_attention(
      queries, keys, values, metadata, output, batch, query_heads, kv_heads,
      page_tokens, max_blocks, key_dim, value_dim, capacity_pages,
      softmax_scale, softcap);
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
    int capacity_pages,
    float softmax_scale,
    float softcap) {
  izwi_paged_decode_attention(
      queries, keys, values, metadata, output, batch, query_heads, kv_heads,
      page_tokens, max_blocks, key_dim, value_dim, capacity_pages,
      softmax_scale, softcap);
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
    int capacity_pages,
    float softmax_scale,
    float softcap) {
  izwi_paged_decode_attention(
      queries, keys, values, metadata, output, batch, query_heads, kv_heads,
      page_tokens, max_blocks, key_dim, value_dim, capacity_pages,
      softmax_scale, softcap);
}

#if defined(IZWI_CUDA_HAS_FP8)
#define IZWI_DEFINE_PAGED_DECODE_FP8(SUFFIX, TYPE)                            \
  extern "C" __global__ void physical_paged_decode_##SUFFIX##_fp8(          \
      const TYPE* queries, const __nv_fp8_e4m3* keys,                         \
      const __nv_fp8_e4m3* values, const unsigned int* metadata, TYPE* output,\
      int batch, int query_heads, int kv_heads, int page_tokens,              \
      int max_blocks, int key_dim, int value_dim, int capacity_pages,         \
      float softmax_scale, float softcap) {                                   \
    izwi_paged_decode_attention<TYPE, __nv_fp8_e4m3>(                         \
        queries, keys, values, metadata, output, batch, query_heads, kv_heads,\
        page_tokens, max_blocks, key_dim, value_dim, capacity_pages,          \
        softmax_scale, softcap);                                              \
  }

IZWI_DEFINE_PAGED_DECODE_FP8(f16, __half)
IZWI_DEFINE_PAGED_DECODE_FP8(bf16, __nv_bfloat16)
#endif

#define IZWI_DEFINE_PAGED_PREFILL(SUFFIX, TYPE)                              \
  extern "C" __global__ void physical_paged_prefill_##SUFFIX(              \
      const TYPE* queries, const TYPE* keys, const TYPE* values,              \
      const unsigned int* metadata, TYPE* output, int sequences,              \
      int total_queries, int query_heads, int kv_heads, int page_tokens,      \
      int max_blocks, int key_dim, int value_dim, int capacity_pages,         \
      int window_tokens, float softmax_scale, float softcap) {                \
    izwi_paged_prefill_attention(                                             \
        queries, keys, values, metadata, output, sequences, total_queries,    \
        query_heads, kv_heads, page_tokens, max_blocks, key_dim, value_dim,   \
        capacity_pages, window_tokens, softmax_scale, softcap);               \
  }

IZWI_DEFINE_PAGED_PREFILL(f32, float)
IZWI_DEFINE_PAGED_PREFILL(f16, __half)
IZWI_DEFINE_PAGED_PREFILL(bf16, __nv_bfloat16)

#if defined(IZWI_CUDA_HAS_FP8)
#define IZWI_DEFINE_PAGED_PREFILL_FP8(SUFFIX, TYPE)                           \
  extern "C" __global__ void physical_paged_prefill_##SUFFIX##_fp8(         \
      const TYPE* queries, const __nv_fp8_e4m3* keys,                         \
      const __nv_fp8_e4m3* values, const unsigned int* metadata, TYPE* output,\
      int sequences, int total_queries, int query_heads, int kv_heads,        \
      int page_tokens, int max_blocks, int key_dim, int value_dim,            \
      int capacity_pages, int window_tokens, float softmax_scale,             \
      float softcap) {                                                        \
    izwi_paged_prefill_attention<TYPE, __nv_fp8_e4m3>(                        \
        queries, keys, values, metadata, output, sequences, total_queries,   \
        query_heads, kv_heads, page_tokens, max_blocks, key_dim, value_dim,   \
        capacity_pages, window_tokens, softmax_scale, softcap);               \
  }

IZWI_DEFINE_PAGED_PREFILL_FP8(f16, __half)
IZWI_DEFINE_PAGED_PREFILL_FP8(bf16, __nv_bfloat16)
#endif

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
