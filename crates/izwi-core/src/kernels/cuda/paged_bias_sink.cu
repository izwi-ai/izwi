#ifndef IZWI_HOST_EMULATION
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#endif
#include <math.h>

// Paged decode attention with an additive position bias and a sink logit
// (`kernels::paged_bias_sink`).
//
// A query at the end of its context scores the key at backward distance d
// (0 is its own key) as
//
//   dot(q, k) * scale + (d < extent ? profile[d] : 0)
//
// over the keys with d < min(context, window), and one extra logit, the
// sink, joins the softmax without a value. Queries, profiles, sinks and the
// output are F32; keys and values stay in the arena's storage type and its
// physical [page, token, kv_head, dim] layout.
//
// One block per (row, query head), `blockDim.x` lanes (32, or 16 for value
// rows too wide for 32 lanes of shared memory). Lane l owns the keys at
// distances l, l + lanes, ... nearest first and runs an online softmax over
// them alone; after one barrier every lane merges the lane states with the
// sink in lane order and writes its share of the output. A row's arithmetic
// depends on its own context only, so its output has the same bits whichever
// rows share the launch, and the grid depends on no length read on the host.
//
// Every multiply-add is an explicit fmaf and the exponential is built from
// them, so the host reference (`paged_bias_sink::reference`) reproduces the
// kernel operation for operation.
//
// Metadata is `[context[T] | first_page_offset[T] | table[T x max_blocks]]`
// for T = rows / span block tables. Each table serves `span` consecutive
// rows: the last one sees the table's whole context, the one before it one
// token less, and so on (a prefill span of one sequence; span 1 is a batch
// of decode rows).

#define PBS_MAX_LANES 32

__device__ __forceinline__ float pbs_load(const float* p, unsigned long long i) { return p[i]; }
__device__ __forceinline__ float pbs_load(const __half* p, unsigned long long i) {
  return __half2float(p[i]);
}
__device__ __forceinline__ float pbs_load(const __nv_bfloat16* p, unsigned long long i) {
  return __bfloat162float(p[i]);
}

// `relpos_attention::det_exp`: Cody-Waite reduction and Cephes' expf
// polynomial, the same literals and operations.
__device__ __forceinline__ float pbs_exp(float x) {
  if (isnan(x)) return x;
  if (x <= -104.0f) return 0.0f;
  const float n = rintf(x * 1.442695f);
  float r = fmaf(-n, 0.6933594f, x);
  r = fmaf(-n, -0.00021219444f, r);
  float p = 0.00019875691f;
  p = fmaf(p, r, 0.0013981999f);
  p = fmaf(p, r, 0.008333452f);
  p = fmaf(p, r, 0.041665796f);
  p = fmaf(p, r, 0.16666666f);
  p = fmaf(p, r, 0.5f);
  const float rr = r * r;
  const float y = fmaf(p, rr, r) + 1.0f;
  const int e = (int)n;
  if (e >= -126) return y * __int_as_float((e + 127) << 23);
  const float low = y * __int_as_float((e + 127 + 64) << 23);
  return low * __int_as_float((127 - 64) << 23);
}

// q . k in eight chains (dim d joins chain d % 8), summed pairwise.
template <typename KV>
__device__ __forceinline__ float pbs_dot(const float* __restrict__ q, const KV* __restrict__ k,
                                         unsigned long long base, int dim) {
  float c0 = 0.0f, c1 = 0.0f, c2 = 0.0f, c3 = 0.0f;
  float c4 = 0.0f, c5 = 0.0f, c6 = 0.0f, c7 = 0.0f;
  int d = 0;
  for (; d + 8 <= dim; d += 8) {
    c0 = fmaf(q[d], pbs_load(k, base + d), c0);
    c1 = fmaf(q[d + 1], pbs_load(k, base + d + 1), c1);
    c2 = fmaf(q[d + 2], pbs_load(k, base + d + 2), c2);
    c3 = fmaf(q[d + 3], pbs_load(k, base + d + 3), c3);
    c4 = fmaf(q[d + 4], pbs_load(k, base + d + 4), c4);
    c5 = fmaf(q[d + 5], pbs_load(k, base + d + 5), c5);
    c6 = fmaf(q[d + 6], pbs_load(k, base + d + 6), c6);
    c7 = fmaf(q[d + 7], pbs_load(k, base + d + 7), c7);
  }
  // The tail dims continue chains 0, 1, ... in order.
  if (d < dim) c0 = fmaf(q[d], pbs_load(k, base + d), c0);
  if (d + 1 < dim) c1 = fmaf(q[d + 1], pbs_load(k, base + d + 1), c1);
  if (d + 2 < dim) c2 = fmaf(q[d + 2], pbs_load(k, base + d + 2), c2);
  if (d + 3 < dim) c3 = fmaf(q[d + 3], pbs_load(k, base + d + 3), c3);
  if (d + 4 < dim) c4 = fmaf(q[d + 4], pbs_load(k, base + d + 4), c4);
  if (d + 5 < dim) c5 = fmaf(q[d + 5], pbs_load(k, base + d + 5), c5);
  if (d + 6 < dim) c6 = fmaf(q[d + 6], pbs_load(k, base + d + 6), c6);
  const float a = c0 + c1;
  const float b = c2 + c3;
  const float c = c4 + c5;
  const float e = c6 + c7;
  const float ab = a + b;
  const float ce = c + e;
  return ab + ce;
}

template <typename KV>
__device__ void pbs_attend(
    const float* __restrict__ queries,
    const float* __restrict__ profiles,
    const float* __restrict__ sinks,
    const KV* __restrict__ keys,
    const KV* __restrict__ values,
    const unsigned int* __restrict__ metadata,
    float* __restrict__ output,
    int rows,
    int span,
    int heads,
    int kv_heads,
    int page_tokens,
    int max_blocks,
    int key_dim,
    int value_dim,
    int extent,
    int capacity_pages,
    unsigned int window,
    float scale) {
  // Lane states: [max, sum, accumulator[value_dim]] per lane.
  extern __shared__ float pbs_shared[];
  const int lanes = (int)blockDim.x;
  const int lane = (int)threadIdx.x;
  const int row_head = (int)blockIdx.x;
  if (row_head >= rows * heads || lanes > PBS_MAX_LANES) {
    return;
  }
  const int row = row_head / heads;
  const int head = row_head - row * heads;
  const int kv_head = head / (heads / kv_heads);
  const int tables = rows / span;
  const int table = row / span;
  // Tokens of the table's context that come after this row's query.
  const unsigned int later = (unsigned int)(span - 1 - (row - table * span));
  const unsigned int table_context = metadata[table];
  const unsigned int first_page_offset = metadata[tables + table];
  const unsigned int* blocks =
      metadata + 2 * (unsigned long long)tables + (unsigned long long)table * max_blocks;

  // Leave zeros if a caller bypasses the host's plan validation. Every
  // condition is the same for all lanes of the block.
  bool valid = page_tokens > 0 && max_blocks > 0 && capacity_pages > 0 &&
               table_context > later &&
               first_page_offset < (unsigned int)page_tokens;
  if (valid) {
    const unsigned long long pages =
        ((unsigned long long)table_context + first_page_offset + page_tokens - 1) /
        (unsigned long long)page_tokens;
    valid = pages <= (unsigned long long)max_blocks;
    for (unsigned long long page = 0; valid && page < pages; ++page) {
      valid = blocks[page] < (unsigned int)capacity_pages;
    }
  }

  const int stride = value_dim + 2;
  float* state = pbs_shared + (unsigned long long)lane * stride;
  float* acc = state + 2;
  for (int dim = 0; dim < value_dim; ++dim) {
    acc[dim] = 0.0f;
  }
  float lane_max = -INFINITY;
  float lane_sum = 0.0f;
  if (valid) {
    const unsigned int context = table_context - later;
    const unsigned int visible = (window != 0u && window < context) ? window : context;
    const float* q = queries + (unsigned long long)row_head * key_dim;
    const float* profile = profiles + (unsigned long long)row_head * extent;
    for (unsigned int distance = (unsigned int)lane; distance < visible;
         distance += (unsigned int)lanes) {
      const unsigned int token = first_page_offset + (context - 1u - distance);
      const unsigned int logical_page = token / (unsigned int)page_tokens;
      const unsigned long long slot =
          (unsigned long long)blocks[logical_page] * page_tokens +
          (token - logical_page * (unsigned int)page_tokens);
      const unsigned long long kv_row = slot * kv_heads + kv_head;
      const float dot = pbs_dot(q, keys, kv_row * key_dim, key_dim);
      const float bias = distance < (unsigned int)extent ? profile[distance] : 0.0f;
      const float score = fmaf(dot, scale, bias);
      float previous_weight = 1.0f;
      float token_weight = 1.0f;
      if (score > lane_max) {
        previous_weight = pbs_exp(lane_max - score);
        lane_max = score;
      } else {
        token_weight = pbs_exp(score - lane_max);
      }
      lane_sum = fmaf(lane_sum, previous_weight, token_weight);
      const unsigned long long value_base = kv_row * value_dim;
      for (int dim = 0; dim < value_dim; ++dim) {
        const float weighted = pbs_load(values, value_base + dim) * token_weight;
        acc[dim] = fmaf(acc[dim], previous_weight, weighted);
      }
    }
  }
  state[0] = lane_max;
  state[1] = lane_sum;
  __syncthreads();

  float* out = output + (unsigned long long)row_head * value_dim;
  if (!valid) {
    for (int dim = lane; dim < value_dim; dim += lanes) {
      out[dim] = 0.0f;
    }
    return;
  }
  // The sink first, then the lanes in order.
  const float sink = sinks[row_head];
  float top = sink;
  for (int l = 0; l < lanes; ++l) {
    const float other = pbs_shared[(unsigned long long)l * stride];
    if (other > top) top = other;
  }
  float weights[PBS_MAX_LANES];
  float total = pbs_exp(sink - top);
  for (int l = 0; l < lanes; ++l) {
    const float* other = pbs_shared + (unsigned long long)l * stride;
    weights[l] = pbs_exp(other[0] - top);
    total = fmaf(other[1], weights[l], total);
  }
  for (int dim = lane; dim < value_dim; dim += lanes) {
    float sum = 0.0f;
    for (int l = 0; l < lanes; ++l) {
      sum = fmaf(pbs_shared[(unsigned long long)l * stride + 2 + dim], weights[l], sum);
    }
    out[dim] = sum / total;
  }
}

#define IZWI_DEFINE_PAGED_BIAS_SINK(suffix, KV)                                  \
  extern "C" __global__ void izwi_paged_bias_sink_##suffix(                      \
      const float* queries, const float* profiles, const float* sinks,           \
      const KV* keys, const KV* values, const unsigned int* metadata,            \
      float* output, int rows, int span, int heads, int kv_heads,                \
      int page_tokens, int max_blocks, int key_dim, int value_dim, int extent,   \
      int capacity_pages, unsigned int window, float scale) {                    \
    pbs_attend<KV>(queries, profiles, sinks, keys, values, metadata, output,     \
                   rows, span, heads, kv_heads, page_tokens, max_blocks,         \
                   key_dim, value_dim, extent, capacity_pages, window, scale);   \
  }

IZWI_DEFINE_PAGED_BIAS_SINK(f32, float)
IZWI_DEFINE_PAGED_BIAS_SINK(f16, __half)
IZWI_DEFINE_PAGED_BIAS_SINK(bf16, __nv_bfloat16)
