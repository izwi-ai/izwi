#include <cuda_bf16.h>
#include <cuda_fp16.h>

// Greedy pick of each logits row (`kernels::greedy`): the largest value
// (ties to the lowest id, NaN never picked) and its margin over the
// runner-up, which counts a duplicate of the best. One block per row writes
// `out[row] = {id, margin bits}`. The merge is exact (max and compare only),
// so the result never depends on the block's partition of the row.

#define GREEDY_THREADS 256
#define GREEDY_NONE 0xFFFFFFFFu

__device__ __forceinline__ float g_load(const float* p, long long i) { return p[i]; }
__device__ __forceinline__ float g_load(const __half* p, long long i) { return __half2float(p[i]); }
__device__ __forceinline__ float g_load(const __nv_bfloat16* p, long long i) {
  return __bfloat162float(p[i]);
}

struct GreedyTop {
  float best;
  unsigned id;
  float second;
};

__device__ __forceinline__ GreedyTop g_merge(GreedyTop a, GreedyTop b) {
  const bool a_wins = b.id == GREEDY_NONE ||
                      (a.id != GREEDY_NONE && (a.best > b.best || (a.best == b.best && a.id < b.id)));
  GreedyTop w = a_wins ? a : b;
  const GreedyTop l = a_wins ? b : a;
  const float runner = l.id == GREEDY_NONE ? l.second : fmaxf(l.best, l.second);
  w.second = fmaxf(w.second, runner);
  return w;
}

template <class T>
__device__ void greedy_row(const T* __restrict__ logits, unsigned* __restrict__ out, int vocab) {
  __shared__ float s_best[GREEDY_THREADS];
  __shared__ unsigned s_id[GREEDY_THREADS];
  __shared__ float s_second[GREEDY_THREADS];
  const long long base = (long long)blockIdx.x * vocab;
  GreedyTop top{-INFINITY, GREEDY_NONE, -INFINITY};
  for (int i = threadIdx.x; i < vocab; i += GREEDY_THREADS) {
    const float v = g_load(logits, base + i);
    if (top.id == GREEDY_NONE ? v > -INFINITY : v > top.best) {
      top.second = top.id == GREEDY_NONE ? top.second : top.best;
      top.best = v;
      top.id = (unsigned)i;
    } else if (v > top.second) {
      top.second = v;
    }
  }
  s_best[threadIdx.x] = top.best;
  s_id[threadIdx.x] = top.id;
  s_second[threadIdx.x] = top.second;
  __syncthreads();
  for (int stride = GREEDY_THREADS / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      const GreedyTop a{s_best[threadIdx.x], s_id[threadIdx.x], s_second[threadIdx.x]};
      const int o = threadIdx.x + stride;
      const GreedyTop b{s_best[o], s_id[o], s_second[o]};
      const GreedyTop m = g_merge(a, b);
      s_best[threadIdx.x] = m.best;
      s_id[threadIdx.x] = m.id;
      s_second[threadIdx.x] = m.second;
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    const bool none = s_id[0] == GREEDY_NONE;
    out[2 * blockIdx.x] = none ? 0u : s_id[0];
    out[2 * blockIdx.x + 1] = __float_as_uint(none ? NAN : s_best[0] - s_second[0]);
  }
}

#define GREEDY_EXPORT(NAME, T)                                                     \
  extern "C" __global__ void izwi_greedy_##NAME(const T* logits, unsigned* out,   \
                                               int vocab) {                       \
    greedy_row<T>(logits, out, vocab);                                            \
  }

GREEDY_EXPORT(f32, float)
GREEDY_EXPORT(f16, __half)
GREEDY_EXPORT(bf16, __nv_bfloat16)
