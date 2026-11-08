// The top candidates of each logits row after a host sampler's own
// per-token arithmetic (`kernels::penalized_topk`), one block per output
// row. Every value is the host's, bit for bit (IEEE round-to-nearest
// intrinsics, never contracted into an FMA):
//
//   m_i = c_i                      (c_i = logits[cond row][i])
//   m_i = c_i + w (c_i - u_i)      (classifier-free guidance, when set)
//   m_i = m_i * scale
//   m_i = m_i + bias               (at one index, when set)
//   g_i = m_i < 0 ? m_i p : m_i / p   (repetition, at the listed ids)
//   g_i = m_i - p                     (presence, at the listed ids)
//   g_i = -inf                        (banned range)
//
// The `count` largest g (by total order, ties to the lowest index) come back
// sorted, with their indices, plus how many raw inputs or m were not finite
// (the caller then samples that row from its whole logits). An 8-bit radix
// select finds the count-th largest key; the keys above it and the
// lowest-index ties at it are bitonic-sorted. Nothing depends on the
// block's partition of the row.

#define PT_THREADS 256
#define PT_MAX_COUNT 256
#define PT_PARAMS 12
#define PT_NONE 0xFFFFFFFFu

__device__ __forceinline__ unsigned pt_key(float v) {
  const unsigned bits = __float_as_uint(v);
  return (bits & 0x80000000u) ? ~bits : (bits | 0x80000000u);
}

__device__ __forceinline__ bool pt_before(unsigned ka, unsigned ia, unsigned kb, unsigned ib) {
  return ka > kb || (ka == kb && ia < ib);
}

// m_i, and whether a raw input or m_i is not finite.
__device__ __forceinline__ float pt_base(const float* __restrict__ logits, long long width,
                                         const unsigned* __restrict__ p, unsigned i, bool* bad) {
  const float c = logits[(long long)p[0] * width + i];
  float m = c;
  bool b = !isfinite(c);
  if (p[1] != PT_NONE) {
    const float u = logits[(long long)p[1] * width + i];
    b = b || !isfinite(u);
    m = __fadd_rn(c, __fmul_rn(__uint_as_float(p[2]), __fsub_rn(c, u)));
  }
  m = __fmul_rn(m, __uint_as_float(p[3]));
  if (i == p[4]) m = __fadd_rn(m, __uint_as_float(p[5]));
  *bad = b || !isfinite(m);
  return m;
}

__device__ __forceinline__ float pt_penalize(float m, unsigned mode, float penalty) {
  if (mode == 1u) return m < 0.0f ? __fmul_rn(m, penalty) : __fdiv_rn(m, penalty);
  if (mode == 2u) return __fsub_rn(m, penalty);
  return m;
}

// params[row] = {cond row, uncond row | NONE, guidance weight bits, scale
// bits, bias index | NONE, bias bits, penalty mode (0 none, 1 repetition,
// 2 presence), penalty bits, banned lo, banned hi, ids start, ids end}.
// scratch [rows, vocab] F32; out [rows, 2 count + 1] U32: value bits, then
// indices, then the non-finite count.
extern "C" __global__ void izwi_penalized_topk(const float* __restrict__ logits, int width,
                                               int vocab, const unsigned* __restrict__ params,
                                               const unsigned* __restrict__ ids,
                                               float* __restrict__ scratch,
                                               unsigned* __restrict__ out, int count) {
  __shared__ unsigned keys[PT_MAX_COUNT];
  __shared__ unsigned idx[PT_MAX_COUNT];
  __shared__ unsigned u_scratch[PT_THREADS];
  __shared__ unsigned s_prefix, s_need, slot;
  const int tid = threadIdx.x;
  const unsigned* p = params + (long long)blockIdx.x * PT_PARAMS;
  float* g = scratch + (long long)blockIdx.x * vocab;
  unsigned* o = out + (long long)blockIdx.x * (2 * count + 1);
  const unsigned mode = p[6];
  const float penalty = __uint_as_float(p[7]);
  const unsigned ban_lo = p[8], ban_hi = p[9];
  const int r_count = min(count, vocab);

  // g without the penalty, and the non-finite count.
  unsigned bad = 0;
  for (int i = tid; i < vocab; i += PT_THREADS) {
    bool b;
    const float m = pt_base(logits, width, p, (unsigned)i, &b);
    bad += b ? 1u : 0u;
    g[i] = ((unsigned)i >= ban_lo && (unsigned)i < ban_hi) ? -INFINITY : m;
  }
  __syncthreads();
  // The listed ids are distinct, so no two threads write one entry.
  if (mode != 0u) {
    for (unsigned j = p[10] + tid; j < p[11]; j += PT_THREADS) {
      const unsigned i = ids[j];
      if (i < (unsigned)vocab && !(i >= ban_lo && i < ban_hi)) {
        bool b;
        g[i] = pt_penalize(pt_base(logits, width, p, i, &b), mode, penalty);
      }
    }
  }
  u_scratch[tid] = bad;
  __syncthreads();
  for (int s = PT_THREADS / 2; s > 0; s >>= 1) {
    if (tid < s) u_scratch[tid] += u_scratch[tid + s];
    __syncthreads();
  }
  const unsigned non_finite = u_scratch[0];
  __syncthreads();

  // Radix select, a byte at a time: the key of the r_count-th largest, and
  // how many of its ties to take.
  unsigned prefix = 0, mask = 0, need = (unsigned)r_count;
  for (int shift = 24; shift >= 0; shift -= 8) {
    u_scratch[tid] = 0;
    __syncthreads();
    for (int i = tid; i < vocab; i += PT_THREADS) {
      const unsigned k = pt_key(g[i]);
      if ((k & mask) == prefix) atomicAdd(&u_scratch[(k >> shift) & 255u], 1u);
    }
    __syncthreads();
    if (tid == 0) {
      unsigned above = 0;
      int bin = 255;
      for (; bin > 0; --bin) {
        if (above + u_scratch[bin] >= need) break;
        above += u_scratch[bin];
      }
      s_prefix = prefix | ((unsigned)bin << shift);
      s_need = need - above;
    }
    __syncthreads();
    prefix = s_prefix;
    need = s_need;
    mask |= 255u << shift;
    __syncthreads();
  }

  // The keys above the threshold, then its lowest-index ties.
  if (tid == 0) slot = 0;
  for (int i = tid; i < PT_MAX_COUNT; i += PT_THREADS) {
    keys[i] = 0u;
    idx[i] = PT_NONE;
  }
  __syncthreads();
  for (int i = tid; i < vocab; i += PT_THREADS) {
    const unsigned k = pt_key(g[i]);
    if (k > prefix) {
      const unsigned at = atomicAdd(&slot, 1u);
      keys[at] = k;
      idx[at] = (unsigned)i;
    }
  }
  __syncthreads();
  const unsigned above = slot;
  const int chunk = (vocab + PT_THREADS - 1) / PT_THREADS;
  const int lo = min(vocab, tid * chunk), hi = min(vocab, lo + chunk);
  unsigned ties = 0;
  for (int i = lo; i < hi; ++i) ties += pt_key(g[i]) == prefix;
  u_scratch[tid] = ties;
  __syncthreads();
  if (tid == 0) {
    unsigned running = 0;
    for (int t = 0; t < PT_THREADS; ++t) {
      const unsigned c = u_scratch[t];
      u_scratch[t] = running;
      running += c;
    }
  }
  __syncthreads();
  unsigned at = above + u_scratch[tid];
  for (int i = lo; i < hi && at < (unsigned)r_count; ++i) {
    const unsigned k = pt_key(g[i]);
    if (k == prefix) {
      keys[at] = k;
      idx[at] = (unsigned)i;
      ++at;
    }
  }
  __syncthreads();

  // Bitonic sort, largest key first and ties by index (padding last).
  int sorted = 1;
  while (sorted < r_count) sorted <<= 1;
  for (int size = 2; size <= sorted; size <<= 1) {
    for (int stride = size >> 1; stride > 0; stride >>= 1) {
      for (int i = tid; i < sorted; i += PT_THREADS) {
        const int j = i ^ stride;
        if (j > i) {
          const bool ascending = (i & size) == 0;
          const bool swap = ascending ? pt_before(keys[j], idx[j], keys[i], idx[i])
                                      : pt_before(keys[i], idx[i], keys[j], idx[j]);
          if (swap) {
            const unsigned tk = keys[i], ti = idx[i];
            keys[i] = keys[j];
            idx[i] = idx[j];
            keys[j] = tk;
            idx[j] = ti;
          }
        }
      }
      __syncthreads();
    }
  }

  for (int r = tid; r < count; r += PT_THREADS) {
    const bool real = r < r_count;
    o[r] = real ? __float_as_uint(g[idx[r]]) : __float_as_uint(-INFINITY);
    o[count + r] = real ? idx[r] : PT_NONE;
  }
  if (tid == 0) o[2 * count] = non_finite;
}
