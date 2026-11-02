// The exponential-race sampler with temperature, top-k and top-p in one
// block per row (`kernels::race_sampler`). Over candidates [first, vocab):
//
//   s_i    = logit_i / temperature
//   rank   = s descending, ties by candidate index (a stable sort)
//   p_r    = exp(s_r - max) / sum_i exp(s_i - max)
//   keep r = r == 0 || (r < cap && cumsum(p)[r] <= top_p)
//   pick   = the first r of the kept prefix maximising p_r / noise[cand_r]
//
// Only the top R ranks (R <= 512) are materialised: a radix select finds the
// R-th largest s, the candidates above it and the lowest-index ties at it
// are gathered and bitonic-sorted. When every one of R < candidates ranks is
// kept and the cap allows more, `overflow` asks the caller to redraw on the
// host. out[row] = {pick, non-finite logits in the row, overflow}. Every
// reduction runs in a fixed order, so a row's draw depends on that row alone.

#define RS_THREADS 256
#define RS_MAX_RANKS 512

__device__ __forceinline__ unsigned rs_key(float v) {
  const unsigned bits = __float_as_uint(v);
  return (bits & 0x80000000u) ? ~bits : (bits | 0x80000000u);
}

__device__ __forceinline__ bool rs_before(unsigned ka, unsigned ia, unsigned kb, unsigned ib) {
  return ka > kb || (ka == kb && ia < ib);
}

extern "C" __global__ void izwi_race_sample(const float* __restrict__ logits,
                                            const float* __restrict__ noise,
                                            unsigned* __restrict__ out, int vocab,
                                            int first, float temperature, int cap,
                                            float top_p, int ranks) {
  __shared__ unsigned keys[RS_MAX_RANKS];
  __shared__ unsigned idx[RS_MAX_RANKS];
  __shared__ float f_scratch[RS_THREADS];
  __shared__ unsigned u_scratch[RS_THREADS];
  __shared__ unsigned slot;
  const int tid = threadIdx.x;
  const long long base = (long long)blockIdx.x * vocab;
  const float* row = logits + base;
  const float* nz = noise + base;
  const int candidates = vocab - first;
  const int r_count = min(ranks, candidates);

  // Non-finite logits anywhere in the row.
  unsigned bad = 0;
  for (int i = tid; i < vocab; i += RS_THREADS) bad += isfinite(row[i]) ? 0u : 1u;
  u_scratch[tid] = bad;
  __syncthreads();
  for (int s = RS_THREADS / 2; s > 0; s >>= 1) {
    if (tid < s) u_scratch[tid] += u_scratch[tid + s];
    __syncthreads();
  }
  const unsigned non_finite = u_scratch[0];
  __syncthreads();
  if (non_finite > 0 || candidates <= 0) {
    if (tid == 0) {
      out[3 * blockIdx.x] = (unsigned)first;
      out[3 * blockIdx.x + 1] = non_finite;
      out[3 * blockIdx.x + 2] = 0;
    }
    return;
  }

  // The largest tempered logit, then the softmax total.
  float m = -INFINITY;
  for (int i = first + tid; i < vocab; i += RS_THREADS) m = fmaxf(m, row[i] / temperature);
  f_scratch[tid] = m;
  __syncthreads();
  for (int s = RS_THREADS / 2; s > 0; s >>= 1) {
    if (tid < s) f_scratch[tid] = fmaxf(f_scratch[tid], f_scratch[tid + s]);
    __syncthreads();
  }
  const float max_s = f_scratch[0];
  __syncthreads();
  float acc = 0.0f;
  for (int i = first + tid; i < vocab; i += RS_THREADS) acc += expf(row[i] / temperature - max_s);
  f_scratch[tid] = acc;
  __syncthreads();
  for (int s = RS_THREADS / 2; s > 0; s >>= 1) {
    if (tid < s) f_scratch[tid] += f_scratch[tid + s];
    __syncthreads();
  }
  const float inverse = 1.0f / f_scratch[0];
  __syncthreads();

  // Radix select: the largest key t with at least r_count keys >= t.
  unsigned prefix = 0;
  for (int bit = 31; bit >= 0; --bit) {
    const unsigned t = prefix | (1u << bit);
    unsigned count = 0;
    for (int i = first + tid; i < vocab; i += RS_THREADS) count += rs_key(row[i] / temperature) >= t;
    u_scratch[tid] = count;
    __syncthreads();
    for (int s = RS_THREADS / 2; s > 0; s >>= 1) {
      if (tid < s) u_scratch[tid] += u_scratch[tid + s];
      __syncthreads();
    }
    if (u_scratch[0] >= (unsigned)r_count) prefix = t;
    __syncthreads();
  }

  // Gather the keys above the threshold, then the lowest-index ties at it.
  if (tid == 0) slot = 0;
  for (int i = tid; i < RS_MAX_RANKS; i += RS_THREADS) {
    keys[i] = 0u;
    idx[i] = 0xFFFFFFFFu;
  }
  __syncthreads();
  for (int i = first + tid; i < vocab; i += RS_THREADS) {
    const unsigned k = rs_key(row[i] / temperature);
    if (k > prefix) {
      const unsigned at = atomicAdd(&slot, 1u);
      keys[at] = k;
      idx[at] = (unsigned)i;
    }
  }
  __syncthreads();
  const unsigned above = slot;
  const int chunk = (candidates + RS_THREADS - 1) / RS_THREADS;
  const int lo = first + tid * chunk, hi = min(vocab, lo + chunk);
  unsigned ties = 0;
  for (int i = lo; i < hi; ++i) ties += rs_key(row[i] / temperature) == prefix;
  u_scratch[tid] = ties;
  __syncthreads();
  if (tid == 0) {
    unsigned running = 0;
    for (int t = 0; t < RS_THREADS; ++t) {
      const unsigned c = u_scratch[t];
      u_scratch[t] = running;
      running += c;
    }
  }
  __syncthreads();
  unsigned at = above + u_scratch[tid];
  for (int i = lo; i < hi && at < (unsigned)r_count; ++i) {
    const unsigned k = rs_key(row[i] / temperature);
    if (k == prefix) {
      keys[at] = k;
      idx[at] = (unsigned)i;
      ++at;
    }
  }
  __syncthreads();

  // Bitonic sort of the r_count entries (padding sorts last).
  int width = 1;
  while (width < r_count) width <<= 1;
  for (int size = 2; size <= width; size <<= 1) {
    for (int stride = size >> 1; stride > 0; stride >>= 1) {
      for (int i = tid; i < width; i += RS_THREADS) {
        const int j = i ^ stride;
        if (j > i) {
          const bool ascending = (i & size) == 0;
          const bool swap = ascending ? rs_before(keys[j], idx[j], keys[i], idx[i])
                                      : rs_before(keys[i], idx[i], keys[j], idx[j]);
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

  // The kept prefix and the race, in rank order.
  if (tid == 0) {
    const int limit = cap > 0 ? cap : 0x7FFFFFFF;
    float cumulative = 0.0f;
    int kept = 0;
    for (int r = 0; r < r_count; ++r) {
      cumulative += expf(row[idx[r]] / temperature - max_s) * inverse;
      if (r > 0 && (r >= limit || cumulative > top_p)) break;
      ++kept;
    }
    unsigned best = idx[0];
    float best_score = -INFINITY;
    for (int r = 0; r < kept; ++r) {
      const float score = expf(row[idx[r]] / temperature - max_s) * inverse / nz[idx[r]];
      if (score > best_score) {
        best_score = score;
        best = idx[r];
      }
    }
    out[3 * blockIdx.x] = best;
    out[3 * blockIdx.x + 1] = 0;
    out[3 * blockIdx.x + 2] =
        (kept == r_count && r_count < candidates && limit > r_count) ? 1u : 0u;
  }
}
