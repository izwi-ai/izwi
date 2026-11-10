// Counter-based Gumbel noise (`kernels::counter_noise`): element j of row r
// is a function of keys[r] and j alone, so a row's noise is the same whoever
// shares the launch, and nothing is drawn on the host or uploaded but one
// 64-bit key a row. SplitMix64's finalizer over key + (j + 1) * golden gives
// 23 uniform bits, u = (k + 0.5) / 2^23 in (0, 1) exactly, then
// g = -log(-log(u)).

extern "C" __global__ void izwi_counter_gumbel(const unsigned long long* __restrict__ keys,
                                               float* __restrict__ out, long long per_row,
                                               long long total) {
  const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= total) return;
  const long long row = i / per_row;
  const long long j = i - row * per_row;
  unsigned long long z = keys[row] + (unsigned long long)(j + 1) * 0x9E3779B97F4A7C15ull;
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
  z ^= z >> 31;
  const float u = ((float)(unsigned)(z >> 41) + 0.5f) * (1.0f / 8388608.0f);
  out[i] = -logf(-logf(u));
}
