// Runs crates/izwi-core/src/kernels/cuda/paged_bias_sink.cu on the host and
// checks it against an independent F64 evaluation over dense keys and values
// (F32, F16 and BF16 storage), then checks that a row has the same bits
// alone and inside a batch.
//
// For machines without a CUDA toolkit or GPU. The kernel uses one shared
// array and one `__syncthreads()` per block and no warp primitives. This
// harness stubs them: a block is `blockDim.x` host threads (`threadIdx` is
// thread-local), `__syncthreads()` is a barrier across them, the dynamic
// shared memory is one static array, and blocks run one after another.
// `__half` and `__nv_bfloat16` are bit containers with exact widening
// conversions. It checks the kernel's arithmetic and indexing, not its
// behaviour on a GPU.
//
//   clang++ -std=c++17 -O1 -o <scratch>/paged_bias_sink scripts/cuda-host-emulation/paged_bias_sink.cpp
//   <scratch>/paged_bias_sink
//
// With a file argument it also replays the cases the Rust host reference
// wrote (`paged_bias_sink::tests::writes_the_cuda_emulator_cases`, run with
// IZWI_PAGED_BIAS_SINK_EMULATOR_DIR set) and requires the kernel's bits to
// equal the reference's for every storage type:
//
//   <scratch>/paged_bias_sink <dir>/paged_bias_sink_cases.bin
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <thread>
#include <vector>

struct Idx { unsigned x, y, z; };
static thread_local Idx threadIdx;
static Idx blockIdx, blockDim;
#define __global__
#define __device__
#define __forceinline__ inline
#define __restrict__
#define __shared__
#define IZWI_HOST_EMULATION 1

struct __half { uint16_t bits; };
struct __nv_bfloat16 { uint16_t bits; };
static float __int_as_float(int v) { float f; std::memcpy(&f, &v, 4); return f; }
static float __bfloat162float(__nv_bfloat16 v) { return __int_as_float((int)((uint32_t)v.bits << 16)); }
static float __half2float(__half v) {
  const uint32_t sign = (uint32_t)(v.bits >> 15) << 31, e = (v.bits >> 10) & 31, m = v.bits & 1023;
  if (e == 0) { float f = std::ldexp((float)m, -24); return sign ? -f : f; }
  return __int_as_float((int)(sign | ((e + 112) << 23) | (m << 13)));
}

// The block's dynamic shared memory and its barrier.
float pbs_shared[32 * (512 + 2)];
static std::mutex sync_mutex;
static std::condition_variable sync_cv;
static unsigned sync_waiting = 0, sync_generation = 0;
static void __syncthreads() {
  std::unique_lock<std::mutex> lock(sync_mutex);
  const unsigned generation = sync_generation;
  if (++sync_waiting == blockDim.x) { sync_waiting = 0; ++sync_generation; sync_cv.notify_all(); }
  else sync_cv.wait(lock, [&] { return sync_generation != generation; });
}

#include "../../crates/izwi-core/src/kernels/cuda/paged_bias_sink.cu"

template <class F> static void launch(unsigned blocks, unsigned threads, F f) {
  blockDim = {threads, 1, 1};
  for (unsigned b = 0; b < blocks; ++b) {
    blockIdx = {b, 0, 0};
    std::vector<std::thread> pool;
    for (unsigned t = 0; t < threads; ++t) pool.emplace_back([&, t] { threadIdx = {t, 0, 0}; f(); });
    for (auto& thread : pool) thread.join();
  }
}

static uint32_t state = 1;
static uint32_t next() { state ^= state << 13; state ^= state >> 17; state ^= state << 5; return state; }
static float unit() { return (float)next() / 4294967295.0f * 2.0f - 1.0f; }
// Multiples of 1/32 in [-2, 2]: exact in F32, F16 and BF16.
static float coarse() { return (float)((int)(next() % 129) - 64) / 32.0f; }
static float as_float(float v) { return v; }
static float as_float(__half v) { return __half2float(v); }
static float as_float(__nv_bfloat16 v) { return __bfloat162float(v); }
static void store(float& slot, float v) { slot = v; }
static void store(__nv_bfloat16& slot, float v) { uint32_t b; std::memcpy(&b, &v, 4); slot.bits = (uint16_t)(b >> 16); }
static void store(__half& slot, float v) {
  uint32_t b; std::memcpy(&b, &v, 4);
  if ((b & 0x7fffffffu) == 0) { slot.bits = (uint16_t)(b >> 16); return; }
  slot.bits = (uint16_t)(((b >> 31) << 15) | ((((b >> 23) & 255) - 112) << 10) | ((b >> 13) & 1023));
}

struct Table { std::vector<unsigned> blocks; unsigned offset, context; };
struct Case {
  const char* name;
  int heads, kv_heads, key_dim, value_dim, extent, page_tokens, capacity_pages, span;
  unsigned window, lanes;
  std::vector<Table> tables;
};

template <class KV, class Kernel> static int run(const Case& c, const char* dtype, Kernel kernel) {
  const int tables = (int)c.tables.size(), rows = tables * c.span;
  int max_blocks = 0;
  for (auto& t : c.tables) max_blocks = std::max(max_blocks, (int)t.blocks.size());
  std::vector<unsigned> metadata(2 * tables + tables * max_blocks, 0u);
  for (int t = 0; t < tables; ++t) {
    metadata[t] = c.tables[t].context; metadata[tables + t] = c.tables[t].offset;
    for (size_t b = 0; b < c.tables[t].blocks.size(); ++b) metadata[2 * tables + t * max_blocks + b] = c.tables[t].blocks[b];
  }
  const size_t slots = (size_t)c.capacity_pages * c.page_tokens;
  std::vector<KV> keys(slots * c.kv_heads * c.key_dim), values(slots * c.kv_heads * c.value_dim);
  for (auto& k : keys) store(k, coarse());
  for (auto& v : values) store(v, coarse());
  std::vector<float> queries((size_t)rows * c.heads * c.key_dim), profiles((size_t)rows * c.heads * c.extent), sinks((size_t)rows * c.heads);
  for (auto& q : queries) q = unit();
  for (auto& p : profiles) p = 2.0f * unit();
  for (auto& s : sinks) s = 3.0f * unit();
  const float scale = 1.0f / std::sqrt((float)c.key_dim);
  std::vector<float> out((size_t)rows * c.heads * c.value_dim, NAN);
  launch(rows * c.heads, c.lanes, [&] {
    kernel(queries.data(), profiles.data(), sinks.data(), keys.data(), values.data(), metadata.data(), out.data(), rows, c.span,
           c.heads, c.kv_heads, c.page_tokens, max_blocks, c.key_dim, c.value_dim, c.extent, c.capacity_pages, c.window, scale);
  });

  // Independent evaluation: the visible keys and values gathered dense,
  // every score computed, one softmax over the scores and the sink.
  double worst = 0, peak = 1e-9;
  for (int row = 0; row < rows; ++row) {
    const Table& t = c.tables[row / c.span];
    const long context = (long)t.context - (c.span - 1 - row % c.span);
    for (int head = 0; head < c.heads; ++head) {
      const int g = head / (c.heads / c.kv_heads);
      const size_t rh = (size_t)row * c.heads + head;
      std::vector<double> scores, want(c.value_dim, 0.0);
      std::vector<size_t> rowsOf;
      for (long token = 0; token < context; ++token) {
        const long distance = context - 1 - token;
        if (c.window && distance >= (long)c.window) continue;
        const size_t physical = t.offset + token;
        const size_t slot = (size_t)t.blocks[physical / c.page_tokens] * c.page_tokens + physical % c.page_tokens;
        double dot = 0;
        for (int d = 0; d < c.key_dim; ++d) dot += (double)queries[rh * c.key_dim + d] * as_float(keys[(slot * c.kv_heads + g) * c.key_dim + d]);
        scores.push_back(dot * scale + (distance < c.extent ? (double)profiles[rh * c.extent + distance] : 0.0));
        rowsOf.push_back(slot * c.kv_heads + g);
      }
      double top = sinks[rh];
      for (double s : scores) top = std::fmax(top, s);
      double total = std::exp(sinks[rh] - top);
      for (double s : scores) total += std::exp(s - top);
      for (size_t i = 0; i < scores.size(); ++i)
        for (int d = 0; d < c.value_dim; ++d) want[d] += std::exp(scores[i] - top) / total * as_float(values[rowsOf[i] * c.value_dim + d]);
      for (int d = 0; d < c.value_dim; ++d) {
        const double error = std::fabs(want[d] - out[rh * c.value_dim + d]);
        worst = std::fmax(worst, std::isfinite(error) ? error : INFINITY);
        peak = std::fmax(peak, std::fabs(want[d]));
      }
    }
  }

  // Each row alone (its own table, its own launch) has the batch's bits.
  int differing = 0;
  for (int row = 0; row < rows; ++row) {
    const Table& t = c.tables[row / c.span];
    const unsigned later = (unsigned)(c.span - 1 - row % c.span);
    std::vector<unsigned> one(2 + max_blocks, 0u);
    one[0] = t.context - later; one[1] = t.offset;
    for (size_t b = 0; b < t.blocks.size(); ++b) one[2 + b] = t.blocks[b];
    std::vector<float> alone((size_t)c.heads * c.value_dim, NAN);
    launch(c.heads, c.lanes, [&] {
      kernel(queries.data() + (size_t)row * c.heads * c.key_dim, profiles.data() + (size_t)row * c.heads * c.extent,
             sinks.data() + (size_t)row * c.heads, keys.data(), values.data(), one.data(), alone.data(), 1, 1, c.heads, c.kv_heads,
             c.page_tokens, max_blocks, c.key_dim, c.value_dim, c.extent, c.capacity_pages, c.window, scale);
    });
    differing += std::memcmp(alone.data(), out.data() + (size_t)row * c.heads * c.value_dim, alone.size() * 4) != 0;
  }
  const bool ok = worst <= 1e-5 * peak && peak > 1e-3 && differing == 0;
  std::printf("%-22s %-4s rows %2d lanes %2u: worst error %.3e at peak %.3e, %d rows differ alone vs batched%s\n",
              c.name, dtype, rows, c.lanes, worst, peak, differing, ok ? "" : "  <-- FAILED");
  return ok ? 0 : 1;
}

// Replays the Rust reference's cases: little-endian 32-bit words, floats as
// their bits. Keys and values are exact in every storage type, so one
// expected output serves the three kernels.
static int replay(const char* path) {
  FILE* file = std::fopen(path, "rb");
  if (!file) { std::printf("cannot open %s\n", path); return 1; }
  std::vector<uint32_t> words;
  for (uint32_t word; std::fread(&word, 4, 1, file) == 1;) words.push_back(word);
  std::fclose(file);
  size_t at = 0;
  auto u = [&] { return at < words.size() ? words[at++] : 0u; };
  auto floats = [&](size_t n) { std::vector<float> v(n); for (auto& x : v) { uint32_t w = u(); std::memcpy(&x, &w, 4); } return v; };
  if (u() != 0x31534250u) { std::printf("%s is not a case file\n", path); return 1; }
  const uint32_t count = u();
  int failures = 0;
  for (uint32_t index = 0; index < count; ++index) {
    const int rows = u(), span = u(), heads = u(), kv_heads = u(), page_tokens = u(), max_blocks = u(), key_dim = u(),
              value_dim = u(), extent = u(), capacity_pages = u();
    const unsigned window = u(), lanes = u();
    const float scale = floats(1)[0];
    const int tables = rows / span;
    std::vector<unsigned> metadata(tables * (2 + max_blocks));
    for (auto& m : metadata) m = u();
    const size_t rh = (size_t)rows * heads, slots = (size_t)capacity_pages * page_tokens * kv_heads;
    const auto queries = floats(rh * key_dim), profiles = floats(rh * extent), sinks = floats(rh);
    const auto keys = floats(slots * key_dim), values = floats(slots * value_dim), expected = floats(rh * value_dim);
    auto differing = [&](auto kv, auto kernel) {
      std::vector<decltype(kv)> k(keys.size()), v(values.size());
      for (size_t i = 0; i < k.size(); ++i) store(k[i], keys[i]);
      for (size_t i = 0; i < v.size(); ++i) store(v[i], values[i]);
      std::vector<float> out(expected.size(), NAN);
      launch(rows * heads, lanes, [&] {
        kernel(queries.data(), profiles.data(), sinks.data(), k.data(), v.data(), metadata.data(), out.data(), rows, span, heads,
               kv_heads, page_tokens, max_blocks, key_dim, value_dim, extent, capacity_pages, window, scale);
      });
      int n = 0;
      for (size_t i = 0; i < out.size(); ++i) n += std::memcmp(&out[i], &expected[i], 4) != 0;
      return n;
    };
    const int f32 = differing(0.0f, [](auto... a) { izwi_paged_bias_sink_f32(a...); });
    const int f16 = differing(__half{}, [](auto... a) { izwi_paged_bias_sink_f16(a...); });
    const int bf16 = differing(__nv_bfloat16{}, [](auto... a) { izwi_paged_bias_sink_bf16(a...); });
    std::printf("reference case %u (rows %d, heads %d, dims %d/%d, lanes %u): %zu values, differing bits f32 %d, f16 %d, bf16 %d%s\n",
                index, rows, heads, key_dim, value_dim, lanes, expected.size(), f32, f16, bf16, f32 + f16 + bf16 ? "  <-- FAILED" : "");
    failures += (f32 + f16 + bf16) != 0;
  }
  if (at != words.size() || count == 0) { std::printf("%s is truncated or empty\n", path); ++failures; }
  return failures;
}

int main(int argc, char** argv) {
  // Twelve pages of four tokens; tables are shuffled and ragged.
  const std::vector<Table> ragged = {
      {{7, 2, 9, 0}, 1, 13}, {{5}, 3, 1}, {{11, 4, 1}, 0, 9}, {{3, 10, 6, 8}, 2, 14}};
  // 20 pages of eight tokens: contexts past one stride of 32 lanes.
  const std::vector<Table> longer = {
      {{12, 3, 17, 0, 9, 5, 14, 1, 19, 7, 2}, 5, 83}, {{4, 16, 8, 11, 6}, 0, 40}, {{13}, 7, 1}};
  const std::vector<Case> cases = {
      {"gqa ragged", 6, 2, 20, 12, 5, 4, 12, 1, 0, 32, ragged},
      {"gqa ragged window", 6, 2, 20, 12, 16, 4, 12, 1, 6, 32, ragged},
      {"long extent short", 4, 4, 16, 16, 24, 8, 20, 1, 0, 32, longer},
      {"long window", 3, 1, 9, 33, 40, 8, 20, 1, 40, 32, longer},
      {"prefill span", 6, 2, 20, 12, 7, 4, 12, 5, 0, 32, {{{7, 2, 9, 0}, 1, 13}, {{3, 10, 6, 8}, 2, 14}}},
      {"prefill span window", 4, 2, 8, 8, 6, 8, 20, 9, 6, 32, {{{12, 3, 17, 0, 9, 5}, 5, 43}}},
      {"sixteen lanes", 4, 2, 24, 300, 30, 8, 20, 1, 0, 16, longer},
      {"one token", 24, 8, 16, 16, 8, 4, 12, 1, 0, 32, {{{5}, 3, 1}}},
  };
  int failures = 0;
  for (const Case& c : cases) {
    state = 4242;
    failures += run<float>(c, "f32", [](auto... a) { izwi_paged_bias_sink_f32(a...); });
    state = 4242;
    failures += run<__half>(c, "f16", [](auto... a) { izwi_paged_bias_sink_f16(a...); });
    state = 4242;
    failures += run<__nv_bfloat16>(c, "bf16", [](auto... a) { izwi_paged_bias_sink_bf16(a...); });
  }
  if (argc > 1) failures += replay(argv[1]);
  std::printf(failures ? "FAILED (%d)\n" : "ok\n", failures);
  return failures != 0;
}
