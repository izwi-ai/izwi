// Runs crates/izwi-core/src/kernels/cuda/moe_packed.cu on the host, one
// "thread" at a time, and checks it against an independent F64 evaluation
// over decoded weights (routing ids and weights, both packed formats).
//
// For machines without a CUDA toolkit or GPU: the kernels use no warp
// primitives or shared memory, so each launch is a loop over the grid. It
// checks the kernels' arithmetic and indexing, not their behaviour on a GPU.
//
//   clang++ -std=c++17 -O1 -o /tmp/moe_packed scripts/cuda-host-emulation/moe_packed.cpp
//   /tmp/moe_packed
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>
struct Idx { unsigned x, y, z; };
static Idx threadIdx, blockIdx, blockDim;
#define __global__
#define __device__
#define __forceinline__ inline
#define __restrict__
#include "../../crates/izwi-core/src/kernels/cuda/moe_packed.cu"

static uint32_t state = 1;
static uint32_t next() { state ^= state << 13; state ^= state >> 17; state ^= state << 5; return state; }
static float unit() { return (float)next() / 4294967295.0f * 2.0f - 1.0f; }
static double e4m3d(unsigned b) { unsigned e=(b>>3)&15, m=b&7; double v = e==0 ? m/8.0*std::pow(2.0,-6) : (1+m/8.0)*std::pow(2.0,(int)e-7); return (b&128)?-v:v; }
static const double E2M1[16] = {0,.5,1,1.5,2,3,4,6,-0.0,-.5,-1,-1.5,-2,-3,-4,-6};

template <class F> static void launch(unsigned gx, unsigned gy, unsigned threads, F f) {
  blockDim = {threads, 1, 1};
  for (unsigned by = 0; by < gy; ++by) for (unsigned bx = 0; bx < gx; ++bx) for (unsigned t = 0; t < threads; ++t) {
    blockIdx = {bx, by, 0}; threadIdx = {t, 0, 0}; f();
  }
}

int main() {
  const int hidden = 96, inter = 64, experts = 9, slots = 3, tokens = 5;
  int failures = 0;
  for (int nvfp4 = 1; nvfp4 >= 0; --nvfp4) {
    state = 77 + nvfp4;
    auto fill = [&](int rows, int cols, std::vector<unsigned char>& codes, std::vector<unsigned char>& scales) {
      const int cb = nvfp4 ? cols / 2 : cols, blk = nvfp4 ? 16 : 32;
      codes.resize((size_t)experts * rows * cb); scales.resize((size_t)experts * rows * cols / blk);
      for (auto& c : codes) { c = (unsigned char)(next() >> 9); if (!nvfp4) c = (c & 0x80) | ((c & 0x7F) % 0x48); }
      for (auto& s : scales) { unsigned v = next() >> 9; s = nvfp4 ? (unsigned char)(0x28 + (v & 0x1F)) : (unsigned char)(123 + (v & 3)); }
    };
    std::vector<unsigned char> gc, gs, dc, ds;
    fill(2 * inter, hidden, gc, gs); fill(hidden, inter, dc, ds);
    std::vector<float> factors(experts);
    for (int e = 0; e < experts; ++e) factors[e] = nvfp4 ? 0.05f + 0.01f * e : 1.0f;
    std::vector<float> x(tokens * hidden), logits(tokens * experts), bias(experts);
    for (auto& v : x) v = unit(); for (auto& v : logits) v = 3 * unit(); for (auto& v : bias) v = unit();
    std::vector<float> routing(tokens * 2 * slots), act(tokens * slots * inter), partial(tokens * slots * hidden), y(tokens * hidden);
    launch((tokens + 127) / 128, 1, 128, [&] { izwi_mpk_route(logits.data(), bias.data(), routing.data(), tokens, experts, slots, 8.0f); });
    const int pairs = tokens * slots;
    if (nvfp4) {
      launch(pairs, (inter + 127) / 128, 128, [&] { izwi_mpk_up_nvfp4(x.data(), routing.data(), gc.data(), gs.data(), factors.data(), act.data(), slots, hidden, inter, experts, 4.0f, 25.0f); });
      launch(pairs, (hidden + 127) / 128, 128, [&] { izwi_mpk_down_nvfp4(act.data(), routing.data(), dc.data(), ds.data(), factors.data(), partial.data(), slots, hidden, inter, experts); });
    } else {
      launch(pairs, (inter + 127) / 128, 128, [&] { izwi_mpk_up_mxfp8(x.data(), routing.data(), gc.data(), gs.data(), factors.data(), act.data(), slots, hidden, inter, experts, 4.0f, 25.0f); });
      launch(pairs, (hidden + 127) / 128, 128, [&] { izwi_mpk_down_mxfp8(act.data(), routing.data(), dc.data(), ds.data(), factors.data(), partial.data(), slots, hidden, inter, experts); });
    }
    launch(tokens, (hidden + 127) / 128, 128, [&] { izwi_mpk_combine(partial.data(), routing.data(), y.data(), slots, hidden); });

    auto w = [&](const std::vector<unsigned char>& codes, const std::vector<unsigned char>& scales, int rows, int cols, int e, int r, int c) {
      const size_t row = (size_t)e * rows + r;
      if (nvfp4) { unsigned byte = codes[row * (cols / 2) + c / 2]; unsigned code = (c % 2 == 0) ? (byte & 15) : (byte >> 4);
        return E2M1[code] * e4m3d(scales[row * (cols / 16) + c / 16]) * (double)factors[e]; }
      return e4m3d(codes[row * cols + c]) * std::pow(2.0, (int)scales[row * (cols / 32) + c / 32] - 127);
    };
    double worst = 0, scale = 1e-9;
    for (int t = 0; t < tokens; ++t) {
      // Independent routing: top `slots` by sigmoid + bias, ties to the lower id.
      std::vector<int> ids; std::vector<double> sc(experts);
      for (int e = 0; e < experts; ++e) sc[e] = 1.0 / (1.0 + std::exp(-(double)logits[t * experts + e]));
      for (int s = 0; s < slots; ++s) { int best = -1; for (int e = 0; e < experts; ++e) { bool used = false; for (int id : ids) used |= id == e;
          if (!used && (best < 0 || (float)(sc[e]) + bias[e] > (float)(sc[best]) + bias[best])) best = e; } ids.push_back(best); }
      for (size_t a = 0; a < ids.size(); ++a) for (size_t b = a + 1; b < ids.size(); ++b) if (ids[b] < ids[a]) std::swap(ids[a], ids[b]);
      double total = 0; for (int id : ids) total += sc[id];
      for (int s = 0; s < slots; ++s) {
        if ((int)routing[t * 2 * slots + s] != ids[s]) { std::printf("routing id mismatch token %d slot %d: %d vs %d\n", t, s, (int)routing[t * 2 * slots + s], ids[s]); ++failures; }
        if (std::fabs(routing[t * 2 * slots + slots + s] - sc[ids[s]] / total * 8.0) > 1e-5) { std::printf("routing weight mismatch\n"); ++failures; }
      }
      std::vector<double> want(hidden, 0.0);
      for (int s = 0; s < slots; ++s) {
        const int e = ids[s]; const double weight = sc[e] / total * 8.0;
        std::vector<double> a(inter);
        for (int i = 0; i < inter; ++i) { double g = 0, u = 0;
          for (int c = 0; c < hidden; ++c) { g += w(gc, gs, 2 * inter, hidden, e, 2 * i, c) * x[t * hidden + c]; u += w(gc, gs, 2 * inter, hidden, e, 2 * i + 1, c) * x[t * hidden + c]; }
          a[i] = std::tanh(g / 4) * 4 / (1 + std::exp(-g)) * std::tanh(u / 25) * 25; }
        for (int h = 0; h < hidden; ++h) { double p = 0; for (int c = 0; c < inter; ++c) p += w(dc, ds, hidden, inter, e, h, c) * a[c]; want[h] += weight * p; }
      }
      for (int h = 0; h < hidden; ++h) { worst = std::fmax(worst, std::fabs(want[h] - y[t * hidden + h])); scale = std::fmax(scale, std::fabs(want[h])); }
    }
    std::printf("%s: worst error %.3e at scale %.3e\n", nvfp4 ? "nvfp4" : "mxfp8", worst, scale);
    if (!(worst <= 2e-5 * scale) || scale < 1e-3) ++failures;
  }
  std::printf(failures ? "FAILED (%d)\n" : "ok\n", failures);
  return failures != 0;
}
