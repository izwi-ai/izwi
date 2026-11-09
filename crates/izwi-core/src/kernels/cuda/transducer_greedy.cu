#include <math.h>

// One greedy transducer joint step for every row of a decode batch, on the
// device (`kernels::transducer_greedy::select_step`): the picks of
// `kernels::greedy` (the largest logit, ties to the lowest id, NaN never
// picked) and the row's transition (`tdt_step`, Parakeet; `rnnt_step`,
// Nemotron ASR), so a decode can run many joint steps between readbacks.
//
// One warp per row. Lanes scan the label logits (and the TDT duration
// logits) with a stride of 32, then an xor butterfly merges the lanes'
// (best, id) pairs; lane 0 applies the transition to the row's state:
//
//   state  [B, 12] u32: t, last emission's frame, emissions at it, joint
//                       steps, tokens, live, emitted this dispatch, joint
//                       steps this dispatch, blanks this dispatch, full
//                       frames this dispatch, RNNT symbols on t, error
//   limits [B, 6]  u32: encoded frames, joint-step limit, token budget,
//                       quota this dispatch, window's first frame, window
//                       frames
//   emits  [B, S, 3] u32: label, frame, duration of each emission
//   labels [B] u32, mask [B] u32: the predictor's input label (0 when the
//                       row emitted nothing) and whether it emitted
//   gather [B] u32: the row's next encoder frame in the stacked windows,
//                       row * window + t - first
//
// A row stops (live = 0) once it is finished (cursor past its frames,
// E * max_symbols tokens, its token budget or its joint-step limit), has
// emitted its quota, or its cursor left its window; a stopped row keeps its
// state and gather index and emits nothing. Grid (B), 32 threads.

#define TG_T 0
#define TG_LAST 1
#define TG_COUNT 2
#define TG_GUARD 3
#define TG_TOKENS 4
#define TG_LIVE 5
#define TG_EMITTED 6
#define TG_JOINT 7
#define TG_BLANKS 8
#define TG_FULL 9
#define TG_SYMBOLS 10
#define TG_ERROR 11
#define TG_FIELDS 12

#define TL_LEN 0
#define TL_GUARD 1
#define TL_BUDGET 2
#define TL_QUOTA 3
#define TL_FIRST 4
#define TL_WINDOW 5
#define TL_FIELDS 6

#define TG_NONE 0xffffffffu
#define TG_FULL_MASK 0xffffffffu

// The lanes' (best, id) over [begin, end), merged: the largest value, ties to
// the lowest id, NaN never picked, id 0 when nothing is.
__device__ __forceinline__ unsigned tg_pick(const float* row, int begin, int end, int lane) {
  float best = -INFINITY;
  unsigned id = TG_NONE;
  for (int i = begin + lane; i < end; i += 32) {
    const float v = row[i];
    if (id == TG_NONE ? v > -INFINITY : v > best) {
      best = v;
      id = (unsigned)(i - begin);
    }
  }
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    const float other_best = __shfl_xor_sync(TG_FULL_MASK, best, offset);
    const unsigned other_id = __shfl_xor_sync(TG_FULL_MASK, id, offset);
    const bool other_wins =
        other_id != TG_NONE &&
        (id == TG_NONE || other_best > best || (other_best == best && other_id < id));
    if (other_wins) {
      best = other_best;
      id = other_id;
    }
  }
  return id == TG_NONE ? 0u : id;
}

extern "C" __global__ void __launch_bounds__(32) izwi_transducer_select_f32(
    const float* __restrict__ logits, int width, unsigned* __restrict__ state,
    const unsigned* __restrict__ limits, unsigned* __restrict__ emits, int max_emits,
    unsigned* __restrict__ labels, unsigned* __restrict__ mask, unsigned* __restrict__ gather,
    int rnnt, int blank, int token_end, int num_durations, int max_symbols) {
  const int b = blockIdx.x;
  const int lane = threadIdx.x;
  unsigned* s = state + (long long)b * TG_FIELDS;
  const unsigned* l = limits + (long long)b * TL_FIELDS;
  if (s[TG_LIVE] == 0u) {  // the whole warp
    if (lane == 0) {
      labels[b] = 0u;
      mask[b] = 0u;
    }
    return;
  }
  const float* row = logits + (long long)b * width;
  const unsigned label = tg_pick(row, 0, rnnt ? width : token_end, lane);
  const unsigned duration = rnnt ? 0u : tg_pick(row, token_end, token_end + num_durations, lane);
  if (lane != 0) return;

  const unsigned frame = s[TG_T];
  bool emitted = false;
  unsigned emitted_duration = 0u;
  s[TG_JOINT] += 1u;
  if (rnnt) {
    if (label > (unsigned)blank) {
      s[TG_ERROR] = 1u;
      s[TG_LIVE] = 0u;
      labels[b] = 0u;
      mask[b] = 0u;
      return;
    }
    if (label == (unsigned)blank) {
      s[TG_T] = frame + 1u;
      s[TG_SYMBOLS] = 0u;
      s[TG_BLANKS] += 1u;
    } else {
      emitted = true;
      s[TG_SYMBOLS] += 1u;
      if (s[TG_SYMBOLS] >= (unsigned)max_symbols) {
        s[TG_T] = frame + 1u;
        s[TG_SYMBOLS] = 0u;
        s[TG_FULL] += 1u;
      }
    }
  } else {
    const unsigned last = (unsigned)(num_durations > 0 ? num_durations - 1 : 0);
    const unsigned clamped = duration < last ? duration : last;
    unsigned jump = clamped;
    if (label == (unsigned)blank && jump == 0u) jump = 1u;
    s[TG_T] = frame + jump;
    s[TG_GUARD] += 1u;
    if (label == (unsigned)blank) {
      s[TG_BLANKS] += 1u;
    } else {
      if (frame == s[TG_LAST]) {
        s[TG_COUNT] += 1u;
      } else {
        s[TG_LAST] = frame;
        s[TG_COUNT] = 1u;
      }
      if (s[TG_COUNT] > (unsigned)max_symbols) {
        s[TG_T] = frame + 1u;
      } else {
        emitted = true;
        emitted_duration = clamped;
      }
    }
  }
  if (emitted) {
    const unsigned slot = s[TG_EMITTED];
    if (slot < (unsigned)max_emits) {
      unsigned* e = emits + ((long long)b * max_emits + slot) * 3;
      e[0] = label;
      e[1] = frame;
      e[2] = emitted_duration;
    }
    s[TG_EMITTED] = slot + 1u;
    s[TG_TOKENS] += 1u;
  }
  labels[b] = emitted ? label : 0u;
  mask[b] = emitted ? 1u : 0u;

  const unsigned t = s[TG_T];
  const unsigned long long tokens = s[TG_TOKENS];
  const unsigned long long cap = (unsigned long long)l[TL_LEN] * (unsigned long long)max_symbols;
  const bool exhausted = t >= l[TL_LEN] || tokens >= cap;
  const bool budget = tokens >= l[TL_BUDGET];
  const bool guard = s[TG_GUARD] >= l[TL_GUARD];
  const bool quota = s[TG_EMITTED] >= l[TL_QUOTA];
  const bool outside = t < l[TL_FIRST] || t - l[TL_FIRST] >= l[TL_WINDOW];
  if (exhausted || budget || guard || quota || outside) {
    s[TG_LIVE] = 0u;
  } else {
    gather[b] = (unsigned)b * l[TL_WINDOW] + (t - l[TL_FIRST]);
  }
}
