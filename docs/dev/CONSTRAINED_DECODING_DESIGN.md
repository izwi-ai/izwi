# Constrained decoding design (DS9.2)

Status: spike implemented for `response_format: json_object` on the shared-sampler
chat families; full JSON Schema support remains a gated follow-up. This note is the
DS9.2 research deliverable and records the decision the roadmap requires.

## Goal

`response_format` on the OpenAI-compatible chat surface must guarantee output
shape. `json_object` guarantees the completion parses as a single JSON value;
`json_schema` would additionally guarantee conformance to a caller schema. Today
the field was not parsed at all (silently ignored).

## Prior art

- **xgrammar** (used by vLLM/SGLang): compiles a grammar (EBNF/JSON Schema) into a
  compressed adaptive FSM and applies **per-step token bitmask** rejection sampling
  at the logits boundary. C++/Python; the bitmask approach is the industry default.
- **outlines** (Python): regex/CFG → FSM over the tokenizer vocabulary; same
  token-mask principle, heavier compile step.
- **llguidance**: Rust-core guidance engine; the only mainstream Rust-native
  candidate. Rejected as a spike dependency: large surface, and the mechanism we
  need to validate (per-state vocab masks at the sampler seam) is small enough to
  hand-roll for the JSON subset. Revisit if full JSON Schema support is pursued.
- Both vLLM and SGLang apply the mask **before** temperature/top-k/top-p and treat
  the masked logits as -inf, then advance the FSM with the accepted token's text.

## Architecture for this codebase

Token-level FSM masking (xgrammar-style), Rust-only, no new dependencies:

1. **Grammar machine** (`models/shared/grammar.rs`): a hand-rolled JSON value FSM
   whose states cover the JSON grammar (objects, arrays, strings with escapes,
   numbers, literals) plus a `Complete` state that only admits insignificant
   whitespace and stop tokens. `feed(text)` advances state char-by-char and
   rejects invalid continuations; a token is **allowed** in state S when
   `S.feed(token_text)` succeeds (ending in any state, including mid-string —
   incremental detokenization tolerates buffering).
2. **Per-state vocab masks**: for each FSM state, the allowed-token bitset over
   the vocabulary is built lazily from single-token decode surfaces and cached
   per machine instance (`Arc<Mutex<HashMap<u32, Arc<[bool]>>>>` survives the
   sampler's `Clone`). Building one state costs one decode per vocab entry; the
   cache makes steady-state stepping O(1) mask lookups.
3. **Sampler seam**: `ChatSampler` owns an optional machine. When active:
   - the deterministic-greedy device fast path is bypassed (it cannot mask);
   - the bounded device path receives the mask through its existing
     `allowed_mask` parameter (applied as `-inf` via `where_cond` before
     penalties/temperature, mask tensor cached per sampler);
   - the host fallback applies the mask before penalties;
   - after the token is drawn, the machine advances with the token's decoded
     surface text.
4. **Family scope**: the machine lives in `ChatSampler`, so every family that
   samples through it (qwen3, gemma3, lfm2 — including continuous-batch rows,
   which each own a sampler) supports `json_object` with no per-family code.
   Qwen3.5/Qwen3.8 use their own samplers and are **rejected at the public
   boundary** with a documented 400 rather than silently unenforced.

## Spike results

- The FSM + mask approach works over real tokenizer vocabularies (WordLevel
  fixture in tests) and over synthetic vocabs; greedy generation under the mask
  yields parseable JSON by construction (the mask only admits continuations of
  the current JSON state).
- Mask build cost is the feasibility risk: one state over a ~150k BPE vocab is
  ~150k single-token decodes (mechanically bounded, cached per state). JSON's
  state space is small (≤ ~15 live states per request), so worst-case warmup is
  ~15 × vocab decodes once per process per tokenizer. An adaptive-token-bitmask
  prefilter (xgrammar's trick: bucket tokens by first byte) would cut this
  further and is the first optimization if real-world warmup matters.
- Logprob interaction: logprob collection forces the host sampler route; the
  mask composes with it (mask → -inf before penalties, raw-logit logprobs are
  computed on the unmasked row — reported logprobs of masked-out tokens remain
  defined, matching vLLM behavior of masking only the *selection*).

## Scope decisions

- `json_object` is accepted and enforced on qwen3/gemma3/lfm2 chat models;
  rejected (400, documented) on models whose sampling path cannot honor it.
- `json_schema` is rejected with a documented 400 for now. The follow-up is
  schema → parameterized JSON FSM (structural subset: types, enums, required
  keys, bounded arrays), reusing this machine as the substrate; it is gated on
  demand rather than built speculatively.
- The worker protocol does not carry `response_format` yet; constrained
  decoding is a local-lane capability until a minor-4 field is justified.

## Kill switch

The constraint activates only when the request carries
`response_format: {"type": "json_object"}`; absent field means byte-identical
legacy behavior.
