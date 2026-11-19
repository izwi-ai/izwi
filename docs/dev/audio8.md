# Audio8 TTS Realistic Air

Developer notes for the `Audio8Tts` family
(`crates/izwi-core/src/models/architectures/audio8`).

- Model: [`Edge0/Audio8-TTS-Realistic-Air`](https://huggingface.co/Edge0/Audio8-TTS-Realistic-Air),
  Apache-2.0, pinned at `6bf7a980759d8022a66922ee975dab99229a2dd9`.
- Catalog id `Audio8-TTS-Realistic-Air`, variant `Audio8TtsRealisticAir`.
- Reference implementation: the four Python files at the root of the model
  repository (`configuration_arktts.py`, `modeling_arktts.py`,
  `modeling_arktts_codec.py`, `processing_arktts.py`).

## What the model is

A 601M-parameter DualAR model with a 44.1 kHz codec.

| Part | Shape |
|---|---|
| Slow transformer | 24 layers, width 896, 14 heads, 2 KV heads, head dim 64, SwiGLU 4864, fused QKV with bias, RMSNorm eps 1e-6, RoPE base 1e6, 2,048 positions, 155,776 tokens, output head tied to the token embedding |
| Fast transformer | 4 layers, same width, fused QKV without bias, 10 positions, 4,096 logits |
| Frame | one semantic token (ids 151,678 to 155,773) and 10 codes; 2,048 samples; 21.53 frames a second |
| Codec | causal conv encoder (strides 2/4/8/8, a 4-layer window transformer), 2x down, pre and post window transformers, 1 semantic book of 4,096 and 9 residual books of 1,024 (dim 8, cosine lookup), 2x up, conv decoder (1536 channels, strides 8/8/4/2) |

Each frame, the slow transformer reads the previous frame (its token
embedding plus the sum of its ten codebook embeddings) and emits one
semantic token; the fast transformer reads the slow hidden state after the
final norm, then the semantic code, and emits the nine residual codes. The
codec decodes frames to audio and encodes a reference clip to frames.

The prompt, the reference frames and the generated frames share the slow
transformer's 2,048 positions: about 95 s of audio in all.

## Family boundaries

The family is its own module and imports no other family. It started as a
copy of the Fish S2 family, whose DualAR design the model follows, and
differs from it wherever the checkpoint does:

| Area | Audio8 |
|---|---|
| Config | flat `arktts` `config.json`, parsed into the slow and fast halves; anything the family does not implement (Q/K norm, scaled codebook embeddings, a fast transformer of another width, another codec post module) is refused at load |
| Weights | one `model.safetensors`, flat tensor names (`layers.N.attention.wqkv`, `fast_layers.N.*`), all BF16; `codec.pth` beside it |
| Slow attention | biased fused QKV projection, no Q/K normalization |
| Codebook embeddings | summed unscaled |
| Codec | post module has 16 heads over 8 KV heads and a 1216 MLP; the pre module keeps 16 KV heads and 3072 |
| Prompt | `ArkttsProcessor`'s parts, each tokenized on its own, caller text with its whitespace collapsed; a prompt for speech with no reference |
| Sampling | the checkpoint's defaults (temperature 0.7, top-p 0.9, top-k 50) |
| Budget | at most 1,024 frames a request; a request that names none gets its text's speaking estimate |

The engine side is the route shape the DualAR families share: a preparation
stage (tokenize, encode the reference through the content-addressed
reference cache), prefill, decode on retained slow state with per-request
fast state, codec frames decoded in batched quanta, and a finalize stage.
The handlers are `engine/executor/handler_audio8.rs` and
`audio8_pending.rs`; the load-time capacity is `runtime/audio8_capacity.rs`.

## Deliberate differences from the reference

- **End of speech on the first frame.** The reference may sample it and
  return no audio. Izwi never allows it on frame 0, so a request always
  produces audio or fails.
- **Budget exhaustion.** A request that reaches its frame budget without an
  end-of-speech token fails as incomplete speech; the reference returns the
  truncated audio.
- **Random numbers.** Sampling is seeded and repeatable on one backend, but
  the draws are not PyTorch's: equal seeds do not give equal audio across
  frameworks, and a seed's audio differs between CPU (F32) and Metal (F16).

The repetition-aware retry matches the reference, including that the
utterance's first frame is never in its ten-frame window.

## Precision

| Backend | Transformers | Codec |
|---|---|---|
| CPU | F32 | F32 |
| Metal | F16 | F32 |
| CUDA | BF16 on compute capability 8.0 or newer, else F32 | F32 |

`IZWI_AUDIO8_DTYPE` overrides the transformers' dtype. RMSNorm runs in F32
and the RoPE tables are the reference's BF16-rounded values used in F32. A
load-time self-check runs every slow layer, the semantic head and a fast
frame, and falls back to F32 when half precision overflows.

## Voices

- **No reference.** The model speaks in a voice of its own. The request's
  seed makes it repeatable for one text; another text may get another voice.
- **Reference.** A clip (at most 30 s) and its transcript. The clip is
  encoded once per distinct audio (reference cache, 4 MiB of codes).
- **Long text with no reference.** The server speaks long text as a segment
  plan (about 300 characters a segment; a CJK character counts as four).
  The first segment's audio and text become the reference of the segments
  after it, so one voice runs through the text
  (`SegmentVoiceAnchor` in `izwi-server/src/api/tts_long_form.rs`). The
  anchor is not journaled: a plan recovered after a restart anchors on the
  first segment it speaks then.

## Environment

| Variable | Effect |
|---|---|
| `IZWI_AUDIO8_DTYPE` | transformer dtype: `f32`, `f16` or `bf16` |
| `IZWI_AUDIO8_NATIVE_BATCHING` | decode concurrent requests as rows of one batch (default: on for CUDA) |
| `IZWI_AUDIO8_BATCHED_SAMPLING` | draw a step's tokens for every row in one device pass (default on) |
| `IZWI_AUDIO8_FAST_GRAPHS` | CUDA graphs for the fast transformer (default on) |

The `IZWI_AUDIO8_MODEL_DIR`, `_BACKEND`, `_REFERENCE_WAV`, `_REFERENCE_TEXT`,
`_TARGET_TEXT` and `_SMOKE_*` variables drive the ignored real-checkpoint
tests; `docs/dev/env/core.md` lists them all.

## Tests

Unit tests run on synthetic weights:

```bash
cargo test -p izwi-core --lib audio8
cargo test -p izwi-server --lib audio8
```

Real checkpoint (ignored; needs the pinned bundle in a directory named
`Audio8-TTS-Realistic-Air`):

```bash
export IZWI_AUDIO8_MODEL_DIR=/path/to/models/Audio8-TTS-Realistic-Air
# headers, tokenizer and memory plan; then a native load
cargo test -p izwi-core --lib -- --ignored audio8_real_artifact
# speech with no reference, written to a WAV
IZWI_AUDIO8_BACKEND=cpu IZWI_AUDIO8_REFERENCE_WAV=none \
IZWI_AUDIO8_TARGET_TEXT="Hello, this is a test." \
IZWI_AUDIO8_SMOKE_OUTPUT_WAV=/tmp/audio8.wav \
cargo test -p izwi-core --lib -- --ignored --nocapture audio8_real_model_smoke
```

Parity with the PyTorch reference:

```bash
python3 -I scripts/fixtures/generate_audio8_reference.py \
    "$IZWI_AUDIO8_MODEL_DIR" /path/to/the-four-python-files \
    crates/izwi-core/tests/fixtures/audio8
cargo test -p izwi-core --lib audio8::reference_tests -- --ignored
```

The generator runs the checkpoint's own code and writes prompt ids, codec
codes and waveform, and a greedy generation; the tests compare izwi's
prompt, codec and greedy output with them.

## Verification status

Checked on a 16 GB Apple M1 Pro with the pinned checkpoint and release
builds of the server:

| Check | CPU (F32) | Metal (F16) |
|---|---|---|
| Native load, every tensor mapped, self-check | yes | yes |
| Speech with no reference, transcribed back by Parakeet | exact | exact |
| Speech in a reference voice, transcribed back | exact | exact |
| German, French (Parakeet) and Chinese (Whisper) | exact | not run |
| Same seed twice gives the same bytes | yes | yes |
| Long text (577 characters, several segments), with and without a reference | not run | 0 word errors in 108 |
| 4 and 8 concurrent requests equal their solo output byte for byte | not run | yes |
| Streamed speech (`stream: true`), time to first audio | 0.9 s | 0.2 to 0.3 s |
| CLI (`izwi list`, `izwi tts` seeded and cloned, missing-transcript error) | not run | yes |
| Real-time factor, one request | about 4 | about 1.05 (1.33 in F32) |

A long text with no reference keeps its voice: across the anchored take the
median pitch of its quarters stayed within 15 Hz, where six separate
requests with no reference ranged over 51 Hz. That is a pitch measurement,
not a listening test.

Sampled speech sometimes ends with a stray extra word after the sentence.
Over 32 sentences on Metal it happened 5 times in F16 and 3 times in F32,
and once in 8 on the CPU in greedy mode, so it is not a half-precision
effect; whether the PyTorch reference does the same is not known until the
greedy fixture has been compared.

CUDA: `cargo clippy -p izwi-core --features cuda -- -D warnings` is clean
with a stand-in `nvcc`.

Not done:

- The parity fixtures have not been generated: the generator script has
  not been run, and `audio8::reference_tests` has only been compiled.
- Voice similarity of a clone has not been measured or listened to.
- CUDA is compile-checked only. Nothing has run on an NVIDIA GPU: the BF16
  path, the fast transformer's CUDA graphs and batched decode are
  unverified there.
- The web UI was checked as far as the model picker (the model is listed
  under Edge0 in the text-to-speech dialog); a take was not created from
  the dialog against a live server.
- `Audio8-TTS-Preview-0.6b`, the earlier checkpoint with the same shapes
  and codec, is not in the catalog.
