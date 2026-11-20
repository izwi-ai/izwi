# Ming 3.0 TTS

Developer notes for the `MingTts` family
(`crates/izwi-core/src/models/architectures/ming_tts`), which runs
[`inclusionAI/Ming-3.0-tts-0.5B`](https://huggingface.co/inclusionAI/Ming-3.0-tts-0.5B)
at revision `ffd36ab5764c39c7e5565dfed66a3aeeced58ab9`.

The Hugging Face repository ships no code. The reference implementation is
<https://github.com/inclusionAI/Ming-3.0-tts> at
`c88c023ea6ccc08fd1e7a1d289cb6246c21deff1`; file names below refer to it.

## The model

It is not a codec-token model. A language model reads the prompt and then,
once per 80 ms of speech, its last hidden state conditions a flow-matching
head that samples one 64-wide audio latent. The latent's embedding is the
language model's next input. A transformer VAE turns the latents into
44.1 kHz audio through an inverse STFT. Nothing is sampled from logits; the
only randomness is each latent's starting noise.

| Part | Checkpoint prefix | Shape | Parameters |
|------|-------------------|-------|-----------:|
| Language model | `model.model.*` | Qwen2.5-0.5B: 24 layers, width 896, 14 heads over 2 KV heads of 64, MLP 4864, RoPE theta 1e6, vocabulary 151,936; biased q/k/v; no stored output head | 494.0M |
| Aggregator | `linear_proj_audio.*` | latent 64 to 1024, a learned first token, 8 bidirectional blocks (16 heads, MLP 4096), RMSNorm, linear to 896; two positions, the first is read | 101.7M |
| Flow head | `flowloss.cfm.model.*` | 8 bidirectional blocks at width 1024; three positions: time + condition, previous latent, noisy latent | 103.1M |
| Stop head | `stop_head.*` | linear 896 to 2 | 1.8K |
| Speaker projection | `spk_head.*` | linear 192 to 896 | 173K |
| VAE decoder | `audio.decoder.*` | linear 64 to 896, x4 linear upsample in time, a 24-layer Qwen2 stack with a 64-frame attention window, linear to 3,530, inverse STFT | 361.1M |
| VAE encoder | `audio.encoder.*` | 882-sample frames to 896, the same 24-layer stack, a 4-layer stack over groups of four frames, linear to mean and scale | 419.3M |

1,180 tensors, 1.48B parameters, all BF16 but two rotary buffers. "0.5B" is
the language model alone.

Audio: 44.1 kHz; 882 samples (20 ms) a VAE frame; four frames a latent, so
12.5 latents a second. The inverse STFT uses a 3,528-sample periodic Hann
window with hop 882 and trims 1,323 samples from each end.

Two rotary conventions live in one checkpoint: the Qwen2 stacks use the
half-split form at theta 1e6, the aggregator and flow head use
x_transformers' interleaved pairs at base 10,000.

## A family of its own

The module imports no other architecture; `isolation_tests` in `mod.rs`
scans its sources for it. It uses the shared paged attention, RoPE tables,
seeded noise, the staged safetensors loader and the torch-archive reader.

| File | Contents |
|------|----------|
| `config.rs` | `config.json`, checked field by field; the pipeline's constants |
| `weights.rs` | The checkpoint mapping and the published tensor list (pinned by a census fixture) |
| `stack.rs` | The Qwen2 layer two ways: the language model on paged KV, and the VAE's windowed stacks with per-layer kept keys |
| `flow.rs` | The aggregator, the flow head and its ten-step sampler |
| `vae.rs` | Incremental decode (upsample, stack, inverse STFT with overlap-add) and encode |
| `speaker.rs` | CAM++, Kaldi fbank and the torchaudio resampler |
| `prompt.rs` | The tokenizer, the prompt pieces, the control captions and text clean-up |
| `stream.rs` | Audio windows and the trailing-silence trim |
| `retained.rs` | The retained row: resumable prefill, batched decode steps, rollback |
| `physical.rs` | The paged-KV state contract |
| `model.rs` | Load, precision, the engine-facing steps, the reference-voice cache |
| `synthetic.rs` | A width-32 model with random weights for tests |
| `reference_tests.rs` | Ignored tests on the published checkpoint |

## The speaker encoder

Upstream ships CAM++ as `campplus.onnx` and runs it with onnxruntime. Izwi
reads no ONNX. It loads the same network from its PyTorch checkpoint,
`funasr/campplus` at `e4b6ede7ce16997aff4ae69fbca1f0175e2afede`
(`campplus_cn_common.bin`, Apache-2.0), with the torch-archive reader.

The two files hold the same weights: 485 of the checkpoint's tensors appear
byte for byte in the ONNX file, 122 are batch-norm step counters an export
drops, and the rest are convolutions the export folded with their batch
norm. Folding them the same way reproduces the ONNX values to within one
float step on the layers sampled (checked 2026-11-19).

The model runs without the speaker encoder. Built-in voices, voice design
and their controls use a zero speaker row. Only a cloned voice needs the
embedding; a model folder without `campplus_cn_common.bin` loads and refuses
cloning with a message naming the file.

## Prompt

Each piece is tokenized on its own and the ids are joined
(`prompt_contract.py`):

```
<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n
<|im_start|>user\n
Please generate speech based on the following description.\n[Target language: German.\n]
[  speaker_1:][<|vision_start|>][<|vision_pad|>][<|vision_end|>\n]
 Text input:\n
[reference transcript]text
<|im_end|>\n
<|im_start|>assistant\n
[caption][<|im_end|>]
<audio>[<audioPatch> x reference latents]
```

The `<|vision_pad|>` row is replaced by `spk_head(embedding)`, or by zeros
for a built-in or described voice. Each `<audioPatch>` row is replaced by the
aggregator's output for one reference latent. The caption is a JSON object in
Python's `json.dumps` form (Chinese keys; English keys for the dialect task)
or a plain sentence.

A tokenizer whose pinned ids differ is refused at load.

## One control per request

Upstream trains each control as its own task and rejects mixes; izwi keeps
that:

| Request | Speaker row | Caption |
|---------|-------------|---------|
| Built-in voice (`voice`) | zeros | the voice's name; an `emotion` picks one of the voice's recorded variants |
| Reference clip, with or without transcript | CAM++ embedding | none, or one of `emotion`, `dialect`, `speaking_rate` / `pitch` / `volume` |
| Voice description | zeros | the description (the voice-design task) |
| No voice, one control | zeros | the control; the model picks the voice |
| No voice, no control | zeros | the default built-in voice of the text's language |

A reference's latents go into the prompt only with its transcript; without
one the embedding alone carries the voice.

## Generation

One latent per decode step:

1. Draw 64 normals from the request's seeded stream.
2. Ten Euler steps of the flow head from that noise, conditioned on the
   pending hidden state and the previous latent, guidance 2.0 (both branches
   in one batch of two), on the reference's time grid with sway.
3. Stop when the stop head's probability passes 0.5 after at least five
   latents, or when the budget ends. The last latent is kept.
4. Otherwise run the latent's aggregator embedding through the language
   model for the next hidden state.

The context is 512 positions as in the reference, so a request generates at
most `512 - prompt positions` latents. A request whose prompt leaves no room
is refused with a message; a budget that ends before the stop head fires is
reported as `LatentBudget`.

Rows from several requests share every call of a decode step, and a
request still gets the bits it gets alone:

| | Language model | Flow head, aggregator |
|---|---|---|
| CPU | projections row by row | one request a call |
| Metal | projections row by row | the step's requests in one call (up to 32) |
| CUDA | one 32-row shape, padded | one 32-request shape, padded |

Each request draws its noise from its own seeded stream, and every op on
the flow path keeps rows apart; only a matmul's kernel depends on the call's
shape. Metal's matmul rounds a row the same for every row count from two up,
but runs a single row as a GEMV that rounds some values differently. Every
flow-head matmul has at least two rows; an aggregator call, whose first
matmul has one row a request, is padded to two. So Metal shares calls
without padding them to a group (padding a lone request to eight would cost
it 60%). The CPU matmul's rounding changes with the row count, so the CPU
runs requests apart. `requests_sampled_together_match_each_alone_*` (every
count from two to eleven) and `rows_decoded_together_match_each_alone`
check it bit for bit, `requests_together_match_each_alone_on_the_checkpoint`
on the published weights (two to eight), and `run-tts-concurrency.py report
--require-parity` through the server.

## Audio

The VAE decodes incrementally. The upsampler waits one latent (the last two
frames of a latent interpolate towards the next one), each stack layer keeps
the keys a later frame can still see, and the overlap-add holds the samples
later frames add to. Latents fed in pieces decode to the samples one pass
gives; `vae.rs` and `retained.rs` test that.

Windows: the first after 4 latents, then every 8, then whatever is left at
the stop. Streamed and final-only requests decode the same windows.

The reference trims trailing silence after decoding everything. The stream
reaches the same cut going forwards: audio is released only up to 0.3 s past
the last 0.1 s frame louder than 1e-3.

### The VAE's attention window

The published config pins the VAE stacks to flash-attn with
`sliding_window: 64`. In transformers 4.46.3 the flash path lets a frame see
itself and the 64 before it; the SDPA path, which upstream's own
requirements and Docker image run, masks keys `j <= i - 64`, so a frame sees
itself and the 63 before it. Izwi follows the SDPA path.
`IZWI_MING_TTS_VAE_WINDOW_KEYS` sets the number of keys (64 by default, 65
for the flash behaviour). The fixture generator records both so the choice
can be checked against the reference.

## Precision

| Backend | Language model | Aggregator, flow head | VAE | Inverse STFT, noise, RoPE tables |
|---------|----------------|-----------------------|-----|----------------------------------|
| CPU | F32 | F32 | F32 | host |
| Metal | F16; residual stream, gated product and down projections in F32 | F16 | F32 | host |
| CUDA | BF16 | BF16 | F32 | host |

On the published weights layer 3's down projection writes values past F16's
65,504 (Qwen2.5's large-activation channel; its gated product already reaches
21,056 on a short prompt). An F16 down projection turns them into infinities
that reach every later layer and the flow head. Under F16 weights the
residual stream, the gated product and the down projections therefore stay in
F32. BF16 has F32's range and needs none of this.

Each load runs a range probe (a built-in voice's prompt and three latents)
and walks a ladder on failure: the device dtype for the language model and
flow path, then the flow path in F32, then everything in F32. The model
diagnostics report the result (`dtype`, `flow_dtype`, `vae_dtype`).
`trace_language_model_magnitudes` (an ignored test) prints each layer's
largest values on the test device. RMSNorm statistics are F32 everywhere:
`eps` 1e-6 is below F16's smallest normal number.

## Long text

The server cuts text longer than one request into segments of whole
sentences of at most 160 characters, a CJK character counting three (about
12 s of speech: `api/tts_policy.rs`), on every speech route, the durable
one included. Each segment is its own request.

A built-in voice or a description is a caption: each request samples a
voice that fits it, so a long text changed speaker at every segment (on
灵语汐 the second segment of a three-segment text measured 0.45-0.62 speaker
similarity to the first, about what two different voices give). When the
request brings no reference, the first segment's audio and text become the
reference of the rest (`SegmentVoiceAnchor`, `FirstSegmentVoice` in
`api/tts_long_form.rs`): later segments clone it with its latents in the
prompt, keep any single control (an emotion, say) and drop a description.
Segments then measure 0.85-0.92 to the first. The speaker embedding alone,
without the transcript, held the voice no better than before. A first
segment under 1 s or over 20 s is not used, and the next one is. Later
segments wait for the first and still run side by side with each other.
`voice_identity_across_stretches` (an ignored reference test) prints CAM++
similarities between stretches of WAV files.

## Kernels and CUDA graphs

- Norms are `kernels::residual_norm` (one launch, F32 statistics, an F32
  gain), the post-attention norm fused with the residual add. In the
  language model they normalize at F32 and cast after.
- Rotations are Candle's fused `rope_i` (the flow transformers'
  interleaved pairs) and `rope_thd` (the Qwen2 stacks' half split).
- The flow transformers' attention over two or three positions is the
  fused kernel where the backend has one (Metal SDPA, CUDA flash
  attention), in the input's dtype. The gated product of the Qwen2 stacks
  is the fused SiLU-multiply.
- The flow head's time MLP runs once at load, for the ten fixed steps; the
  condition projection and the previous latent's embedding once a latent.

On CUDA two sets of graphs cut launch overhead (`graphs.rs`):

| Graphs | What | Switch | Diagnostics |
|---|---|---|---|
| Flow path (`ShapeGraphs`) | the whole ten-step sampler, and the aggregator, at 32 padded requests | `IZWI_MING_TTS_FLOW_GRAPHS` | `flow_graphs` |
| Decode step (`StretchGraphs`) | the work between two layers' paged attention, at 1 to 4 or 8 decode rows | `IZWI_MING_TTS_DECODE_GRAPHS`, `cuda.decode_graphs` | `decode_graphs` |

Both are on by default. A graph is captured, checked against the eager run
bit for bit, verified once more on new inputs, then replayed, and any
failure turns its set off with the reason in the diagnostics. The flow-path
graphs are self-checked at load (after the range probe) and priced at
200 MiB. Not yet run on a GPU.

## Performance

`ming_tts::reference_tests::bench_components` (ignored, the published
checkpoint; `IZWI_MING_TTS_TEST_DEVICE`, `IZWI_MING_TTS_BENCH_CALLS`) times
each stage of a latent with the device drained around every call. M1 Pro,
2026-11-20, per latent (80 ms of audio):

| Stage | Metal F16 before | Metal F16 now | CPU F32 before | CPU F32 now |
|---|---:|---:|---:|---:|
| Flow head, one request | 96.6 ms | 26.5 ms | 254 ms | 234 ms |
| Flow head, 8 requests together | 8 x 96.6 | 93.6 ms (11.7 a request) | n/a | n/a |
| Language-model step, 1 / 4 rows | 12.7 / 35.0 ms | 9.9 / 21.8 ms | 42 / 163 ms | 41 / 160 ms |
| Aggregator | 7.7 ms | 3.0 ms | 19 ms | 19 ms |
| Prefill, 160 positions | 84 ms | 55 ms | 447 ms | 342 ms |
| A sentence end to end, RTF | 1.57 | 0.54 | 4.31 | 4.10 |

Through the server on Metal (`run-tts-concurrency.py`, buffered requests of
about 8 s of audio, byte parity with one stream at every level):

| Streams | 1 | 2 | 4 | 8 |
|---|---:|---:|---:|---:|
| Before, audio-s/s | 0.65 | 0.66 | 0.67 | 0.66 |
| Now, audio-s/s | 1.90 | 2.44 | 3.05 | 3.18 |

Ming decodes eight rows on Metal (`catalog::tts_decode_width`), two on the
CPU and 32 on CUDA.

The flow head is most of a latent everywhere. On Metal its linear layers run
on Izwi's multi-row mat-vec (`kernels::metal_dense_mv`, switch
`IZWI_MING_TTS_METAL_MATVEC`): Candle's GEMM read a few rows' weights well
below the memory bandwidth (about 40 of the flow head's 59 ms). The kernel
sums a row the same at any row count, so sharing stays bit-exact. From about
eight requests (48 rows) Candle's tiled GEMM would be as fast. On the CPU the
flow head is bound by its matmuls, and requests do not share calls (see
Generation).

## Checks

- `ming_tts::` unit tests run the whole pipeline on a width-32 model:
  windowed stack against a dense masked reference, chunked against one-pass
  decode, retained steps against the one-shot loop, rollback, rows together
  against rows alone, the trim against a transcription of the reference's.
- `weights::tests` pins the 1,180 published tensors.
- `reference_tests` (ignored) run the published checkpoint:
  `IZWI_MING_TTS_MODEL_DIR=<folder> cargo test -p izwi-core --lib --release
  ming_tts::reference_tests -- --ignored --nocapture`, with
  `IZWI_MING_TTS_TEST_DEVICE=metal` for Metal and `IZWI_MING_TTS_OUT_DIR`
  to keep the audio.
- `scripts/fixtures/generate_ming_tts_reference.py` writes the PyTorch
  reference fixture (prompt ids, hidden states, a flow sample with supplied
  noise, VAE decode under both attention implementations, the posterior and
  the speaker embedding).
