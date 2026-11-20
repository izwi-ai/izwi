# Rumble 2 Flash

Developer notes for the `RumbleAsr` family
(`crates/izwi-core/src/models/architectures/rumble`), which runs
[`Reson8/Rumble-2-Flash`](https://huggingface.co/Reson8/Rumble-2-Flash) at
revision `92ec9141fb1178c5e77fd0020fe86a33c355060b`.

## The model

| Part | Shape |
|------|-------|
| Checkpoint | One `model.safetensors`: 457 tensors, all BF16, 508,631,552 parameters |
| Audio tower | 3 stride-2 conv2d (480 channels), `conv_out` 7680 to 896, 18 layers (d 896, 14 heads, FFN 3584), `ln_post`; 128 mel bins in, 13 tokens per second out |
| Projector | `linear_1` (896 to 896), exact GELU, `linear_2` (896 to 1024) |
| Decoder | Qwen3 text trunk: 14 layers, hidden 1024, 16 heads, 8 KV heads of 128, MLP 3072, RoPE theta 1e6 |
| Vocabulary | 99,630 tokens (a pruned, re-indexed Qwen vocabulary); the output head is the token table |
| Context | 65,536 positions; izwi leases 1,536 per invocation |

The checkpoint declares `model_type: qwen3_asr` and is published in the
Transformers-native layout: `model.audio_tower.*`,
`model.multi_modal_projector.*`, `model.language_model.*`, with a flat
`config.json` whose RoPE settings sit under `rope_parameters`.

## A family of its own

Rumble is not a variant of the Qwen3-ASR family. Its module owns the
config, the checkpoint layout and census, the front end, the audio tower and
projector, the tokenizer, the prompt and the transcript handling. Neither
family's module names the other; `family_separation` in `rumble/mod.rs`
scans both for it.

The decoder is the shared Qwen3 text trunk in `qwen3::core`
(`Qwen3Model::load_with_layout` with a `language_model` prefix and no stored
head), which chat, TTS and VibeVoice also load. The paged KV contract and the
Metal and CUDA attention paths come with it.

| File | Contents |
|------|----------|
| `config.rs` | `config.json` and `preprocessor_config.json`; the trunk config |
| `checkpoint.rs` | The tensor plan and the header check that runs before any weight is mapped |
| `frontend.rs` | Resampling, the 0.5 s minimum, log-mel features with the reference's padding |
| `audio_tower.rs` | Conv stem, windowed encoder layers, projector |
| `tokenizer.rs` | `tokenizer.json`, special ids by name, checked against the config |
| `prompt.rs` | The chat template's prompt, the style tokens, the language tables |
| `transcript.rs` | Splitting `language {Name}<asr_text>{text}` on token ids |
| `precision.rs` | Per-device dtypes and their overrides |
| `model.rs` | Load, one-invocation transcription, the published state |
| `retained.rs` | The retained row: prompt artifact, resumable prefill, single and batched decode steps |
| `synthetic.rs`, `tests.rs`, `retained_tests.rs` | A tiny checkpoint in the published layout and tests on it |
| `reference_tests.rs` | Ignored tests on the published checkpoint |

## Loading

1. `ensure_rumble_artifacts` requires `config.json`,
   `preprocessor_config.json`, `tokenizer.json` and `model.safetensors`.
2. `check_rumble_census` compares the file's header with the tensors the
   config implies. A missing or re-shaped tensor fails by name, and so does a
   tensor the loader would not read (a re-laid-out release must not load
   with a silently skipped projector).
3. The tokenizer resolves every special id by name and checks the audio ids
   and stop ids against `config.json`.
4. The audio tower and the decoder load through
   `staged::metal_synced_safetensors`, each at its own dtype, behind
   `load_with_precision_fallback`.

| Device | Decoder | Audio tower and projector |
|--------|---------|---------------------------|
| CPU | F32 | F32 |
| Metal | F32 (F16 text drifts on long audio on this trunk) | F32 |
| CUDA | The device's half precision (BF16 where supported), behind a load-time probe | F32 (batched BF16 audio encoders were batch-variant on the H100 for other families) |

`IZWI_RUMBLE_TEXT_DTYPE` and `IZWI_RUMBLE_AUDIO_DTYPE` override them.

## Front end

The published feature extractor pads every clip with silence to a 30 s
window before the STFT and marks `ceil(samples / hop)` frames valid. Two
consequences that izwi reproduces:

- a clip's last frames see silence after it, not the reflected signal;
- the tower splits features into 100-frame chunks, and the last chunk is
  filled with the mel of that silence, not with zeros.

`RumbleFrontend::prepare` pads the waveform to the next whole chunk (plus one
FFT window, never past 30 s) and returns the padded frames beside the valid
count; the tower reads its last chunk's padding from the frames it is given.
Clips under 0.5 s are zero-padded to 8,000 samples and the padding counts as
audio, as upstream.

One invocation takes at most 30 s. The generic chunk planner cuts longer
audio at 95% of that, so a 29.7 s clip runs as two chunks.

## Prompt and output

```text
<|im_start|>system\n{style}{context}<|im_end|>\n
<|im_start|>user\n<|audio_start|><|audio_pad|>*N<|audio_end|><|im_end|>\n
<|im_start|>assistant\n[language {Name}<asr_text>]
```

`{style}` is three control tokens, always present: `<digits>` or
`<nodigits>`, `<filler>` or `<nofiller>`, `<verbatim>` or `<noverbatim>`. The
defaults (digits, no fillers, verbatim) are the template's. A request's
`AsrTaskOptions.style` overrides them field by field; it is part of the
prepared-artifact key and only variants with
`supports_asr_transcript_style` accept a set field.

With a forced language the assistant turn is prefilled and the model emits
the transcript alone. Without one it emits `language {Name}<asr_text>` first;
`split_rumble_output` splits on the `<asr_text>` id, so transcript text that
reads "language ..." is never taken for the header, and a stream emits
nothing before the marker. `<non_speech>` is dropped from transcripts.

`resolve_rumble_language` maps a code or name to the model's name for the 30
documented languages and for the 15 the card lists without a documented
name; other values pass through title-cased.

## Execution

The ASR capability offers two graphs
(`row(F::RumbleAsr, C::Asr, Loaded::RumbleAsr, RetainedAndInvocation).sequence().resumable().chunked()`).

One pass covers audio up to the model's whole 30 s window
(`asr_single_pass_fills_invocation_window`): the window is a fixed feature
window the front end checks itself, so the planner's 5% margin applies only
to the chunks of longer audio (28.5 s each) and to a context too small for a
full window.

**Retained rows** take audio that fits one pass, on a load that offers the
route. Request preparation computes the log-mel on a host worker, then
`RumbleEncoderBatcher` encodes the audio and splices it into each prompt on
the device (`asr.encoder.rumble`). Requests that queue while a pass runs
encode together in the next one, up to the engine's rows: the tower packs
every row's conv chunks and attends inside each row's own windows, so rows
of any length share a pass with no padding (a ragged batch, each row priced
at its own geometry). A packed and a lone encode agree up to rounding; one
row takes the scalar path.
The prompt prefills onto retained decoder pages in resumable spans
(`asr.rumble.prefill.scalar`), and the row then decodes beside the other
rows, one batched step per token (`asr.rumble.decode.tensor_continuous`).
Each row claims its prompt plus its decode budget from a pool that holds one
sequence per retained row and never less than one 1,536-token sequence. The
budget is 512 tokens by default; any budget is narrowed to the room the
sequence leaves after the prompt. A row decodes the same tokens alone and in
a batch, and the same tokens as an atomic invocation.

**Long files** run as chunk rows (`runtime/asr/chunk_rows.rs`,
`IZWI_ASR_LONG_FORM_FANOUT`): the runtime plans the file once and submits
each 28.5 s chunk as its own single-pass request, up to the ASR row width at
once, and commits the texts in plan order. On a load with the retained route
the chunks of one file encode and decode as a batch, beside other users'
requests; without it each chunk is one atomic invocation, so other requests
wait at most one chunk. Before chunk rows, a long file ran its chunks one
after another in one dispatch (see the H100 run under What is verified).

**Atomic invocations** take long files when chunk rows are off, and every
request on a load without the retained route: each chunk is one invocation
on a lifecycle-owned paged KV lease of 1,536 tokens (the prompt's markers and
context, at most 256 tokens; one window of audio tokens, 390; and the decode
budget), chunked inside the dispatch. An atomic invocation holds the device
for its whole decode. A chunked transcript reports the language most of its
chunks named (the earliest on a tie), and a streamed one names the detected
language in its final event.

The retained route is on by default on CUDA and off on CPU and Metal;
`IZWI_RUMBLE_RETAINED_DECODE` overrides the default on any backend
(`rumble_retained_decode_enabled(backend)`). With it off every request runs
atomically and the load fits no retained decoder state. The measurements
below are why: at a width of a few rows the route costs a single request
latency and gains the CPU nothing under load.

The decode width is the engine's rows on CPU and Metal. On CUDA with no
width fixed it is up to 32 rows: each load fits 32, 16 or 8 to its memory
from a priced row (14 layers of BF16 K and V over the 1,051-token minimum
sequence, about 59 MiB). The encoder and prefill keep the engine's rows.

On CUDA the batched decode step replays as one CUDA graph per padded width
(1, 2, 3, 4, 6, 8, 12, 16, 24 or 32 rows), the graphs Qwen3-ASR uses on the
same trunk. A graph is checked against the eager body on its first two
steps and each width's first step against the standard decode; a mismatch
turns the graphs off for the process. `IZWI_RUMBLE_DECODE_GRAPHS=false`
keeps the standard decode, a load prices 256 MiB for the graphs, and
`decode_graph` in a row's diagnostics reports captures, replays and why the
graphs are off. They need a BF16 or F16 decoder: a decoder that reloaded in
F32 decodes without them. None of this has run on a GPU.

Measured on an M1 Pro, where the engine's automatic width is 2, with the
route switched on against the same build with it off:

| | Retained route | Atomic only |
|---|---|---|
| Metal, 9 concurrent requests (3.6 s, 12.4 s and 27.3 s clips) | 6.1 to 6.9 s | 7.5 s |
| Metal, one 3.6 s clip | 0.35 s | 0.25 s |
| Metal, one 27.3 s clip | 1.77 s | 1.63 s |
| CPU, 8 concurrent requests | 25.9 s | 21.5 to 25.2 s over four runs |

A retained row pays for its extra dispatches (the encoder job, the prefill
and one scheduler quantum per token), about 0.1 s a request here. At width 2
that buys little on Metal and nothing on the CPU, where a two-row step takes
as long as two steps. The route is built for wider
devices, which is why CUDA alone takes it by default; it has not been
measured on one.

Live transcription and dictation use the generic rolling-window route, the
diarization pipeline uses the model as its transcription stage, and word
timestamps come from the forced aligner.

Not implemented: the checkpoint's auxiliary CTC head, which izwi does not
download, and argmax inside the decode graph (the logits are read back each
step, as for Qwen3-ASR).

## Memory

`price_rumble` in `runtime/lifecycle/asr_memory.rs` prices the published
census (`tests/fixtures/asr_residency/rumble_2_flash.json`):

| Backend | Resident weights | Catalog figure |
|---------|------------------|----------------|
| CPU | 4 bytes a parameter plus the position table (1.9 GiB) | 3.5 GiB |
| Metal | The same weights; the loader holds staged uploads until its fence | 4.0 GiB |
| CUDA | F32 audio (0.69 GiB) and a BF16 decoder (0.60 GiB) | 2.0 GiB |

## What is verified

- Unit tests on the synthetic checkpoint: the loader and its census refusal,
  the prompt against the template, the output split, streaming, the decode
  loop's budget and cache bookkeeping, and that the audio conditions the
  decode. For the retained route: it emits the atomic route's tokens when
  prefilled in spans, a row decodes the same alone and in a batch, and
  misuse is refused. The loaded adapter's graphs and the load's retained
  state are tested per backend.
- The tensor plan and the memory plan against the published header.
- On the published checkpoint (`reference_tests.rs`, CPU): the special ids
  against the published added-token table, the default prompt, and a
  transcript of `data/fox.wav` with a forced and a detected language. On
  CPU and Metal: four requests (detected and forced language, a style, a
  context prompt on a shorter clip) run atomically, as retained rows
  prefilled in 48-token spans and decoded alone, and as one batch of four;
  the text, language and tokens are the same each way.
- Live, through a release server on CPU and on Metal (M1 Pro): download by
  the server's own downloader at the pin, load, and 13 requests with
  identical transcripts on both backends: forced and detected language, each
  style option, a context prompt, a partial chunk, a 0.3 s clip, silence,
  stereo 44.1 kHz input, a 150 s recording, forced-aligner word timestamps
  and SSE streaming. Also a durable transcription job, four concurrent
  requests, the realtime WebSocket, a diarization job with Rumble as the
  transcription stage, and the CLI. Real-time factor about 0.06 on Metal and
  0.2 on the CPU.
- With the final defaults (retained route off on CPU and Metal), the same
  13 requests on both backends, and again on Metal with the route forced
  on: identical transcripts and languages each way, including a 29.7 s clip
  in one pass and the streamed final event's language. Speech synthesised
  by the macOS system voices in German, French, Spanish and Italian was
  transcribed word for word with the language detected, and a 52 s German
  clip and a 150 s English one reported their language from three and seven
  chunks. A durable job on the German clip and the realtime WebSocket were
  rerun on Metal.
- With the batched encoder (route forced on): on the published checkpoint,
  four requests encoded in one tower pass and decoded as one batch emit the
  atomic route's text, language and tokens on CPU and Metal. Live on Metal,
  the 13 requests and the language clips returned the same transcripts as
  before, 9 and 16 concurrent requests matched their solo runs, and the
  metrics showed two-row encoder passes.
- With the retained route, the same 13 requests returned the same
  transcripts, languages and diagnostics on both backends as the atomic
  route had. Under 8, 9 and 16 concurrent requests every transcript equalled
  its solo run, and the engine's metrics showed two-row decode steps. On
  Metal the realtime WebSocket, a transcription job with a style option and
  a diarization job were rerun on the retained route.
- On an H100 (80 GB HBM3, BF16 decoder, build 65070de7b, before chunk
  rows), the ASR serving matrix with the client on the same host:
  - Clips of 5–15 s: 110 audio s/s for one stream (p50 0.10 s), 497 for
    32 streams.
  - 30 s clips: 91 audio s/s for one stream, 662 for 32 streams.
  - The decode graphs captured and replayed, and decode steps reached 32
    rows.
  - At 32 streams of 30 s clips, the GPU time split 45% decode, 35%
    encoder (two rows a pass on average) and 20% prefill.
  - Transcripts under load matched their solo runs, except for four
    punctuation, case or one-word differences in 32 compared.
  - 600 s files ran their 29 chunks serially in one dispatch: 69 audio s/s
    for one file and 77 for four. Chunk rows were added for this.
- Chunk rows, live on Metal (M1 Pro, two rows in flight) with a 150 s and a
  600 s interview recording:
  - Transcripts and detected language were identical with chunk rows on and
    off, on the retained route and on the default atomic route.
  - Three concurrent 150 s files each matched their solo run.
  - With the retained route forced on, the 600 s file took 39.6 s against
    57.6 s without chunk rows.
  - On the default route it took 49.1 s against 55.6 s, and three
    concurrent 150 s files took 33.3 s against 41.3 s.

Not verified:

- Parity against transformers. `scripts/fixtures/generate_rumble_reference.py`
  writes the fixture the ignored parity tests read; it has not been run.
- Long files as chunk rows on CUDA (only on Metal so far), and the decode
  graphs' gain over the standard decode (the A/B was cut short when the spot VM was preempted).
- Languages other than English, and accuracy on a benchmark set.
