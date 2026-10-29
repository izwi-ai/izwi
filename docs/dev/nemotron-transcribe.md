# Nemotron 3.5 Transcribe

izwi serves NVIDIA's Nemotron 3.5 Transcribe 30B-A4B FP8 as an offline speech
recognition model. It transcribes 40 languages in one pass of up to 60
minutes, translates to or from English, takes context terms, tags speakers,
times words with its own CTC head, and can diarize a recording on its own. It
is its own model family
(`nemotron_transcribe`), separate from `nemotron`
(`Nemotron-3.5-ASR-Streaming-0.6B`, the FastConformer RNNT model) and
`sortformer` (`Nemotron-3-Diarization`); it imports neither.

CPU, Metal and CUDA are implemented. None is certified: the model has run end
to end only with a tiny synthetic checkpoint, on CPU and Metal. It has not run
with the real weights on any backend, because the development laptop cannot
hold it. CUDA is compile-checked only. Certification is done with the
[hand-off test kit](../../scripts/e2e/nemotron-transcribe/README.md); see
[What is verified](#what-is-verified).

User-facing behaviour is documented in the
[support matrix](../user/support-matrix.md), the
[models guide](../user/models/index.md), the
[transcription guide](../user/features/transcription.md), the
[API reference](../user/api.md) and the
[`izwi transcribe` reference](../user/cli/transcribe.md).

| Variant | Files | Family |
| --- | --- | --- |
| `Nemotron-3.5-Transcribe-30B-A4B-FP8` | 7 weight files, 34.7 GB | `NemotronTranscribeAsr` |

## Pinned checkpoint

| Repository | Revision | Licence |
| --- | --- | --- |
| `nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B-FP8` | `006d73cd8385237ed5603fc6c7e76200e50abbd2` | OpenMDW-1.1 |

The repository and revision constants are in
`crates/izwi-model-catalog/src/catalog/metadata.rs`. The download
(`crates/izwi-core/src/artifacts/downloader.rs`) fetches the configs, the
tokenizer, the chat template, the index and the seven weight files: four
backbone shards, `perception.safetensors`, `model-ctc-bf16.safetensors` (the
CTC timestamp head) and `model-mtp-fp8.safetensors` (the MTP head). A bundle
is complete only when every weight file has its exact size at the pinned
revision. The MTP head is downloaded but never loaded.

The NVFP4 and BF16 siblings are not supported; resolution refuses them by
repository id.

## Architecture

The checkpoint is a NeMo SpeechLM bundle (`model_type: nemo_speechlm`). From
audio to text:

1. **Log-mel front end** (`frontend.rs`, host): NeMo's `FilterbankFeatures`
   at inference, with the checkpoint's stored filterbank and window after a
   check against a recomputation. The ASR branch normalizes per feature over
   the whole recording; the diarizer reads the raw log-mel.
2. **Parallel Expert Encoder** (`perception.rs`): a 32-layer Transformer ASR
   encoder with FP8 linears (`speech_transformer.rs`) and the
   Nemotron-3-Diarization Sortformer run as a streaming speaker encoder
   (`sortformer.rs`). Both run over 40 s windows (500 encoder frames of 80 ms)
   with 4 s of context on each side. Each core frame fuses the ASR output with
   the binarized speaker activity through a sin/cos speaker kernel, and is
   projected to the decoder width: one audio token per 80 ms.
3. **Nemotron-H 30B-A3B decoder** (`decoder/`): 52 pre-norm layers, 23 Mamba2,
   23 mixture-of-experts and 6 attention layers. The MoE layers route each
   token to 6 of 128 relu² experts with a sigmoid router and an F32 selection
   bias, plus one shared expert. Attention is GQA (32 query heads, 2 KV heads,
   head dim 128) with no RoPE. The residual stream is F32.
4. **CTC timestamp head** (`ctc_head.rs`, `ctc/`): times the words of a
   finished transcript. See [Word timestamps](#word-timestamps).

The published dimensions are in
`crates/izwi-core/tests/fixtures/nemotron_transcribe/config.json`.
`config.rs` checks every switch the family implements one way (activations,
norms, routing, quantization) and fails the load when one differs, and the
checkpoint census (`checkpoint.rs`) fails it on any missing, extra or
reshaped tensor before a byte is uploaded.

### Module layout

`crates/izwi-core/src/models/architectures/nemotron_transcribe/`:

| Module | Holds |
| --- | --- |
| `mod.rs` | The family's exports and the isolation test that keeps it from importing other families |
| `config.rs` | Config parsing; fails closed on any switch the family does not implement |
| `checkpoint.rs` | The index, memory maps, tensor census and FP8 ingestion |
| `residency.rs` | What a load holds on each backend, priced before any weight is read |
| `precision.rs` | FP8 semantics and the `vllm-fp8` parity mode |
| `linear.rs` | Linear layers in each backend's resident form |
| `frontend.rs` | The NeMo log-mel front end and per-feature normalization, on the host |
| `speech_transformer.rs` | NeMo's `TransformerEncoder`: the ASR encoder, the diarizer's encoder and the CTC trunk |
| `sortformer.rs` | The embedded Nemotron-3-Diarization Sortformer as a streaming speaker encoder |
| `perception.rs` | The Parallel Expert Encoder, run lazily one 40 s window at a time |
| `decoder/mod.rs` | The Nemotron-H decoder |
| `decoder/mamba.rs` | The Mamba2 mixer |
| `decoder/moe.rs` | The sigmoid-routed mixture of experts |
| `decoder/attention.rs` | GQA over the paged KV cache, without RoPE |
| `decoder/state.rs` | A row's decoder state and its state contract |
| `prompt.rs` | The `nemotron3p5` prompt, the card's instructions, the language table and the audio splice |
| `model.rs` | The loaded model and its request rows |
| `transcript.rs` | Output tags, speaker turns, the generation-loop guardrail and the online loop detector |
| `ctc_head.rs` | The CTC timestamp head (NeMo's `TransformerCTCDecoder`) |
| `ctc/` | The CTC tokenizer and NeMo's multi-speaker word timestamp aligner, on the host |
| `synthetic.rs` | A tiny checkpoint in the published layout, for tests |
| `model_tests.rs`, `reference_tests.rs`, `ctc/golden_tests.rs` | Tests; see [Tests](#tests) |

### Prompt

`prompt.rs` renders the card's template with an empty system turn and thinking
off. The single `<|audio|>` placeholder expands to one position per audio
token. The instruction is chosen in this order:

- `prompt` on the request replaces the instruction entirely, after tags are
  stripped (at most 4,000 characters). Combined with any option below it is
  refused (`400`), since that option would be dropped.
- `target_language` renders "Please translate the given speech to …". A
  translation runs to or from English, never between two other languages
  (31 languages translate; Turkish, Bengali, Indonesian, Marathi, Tagalog,
  Tamil, Telugu and Vietnamese do not). Context terms and speaker tags
  cannot be combined with it (`400`).
- Speaker labels render the card's speaker-tag instruction, with context
  terms appended when given. They are refused when the named language is not
  one of the card's 13 multi-speaker languages (en, fr, de, hi, it, ko, pt,
  ru, es, ja, th, ar, zh).
- Context terms (`keywords`) render the card's
  "Transcribe with punctuation and capitalization. Context terms: …" form: at
  most 64 terms of at most 80 characters, tags stripped, duplicates dropped.
- Otherwise, the card's "Readable text" instruction (fillers omitted,
  punctuation and capitalization), naming the language when the request
  gives one.

The language table is the card's support matrix: 40 languages transcribe, and
Norwegian is listed without ASR. A request without a language, or with `auto`,
lets the model detect it. Codes, English names and a few aliases resolve
(`pt-BR`, `nb`, `fil`, `iw`, `Mandarin`); anything else is refused before
admission (`validate_asr_language` and `validate_asr_task` in
`runtime/asr.rs`).

### Output

`transcript.rs` post-processes the decoded text:

- The model writes `<spk:N>` speaker tags (when prompted), `<ev:*>` event
  tags and `<sty:*>` style tags. The returned text has all of them removed;
  the raw text in the diagnostics keeps them. Speaker tags also split the
  text into turns (`speaker` is null for text before the first tag).
- The card's generation-loop guardrail, adapted from Qwen3-ASR, runs once on
  the final text: a character repeated 21 or more times, then a block of
  1–4,096 characters repeated 20 or more times, collapse to one copy. Rust's
  `regex` has no backreferences, so both are hand-written scanners with
  Python's semantics.
- An online loop detector stops a row once its last tokens are one block
  (up to 256 tokens) repeated 20 times, instead of letting it run to the
  token budget.

`finish_reason` in the ASR diagnostics is `stop` (the stop token), `length`
(the token budget) or `repetition` (the loop detector).

## Residency and precision

`linear.rs` and `residency.rs` hold the per-backend forms:

| Weight | CPU | Metal | CUDA |
| --- | --- | --- | --- |
| FP8 linears | Q8_0, re-quantized at load | E4M3 bytes + scale grid | E4M3 bytes padded to 64×128 + scale grid |
| BF16 linears and tables | F32 | F32 | BF16 |
| Router gates, norms, Mamba parameters | F32 | F32 | F32 |
| Language-model head | Q8_0 | F16 | BF16 |
| Embedding table | BF16, host | BF16, host | BF16, host |
| Activations | F32 | F32 | BF16 |
| Resident total | about 33.6 GiB | about 31.6 GiB | about 31.6 GiB |

CUDA needs compute capability 8.0 or newer; an older GPU fails the load. CUDA's
FP8 provider needs `rows % 64 == 0` and `cols % 128 == 0`, so CUDA pads both
with zeros, which is exact. Metal keeps F32 activations so relu² expert
intermediates never meet F16's range.

FP8 linears run W8A16 by default (`IZWI_NEMOTRON_TRANSCRIBE_FP8_MODE=w8a16`):
the checkpoint's weights exactly (`decode(byte) * weight_scale`), unquantized
activations and a dense KV cache. That is also what vLLM's Marlin path computes
on SM80 and SM86 GPUs. `vllm-fp8` emulates vLLM's W8A8 path on SM89 and newer:
every FP8 linear rounds its input to E4M3 with its static input scale, q/k/v
share the largest of their scales as vLLM's fused QKV does, and K and V
round-trip through E4M3 before the cache. It exists to tell an izwi bug from
the W8A8/W8A16 difference when a transcript disagrees with a vLLM run; the
activation rounding runs on the host, so it is slow. Any other value fails the
load.

Load admission prices the residency from the config's census before any weight
is read (`nemotron_transcribe_plan` in
`runtime/lifecycle/asr_memory.rs`), so a machine that cannot hold the model is
refused with a clear message instead of failing mid-load. At load, in W8A16
mode, a sample of each resident projection kind is checked against the
exactly dequantized checkpoint weights, and on Metal and CUDA the device
Mamba2 scan is checked against the CPU scan; either check failing fails the
load.

## Engine path

### Retained rows

Every request is a retained engine row. The pieces:

- **Adapter**: `NemotronTranscribeAsrExecutionAdapter` in
  `crates/izwi-core/src/runtime/adapters/loaded/asr.rs`. Its contract has a
  static prefill stage (`asr.prefill.nemotron_transcribe.tensor_static`,
  bounded spans per row) and a continuous decode stage
  (`asr.decode.nemotron_transcribe.tensor_continuous`), with cancellation at
  each sequence step and prefix reuse off.
- **Preparation**: `prepare_nemotron_transcribe_row` in
  `crates/izwi-core/src/engine/request/asr_preparation.rs`, called from
  `runtime/service/asr_preparation.rs`. It decodes and resamples the audio,
  computes the log-mel and its normalization, renders the instruction, plans
  the windows and the decode budget, and installs the row. No device work
  happens here: the perception windows encode inside the prefill spans.
- **Rows handler**: `crates/izwi-core/src/engine/executor/handler_nemotron_transcribe_rows.rs`.
  Each quantum either prefills one span of a row's prompt (encoding the
  perception windows its audio positions need) or steps every decoding row
  one token, with the decoder batched across rows. A quantum opens a
  checkpoint; the Mamba state is staged into the tensor-state arena and
  committed with the KV pages, so a failed or cancelled quantum rolls the row
  back to where it started. Text streams as it decodes. A finished row
  reports `finish_reason`, `raw_text`, `speaker_turns`, token counts and the
  number of perception windows in its ASR diagnostics. Rows are cleaned up on
  cancel, session end and purge.
- **Scheduler**: the family sets `ModelFamily::requires_prefill_spans`, which
  becomes the execution profile's `prefill_spans_required`
  (`engine/executor.rs`, `engine/scheduler/mod.rs`). The scheduler then
  prefills this model's prompts in resumable spans whatever the
  chunked-prefill switch says. A one-hour prompt (about 45,000 audio tokens)
  is 23 spans of 2,048 tokens on CUDA or 88 of 512 on CPU and Metal, so it
  never holds the device for its whole prefill while other rows wait.

Decode rows run at the engine's tensor batch width for speech models: two rows
on CPU and Metal unless `IZWI_MAX_BATCH_SIZE` sets one, and CUDA's automatic
width.

### Row state

`decoder/state.rs` declares one transactional consistency group per row:

| Domain | Kind | Holds |
| --- | --- | --- |
| 1 | Paged attention, raw keys | KV for the 6 attention layers |
| 2 | Tensor | The Mamba2 recurrent state, F32, per Mamba layer |
| 3 | Tensor | The Mamba2 convolution state, F32, per Mamba layer |

The Mamba state is about 47.6 MiB per row. The state is allocated through the
managed arena at load (`allocate_nemotron_transcribe_state` in
`runtime/lifecycle/load/state.rs`). Prompt and completion share a
65,536-token context; the completion budget is the card's 16,384 tokens. A
row books KV for a planned decode of `min(16384, ceil(8 × seconds) + 128)`
tokens (`planned_decode_tokens` in `model.rs`), which covers CJK and
tag-heavy output. Prefixes are not shared: prompts share only about 15 tokens
before the audio, and vLLM serves this model without prefix caching too.

## Word timestamps

Word timestamps come from the checkpoint's own CTC head and NeMo's multi-speaker
forced aligner, ported. No forced-aligner model is needed or loaded.

- `ctc_head.rs` runs NeMo's `TransformerCTCDecoder` on the stored ASR encoder
  output (before speaker fusion), in the encoder's 40 s windows: a transposed
  convolution doubles the 80 ms frames to 40 ms, a two-layer trunk and a
  1×1 convolution give log-probabilities over 23,863 classes (blank last).
  Only the columns the transcript's targets use leave the device, so an hour
  (90,000 CTC frames) never materialises whole.
- `ctc/` is NeMo's `MultiSpeakerSOTWordTimestampAligner`
  (`speechlm2/parts/ctc_timestamp_utils.py` at NeMo Speech `743beaf4`) on the
  host. `prepare` splits the transcript into speaker streams and
  re-tokenises each with the CTC SentencePiece model (`ctc/tokenizer.rs`, a
  port of sentencepiece's normalizer and BPE). `align` maps speaker tags to
  diarizer columns (`ctc/assignment.rs`, a port of SciPy's
  `linear_sum_assignment`), runs the anchor-guided banded max-sum CTC search
  per stream (`ctc/search.rs`), refines word boundaries against the
  diarization segments and rounds times to milliseconds. The result has the
  shape of the `ctc_timestamps` object NeMo's vLLM integration returns.
- The rows handler puts that object in `asr_diagnostics["ctc_timestamps"]`,
  and its words without event and style tags (which the returned text leaves
  out) in `asr_diagnostics["timestamp_words"]`, the field the transcription
  routes read. `speaker_turns` take their start and end from their own
  words. When the transcript cannot align (too many tokens for the CTC
  timeline, a character the CTC vocabulary lacks), `error` is
  `alignment_failed` and the diarizer's segments are still returned, as
  upstream does.
- The aligner reads the gathered columns on the host, `frames × columns` in
  F32 as upstream: an hour with about 2,000 distinct tokens is roughly
  700 MB for the duration of the stage.

Numerics follow upstream, including its tie order. Deviations, listed in the
module docs:

- Thai keeps whitespace words: upstream segments Thai with pythainlp's
  `newmm`, which izwi does not ship.
- The banded search keeps one score row every √frames frames and recomputes
  backpointers segment by segment, instead of upstream's full backpointer
  matrix: the same path, in bounded memory, at about twice the forward time.

A translation has no word timestamps. Without `model-ctc-bf16.safetensors` the
model loads without them. On a transcription job, when the model's alignment
yields no words, the server falls back to the forced aligner
(`crates/izwi-server/src/api/transcription/handlers.rs`).

## Speaker labels

The model tags speakers itself: the speaker-tag instruction makes it write
`<spk:N>`, and the Sortformer's speaker activity is already fused into its
audio tokens. The catalog marks this with `supports_asr_speaker_tags`.

- **`/v1/audio/transcriptions`**: the izwi extension field
  `speaker_labels=true` (JSON or multipart) asks for tags, and `verbose_json`
  carries `speaker_turns: [{speaker, text, start?, end?}]`. `speaker` is the
  model's tag number, null for text before the first tag; times are present
  when the CTC stage timed the words. The CLI's `--speaker-labels` sets the
  field and `verbose_json`.
- **Speaker-attributed ASR jobs** (`job_kind=speaker_attributed_asr`) accept
  the model; Granite Speech 4.1 2B Plus stays the default. The job runs one
  transcription row with speaker tags and word timestamps, labels tag N as
  "Speaker N+1", merges consecutive turns of one speaker and times turns when
  the word timestamps align (`speaker_tagged_turns` and
  `speaker_tagged_asr_input_with_progress` in
  `runtime/asr/speaker_attribution.rs`). `min_speakers` and `max_speakers`
  only raise warnings; they do not steer the model.
- **The diarization pipeline** refuses the model as its ASR stage: it
  attributes speakers in its own transcripts, and its transcripts run only as
  resumable rows, never as the one blocking stage the pipeline needs
  (`serves_diarization_pipeline_asr`, and
  `resolve_diarization_asr_model_variant` in
  `crates/izwi-model-catalog/src/catalog/variant.rs`). A diarization job can
  name it as its diarization model instead; see
  [One-model diarization](#one-model-diarization).

## One-model diarization

A model that tags speakers and times its own words diarizes a recording in one
transcription (`ModelVariant::diarizes_in_one_model`:
`supports_asr_speaker_tags && supports_native_word_timestamps`; only this
model). `resolve_diarization_model_variant_strict` accepts it as a diarization
`model`, and the admin API reports `one_model_diarization: true`.

- **Server gate**: `reject_pipeline_models_for_one_model_diarization` in
  `crates/izwi-server/src/api/diarization/handlers.rs` returns 400 ("diarizes
  on its own") when the request also names `asr_model` or `aligner_model`.
  `validate_speaker_bounds` caps `min_speakers` and `max_speakers` at the
  embedded Nemotron-3-Diarization head's 8 channels
  (`diarization_speaker_channels`).
- **Runtime branch**: `diarize_with_transcript_input` in
  `crates/izwi-core/src/runtime/diarization.rs` checks the same rule, then
  calls `diarize_in_one_model` (`runtime/diarization/one_model.rs`) instead of
  the pipeline. No diarizer, pipeline ASR or aligner is loaded. It runs one
  background transcription with speaker tags and word timestamps, passing the
  job's `language`; canonical 16 kHz samples go in as a WAV.
- **The record** comes from `asr_diagnostics["ctc_timestamps"]`
  (`one_model_transcript`):
  - `segments`: the embedded diarizer's `diarization` segments (10 ms
    predictions above 0.5), labelled `SPEAKER_NN` by diarizer column;
  - `words`: the aligner's words without event and style tags, each labelled
    with the column `speaker_tag_to_diarization_speaker` assigned its
    `<spk:N>` tag (the Hungarian mapping), or `UNKNOWN` when the tag has none;
  - `utterances` and `transcript`: `build_utterances` and
    `format_utterance_transcript`, as in the pipeline;
  - `asr_text`: the model's transcript; `alignment_coverage`: the share of
    words with a column; `speaker_count`: the columns seen in segments and
    words.
- **Warnings, not failures**: missing or unreadable timestamps, an alignment
  `error` (the job keeps the transcript and the segments, with no words), and
  a speaker count outside `min_speakers`/`max_speakers`. The model counts
  speakers itself; the bounds do not steer it.
- `min_speech_duration_ms`, `min_silence_duration_ms` and the VAD gate are not
  used. The UI offers the model under "Single model" in the diarization
  picker and hides and omits the ASR, aligner and timing fields; `izwi
  diarize --asr-model` is optional, so `izwi diarize call.wav --model
  Nemotron-3.5-Transcribe-30B-A4B-FP8` runs this path.

Unit tests in `one_model.rs` cover word-to-column mapping with event tags
dropped, and a failed alignment that keeps the segments. Like the rest of the
family, it has not run on the real weights.

## Routes and capabilities

The admin API's `route_capabilities` for this model:

| Flag | Value | Why |
| --- | --- | --- |
| `speaker_attributed_asr` | `true` | It tags speakers on its ASR rows |
| `speech_translation` | `true` | To or from English |
| `keyword_biasing` | `true` | The card's context terms |
| `native_word_timestamps` | `true` | The CTC head times words |
| `diarization_pipeline_asr` | `false` | It attributes speakers itself |
| `one_model_diarization` | `true` | It diarizes on its own |
| `speech_to_text_realtime` | `false` | The card says offline |
| `voice_realtime_modular_asr` | `false` | The card says offline |

`native_word_timestamps`, `diarization_pipeline_asr` and
`one_model_diarization` are new flags; the UI reads them to skip the aligner,
to filter the diarization pipeline's ASR choices and to offer the model as a
single-model diarizer. Live transcription, voice-mode ASR and audio chat
refuse the model (`serves_live_asr` in
`crates/izwi-model-catalog/src/catalog/metadata.rs`).
The no-speech gate applies: audio without speech returns an empty transcript
without running the model.

Not supported: prefix caching, MTP speculative decoding (the head is
downloaded but not loaded), the NVFP4 and BF16 siblings, and izwi-side
chunking (one pass covers the 60-minute cap).

## Tests

**Synthetic checkpoint** (no weights; normal suite). `synthetic.rs` writes a
tiny checkpoint with every real tensor name and dtype. `model_tests.rs` covers
load and census, the retained-row API, span and batch invariance, rollback
replay and CPU/Metal agreement; `runtime/nemotron_transcribe_tests.rs` runs it
through `RuntimeService` on the CPU (load admission, 16-token resumable spans,
three concurrent rows transcribing exactly as alone, request validation).

```bash
cargo test -p izwi-core --lib nemotron_transcribe
cargo test -p izwi-core --features metal --lib nemotron_transcribe
```

**CTC goldens** (`ctc/golden_tests.rs`). Replays
`crates/izwi-core/tests/fixtures/nemotron_transcribe/ctc_goldens.json`, which
`scripts/fixtures/generate_nemotron_transcribe_ctc_goldens.py` writes by
running NeMo's aligner on synthetic inputs. Alignment goldens run everywhere;
tokenizer and `prepare` goldens need the real CTC SentencePiece model, read
from `$IZWI_NEMOTRON_TRANSCRIBE_MODEL_DIR`, and report themselves as skipped
without it.

**L1, tiny upstream fixture** (`reference_tests.rs`). A random checkpoint in
the published layout and what NeMo's perception and transformers' NemotronH
computed from it, written once by
`scripts/fixtures/generate_nemotron_transcribe_tiny.py` into
`crates/izwi-core/tests/fixtures/nemotron_transcribe/tiny/`. Runs on CPU and
Metal when the fixture is present and reports itself as not run otherwise.

**L2, real checkpoint** (`reference_tests.rs`, ignored). Stage-by-stage parity
against the reference `scripts/fixtures/generate_nemotron_transcribe_reference.py`
writes on a GPU box: Leg A is an F32 forward pass with dequantized weights,
Leg B is vLLM. It reads `IZWI_NEMOTRON_TRANSCRIBE_MODEL_DIR`,
`IZWI_NEMOTRON_TRANSCRIBE_AUDIO_DIR`, `IZWI_NEMOTRON_TRANSCRIBE_BACKEND`
(`cpu`, `metal` or `cuda`) and optionally `IZWI_NEMOTRON_TRANSCRIBE_REFERENCE`.

**Hand-off kit.** [`scripts/e2e/nemotron-transcribe/README.md`](../../scripts/e2e/nemotron-transcribe/README.md)
is the runbook for machines that can hold the model: `fetch.sh`, `build.sh`,
`run-parity.sh`, `run-e2e.sh`, `run-load.sh`, `wer-mini.py` and
`collect.sh`, with the hardware each needs and what to send back.

The tests that read real weights report `not run` or `skipped` when the files
are missing; a pass without them proves nothing.

## Environment

The family's knobs, in [`env/core.md`](env/core.md):

- `IZWI_NEMOTRON_TRANSCRIBE_FP8_MODE` (read at load): `w8a16` (default) or
  `vllm-fp8`. Anything else fails the load.
- `IZWI_NEMOTRON_TRANSCRIBE_PREFILL_SPAN` (read at load): the most prompt
  tokens one row prefills per engine step, 16 to 16,384. Defaults to 2,048 on
  CUDA and 512 on CPU and Metal; values outside the range are ignored.

`IZWI_NEMOTRON_TRANSCRIBE_MODEL_DIR`, `_AUDIO_DIR`, `_BACKEND` and
`_REFERENCE` are read only by the tests.

## What is verified

- **Synthetic checkpoint, CPU and Metal**: verified end to end, including
  retained spans, concurrent rows and rollback.
- **Real weights**: not run on any backend. Real-weight parity (L2), the HTTP
  end-to-end matrix, load parity and WER against the card are what the kit
  collects.
- **CUDA**: compile-checked only. No NVIDIA GPU has run it, including the
  padded FP8 provider and BF16 activations.
- **Memory**: the residency figures are computed from the published config,
  not measured on a loaded model.
- **Word timestamps**: the host aligner is checked against upstream goldens
  (`ctc/golden_tests.rs`); the CTC head has not run on real weights.
- **One-model diarization**: unit-tested on synthetic `ctc_timestamps`
  only. The kit's `one_model_diarization` e2e case still skips (it predates
  the request shape), so the kit does not exercise it yet.
