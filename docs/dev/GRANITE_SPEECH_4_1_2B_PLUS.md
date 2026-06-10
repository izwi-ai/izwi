# Granite Speech 4.1 2B Plus

Implementation status for `ibm-granite/granite-speech-4.1-2b-plus`.

## Supported Surface

- Catalog variant: `Granite-Speech-4.1-2B-Plus`.
- Native Rust/Candle ASR loader and offline transcription runtime.
- Hugging Face artifact bundle validation for config, processor, tokenizer,
  chat template, generation config, safetensor index, and three weight shards.
- Granite Speech Plus audio frontend with 16 kHz input, 80-bin log-mel
  extraction, adjacent-frame stacking to 160 dims, Conformer encoder,
  Q-Former projector, and Granite causal decoder.
- OpenAI-compatible `/v1/audio/transcriptions` routing with `prompt`,
  `max_tokens`, `stream`, and `timestamp_granularities`.
- CLI request support through `izwi transcribe --model
  Granite-Speech-4.1-2B-Plus --prompt ... --max-tokens ...`.
- Shared UI API request support for direct ASR prompt, max-token, and
  timestamp controls.

## Rich Transcript Behavior

Granite Speech diagnostics expose:

- `speaker_segments` parsed from `[Speaker N]:` markers.
- `timestamp_words` parsed from `[T:N]` word-end centisecond markers.
- Prompt/audio/decode metadata, including `prompt_prefix_tokens` for chunked
  long-form requests.

The OpenAI-compatible transcription endpoint prefers model-provided
`timestamp_words` and `segments` before falling back to forced alignment. When
Granite only provides word end times, OpenAI-style word starts are inferred
from the previous word end.

## Chunking And Realtime Boundaries

Long-form chunking uses the shared ASR chunk planner. For Granite Speech, each
chunk receives the already-assembled prior transcript through the official
chat-template `prefix_text` slot.

Granite Speech is intentionally advertised as batch/offline ASR only:

- no realtime ASR adapter,
- no audio-chat adapter,
- no incremental decode state.

OpenAI `stream=true` can still emit chunk/result deltas through Izwi's offline
streaming path; it is not native realtime audio state.

## Verification

Focused verification used for this implementation:

```bash
cargo check -p izwi-core
cargo test -p izwi-core granite_speech
cargo test -p izwi-core chunk_plan_context_callback_receives_assembled_prefix_text
cargo check -p izwi-server -p izwi-cli
cargo test -p izwi-server granite
cargo test -p izwi-cli request_body_includes_prompt_max_tokens_and_word_timestamps
cd ui && npm run test -- src/shared/api/audio.test.ts
```

`cargo fmt -- --check` is currently not a useful project-wide gate because
unrelated existing Rust files fail formatting checks. For this change, only the
edited Rust files were formatted and `git diff --check` was clean.

## Optional Real-Model Smoke

After downloading the model artifacts, run a short local smoke through the
server or CLI:

```bash
izwi pull Granite-Speech-4.1-2B-Plus
izwi transcribe ./fixtures/audio/short.wav \
  --model Granite-Speech-4.1-2B-Plus \
  --format verbose-json \
  --max-tokens 128
```

For word timestamps, use:

```bash
izwi transcribe ./fixtures/audio/short.wav \
  --model Granite-Speech-4.1-2B-Plus \
  --word-timestamps \
  --max-tokens 256
```

Expected smoke result: non-empty `text`, Granite diagnostics in the server
payload, and no forced-alignment fallback when model-provided timestamp words
are present.
