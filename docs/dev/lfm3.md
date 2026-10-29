# LFM3 and LFM3-Audio

izwi serves Liquid AI's LFM3-1.7B as a text chat model and LFM3-Audio-2B as one
model for speech recognition, speech synthesis and voice chat. Both are
verified on real weights on CPU and Metal. CUDA builds and is compile-checked
only; see [Known gaps](#known-gaps) and the LFM3 section of
[LFM CUDA implementation and qualification](lfm-cuda.md#lfm3).

User-facing behaviour is documented in the
[support matrix](../user/support-matrix.md), the
[models guide](../user/models/index.md), the [API reference](../user/api.md)
(deferred tools, audio content parts) and the
[voice guide](../user/features/voice.md) (spoken tool calls).

| Variant | Files | Family |
| --- | --- | --- |
| `LFM3-1.7B-GGUF` (default) | `LFM3-1.7B-QAD-Q4_0.gguf` | `Lfm3Chat` |
| `LFM3-1.7B-Q8_0-GGUF` | `LFM3-1.7B-Q8_0.gguf` | `Lfm3Chat` |
| `LFM3-Audio-2B-GGUF` (default) | `LFM3-Audio-2B-Q4_K_M.gguf`, `mmproj-LFM3-Audio-2B-Q8_0.gguf` | `Lfm3Audio` |
| `LFM3-Audio-2B-Q8_0-GGUF` | `LFM3-Audio-2B-Q8_0.gguf`, `mmproj-LFM3-Audio-2B-Q8_0.gguf` | `Lfm3Audio` |

## Pinned checkpoints

Downloads use pinned revisions. The GGUF repositories ship no tokenizer, so the
tokenizer, chat template and configs come from the safetensors repositories,
also pinned. The constants are in
`crates/izwi-model-catalog/src/catalog/metadata.rs`; the companion download is
in `crates/izwi-core/src/artifacts/downloader.rs`.

| Repository | Revision | Files used |
| --- | --- | --- |
| `LiquidAI/LFM3-1.7B-GGUF` | `b3f2ae5d73cd3c91bf40e8ba752c648aa05dffdf` | QAD-Q4_0 and Q8_0 GGUFs |
| `LiquidAI/LFM3-1.7B` | `40ebe0aa6662c9d40389c2f5d565b6a2c6aebacd` | tokenizer, chat template, configs |
| `LiquidAI/LFM3-Audio-2B-GGUF` | `fcfae9027c00a6f3f872c48a841cfeb77e2b5a1c` | Q4_K_M and Q8_0 language models, Q8_0 mmproj |
| `LiquidAI/LFM3-Audio-2B` | `9d49701c372d6011d900f6abb40c6b296fb706dd` | tokenizer, chat template, configs |

The Rust chat templates, the tool-result renderer and the reference fixtures
name the companion revisions they were written against.

## Architecture

### The LFM2 backbone

LFM3 is a tri-hybrid: gated ShortConv layers, sliding-window attention layers
and global attention layers. izwi runs it on the LFM2 backbone
(`crates/izwi-core/src/models/architectures/lfm2/backbone.rs`) with its LFM3
features switched on. `lfm3/config.rs` parses the `lfm3.*` GGUF metadata into an
`Lfm2BackboneConfig` whose `lfm3` field (`Lfm3TrunkFeatures`) holds what LFM3
adds:

- a sliding window per layer, from `lfm3.attention.sliding_window` and
  `lfm3.attention.sliding_window_pattern` (256 tokens in LFM3-1.7B);
- a RoPE base of its own for the sliding layers (`lfm3.rope.freq_base_swa`);
- partial rotation on the global layers: `rope_freqs.weight` carries 1 for a
  rotated frequency and 1e30 for an unrotated one, and the loader refuses any
  other factor;
- a per-head attention output gate, `sigmoid(attn_gate(x))`, before the output
  projection;
- sandwich norms: an RMSNorm after each operator and after each FFN, before the
  residual add.

LFM2 and LFM2.5 checkpoints load with `lfm3: None` and keep their forward pass.
The same backbone runs LFM3-Audio's language model, its speech encoder and the
LiquidCodec decoder's trunk, each under its own tensor names. On Metal, with
fused kernels on and `IZWI_LFM2_DIAGNOSTICS` off, the sandwich norm and its
residual add and the output gate run as fused kernels
(`kernels::metal_lfm_norm`). Decode is bound by weight bandwidth: LFM3-1.7B
Q4_0 reads about 1.05 GB per token, so about 105 tok/s on an M1 Pro is
roughly 113 GB/s of its 200 GB/s (indicative, shared machine). Fusing the
decode rows' q/k norms as well cut dispatches but measured no gain, so it was
not kept; the lever is the Q4_0 mat-vec's bandwidth.

### Attention pages and ShortConv rings

The chat state contract (`lfm2/physical.rs`) has two domains in one retained,
transactional group:

- One paged-attention domain for every attention layer. Global and
  sliding-window layers share one page table; a sliding layer narrows its own
  view of it (`PhysicalPagedKvCache::sequence_table_with_window` for decode, a
  windowed view and `window_tokens` for prefill). Pages behind a sliding
  layer's window are kept: a request that attaches a shared prefix needs the
  window behind its boundary, which a released page would no longer hold.
- One ring domain with an F32 component per ShortConv layer, holding the
  recent inputs its causal convolution reads (`lfm3.shortconv.l_cache` steps).
  A step's ring update commits in the same transaction as its KV writes, so a
  rolled-back quantum restores both.

With prefix sharing on, attention publishes committed pages and the ring
publishes a snapshot at each page boundary; a request attaches a shared
prefix's pages and restores the ring at its end. The catalog turns it on for
LFM3 chat on CPU and Metal and leaves it off on CUDA
(`crates/izwi-model-catalog/src/catalog/prefix_reuse.rs`).

Chat KV is F16 on Metal and CUDA and F32 on the CPU (`lfm2_chat_kv_dtype` in
`lfm2/chat.rs`); activations and rings are F32. LFM3 chat serves 32,768 tokens
by default on CPU and Metal and the checkpoint's 262,144 on CUDA
(`lfm3_chat_context_limit`); the context sizes the RoPE tables and the KV
reservation at load. Chat prefill runs in resumable spans. `catalog/batch_width.rs`
gives LFM3 chat the LFM2 family's decode width, up to eight rows on CPU and
Metal; the evidence that table cites was measured on LFM2.5.

### Chat template

`lfm3/prompt.rs` renders LFM3-1.7B's `chat_template.jinja` in Rust. The
rendered string is tokenized once, as `apply_chat_template` does.

- **Thinking.** The generation prompt opens `<think>` by default, and
  `enable_thinking=false` closes an empty `<think></think>`. An assistant turn
  at or before the last user turn keeps only the text after its last
  `</think>`; `preserve_thinking` keeps every turn whole.
- **Tools.** Definitions are appended to the system turn as `<tools>`, one
  function object per line, serialized as Python's `json.dumps` writes it
  (`izwi_model_catalog::chat::tools::tool_to_json`). The model writes calls as
  `<tool_call><function=name><parameter=key>value</parameter></function></tool_call>`,
  which the gateway parses as `ToolCallFormat::QwenXml`
  (`crates/izwi-gateway/src/api/openai/tool_calls.rs`). Past calls reach the
  prompt already rendered by the gateway. A tool turn is wrapped in
  `<tool_response>` unless it already holds one or is empty.
- **Deferred tools.** A tool with `"defer_loading": true` (on the tool or its
  `function`) stays out of `<tools>` until a tool result names it.
  `crates/izwi-gateway/src/api/openai/tool_response.rs` renders such a result
  as the template's `<tool_response>` block, with the named definitions in
  `<additional_tools>`. A result names tools with `tool_reference` content
  items (resolved against the request's `tools`, the last match winning; an
  unknown name is a 400), with `{"result": {"tools": [...]}}` items, or with an
  `additional_tools` key on the tool message (definitions, or a flag when the
  content holds them). For LFM3 chat, consecutive results with the same
  `tool_call_id` merge into one block. LFM3-Audio's dialect writes other items
  as JSON, refuses `tool_reference` items and does not merge. Other models
  refuse both constructs (`reject_deferred_tool_results`). The chat
  completions handler and the Responses handler
  (`crates/izwi-server/src/api/openai/responses/handlers.rs`) both render
  through it, and the prompt renderer passes the block through unchanged.

Sampling defaults come from LFM3-1.7B's card and `generation_config.json`:
temperature 0.1, top-k 50, repetition penalty 1.1
(`crates/izwi-model-catalog/src/chat/sampling.rs`). Request values win.

### LFM3-Audio

`crates/izwi-core/src/models/architectures/lfm3_audio/` holds the model. The
language model GGUF is an LFM3 trunk; the mmproj GGUF holds the other stages.

1. **Mel features** (`preprocessor.rs`, host): 16 kHz input, a 25 ms Hann
   window every 10 ms, 128 mel filters, no per-clip normalization, so features
   stream.
2. **Audio encoder** (`encoder.rs`): three stride-2 3x3 convolutions turn 128
   mels and 8 frames into one vector, then a 1x1 projects it to 768: one encoder
   token per 80 ms. A 17-layer LFM2 model (10 width-3 ShortConvs, 7 attention
   layers) runs over the tokens with causal attention over the last 32, and a
   projector (LayerNorm, Linear, GELU, Linear) maps them into the language
   model. The encoder is causal with bounded reach, so it streams.
3. **Language model** (`model.rs`): text mode samples tokens;
   `<|audio_start|>` switches to speech and `<|mixed_start|>` to interleaved
   blocks of 6 text tokens and 12 speech frames. Speech mode samples one frame
   per step until a frame holds the end-of-audio code. Every slot, a token or a
   frame (fed back as the sum of its codes' embeddings), is one language-model
   step. Text decodes greedily by default.
4. **MaskGIT depth decoder** (`depth.rs`): each of the 12 codebooks (2,016 FSQ
   codes plus end-of-audio) gets a slot holding the frame's projected hidden
   state plus its known code's embedding or the mask embedding. A 3-layer
   bidirectional LFM3 trunk (per-head q/k RMSNorm, NEOX RoPE over slots 0..11,
   sigmoid output gates, no sandwich norms) runs over the slots, and each
   codebook scores its codes through its own Q8_0 embedding table. Codebooks
   are revealed in a random order, four per pass; a pass scores only the four
   it reveals. Speech sampling follows the model card: temperature 1, top-k
   128, repetition penalty 1.1 over each codebook's last 4,096 frames. A frame
   that draws end-of-audio in any codebook ends speech. On Metal with fused
   kernels on, a pass uses fused RMSNorm, q/k RMSNorm, RoPE and SiLU-gate
   kernels (`depth_fusion_on`); CPU and CUDA run the op-by-op pass.
5. **LiquidCodec decoder** (`codec.rs`): each code's FSQ digits give a 48-dim
   latent per frame, projected to 512 wide and upsampled six times to 75 rows a
   second; an 8-layer LFM3 trunk (width-7 ShortConvs, gated attention over the
   last 30 rows, sandwich norms) and a linear layer give log magnitudes and
   phases for an ISTFT (n_fft 1280, hop 320) on the host. Output is 24 kHz,
   1,920 samples per frame (12.5 frames a second). The decoder is causal with
   bounded reach, so a stream decodes exactly what the whole utterance does. A
   stream's first codec call emits after two frames (160 ms), later ones every
   six (480 ms).

Task prompts (`prompts.rs`) follow the checkpoint template: `Perform ASR in
<language>.` (18 languages) or `Perform ASR.`; `Perform TTS in <locale> using a
<gender> voice.` (19 locales, female or male) prefilled with `<|audio_start|>`;
`Perform live ASR.` (English only); and voice replies under `Respond with
interleaved text and audio.`, or `Respond with function calls, and interleaved
text and audio.` and `List of tools:` when tools are offered.

LFM3-Audio serves 32,768 tokens of context, the model card's figure (the GGUF
says 128,000). A row leases only the context its workload can use
(`lfm3_audio/physical.rs`): 2,048 tokens for a transcription clip of at most
30 s, 4,096 for a TTS segment of at most 2,048 frames, 16,384 for a
conversation, and 2,048 for a LiveASR segment. Paged KV is F16 on Metal and
CUDA and F32 on the CPU. The state domain ids are LFM2.5-Audio's, so the
engine's audio-chat leases serve both families. The depth decoder is
stateless, and the encoder and codec streams are bounded host state.

## Engine paths

### Retained rows: ASR, TTS and voice-chat replies

`lfm3_audio/retained.rs` and `engine/executor/handler_lfm3_audio_rows.rs`. Each
request is a retained engine row: its paged KV and ShortConv ring stay in the
managed arena between quanta, and its generate loop's host state in an
`Lfm3AudioRow`. Each quantum opens a checkpoint, so a failed or cancelled
quantum rolls back to the frame boundary it started at. Prompts prefill one row
at a time; decode steps every row together, with the language model in one
batched forward and the depth decoder sampling one frame per row. CPU and Metal
run two rows at a time (the portable width) unless `IZWI_MAX_BATCH_SIZE` sets
one; CUDA follows the engine's tensor width.

- **Speech** rows stream audio as the codec decodes it; a final-only request
  gets it when its row finishes. A request without its own budget may speak
  about 1.5 times a slow reading of its text plus 2 s, between 4 s and 2,048
  frames (`lfm3_audio_tts_auto_max_frames`).
- **Transcription** of one clip (at most 30 s) is a retained row whose prompt
  holds the clip's encoder embeddings. Longer audio takes the long-form route,
  which transcribes it in clips of at most 30 s and leases the language model's
  pages and ring for one atomic invocation
  (`prepare_lfm3_audio_asr_shape_for_binding` in
  `runtime/service/asr_preparation.rs`). The diarization pipeline leases the
  same invocation workspace.
- **Voice-chat replies** stream text and speech in the order they decode. The
  reply row hears the spoken turn but does not transcribe it; the turn
  transcribes as its own row beside the reply (`lfm3_audio_turn_transcript` in
  `runtime/speech_to_speech.rs`).

### Chat completions with audio input

A chat request that names LFM3-Audio is built by `runtime/lfm3_audio_chat.rs`
as a reply row with `SpeechReplyTurn::Chat` (`engine/request.rs`): the
conversation, with an `<|audio|>` placeholder for each clip, answered in text
only (`Lfm3AudioReplyOutput::Text`; the reply never starts speech). The gateway
reads `input_audio` parts, and `audio_url` or `audio` parts holding a `data:`
URL (`crates/izwi-gateway/src/app/chat_content.rs`). Only user messages carry
audio, at most 8 clips per request and 120 s per clip; images and video are
refused. The request's sampling applies over LFM3-Audio's greedy text defaults,
`preserve_thinking` is honoured, and tools render as described above. A chat
invocation that carries a gateway's token ceiling (`max_tokens_ceiling`, which
serving workers pass) is refused: LFM3-Audio answers chat only on a local
server.

### Spoken tool rounds

`RuntimeService::speech_to_speech_round_streaming_bytes`
(`runtime/speech_to_speech.rs`) runs one model round of a spoken turn with the
tools it may call; only LFM3-Audio accepts tools there. A continuation round
passes the round so far in its messages (the spoken turn as an `<|audio|>` user
turn, then the calls and their results) and `transcribe: false`, since the
first round already transcribed the turn.

On the server, a unified voice session on LFM3-Audio offers the built-in tools
(`NoopTool` and `TimeTool`, shared with agent turns) unless `builtin_tools` is
`false`, plus up to 64 client function tools from `input_stream_start`. The
client answers its calls with `function_call_output` within
`function_call_timeout_ms` (default 10 s, 1 to 60 s). A turn runs at most three
model rounds and eight calls (`MAX_TOOL_ROUNDS`, `MAX_TURN_FUNCTION_CALLS` in
`crates/izwi-server/src/app/voice_tools.rs`); each round holds its own
admission permit, so none is held while a tool runs
(`stream_unified_s2s_to_socket` in `app/voice_realtime.rs`).

### LiveASR realtime sessions

`lfm3_audio/live_asr.rs` follows Liquid's `transcribe_live` and transcribes
English only. Mel features reach the encoder in chunks of 96 frames (0.96 s).
Each chunk's embeddings feed the language model, which writes the chunk's text
greedily until `<|text_end|>`: at most 48 tokens, 96 for the last chunk. Before
the last chunk, speech, interleaved blocks and the end of the turn are masked.

One `Lfm3LiveAsr` is one segment, a text context of at most 30 s of audio. A
longer stream continues in segments of 29 s; each new one first hears the
previous one's last 2.88 s again, `Lfm3LiveTranscript` drops the words that
overlap repeats, and the old segment's last block is not kept.

Sessions run on the engine realtime path (`EngineRealtimeModel::Lfm3Audio` in
`runtime/asr/realtime_session.rs`), bound to a four-stage graph checked by
`authenticate_lfm3_audio_realtime_execution_binding` (`lfm3_audio/mod.rs`):
static-batched preparation (mel features and the streaming encoder, ragged
across sessions), one session's chunk prefill at a time, decode batched across
sessions, and a scalar completion. A session retains the language model's
pages; the mel and encoder streams' tails, queued chunk embeddings, the
ShortConv ring and the transcript are budgeted outside the arena's pages. A
push's cost follows the mel frames it adds (`lfm3_live_asr_operation_bounds`).

## Real-weight tests

The real-weight tests need the published GGUFs and report themselves as not
run, or skip, when the files are missing; a pass without the weights proves
nothing. Check the output for `not run` or `skipping`.

**Runtime routes** (`crates/izwi-core/src/runtime/lfm3_audio_tests.rs`): ASR,
TTS and voice chat, LiveASR on engine sessions, chat about audio clips with
tool calls, and a four-language check. They run when
`IZWI_LFM3_AUDIO_MODELS_DIR` names the models directory, on the CPU unless
`IZWI_LFM3_AUDIO_BACKEND=metal` (or `cuda`) selects a device.
`IZWI_LFM3_AUDIO_BATCH=4` runs speech and transcription rows four wide. The
four-language check also loads `Qwen3-TTS-12Hz-1.7B-CustomVoice` and
`Whisper-Large-v3-Turbo` from the same directory.

```bash
IZWI_LFM3_AUDIO_MODELS_DIR=~/Library/Application\ Support/izwi/models cargo test -p izwi-core --features metal --release --lib lfm3_audio_tests -- --nocapture
```

**Model checks** (`lfm3_audio/model_tests.rs`, ignored): ASR of
`data/fox.wav`, TTS transcribed back for both voices, streamed codec output
equal to one offline decode, seeded reproducibility, batch invariance (decode
rows alone and four wide, and speech and voice-chat reply rows joining a
running batch), LiveASR in 29 s segments, component timing, and the fused
depth pass against the op-by-op one on Metal. They read
`IZWI_LFM25_MODELS_DIR`, else the default models directory, and skip when
`LFM3-Audio-2B-GGUF` is missing.

```bash
cargo test -p izwi-core --features metal --release --lib lfm3_audio::model::tests -- --ignored --nocapture
```

**Chat parity** (`lfm2/chat_lfm3_reference_tests.rs`, ignored): prompt ids
against the checkpoint template (thinking on and off, preserved and stripped
reasoning, tools, tool round trips), one-shot against resumable prefill cut at
255, 256 and 257 tokens across the sliding window, replay after preemption,
and teacher-forced scalar and two-row decode against the top-8 logits of
`crates/izwi-core/tests/fixtures/lfm3_chat_reference.json`, an independent F32
forward pass written by `scripts/fixtures/generate_lfm3_reference.py`. They
read `IZWI_LFM25_MODELS_DIR` and skip when `LFM3-1.7B-GGUF` is missing.

```bash
cargo test -p izwi-core --features metal --lib lfm2::chat::lfm3_reference_tests -- --ignored --nocapture
```

**Trunk** (`lfm3/tests.rs`, no weights): a tiny `lfm3` GGUF with every LFM3
feature against an independent F64 forward pass written from HF
`modeling_lfm3.py`, and any prefill split against one shot. These run in the
normal suite; `--features metal` adds the Metal legs.

**Server tool rounds**: `unified_turns_call_builtin_and_client_tools_if_env_set`
in `crates/izwi-server/src/app/voice_realtime.rs` makes a built-in call, then a
client call answered over the socket, each followed by a spoken answer. It
reads `IZWI_LFM3_AUDIO_MODELS_DIR`; on Metal set `IZWI_LFM3_AUDIO_BACKEND=metal`
and build with `--features metal`.

```bash
IZWI_LFM3_AUDIO_MODELS_DIR=~/Library/Application\ Support/izwi/models IZWI_LFM3_AUDIO_BACKEND=metal \
  cargo test -p izwi-server --features metal --release --lib unified_turns_call_builtin_and_client_tools_if_env_set -- --nocapture
```

The `IZWI_LFM3_AUDIO_*` and `IZWI_LFM25_MODELS_DIR` variables are read only by
these tests and are not in the engine's environment registry.

## Environment

LFM3's own knob is in [`env/core.md`](env/core.md):

- `IZWI_LFM3_CONTEXT_TOKENS` (read at model load) sets LFM3 chat's context, up
  to the checkpoint's 262,144 tokens. It defaults to 32,768 on CPU and Metal
  and to the whole context on CUDA, where the arena is fitted to device memory.

Shared knobs that reach LFM3, also in [`env/core.md`](env/core.md):

- `IZWI_MAX_BATCH_SIZE` sets the tensor batch width, and with it the LFM3-Audio
  rows' width on CPU and Metal.
- `IZWI_ENABLE_PREFIX_CACHING` overrides the catalog's prefix reuse.
- `IZWI_FUSED_KERNELS=0` sends the Metal sandwich-norm, output-gate and depth
  passes, and the accelerator ShortConv ring, down the op-by-op path.
- `IZWI_LFM2_DIAGNOSTICS=1` adds the LFM2 backbone's activation checks, which
  also turns off the Metal sandwich-norm and output-gate fusion.

## Known gaps

- **CUDA** builds for both models and is compile-checked only: neither has run
  on an NVIDIA GPU. See [LFM CUDA](lfm-cuda.md#lfm3) for the tests to run on
  one.
- **Prefix reuse** is not enabled for LFM3-Audio on any backend: no audio-chat
  reuse evidence has been collected. For LFM3 chat it is off on CUDA.
- **Benchmarks.** `LFM3-1.7B-GGUF` and `LFM3-Audio-2B-GGUF` are in the CPU,
  Metal and CUDA chat and audio-concurrency manifests, `lfm-cuda.toml` and the
  CUDA family manifests under `benchmarks/manifests/`, but no run of those
  cells has been retained.
- **Voxtral realtime** sessions run on the same engine realtime path. LiveASR
  was the first real-weight stream through it, and the gaps it exposed on the
  shared path are fixed, but no real-weight Voxtral stream has run through it.
- **French ASR** on imperfect speech drifts into English: where Qwen3-TTS
  mispronounces a word, LFM3-Audio's transcript of its French has a word error
  rate of 0.50 on Metal, while it hears its own clean French exactly. See the
  LFM3 Audio row of the [support matrix](../user/support-matrix.md).
- LiveASR transcribes English only.
