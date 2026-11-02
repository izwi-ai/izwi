# Fish Audio Drama 3

izwi serves Fish Audio's Drama 3 (`fishaudio/drama-3`) as a text-to-speech
model: a 35.9B-parameter mixture of experts with about 3B parameters active
per frame. It speaks in a voice its sampling seed sets, or clones a reference
voice given its transcript; it takes inline performance tags; and it speaks
long text as a chunk session that carries the voice from chunk to chunk. It
is its own model family (`fish_drama3`), separate from `fish_s2`
(`FishAudio-S2-Pro`, a dense dual-AR model with another tower, prompt and
checkpoint) and from the Qwen3.6 MoE chat family; it imports neither.

It runs on CPU, Metal and CUDA (compute capability 8.0 or newer). The suites
that need a GPU or the published weights, and how to run them, are in
[Tests](#tests).

User-facing behaviour is documented in the
[support matrix](../user/support-matrix.md), the
[models guide](../user/models/index.md), the
[text-to-speech guide](../user/features/text-to-speech.md), the
[API reference](../user/api.md) and the
[`izwi tts` reference](../user/cli/tts.md).

| Variant | Files | Family |
| --- | --- | --- |
| `FishAudio-Drama-3` | 24 files, 41.5 GB | `FishDrama3Tts` |

## Pinned checkpoint

| Repository | Revision | Licence |
| --- | --- | --- |
| `fishaudio/drama-3` | `78e5ab79e49aa9a25abe8799eff2c4d73a1c3216` | Fish Audio Community License v1.0 |

`fish_drama3/artifacts.rs` holds the repository, the revision and the manifest
(`FISH_DRAMA3_FILES`): every file of the revision except `.gitattributes`, 24
files and 41,502,569,692 bytes, each with its size and SHA-256. The weights
were re-uploaded on 2026-10-29 (8 shards became 15), so the family follows
the one revision, never `main`. The downloader
(`crates/izwi-core/src/artifacts/downloader.rs`) fetches the manifest's files,
then `verify_full` hashes every one once, before the model is marked
downloaded. Every load runs `verify_identity` instead: each file present at
its exact size and `config.json` at its pinned hash, a few `stat` calls and
5.5 kB of hashing. `codec.pth` is byte-identical to Fish S2 Pro's.
`chat_template.jinja` is the stock Qwen3.5 template; it ships with the
repository and is never rendered.

The licence is shown, never enforced. `ModelVariant::license_label` and
`license_notice` (`crates/izwi-model-catalog/src/catalog/metadata.rs`) feed
the admin model API's `license` and `license_notice`; `izwi pull` prints them
before it downloads; the server logs them when a download starts and when the
model loads. Nothing is refused because of them. The licence caps commercial
use at annual revenue below USD 500,000 (§III(a)) and grants no licence to
offer a Hosted Service (§III(c)), which an izwi server or Docker deployment
open to third parties is. [`NOTICE`](../../NOTICE) carries the attribution the
licence asks for.

## Architecture

From text to audio, one frame (2,048 samples at 44.1 kHz, 21.5 frames a
second) at a time:

1. **Trunk** (`trunk/`): Qwen3.5-35B-A3B's language model. 40 pre-norm layers,
   30 Gated DeltaNet and 10 gated full attention (every fourth layer). Every
   layer is a mixture of experts: 256 routed experts, top 8, FP8 with F32
   128×128 block scales, plus a sigmoid-gated shared expert. Its main token is
   the frame's semantic code (`<|semantic:N|>`), or `<|im_end|>` to stop.
2. **Codebook tower** (`tower.rs`): 13 dense layers at the trunk's width, 4
   shared and then one per residual codebook, each followed by the head that
   samples that codebook. Every tower layer keeps its own KV over the whole
   sequence, text rows included.
3. **Codec** (`codec/`): the Fish S2 DAC, unchanged, as the family's own copy.
   One semantic codebook of 4,096 and 9 residual codebooks of 1,024, fully
   causal, F32 on every backend.

Every position is a row `[main, c0, …, c9]`: text rows are `[token, 0, …, 0]`,
audio rows `[<|semantic:c0|>, c0, …, c9]`. A frame goes back into the trunk as
`(E[main] + CB[4096 + c1]) / √2` (`embed.rs`). The main head is compact
(`head.rs`): the load keeps only the 4,097 candidate rows, `<|im_end|>` and
the semantic band.

`config.rs` checks every switch the family implements one way and fails the
load on any other value; the checkpoint census (`checkpoint.rs`, 62,245
tensors) fails it on a missing, extra or reshaped tensor before a weight is
read. The published config and census are in
`crates/izwi-core/tests/fixtures/fish_drama3/`.

### Prompt

`prompt.rs` builds fish-speech's `reference-bracket-v1` prompt: a ChatML
system block that holds each reference's transcript and audio rows, then the
history turns (earlier chunks with their audio), then the user text and
`<|im_start|>assistant\n<|voice|>`, where generation starts. Each literal is
tokenized on its own (`tokenizer.rs`), as upstream does. Text without a
leading speaker tag gets `<|speaker:0|>`. A prompt holds at most 8,192 rows,
and a request at most 12,288 positions. Text or a transcript that holds a
control token (`<|im_start|>`, `<|im_end|>`, `<|voice|>`, `<|text|>`,
`<|interleave|>`, `<|semantic:N|>`) or a speaker tag outside
`<|speaker:0|>`..`<|speaker:15|>` is refused, by the server
(`validate_speech_control_tokens` in `crates/izwi-server/src/api/tts_policy.rs`,
for Fish S2 Pro too) and by the prompt builder.

Reference clips are mixed to mono, resampled and encoded once per load, then
cached (`reference_cache.rs`); a clip may last at most 60 s.

### Sampling and stop

`sampling.rs` is the fish-speech sampler: temperature, then top-k and top-p
over the tempered distribution, drawn by an exponential race whose noise is
indexed by candidate, so a recorded rollout replays exactly. The main token
draws twice every frame, at the request's policy and at the fixed
repetition-aware policy (temperature 1.0, top-k 30, top-p 0.95); the second
replaces the first when the first is a semantic token among the last 16 main
ids, prompt rows included. The 9 codebook heads draw once each. Defaults are
temperature 1.0 and top-p 0.9, the main token uncapped and the heads at top-k
30; a request's `top_k` caps both (the server maps `top_k: 0` to no cap).
There is no repetition penalty. Eos is masked until the generation has a
frame, as for Fish S2.

`model/budget.rs` holds the stop contract. A chunk's budget is the request's
`max_tokens` (at most 4,096 frames, the codec stream's capacity) or an
automatic one: Fish S2's CJK-aware speaking estimate times 2.5, at most 1,024
frames. A budget of N allows N frames, and the draw after the N-th must be
eos; otherwise the request fails with `frame_limit`, never a cut-off take.

### Module layout

`crates/izwi-core/src/models/architectures/fish_drama3/`:

| Module | Holds |
| --- | --- |
| `mod.rs` | The family's exports and the isolation test that keeps it from importing other families |
| `artifacts.rs` | The pinned manifest, `verify_identity` and `verify_full` |
| `config.rs` | Config parsing; fails closed on any switch the family does not implement |
| `checkpoint.rs` | The shard index, memory maps, the tensor census, FP8 experts and row-slice reads |
| `residency.rs`, `linear.rs` | Each backend's weight forms and the linear layers in them |
| `precision.rs` | The dtype policy and the CPU's Q8_0 self-check |
| `tokenizer.rs` | The published tokenizer, checked against the config; one literal at a time |
| `prompt.rs` | The `reference-bracket-v1` prompt, speaker tags and control-token refusal |
| `embed.rs`, `head.rs` | The row embedding and the compact main head |
| `trunk/` | The CPU reference trunk: gated attention, Gated DeltaNet, MoE and the DeltaNet state |
| `tower.rs` | The CPU reference codebook tower |
| `sampling.rs` | The sampler and repetition-aware sampling |
| `codec/` | The family's copy of the Fish S2 DAC, its streaming decoder and workspace |
| `reference_cache.rs` | Reference clips to codes, cached per load |
| `chunking.rs` | fish-speech's long-text chunker and the history policy of its chunk loop |
| `model/` | The model loop: prefill spans, decode quanta, the codec stream (`audio.rs`), budgets (`budget.rs`) and the chunk session (`session.rs`) |
| `device/` | The trunk, tower and resident sampling shared by Metal and CUDA, and the load-time table checks |
| `metal/`, `cuda/` | What only that backend runs: its residency, load checks and device tests |
| `physical.rs` | The engine state contract and the invocation workspace |
| `synthetic.rs`, `test_support.rs` | The synthetic checkpoint, and its lifecycle form for other crates' tests |
| `reference_tests.rs` | The ignored parity suites; see [Tests](#tests) |

## Residency and precision

| Part | CPU | Metal | CUDA (SM80+) |
| --- | --- | --- | --- |
| Routed experts | Q8_0 (FP8 → F32 → Q8_0 at load) | Raw E4M3 + F32 block scales, fused MoE kernels | Raw E4M3 + F32 block scales, fused MoE kernels |
| Dense trunk projections, shared expert | Q8_0 | F16; q/k/v and qkv+z packed | BF16; q/k/v and qkv+z packed |
| Router, gates, norms, conv, `A_log`, `dt_bias` | F32 | F32 | F32 |
| Tower projections and codebook heads | Q8_0 | F16 | BF16 |
| Compact main head | F32 | F16 | BF16 |
| Embedding and code tables | BF16 host tables | F16 device tables | BF16 device tables |
| Residual stream | F32 | F32 | BF16 |
| Paged KV (both domains) | F32 | F16 (F32 after a breached probe) | BF16 |
| Sampling | Host | Device, one readback a frame | Device, one readback a frame |
| Codec | F32 | F32 | F32 |
| Resident total | about 37 GiB | about 37 GiB | about 37 GiB |

Each load checks what it is about to run, and each check has a counter in the
backend's report:

- **CPU** (`precision.rs`): every Q8_0 projection class against F32; a class
  over tolerance stays F32.
- **Metal** (`metal/self_check.rs`): the host RoPE tables (NeoX half-split for
  the trunk, interleaved pairs for the tower), the FP8 GEMV against the
  original tensors, and an F16 range probe over every layer on xorshift input.
  A layer whose activations leave F16's range runs in F32, and a breached MoE
  layer runs the host expert loop.
- **CUDA** (`cuda/self_check.rs`): the same RoPE and FP8 checks, which refuse
  the load on failure. BF16 has F32's exponent range, so there is no range
  probe. The kernel checks gate a kernel, never the load: a failed
  row-invariant GEMV check leaves steps on one GEMV per row, a failed chunked
  DeltaNet check leaves spans on the per-row kernels, and a failed tensor-core
  check leaves the grouped MoE on its warp kernels. A GPU older than SM80 is
  refused before any weight is read.

`IZWI_FISH_DRAMA3_MOE_BACKEND=loop` runs the routed experts as a host-routed
loop of one FP8 GEMV pair per expert on Metal and CUDA, for diagnosis.

### Fused forwards

On Metal and CUDA the trunk and tower run through fused kernels shared with
other families (`crates/izwi-core/src/kernels/`):

- `residual_norm`: each residual add with the norm after it in one launch,
  the sum rounded once to the stream's dtype (BF16 on CUDA, F32 on Metal)
  and the norm read through the next layer's activation dtype.
- `qk_norm_rope`: q, k and v read straight from the packed projection, each
  head's q/k norm and RoPE in one launch from cos/sin tables built on the
  host at load (one set per stack and device, shared by every layer) and
  one positions upload per stack forward.
- `swiglu`: the shared expert's and the tower's gate and up projections are
  packed into one `[gate | up]` projection at load; one GEMV and one launch
  replace two GEMVs, two casts, SiLU, the product and a cast.
- `race_sampler`: each device draw is one launch instead of a bounded sort
  and about 40 Candle ops. A row's main draw uploads both its exponentials
  (the request's draw and RAS's) at once; a codebook head races every row
  that shares a policy in one launch, over exponentials uploaded head-major.
  A kept prefix longer than the 512 sorted ranks (an uncapped main draw over
  a flat distribution) is bounded by mass on the device, so no draw falls
  back to the host.

A fused trunk step runs as pieces: stretches of DeltaNet and MoE layers
between attention mixers (each opens with the previous attention layer's
MoE half), pure functions of the residual, the normed hidden and each row's
conv history and recurrent state, and the attention mixers with their post
norms, eager. On CUDA each stretch replays a CUDA graph
(`kernels::cuda::stretch_graphs`), captured per row count up to 4 rows and
checked bit for bit against the eager stretch on its capture step and the
next; any failure turns the graphs off for the process, and a host-routed
MoE layer keeps them off. A replay copies the DeltaNet states in and out of
the graph's buffers. The trunk's `decode_graphs` diagnostics count warmups,
captures, verified steps and replays. A device-drawn tower step runs as
pieces too: each layer's attention with its FFN norm, eager, and the
stretches between them (a layer's FFN and final norm, its head's race, the
next layer's injection and pre-attention norms). The race takes each row's
heads policy as launch arguments, so the tower's graphs are kept per row
count and policy set (at most 8 shapes); the tower's `decode_graphs`
diagnostics report them. `IZWI_FISH_DRAMA3_SEGMENT_GRAPHS=0` or
`IZWI_CUDA_DECODE_GRAPHS=off` keeps every stretch of both eager. Prefill
spans run the fused layers whole. The graphs have been compiled and
type-checked but not yet run on a GPU.

On the fused path a frame reads back once. The main draws, RAS (the
retry replaces a semantic first draw already in the row's 16-token window,
uploaded per frame), the tower and the trunk all run on the device for every
row under its frame limit, the next trunk row is embedded from the device's
token and codebook-1 code, and every pick, chosen candidate, code and
non-finite count then comes back in one transfer. The tower and trunk
appends and DeltaNet states stay prepared until then: a row that drew eos
has its step aborted (one wasted step), and the host replays RAS to check
the device's choice. Frames, noise and samplers are those of the
two-readback path bit for bit.

The load compares the kernels on the device with their CPU forms at the
model's widths and dtype mixes (relative L2 at most 1e-3; the race must
pick exactly as its CPU form does), and
`decode_path` in the diagnostics says which path runs and why.
`IZWI_FISH_DRAMA3_FUSED_DECODE=0` runs the eager per-layer chain. The Metal
range probe and the trunk's traced test forwards always run eagerly.

## Engine path

### Retained rows

Every request is a retained engine row on Fish S2's route shape. The adapter
is `builtin.fish_drama3_tts.physical_sequence` (`FishDrama3TtsExecutionAdapter`
in `crates/izwi-core/src/runtime/adapters/loaded/tts.rs`), with rows for both the
`Tts` and `StreamingTts` capabilities. The executor is
`crates/izwi-core/src/engine/executor/handler_fish_drama3.rs`, with its own
pending coordinator (`fish_drama3_pending.rs`). The stages:

| Stage | Work |
| --- | --- |
| `tts.prepare.fish_drama3` | Decode, resample and codec-encode the reference (through the cache); tokenize; build the rows; plan the frame budget and KV positions |
| `tts.prefill.fish_drama3` | One cell of the family's prefill grid per quantum (256 rows on CPU and Metal, 1,024 on CUDA): the trunk over the rows, then the tower teacher-forced over them. A request's cells are never batched with another's |
| `tts.decode.fish_drama3` | One frame per row per quantum, rows of many users together, each row alone through every kernel |
| `tts.audio_decode.fish_drama3` | Committed frames through the codec in batched quanta: the first after 4 frames, then 16 at a time, within each stream's output credit |
| `tts.codec.fish_drama3.scalar` | Flush; the PCM is frames × 2,048 samples |

The backend report's fast-path counters time each decode step's three parts
on the host and count the step's host-to-device uploads
(`step_uploads_total`), beside the resident sampler's frames and readbacks.
On the one-readback frame `step_main_draw_ms_total` is the main head, its
draws and RAS, `step_tower_ms_total` the tower and its heads, and
`step_trunk_ms_total` the embedding, the trunk, the frame's readback (where
the host waits for the GPU) and the commits. With the fused path off,
`step_main_draw_ms_total` ends with readback 1 and `step_tower_ms_total`
with readback 2.

Every model quantum runs inside the family's quantum checkpoint, and the
DeltaNet tensors move through the engine's transactional tensor arena, so an
aborted quantum restores the request bit for bit. Cancellation and the
deadline are checked between prefill cells, decode quanta and codec quanta.

Rows decoded together: one on the CPU, one on Metal (two with
`IZWI_FISH_DRAMA3_NATIVE_BATCHING=true`) and eight on CUDA
(`crates/izwi-core/src/catalog/batch_width.rs`). A row's output does not
depend on the rows beside it: steps run one GEMV per row on Metal and the
row-invariant multi-row GEMV (`kernels::cuda::row_gemv`) on CUDA, and CUDA's
split decode partitions each row by its own length
(`SplitDecodePartitioning::RowLength`).

### Row state

`physical.rs` declares one transactional consistency group per row. Every
domain is retained and clocked by decoder rows, and none shares prefixes.

| Domain | Kind | Holds |
| --- | --- | --- |
| 1 | Paged attention, raw keys | KV for the 10 trunk attention layers, 2 KV heads × 256 |
| 2 | Paged attention, raw keys | KV for the 13 tower layers, 2 KV heads × 256 |
| 3 | Tensor | DeltaNet recurrent state, 30 × [32, 128, 128] F32 |
| 4 | Tensor | DeltaNet convolution state, 30 × [3, 8192] F32 |

The tensor state is about 62.8 MiB a row. The tower writes one row behind the
trunk (after prefill `tower_len == trunk_len - 1`, asserted at every commit),
which is why the two stacks are separate paged domains. Keys are stored after
RoPE. The state is allocated through the managed arena at load
(`allocate_fish_drama3_tts_state` in `runtime/lifecycle/load/state.rs`) and
released when the request ends.

### Admission

- **Load** (`crates/izwi-core/src/runtime/lifecycle/fish_drama3_memory.rs`):
  prices the backend's resident weights, the load peak, one row and the
  reference cache from the pinned census, opening nothing but `config.json`.
  A load the budget cannot hold fails before a weight is read: "Fish Audio
  Drama 3 needs about … GiB of … memory (… GiB of weights, a … GiB load peak,
  … GiB for one request); this device has … GiB".
- **Rows** (`crates/izwi-core/src/runtime/fish_drama3_capacity.rs`): sealed
  into the adapter at load through Fish S2's interface. One row is its tensor
  state, KV for the worst admitted request (8,192 prompt rows and 4,096
  frames), the span and decode workspaces, two copies of the codec stream
  history and the codec quantum's workspace and output credit; the rows are
  what the headroom holds, capped by the width above. Live admission still
  prices every allocation.

## Long text: the chunk session

The catalog marks the family with `continues_speech_across_chunks`, so the
server plans its speech itself (`plans_own_speech_segments`) as a chunk
session (`SpeechTextPlan::chunk_session` in
`crates/izwi-server/src/api/tts_long_form.rs`):

- `chunking.rs`, a port of fish-speech's chunker, cuts the text into chunks of
  about 300 UTF-8 bytes: sentence ends first, then clause marks, then spaces.
  Decimals, `<|…|>` tokens, `<tag>…</tag>` and `[…]` stay whole, a speaker's
  lines keep their tag, and a chunk holds at most three speakers.
- Each chunk is one engine request with its own frame budget, sampled from
  `seed + i`. The request carries the chunk history (`chunk_history`, internal
  only): the first chunk that made audio, pinned, and the latest one, each
  with its generated codes. The engine hands a finished chunk's codes back to
  the server (`generated_codes`), which adds it to the history
  (`SpeechChunkHistory`). When a prompt would be too long, the latest chunk
  goes first, then the pinned one; references always stay.
- A chunk whose prompt is too long before any audio
  (`ContextLengthExceeded`) is cut in two by the chunker, tags kept whole,
  and retried.
- Within a chunk, audio streams as it is generated; chunks follow one another
  with no added pause. A stream's final `audio.done` reports
  `planned_segment_count` and `token_unit: "semantic_frames"`.
- First-party text-to-speech records journal their audio per chunk, so an
  unfinished record plays from its start
  (`GET /v1/text-to-speech/{record_id}/events`). A durable job's checkpoint
  (`crates/izwi-server/src/api/speech_history/durable.rs`) holds the history
  (codes as little-endian u16, base64; about 27 KB for a 1,024-frame chunk),
  so a job that yields or recovers between chunks keeps its voice. The
  checkpoint records the model's artifact fingerprint, and a plan refuses to
  continue once it changes.
- `IZWI_FISH_DRAMA3_LONG_FORM_ENABLED=false` (server) refuses text that needs
  more than one chunk with `400`.

## Dialogue

A request may voice up to 16 speakers from up to 8 references
(`reference_cache::MAX_REFERENCES`, `prompt::MAX_SPEAKER_ID`). The catalog
says so (`supports_speaker_dialogue`, `max_speakers`, `max_references`), and
the server's policy reads those fields, not the model id.

- **Request shape** (`crates/izwi-server/src/api/speech_dialogue.rs`):
  `references: [{speaker, audio | saved_voice_id, text}]` and
  `script: [{speaker, text}]` on `/v1/audio/speech` and the first-party
  text-to-speech and voice-cloning record routes. Studio stays one voice per
  paragraph. `reference_audio` / `reference_text` / `saved_voice_id` mean one
  reference for speaker 0; giving both forms is a `400`. A script replaces
  `input` (`text`): the server renders it with `prompt::render_script`
  (`<|speaker:N|> text` lines joined by `"\n"`), and a record stores that
  rendered text as its `input_text`. Raw tags in `input` work the same way.
- **Speakers.** A reference's speaker is its `speaker`, else the tag leading
  its transcript, else 0. Its transcript is tagged
  `<|speaker:N|> ` unless a tag already leads it
  (`prompt::speaker_reference_transcript`, upstream's
  `tag_reference_text`); a leading tag for another speaker is refused, where
  upstream keeps it. Several references may voice one speaker, as upstream's
  prompt allows (two clips of one voice share a tag). A speaker the text names
  without a reference speaks in a voice the seed sets beside referenced ones,
  as upstream allows; a dialogue model without a seeded voice would need a
  reference per speaker (`check_unreferenced_speakers`).
- **Checks before any load** (`check_dialogue_request`,
  `check_dialogue_references`): dialogue fields for a model without
  `supports_speaker_dialogue`, both reference forms, `script` with `input`,
  more than `max_references`, a speaker past `max_speakers`, a reference that
  is not exactly one of `audio` and `saved_voice_id`, a saved voice given its
  own `text` or saved without one, a missing transcript, control tokens in a
  transcript or script line, a transcript led by another speaker's tag, and
  inline clips over 16 MiB of base64 (`MAX_INLINE_REFERENCE_BYTES`, below the
  25 MiB default body limit of `/v1/audio/speech` and the 64 MiB of the record
  routes). The runtime repeats the structural checks
  (`TtsRuntimeRequest::from_generation`).
- **Plumbing.** `GenerationRequest::speaker_references`
  (`Vec<SpeechSpeakerReference>`: speaker, base64 audio, transcript) travels
  through `TtsRuntimeRequest` into `EngineCoreRequest::tts_speaker_references`
  (crate-private, like the chunk history; `TtsEngineInput`, the direct
  engine's payload, stays single-reference, since only the runtime prepares
  Drama 3). Each chunk of a long text is a clone of the base request, so every
  chunk holds every reference, as upstream's `generate_long_drama3` does. The
  retained-bytes ledger counts them.
- **Preparation** (`prepare_fish_drama3_tts_for_binding`): decodes each clip
  (60 s inline; saved voices hold 30 s from upload), tags its transcript, and
  encodes them in order through the reference cache. A clip it cannot take is
  named by its place (`` `references[i]` ``). `prepare_request` then refuses
  references whose shortest prompt passes the rows left for the text
  (`min(8,192, positions − budget)`) as invalid input naming them, since no
  shorter chunk helps; text too long beside them stays
  `ContextLengthExceeded`, now naming the references' share, and the plan
  cuts the chunk in two as before.
- **Durable records.** Each dialogue reference is its own pair of input
  artifacts (original and canonical audio), whose metadata carries
  `reference_index`; the stored request keeps each reference's speaker,
  transcript and saved voice id but not its audio, and not the script. Every
  stage attempt rehydrates every reference from its artifact
  (`hydrate_batch_reference_audio`). The model snapshot lists each
  reference's speaker and audio and transcript hashes, so a recovered plan's
  identity changes when any of them does; an idempotency key's digest hashes
  each reference's audio in order. Requests without dialogue serialize as
  before, so their digests are unchanged.
- **CLI.** `izwi tts` takes `--speaker-reference N=AUDIO[:TRANSCRIPT_FILE]`
  and `--speaker-voice N=SAVED_ID` (repeatable; the single-reference flags join
  them as speaker 0), `--script FILE.jsonl` and `--seed`. Its request timeout is
  900 s per chunk the family's chunker plans for the text (`plan_chunks`), one
  fence for every other model.

## Take editing

A take can continue earlier takes (plan §8.4). No public engine implements
upstream's `previous_takes`; Izwi builds on the trained history-turn layout
(`FS-omni/generate.py:158-169`, Report Fig. 12): an earlier take is a
completed `user` turn (its text) and `assistant` turn (`<|voice|>`, its frames,
`<|im_end|>`) before the new text.

- **No second history mechanism.** The takes' chunks enter the chunk
  session's history (`SpeechChunkHistory`) before the new text's first chunk,
  as if they were the session's own earlier chunks
  (`generate_speech_plan_stream_with_progress` reads them from the base
  request's `chunk_history`). The family's policy then applies unchanged: the
  first take's first chunk is pinned, the latest chunk follows, and
  `build_chunk_prompt` drops the latest, then the pinned one, when the prompt
  would pass its rows. A long new text is a chunk session after them. The
  engine and `prepare_request` are untouched.
- **What a take keeps** (`crates/izwi-server/src/api/speech_takes.rs`). Under
  that policy only a take's first chunk (pinned when it is the first take) and
  its last chunk (the latest when it is the last take) ever reach a later
  prompt, so a take keeps exactly those two (one when it had one chunk). That
  bounds the stored codes and the durable checkpoint at two chunks whatever
  the take's length. At most 2 `previous_take_ids` (`MAX_PREVIOUS_TAKES`): a
  third take could never reach the prompt.
- **Storage** (migration 15, nullable, Drama 3 records only:
  `keeps_takes` is `continues_speech_across_chunks`):
  - `generated_codes_json`, written only by a take's completion (the
    synchronous `complete_record`, the attempt-fenced
    `complete_record_for_attempt`, and the opaque final-audio settlement), so
    a failed or cancelled take has none. Versioned JSON
    (`SpeechTakeCodes`, `version: 1`):
    `{"version": 1, "chunks": N, "turns": [{"chunk": i, "text": "…",
    "codebooks": 10, "codes": "<base64>"}]}`. `chunks` counts the take's
    chunks; each kept turn carries its chunk index, the text the chunker gave
    it (re-tagged pieces included) and its codes `[codebooks][frames]`
    flattened codebook by codebook as little-endian u16, base64, the same
    encoding as a checkpoint's history turn (`encode_chunk_turn`). About 13 KB
    of codes per 30 s of a chunk, 4/3 of that in base64.
  - `references_json`: `[{"speaker": N, "saved_voice_id": "…"}]` in request
    order (`SpeechTakeReference`); an inline clip is speaker-only, the
    single-reference form is speaker 0, and no reference is `[]`. Written when
    the record is created, so a failed take reports its voices too.
  - `seed`: the request's seed, its 64 bits in the signed column.
  - `parent_take_id`: the last of `previous_take_ids`. No foreign key: a
    parent may be deleted, and its children keep the id.
  - `GET` reports `references`, `seed`, `parent_take_id` and
    `editable` (`generated_codes_json IS NOT NULL`), never the codes. Every
    other record serializes as before.
- **Durable records.** A take's kept chunks ride in the speech checkpoint
  (`SpeechCheckpoint::take`) beside the chunk history, so a take whose chunks
  run as separate stage attempts keeps its first chunk. The stored request
  keeps `previous_take_ids`; every stage attempt reads the takes again
  (`load_previous_takes`), and the plan identity lists each take's id and
  the sha256 of its stored codes, so a recovered plan continues the same
  takes. A take deleted meanwhile fails the record with its `404` message
  (a client error, so the job does not retry). An idempotency key's digest
  adds the codes' hashes in order (`previous_takes_codes_sha256`); requests
  without previous takes digest as before.
- **Checks before any load**: the shape (`check_previous_take_ids`: a model
  that keeps no takes, more than 2, an empty or repeated id), then each take
  read from the request's route (`get_take`): `404` when it is not there,
  `400` for another model's take or one without kept codes (not ready, or made
  before migration 15). Records belong to no caller, so a take is any record
  the route's `GET` returns.
- **Not on `/v1/audio/speech`** or the CLI (which uses it): that route keeps no
  records, and a field that names records belongs on the routes that keep
  them.
- **UI.** **Edit this take** reopens the generation dialog from the record
  (`ui/src/features/text-to-speech/take.ts`) and sends
  `previous_take_ids: [id]`; **Open in dialogue editor** restores the saved
  voices and seed the same way, without continuing the take.

## Routes and capabilities

The admin API's `speech_capabilities` for this model:

| Field | Value |
| --- | --- |
| `supports_builtin_voices` | `false` |
| `supports_reference_voice` | `true` |
| `reference_transcript_required` | `true` |
| `supports_seeded_voice` | `true` (a new field, present only when `true`) |
| `supports_speaker_dialogue` | `true` (present only when `true`) |
| `max_speakers` | `16` (present only with `supports_speaker_dialogue`) |
| `max_references` | `8` (present only with `supports_speaker_dialogue`) |
| `supports_streaming` | `true` |
| `supports_auto_long_form` | `true` |
| `supports_in_progress_replay` | `true` |
| `inline_tags` | `[sighs]`, `[laughs]`, `[pause]`, `[gasps]`, `[clears throat]`, `[hesitates]` |

The server's voice policy (`validate_speech_voice` in `tts_policy.rs`) accepts
a request with no voice input, or references with their transcripts, inline
or from saved voices (see [Dialogue](#dialogue)); inline clips may last 60 s,
saved voices 30 s. The
OpenAI `voice` field is ignored, and `language` is a hint the model does not
read. Realtime voice sessions refuse the model, since they need built-in
voices. Prefix reuse is excluded in the catalog (`prefix_reuse.rs`).

## Environment

The family's engine knobs, in [`env/core.md`](env/core.md), all read at model
load:

- `IZWI_FISH_DRAMA3_DTYPE`: the activation dtype, f16 or f32 on Metal (f32
  also stores the paged KV in F32 and runs every MoE layer's host loop), bf16
  on CUDA, f32 on CPU. Anything else fails the load.
- `IZWI_FISH_DRAMA3_FUSED_DECODE`: the fused trunk, tower and draws and
  the one-readback frame on Metal and CUDA (default `true`); any false value
  runs the eager chain with two readbacks a frame.
- `IZWI_FISH_DRAMA3_MOE_BACKEND`: `fused` (default) or `loop`, on Metal and
  CUDA.
- `IZWI_FISH_DRAMA3_SEGMENT_GRAPHS`: CUDA graphs of the fused trunk and
  tower steps' stretches (default `true`).
- `IZWI_FISH_DRAMA3_NATIVE_BATCHING`: two rows at once on Metal instead of
  one.
- `IZWI_FISH_DRAMA3_PREFILL_SPAN`: rows per prefill cell, 1 to 8,192; 256 on
  CPU and Metal, 1,024 on CUDA.

The server's `IZWI_FISH_DRAMA3_LONG_FORM_ENABLED` (default `true`) is
described above; Fish S2 Pro keeps `IZWI_FISH_LONG_FORM_ENABLED`.

`IZWI_FISH_DRAMA3_BACKEND`, `_CODEBOOK_FIXTURE`, `_CODEC_PATH`, `_MODEL_DIR`,
`_REFERENCE_DIR`, `_SLICE_DIR` and `_TOKENIZER_PATH` are read only by the
tests below.

## Tests

### Unit and synthetic suites (no weights)

`synthetic.rs` writes a checkpoint with every real tensor name and dtype at a
tiny geometry: hidden 128, 8 trunk layers (full attention at 3 and 7), 8
experts, a 64-code semantic band and 4 + 9 tower layers. A stub codec makes
2,048 samples a frame. Other crates' tests load it through izwi-core's
`test-support` feature
(`fish_drama3::test_support::write_lifecycle_synthetic_model`).

```bash
cargo test -p izwi-core --lib fish_drama3
cargo test -p izwi-core --features metal --lib fish_drama3
cargo test -p izwi-server --lib fish_drama3
```

The izwi-core suites cover the family's units, the CPU and Metal device
columns, batch invariance and the runtime
(`crates/izwi-core/src/runtime/fish_drama3_tests.rs`: load, streaming and
non-streaming speech, a cloned voice, speaker references (one for speaker 0
speaks as the single reference does; a second one, and its speaker, change
the take; references that fill the prompt are refused), requests alone and
together, cancellation mid-prefill and mid-decode, the frame budget,
admission refusal and unload). The izwi-server suite
(`crates/izwi-server/src/api/fish_drama3_speech_tests.rs`) runs the speech
routes end to end: a seeded and a cloned voice in each native format, refused
references and control tokens, SSE streams, long text as a chunk session
carrying its history, and a durable record whose chunks recover their history
from the checkpoint; for dialogue, a two-speaker script with a reference per
speaker, raw tags in `input`, a referenced speaker beside a seeded one, saved
voices per speaker, the voice-cloning record route, a long dialogue of three
or more chunks, a durable record with an inline and a saved-voice reference
across stage attempts, and every dialogue `400` on both route families; for
take editing, takes continuing one and two earlier takes on the durable route,
a long take continued by a long edit across stage attempts (and failed when
its earlier take is deleted between attempts), a dialogue take on the
synchronous voice-cloning route continued with its stored voices, and every
`previous_take_ids` refusal. Each take's expected audio is the runtime's chunk
session with every chunk of the earlier takes pushed into its history first.

### Fixtures and generators

`crates/izwi-core/tests/fixtures/fish_drama3/` holds the pinned `config.json`,
the tensor census and the published tensor names, and its
[`README.md`](../../crates/izwi-core/tests/fixtures/fish_drama3/README.md)
has every schema and test in detail. The generators in `scripts/fixtures/`
run upstream fish-speech at `0f8a12e8` from a checkout passed as
`--fish-speech`, with `python3 -I` from the repository root:

| Script | Writes | Needs |
| --- | --- | --- |
| `generate_fish_drama3_goldens.py` | `prompt_goldens.json`, `chunk_goldens.json` (prompt rows, validation and chunking from upstream's builder and chunker) | CPU; the pinned `config.json` and `tokenizer.json`, no weights |
| `generate_fish_drama3_tiny.py` | `tiny/`: a tiny checkpoint, its tokenizer and upstream's prompts, traces and rollouts under recorded noise (L1) | CPU torch; about 1–2 minutes, under 2 GB of RAM |
| `fetch_fish_drama3_slices.py` | Range-downloaded real-weight slices (the tower, head rows, embedding rows, trunk layers 0 and 3), checked against the pins (L1.5) | Standard library only; about 4.2 GB of downloads |
| `generate_fish_drama3_slices.py` | `slice_expected.{json,safetensors}`: upstream's outputs on the slices (L1.5) | CPU torch; under 8 GB of RAM |
| `generate_fish_drama3_reference.py` | `reference.{json,safetensors}` and `audio/`: whole-model generations under recorded noise, codec PCM (L2) | A GPU with 80 GB or more (about 20–25 minutes) or a CPU box with 64 GB or more (about 1–1.5 hours) |

`fish_drama3_fixture_common.py` holds what they share: the xorshift
generator, checksums, trace entries and the recorded-noise layout.

### Parity suites (ignored)

Each ignored test names the generator or variable it needs and fails at once
without it. The real-weight levels need a release build.

```bash
cargo test --release -p izwi-core --lib fish_drama3::reference_tests -- --ignored
```

| Level | Tests | Needs |
| --- | --- | --- |
| L1, tiny fixture | `tiny_*`: the synthetic checkpoint is upstream's, prompt rows, prefill and forced steps, rollouts under recorded noise; `_on_metal` and `_on_cuda` variants | `tests/fixtures/fish_drama3/tiny/` |
| L1.5, real-weight slices | `slice_*`: DeltaNet layer 0, attention layer 3, the tower on a forced step, the main head and embedding; `_on_metal` and `_on_cuda` variants | `IZWI_FISH_DRAMA3_SLICE_DIR`; 4–7 GB of RAM; `--test-threads 1` |
| L2, whole model | `reference_prompts_match_upstream`, `reference_utterance_pcm_matches_the_upstream_codec`, `reference_generations_match_upstream` | `IZWI_FISH_DRAMA3_REFERENCE_DIR` and `IZWI_FISH_DRAMA3_MODEL_DIR`; for generations also `IZWI_FISH_DRAMA3_BACKEND` |

`IZWI_FISH_DRAMA3_BACKEND` picks the L2 form: `cpu` (F32 weights, about
140 GiB of RAM), `cpu-q8_0` (the production Q8_0 form, about 40 GiB, where
divergence is reported only), `metal` (a Mac with 64 GB or more, built with
`--features metal`) or `cuda` (an SM80+ GPU, built with `--features cuda`).

```bash
IZWI_FISH_DRAMA3_MODEL_DIR=/path/to/drama-3 \
IZWI_FISH_DRAMA3_REFERENCE_DIR=$PWD/target/fish-drama3-reference \
IZWI_FISH_DRAMA3_BACKEND=cuda \
  cargo test --release -p izwi-core --features cuda --lib \
  fish_drama3::reference_tests::reference_generations -- --ignored
```

Other ignored tests read the published files directly:

- `IZWI_FISH_DRAMA3_TOKENIZER_PATH` (the pinned `tokenizer.json`): the golden
  id tests in `tokenizer.rs` and `prompt.rs`, and with the goldens above,
  the prompt and chunk golden tests.
- `IZWI_FISH_DRAMA3_CODEC_PATH` (the published `codec.pth`): the cross-family
  codec tests in `models/architectures/fish_codec_equivalence.rs` (Drama 3's
  codec bit-identical to Fish S2's, on CPU, Metal and CUDA) and the codec
  pricing check in `fish_drama3_memory.rs`.
- `IZWI_FISH_DRAMA3_CODEC_PATH` with `IZWI_FISH_DRAMA3_CODEBOOK_FIXTURE` (a
  JSON array of more than 128 real generated codebook rows): the CUDA codec
  streaming test in `codec/streaming.rs`.

### CUDA device suites

On a CUDA box (SM80 or newer), the device lane runs the Drama 3 CUDA suites
with the other CUDA checks:

```bash
scripts/ci/check-backend-truth.sh cargo-cuda-device
```

or the Drama 3 suites alone:

```bash
CUDA_COMPUTE_CAP=<cap> IZWI_REQUIRE_CUDA_TEST_DEVICE=1 cargo test -p izwi-core --features cuda --lib -- \
  --test-threads=1 kernels::cuda::row_gemv kernels::cuda::split_decode fish_drama3::cuda fish_drama3::device \
  runtime::fish_drama3_tests::cuda a_cuda_load_holds_what_its_census_prices
```

They cover the row-invariant GEMV and row-length split decode kernels,
component parity against the CPU reference, batch invariance at widths 1 to
8, the load checks, device sampling, the runtime at widths 1 and 8, and a load
holding what its census prices. `IZWI_REQUIRE_CUDA_TEST_DEVICE=1`
makes a missing device a failure rather than a skip.

### Benchmarks

- **Manifests.** `benchmarks/manifests/{cpu,metal}-audio-concurrency.toml`
  measure Drama 3 at c1 and c2, the rows those backends decode together;
  `cuda-audio-concurrency.toml` at c1 to c32 (eight rows decode together,
  so c16 and c32 measure the queue). Every case clones `data/fox.wav` with
  `data/fox.md`, as Fish S2 Pro's do. `cuda-family-api.toml` and
  `cuda-family-load.txt` list the model for the CUDA certification runner.
- **TTS matrix.** `scripts/bench/run-tts-matrix.sh --models
  FishAudio-Drama-3:drama3` runs a seeded voice, a cloned one, a stream,
  numbers, tags, a two-speaker dialogue and about five chunks of long text,
  all from seed 7. The family's footprint guard defaults to 45,000 MiB (the
  12,000 MiB default would kill the server while it loads) and its load
  timeout to an hour.

## Hand-off kit

`scripts/e2e/fish-drama3/` runs the parity suites, the end-to-end matrix,
load parity and eval-mini on machines that hold the model, and packs the
results into one tarball; its
[`README.md`](../../scripts/e2e/fish-drama3/README.md) has the hardware
table, the prerequisites and every option. From the repository root:

```bash
K=scripts/e2e/fish-drama3
bash $K/fetch.sh            # once per machine: 41.5 GB at the pin, hashed once
bash $K/build.sh cuda && bash $K/run-parity.sh cuda && \
  bash $K/run-e2e.sh cuda && bash $K/run-load.sh cuda && bash $K/collect.sh cuda
```

with `metal` on a Mac with 64 GB or more and `cpu` on a box with 64 GB of
RAM or more. Every shell script takes `--help` and `--dry-run`.

- `fetch.sh` downloads with a resumable curl loop, checks every file's size
  and SHA-256 against `FISH_DRAMA3_FILES`, then writes `izwi-artifact.json`
  as izwi's downloader does after `verify_full`, so izwi reads the bundle as
  downloaded.
- `build.sh` builds `izwi-server` in a private target directory with
  `IZWI_BUILD_GIT_SHA`, and `run-e2e.sh` and `run-load.sh` refuse a server
  whose logged `build_git_sha` is not that build's.
- `run-parity.sh` runs the [parity suites](#parity-suites-ignored) leg by
  leg in one release test build: the family's suites, the codec device
  tests, the published tokenizer and codec, then L1, L1.5 and L2 when their
  fixtures exist (a missing fixture skips its leg; `--require-all` fails
  it). A missing Metal or CUDA device fails the run.
- `run-e2e.sh` runs the end-to-end matrix over HTTP: a seeded and a cloned
  voice (inline and saved), tags, a two-speaker script, long
  text with its history, SSE time to first audio with streamed PCM equal to
  the WAV, the formats, a take edit, cancellation at each stage the API
  reaches, a soak with untracked device memory flat (`--soak-minutes`, 30 by
  default), the backend's fast-path counters, unload and reload, and a load
  refused under an undersized budget. It keeps every take under `audio/`.
- `run-load.sh` runs `scripts/bench/run-audio-load.py --require-parity` with
  a tolerance of 0 (rows are batch-invariant) at the backend's widths, a
  streaming probe of time to first audio, RTF and frames/s beside the card's
  H200 table, and chat inter-token latency beside a long take
  (`FD3_CHAT_MODEL`).
- `eval-mini.py` clones 100 LibriSpeech test-clean pairs and 25 + 25
  Seed-TTS-eval items at both sampling profiles (the defaults, and 0.7 / 0.7
  / 30) and scores any directory of WAVs for WER and ECAPA similarity, so
  izwi's takes are compared with the official engine's (Leg B), which the
  tester runs on the same machine. It runs no upstream code.

`build.sh cpu --synthetic` and `run-e2e.sh cpu --synthetic` run the same
matrix against the synthetic checkpoint (written by the ignored izwi-server
test `write_the_e2e_kit_synthetic_checkpoint`, served by a debug build with
izwi-core's `test-support` feature), which checks the kit and the routes
without the weights.

Send back `target/fish-drama3/<backend>/fish-drama3-report-<backend>-<sha>.tar.gz`
per backend: the report (build, parity, e2e, load and eval sections), the
admission and memory snapshots, the path counters, timings, log tails and
the WAVs. From the generator runs, also send the L1 `tiny/` directory, the
L1.5 `slice_expected.json` and the L2 `reference.json` and
`reference.safetensors`.
