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
| Sampling | Host | Device, two readbacks a frame | Device, two readbacks a frame |
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

## Routes and capabilities

The admin API's `speech_capabilities` for this model:

| Field | Value |
| --- | --- |
| `supports_builtin_voices` | `false` |
| `supports_reference_voice` | `true` |
| `reference_transcript_required` | `true` |
| `supports_seeded_voice` | `true` (a new field, present only when `true`) |
| `supports_streaming` | `true` |
| `supports_auto_long_form` | `true` |
| `supports_in_progress_replay` | `true` |
| `inline_tags` | `[sighs]`, `[laughs]`, `[pause]`, `[gasps]`, `[clears throat]`, `[hesitates]` |

The server's voice policy (`validate_speech_voice` in `tts_policy.rs`) accepts
a request with no voice input, or one reference with its transcript, inline
or from a saved voice; inline clips may last 60 s, saved voices 30 s. The
OpenAI `voice` field is ignored, and `language` is a hint the model does not
read. Realtime voice sessions refuse the model, since they need built-in
voices. Prefix reuse is excluded in the catalog (`prefix_reuse.rs`).

## Environment

The family's engine knobs, in [`env/core.md`](env/core.md), all read at model
load:

- `IZWI_FISH_DRAMA3_DTYPE`: the activation dtype, f16 or f32 on Metal (f32
  also stores the paged KV in F32 and runs every MoE layer's host loop), bf16
  on CUDA, f32 on CPU. Anything else fails the load.
- `IZWI_FISH_DRAMA3_MOE_BACKEND`: `fused` (default) or `loop`, on Metal and
  CUDA.
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
non-streaming speech, a cloned voice, requests alone and together,
cancellation mid-prefill and mid-decode, the frame budget, admission refusal
and unload). The izwi-server suite
(`crates/izwi-server/src/api/fish_drama3_speech_tests.rs`) runs the speech
routes end to end: a seeded and a cloned voice in each native format, refused
references and control tokens, SSE streams, long text as a chunk session
carrying its history, and a durable record whose chunks recover their history
from the checkpoint.

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
