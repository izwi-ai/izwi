# SOV-Transcribe-1

izwi serves Salesforce's SOV-Transcribe-1 as an offline speech recognition
model on NVIDIA GPUs (CUDA) only. It is its own model family
(`sov_transcribe`): the Whisper large-v3 encoder, a projector that stacks
four encoder frames into one audio token, and the Gemma 4 26B-A4B decoder
(30 layers, sliding and full attention with different head geometries, and a
dense MLP beside 128 routed experts in every layer). It imports no other
family; `the_family_imports_no_other_architecture` holds that line.

User-facing behaviour is in the [support matrix](../user/support-matrix.md),
the [models guide](../user/models/index.md) and the
[transcription guide](../user/features/transcription.md).

| Variant | Files | Family |
| --- | --- | --- |
| `SOV-Transcribe-1` | 16 shards, 53.3 GB, BF16 (routers F32) | `SovTranscribeAsr` |

## Pinned checkpoint

| Repository | Revision | Licence |
| --- | --- | --- |
| `salesforce-sovereign-intelligence/SOV-Transcribe-1` | `505d8fae243924c0d8acdc622f7f0abab344eba3` | Apache-2.0 |

`tests/fixtures/sov_transcribe/` holds the published configs and the census
of all 1,147 tensors, read from the shards' safetensors headers.

## CUDA only

The catalogue gives the variant `required_backend() == Some(Cuda)`. On a CPU
or Metal server:

- `/v1/admin/models` lists it with `required_backend: "cuda"` and
  `device_fit: "unsupported_backend"`; the app shows "CUDA only" and offers
  no download or load; `izwi list` shows `cuda only`.
- `RuntimeService::ensure_model_runs_on_this_backend` refuses every load and
  request with `InvalidInput` ("SOV-Transcribe-1 needs an NVIDIA GPU (CUDA);
  this server runs on Metal"), a 400, before anything is queued or read.
- The loader refuses the published checkpoint off CUDA as well
  (`precision::require_serving_backend_for`).

The model code is device-generic Candle. The CPU runs a tiny synthetic
checkpoint (`synthetic.rs`) in F32 as the test reference, so the family's
tests and the runtime end-to-end tests run on any machine and in CI. No
Metal path exists.

## Inference contract

- **Audio.** 16 kHz mono, right-padded to whole 30 s windows; Whisper log-mel
  over the whole padded clip at once (the `max - 8` clamp sees every window),
  then cut into windows. Padded windows are kept: every window is 375 audio
  tokens.
- **Encoder.** One batch row per window, the stored position table added from
  row 0 each window, no mask. Windows of different requests share batches of
  `IZWI_SOV_TRANSCRIBE_ENCODER_BATCH` (8).
- **Prompt.** `<bos><|turn>user\n{instruction}<|audio>{audio}<audio|><turn|>\n
  <|turn>model\n<|channel>thought\n<channel|>`, control tokens placed by id.
  The instruction is the API's `prompt` when given, else the card's language
  hint or automatic-language instruction; keywords are prefixed in the card's
  wording. Seven languages.
- **Decoder.** Text embeddings times `sqrt(2816)` rounded to BF16 (53.0);
  audio embeddings unscaled. Attention scale 1.0; full layers take V from
  `k_proj` before `k_norm` and RoPE and rotate 64 of 256 pairs over a
  512-wide exponent. Router in F32 on the raw residual: softmax over all
  experts, top 8, renormalize, then `per_expert_scale`. Greedy, stops on ids
  1, 106 or 50, final logits softcapped at 30.
- **Loops.** A row stops once its last tokens are one block repeated 20
  times; the repeats are dropped.

## Serving

Every request is a retained row (`handler_sov_transcribe_rows.rs`): the first
prefill span encodes the audio of every row in the quantum together, spans
pack into one decoder forward (`IZWI_SOV_TRANSCRIBE_BATCHED_PREFILL`), and
decode steps batch across rows, several per quantum while nothing waits
(`IZWI_SOV_TRANSCRIBE_DECODE_QUANTUM`). The state is one managed paged-KV
domain over all 30 layers (220 KiB per token in BF16); a quantum rolls back
by rewinding the block table and the host decode loop.

A request covers one 30 s window. A longer file is cut at pauses by the
shared chunk planner and runs as concurrent chunk rows
(`runtime/asr/chunk_rows.rs`), joined in order.

### Performance on CUDA

- **Fused experts** (`kernels::moe_gelu`, `IZWI_SOV_TRANSCRIBE_MOE_BACKEND`):
  routing (softmax, top 8, renormalize, per-expert scale) and the BF16
  experts in five launches with the ids on the device. Expert-major kernels
  read each used expert's rows once for every row of the step and sum each
  pair in a fixed order, so a row decodes the same alone as in a batch.
  `auto` takes them for calls of up to 512 routed pairs (every decode step
  up to 64 rows); prefill spans route on the host, where each used expert's
  tokens run as one GEMM. They serve only after a load-time check against
  the host path (`moe_check`).
- **Decode width 32** on CUDA (`catalog/batch_width.rs`).
- **CUDA graphs**: the encoder and projector per batch size of windows
  (`IZWI_SOV_TRANSCRIBE_ENCODER_GRAPHS`), and each decode step as 31
  stretches between the layers' paged attention, per row count 1 to 4 and 8
  (`IZWI_SOV_TRANSCRIBE_DECODE_GRAPHS`). Both verify every graph bit for
  bit against the eager run before replaying it; any failure turns that set
  off for the process (`encoder_graphs`, `decode_graphs` in the
  diagnostics).
- **Fewer launches per layer**: the q, k, v projections and the dense
  gate/up projections are single fused weights; every RMSNorm is one launch
  of `kernels::residual_norm`, with the residual adds fused into the
  following norm; the RoPE tables for the whole context are resident and a
  step gathers its rows with one upload.
- **Encoder attention** on the shared fused kernel (FlashAttention on CUDA
  builds that have it), never the 1500 x 1500 score matrices.
- **Packed prefill** across rows and **cross-request encoder batches**.

Considered and not done:

- **Freeing KV pages that leave the sliding window.** A served request is
  one 30 s window, at most about 410 prompt and 424 planned decode tokens,
  so a row never passes the 1,024-token window and the sliding layers
  already hold exactly what they attend to. Rotating them separately would
  split the KV state into two domains for no saving on this route.
- **Greedy picks on the raw logits** (skipping the softcap). The reference
  picks over the softcapped logits, whose rounding can tie tokens the raw
  logits order; the saving is three element-wise passes over 1 MB a row.

## What is verified

- Unit and fixture tests: config fail-closed switches, the census against the
  published headers (parameter count exact), whole-clip mel, RoPE tables,
  RMSNorm, routing order, prompt layout, loop detection, residency pricing.
- The paged decoder against a dense, cache-free reference with explicit
  causal and sliding-window masks, in spans of 1, 5 and 24 tokens.
- Transcripts independent of the prefill span, the decode quantum, packing,
  batching, rollback and the expert path.
- The runtime on the synthetic checkpoint with published 30 s windows:
  the CUDA-only refusal, concurrent rows, request validation, and a 75 s
  file as chunk rows.

Not verified: any run on an NVIDIA GPU or on the real checkpoint. The CUDA
path builds (`--features cuda`) but has not run. `reference_tests.rs` is the ignored parity test
for the H100: generate its reference with
`scripts/fixtures/generate_sov_transcribe_reference.py`, then run it with
`IZWI_SOV_TRANSCRIBE_MODEL_DIR` and `IZWI_SOV_TRANSCRIBE_AUDIO_DIR` set.

## Not done

- Any run on an NVIDIA GPU: the fused kernels and the graphs build and the
  CUDA source passes a host-compiler syntax check, but none has run. The
  kernels' CUDA test (`moe_gelu::tests::the_cuda_kernels_match_the_reference`)
  and the load-time checks are the first things to watch on the H100.
- Throughput and the 32-row width are unmeasured.
- The DFlash speculative draft, a smaller (FP8) build, and single-pass long
  audio.
