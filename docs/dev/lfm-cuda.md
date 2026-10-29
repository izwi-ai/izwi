# LFM CUDA implementation and qualification

LFM uses Candle 0.11.0. The text backbone and audio depthformer retain quantized
projections; audio encoder/detokenizer components are dense. Backbone
activations and recurrent rings are F32. Chat paged KV (LFM2.5 and LFM3) is F16
on CUDA and Metal and F32 on the CPU; LFM2.5-Audio's KV is F32. A generic
configured BF16 preference does not convert this model. CUDA support remains
source-reviewed and unverified until device evidence is collected.

The implementation fixes the text executor/sequence-adapter mismatch, uses
Candle's native noninterleaved RoPE (including different positions per decode
row), softmax, SiLU and tanh GELU, and packs greedy token IDs before host readback.
Retained singleton audio prefill uses the existing full-span path. ASR declares
its existing full-prompt prefill workspace estimate as well as decode workspace,
including when the execution adapter is rebound. ShortConv only
constructs ring updates for the suffix that survives. Nonstreaming audio-chat
decodes the final waveform without repeatedly decoding intermediate prefixes.
Relative-position Conformer attention and GLU semantics are preserved.

## Optional depthwise convolution

`IZWI_LFM_CUDA_DEPTHWISE_CONV=1` enables an experimental Candle
unfold/multiply/reduce implementation for the encoder's F32 kernel-9, padding-4,
stride-1 depthwise Conv1d. It reuses folded affine weights and limits each expanded
product to 1 MiB. Padded input, reduced chunks, output concatenation and allocator
retention consume additional memory; the limit is not a total workspace bound.
Unsupported geometry uses the existing convolution. Unset the variable or set it
to `0` to retain the default path. Two-dimensional subsampling is unchanged.

Keep this option off until end-to-end quality, peak memory and latency improve
on the deployment GPU. No additional custom CUDA kernel was introduced.

## Validation on NVIDIA hardware

Build/link checks, device numerical checks and model performance evidence are
separate requirements. The portable tests alone satisfy neither CUDA compilation
nor GPU qualification. With the CUDA toolkit and an NVIDIA device, run the
explicit kernel and generation-guard tests (these fail if the device is unavailable):

```bash
cargo test --locked -p izwi-core --lib --features cuda \
  cuda_lfm2_batched_rotary_matches_scalar_at_ragged_positions -- --ignored
cargo test --locked -p izwi-core --lib --features cuda \
  cuda_lfm2_finite_validation_rejects_nonfinite_values -- --ignored
cargo test --locked -p izwi-core --lib --features cuda \
  cuda_lfm2_decode_rejects_nonfinite_logits_before_argmax -- --ignored
cargo test --locked -p izwi-core --lib --features cuda \
  cuda_candle_conformer_operations_and_depthwise_match_reference -- --ignored
cargo test --locked -p izwi-core --lib --features cuda \
  lfm25_cuda_packed_frame_tokens_preserve_greedy_and_mixed_rows -- --ignored
```

Run the normal LFM regressions with the deployment features, including
`flash-attn` and `cudnn` if deployed. Do not infer a provider from enabled features;
retain the provider actually observed by runtime telemetry.

Against a server built from the same clean commit, with all models provisioned:

```bash
scripts/bench/run-model-evidence.sh --backend cuda \
  --manifest benchmarks/manifests/lfm-cuda.toml \
  --server http://127.0.0.1:8080 --output target/lfm-cuda-evidence
```

The manifest covers the LFM2.5 Instruct, LFM2.5 Thinking and LFM3 chat models,
and LFM2.5-Audio and LFM3-Audio ASR/TTS, at concurrency 1/2/4/8, with streaming
and nonstreaming requests. Compare the baseline and changed builds on the same
GPU, weights, inputs and limits. Retain TTFT, prefill/decode latency,
tokens or audio seconds per second, peak VRAM, workspace, kernel launches and
host synchronization counts. Check transcript quality, audio quality, finite
outputs, cancellation, EOS and rollback. Include long prompts and unequal prompt
lengths in a separate stress run. Audio-chat has no benchmark command: explicitly
exercise its streaming and nonstreaming API, checking final transcript/audio and
stream output boundaries. The manifest does not cover that route.

## LFM3

LFM3-1.7B chat and LFM3-Audio-2B build for CUDA and are compile-checked only:
neither has run on an NVIDIA GPU. [LFM3 and LFM3-Audio](lfm3.md) describes the
models; this section covers what CUDA adds.

**Sliding windows.** LFM3's global and sliding-window attention layers share one
page table (`lfm2/physical.rs`). A sliding layer's decode reads a narrowed block
table that starts at its window's first page, with a first-page offset
(`PhysicalPagedKvCache::sequence_table_with_window`). Its paged prefill passes
`window_tokens` to the arena (`backends/kv/accelerator.rs`): with the
`flash-attn` feature that is Candle's `flash_attn_varlen_paged_windowed` with a
left window of `window - 1`, and the native paged prefill kernel takes the window
itself. Stateless forwards (LFM3-Audio's speech encoder and LiquidCodec trunk)
pass the same left window to CUDA flash attention, when it is requested, on
masked prefill only (`lfm25_cuda_flash_attention_options`). Pages behind a
window are not released, so the KV reservation covers every attention layer at
the full context.

**F16 KV.** LFM3 chat and LFM3-Audio store paged KV in F16 on CUDA
(`lfm2_chat_kv_dtype`, `lfm3_audio_kv_dtype`). Queries and the attention output
are cast at the call sites and scores accumulate in F32; activations and
ShortConv rings stay F32. The LFM3 chat parity leg runs with the backend's KV
dtype against the F32 reference, so on CUDA it checks F16 KV.

**Context.** LFM3-1.7B is trained to 262,144 tokens. On CUDA LFM3 chat defaults
to the checkpoint's whole context, and its one-context resident pool
(`CudaResidentPool::OneContext`) is fitted to device memory at load, which caps
the reach; CPU and Metal default to 32,768. `IZWI_LFM3_CONTEXT_TOKENS` sets it on
any backend (see the support matrix's context table). The RoPE tables are built
in F64; `rope_tables_match_f64_at_far_positions_on_cuda` checks them at positions
up to 127,999, not across the whole 262,144. LFM3-Audio serves 32,768 tokens on
every backend. Prefix reuse is off for LFM3 chat on CUDA: the contract declares
committed pages and ring snapshots, but no GPU run has checked them.

**Other paths.** With fused kernels on, a CUDA ShortConv ring kernel that
declines is an error, not a silent fallback; `IZWI_FUSED_KERNELS=0` takes the
op path. LFM3-Audio's depth decoder runs op-by-op on CUDA (its fused kernels
are Metal only), and its mel features, sampling and inverse FFT run on the host.

On an NVIDIA host, run the LFM `*_on_cuda` tests. The two backbone tests fail
without a device. The model legs print `skipping: no CUDA device` and pass when
the selected device is not CUDA, and the trunk test passes silently when
`Device::new_cuda(0)` fails, so read their output. The real-weight legs read
`IZWI_LFM25_MODELS_DIR`, else the default models directory, and skip when the
model folder is missing.

```bash
# Shared backbone: RoPE tables at far positions, the op-path ShortConv ring.
cargo test --locked -p izwi-core --lib --features cuda \
  rope_tables_match_f64_at_far_positions_on_cuda -- --ignored
cargo test --locked -p izwi-core --lib --features cuda \
  unfused_shortconv_ring_falls_back_on_cuda -- --ignored
# The LFM3 trunk against its F64 reference (synthetic weights).
cargo test --locked -p izwi-core --lib --features cuda \
  lfm3_trunk_matches_the_reference_on_cuda -- --nocapture
# Real weights: LFM3 chat against the F32 reference, LFM3-Audio ASR and TTS.
cargo test --locked -p izwi-core --lib --features cuda \
  lfm3_chat_matches_the_f32_reference_on_cuda -- --ignored --nocapture
cargo test --locked -p izwi-core --release --lib --features cuda \
  lfm3_audio_transcribes_and_speaks_on_cuda -- --ignored --nocapture
# LFM2.5 on the same backbone.
cargo test --locked -p izwi-core --lib --features cuda \
  lfm25_chat_matches_the_f32_reference_on_cuda -- --ignored --nocapture
cargo test --locked -p izwi-core --lib --features cuda \
  lfm25_f16_kv_tracks_f32_kv_on_cuda -- --ignored --nocapture
```

The runtime routes (ASR, TTS, voice chat, LiveASR, spoken tool rounds) run on CUDA
through `runtime/lfm3_audio_tests.rs` with `IZWI_LFM3_AUDIO_BACKEND=cuda`:

```bash
IZWI_LFM3_AUDIO_MODELS_DIR=/path/to/models IZWI_LFM3_AUDIO_BACKEND=cuda \
  cargo test --locked -p izwi-core --features cuda --release --lib lfm3_audio_tests -- --nocapture
```

`lfm-cuda.toml` covers LFM3 chat and LFM3-Audio ASR/TTS at concurrency 1/2/4/8,
`cuda-family-api.toml` has one LFM3 chat and one LFM3-Audio TTS case, and
`cuda-family-load.txt` lists both models.

## Remaining evidence-dependent work

- Automatic retained audio context remains capped at 4,096; explicit context can
  exceed this. Request generation is already clamped to effective context minus
  prompt length. Raising automatic capacity requires fitting every physical state
  domain and request workspace, not merely changing the output-token policy.
- Measure multirow token-wave prefill before adding packed/span batching. Preserve
  ragged positions, recurrent state, transaction rollback and cancellation.
- Qualify BF16 attention/KV separately from F16. Update physical contracts,
  allocation byte counts, queries and outputs together; keep recurrence F32.
  Test raw KV finiteness and long-generation quality. F16's narrower exponent
  range needs separate evidence. Dense audio precision is another experiment.
- Batch audio heads only across requests at the same dependent codebook step.
  Preserve stochastic sampling and first-codebook EOS.
- Share ASR/response encoder preparation only when preprocessing and long-form
  chunking agree. Establish detokenizer dependencies before adding incremental
  streaming state. Profile existing graph/provider paths before new kernels.

## Multi-turn incident replay

Use the bounded replay on an already running, dedicated CUDA server. It does not
build, deploy, load models, or certify model quality. Provision both text models,
start the server with `IZWI_LFM2_DIAGNOSTICS=1`, and retain its logs. The script
checks `/v1/health` for the supplied build SHA, CUDA selection, compiled support
and device usability, and `/v1/metrics` for each model's actual CUDA placement.
For the repository CUDA Compose service, explicitly pass the diagnostic setting
and recreate the container (exporting a host variable does not update an already
running server):

```sh
IZWI_LFM2_DIAGNOSTICS=1 docker compose --profile cuda up -d --build --force-recreate izwi-cuda
docker compose --profile cuda exec -T izwi-cuda printenv IZWI_LFM2_DIAGNOSTICS
# Reproduce the failing chat, then collect logs:
docker compose --profile cuda logs --since 10m izwi-cuda > /tmp/lfm-diagnostics.log
```

The `printenv` result must be `1`. Reproduce the failing chat before collecting
the log. For a direct binary or another container launcher, set the variable in
that server's environment and restart it. Layer checks include output-projection
input, Candle QMatMul and optional bias; raw-logit errors also state whether
activation diagnostics were enabled. Capture the first failing stage, preceding
finite stage, request context, deployed SHA and GPU/driver. Do not infer a failing
operator from the final-logit error alone.

Run against the exact baseline and changed deployment, using separate output
folders:

```bash
python3 scripts/bench/run-lfm-cuda-replay.py \
  --server http://127.0.0.1:8080 --expected-sha FULL_DEPLOYED_COMMIT_SHA \
  --output target/lfm-replay-baseline --server-log /path/to/server.log
```

The built-in cases preserve the user prompts from the Thinking and Instruct
incident screenshots, including their original spelling. Each case runs once
with streaming and once without it, in independent server conversations, and
uses the model's actual replies as subsequent history. The script saves the
persisted conversation before deleting only the temporary conversations it
created. Use `--route stateless` to repeat through `/v1/chat/completions`, supplying
that same generated history explicitly.

Each turn retains its request, raw response/SSE, response headers, final parsed
result and client-observed elapsed time. Health and metrics snapshots preserve
only telemetry the server actually exposes; enabled features are not treated as
provider observations. `--server-log` copies bytes appended during the run from
a log on the harness host: use a server-side copy when the server is remote.
The report does not invent kernel counts, GPU timings or logits diagnostics when
those are absent. Capture server diagnostics separately if the log is unavailable.

Requests default to a 512-token diagnostic cap and a 120-second absolute deadline
per HTTP exchange, with an 8 MiB response bound. `--max-tokens`, `--timeout` and
`--temperature` make these controls explicit. A cap is a diagnostic constraint,
not evidence that the model completed its reasoning. HTTP/SSE failures, missing
stream completion, reasoning-only or empty answers, exact long repeated answers,
and repeated suffixes fail the smoke gate. Repetition is a review flag and can be
legitimate; a passing report still requires human review for relevance and
accuracy. Reasoning text alone never satisfies the visible-answer gate.

For a fixed-history comparison, pass `--route stateless --cases fixture.json`:

```json
[
  {
    "name": "fixed-history",
    "model": "LFM2.5-1.2B-Instruct-GGUF",
    "history": [
      {"role": "user", "content": "Tell me about victoria falls"},
      {"role": "assistant", "content": "PASTE THE CAPTURED BASELINE ANSWER HERE"}
    ],
    "turns": ["Where is it"]
  }
]
```

Fixture history must be copied from captured evidence for a teacher-forced
comparison. Subsequent replies within the fixture still come from the model.
Portable harness checks: `python3 scripts/bench/test-lfm-cuda-replay.py`.
These fixtures exercise parsing and acceptance; they are not device evidence.

## Response integrity and sampling

LFM validates raw logits with Candle finite-value reductions before sampling.
NaN/infinity, invalid control selections, empty terminal text and repetition-stop
failures return explicit inference errors rather than successful blank responses.
Padding is not treated as an implicit end token. Declared stop IDs are honored.
Successful terminal logs include request ID, model, stop reason (`eos`,
`configured_stop`, or `length`), prompt/output counts and resolved output budget.

Managed generation uses the existing shared ChatSampler for temperature, top-k,
top-p, repetition/presence penalties and seed. Repetition penalties below 1 are
explicitly rejected because that shared path only supports penalties of 1 or
higher. Default direct generation remains greedy. The default greedy batched
path retains packed token readback; sampled rows retain independent RNG history.
Sampler, incremental UTF-8 decoder and prefill cursor are part of rollback.
The prompt is rendered exactly like the checkpoint's chat template: one BOS, a
system turn only when the request sends one, and no extra instructions.

`IZWI_LFM2_DIAGNOSTICS=1` additionally checks intermediate norms, projections,
attention, recurrent convolution and MLP outputs. It logs layer/row/position,
shape, dtype, device and scalar min/max, plus at most the first 64 selected IDs
per request attempt. Request/row context logs associate these with serving work.
This mode intentionally synchronizes and can substantially slow inference;
disable it for performance measurement. It does not log full prompt or response
text, but token IDs and captured replay artifacts can reveal user content.

The native attention synchronization repair below addresses a code-level race;
its relationship to application nonfinite logits still requires hardware evidence.
The fixed-history harness compares prompt replays; it does not force every
generated token or prove per-layer CUDA/CPU equivalence. Use the activation trace
to locate the first failing layer before selecting further operator or precision
changes.

### Native paged-attention synchronization qualification

The native CUDA attention kernels reuse shared reduction storage between tokens
and partitions. Their trailing barrier must follow every read of that storage
before the next iteration can overwrite it. This correctness repair does not by
itself establish the source of an application's nonfinite logits.

On an NVIDIA host with Compute Sanitizer installed, compile and run the explicit
hardware regression under Racecheck:

```sh
cargo test --locked -p izwi-core --features cuda --lib --no-run \
  --message-format=json > /tmp/izwi-cuda-test-build.jsonl
lfm_cuda_test_binary=$(python3 - <<'PY'
import json
from pathlib import Path
executables = []
for line in Path('/tmp/izwi-cuda-test-build.jsonl').read_text().splitlines():
    message = json.loads(line)
    if (message.get('reason') == 'compiler-artifact'
            and message.get('target', {}).get('name') == 'izwi_core'
            and message.get('profile', {}).get('test')
            and message.get('executable')):
        executables.append(message['executable'])
assert len(executables) == 1, executables
print(executables[0])
PY
)
compute-sanitizer --tool racecheck --error-exitcode 1 \
  "$lfm_cuda_test_binary" \
  kernels::cuda::tests::cuda_native_paged_attention_scratch_reuse_matches_candle \
  --ignored --exact --test-threads=1
```

The test requires CUDA device 0 and fails if unavailable. It repeatedly compares
F32 native causal prefill, one-pass decode and forced partitioned decode/reduction
with CPU Candle attention using nontrivial inputs and multiple warps. Record the
GPU, build SHA, numerical result and sanitizer output; portable compilation or a
passing numerical run alone does not establish absence of a shared-memory race.
Then rerun the model conversation replay and capture the first failing activation
if nonfinite logits persist.
