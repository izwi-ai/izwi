# Nemotron 3.5 Transcribe: hand-off test kit

Scripts that check izwi's Nemotron 3.5 Transcribe 30B-A4B FP8 support on
machines big enough to hold it, and pack the results into one file to send
back. Each layer must pass before the next is worth running.

| Layer | What | Where | Command |
|---|---|---|---|
| L1 | Upstream tiny-random fixture (no weights, no GPU) | any Linux or macOS box with Python | `scripts/fixtures/generate_nemotron_transcribe_tiny.py` |
| L2 | Real-weight reference: Leg A (F32 dequantized) and Leg B (vLLM 0.32 + NeMo plugin) | the CUDA GPU box | `scripts/fixtures/generate_nemotron_transcribe_reference.py` |
| L2 | izwi's ignored parity tests against that reference | every backend | `run-parity.sh <backend>` |
| L3 | HTTP end to end, load parity, WER-mini | every backend | `run-e2e.sh`, `run-load.sh`, `wer-mini.py` |

Model: [`nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B-FP8`](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B-FP8)
at revision `006d73cd8385237ed5603fc6c7e76200e50abbd2`, 34.7 GB in 7 weight
files. izwi's variant id is `Nemotron-3.5-Transcribe-30B-A4B-FP8`.

## Hardware

About 31.6 GiB resident on Metal and CUDA (raw FP8), about 33.6 GiB on the
CPU (FP8 to Q8_0 at load). Machines that cannot fit it are refused before any
weight is read.

| Machine | Fits? | Expected single-request RTF, 1 h of audio | Notes |
|---|---|---|---|
| H100 80 GB / RTX PRO 6000 96 GB | yes | 0.03–0.08 | many rows; vLLM serves 64 sequences on 96 GB |
| L40S / RTX 6000 Ada / A6000 48 GB | yes | 0.05–0.15 | about 8–16 short rows; A6000 (SM86) runs W8A16 in vLLM too |
| Apple M-Max/Ultra, 64 GB or more | yes | 0.1–0.3 | decode is bandwidth-bound: about 110 tok/s ideal at 400 GB/s |
| Apple 48 GB | doubtful | — | Metal working set about 36 GB; likely one short row only |
| CPU box, 64 GB RAM or more | yes | 0.3–1.0 | prefill is compute-bound: minutes per hour of audio |
| 16 GB Macs, 24–32 GB GPUs | no | — | admission refuses with a clear message |

CUDA needs compute capability 8.0 or newer. The L2 generator's Leg A needs
about 40 GB of GPU memory with the FP8 expert codes resident (`--expert-store
device`), about 13 GB with `--expert-store lazy` (slower), or a 64 GB+ CPU box.

## Prerequisites

* git, curl, ffmpeg, Python 3.9+ (the kit itself uses only the standard library).
* Rust (the workspace's `rust-version`, 1.99 or newer); about 25 GB free for
  the build and 36 GB for the weights (`fetch.sh` and `build.sh` check `df`).
* CUDA: the toolkit with `nvcc` on `PATH`; `CUDA_COMPUTE_CAP` is read from
  `nvidia-smi` when unset.
* For the generators only (GPU box, or any box for L1): torch 2.5+, numpy,
  safetensors, tokenizers, omegaconf, transformers main with native NemotronH
  (bb666bd0 or newer), NeMo Speech main (743beaf4 pinned by the plan; the
  generators record what ran). Leg B needs the card's environment: vLLM
  0.32.0 + NeMo Speech main in one venv (see the model card's "Quantized
  Runtime and Environment"); pass its interpreter with `--vllm-python`.

Everything the kit writes goes to `target/nemotron-transcribe/<backend>/`
(override with `NT_OUT_ROOT`). Weights go to izwi's models directory
(`IZWI_MODELS_DIR`, default `~/.local/share/izwi/models` on Linux and
`~/Library/Application Support/izwi/models` on macOS) as
`Nemotron-3.5-Transcribe-30B-A4B-FP8/`.

## One command per backend

Run from the repository root, at the commit you were asked to test.

```bash
K=scripts/e2e/nemotron-transcribe

# Once per machine: the weights (resumable; rerun after any interruption).
bash $K/fetch.sh

# CUDA box
bash $K/build.sh cuda && bash $K/run-parity.sh cuda --both-modes && \
  bash $K/run-e2e.sh cuda && bash $K/run-load.sh cuda && bash $K/collect.sh cuda

# Mac (64 GB or more)
bash $K/build.sh metal && bash $K/run-parity.sh metal && \
  bash $K/run-e2e.sh metal && bash $K/run-load.sh metal && bash $K/collect.sh metal

# CPU box (64 GB RAM or more)
bash $K/build.sh cpu && bash $K/run-parity.sh cpu && \
  bash $K/run-e2e.sh cpu --skip-long && bash $K/run-load.sh cpu && bash $K/collect.sh cpu
```

`run-parity.sh` needs `crates/izwi-core/tests/fixtures/nemotron_transcribe/reference.json`
and its `reference-audio/` from the L2 generator (below). Without them, run
the other steps and say so when you send results back.

WER-mini (any backend). The acceptance gate is izwi within +0.3 absolute of
Leg B on the same machine, so on the CUDA box score vLLM first, then izwi
with `--compare`; elsewhere run the izwi line alone (it still reports the
card's numbers beside its own):

```bash
D=target/nemotron-transcribe/wer-data
python3 $K/wer-mini.py prepare --out $D
# Leg B, with `vllm serve` running as in the model card minus --speculative-config:
python3 $K/wer-mini.py run --engine vllm --base-url http://127.0.0.1:8000 \
  --vllm-model <served name> --manifest $D/manifest.jsonl \
  --output target/nemotron-transcribe/cuda/wer-vllm.json
# izwi (stop vLLM first; this starts izwi-server and loads the model):
python3 $K/wer-mini.py run --manifest $D/manifest.jsonl \
  --backend cuda --binary target/nemotron-transcribe/cuda/bin/izwi-server \
  --models-dir "$IZWI_MODELS_DIR" --out target/nemotron-transcribe/cuda \
  --output target/nemotron-transcribe/cuda/wer-izwi.json \
  --compare target/nemotron-transcribe/cuda/wer-vllm.json \
  --report target/nemotron-transcribe/cuda/report.json
```

## The fixture generators

L1, once, on any box (CPU only, a few minutes; about 4–5 MB of output):

```bash
python3 -I scripts/fixtures/generate_nemotron_transcribe_tiny.py
# writes crates/izwi-core/tests/fixtures/nemotron_transcribe/tiny/
```

It records whether NeMo's perception ran or the SGLang-Omni fallback
(`audio_reference` in `expected.json`); if the fallback ran, also run it once
with `--audio-impl nemo` on the GPU box and send both.

L2, on the GPU box. Leg B (vLLM) and Leg A (F32 reference) can run in
different environments; `--update` merges them into one file:

```bash
M="$IZWI_MODELS_DIR/Nemotron-3.5-Transcribe-30B-A4B-FP8"
python3 -I scripts/fixtures/generate_nemotron_transcribe_reference.py --model-dir "$M" \
  --leg-b --skip-leg-a --vllm-python /path/to/nemo-speech/.venv/bin/python
python3 -I scripts/fixtures/generate_nemotron_transcribe_reference.py --model-dir "$M" --update
# writes crates/izwi-core/tests/fixtures/nemotron_transcribe/reference.json
#    and crates/izwi-core/tests/fixtures/nemotron_transcribe/reference-audio/
```

Leg B's GPU decides what it measured: SM89+ (L40S, H100, Blackwell) runs W8A8
with an FP8 KV cache, so compare it with izwi's
`IZWI_NEMOTRON_TRANSCRIBE_FP8_MODE=vllm-fp8` (`run-parity.sh --both-modes`);
SM80/86 (A100, A6000) runs Marlin W8A16, izwi's default. Both generators exit
with status 3 when a self-check fails; send the output anyway.

## What each script does

| Script | Does | Writes (under `target/nemotron-transcribe/<backend>/`) |
|---|---|---|
| `fetch.sh [--dest DIR] [--jobs 1\|2] [--verify-only]` | resumable download at the pin, a curl loop per file appending in place, at most 2 connections, exact size check | the model directory |
| `build.sh <backend>` | release `izwi-server` with a private `CARGO_TARGET_DIR`, `IZWI_BUILD_GIT_SHA=HEAD`, `CARGO_INCREMENTAL=0`, a `df` check; prints the binary size and sha256 | `bin/izwi-server`, `build-info.json`, `build.log` |
| `run-parity.sh <backend> [--both-modes] [--skip-unit]` | `cargo test -p izwi-core --release --features <metal\|cuda> --lib -- nemotron_transcribe::reference_tests --ignored` (plus the family's other tests) with `IZWI_NEMOTRON_TRANSCRIBE_{MODEL_DIR,AUDIO_DIR,BACKEND}`; checks the audio's sha256 against reference.json first | `parity-*.log`, `parity.json` |
| `run-e2e.sh <backend> [options]` | starts the server (`RUST_LOG=info`, JSON logs), refuses a binary whose logged `build_git_sha` is not the built one, then the plan §10.4 matrix | `report.json` (`e2e`), `server-*.log`, `snapshots/`, `audio/` |
| `run-load.sh <backend> [options]` | `scripts/bench/run-audio-load.py --require-parity` at c = 1, 2, 4 (cpu, metal) or 1…32 (cuda), mixed 2–95 s clips, Poisson arrivals; chat ITL beside a 1-hour job with `NT_CHAT_MODEL` | `load-c*.json`, `report.json` (`load`) |
| `wer-mini.py prepare\|run` | 100 LibriSpeech test-clean + 100 FLEURS (en, de, fr, es, ja), WER/CER beside the card's numbers; `--compare` gates against Leg B | `wer-*.json`, `report.json` (`wer_*`) |
| `collect.sh <backend> [--base-url URL]` | one tarball with the report, snapshots, timings and log tails | `nemotron-transcribe-report-<backend>-<sha>.tar.gz` |
| `nt_harness.py` | the Python behind run-e2e, run-load and the report merging (stdlib only) | — |

E2E cases (`run-e2e.sh --only a,b` / `--skip a,b`): `load`, `format_json`,
`format_verbose_json`, `format_srt`, `format_vtt`, `stream_sse`, `language`,
`language_refused`, `prompt_override`, `keywords`, `translate_en_de`,
`translate_de_en` (needs `--de-clip`), `translate_refused`,
`translate_with_timestamps_refused`, `word_timestamps`, `speaker_attributed`,
`one_model_diarization` (a diarization job naming the model alone; a pipeline ASR beside it must be refused), `cancel_mid_prefill`,
`cancel_mid_decode`, `cancel_by_disconnect`, `long_60min` (`--skip-long` on
the CPU), `soak_untracked_memory`, `unload_reload`,
`oom_refusal_undersized_budget` (a second server with an 8 GiB
`IZWI_<BACKEND>_MEMORY_BUDGET_BYTES`). Cases for phases that have not landed
yet (word timestamps, speaker attribution) fail with the server's message;
that is expected until they land.

Useful environment: `IZWI_NEMOTRON_TRANSCRIBE_FP8_MODE` (`w8a16` default,
`vllm-fp8`), `NT_MODELS_DIR`, `NT_OUT_ROOT`, `NT_CARGO_TARGET_DIR`,
`CUDA_COMPUTE_CAP`, `NT_LOAD_CONCURRENCY`, `NT_CHAT_MODEL`, `NT_PYTHON`.

## Sending results back

Per backend, send `target/nemotron-transcribe/<backend>/nemotron-transcribe-report-<backend>-<sha>.tar.gz`
(`collect.sh` prints its path and a one-line summary per section). From the
fixture runs, also send:

* L1: the whole `crates/izwi-core/tests/fixtures/nemotron_transcribe/tiny/` directory;
* L2: `reference.json`, `vllm-leg-b.log` (beside it) and `reference-audio/`
  (about 90 MB; izwi's tests check its sha256 against reference.json).

Say which machine each came from (GPU and SM, Mac model and RAM, CPU and
RAM) if `host.txt` in the tarball does not already show it.
