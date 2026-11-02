# Fish Audio Drama 3: hand-off test kit

Scripts that check izwi's Fish Audio Drama 3 support on machines big enough
to hold it, and pack the results into one file to send back. Each layer must
pass before the next is worth running.

| Layer | What | Where | Command |
|---|---|---|---|
| L1 | Upstream tiny fixture (no weights, no GPU) | any box with Python and CPU torch | `scripts/fixtures/generate_fish_drama3_tiny.py` |
| L1.5 | Real-weight slices (about 4.2 GB of downloads) | any box with Python and CPU torch, under 8 GB of RAM | `scripts/fixtures/fetch_fish_drama3_slices.py`, then `generate_fish_drama3_slices.py` |
| L2 | Whole-model reference (Leg A) | a GPU with 80 GB or more, or a CPU box with 64 GB or more | `scripts/fixtures/generate_fish_drama3_reference.py` |
| L1–L2 | izwi's parity suites against those fixtures | every backend | `run-parity.sh <backend>` |
| L3 | HTTP end to end, load parity, eval-mini | every backend | `run-e2e.sh`, `run-load.sh`, `eval-mini.py` |

The generators and their outputs are described in
`crates/izwi-core/tests/fixtures/fish_drama3/README.md`; this kit runs izwi
against what they wrote.

Model: [`fishaudio/drama-3`](https://huggingface.co/fishaudio/drama-3) at
revision `78e5ab79e49aa9a25abe8799eff2c4d73a1c3216`, 41.5 GB in 24 files.
izwi's variant id is `FishAudio-Drama-3`. The weights are licensed under the
Fish Audio Community License v1.0: research and non-commercial use, and
commercial use below USD 500,000 in annual revenue; offering the model as a
hosted service needs a licence from Fish Audio.

## Hardware

About 37 GiB resident on every backend (raw FP8 experts on Metal and CUDA,
Q8_0 on the CPU), plus load transients of about 3 GiB. Admission refuses a
machine that cannot hold it before any weight is read.

| Machine | Fits? | Rows decoded together | Notes |
|---|---|---|---|
| H100 / H200 80+ GB, RTX PRO 6000 96 GB | yes | 8 | the card's serving numbers are from one H200 |
| L40S / RTX 6000 Ada / A6000 48 GB | yes | 8 | about 7 GiB left for rows |
| A100 80 GB | yes | 8 | SM80: the minimum |
| Apple silicon with 64 GB or more | yes | 1 (2 with `IZWI_FISH_DRAMA3_NATIVE_BATCHING=true`) | a 64 GB Mac has about 48 GiB of GPU working set; 96 or 128 GB is comfortable |
| CPU box with 64 GB of RAM or more | yes | 1 | FP8 experts become Q8_0 at load |
| Apple 48 GB, 24–40 GB GPUs, GPUs older than SM80 | no | — | refused with a message that names the memory it needs |

The L2 generator's Leg A needs a GPU with 80 GB or more, or a CPU box with
64 GB or more. The L2 parity test's CPU forms need about 40 GiB of RAM
(`cpu-q8_0`, the default) or about 140 GiB (`cpu`, F32 weights).

## Prerequisites

* git, curl, ffmpeg (eval-mini only), Python 3.9+ (the kit itself uses only
  the standard library).
* Rust (the workspace's `rust-version`, 1.99 or newer); about 25 GB free for
  each backend's build and 42 GB for the weights (`fetch.sh` and `build.sh`
  check `df`).
* CUDA: the toolkit with `nvcc` on `PATH`; `CUDA_COMPUTE_CAP` is read from
  `nvidia-smi` when unset.
* For eval-mini's scoring only: `faster-whisper` or `openai-whisper` (Whisper
  large-v3), `speechbrain` with `torchaudio` (ECAPA speaker similarity) and,
  optionally, `whisper_normalizer`; LibriSpeech test-clean (downloaded from
  openslr by default) and a local copy of the seed-tts-eval test set.

Everything the kit writes goes to `target/fish-drama3/<backend>/` (override
with `FD3_OUT_ROOT`). Weights go to izwi's models directory
(`IZWI_MODELS_DIR`, default `~/.local/share/izwi/models` on Linux and
`~/Library/Application Support/izwi/models` on macOS) as
`FishAudio-Drama-3/`. Every script takes `--help`, and all but eval-mini's
`score` take `--dry-run`.

## One command per backend

Run from the repository root, at the commit you were asked to test.

```bash
K=scripts/e2e/fish-drama3

# Once per machine: the weights (resumable; rerun after any interruption).
bash $K/fetch.sh

# CUDA box
bash $K/build.sh cuda && bash $K/run-parity.sh cuda && \
  bash $K/run-e2e.sh cuda && bash $K/run-load.sh cuda && bash $K/collect.sh cuda

# Mac (64 GB or more)
bash $K/build.sh metal && bash $K/run-parity.sh metal && \
  bash $K/run-e2e.sh metal && bash $K/run-load.sh metal && bash $K/collect.sh metal

# CPU box (64 GB of RAM or more)
bash $K/build.sh cpu && bash $K/run-parity.sh cpu && \
  bash $K/run-e2e.sh cpu && bash $K/run-load.sh cpu && bash $K/collect.sh cpu
```

`run-parity.sh` runs every leg whose fixture exists and reports the rest as
skipped (`incomplete`); `--require-all` makes a skipped leg fail. Without the
L1 fixture, the slices and the L2 reference it still runs the family's
suites, the device tests and the real tokenizer and codec checks.

`run-e2e.sh` takes about 45 minutes on a GPU: the soak alone is 30
(`--soak-minutes`). Add `--asr-model Whisper-Large-v3-Turbo` (or any ASR
model izwi has downloaded) to transcribe every take; without it the takes
are kept for listening and eval-mini scores the model.

Before spending GPU time, the kit can check itself on any machine against
the synthetic checkpoint (random tiny weights and a stub codec, written by an
izwi-server test; no weights, a few minutes):

```bash
bash $K/build.sh cpu --synthetic && bash $K/run-e2e.sh cpu --synthetic && \
  bash $K/run-load.sh cpu --synthetic
```

## eval-mini

100 LibriSpeech test-clean pairs and 25 + 25 Seed-TTS-eval items (en, zh),
each cloned from its reference, at the sampling defaults and at 0.7 / 0.7 /
30. Score izwi's WAVs, then the WAVs your own Leg B run made for the same
manifest (sglang-omni at `d9c3e708` or vllm-omni at `e45e43bc`, on the same
machine, cloning each item's `reference_audio` with its `reference_text` and
writing `<id>.wav`); the gate is izwi's WER at most 0.5 points above Leg B's
and its similarity at most 0.02 below.

```bash
D=target/fish-drama3/eval
python3 $K/eval-mini.py prepare --out $D --seed-tts-dir /path/to/seedtts_testset \
  --librispeech-pc-list /path/to/librispeech_pc_test_clean_cross_sentence.lst
python3 $K/eval-mini.py synth --manifest $D/manifest.jsonl --wavs $D/izwi \
  --backend cuda --binary target/fish-drama3/cuda/bin/izwi-server \
  --models-dir "$IZWI_MODELS_DIR" --out target/fish-drama3/cuda
# Leg B: your sglang-omni or vllm-omni run writes $D/legb/<profile>/<id>.wav.
for P in defaults fish; do
  python3 $K/eval-mini.py score --manifest $D/manifest.jsonl --wavs $D/legb/$P \
    --label legb-$P --output target/fish-drama3/cuda/eval-legb-$P.json
  python3 $K/eval-mini.py score --manifest $D/manifest.jsonl --wavs $D/izwi/$P \
    --label izwi-$P --output target/fish-drama3/cuda/eval-izwi-$P.json \
    --compare target/fish-drama3/cuda/eval-legb-$P.json \
    --report target/fish-drama3/cuda/report.json
done
```

Without `--librispeech-pc-list` (F5-TTS publishes the LibriSpeech-PC
cross-sentence list), the LibriSpeech pairs come from the archive in order,
with its upper-case transcripts sentence-cased. Without `--seed-tts-dir` the
Seed-TTS sets are left out. On a machine without Leg B, score izwi alone and
say so.

## What each script does

| Script | Does | Writes (under `target/fish-drama3/<backend>/`) |
|---|---|---|
| `fetch.sh [--dest DIR] [--jobs 1\|2] [--verify-only] [--dry-run]` | resumable download at the pin, a curl loop per file appending in place, at most 2 connections, exact sizes, then every file's SHA-256 once and `izwi-artifact.json`, as izwi's downloader writes it after the same check | the model directory |
| `build.sh <backend> [--synthetic]` | release `izwi-server` with a private `CARGO_TARGET_DIR`, `IZWI_BUILD_GIT_SHA=HEAD`, `CARGO_INCREMENTAL=0`, a `df` check; prints the binary size and sha256 (`--synthetic`: a debug build with izwi-core's `test-support`, into `<backend>-synthetic/`) | `bin/izwi-server`, `build-info.json`, `build.log` |
| `run-parity.sh <backend> [--only/--skip legs] [--l2-cpu-form cpu\|cpu-q8_0] [--require-all]` | one release test build of izwi-core; legs `unit` (the family's suites; on CUDA the row GEMV and split decode kernels too), `devices` (the codec device tests), `tokenizer`, `codec` (the published files), `tiny` (L1), `slice` (L1.5) and `reference` (L2), with a missing Metal or CUDA device a failure | `parity-<leg>.log`, `parity.json` |
| `run-e2e.sh <backend> [options]` | starts the server (`RUST_LOG=info`, JSON logs), refuses a binary whose logged `build_git_sha` is not the built one, then the E2E cases below | `report.json` (`e2e`), `audio/`, `server-*.log`, `snapshots/` |
| `run-load.sh <backend> [options]` | `scripts/bench/run-audio-load.py --task tts --require-parity --tts-tolerance 0` at c = 1, 2 (cpu, metal) or 1…32 (cuda), seeded, 4–60 word inputs, Poisson arrivals; a streaming probe of TTFA, RTF and frames/s at each width beside the card's H200 table; chat ITL beside a long take with `FD3_CHAT_MODEL` | `load-c*.json`, `report.json` (`load`) |
| `eval-mini.py prepare\|synth\|score` | the item set; izwi's takes at both sampling profiles; WER/CER and ECAPA similarity of any WAV directory, `--compare` gating against Leg B | `eval-*.json`, `report.json` (`eval_*`) |
| `collect.sh <backend> [--base-url URL] [--eval-wavs]` | one tarball with the report, snapshots, timings, log tails and the E2E WAVs | `fish-drama3-report-<backend>-<sha>.tar.gz` |
| `fd3_harness.py` | the Python behind run-e2e, run-load, eval-mini's synth and the report merging (stdlib only) | — |

E2E cases (`run-e2e.sh --only a,b` / `--skip a,b`): `load` (the licence is
logged), `plain` (a voice the seed sets; the same seed twice is the same
take), `reference_inline` (`data/fox.wav` with its transcript; without it,
400), `reference_saved_voice`, `tags` (two lines open with a tag: listen for
the leading-silence quirk), `dialogue_two_speakers` (a script with a
reference per speaker, then each line alone in the same request shape, since
the stream does not mark where a speaker's chunk ends), `long_text_history`
(about five chunks with their history; listening samples where the chunks
meet, found from the stream's prefill pauses; stream equals WAV),
`stream_sse` (time to first audio; streamed PCM equals the WAV for one seed),
`formats`, `take_edit` (a durable take, then an edit through
`previous_take_ids`), `cancel_mid_preparation`, `cancel_mid_prefill`,
`cancel_mid_decode` (each `inconclusive` when the request had passed that
stage before the cancel landed), `cancel_after_first_audio`,
`soak_untracked_memory`, `fast_paths` (resident sampling on Metal; also the
row-invariant GEMV, chunked DeltaNet and grouped MoE on CUDA),
`asr_round_trip`, `unload_reload`, `oom_refusal_undersized_budget` (a second
server with an 8 GiB `IZWI_<BACKEND>_MEMORY_BUDGET_BYTES`).

Useful environment: `IZWI_FISH_DRAMA3_NATIVE_BATCHING`,
`IZWI_FISH_DRAMA3_MOE_BACKEND`, `IZWI_FISH_DRAMA3_DTYPE` (each reaches the
server and is recorded in the report), `FD3_MODELS_DIR`, `FD3_OUT_ROOT`,
`FD3_CARGO_TARGET_DIR`, `CUDA_COMPUTE_CAP`, `FD3_LOAD_CONCURRENCY`,
`FD3_CHAT_MODEL`, `FD3_PYTHON`, `HF_TOKEN`.

## Sending results back

Per backend, send `target/fish-drama3/<backend>/fish-drama3-report-<backend>-<sha>.tar.gz`
(`collect.sh` prints its path and a one-line summary per section). From the
fixture runs, also send:

* L1: the whole `crates/izwi-core/tests/fixtures/fish_drama3/tiny/` directory;
* L1.5: `slice_expected.json` and the generator's log (the slices themselves
  stay on your machine);
* L2: `reference.json` and `reference.safetensors` from the generator's
  output directory, and its log.

Say which machine each came from (GPU and SM, Mac model and RAM, CPU and
RAM) if `host.txt` in the tarball does not already show it.
