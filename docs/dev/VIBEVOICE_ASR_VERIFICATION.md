# VibeVoice ASR Verification Notes

Date: 2026-05-29

## Scope

This note records the Phase 4 verification state for the native Rust/Candle
VibeVoice ASR pipeline. The implementation covers processor parity, request
generation controls, structured output parsing, and long-audio tokenizer
streaming. The remaining work is full real-model smoke completion on each
backend and profiling-driven kernel work.

## Implemented Runtime Path

- The ASR loader uses the local VibeVoice-ASR safetensors artifacts and Qwen
  tokenizer files.
- Audio preprocessing reads `preprocessor_config.json`, resamples to the
  configured sample rate, normalizes to the configured target dBFS, and pads
  audio to the speech-token placeholder contract.
- Prompt construction follows the VibeVoice ASR object-reference placeholder
  shape.
- Generation accepts request max-token limits and stop controls through the
  ASR runtime and OpenAI transcription endpoint.
- VibeVoice rich transcription output is parsed into plain transcript text plus
  segment/timestamp diagnostics.
- Long audio uses 60-second tokenizer streaming chunks before final-only LM
  decode.

## Device Policy

- CPU: F32 model dtype. This avoids BF16/F16 CPU compatibility issues and
  matches the existing device policy tests.
- Metal: F32 model dtype. This keeps Apple Silicon execution on the same
  numerically conservative path as CPU until profiles justify splitting dtypes.
- CUDA: VibeVoice family policy selects BF16 when supported, then F16, then
  F32. `IZWI_VIBEVOICE_ASR_DTYPE` remains validated by
  `select_model_dtype_checked`.

If CUDA profiling finds tokenizer instability in reduced precision, split the
policy before adding kernels: keep the Qwen decoder in BF16/F16 and keep the
acoustic tokenizer, semantic tokenizer, and connectors in F32.

## Local Verification

Commands that passed after the implementation phases:

```bash
cargo test -p izwi-core vibevoice --lib
cargo test -p izwi-server transcriptions --lib
git diff --check
```

The local VibeVoice-ASR artifacts were present under:

```text
/Users/lennex/Library/Application Support/izwi/models/VibeVoice-ASR
```

The checkpoint is a large BF16 safetensors model, approximately 17 GB on disk.
On CPU, Candle converts the BF16 tensors to F32 at load time, so memory use is
substantially larger than the on-disk size.

CPU smoke status:

- `izwi-server --backend cpu` started successfully.
- `/livez` returned HTTP 200.
- A short `/v1/audio/transcriptions` request for `VibeVoice-ASR` entered
  `VibeVoiceAsrModel::load` and the Qwen model safetensors load path.
- The run was interrupted before transcript generation completed. A sample from
  the interrupted process showed 18.5 GB physical footprint while loading Qwen
  layer weights through Candle safetensors conversion.

No `izwi-server` process was left running after the interrupted smoke.

## Smoke Commands

CPU:

```bash
cargo run -p izwi-server -- --backend cpu --host 127.0.0.1 --port 4975
curl -sS http://127.0.0.1:4975/livez
curl -sS \
  -F file=@data/fox.wav \
  -F model=VibeVoice-ASR \
  -F response_format=verbose_json \
  -F max_tokens=8 \
  http://127.0.0.1:4975/v1/audio/transcriptions
```

Metal:

```bash
cargo run -p izwi-server -- --backend metal --host 127.0.0.1 --port 4976
curl -sS http://127.0.0.1:4976/livez
curl -sS \
  -F file=@data/fox.wav \
  -F model=VibeVoice-ASR \
  -F response_format=verbose_json \
  -F max_tokens=8 \
  http://127.0.0.1:4976/v1/audio/transcriptions
```

CUDA source build:

```bash
cargo build -p izwi-server --release --features cuda,cudnn
./target/release/izwi-server --backend cuda --host 127.0.0.1 --port 4977
curl -sS http://127.0.0.1:4977/livez
curl -sS \
  -F file=@data/fox.wav \
  -F model=VibeVoice-ASR \
  -F response_format=verbose_json \
  -F max_tokens=8 \
  http://127.0.0.1:4977/v1/audio/transcriptions
```

CUDA Docker:

```bash
CUDA_COMPUTE_CAP=80 docker compose --profile cuda up --build izwi-cuda
curl -sS http://127.0.0.1:8080/livez
curl -sS \
  -F file=@data/fox.wav \
  -F model=VibeVoice-ASR \
  -F response_format=verbose_json \
  -F max_tokens=8 \
  http://127.0.0.1:8080/v1/audio/transcriptions
```

For long-audio tokenizer streaming coverage, repeat the transcription command
with a file longer than 60 seconds and inspect diagnostics for:

```text
audio.tokenizer_streaming=true
audio.tokenizer_streaming_chunks>1
```

Useful diagnostics fields for all backend smokes:

```text
device.device_kind
device.model_dtype
decode.generated_tokens
decode.stop_reason
audio.tokenizer_streaming
audio.tokenizer_streaming_chunks
output.segment_count
```

## Kernel Follow-Up

Do not add backend-specific kernels until real CPU, Metal, and CUDA smokes are
correct and profiles identify a specific bottleneck. The first profiles should
separate:

- Qwen decoder prefill and token decode.
- Acoustic and semantic tokenizer Conv1d blocks.
- Transposed-conv overlap in tokenizer streaming.
- Connector linears and rank-3 layout conversions.
- KV-cache append and attention implementation.

Potential kernel work, if profiling proves Candle overhead remains:

- Fused tokenizer block norm, depthwise convolution, and FFN pieces.
- Backend-specific conv layout helpers for tokenizer stages.
- Preallocated dense KV append for final-only ASR decode.
- Narrow fused activation kernels for connector or tokenizer MLPs.
