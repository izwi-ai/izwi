# Audio Processing Contract

Date: 2026-06-02

This contract defines the production boundary for audio work in Izwi. It is
paired with `docs/audio-processing-contract.json`, which is parsed by tests so
the route-level contract stays executable.

## Boundary

Shared audio infrastructure may:

- accept JSON and multipart audio payloads;
- normalize raw base64, data URLs, and ASCII whitespace;
- capture source MIME, filename, byte count, decoded sample rate, channel count,
  duration, and validation status;
- decode user audio into finite mono PCM samples when a route needs decoded
  audio;
- report user-facing validation errors before long-running jobs start.

Shared audio infrastructure must not:

- change model-specific feature extraction by default;
- replace model-specific codec paths by default;
- alter model-owned reference-audio preprocessing by default;
- silently change model target sample rates;
- globally replace resamplers in model adapters without fixture-backed proof.

Model adapters keep ownership of their audio details. Whisper, Qwen, Parakeet,
Sortformer, VibeVoice, Voxtral, Kokoro, LFM, and future model families receive
the decoded PCM plus original sample rate unless an existing route intentionally
passes encoded bytes.

## Ingress Policy

Supported payload shapes are:

- `json.audio_base64`;
- `multipart.file`;
- `multipart.audio`;
- `multipart.audio_base64`;
- route-specific reference audio fields for TTS and saved voices.

Base64 parsing should be uniform across routes: raw base64, data URLs, and ASCII
whitespace are accepted; empty or invalid audio is rejected as a bad request.
MIME and filename are hints until decoded/sniffed metadata confirms the payload.

OpenAI audio routes keep the current upload limit policy: 25 MiB by default,
64 MiB in relaxed compatibility mode or when configured. First-party audio
uploads stay capped at 64 MiB while multipart ingestion remains memory-backed.

## Decode Policy

The current decode policy is Symphonia first, Hound WAV fallback, mono downmix
by channel averaging, finite-sample sanitization, and clamp to `[-1, 1]`.

Decoded metadata should be available before job queueing whenever possible:
source bytes, source MIME, decoded MIME or codec, sample rate, channels,
duration, peak/RMS, clipping count, and validation reason.

## Realtime Policy

Realtime transcription uses `/v1/speech-to-text/realtime/ws` with `ITRW` PCM16
binary frames. Realtime voice uses `/v1/voice/realtime/ws` with `IVWS` PCM16
binary frames. Existing frame names and event names are compatibility contracts
unless a dedicated contract-alignment phase changes the docs and code together.

## Output Policy

WAV and raw PCM remain the stable native output formats. Compressed outputs
such as MP3, Opus, AAC, Ogg, and FLAC are additive only; unsupported compressed
formats are rejected unless explicit WAV fallback is requested. Encoders should
use the actual generated sample rate from the model result when available.

## Verification

Every phase that changes audio behavior needs tests at the boundary it touches:
payload parsing, decode metadata, format fixtures, resampling, realtime
protocols, TTS sample-rate headers, CLI behavior, and OpenAPI/docs alignment.
