# Inference Capability Inventory

Date: 2026-05-09

## Purpose

This document is the Phase 0 implementation contract for the full inference
engine migration. It records the current model variants, their capabilities, and
the public surfaces that must continue working while the architecture is moved
behind `RuntimeService`.

No runtime behavior is changed by this phase.

## Capability Names

These canonical capability names should be used by typed runtime requests,
capability adapters, broker metrics, conformance fixtures, and documentation.

| Capability | Meaning | Current examples |
| --- | --- | --- |
| `tts` | Text or prompt input to generated audio output. | Qwen3-TTS, Kokoro, LFM2.5 Audio TTS path. |
| `streaming_tts` | TTS with progressive audio chunks. | Qwen3-TTS, Kokoro. |
| `asr` | Audio input to transcript output. | Parakeet, Whisper, Qwen3 ASR, audio-chat transcription path. |
| `realtime_asr` | Session-oriented ASR with partial/final updates. | Realtime transcription websocket, Voxtral realtime when enabled. |
| `chat` | Text/multimodal prompt to generated text output. | Qwen3, Qwen3.5, Gemma, LFM2.5 chat. |
| `audio_chat` | Audio/text prompt to audio-language model output. | LFM2.5 Audio. |
| `speech_to_speech` | Audio input to generated audio/text response. | LFM2.5 Audio unified voice path. |
| `diarization` | Audio input to speaker-attributed segments. | Sortformer. |
| `forced_alignment` | Transcript/audio input to word timestamps. | Qwen3 ForcedAligner. |
| `vad` | Audio input to speech activity events. | Voice realtime VAD stage. |
| `endpointing` | Realtime speech turn boundary detection. | Voice realtime endpointing stage. |
| `tokenizer` | Codec/tokenizer artifact used by other capabilities. | Qwen3-TTS Tokenizer 12Hz. |

## Model Variant Inventory

Every current `ModelVariant::all()` entry is listed here so implementation
phases can verify registry coverage without changing public behavior.

| Variant | Family | Primary task | Runtime capabilities | Enabled |
| --- | --- | --- | --- | --- |
| `Qwen3Tts12Hz06BBase` | Qwen3Tts | TTS | `tts`, `streaming_tts` | Yes |
| `Qwen3Tts12Hz06BBase4Bit` | Qwen3Tts | TTS | `tts`, `streaming_tts` | Yes |
| `Qwen3Tts12Hz06BBase8Bit` | Qwen3Tts | TTS | `tts`, `streaming_tts` | No |
| `Qwen3Tts12Hz06BBaseBf16` | Qwen3Tts | TTS | `tts`, `streaming_tts` | No |
| `Qwen3Tts12Hz06BCustomVoice` | Qwen3Tts | TTS | `tts`, `streaming_tts` | Yes |
| `Qwen3Tts12Hz06BCustomVoice4Bit` | Qwen3Tts | TTS | `tts`, `streaming_tts` | Yes |
| `Qwen3Tts12Hz06BCustomVoice8Bit` | Qwen3Tts | TTS | `tts`, `streaming_tts` | No |
| `Qwen3Tts12Hz06BCustomVoiceBf16` | Qwen3Tts | TTS | `tts`, `streaming_tts` | No |
| `Qwen3Tts12Hz17BBase` | Qwen3Tts | TTS | `tts`, `streaming_tts` | Yes |
| `Qwen3Tts12Hz17BBase4Bit` | Qwen3Tts | TTS | `tts`, `streaming_tts` | Yes |
| `Qwen3Tts12Hz17BCustomVoice` | Qwen3Tts | TTS | `tts`, `streaming_tts` | Yes |
| `Qwen3Tts12Hz17BCustomVoice4Bit` | Qwen3Tts | TTS | `tts`, `streaming_tts` | Yes |
| `Qwen3Tts12Hz17BVoiceDesign` | Qwen3Tts | TTS | `tts`, `streaming_tts` | Yes |
| `Qwen3Tts12Hz17BVoiceDesign4Bit` | Qwen3Tts | TTS | `tts`, `streaming_tts` | Yes |
| `Qwen3Tts12Hz17BVoiceDesign8Bit` | Qwen3Tts | TTS | `tts`, `streaming_tts` | No |
| `Qwen3Tts12Hz17BVoiceDesignBf16` | Qwen3Tts | TTS | `tts`, `streaming_tts` | No |
| `Qwen3TtsTokenizer12Hz` | Tokenizer | Tokenizer | `tokenizer` | Yes |
| `Lfm2512BInstructGguf` | Lfm2Chat | Chat | `chat` | Yes |
| `Lfm2512BThinkingGguf` | Lfm2Chat | Chat | `chat` | Yes |
| `Lfm25Audio15BGguf` | Lfm25Audio | AudioChat | `audio_chat`, `asr`, `speech_to_speech`, `tts` | Yes |
| `Kokoro82M` | KokoroTts | TTS | `tts`, `streaming_tts` | Yes |
| `ParakeetTdt06BV3` | ParakeetAsr | ASR | `asr` | Yes |
| `WhisperLargeV3Turbo` | WhisperAsr | ASR | `asr` | Yes |
| `Qwen3Asr06BGguf` | Qwen3Asr | ASR | `asr` | Yes |
| `Qwen3Asr17BGguf` | Qwen3Asr | ASR | `asr` | Yes |
| `Nemotron35AsrStreaming06B` | NemotronAsr | ASR | `asr` | Yes |
| `DiarStreamingSortformer4SpkV21` | SortformerDiarization | Diarization | `diarization` | Yes |
| `Qwen306B` | Qwen3Chat | Chat | `chat` | No |
| `Qwen306B4Bit` | Qwen3Chat | Chat | `chat` | No |
| `Qwen306BGguf` | Qwen3Chat | Chat | `chat` | Yes |
| `Qwen317B` | Qwen3Chat | Chat | `chat` | No |
| `Qwen317B4Bit` | Qwen3Chat | Chat | `chat` | No |
| `Qwen317BGguf` | Qwen3Chat | Chat | `chat` | Yes |
| `Qwen34BGguf` | Qwen3Chat | Chat | `chat` | Yes |
| `Qwen38BGguf` | Qwen3Chat | Chat | `chat` | Yes |
| `Qwen314BGguf` | Qwen3Chat | Chat | `chat` | No |
| `Qwen3508BGguf` | Qwen35Chat | Chat | `chat` | Yes |
| `Qwen352BGguf` | Qwen35Chat | Chat | `chat` | Yes |
| `Qwen354BGguf` | Qwen35Chat | Chat | `chat` | Yes |
| `Qwen359BGguf` | Qwen35Chat | Chat | `chat` | Yes |
| `Gemma31BIt` | Gemma3Chat | Chat | `chat` | Yes |
| `Gemma34BIt` | Gemma3Chat | Chat | `chat` | No |
| `Qwen3ForcedAligner06B` | Qwen3ForcedAligner | ForcedAlign | `forced_alignment` | Yes |
| `Qwen3ForcedAligner06B4Bit` | Qwen3ForcedAligner | ForcedAlign | `forced_alignment` | Yes |
| `VoxtralMini4BRealtime2602` | Voxtral | Asr | `asr` | Yes |

## Public Compatibility Surfaces

Implementation phases must preserve these surfaces unless an intentional
behavior change is approved separately.

| Surface | Required coverage |
| --- | --- |
| OpenAI TTS | `/v1/audio/speech`, binary response, streaming response, voice/reference options, response format mapping. |
| First-party TTS | Speech history creation, background generation, long-form chunking, SSE generation, saved voice resolution. |
| OpenAI ASR | `/v1/audio/transcriptions`, JSON and multipart inputs, `json`, `text`, `srt`, `vtt`, and verbose-style outputs. |
| First-party ASR | Transcription jobs, background execution, summaries, export paths, realtime transcription websocket. |
| Chat/Responses | `/v1/chat/completions`, `/v1/responses`, streaming and non-streaming responses, media/tool normalization. |
| Realtime voice | Modular voice, unified audio-chat voice, barge-in, interruption, transcript events, audio events, turn persistence. |
| Diarization | First-party speech-to-text jobs with `job_kind=diarization`, ASR attribution, forced alignment, optional LLM refinement, exports. |
| Model ops | Model list, download, cancel download, load, unload, delete, lifecycle snapshots. |
| Metrics | `/v1/metrics`, `/v1/metrics/prometheus`, `/internal/*` aliases, existing runtime and voice counters. |
| CLI/UI | `tts`, `transcribe`, `diarize`, `chat`, bench commands, and existing UI routes for TTS, speech-text, diarization, chat, voice, and models. |

## Conformance Fixtures

These fixtures should be present before a phase changes execution behavior. When
the real model is optional or too large for CI, the fixture may use a fake
adapter/model handle, but the public contract should match the real path.

| Capability | Minimum fixture |
| --- | --- |
| `tts` | Short text, explicit model, voice option, binary response format, streaming chunk sequence. |
| `streaming_tts` | First audio chunk timing marker, multiple chunk delivery, terminal event/error handling. |
| `asr` | Short WAV input, bytes and path request forms, transcript text, language option, output format mapping. |
| `realtime_asr` | PCM frames, partial update, final update, cancellation/close behavior. |
| `chat` | Single-user prompt, multi-message prompt, streaming delta, terminal usage/event shape. |
| `audio_chat` | Audio bytes plus optional text prompt, transcript mode, speech-to-speech mode. |
| `speech_to_speech` | Audio request to streaming audio response with cancellation. |
| `diarization` | Short multi-speaker fixture, segment speaker labels, ASR attribution, optional refinement disabled/enabled. |
| `forced_alignment` | Transcript plus audio fixture producing ordered word timestamps. |
| `vad` | Silence/speech/silence frames producing stable activity transitions. |
| `endpointing` | Realtime speech frames producing one completed turn boundary. |
| `tokenizer` | Artifact availability and dependent TTS model load path. |

## Phase 0 Exit Criteria

- Architecture research plan exists and includes regression controls.
- Every current model variant has an inventory row.
- Canonical capability names are documented.
- Public compatibility surfaces are documented.
- Minimum conformance fixture expectations are documented.
- No production runtime behavior changed.
