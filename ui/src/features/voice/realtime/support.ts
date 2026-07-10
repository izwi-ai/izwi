import type { ModelInfo } from "@/api";
import { createUuid } from "@/lib/ids";
import { isKokoroVariant } from "@/types";

export type RuntimeStatus =
  | "idle"
  | "listening"
  | "user_speaking"
  | "processing"
  | "assistant_speaking";

export type VoiceRealtimeMode = "modular" | "unified";
export const VOICE_REALTIME_PROTOCOL = "voice_realtime" as const;
export const VOICE_REALTIME_VERSION = 2 as const;

export type VoiceRealtimeServerEvent =
  | { type: "connected"; protocol: string; server_time_ms?: number }
  | {
      type: "session_ready";
      protocol: string;
      session_id: string;
      owner_instance_id: string;
      connection_epoch: number;
      resumable: false;
      resume_window_ms: 0;
    }
  | {
      type: "input_stream_ready";
      vad?: {
        backend?: string;
        threshold?: number;
        score_sample_rate?: number;
        score_frame_ms?: number;
        min_speech_ms?: number;
        silence_duration_ms?: number;
      };
    }
  | { type: "input_stream_stopped" }
  | { type: "user_speech_start"; utterance_id: string; utterance_seq: number }
  | { type: "user_speech_rejected"; utterance_id: string; utterance_seq: number }
  | {
      type: "user_speech_end";
      utterance_id: string;
      utterance_seq: number;
      reason?:
        | "silence"
        | "max_duration"
        | "stream_stopped"
        | "client_pause";
    }
  | { type: "turn_processing"; utterance_id: string; utterance_seq: number }
  | {
      type: "user_transcript_start";
      utterance_id: string;
      utterance_seq: number;
    }
  | {
      type: "user_transcript_delta";
      utterance_id: string;
      utterance_seq: number;
      delta: string;
    }
  | {
      type: "user_transcript_snapshot";
      utterance_id: string;
      utterance_seq: number;
      text: string;
    }
  | {
      type: "user_transcript_final";
      utterance_id: string;
      utterance_seq: number;
      text: string;
      language?: string | null;
      audio_duration_secs?: number;
    }
  | {
      type: "assistant_text_start";
      utterance_id: string;
      utterance_seq: number;
    }
  | {
      type: "assistant_text_final";
      utterance_id: string;
      utterance_seq: number;
      text: string;
      raw_text?: string;
    }
  | {
      type: "assistant_text_delta";
      utterance_id: string;
      utterance_seq: number;
      delta: string;
    }
  | {
      type: "assistant_text_snapshot";
      utterance_id: string;
      utterance_seq: number;
      text: string;
    }
  | {
      type: "assistant_audio_start";
      utterance_id: string;
      utterance_seq: number;
      sample_rate: number;
      audio_format: "pcm_i16" | "pcm_f32" | "wav";
    }
  | {
      type: "assistant_audio_done";
      utterance_id: string;
      utterance_seq: number;
    }
  | {
      type: "turn_interrupted";
      utterance_id: string;
      utterance_seq: number;
      reason?: string;
    }
  | {
      type: "turn_done";
      utterance_id: string;
      utterance_seq: number;
      status: "ok" | "error" | "timeout" | "interrupted" | "no_input";
      reason?: string;
    }
  | {
      type: "error";
      utterance_id?: string | null;
      utterance_seq?: number | null;
      message: string;
      code?: string;
      fatal?: boolean;
    }
  | { type: "pong"; timestamp_ms?: number; server_time_ms?: number };

type VoiceRealtimeV2Payload =
  | {
      type: "session_ready";
      data: {
        accepted_version: typeof VOICE_REALTIME_VERSION;
        owner_instance_id: string;
        resumable: false;
        resume_window_ms: 0;
      };
    }
  | { type: "session_started" }
  | {
      type: "audio_accepted";
      data: {
        frame_sequence: number;
        buffer_depth_samples: number;
        ingress_queue_depth: number;
      };
    }
  | {
      type: "audio_gap";
      data: {
        expected_frame_sequence: number;
        received_frame_sequence: number;
        missing_frames: number;
        action: "continue" | "reset_segment" | "close_session";
      };
    }
  | { type: "speech_started" }
  | {
      type: "speech_ended";
      data: {
        reason: "silence" | "max_duration" | "client_pause" | "stream_stopped";
      };
    }
  | {
      type: "transcript_partial" | "transcript_stable" | "transcript_correction";
      data: { text: string; revision: number; language?: string | null };
    }
  | {
      type: "transcript_final";
      data: { text: string; revision: number; language?: string | null };
    }
  | { type: "assistant_text_started" }
  | {
      type: "assistant_text_partial";
      data: { text: string; revision: number };
    }
  | { type: "assistant_text_final"; data: { text: string } }
  | {
      type: "assistant_audio_started";
      data: { sample_rate: number; channels: number; format: "pcm_i16" | "pcm_f32" };
    }
  | {
      type: "assistant_audio_completed";
      data: { last_chunk_sequence: number };
    }
  | {
      type: "interruption";
      data: {
        reason:
          | "client_request"
          | "barge_in"
          | "preempted_by_new_turn"
          | "backpressure"
          | "session_closing";
        cutoff_event_id: number;
        cutoff_sequence: number;
      };
    }
  | {
      type: "turn_completed";
      data: { status: "ok" | "no_input" | "interrupted" | "error" | "timeout" };
    }
  | {
      type: "recoverable_error";
      data: { code: string; message: string; retry_after_ms?: number | null };
    }
  | {
      type: "fatal_error";
      data: { code: string; message: string; close: RealtimeVoiceClose };
    }
  | { type: "closing" | "closed"; data: { close: RealtimeVoiceClose } }
  | {
      type: "pong";
      data: { client_timestamp_ms?: number | null; server_timestamp_ms: number };
    };

type RealtimeVoiceClose = {
  code: string;
  reason: string;
  message: string;
  retryable: boolean;
};

export type VoiceRealtimeV2ServerEnvelope = {
  protocol: typeof VOICE_REALTIME_PROTOCOL;
  version: typeof VOICE_REALTIME_VERSION;
  event_id: number;
  sequence: number;
  session_id: string;
  connection_epoch: number;
  timestamp_ms: number;
  utterance_id?: string;
  turn_id?: string;
  segment_id?: string;
} & VoiceRealtimeV2Payload;

export type VoiceRealtimeClientMessage =
  | {
      type: "session_start";
      system_prompt?: string;
      protocol?: typeof VOICE_REALTIME_PROTOCOL;
      version?: typeof VOICE_REALTIME_VERSION;
      resume_from_event_id?: number;
    }
  | {
      type: "input_stream_start";
      mode?: VoiceRealtimeMode;
      asr_model_id?: string;
      text_model_id?: string;
      tts_model_id?: string;
      s2s_model_id?: string;
      speaker?: string;
      asr_language?: string;
      max_output_tokens?: number;
      vad_threshold?: number;
      min_speech_ms?: number;
      silence_duration_ms?: number;
      max_utterance_ms?: number;
      pre_roll_ms?: number;
      input_sample_rate?: number;
    }
  | { type: "input_stream_stop" }
  | { type: "interrupt"; reason?: string }
  | { type: "ping"; timestamp_ms?: number };

const VOICE_WS_BIN_MAGIC = "IVWS";
const VOICE_WS_BIN_VERSION = 1;
const VOICE_WS_BIN_KIND_CLIENT_PCM16 = 1;
const VOICE_WS_BIN_KIND_ASSISTANT_PCM16 = 2;
const VOICE_WS_BIN_CLIENT_HEADER_LEN = 16;
const VOICE_WS_BIN_ASSISTANT_HEADER_LEN = 24;

export interface VoiceRealtimeAssistantAudioBinaryChunk {
  utteranceSeq: number;
  sequence: number;
  sampleRate: number;
  isFinal: boolean;
  pcm16Bytes: Uint8Array;
}

export interface TranscriptEntry {
  id: string;
  role: "user" | "assistant";
  text: string;
  timestamp: number;
}

export interface VoicePageProps {
  models: ModelInfo[];
  loading: boolean;
  downloadProgress: Record<
    string,
    {
      percent: number;
      currentFile: string;
      status: string;
      downloadedBytes: number;
      totalBytes: number;
    }
  >;
  onDownload: (variant: string) => void;
  onCancelDownload?: (variant: string) => void;
  onLoad: (variant: string) => void;
  onUnload: (variant: string) => void;
  onDelete: (variant: string) => void;
  onError?: (message: string) => void;
}

export const VOICE_AGENT_SYSTEM_PROMPT =
  "You are a helpful voice assistant. Reply with concise spoken-friendly language. Avoid markdown. Do not output <think> tags or internal reasoning. Return only the final spoken answer. Keep responses brief unless asked for details.";
export const VOICE_PIPELINE_LABEL = "Modular Voice Stack";
export const UNIFIED_VOICE_PIPELINE_LABEL = "Unified LFM2.5 Audio";

export const MODULAR_STACK_VARIANTS = {
  asr: "Parakeet-TDT-0.6B-v3",
  text: "Qwen3-1.7B-GGUF",
  tts: "Kokoro-82M",
} as const;

export function parseFinalAnswer(content: string): string {
  const openTag = "<think>";
  const closeTag = "</think>";
  let out = content;

  while (true) {
    const start = out.indexOf(openTag);
    if (start === -1) break;
    const end = out.indexOf(closeTag, start + openTag.length);
    if (end === -1) {
      out = out.slice(0, start);
      break;
    }
    out = `${out.slice(0, start)}${out.slice(end + closeTag.length)}`;
  }

  return out.trim();
}

export function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  if (bytes < 1024 * 1024 * 1024) {
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  }
  return `${(bytes / (1024 * 1024 * 1024)).toFixed(2)} GB`;
}

export function isAsrVariant(variant: string): boolean {
  return (
    variant.includes("Whisper-Large-v3-Turbo") ||
    variant.includes("Parakeet-TDT")
  );
}

export function isUnifiedAudioChatVariant(variant: string): boolean {
  return variant.trim() === "LFM2.5-Audio-1.5B-GGUF";
}

export function formatModelVariantLabel(variant: string): string {
  const normalized = variant
    .replace(/-4bit\b/g, "-4-bit")
    .replace(/-8bit\b/g, "-8-bit");

  if (normalized.startsWith("Parakeet-TDT-")) {
    return normalized.replace("Parakeet-TDT-", "Parakeet ");
  }

  if (normalized.startsWith("Whisper-Large-v3-Turbo")) {
    return "Whisper Large v3 Turbo";
  }

  if (normalized.startsWith("Qwen3-TTS-12Hz-")) {
    return normalized.replace("Qwen3-TTS-12Hz-", "TTS ");
  }

  if (normalized.startsWith("Qwen3-ForcedAligner-")) {
    return normalized.replace("Qwen3-ForcedAligner-", "ForcedAligner ");
  }

  if (normalized.startsWith("Qwen3-")) {
    return normalized.replace("Qwen3-", "Qwen3 ");
  }

  if (normalized.startsWith("Gemma-3-")) {
    return normalized
      .replace("Gemma-3-1b-it", "Gemma 3 1B Instruct")
      .replace("Gemma-3-4b-it", "Gemma 3 4B Instruct");
  }

  if (normalized === "LFM2.5-Audio-1.5B-GGUF") {
    return "LFM2.5 Audio 1.5B GGUF";
  }

  if (isKokoroVariant(normalized)) {
    return "Kokoro 82M";
  }

  return normalized.replace(/-/g, " ");
}

export function isRunnableModelStatus(status: ModelInfo["status"]): boolean {
  return status === "ready";
}

export function encodeWavPcm16(
  samples: Float32Array,
  sampleRate: number,
): Blob {
  const bytesPerSample = 2;
  const blockAlign = bytesPerSample;
  const byteRate = sampleRate * blockAlign;
  const dataSize = samples.length * bytesPerSample;
  const buffer = new ArrayBuffer(44 + dataSize);
  const view = new DataView(buffer);

  const writeString = (offset: number, value: string) => {
    for (let i = 0; i < value.length; i += 1) {
      view.setUint8(offset + i, value.charCodeAt(i));
    }
  };

  writeString(0, "RIFF");
  view.setUint32(4, 36 + dataSize, true);
  writeString(8, "WAVE");
  writeString(12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, byteRate, true);
  view.setUint16(32, blockAlign, true);
  view.setUint16(34, 16, true);
  writeString(36, "data");
  view.setUint32(40, dataSize, true);

  let offset = 44;
  for (let i = 0; i < samples.length; i += 1) {
    const clamped = Math.max(-1, Math.min(1, samples[i]));
    const int16 = clamped < 0 ? clamped * 0x8000 : clamped * 0x7fff;
    view.setInt16(offset, int16, true);
    offset += 2;
  }

  return new Blob([buffer], { type: "audio/wav" });
}

export function decodePcmI16Base64(base64Data: string): Float32Array {
  const binary = atob(base64Data);
  const sampleCount = Math.floor(binary.length / 2);
  const out = new Float32Array(sampleCount);

  for (let i = 0; i < sampleCount; i += 1) {
    const lo = binary.charCodeAt(i * 2);
    const hi = binary.charCodeAt(i * 2 + 1);
    let value = (hi << 8) | lo;
    if (value & 0x8000) {
      value -= 0x10000;
    }
    out[i] = value / 0x8000;
  }

  return out;
}

export function mergeSampleChunks(chunks: Float32Array[]): Float32Array {
  const totalSamples = chunks.reduce((sum, chunk) => sum + chunk.length, 0);
  const merged = new Float32Array(totalSamples);
  let offset = 0;
  for (const chunk of chunks) {
    merged.set(chunk, offset);
    offset += chunk.length;
  }
  return merged;
}

function encodeFloat32ToPcm16Bytes(samples: Float32Array): Uint8Array {
  const out = new Uint8Array(samples.length * 2);
  const view = new DataView(out.buffer);
  for (let i = 0; i < samples.length; i += 1) {
    const clamped = Math.max(-1, Math.min(1, samples[i]));
    const int16 = clamped < 0 ? clamped * 0x8000 : clamped * 0x7fff;
    view.setInt16(i * 2, int16, true);
  }
  return out;
}

export function encodeLiveMicPcm16(samples: Float32Array): Uint8Array {
  return encodeFloat32ToPcm16Bytes(samples);
}

export function encodeVoiceRealtimeClientPcm16Frame(
  pcm16Bytes: Uint8Array,
  sampleRate: number,
  frameSeq: number,
): Uint8Array {
  const frame = new Uint8Array(
    VOICE_WS_BIN_CLIENT_HEADER_LEN + pcm16Bytes.length,
  );
  frame[0] = VOICE_WS_BIN_MAGIC.charCodeAt(0);
  frame[1] = VOICE_WS_BIN_MAGIC.charCodeAt(1);
  frame[2] = VOICE_WS_BIN_MAGIC.charCodeAt(2);
  frame[3] = VOICE_WS_BIN_MAGIC.charCodeAt(3);
  frame[4] = VOICE_WS_BIN_VERSION;
  frame[5] = VOICE_WS_BIN_KIND_CLIENT_PCM16;
  frame[6] = 0;
  frame[7] = 0;
  const view = new DataView(frame.buffer);
  view.setUint32(8, sampleRate >>> 0, true);
  view.setUint32(12, frameSeq >>> 0, true);
  frame.set(pcm16Bytes, VOICE_WS_BIN_CLIENT_HEADER_LEN);
  return frame;
}

export function parseVoiceRealtimeAssistantAudioBinaryChunk(
  data: ArrayBuffer,
): VoiceRealtimeAssistantAudioBinaryChunk | null {
  if (data.byteLength < VOICE_WS_BIN_ASSISTANT_HEADER_LEN) {
    return null;
  }
  const bytes = new Uint8Array(data);
  if (
    String.fromCharCode(bytes[0], bytes[1], bytes[2], bytes[3]) !==
    VOICE_WS_BIN_MAGIC
  ) {
    return null;
  }
  const view = new DataView(data);
  const version = view.getUint8(4);
  const kind = view.getUint8(5);
  if (
    version !== VOICE_WS_BIN_VERSION ||
    kind !== VOICE_WS_BIN_KIND_ASSISTANT_PCM16
  ) {
    return null;
  }
  const flags = view.getUint16(6, true);
  const utteranceSeq = Number(view.getBigUint64(8, true));
  const sequence = view.getUint32(16, true);
  const sampleRate = view.getUint32(20, true);
  const pcm16Bytes = bytes.slice(VOICE_WS_BIN_ASSISTANT_HEADER_LEN);
  return {
    utteranceSeq,
    sequence,
    sampleRate,
    isFinal: (flags & 1) === 1,
    pcm16Bytes,
  };
}

export function decodePcmI16Bytes(pcm16Bytes: Uint8Array): Float32Array {
  const sampleCount = Math.floor(pcm16Bytes.length / 2);
  const out = new Float32Array(sampleCount);
  const view = new DataView(
    pcm16Bytes.buffer,
    pcm16Bytes.byteOffset,
    pcm16Bytes.byteLength,
  );
  for (let i = 0; i < sampleCount; i += 1) {
    out[i] = view.getInt16(i * 2, true) / 0x8000;
  }
  return out;
}

export function buildVoiceRealtimeWebSocketUrl(apiBaseUrl: string): string {
  const base = new URL(apiBaseUrl, window.location.origin);
  base.protocol = base.protocol === "https:" ? "wss:" : "ws:";
  base.pathname = `${base.pathname.replace(/\/$/, "")}/voice/realtime/ws`;
  base.search = "";
  base.hash = "";
  return base.toString();
}

export function buildVoiceRealtimeV2SessionStart(
  systemPrompt?: string,
): VoiceRealtimeClientMessage {
  return {
    type: "session_start",
    protocol: VOICE_REALTIME_PROTOCOL,
    version: VOICE_REALTIME_VERSION,
    ...(systemPrompt?.trim() ? { system_prompt: systemPrompt } : {}),
  };
}

export function isVoiceRealtimeV2ServerEnvelope(
  value: unknown,
): value is VoiceRealtimeV2ServerEnvelope {
  if (!value || typeof value !== "object") return false;
  const event = value as Record<string, unknown>;
  if (
    event.protocol !== VOICE_REALTIME_PROTOCOL ||
    event.version !== VOICE_REALTIME_VERSION ||
    !isNonNegativeSafeInteger(event.event_id) ||
    !isNonNegativeSafeInteger(event.sequence) ||
    typeof event.session_id !== "string" ||
    !event.session_id ||
    !isNonNegativeSafeInteger(event.connection_epoch) ||
    !isNonNegativeSafeInteger(event.timestamp_ms) ||
    typeof event.type !== "string"
  ) {
    return false;
  }
  if (event.utterance_id !== undefined && typeof event.utterance_id !== "string") {
    return false;
  }
  const data = event.data;
  const record = data && typeof data === "object" ? (data as Record<string, unknown>) : null;
  const hasTextRevision = () =>
    record !== null &&
    typeof record.text === "string" &&
    isNonNegativeSafeInteger(record.revision);

  switch (event.type) {
    case "session_ready":
      return (
        record !== null &&
        record.accepted_version === VOICE_REALTIME_VERSION &&
        typeof record.owner_instance_id === "string" &&
        record.resumable === false &&
        record.resume_window_ms === 0
      );
    case "session_started":
    case "speech_started":
    case "assistant_text_started":
      return data === undefined;
    case "audio_accepted":
      return (
        record !== null &&
        isNonNegativeSafeInteger(record.frame_sequence) &&
        isNonNegativeSafeInteger(record.buffer_depth_samples) &&
        isNonNegativeSafeInteger(record.ingress_queue_depth)
      );
    case "audio_gap":
      return (
        record !== null &&
        isNonNegativeSafeInteger(record.expected_frame_sequence) &&
        isNonNegativeSafeInteger(record.received_frame_sequence) &&
        isNonNegativeSafeInteger(record.missing_frames) &&
        ["continue", "reset_segment", "close_session"].includes(
          String(record.action),
        )
      );
    case "speech_ended":
      return (
        record !== null &&
        ["silence", "max_duration", "client_pause", "stream_stopped"].includes(
          String(record.reason),
        )
      );
    case "transcript_partial":
    case "transcript_stable":
    case "transcript_correction":
    case "transcript_final":
    case "assistant_text_partial":
      return hasTextRevision();
    case "assistant_text_final":
      return record !== null && typeof record.text === "string";
    case "assistant_audio_started":
      return (
        record !== null &&
        isPositiveSafeInteger(record.sample_rate) &&
        isPositiveSafeInteger(record.channels) &&
        ["pcm_i16", "pcm_f32"].includes(String(record.format))
      );
    case "assistant_audio_completed":
      return record !== null && isNonNegativeSafeInteger(record.last_chunk_sequence);
    case "interruption":
      return (
        record !== null &&
        [
          "client_request",
          "barge_in",
          "preempted_by_new_turn",
          "backpressure",
          "session_closing",
        ].includes(String(record.reason)) &&
        isNonNegativeSafeInteger(record.cutoff_event_id) &&
        isNonNegativeSafeInteger(record.cutoff_sequence)
      );
    case "turn_completed":
      return (
        record !== null &&
        ["ok", "no_input", "interrupted", "error", "timeout"].includes(
          String(record.status),
        )
      );
    case "recoverable_error":
      return (
        record !== null &&
        typeof record.code === "string" &&
        typeof record.message === "string"
      );
    case "fatal_error":
      return (
        record !== null &&
        typeof record.code === "string" &&
        typeof record.message === "string" &&
        isRealtimeVoiceClose(record.close)
      );
    case "closing":
    case "closed":
      return record !== null && isRealtimeVoiceClose(record.close);
    case "pong":
      return record !== null && isNonNegativeSafeInteger(record.server_timestamp_ms);
    default:
      return false;
  }
}

export function isValidVoiceRealtimeV2Successor(
  previous: VoiceRealtimeV2ServerEnvelope | null,
  current: VoiceRealtimeV2ServerEnvelope,
): boolean {
  if (previous === null) {
    return current.sequence === 0 && current.connection_epoch === 0;
  }
  if (
    current.session_id !== previous.session_id ||
    current.event_id <= previous.event_id ||
    current.connection_epoch < previous.connection_epoch
  ) {
    return false;
  }
  if (current.connection_epoch === previous.connection_epoch) {
    return current.sequence === previous.sequence + 1;
  }
  return (
    current.connection_epoch === previous.connection_epoch + 1 &&
    current.sequence === 0
  );
}

export function normalizeVoiceRealtimeV2Event(
  envelope: VoiceRealtimeV2ServerEnvelope,
): VoiceRealtimeServerEvent | null {
  const identity = typedVoiceIdentity(envelope);
  switch (envelope.type) {
    case "session_ready":
      return {
        type: "session_ready",
        protocol: VOICE_REALTIME_PROTOCOL,
        session_id: envelope.session_id,
        owner_instance_id: envelope.data.owner_instance_id,
        connection_epoch: envelope.connection_epoch,
        resumable: false,
        resume_window_ms: 0,
      };
    case "session_started":
      return { type: "input_stream_ready" };
    case "audio_accepted":
    case "audio_gap":
      return null;
    case "speech_started":
      return identity ? { type: "user_speech_start", ...identity } : null;
    case "speech_ended":
      return identity
        ? { type: "user_speech_end", ...identity, reason: envelope.data.reason }
        : null;
    case "transcript_partial":
    case "transcript_stable":
    case "transcript_correction":
      return identity
        ? { type: "user_transcript_snapshot", ...identity, text: envelope.data.text }
        : null;
    case "transcript_final":
      return identity
        ? {
            type: "user_transcript_final",
            ...identity,
            text: envelope.data.text,
            language: envelope.data.language,
          }
        : null;
    case "assistant_text_started":
      return identity ? { type: "assistant_text_start", ...identity } : null;
    case "assistant_text_partial":
      return identity
        ? { type: "assistant_text_snapshot", ...identity, text: envelope.data.text }
        : null;
    case "assistant_text_final":
      return identity
        ? { type: "assistant_text_final", ...identity, text: envelope.data.text }
        : null;
    case "assistant_audio_started":
      return identity
        ? {
            type: "assistant_audio_start",
            ...identity,
            sample_rate: envelope.data.sample_rate,
            audio_format: envelope.data.format,
          }
        : null;
    case "assistant_audio_completed":
      return identity ? { type: "assistant_audio_done", ...identity } : null;
    case "interruption":
      return identity
        ? { type: "turn_interrupted", ...identity, reason: envelope.data.reason }
        : null;
    case "turn_completed":
      return identity
        ? { type: "turn_done", ...identity, status: envelope.data.status }
        : null;
    case "recoverable_error":
      return { type: "error", code: envelope.data.code, message: envelope.data.message };
    case "fatal_error":
      return {
        type: "error",
        code: envelope.data.code,
        message: envelope.data.message,
        fatal: true,
      };
    case "closing":
    case "closed":
      return null;
    case "pong":
      return {
        type: "pong",
        timestamp_ms: envelope.data.client_timestamp_ms ?? undefined,
        server_time_ms: envelope.data.server_timestamp_ms,
      };
  }
  return null;
}

function typedVoiceIdentity(
  envelope: VoiceRealtimeV2ServerEnvelope,
): { utterance_id: string; utterance_seq: number } | null {
  if (typeof envelope.utterance_id !== "string" || !envelope.utterance_id) {
    return null;
  }
  const match = /^turn-(\d+)$/.exec(envelope.turn_id ?? "");
  const utteranceSeq = match ? Number(match[1]) : Number.NaN;
  if (!Number.isSafeInteger(utteranceSeq) || utteranceSeq < 0) {
    return null;
  }
  return { utterance_id: envelope.utterance_id, utterance_seq: utteranceSeq };
}

function isRealtimeVoiceClose(value: unknown): value is RealtimeVoiceClose {
  if (!value || typeof value !== "object") return false;
  const close = value as Record<string, unknown>;
  return (
    typeof close.code === "string" &&
    typeof close.reason === "string" &&
    typeof close.message === "string" &&
    typeof close.retryable === "boolean"
  );
}

function isNonNegativeSafeInteger(value: unknown): value is number {
  return Number.isSafeInteger(value) && Number(value) >= 0;
}

function isPositiveSafeInteger(value: unknown): value is number {
  return Number.isSafeInteger(value) && Number(value) > 0;
}

export function isVoiceRealtimeServerEvent(
  value: unknown,
): value is VoiceRealtimeServerEvent {
  if (!value || typeof value !== "object") return false;
  const event = value as Record<string, unknown>;
  const type = event.type;
  if (typeof type !== "string") return false;

  const hasTurnIdentity = () =>
    typeof event.utterance_id === "string" &&
    typeof event.utterance_seq === "number";
  const hasTurnText = (field: "delta" | "text") =>
    hasTurnIdentity() && typeof event[field] === "string";

  switch (type) {
    case "connected":
      return typeof event.protocol === "string";
    case "session_ready":
      return (
        typeof event.protocol === "string" &&
        typeof event.session_id === "string" &&
        typeof event.owner_instance_id === "string" &&
        typeof event.connection_epoch === "number" &&
        event.resumable === false &&
        event.resume_window_ms === 0
      );
    case "input_stream_ready":
    case "input_stream_stopped":
    case "pong":
      return true;
    case "user_speech_start":
    case "user_speech_rejected":
    case "user_speech_end":
    case "turn_processing":
    case "user_transcript_start":
    case "assistant_text_start":
    case "assistant_audio_done":
    case "turn_interrupted":
      return hasTurnIdentity();
    case "user_transcript_delta":
    case "assistant_text_delta":
      return hasTurnText("delta");
    case "user_transcript_snapshot":
    case "user_transcript_final":
    case "assistant_text_snapshot":
    case "assistant_text_final":
      return hasTurnText("text");
    case "assistant_audio_start":
      return (
        hasTurnIdentity() &&
        typeof event.sample_rate === "number" &&
        ["pcm_i16", "pcm_f32", "wav"].includes(String(event.audio_format))
      );
    case "turn_done":
      return (
        hasTurnIdentity() &&
        ["ok", "error", "timeout", "interrupted", "no_input"].includes(
          String(event.status),
        )
      );
    case "error":
      return typeof event.message === "string";
    default:
      return false;
  }
}

export function shouldStopVoiceRealtimePlayback(
  event: VoiceRealtimeServerEvent,
  activeUtteranceSeq: number | null,
): boolean {
  if (activeUtteranceSeq == null) {
    return false;
  }

  if (event.type === "user_speech_start") {
    return activeUtteranceSeq < event.utterance_seq;
  }
  if (event.type === "turn_interrupted") {
    return activeUtteranceSeq === event.utterance_seq;
  }
  return (
    event.type === "turn_done" &&
    event.status === "interrupted" &&
    activeUtteranceSeq === event.utterance_seq
  );
}

export function makeTranscriptEntryId(_role: "user" | "assistant"): string {
  return createUuid();
}

export async function transcodeToWav(
  inputBlob: Blob,
  targetSampleRate = 16000,
): Promise<Blob> {
  if (inputBlob.type === "audio/wav" || inputBlob.type === "audio/x-wav") {
    return inputBlob;
  }

  const decodeContext = new AudioContext();
  try {
    const sourceBytes = await inputBlob.arrayBuffer();
    const decoded = await decodeContext.decodeAudioData(sourceBytes.slice(0));

    const monoBuffer = decodeContext.createBuffer(
      1,
      decoded.length,
      decoded.sampleRate,
    );
    const mono = monoBuffer.getChannelData(0);

    for (let i = 0; i < decoded.length; i += 1) {
      let sum = 0;
      for (let ch = 0; ch < decoded.numberOfChannels; ch += 1) {
        sum += decoded.getChannelData(ch)[i] ?? 0;
      }
      mono[i] = sum / decoded.numberOfChannels;
    }

    const rendered = await (() => {
      if (decoded.sampleRate === targetSampleRate) {
        return Promise.resolve(monoBuffer);
      }

      const targetLength = Math.ceil(
        (monoBuffer.length * targetSampleRate) / monoBuffer.sampleRate,
      );
      const offline = new OfflineAudioContext(
        1,
        targetLength,
        targetSampleRate,
      );
      const source = offline.createBufferSource();
      source.buffer = monoBuffer;
      source.connect(offline.destination);
      source.start(0);
      return offline.startRendering();
    })();

    return encodeWavPcm16(rendered.getChannelData(0), targetSampleRate);
  } finally {
    decodeContext.close().catch(() => {});
  }
}
