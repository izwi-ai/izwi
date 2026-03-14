import type { ModelInfo } from "@/api";
import { isKokoroVariant } from "@/types";

export type RuntimeStatus =
  | "idle"
  | "listening"
  | "user_speaking"
  | "processing"
  | "assistant_speaking";

export type VoiceRealtimeServerEvent =
  | { type: "connected"; protocol: string; server_time_ms?: number }
  | { type: "session_ready"; protocol: string }
  | {
      type: "input_stream_ready";
      vad?: {
        threshold?: number;
        min_speech_ms?: number;
        silence_duration_ms?: number;
      };
    }
  | { type: "input_stream_stopped" }
  | { type: "listening"; utterance_id: string; utterance_seq: number }
  | { type: "user_speech_start"; utterance_id: string; utterance_seq: number }
  | {
      type: "user_speech_end";
      utterance_id: string;
      utterance_seq: number;
      reason?: "silence" | "max_duration" | "stream_stopped";
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
    }
  | { type: "pong"; timestamp_ms?: number; server_time_ms?: number };

export type VoiceRealtimeClientMessage =
  | { type: "session_start"; system_prompt?: string }
  | {
      type: "input_stream_start";
      mode?: "modular";
      asr_model_id?: string;
      text_model_id?: string;
      tts_model_id?: string;
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
    variant.includes("Qwen3-ASR") ||
    variant.includes("Whisper-Large-v3-Turbo") ||
    variant.includes("Parakeet-TDT") ||
    variant.includes("Voxtral")
  );
}

export function formatModelVariantLabel(variant: string): string {
  const normalized = variant
    .replace(/-4bit\b/g, "-4-bit")
    .replace(/-8bit\b/g, "-8-bit");

  if (normalized.startsWith("Qwen3-ASR-")) {
    return normalized.replace("Qwen3-ASR-", "ASR ");
  }

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

export function isVoiceRealtimeServerEvent(
  value: unknown,
): value is VoiceRealtimeServerEvent {
  return (
    !!value &&
    typeof value === "object" &&
    "type" in value &&
    typeof (value as { type?: unknown }).type === "string"
  );
}

export function makeTranscriptEntryId(role: "user" | "assistant"): string {
  return `${role}-${Date.now()}-${Math.random().toString(16).slice(2)}`;
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
