import type {
  TranscriptionRecord,
  TranscriptionRecordSummary,
} from "@/api";

export interface ModelOption {
  value: string;
  label: string;
  statusLabel: string;
  isReady: boolean;
}

export interface TranscriptionPlaygroundProps {
  selectedModel: string | null;
  selectedModelReady?: boolean;
  modelOptions?: ModelOption[];
  onSelectModel?: (variant: string) => void;
  onOpenModelManager?: () => void;
  onModelRequired: () => void;
  timestampAlignerModelId?: string | null;
  timestampAlignerReady?: boolean;
  onTimestampAlignerRequired?: () => void;
  historyActionContainer?: HTMLElement | null;
}

export interface ProcessAudioOptions {
  filename?: string;
  transcode?: boolean;
  preserveTranscript?: boolean;
}

export const LANGUAGE_OPTIONS = [
  "English",
  "Chinese",
  "Cantonese",
  "Arabic",
  "German",
  "French",
  "Spanish",
  "Portuguese",
  "Indonesian",
  "Italian",
  "Korean",
  "Russian",
  "Thai",
  "Vietnamese",
  "Japanese",
  "Turkish",
  "Hindi",
  "Malay",
  "Dutch",
  "Swedish",
  "Danish",
  "Finnish",
  "Polish",
  "Czech",
  "Filipino",
  "Persian",
  "Greek",
  "Romanian",
  "Hungarian",
  "Macedonian",
] as const;

const TRANSCRIPTION_WS_BIN_MAGIC = "ITRW";
const TRANSCRIPTION_WS_BIN_VERSION = 1;
const TRANSCRIPTION_WS_BIN_KIND_CLIENT_PCM16 = 1;
const TRANSCRIPTION_WS_BIN_CLIENT_HEADER_LEN = 16;
export const LIVE_MIC_PCM_FRAME_SIZE = 2048;

export type TranscriptionRealtimeServerEvent =
  | { type: "session_ready"; protocol?: string }
  | { type: "session_started" }
  | {
      type: "transcript_partial";
      sequence: number;
      text: string;
      language?: string | null;
    }
  | { type: "error"; message?: string }
  | { type: "session_done" }
  | { type: "pong"; timestamp_ms?: number | null };

export function buildTranscriptionRealtimeWebSocketUrl(
  apiBaseUrl: string,
): string {
  const base = new URL(apiBaseUrl, window.location.origin);
  base.protocol = base.protocol === "https:" ? "wss:" : "ws:";
  base.pathname = `${base.pathname.replace(/\/$/, "")}/transcription/realtime/ws`;
  base.search = "";
  base.hash = "";
  return base.toString();
}

export function isTranscriptionRealtimeServerEvent(
  value: unknown,
): value is TranscriptionRealtimeServerEvent {
  return (
    !!value &&
    typeof value === "object" &&
    "type" in value &&
    typeof (value as { type?: unknown }).type === "string"
  );
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

export function encodeLiveMicChunk(samples: Float32Array): Uint8Array {
  return encodeFloat32ToPcm16Bytes(samples);
}

export function encodeTranscriptionRealtimePcm16Frame(
  pcm16Bytes: Uint8Array,
  sampleRate: number,
  frameSeq: number,
): Uint8Array {
  const frame = new Uint8Array(
    TRANSCRIPTION_WS_BIN_CLIENT_HEADER_LEN + pcm16Bytes.length,
  );
  frame[0] = TRANSCRIPTION_WS_BIN_MAGIC.charCodeAt(0);
  frame[1] = TRANSCRIPTION_WS_BIN_MAGIC.charCodeAt(1);
  frame[2] = TRANSCRIPTION_WS_BIN_MAGIC.charCodeAt(2);
  frame[3] = TRANSCRIPTION_WS_BIN_MAGIC.charCodeAt(3);
  frame[4] = TRANSCRIPTION_WS_BIN_VERSION;
  frame[5] = TRANSCRIPTION_WS_BIN_KIND_CLIENT_PCM16;
  frame[6] = 0;
  frame[7] = 0;
  const view = new DataView(frame.buffer);
  view.setUint32(8, sampleRate >>> 0, true);
  view.setUint32(12, frameSeq >>> 0, true);
  frame.set(pcm16Bytes, TRANSCRIPTION_WS_BIN_CLIENT_HEADER_LEN);
  return frame;
}

function encodeWavPcm16(samples: Float32Array, sampleRate: number): Blob {
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

function normalizeTranscript(text: string): string {
  return text.trim().replace(/\s+/g, " ");
}

function buildTranscriptPreview(text: string, maxChars = 160): string {
  const normalized = normalizeTranscript(text);
  if (!normalized) {
    return "No transcript";
  }
  if (normalized.length <= maxChars) {
    return normalized;
  }
  return `${normalized.slice(0, maxChars)}...`;
}

export function summarizeRecord(
  record: TranscriptionRecord,
): TranscriptionRecordSummary {
  return {
    id: record.id,
    created_at: record.created_at,
    model_id: record.model_id,
    language: record.language,
    duration_secs: record.duration_secs,
    processing_time_ms: record.processing_time_ms,
    rtf: record.rtf,
    audio_mime_type: record.audio_mime_type,
    audio_filename: record.audio_filename,
    transcription_preview: buildTranscriptPreview(record.transcription),
    transcription_chars: Array.from(record.transcription).length,
  };
}

export function formatCreatedAt(timestampMs: number): string {
  if (!Number.isFinite(timestampMs)) {
    return "Unknown time";
  }
  const value = new Date(timestampMs);
  if (Number.isNaN(value.getTime())) {
    return "Unknown time";
  }
  return value.toLocaleString([], {
    month: "short",
    day: "numeric",
    hour: "numeric",
    minute: "2-digit",
  });
}

export function formatAudioDuration(durationSecs: number | null): string {
  if (
    durationSecs === null ||
    !Number.isFinite(durationSecs) ||
    durationSecs < 0
  ) {
    return "Unknown length";
  }
  if (durationSecs < 60) {
    return `${durationSecs.toFixed(1)}s`;
  }
  const minutes = Math.floor(durationSecs / 60);
  const seconds = Math.floor(durationSecs % 60);
  return `${minutes}m ${seconds}s`;
}

export function formatClockTime(totalSeconds: number): string {
  if (!Number.isFinite(totalSeconds) || totalSeconds < 0) {
    return "0:00";
  }
  const minutes = Math.floor(totalSeconds / 60);
  const seconds = Math.floor(totalSeconds % 60);
  return `${minutes}:${seconds.toString().padStart(2, "0")}`;
}
