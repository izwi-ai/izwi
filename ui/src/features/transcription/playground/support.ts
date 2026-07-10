import type {
  ModelInfo,
  TranscriptionProcessingStatus,
  TranscriptionRecord,
  TranscriptionRecordSummary,
  TranscriptionSummaryStatus,
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
  summaryModelId?: string | null;
  summaryModelReady?: boolean;
  summaryModelStatus?: ModelInfo["status"] | null;
  onSummaryModelRequired?: () => void;
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

export const TRANSCRIPTION_REALTIME_PROTOCOL = "transcription_realtime" as const;
export const TRANSCRIPTION_REALTIME_VERSION = 3 as const;

export type LegacyTranscriptionRealtimeServerEvent =
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

type TranscriptionRealtimeV3Payload =
  | {
      type: "session_ready";
      data: {
        accepted_version: number;
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
  | {
      type: "transcript_partial";
      data: { text: string; revision: number; language?: string | null };
    }
  | {
      type: "transcript_stable";
      data: { text: string; revision: number; stable_prefix_chars: number };
    }
  | {
      type: "transcript_correction";
      data: {
        text: string;
        revision: number;
        replaces_revision: number;
        reason: string;
      };
    }
  | {
      type: "transcript_final";
      data: { text: string; revision: number; language?: string | null };
    }
  | {
      type: "recoverable_error";
      data: { code: string; message: string; retry_after_ms?: number };
    }
  | {
      type: "fatal_error";
      data: { code: string; message: string; close: RealtimeClosePayload };
    }
  | { type: "closing"; data: { close: RealtimeClosePayload } }
  | { type: "closed"; data: { close: RealtimeClosePayload } }
  | {
      type: "pong";
      data: { client_timestamp_ms?: number; server_timestamp_ms: number };
    };

interface RealtimeClosePayload {
  code: string;
  reason: string;
  message: string;
  retryable: boolean;
}

export type TranscriptionRealtimeV3ServerEnvelope = {
  protocol: typeof TRANSCRIPTION_REALTIME_PROTOCOL;
  version: typeof TRANSCRIPTION_REALTIME_VERSION;
  event_id: number;
  sequence: number;
  session_id: string;
  connection_epoch: number;
  timestamp_ms: number;
  utterance_id?: string;
  turn_id?: string;
  segment_id?: string;
} & TranscriptionRealtimeV3Payload;

export type TranscriptionRealtimeServerEvent =
  | LegacyTranscriptionRealtimeServerEvent
  | TranscriptionRealtimeV3ServerEnvelope;

export function buildTranscriptionRealtimeV3SessionStart(
  modelId: string | null,
  language: string,
) {
  return {
    type: "session_start" as const,
    protocol: TRANSCRIPTION_REALTIME_PROTOCOL,
    version: TRANSCRIPTION_REALTIME_VERSION,
    model_id: modelId || undefined,
    language,
  };
}

export function buildTranscriptionRealtimeWebSocketUrl(
  apiBaseUrl: string,
): string {
  const base = new URL(apiBaseUrl, window.location.origin);
  base.protocol = base.protocol === "https:" ? "wss:" : "ws:";
  base.pathname = `${base.pathname.replace(/\/$/, "")}/speech-to-text/realtime/ws`;
  base.search = "";
  base.hash = "";
  return base.toString();
}

export function isTranscriptionRealtimeServerEvent(
  value: unknown,
): value is TranscriptionRealtimeServerEvent {
  if (!isRecord(value) || typeof value.type !== "string") return false;
  if (
    value.protocol === TRANSCRIPTION_REALTIME_PROTOCOL ||
    value.version === TRANSCRIPTION_REALTIME_VERSION
  ) {
    return isTranscriptionRealtimeV3ServerEnvelope(value);
  }
  return [
    "session_ready",
    "session_started",
    "transcript_partial",
    "error",
    "session_done",
    "pong",
  ].includes(value.type);
}

export function isTranscriptionRealtimeV3ServerEnvelope(
  value: unknown,
): value is TranscriptionRealtimeV3ServerEnvelope {
  if (!value || typeof value !== "object") return false;
  const candidate = value as Record<string, unknown>;
  const validEnvelope =
    candidate.protocol === TRANSCRIPTION_REALTIME_PROTOCOL &&
    candidate.version === TRANSCRIPTION_REALTIME_VERSION &&
    isNonNegativeSafeInteger(candidate.event_id) &&
    isNonNegativeSafeInteger(candidate.sequence) &&
    typeof candidate.session_id === "string" &&
    candidate.session_id.length > 0 &&
    isNonNegativeSafeInteger(candidate.connection_epoch) &&
    typeof candidate.timestamp_ms === "number" &&
    Number.isFinite(candidate.timestamp_ms) &&
    typeof candidate.type === "string";
  if (!validEnvelope) return false;

  const data = isRecord(candidate.data) ? candidate.data : null;
  switch (candidate.type) {
    case "session_started":
      return candidate.data === undefined;
    case "session_ready":
      return (
        data !== null &&
        data.accepted_version === TRANSCRIPTION_REALTIME_VERSION &&
        typeof data.owner_instance_id === "string" &&
        data.owner_instance_id.length > 0 &&
        data.resumable === false &&
        data.resume_window_ms === 0
      );
    case "audio_accepted":
      return (
        data !== null &&
        isNonNegativeSafeInteger(data.frame_sequence) &&
        isNonNegativeSafeInteger(data.buffer_depth_samples) &&
        isNonNegativeSafeInteger(data.ingress_queue_depth)
      );
    case "audio_gap":
      return (
        data !== null &&
        isNonNegativeSafeInteger(data.expected_frame_sequence) &&
        isNonNegativeSafeInteger(data.received_frame_sequence) &&
        isNonNegativeSafeInteger(data.missing_frames) &&
        ["continue", "reset_segment", "close_session"].includes(
          String(data.action),
        )
      );
    case "transcript_partial":
    case "transcript_final":
      return (
        data !== null &&
        typeof data.text === "string" &&
        isNonNegativeSafeInteger(data.revision) &&
        (data.language === undefined ||
          data.language === null ||
          typeof data.language === "string")
      );
    case "transcript_stable":
      return (
        data !== null &&
        typeof data.text === "string" &&
        isNonNegativeSafeInteger(data.revision) &&
        isNonNegativeSafeInteger(data.stable_prefix_chars)
      );
    case "transcript_correction":
      return (
        data !== null &&
        typeof data.text === "string" &&
        isNonNegativeSafeInteger(data.revision) &&
        isNonNegativeSafeInteger(data.replaces_revision) &&
        typeof data.reason === "string"
      );
    case "recoverable_error":
      return (
        data !== null &&
        typeof data.code === "string" &&
        typeof data.message === "string" &&
        (data.retry_after_ms === undefined ||
          isNonNegativeSafeInteger(data.retry_after_ms))
      );
    case "fatal_error":
      return (
        data !== null &&
        typeof data.code === "string" &&
        typeof data.message === "string" &&
        isRealtimeClosePayload(data.close)
      );
    case "closing":
    case "closed":
      return data !== null && isRealtimeClosePayload(data.close);
    case "pong":
      return (
        data !== null &&
        (data.client_timestamp_ms === undefined ||
          isNonNegativeSafeInteger(data.client_timestamp_ms)) &&
        isNonNegativeSafeInteger(data.server_timestamp_ms)
      );
    default:
      return false;
  }
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}

function isNonNegativeSafeInteger(value: unknown): value is number {
  return (
    typeof value === "number" && Number.isSafeInteger(value) && value >= 0
  );
}

function isRealtimeClosePayload(value: unknown): value is RealtimeClosePayload {
  if (!isRecord(value)) return false;
  return (
    typeof value.code === "string" &&
    typeof value.reason === "string" &&
    typeof value.message === "string" &&
    typeof value.retryable === "boolean"
  );
}

export function isValidTranscriptionRealtimeV3Successor(
  previous: TranscriptionRealtimeV3ServerEnvelope | null,
  current: TranscriptionRealtimeV3ServerEnvelope,
): boolean {
  if (!previous) {
    return (
      current.event_id === 1 &&
      current.sequence === 0 &&
      current.connection_epoch === 0
    );
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

function buildSummaryPreview(
  summaryText: string | null | undefined,
  maxChars = 200,
): string | null {
  if (!summaryText) {
    return null;
  }
  const normalized = normalizeTranscript(summaryText);
  if (!normalized) {
    return null;
  }
  if (normalized.length <= maxChars) {
    return normalized;
  }
  return `${normalized.slice(0, maxChars)}...`;
}

export function normalizeSummaryStatus(
  status: string | null | undefined,
  summaryText?: string | null,
  summaryError?: string | null,
): TranscriptionSummaryStatus {
  if (status === "not_requested" || status === "pending" || status === "ready" || status === "failed") {
    return status;
  }
  if ((summaryText ?? "").trim().length > 0) {
    return "ready";
  }
  if ((summaryError ?? "").trim().length > 0) {
    return "failed";
  }
  return "not_requested";
}

export function normalizeProcessingStatus(
  status: string | null | undefined,
  processingError?: string | null,
): TranscriptionProcessingStatus {
  if (
    status === "pending" ||
    status === "processing" ||
    status === "ready" ||
    status === "failed"
  ) {
    return status;
  }
  if ((processingError ?? "").trim().length > 0) {
    return "failed";
  }
  return "ready";
}

export function summaryStatusLabel(status: TranscriptionSummaryStatus): string {
  switch (status) {
    case "pending":
      return "Summary pending";
    case "ready":
      return "Summary ready";
    case "failed":
      return "Summary failed";
    case "not_requested":
    default:
      return "Summary not requested";
  }
}

export function summaryStatusTone(
  status: TranscriptionSummaryStatus,
): "neutral" | "warning" | "success" | "danger" {
  switch (status) {
    case "pending":
      return "warning";
    case "ready":
      return "success";
    case "failed":
      return "danger";
    case "not_requested":
    default:
      return "neutral";
  }
}

export function summarizeRecord(
  record: TranscriptionRecord,
): TranscriptionRecordSummary {
  const summaryStatus = normalizeSummaryStatus(
    record.summary_status,
    record.summary_text,
    record.summary_error,
  );
  const summaryPreview = buildSummaryPreview(record.summary_text);
  return {
    id: record.id,
    created_at: record.created_at,
    model_id: record.model_id,
    language: record.language,
    processing_status: normalizeProcessingStatus(
      record.processing_status,
      record.processing_error,
    ),
    processing_error: record.processing_error ?? null,
    duration_secs: record.duration_secs,
    processing_time_ms: record.processing_time_ms,
    rtf: record.rtf,
    audio_mime_type: record.audio_mime_type,
    audio_filename: record.audio_filename,
    transcription_preview: buildTranscriptPreview(record.transcription),
    transcription_chars: Array.from(record.transcription).length,
    summary_status: summaryStatus,
    summary_preview: summaryPreview,
    summary_chars: Array.from(record.summary_text ?? "").length,
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
