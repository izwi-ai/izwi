import type {
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
