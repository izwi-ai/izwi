export type SpeechTextUploadPhase =
  | "idle"
  | "preparing"
  | "uploading"
  | "accepted"
  | "opening"
  | "failed"
  | "cancelled";

export interface SpeechTextUploadState {
  phase: SpeechTextUploadPhase;
  fileName: string;
  fileSizeBytes: number;
  fileKind: string;
  loadedBytes: number;
  totalBytes: number | null;
  percent: number | null;
  errorMessage?: string;
}

const FILE_SIZE_UNITS = ["B", "KB", "MB", "GB"] as const;

const MIME_KIND_LABELS: Record<string, string> = {
  "audio/aac": "AAC",
  "audio/flac": "FLAC",
  "audio/mp3": "MP3",
  "audio/mp4": "M4A",
  "audio/mpeg": "MP3",
  "audio/ogg": "OGG",
  "audio/wav": "WAV",
  "audio/wave": "WAV",
  "audio/webm": "WEBM",
  "audio/x-m4a": "M4A",
  "audio/x-wav": "WAV",
};

export function clampUploadPercent(value: number): number {
  if (!Number.isFinite(value)) {
    return 0;
  }
  return Math.min(100, Math.max(0, value));
}

export function formatUploadFileSize(bytes: number): string {
  if (!Number.isFinite(bytes) || bytes <= 0) {
    return "0 B";
  }

  let size = bytes;
  let unitIndex = 0;
  while (size >= 1024 && unitIndex < FILE_SIZE_UNITS.length - 1) {
    size /= 1024;
    unitIndex += 1;
  }

  if (unitIndex === 0) {
    return `${Math.round(size)} ${FILE_SIZE_UNITS[unitIndex]}`;
  }

  const precision = size >= 100 ? 0 : 1;
  return `${size.toFixed(precision)} ${FILE_SIZE_UNITS[unitIndex]}`;
}

export function resolveUploadFileKind(blob: Blob, filename?: string | null): string {
  const extension = filename?.match(/\.([^.]+)$/)?.[1]?.trim();
  if (extension) {
    return extension.toUpperCase();
  }

  const mimeType = blob.type.split(";")[0]?.trim().toLowerCase();
  if (!mimeType) {
    return "Audio";
  }

  const mappedLabel = MIME_KIND_LABELS[mimeType];
  if (mappedLabel) {
    return mappedLabel;
  }

  if (mimeType.startsWith("audio/")) {
    return mimeType.slice("audio/".length).toUpperCase();
  }

  return "Audio";
}

export function createSpeechTextUploadState(
  blob: Blob,
  filename: string,
  phase: SpeechTextUploadPhase = "preparing",
): SpeechTextUploadState {
  return {
    phase,
    fileName: filename,
    fileSizeBytes: blob.size,
    fileKind: resolveUploadFileKind(blob, filename),
    loadedBytes: 0,
    totalBytes: blob.size > 0 ? blob.size : null,
    percent: 0,
  };
}

export function createUploadAbortError(message = "Upload cancelled"): Error {
  if (typeof DOMException !== "undefined") {
    return new DOMException(message, "AbortError");
  }

  const error = new Error(message);
  error.name = "AbortError";
  return error;
}
