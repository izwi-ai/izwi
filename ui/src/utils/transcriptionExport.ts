import type { TranscriptionRecord } from "@/api";
import {
  formatTranscriptionText,
  transcriptionEntriesFromRecord,
  transcriptionHasTimestamps,
} from "@/features/transcription/transcript";

export type TranscriptionExportFormat = "txt" | "json" | "srt" | "vtt";

export interface TranscriptionExportOptions {
  includeMetadata?: boolean;
}

export interface TranscriptionExportPayload {
  content: string;
  extension: string;
  filename: string;
  mimeType: string;
}

export type ExportableTranscriptionRecord = Pick<
  TranscriptionRecord,
  "id" | "duration_secs" | "transcription" | "segments" | "words"
> & {
  created_at?: number | null;
  model_id?: string | null;
  aligner_model_id?: string | null;
  language?: string | null;
  audio_filename?: string | null;
};

function stripExtension(filename: string): string {
  return filename.replace(/\.[^.]+$/, "");
}

function baseFilename(record: ExportableTranscriptionRecord): string {
  const source =
    record.audio_filename?.trim() || `transcription-${record.id}`;
  return stripExtension(source);
}

function formatSeconds(seconds: number): string {
  if (!Number.isFinite(seconds) || seconds < 0) {
    return "0.00";
  }
  return seconds.toFixed(2);
}

function formatSrtTimestamp(totalSeconds: number): string {
  const clamped = Math.max(0, totalSeconds);
  const hours = Math.floor(clamped / 3600);
  const minutes = Math.floor((clamped % 3600) / 60);
  const seconds = Math.floor(clamped % 60);
  const milliseconds = Math.round((clamped - Math.floor(clamped)) * 1000);
  return `${hours.toString().padStart(2, "0")}:${minutes
    .toString()
    .padStart(2, "0")}:${seconds.toString().padStart(2, "0")},${milliseconds
    .toString()
    .padStart(3, "0")}`;
}

function formatVttTimestamp(totalSeconds: number): string {
  return formatSrtTimestamp(totalSeconds).replace(",", ".");
}

function createdAtLabel(createdAt: number | null | undefined): string {
  if (typeof createdAt !== "number" || !Number.isFinite(createdAt)) {
    return "Unknown";
  }
  return new Date(createdAt).toISOString();
}

function metadataBlock(record: ExportableTranscriptionRecord): string[] {
  const timestamped = transcriptionHasTimestamps(record);
  const alignerLabel =
    timestamped && record.aligner_model_id
      ? ` (${record.aligner_model_id})`
      : "";

  return [
    `File: ${record.audio_filename ?? `${record.id}.wav`}`,
    `Model: ${record.model_id ?? "Unknown model"}`,
    `Language: ${record.language ?? "Unknown language"}`,
    `Timestamps: ${timestamped ? `Enabled${alignerLabel}` : "Disabled"}`,
    `Duration: ${formatSeconds(record.duration_secs ?? 0)}s`,
    `Created At: ${createdAtLabel(record.created_at)}`,
  ];
}

function txtContent(
  record: ExportableTranscriptionRecord,
  options: TranscriptionExportOptions,
): string {
  const transcript = formatTranscriptionText(record);
  if (!options.includeMetadata) {
    return transcript;
  }
  return `${metadataBlock(record).join("\n")}\n\n${transcript}`;
}

function jsonContent(
  record: ExportableTranscriptionRecord,
  options: TranscriptionExportOptions,
): string {
  const entries = transcriptionEntriesFromRecord(record).map((entry) => ({
    start: entry.start,
    end: entry.end,
    text: entry.text,
    word_start: entry.wordStart,
    word_end: entry.wordEnd,
    timed: entry.timed,
  }));

  const payload = options.includeMetadata
    ? {
        metadata: {
          id: record.id,
          created_at: record.created_at ?? null,
          model_id: record.model_id ?? null,
          aligner_model_id: record.aligner_model_id ?? null,
          audio_filename: record.audio_filename ?? null,
          language: record.language ?? null,
          duration_secs: record.duration_secs ?? null,
          timestamps_enabled: transcriptionHasTimestamps(record),
        },
        transcript: entries,
      }
    : entries;

  return JSON.stringify(payload, null, 2);
}

function subtitleContent(
  record: ExportableTranscriptionRecord,
  format: "srt" | "vtt",
): string {
  const entries = transcriptionEntriesFromRecord(record)
    .map((entry, index, allEntries) => {
      const start = Number.isFinite(entry.start) ? Math.max(0, entry.start) : 0;
      const nextStart =
        index < allEntries.length - 1 &&
        Number.isFinite(allEntries[index + 1]?.start)
          ? Math.max(start, allEntries[index + 1]?.start ?? start)
          : null;
      const durationEnd =
        typeof record.duration_secs === "number" &&
        Number.isFinite(record.duration_secs) &&
        record.duration_secs > start
          ? record.duration_secs
          : null;
      const fallbackEnd = durationEnd ?? nextStart ?? start + 1;
      const end =
        Number.isFinite(entry.end) && entry.end > start
          ? entry.end
          : Math.max(fallbackEnd, start + 1);

      return {
        start,
        end,
        text: entry.text.trim(),
      };
    })
    .filter((entry) => entry.text.length > 0 && entry.end > entry.start);

  const lines = entries.map((entry, index) => {
    const timestamp =
      format === "srt"
        ? `${formatSrtTimestamp(entry.start)} --> ${formatSrtTimestamp(entry.end)}`
        : `${formatVttTimestamp(entry.start)} --> ${formatVttTimestamp(entry.end)}`;

    if (format === "srt") {
      return `${index + 1}\n${timestamp}\n${entry.text}`;
    }

    return `${timestamp}\n${entry.text}`;
  });

  if (format === "vtt") {
    return `WEBVTT\n\n${lines.join("\n\n")}`;
  }

  return lines.join("\n\n");
}

export function buildTranscriptionExport(
  record: ExportableTranscriptionRecord,
  format: TranscriptionExportFormat,
  options: TranscriptionExportOptions = {},
): TranscriptionExportPayload {
  const includeMetadata = options.includeMetadata ?? format === "json";
  const normalizedOptions = { includeMetadata };

  switch (format) {
    case "txt":
      return {
        content: txtContent(record, normalizedOptions),
        extension: "txt",
        filename: `${baseFilename(record)}.txt`,
        mimeType: "text/plain; charset=utf-8",
      };
    case "json":
      return {
        content: jsonContent(record, normalizedOptions),
        extension: "json",
        filename: `${baseFilename(record)}.json`,
        mimeType: "application/json; charset=utf-8",
      };
    case "srt":
      return {
        content: subtitleContent(record, "srt"),
        extension: "srt",
        filename: `${baseFilename(record)}.srt`,
        mimeType: "application/x-subrip; charset=utf-8",
      };
    case "vtt":
      return {
        content: subtitleContent(record, "vtt"),
        extension: "vtt",
        filename: `${baseFilename(record)}.vtt`,
        mimeType: "text/vtt; charset=utf-8",
      };
    default:
      return {
        content: txtContent(record, normalizedOptions),
        extension: "txt",
        filename: `${baseFilename(record)}.txt`,
        mimeType: "text/plain; charset=utf-8",
      };
  }
}
