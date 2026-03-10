import type { DiarizationRecord } from "../api";
import {
  formattedTranscriptFromRecord,
  transcriptEntriesFromRecord,
} from "./diarizationTranscript";

export type DiarizationExportFormat = "txt" | "json" | "srt" | "vtt";

export interface DiarizationExportOptions {
  includeMetadata?: boolean;
}

export interface DiarizationExportPayload {
  content: string;
  extension: string;
  filename: string;
  mimeType: string;
}

type ExportableRecord = Pick<
  DiarizationRecord,
  | "id"
  | "created_at"
  | "model_id"
  | "speaker_count"
  | "corrected_speaker_count"
  | "duration_secs"
  | "audio_filename"
  | "speaker_name_overrides"
  | "utterances"
  | "transcript"
  | "raw_transcript"
>;

function stripExtension(filename: string): string {
  return filename.replace(/\.[^.]+$/, "");
}

function baseFilename(record: ExportableRecord): string {
  const source = record.audio_filename?.trim() || `diarization-${record.id}`;
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

function metadataBlock(record: ExportableRecord): string[] {
  return [
    `File: ${record.audio_filename ?? `${record.id}.wav`}`,
    `Model: ${record.model_id ?? "Unknown model"}`,
    `Speakers: ${record.corrected_speaker_count ?? record.speaker_count}`,
    `Duration: ${formatSeconds(record.duration_secs ?? 0)}s`,
    `Created At: ${new Date(record.created_at).toISOString()}`,
  ];
}

function txtContent(
  record: ExportableRecord,
  options: DiarizationExportOptions,
): string {
  const transcript = formattedTranscriptFromRecord(record);
  if (!options.includeMetadata) {
    return transcript;
  }
  return `${metadataBlock(record).join("\n")}\n\n${transcript}`;
}

function jsonContent(
  record: ExportableRecord,
  options: DiarizationExportOptions,
): string {
  const entries = transcriptEntriesFromRecord(record).map((entry) => ({
    speaker: entry.speaker,
    start: entry.start,
    end: entry.end,
    text: entry.text,
  }));

  const payload = options.includeMetadata
    ? {
        metadata: {
          id: record.id,
          created_at: record.created_at,
          model_id: record.model_id,
          audio_filename: record.audio_filename,
          speaker_count: record.speaker_count,
          corrected_speaker_count:
            record.corrected_speaker_count ?? record.speaker_count,
          duration_secs: record.duration_secs,
        },
        transcript: entries,
      }
    : entries;

  return JSON.stringify(payload, null, 2);
}

function subtitleContent(
  record: ExportableRecord,
  format: "srt" | "vtt",
): string {
  const entries = transcriptEntriesFromRecord(record);
  const lines = entries.map((entry, index) => {
    const timestamp =
      format === "srt"
        ? `${formatSrtTimestamp(entry.start)} --> ${formatSrtTimestamp(entry.end)}`
        : `${formatVttTimestamp(entry.start)} --> ${formatVttTimestamp(entry.end)}`;
    const cueText = `${entry.speaker}: ${entry.text}`;

    if (format === "srt") {
      return `${index + 1}\n${timestamp}\n${cueText}`;
    }

    return `${timestamp}\n${cueText}`;
  });

  if (format === "vtt") {
    return `WEBVTT\n\n${lines.join("\n\n")}`;
  }

  return lines.join("\n\n");
}

export function buildDiarizationExport(
  record: ExportableRecord,
  format: DiarizationExportFormat,
  options: DiarizationExportOptions = {},
): DiarizationExportPayload {
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
