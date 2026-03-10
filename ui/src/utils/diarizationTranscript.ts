import type {
  DiarizationRecord,
  DiarizationResponse,
  DiarizationSegment,
  DiarizationUtterance,
  DiarizationWord,
} from "../api";

export interface TranscriptEntry {
  speaker: string;
  start: number;
  end: number;
  text: string;
}

export interface RawSpeakerSummary {
  rawSpeaker: string;
  displaySpeaker: string;
  utteranceCount: number;
  totalDuration: number;
  wordCount: number;
}

export interface SpeakerSummary {
  displaySpeaker: string;
  rawSpeakers: string[];
  utteranceCount: number;
  totalDuration: number;
  wordCount: number;
}

type SpeakerNameOverrideSource = {
  speaker_name_overrides?: Record<string, string> | null;
};

type TranscriptSource = Pick<
  DiarizationResponse,
  "utterances" | "transcript" | "raw_transcript"
> &
  SpeakerNameOverrideSource;

type TranscriptRecordSource = Pick<
  DiarizationRecord,
  "utterances" | "transcript" | "raw_transcript"
> &
  SpeakerNameOverrideSource;

function speakerNameOverridesFromSource(
  source: SpeakerNameOverrideSource | null | undefined,
): Record<string, string> {
  if (!source?.speaker_name_overrides) {
    return {};
  }

  return Object.fromEntries(
    Object.entries(source.speaker_name_overrides)
      .map(([raw, corrected]) => [String(raw).trim(), String(corrected).trim()])
      .filter(([raw, corrected]) => raw.length > 0 && corrected.length > 0),
  );
}

export function resolveSpeakerName(
  rawSpeaker: string,
  source: SpeakerNameOverrideSource | null | undefined,
): string {
  const raw = String(rawSpeaker ?? "").trim();
  if (!raw) {
    return "UNKNOWN";
  }

  const corrected = speakerNameOverridesFromSource(source)[raw];
  return corrected || raw;
}

function coerceUtteranceEntries(
  utterances: DiarizationUtterance[] | null | undefined,
  source: SpeakerNameOverrideSource | null | undefined,
): TranscriptEntry[] {
  if (!Array.isArray(utterances)) {
    return [];
  }

  return utterances
    .map((utterance) => ({
      speaker: resolveSpeakerName(String(utterance.speaker ?? "UNKNOWN"), source),
      start: Number(utterance.start ?? 0),
      end: Number(utterance.end ?? 0),
      text: String(utterance.text ?? "").trim(),
    }))
    .filter(
      (entry) =>
        entry.text.length > 0 &&
        Number.isFinite(entry.start) &&
        Number.isFinite(entry.end) &&
        entry.end > entry.start,
    );
}

function parseTranscriptEntriesFromText(
  text: string,
  source: SpeakerNameOverrideSource | null | undefined,
): TranscriptEntry[] {
  return text
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter(Boolean)
    .map((line) => line.replace(/^[-*]\s+/, "").replace(/^\d+\.\s+/, ""))
    .map((line): TranscriptEntry | null => {
      const match = line.match(
        /^([A-Za-z0-9_]+)\s+\[([0-9]+(?:\.[0-9]+)?)s\s*-\s*([0-9]+(?:\.[0-9]+)?)s\]:\s*(.*)$/,
      );
      if (!match) {
        return null;
      }
      const start = Number(match[2]);
      const end = Number(match[3]);
      if (!Number.isFinite(start) || !Number.isFinite(end) || end <= start) {
        return null;
      }
      return {
        speaker: resolveSpeakerName(match[1], source),
        start,
        end,
        text: match[4].trim(),
      };
    })
    .filter((entry): entry is TranscriptEntry => entry !== null);
}

function sanitizeTranscriptText(transcript: string, rawTranscript: string): string {
  const source = (transcript || rawTranscript || "").trim();
  if (!source) {
    return "";
  }

  return source
    .replace(/<think>[\s\S]*?<\/think>/gi, " ")
    .replace(/```text/gi, "")
    .replace(/```/g, "")
    .trim();
}

export function transcriptEntriesFromUtterances(
  utterances: DiarizationUtterance[] | null | undefined,
  source?: SpeakerNameOverrideSource | null,
): TranscriptEntry[] {
  return coerceUtteranceEntries(utterances, source);
}

export function transcriptEntriesFromResult(result: TranscriptSource): TranscriptEntry[] {
  const entries = coerceUtteranceEntries(result.utterances, result);
  if (entries.length > 0) {
    return entries;
  }
  return parseTranscriptEntriesFromText(
    sanitizeTranscriptText(result.transcript, result.raw_transcript),
    result,
  );
}

export function formattedTranscriptFromResult(result: TranscriptSource): string {
  const entries = transcriptEntriesFromResult(result);
  if (entries.length > 0) {
    return formatTranscriptFromEntries(entries);
  }
  return sanitizeTranscriptText(result.transcript, result.raw_transcript);
}

export function transcriptEntriesFromRecord(record: TranscriptRecordSource): TranscriptEntry[] {
  const entries = coerceUtteranceEntries(record.utterances, record);
  if (entries.length > 0) {
    return entries;
  }
  return parseTranscriptEntriesFromText(
    sanitizeTranscriptText(record.transcript, record.raw_transcript),
    record,
  );
}

export function formattedTranscriptFromRecord(record: TranscriptRecordSource): string {
  const entries = transcriptEntriesFromRecord(record);
  if (entries.length > 0) {
    return formatTranscriptFromEntries(entries);
  }
  return sanitizeTranscriptText(record.transcript, record.raw_transcript);
}

export function formatTranscriptFromEntries(entries: TranscriptEntry[]): string {
  return entries
    .map(
      (entry) =>
        `${entry.speaker} [${entry.start.toFixed(2)}s - ${entry.end.toFixed(2)}s]: ${entry.text}`,
    )
    .join("\n");
}

export function previewTranscript(
  entries: TranscriptEntry[],
  transcript: string,
  rawTranscript: string,
  maxChars = 180,
): string {
  const formatted =
    entries.length > 0
      ? formatTranscriptFromEntries(entries)
      : sanitizeTranscriptText(transcript, rawTranscript);
  if (formatted.length <= maxChars) {
    return formatted;
  }
  return `${formatted.slice(0, maxChars)}...`;
}

function aggregateRawSpeakerSummaries(
  utterances: DiarizationUtterance[] | null | undefined,
  words: DiarizationWord[] | null | undefined,
  segments: DiarizationSegment[] | null | undefined,
  source: SpeakerNameOverrideSource | null | undefined,
): RawSpeakerSummary[] {
  const wordCounts = new Map<string, number>();
  for (const word of words ?? []) {
    const rawSpeaker = String(word.speaker ?? "").trim();
    if (!rawSpeaker) {
      continue;
    }
    wordCounts.set(rawSpeaker, (wordCounts.get(rawSpeaker) ?? 0) + 1);
  }

  const summaries = new Map<string, RawSpeakerSummary>();
  const ensureSummary = (rawSpeaker: string) => {
    const trimmed = rawSpeaker.trim();
    if (!trimmed) {
      return null;
    }
    const existing = summaries.get(trimmed);
    if (existing) {
      return existing;
    }
    const next: RawSpeakerSummary = {
      rawSpeaker: trimmed,
      displaySpeaker: resolveSpeakerName(trimmed, source),
      utteranceCount: 0,
      totalDuration: 0,
      wordCount: wordCounts.get(trimmed) ?? 0,
    };
    summaries.set(trimmed, next);
    return next;
  };

  for (const utterance of utterances ?? []) {
    const summary = ensureSummary(String(utterance.speaker ?? ""));
    if (!summary) {
      continue;
    }
    summary.utteranceCount += 1;
    const start = Number(utterance.start ?? 0);
    const end = Number(utterance.end ?? 0);
    if (Number.isFinite(start) && Number.isFinite(end) && end > start) {
      summary.totalDuration += end - start;
    }
  }

  if (summaries.size === 0) {
    for (const segment of segments ?? []) {
      const summary = ensureSummary(String(segment.speaker ?? ""));
      if (!summary) {
        continue;
      }
      summary.utteranceCount += 1;
      const start = Number(segment.start ?? 0);
      const end = Number(segment.end ?? 0);
      if (Number.isFinite(start) && Number.isFinite(end) && end > start) {
        summary.totalDuration += end - start;
      }
    }
  }

  return Array.from(summaries.values()).sort((left, right) =>
    left.rawSpeaker.localeCompare(right.rawSpeaker),
  );
}

export function rawSpeakerSummariesFromRecord(
  record: Pick<
    DiarizationRecord,
    "utterances" | "words" | "segments" | "speaker_name_overrides"
  >,
): RawSpeakerSummary[] {
  return aggregateRawSpeakerSummaries(
    record.utterances,
    record.words,
    record.segments,
    record,
  );
}

export function speakerSummariesFromRecord(
  record: Pick<
    DiarizationRecord,
    "utterances" | "words" | "segments" | "speaker_name_overrides"
  >,
): SpeakerSummary[] {
  const grouped = new Map<string, SpeakerSummary>();

  for (const rawSummary of rawSpeakerSummariesFromRecord(record)) {
    const existing = grouped.get(rawSummary.displaySpeaker);
    if (existing) {
      existing.rawSpeakers.push(rawSummary.rawSpeaker);
      existing.utteranceCount += rawSummary.utteranceCount;
      existing.totalDuration += rawSummary.totalDuration;
      existing.wordCount += rawSummary.wordCount;
      continue;
    }

    grouped.set(rawSummary.displaySpeaker, {
      displaySpeaker: rawSummary.displaySpeaker,
      rawSpeakers: [rawSummary.rawSpeaker],
      utteranceCount: rawSummary.utteranceCount,
      totalDuration: rawSummary.totalDuration,
      wordCount: rawSummary.wordCount,
    });
  }

  return Array.from(grouped.values()).sort((left, right) =>
    left.displaySpeaker.localeCompare(right.displaySpeaker),
  );
}

export function correctedSpeakerCount(
  record: Pick<
    DiarizationRecord,
    | "speaker_count"
    | "corrected_speaker_count"
    | "utterances"
    | "words"
    | "segments"
    | "speaker_name_overrides"
  >,
): number {
  if (typeof record.corrected_speaker_count === "number") {
    return record.corrected_speaker_count;
  }

  const summaries = speakerSummariesFromRecord(record);
  return summaries.length > 0 ? summaries.length : record.speaker_count ?? 0;
}
