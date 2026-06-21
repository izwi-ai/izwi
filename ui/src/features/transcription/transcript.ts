import type {
  SpeakerAttributedAsrTurn,
  TranscriptionRecord,
  TranscriptionSegment,
  TranscriptionWord,
} from "@/api";

const ENTRY_GAP_BREAK_SECS = 0.85;
const MAX_ENTRY_WORDS = 18;
const MAX_ENTRY_DURATION_SECS = 9;

export interface TranscriptionTranscriptEntry {
  start: number;
  end: number;
  text: string;
  wordStart: number;
  wordEnd: number;
  timed: boolean;
}

type TranscriptRecordLike = Pick<
  TranscriptionRecord,
  | "duration_secs"
  | "transcription"
  | "segments"
  | "words"
  | "speaker_attributed_text"
  | "speaker_turns"
>;

function normalizeTranscriptText(text: string): string {
  return text.trim().replace(/\s+/g, " ");
}

function speakerLabel(speaker: string): string {
  const normalized = normalizeTranscriptText(speaker);
  if (!normalized) {
    return "[Speaker]";
  }
  if (normalized.startsWith("[") && normalized.endsWith("]")) {
    return normalized;
  }
  return `[${normalized}]`;
}

function speakerTurnLines(turns: SpeakerAttributedAsrTurn[]): string[] {
  return turns
    .map((turn) => {
      const text = normalizeTranscriptText(turn.text ?? "");
      if (!text) {
        return null;
      }
      return `${speakerLabel(turn.speaker ?? "")}: ${text}`;
    })
    .filter((line): line is string => line !== null);
}

function formatExportTimestamp(totalSeconds: number): string {
  if (!Number.isFinite(totalSeconds) || totalSeconds < 0) {
    return "00:00.000";
  }

  const totalMilliseconds = Math.round(totalSeconds * 1000);
  const hours = Math.floor(totalMilliseconds / 3_600_000);
  const minutes = Math.floor((totalMilliseconds % 3_600_000) / 60_000);
  const seconds = Math.floor((totalMilliseconds % 60_000) / 1000);
  const milliseconds = totalMilliseconds % 1000;

  if (hours > 0) {
    return `${hours.toString().padStart(2, "0")}:${minutes
      .toString()
      .padStart(2, "0")}:${seconds.toString().padStart(2, "0")}.${milliseconds
      .toString()
      .padStart(3, "0")}`;
  }

  return `${minutes.toString().padStart(2, "0")}:${seconds
    .toString()
    .padStart(2, "0")}.${milliseconds.toString().padStart(3, "0")}`;
}

function isValidTimedSegment(
  segment: TranscriptionSegment | null | undefined,
): segment is TranscriptionSegment {
  if (!segment) {
    return false;
  }

  return (
    Number.isFinite(segment.start) &&
    Number.isFinite(segment.end) &&
    segment.end > segment.start &&
    String(segment.text ?? "").trim().length > 0
  );
}

function isValidTimedWord(
  word: TranscriptionWord | null | undefined,
): word is TranscriptionWord {
  if (!word) {
    return false;
  }

  return (
    Number.isFinite(word.start) &&
    Number.isFinite(word.end) &&
    word.end > word.start &&
    String(word.word ?? "").trim().length > 0
  );
}

function buildEntriesFromSegments(
  segments: TranscriptionSegment[],
): TranscriptionTranscriptEntry[] {
  return segments
    .filter(isValidTimedSegment)
    .map((segment) => ({
      start: segment.start,
      end: segment.end,
      text: String(segment.text ?? "").trim(),
      wordStart: Number(segment.word_start ?? 0),
      wordEnd: Number(segment.word_end ?? 0),
      timed: true,
    }))
    .sort((left, right) => left.start - right.start);
}

function buildEntriesFromWords(
  transcription: string,
  words: TranscriptionWord[],
): TranscriptionTranscriptEntry[] {
  const timedWords = words.filter(isValidTimedWord);
  if (timedWords.length === 0) {
    return [];
  }

  const originalTokens = transcription.split(/\s+/).filter(Boolean);
  const useOriginalTokens = originalTokens.length === timedWords.length;
  const entries: TranscriptionTranscriptEntry[] = [];
  let entryStart = 0;

  const pushEntry = (entryEnd: number) => {
    const firstWord = timedWords[entryStart];
    const lastWord = timedWords[entryEnd];
    const text = useOriginalTokens
      ? originalTokens.slice(entryStart, entryEnd + 1).join(" ")
      : timedWords
          .slice(entryStart, entryEnd + 1)
          .map((word) => word.word)
          .join(" ");

    const normalizedText = normalizeTranscriptText(text);
    if (!normalizedText) {
      return;
    }

    entries.push({
      start: firstWord.start,
      end: lastWord.end,
      text: normalizedText,
      wordStart: entryStart,
      wordEnd: entryEnd,
      timed: true,
    });
  };

  for (let index = 1; index <= timedWords.length; index += 1) {
    const shouldBreak =
      index >= timedWords.length ||
      timedWords[index].start - timedWords[index - 1].end >= ENTRY_GAP_BREAK_SECS ||
      index - entryStart >= MAX_ENTRY_WORDS ||
      timedWords[index - 1].end - timedWords[entryStart].start >=
        MAX_ENTRY_DURATION_SECS;

    if (!shouldBreak) {
      continue;
    }

    pushEntry(index - 1);
    entryStart = index;
  }

  return entries;
}

function buildEntriesFromSpeakerTurns(
  turns: SpeakerAttributedAsrTurn[],
  durationSecs: number | null,
): TranscriptionTranscriptEntry[] {
  return turns
    .map((turn, index) => {
      const text = normalizeTranscriptText(turn.text ?? "");
      if (!text) {
        return null;
      }
      const hasTiming =
        typeof turn.start === "number" &&
        Number.isFinite(turn.start) &&
        typeof turn.end === "number" &&
        Number.isFinite(turn.end) &&
        turn.end > turn.start;
      const fallbackEnd =
        typeof durationSecs === "number" && Number.isFinite(durationSecs)
          ? Math.max(durationSecs, 0)
          : 0;
      const wordCount = text.split(/\s+/).filter(Boolean).length;
      return {
        start: hasTiming ? Number(turn.start) : 0,
        end: hasTiming ? Number(turn.end) : fallbackEnd,
        text: `${speakerLabel(turn.speaker ?? "")}: ${text}`,
        wordStart: 0,
        wordEnd: Math.max(wordCount - 1, 0),
        timed: hasTiming,
        index,
      };
    })
    .filter(
      (entry): entry is TranscriptionTranscriptEntry & { index: number } =>
        entry !== null,
    )
    .sort((left, right) =>
      left.timed && right.timed ? left.start - right.start : left.index - right.index,
    )
    .map(({ index: _index, ...entry }) => entry);
}

function buildFallbackEntry(
  transcription: string,
  durationSecs: number | null,
): TranscriptionTranscriptEntry[] {
  const normalized = normalizeTranscriptText(transcription);
  if (!normalized) {
    return [];
  }

  const wordCount = normalized.split(/\s+/).filter(Boolean).length;
  return [
    {
      start: 0,
      end:
        typeof durationSecs === "number" && Number.isFinite(durationSecs)
          ? Math.max(durationSecs, 0)
          : 0,
      text: normalized,
      wordStart: 0,
      wordEnd: Math.max(wordCount - 1, 0),
      timed: false,
    },
  ];
}

export function transcriptionEntriesFromRecord(
  record: TranscriptRecordLike | null | undefined,
): TranscriptionTranscriptEntry[] {
  if (!record) {
    return [];
  }

  const segmentEntries = buildEntriesFromSegments(record.segments ?? []);
  if (segmentEntries.length > 0) {
    return segmentEntries;
  }

  const wordEntries = buildEntriesFromWords(record.transcription, record.words ?? []);
  if (wordEntries.length > 0) {
    return wordEntries;
  }

  const speakerTurnEntries = buildEntriesFromSpeakerTurns(
    record.speaker_turns ?? [],
    record.duration_secs,
  );
  if (speakerTurnEntries.length > 0) {
    return speakerTurnEntries;
  }

  return buildFallbackEntry(record.transcription, record.duration_secs);
}

export function transcriptionHasTimestamps(
  record: TranscriptRecordLike | null | undefined,
): boolean {
  if (!record) {
    return false;
  }

  return (
    (record.segments ?? []).some(isValidTimedSegment) ||
    (record.words ?? []).some(isValidTimedWord)
  );
}

export function transcriptionWordCount(
  record: TranscriptRecordLike | null | undefined,
): number {
  if (!record) {
    return 0;
  }

  const timedWords = (record.words ?? []).filter(isValidTimedWord);
  if (timedWords.length > 0) {
    return timedWords.length;
  }

  const speakerWords = (record.speaker_turns ?? [])
    .map((turn) => normalizeTranscriptText(turn.text ?? ""))
    .filter(Boolean)
    .join(" ");
  if (speakerWords) {
    return speakerWords.split(/\s+/).filter(Boolean).length;
  }

  const normalized = normalizeTranscriptText(record.transcription);
  if (!normalized) {
    return 0;
  }

  return normalized.split(/\s+/).filter(Boolean).length;
}

export function formatTranscriptionText(
  record: TranscriptRecordLike | null | undefined,
  options: {
    includeTimestamps?: boolean;
  } = {},
): string {
  if (!record) {
    return "";
  }

  const turnLines = speakerTurnLines(record.speaker_turns ?? []);
  if (turnLines.length > 0) {
    return turnLines.join("\n");
  }

  const nativeSpeakerText =
    record.speaker_attributed_text
      ?.split(/\r?\n/)
      .map((line) => normalizeTranscriptText(line))
      .filter(Boolean)
      .join("\n") ?? "";
  if (nativeSpeakerText) {
    return nativeSpeakerText;
  }

  const normalizedTranscript = normalizeTranscriptText(record.transcription);
  const includeTimestamps =
    options.includeTimestamps ?? transcriptionHasTimestamps(record);
  if (!includeTimestamps) {
    return normalizedTranscript;
  }

  const timedEntries = transcriptionEntriesFromRecord(record).filter(
    (entry) => entry.timed && Number.isFinite(entry.start) && entry.end > entry.start,
  );
  if (timedEntries.length === 0) {
    return normalizedTranscript;
  }

  return timedEntries
    .map((entry) => {
      const text = normalizeTranscriptText(entry.text);
      return `[${formatExportTimestamp(entry.start)} - ${formatExportTimestamp(entry.end)}] ${text}`;
    })
    .join("\n");
}
