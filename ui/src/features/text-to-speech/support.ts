import { type SavedVoiceSummary, type SpeechHistoryProcessingStatus } from "@/api";
import { getSpeakerProfilesForVariant } from "@/types";

export function normalizeSpeechProcessingStatus(
  status: SpeechHistoryProcessingStatus | null | undefined,
  error: string | null | undefined,
): SpeechHistoryProcessingStatus {
  if (status === "pending" || status === "processing" || status === "failed") {
    return status;
  }
  if (error) {
    return "failed";
  }
  return "ready";
}

const PROCESSING_PROGRESS: Record<SpeechHistoryProcessingStatus, number> = {
  pending: 0,
  processing: 1,
  ready: 2,
  failed: 2,
};

type ProcessingState = {
  processing_status?: SpeechHistoryProcessingStatus | null;
  processing_error?: string | null;
};

/**
 * The further along of a polled record's status and a local copy's. The
 * copy (a created job's placeholder, a stream's start or final event) may
 * run ahead of the record, never behind it: a long job created without a
 * stream would otherwise read as queued until it finished. A tie keeps the
 * record's.
 */
export function furthestSpeechProcessingStatus(
  record: ProcessingState,
  local: ProcessingState,
): SpeechHistoryProcessingStatus {
  const polled = normalizeSpeechProcessingStatus(
    record.processing_status,
    record.processing_error,
  );
  const copy = normalizeSpeechProcessingStatus(
    local.processing_status,
    local.processing_error,
  );
  return PROCESSING_PROGRESS[copy] > PROCESSING_PROGRESS[polled] ? copy : polled;
}

export function formatSpeechCreatedAt(timestampMs: number): string {
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

export function formatSpeechDuration(durationSecs: number | null): string {
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

const SPEAKER_TAG = /<\|speaker:\d+\|>/g;

/** A take's text as titles and lists show it: a dialogue's
 * `<|speaker:N|>` turn tags dropped, white space collapsed. */
export function speechTextLabel(text: string): string {
  return text.replace(SPEAKER_TAG, " ").replace(/\s+/g, " ").trim();
}

/** A saved voice as voice pickers list it: its name and where it came from. */
export function savedVoiceLabel(voice: SavedVoiceSummary): string {
  const source =
    voice.source_route_kind === "voice_cloning"
      ? "Clone"
      : voice.source_route_kind === "voice_design"
        ? "Design"
        : "Saved";
  return `${voice.name} (${source})`;
}

export function resolveSpeechVoiceLabel({
  savedVoiceId,
  speaker,
  modelId,
  savedVoiceNameById,
}: {
  savedVoiceId: string | null | undefined;
  speaker: string | null | undefined;
  modelId: string | null | undefined;
  savedVoiceNameById: Record<string, string>;
}): string {
  if (savedVoiceId) {
    return savedVoiceNameById[savedVoiceId] || "Saved voice";
  }
  if (speaker) {
    const matchedSpeaker = getSpeakerProfilesForVariant(modelId ?? null).find(
      (profile) => profile.id === speaker,
    );
    if (matchedSpeaker) {
      return matchedSpeaker.name;
    }
    return speaker;
  }
  if (modelId) {
    return modelId;
  }
  return "Generated voice";
}

const DOWNLOAD_NAME_MAX_CHARS = 60;

/**
 * A take's file name: the opening words of what it says, which name it
 * better than its id. Characters file systems refuse are dropped.
 */
export function speechDownloadFilename(record: {
  id: string;
  input_text?: string | null;
  input_preview?: string | null;
  audio_mime_type: string;
}): string {
  const text = speechTextLabel(record.input_text ?? record.input_preview ?? "")
    .replace(/[\\/:*?"<>|]/g, " ")
    .replace(/\p{Cc}/gu, " ")
    .replace(/\s+/g, " ")
    .trim();
  const cut =
    text.length > DOWNLOAD_NAME_MAX_CHARS
      ? text.slice(0, DOWNLOAD_NAME_MAX_CHARS).replace(/\s+\S*$/, "")
      : text;
  const base = cut.replace(/[.\s]+$/, "") || `speech-${record.id.slice(0, 8)}`;
  const subtype = record.audio_mime_type.split("/")[1]?.split(";")[0] ?? "";
  const extension = subtype.includes("wav") ? "wav" : subtype || "audio";
  return `${base}.${extension}`;
}
