import { type SpeechHistoryProcessingStatus } from "@/api";
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
  const text = (record.input_text ?? record.input_preview ?? "")
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
