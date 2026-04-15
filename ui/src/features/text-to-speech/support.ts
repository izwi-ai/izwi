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
