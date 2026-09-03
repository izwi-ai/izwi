import type { SpeechTextJobKind } from "@/api";

export type SpeechTextHistoryStatus =
  | "queued"
  | "processing"
  | "ready"
  | "failed"
  | "canceled"
  | "unknown";

export interface SpeechTextHistoryStatusPresentation {
  status: SpeechTextHistoryStatus;
  label: string;
  tone: "neutral" | "info" | "success" | "danger";
  description: string;
}

function jobLabel(kind: SpeechTextJobKind): string {
  switch (kind) {
    case "diarization":
      return "Diarization";
    case "speaker_attributed_asr":
      return "Speaker transcription";
    case "transcription":
    default:
      return "Transcription";
  }
}

export function presentSpeechTextHistoryStatus(
  status: string | null | undefined,
  processingError: string | null | undefined,
  kind: SpeechTextJobKind,
): SpeechTextHistoryStatusPresentation {
  const normalizedStatus = status?.trim().toLowerCase();
  const hasError = Boolean(processingError?.trim());
  const label = jobLabel(kind);

  if (hasError || normalizedStatus === "failed" || normalizedStatus === "error") {
    return {
      status: "failed",
      label: "Failed",
      tone: "danger",
      description:
        kind === "diarization"
          ? "Diarization failed. Review the error, then retry when you’re ready."
          : `${label} failed. Open the record to review the error and choose how to continue.`,
    };
  }

  switch (normalizedStatus) {
    case "pending":
    case "queued":
      return {
        status: "queued",
        label: "Queued",
        tone: "neutral",
        description: `${label} is queued and waiting to start.`,
      };
    case "processing":
    case "running":
      return {
        status: "processing",
        label: "Processing",
        tone: "info",
        description: `${label} is in progress.`,
      };
    case "ready":
    case "complete":
    case "completed":
      return {
        status: "ready",
        label: "Ready",
        tone: "success",
        description: `${label} is ready.`,
      };
    case "canceled":
    case "cancelled":
      return {
        status: "canceled",
        label: "Canceled",
        tone: "neutral",
        description: `${label} was canceled before completion.`,
      };
    default:
      return {
        status: "unknown",
        label: "Status Unknown",
        tone: "neutral",
        description: "Job status is unavailable. Open the record for details.",
      };
  }
}
