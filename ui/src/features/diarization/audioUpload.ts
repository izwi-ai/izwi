export function formatDraftValue(value: number | null | undefined): string {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    return "";
  }
  if (Number.isInteger(value)) {
    return String(value);
  }
  return value.toFixed(2).replace(/\.?0+$/, "");
}

export function parseOptionalInteger(value: string): number | undefined {
  const trimmed = value.trim();
  if (!trimmed) {
    return undefined;
  }
  const parsed = Number.parseInt(trimmed, 10);
  return Number.isFinite(parsed) ? parsed : undefined;
}

export function clampIntegerDraft(
  value: string,
  fallback: number,
  min: number,
  max: number,
): number {
  const parsed = parseOptionalInteger(value) ?? fallback;
  return Math.max(min, Math.min(max, parsed));
}

export {
  prepareSpeechTextUploadBlob as prepareDiarizationUploadBlob,
  resolveSourceAudioFilename,
  resolveSpeechTextUploadFilename as resolveDiarizationUploadFilename,
  transcodeToWav,
} from "@/shared/audioUpload";
