import { api, type TranscriptionRecord } from "@/api";
import {
  type UseSpeechTextRecordResult,
  useSpeechTextRecord,
} from "@/features/speech-text/hooks/useSpeechTextRecord";

export type UseTranscriptionRecordResult =
  UseSpeechTextRecordResult<TranscriptionRecord>;

export function useTranscriptionRecord(
  recordId: string | null | undefined,
): UseTranscriptionRecordResult {
  return useSpeechTextRecord<TranscriptionRecord>({
    recordId,
    getRecord: api.getTranscriptionRecord,
    loadErrorMessage: "Failed to load transcription record.",
    enablePolling: true,
  });
}
