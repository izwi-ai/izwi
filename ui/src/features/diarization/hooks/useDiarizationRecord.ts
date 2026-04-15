import { api, type DiarizationRecord } from "@/api";
import {
  type UseSpeechTextRecordResult,
  useSpeechTextRecord,
} from "@/features/speech-text/hooks/useSpeechTextRecord";

export type UseDiarizationRecordResult =
  UseSpeechTextRecordResult<DiarizationRecord>;

export function useDiarizationRecord(
  recordId: string | null | undefined,
): UseDiarizationRecordResult {
  return useSpeechTextRecord<DiarizationRecord>({
    recordId,
    getRecord: api.getDiarizationRecord,
    loadErrorMessage: "Failed to load diarization record.",
    enablePolling: true,
  });
}
