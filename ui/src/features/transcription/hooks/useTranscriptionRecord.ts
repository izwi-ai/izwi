import { api, type TranscriptionRecord } from "@/api";
import {
  type UseSpeechTextRecordResult,
  useSpeechTextRecord,
} from "@/features/speech-text/hooks/useSpeechTextRecord";

export type UseTranscriptionRecordResult =
  UseSpeechTextRecordResult<TranscriptionRecord>;

const getTranscriptionRecord = api.getTranscriptionRecord.bind(api);
const getSpeakerAttributedAsrRecord =
  api.getSpeakerAttributedAsrRecord.bind(api);

export function useTranscriptionRecord(
  recordId: string | null | undefined,
  options: { jobKind?: "transcription" | "speaker_attributed_asr" } = {},
): UseTranscriptionRecordResult {
  const getRecord =
    options.jobKind === "speaker_attributed_asr"
      ? getSpeakerAttributedAsrRecord
      : getTranscriptionRecord;

  return useSpeechTextRecord<TranscriptionRecord>({
    recordId,
    getRecord,
    loadErrorMessage: "Failed to load transcription record.",
    enablePolling: true,
  });
}
