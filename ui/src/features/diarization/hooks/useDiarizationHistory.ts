import { api, type DiarizationRecordSummary } from "@/api";
import {
  type UseSpeechTextHistoryResult,
  useSpeechTextHistory,
} from "@/features/speech-text/hooks/useSpeechTextHistory";

export type UseDiarizationHistoryResult =
  UseSpeechTextHistoryResult<DiarizationRecordSummary>;

export function useDiarizationHistory(): UseDiarizationHistoryResult {
  return useSpeechTextHistory<DiarizationRecordSummary>({
    listPage: api.listDiarizationRecordPage,
    loadErrorMessage: "Failed to load diarization history.",
    loadMoreErrorMessage: "Failed to load more diarization history.",
    enablePolling: false,
  });
}
