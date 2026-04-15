import { api, type TranscriptionRecordSummary } from "@/api";
import {
  type UseSpeechTextHistoryResult,
  useSpeechTextHistory,
} from "@/features/speech-text/hooks/useSpeechTextHistory";

export type UseTranscriptionHistoryResult =
  UseSpeechTextHistoryResult<TranscriptionRecordSummary>;

export function useTranscriptionHistory(): UseTranscriptionHistoryResult {
  return useSpeechTextHistory<TranscriptionRecordSummary>({
    listPage: api.listTranscriptionRecordPage,
    loadErrorMessage: "Failed to load transcription history.",
    loadMoreErrorMessage: "Failed to load more transcription history.",
    enablePolling: true,
  });
}
