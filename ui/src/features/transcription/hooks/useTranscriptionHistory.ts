import { api, type SpeechTextJobSummary } from "@/api";
import {
  type UseSpeechTextHistoryResult,
  useSpeechTextHistory,
} from "@/features/speech-text/hooks/useSpeechTextHistory";
import type { CursorPaginationQuery } from "@/shared/api/pagination";

export type UseTranscriptionHistoryResult =
  UseSpeechTextHistoryResult<SpeechTextJobSummary>;

function listSpeechTextHistoryPage(query?: CursorPaginationQuery) {
  return api.listSpeechTextJobPage(query);
}

export function useTranscriptionHistory(): UseTranscriptionHistoryResult {
  return useSpeechTextHistory<SpeechTextJobSummary>({
    listPage: listSpeechTextHistoryPage,
    loadErrorMessage: "Failed to load speech-text history.",
    loadMoreErrorMessage: "Failed to load more speech-text history.",
    enablePolling: true,
  });
}
