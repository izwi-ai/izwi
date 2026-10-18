import { api, type SpeechHistoryRecordSummary } from "@/api";
import { normalizeSpeechProcessingStatus } from "@/features/text-to-speech/support";
import {
  usePolledCursorHistory,
  type UsePolledCursorHistoryResult,
} from "@/hooks/usePolledCursorHistory";

export type UseTextToSpeechHistoryResult =
  UsePolledCursorHistoryResult<SpeechHistoryRecordSummary>;

const listTextToSpeechPage: typeof api.listTextToSpeechRecordPage = (query) =>
  api.listTextToSpeechRecordPage(query);

function isTextToSpeechRecordActive(record: SpeechHistoryRecordSummary): boolean {
  const status = normalizeSpeechProcessingStatus(
    record.processing_status,
    record.processing_error,
  );
  return status === "pending" || status === "processing";
}

export function useTextToSpeechHistory(): UseTextToSpeechHistoryResult {
  return usePolledCursorHistory({
    listPage: listTextToSpeechPage,
    isRecordActive: isTextToSpeechRecordActive,
    loadErrorMessage: "Failed to load text-to-speech history.",
    loadMoreErrorMessage: "Failed to load more text-to-speech history.",
  });
}
