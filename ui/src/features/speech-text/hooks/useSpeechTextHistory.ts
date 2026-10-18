import { useCallback } from "react";

import type { CursorPageResult, CursorPaginationQuery } from "@/api";
import {
  usePolledCursorHistory,
  type UsePolledCursorHistoryResult,
} from "@/hooks/usePolledCursorHistory";

interface SpeechTextHistoryRecordBase {
  id: string;
  created_at: number;
  processing_status: "pending" | "processing" | "ready" | "failed";
  summary_status?: "not_requested" | "pending" | "ready" | "failed";
}

export type UseSpeechTextHistoryResult<TSummary extends SpeechTextHistoryRecordBase> =
  UsePolledCursorHistoryResult<TSummary>;

interface UseSpeechTextHistoryOptions<TSummary extends SpeechTextHistoryRecordBase> {
  listPage: (query?: CursorPaginationQuery) => Promise<CursorPageResult<TSummary>>;
  loadErrorMessage: string;
  loadMoreErrorMessage: string;
  enablePolling?: boolean;
}

function isSpeechTextRecordActive(record: SpeechTextHistoryRecordBase): boolean {
  return (
    record.processing_status === "pending" ||
    record.processing_status === "processing" ||
    record.summary_status === "pending"
  );
}

export function useSpeechTextHistory<TSummary extends SpeechTextHistoryRecordBase>(
  options: UseSpeechTextHistoryOptions<TSummary>,
): UseSpeechTextHistoryResult<TSummary> {
  const isRecordActive = useCallback(
    (record: TSummary) => isSpeechTextRecordActive(record),
    [],
  );
  return usePolledCursorHistory({ ...options, isRecordActive });
}
