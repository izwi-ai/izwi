import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import type { CursorPageResult, CursorPaginationQuery } from "@/api";

const HISTORY_PAGE_LIMIT = 25;
const HISTORY_POLL_INTERVAL_MS = 2500;

interface SpeechTextHistoryRecordBase {
  id: string;
  processing_status: "pending" | "processing" | "ready" | "failed";
  summary_status?: "not_requested" | "pending" | "ready" | "failed";
}

export interface UseSpeechTextHistoryResult<TSummary extends SpeechTextHistoryRecordBase> {
  records: TSummary[];
  loading: boolean;
  loadingMore: boolean;
  error: string | null;
  hasMoreRecords: boolean;
  loadMoreRecords: () => Promise<void>;
  refresh: () => Promise<void>;
}

interface UseSpeechTextHistoryOptions<TSummary extends SpeechTextHistoryRecordBase> {
  listPage: (query?: CursorPaginationQuery) => Promise<CursorPageResult<TSummary>>;
  loadErrorMessage: string;
  loadMoreErrorMessage: string;
  enablePolling?: boolean;
}

function defaultHistoryPollingPredicate(
  records: SpeechTextHistoryRecordBase[],
): boolean {
  return records.some((record) => {
    if (
      record.processing_status === "pending" ||
      record.processing_status === "processing"
    ) {
      return true;
    }

    return record.summary_status === "pending";
  });
}

export function useSpeechTextHistory<TSummary extends SpeechTextHistoryRecordBase>(
  options: UseSpeechTextHistoryOptions<TSummary>,
): UseSpeechTextHistoryResult<TSummary> {
  const { listPage, loadErrorMessage, loadMoreErrorMessage, enablePolling = true } =
    options;

  const [records, setRecords] = useState<TSummary[]>([]);
  const [loading, setLoading] = useState(true);
  const [loadingMore, setLoadingMore] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [nextCursor, setNextCursor] = useState<string | null>(null);
  const [hasMore, setHasMore] = useState(false);
  const recordsRef = useRef<TSummary[]>([]);

  useEffect(() => {
    recordsRef.current = records;
  }, [records]);

  const loadFirstPage = useCallback(
    async (background = false) => {
      const hasVisibleRecords = recordsRef.current.length > 0;
      const backgroundRefresh = background && hasVisibleRecords;
      if (!backgroundRefresh) {
        setLoading(true);
        setError(null);
      }

      try {
        const page = await listPage({
          limit: HISTORY_PAGE_LIMIT,
          cursor: null,
        });
        setRecords(page.items);
        setNextCursor(page.pagination.next_cursor);
        setHasMore(page.pagination.has_more);
        setError(null);
      } catch (err) {
        setError(err instanceof Error ? err.message : loadErrorMessage);
      } finally {
        setLoading(false);
      }
    },
    [listPage, loadErrorMessage],
  );

  const refresh = useCallback(async () => {
    await loadFirstPage(false);
  }, [loadFirstPage]);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  const pollingRequired = useMemo(
    () => defaultHistoryPollingPredicate(records),
    [records],
  );

  const pollingEnabled = records.length <= HISTORY_PAGE_LIMIT;

  useEffect(() => {
    if (!enablePolling || !pollingRequired || !pollingEnabled) {
      return;
    }

    const intervalId = window.setInterval(() => {
      void loadFirstPage(true);
    }, HISTORY_POLL_INTERVAL_MS);

    return () => {
      window.clearInterval(intervalId);
    };
  }, [enablePolling, loadFirstPage, pollingEnabled, pollingRequired]);

  const loadMoreRecords = useCallback(async () => {
    if (loading || loadingMore || !nextCursor || !hasMore) {
      return;
    }
    setLoadingMore(true);
    setError(null);
    try {
      const page = await listPage({
        limit: HISTORY_PAGE_LIMIT,
        cursor: nextCursor,
      });
      setRecords((current) => {
        const seen = new Set(current.map((record) => record.id));
        const nextItems = page.items.filter((record) => !seen.has(record.id));
        return [...current, ...nextItems];
      });
      setNextCursor(page.pagination.next_cursor);
      setHasMore(page.pagination.has_more);
    } catch (err) {
      setError(err instanceof Error ? err.message : loadMoreErrorMessage);
    } finally {
      setLoadingMore(false);
    }
  }, [hasMore, listPage, loadMoreErrorMessage, loading, loadingMore, nextCursor]);

  return {
    records,
    loading,
    loadingMore,
    error,
    hasMoreRecords: hasMore && Boolean(nextCursor),
    loadMoreRecords,
    refresh,
  };
}
