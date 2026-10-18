import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import type { CursorPageResult, CursorPaginationQuery } from "@/api";
import type { ListFilters } from "@/shared/api/pagination";

const HISTORY_PAGE_LIMIT = 25;
const HISTORY_POLL_INTERVAL_MS = 2500;

export interface HistoryRecordBase {
  id: string;
  created_at: number;
}

export interface UsePolledCursorHistoryOptions<TRecord extends HistoryRecordBase> {
  listPage: (
    query?: CursorPaginationQuery,
  ) => Promise<CursorPageResult<TRecord>>;
  /** True while a record still changes server-side and needs polling. */
  isRecordActive: (record: TRecord) => boolean;
  loadErrorMessage: string;
  loadMoreErrorMessage: string;
  enablePolling?: boolean;
  /** Server-side filters; a change reloads the list from the first page. */
  filters?: ListFilters;
}

export interface UsePolledCursorHistoryResult<TRecord> {
  records: TRecord[];
  /** Rows matching the filters, when the server reports it. */
  total: number | null;
  loading: boolean;
  loadingMore: boolean;
  error: string | null;
  hasMoreRecords: boolean;
  loadMoreRecords: () => Promise<void>;
  refresh: () => Promise<void>;
}

/** Same order the server pages in: created_at DESC, then id DESC. */
function isOlderThan(record: HistoryRecordBase, boundary: HistoryRecordBase) {
  return (
    record.created_at < boundary.created_at ||
    (record.created_at === boundary.created_at && record.id < boundary.id)
  );
}

/**
 * Fold a freshly fetched first page into a list that may already hold later
 * pages. Rows in the page replace their old copies; rows older than the page
 * are kept so loaded pages survive a poll; rows inside the page's range that
 * the page no longer contains were deleted and are dropped.
 */
export function mergeRefreshedFirstPage<TRecord extends HistoryRecordBase>(
  current: TRecord[],
  firstPage: TRecord[],
  firstPageHasMore: boolean,
): TRecord[] {
  const oldestInPage = firstPage[firstPage.length - 1];
  if (!oldestInPage || !firstPageHasMore) {
    return firstPage;
  }
  const pageIds = new Set(firstPage.map((record) => record.id));
  const olderRows = current.filter(
    (record) => !pageIds.has(record.id) && isOlderThan(record, oldestInPage),
  );
  return [...firstPage, ...olderRows];
}

/**
 * Cursor-paginated history list that polls its first page while any loaded
 * record is still active. Polling merges the first page in place, so it keeps
 * running after "Load more", and responses that a newer request superseded
 * are ignored.
 */
export function usePolledCursorHistory<TRecord extends HistoryRecordBase>(
  options: UsePolledCursorHistoryOptions<TRecord>,
): UsePolledCursorHistoryResult<TRecord> {
  const {
    listPage,
    isRecordActive,
    loadErrorMessage,
    loadMoreErrorMessage,
    enablePolling = true,
    filters,
  } = options;
  // Callers usually pass a fresh object each render: key on its contents so
  // only a real change makes new loaders (and so reloads the list).
  const filtersKey = JSON.stringify(filters ?? {});
  const stableFilters = useMemo(
    () => JSON.parse(filtersKey) as ListFilters,
    [filtersKey],
  );

  const [records, setRecords] = useState<TRecord[]>([]);
  const [loading, setLoading] = useState(true);
  const [loadingMore, setLoadingMore] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [nextCursor, setNextCursor] = useState<string | null>(null);
  const [hasMore, setHasMore] = useState(false);
  const [total, setTotal] = useState<number | null>(null);
  const recordsRef = useRef<TRecord[]>([]);
  // Bumped by every first-page request; a response is applied only if no
  // newer first-page request started after it.
  const firstPageRequestRef = useRef(0);
  // Bumped by full (non-background) reloads, which reset the cursor; a
  // load-more that started before one belongs to a list that no longer exists.
  const listGenerationRef = useRef(0);

  useEffect(() => {
    recordsRef.current = records;
  }, [records]);

  const loadFirstPage = useCallback(
    async (background = false) => {
      const requestId = ++firstPageRequestRef.current;
      const backgroundRefresh = background && recordsRef.current.length > 0;
      if (!backgroundRefresh) {
        listGenerationRef.current += 1;
        setLoading(true);
        setError(null);
      }

      try {
        const page = await listPage({
          limit: HISTORY_PAGE_LIMIT,
          cursor: null,
          filters: stableFilters,
        });
        if (requestId !== firstPageRequestRef.current) {
          return;
        }
        setTotal(page.total ?? null);
        if (backgroundRefresh) {
          // Keep the cursor: it points past the oldest loaded row, which a
          // merge preserves.
          setRecords((current) =>
            mergeRefreshedFirstPage(current, page.items, page.pagination.has_more),
          );
          if (!page.pagination.has_more) {
            setNextCursor(null);
            setHasMore(false);
          }
        } else {
          setRecords(page.items);
          setNextCursor(page.pagination.next_cursor);
          setHasMore(page.pagination.has_more);
        }
        setError(null);
      } catch (err) {
        if (requestId === firstPageRequestRef.current) {
          setError(err instanceof Error ? err.message : loadErrorMessage);
        }
      } finally {
        if (requestId === firstPageRequestRef.current) {
          setLoading(false);
        }
      }
    },
    [listPage, loadErrorMessage, stableFilters],
  );

  const refresh = useCallback(async () => {
    await loadFirstPage(false);
  }, [loadFirstPage]);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  const pollingRequired = useMemo(
    () => records.some(isRecordActive),
    [isRecordActive, records],
  );

  useEffect(() => {
    if (!enablePolling || !pollingRequired) {
      return;
    }

    const intervalId = window.setInterval(() => {
      void loadFirstPage(true);
    }, HISTORY_POLL_INTERVAL_MS);

    return () => {
      window.clearInterval(intervalId);
    };
  }, [enablePolling, loadFirstPage, pollingRequired]);

  const loadMoreRecords = useCallback(async () => {
    if (loading || loadingMore || !nextCursor || !hasMore) {
      return;
    }
    const generation = listGenerationRef.current;
    setLoadingMore(true);
    setError(null);
    try {
      const page = await listPage({
        limit: HISTORY_PAGE_LIMIT,
        cursor: nextCursor,
        filters: stableFilters,
      });
      if (generation !== listGenerationRef.current) {
        return;
      }
      setRecords((current) => {
        const seen = new Set(current.map((record) => record.id));
        const nextItems = page.items.filter((record) => !seen.has(record.id));
        return [...current, ...nextItems];
      });
      setNextCursor(page.pagination.next_cursor);
      setHasMore(page.pagination.has_more);
    } catch (err) {
      if (generation === listGenerationRef.current) {
        setError(err instanceof Error ? err.message : loadMoreErrorMessage);
      }
    } finally {
      setLoadingMore(false);
    }
  }, [
    hasMore,
    listPage,
    loadMoreErrorMessage,
    loading,
    loadingMore,
    nextCursor,
    stableFilters,
  ]);

  return {
    records,
    total,
    loading,
    loadingMore,
    error,
    hasMoreRecords: hasMore && Boolean(nextCursor),
    loadMoreRecords,
    refresh,
  };
}
