import { act, renderHook, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import type { CursorPageResult, CursorPaginationQuery } from "@/api";

import {
  mergeRefreshedFirstPage,
  usePolledCursorHistory,
} from "./usePolledCursorHistory";

interface Row {
  id: string;
  created_at: number;
  active?: boolean;
}

const row = (id: string, created_at: number, active = false): Row => ({
  id,
  created_at,
  active,
});

function page(items: Row[], next_cursor: string | null): CursorPageResult<Row> {
  return {
    items,
    pagination: { next_cursor, has_more: next_cursor !== null, limit: 25 },
  };
}

describe("mergeRefreshedFirstPage", () => {
  const loaded = [row("e", 50), row("d", 40), row("c", 30), row("b", 20), row("a", 10)];

  it("replaces rows from the page and keeps older loaded rows", () => {
    const refreshed = [row("e", 50, true), row("d", 40), row("c", 30)];
    expect(mergeRefreshedFirstPage(loaded, refreshed, true).map((r) => r.id)).toEqual([
      "e",
      "d",
      "c",
      "b",
      "a",
    ]);
    expect(mergeRefreshedFirstPage(loaded, refreshed, true)[0].active).toBe(true);
  });

  it("drops rows inside the page range that the page no longer has", () => {
    const refreshed = [row("e", 50), row("c", 30)];
    expect(mergeRefreshedFirstPage(loaded, refreshed, true).map((r) => r.id)).toEqual([
      "e",
      "c",
      "b",
      "a",
    ]);
  });

  it("puts new rows first and keeps rows pushed past the page boundary", () => {
    const refreshed = [row("g", 70), row("f", 60), row("e", 50)];
    expect(mergeRefreshedFirstPage(loaded, refreshed, true).map((r) => r.id)).toEqual([
      "g",
      "f",
      "e",
      "d",
      "c",
      "b",
      "a",
    ]);
  });

  it("returns the page alone when the server has nothing more", () => {
    const refreshed = [row("e", 50), row("d", 40)];
    expect(mergeRefreshedFirstPage(loaded, refreshed, false)).toEqual(refreshed);
  });

  it("orders equal timestamps by id, like the server", () => {
    const current = [row("b", 10), row("a", 10)];
    expect(
      mergeRefreshedFirstPage(current, [row("b", 10)], true).map((r) => r.id),
    ).toEqual(["b", "a"]);
  });
});

describe("usePolledCursorHistory", () => {
  afterEach(() => {
    vi.useRealTimers();
  });

  function setup(listPage: (query?: CursorPaginationQuery) => Promise<CursorPageResult<Row>>) {
    return renderHook(() =>
      usePolledCursorHistory<Row>({
        listPage,
        isRecordActive: (record) => Boolean(record.active),
        loadErrorMessage: "load failed",
        loadMoreErrorMessage: "load more failed",
      }),
    );
  }

  it("keeps polling after load more and preserves loaded pages", async () => {
    vi.useFakeTimers({ shouldAdvanceTime: true });
    let firstPageCalls = 0;
    const listPage = vi.fn(async (query?: CursorPaginationQuery) => {
      if (query?.cursor === "page-2") {
        return page([row("b", 20), row("a", 10)], null);
      }
      firstPageCalls += 1;
      // The active job finishes on the second poll.
      const active = firstPageCalls < 3;
      return page([row("d", 40, active), row("c", 30)], "page-2");
    });

    const { result } = setup(listPage);
    await waitFor(() => expect(result.current.records).toHaveLength(2));
    await act(async () => {
      await result.current.loadMoreRecords();
    });
    expect(result.current.records.map((r) => r.id)).toEqual(["d", "c", "b", "a"]);

    await act(async () => {
      await vi.advanceTimersByTimeAsync(2600);
    });
    await act(async () => {
      await vi.advanceTimersByTimeAsync(2600);
    });

    await waitFor(() => expect(result.current.records[0].active).toBe(false));
    expect(result.current.records.map((r) => r.id)).toEqual(["d", "c", "b", "a"]);
    expect(result.current.hasMoreRecords).toBe(false);
  });

  it("ignores a first-page response superseded by a newer refresh", async () => {
    const resolvers: Array<(value: CursorPageResult<Row>) => void> = [];
    const listPage = vi.fn(
      () =>
        new Promise<CursorPageResult<Row>>((resolve) => {
          resolvers.push(resolve);
        }),
    );

    const { result } = setup(listPage);
    await waitFor(() => expect(resolvers).toHaveLength(1));
    let refreshDone: Promise<void> = Promise.resolve();
    act(() => {
      refreshDone = result.current.refresh();
    });
    await waitFor(() => expect(resolvers).toHaveLength(2));

    await act(async () => {
      resolvers[1](page([row("new", 20)], null));
      await refreshDone;
    });
    await act(async () => {
      resolvers[0](page([row("stale", 10)], null));
    });

    expect(result.current.records.map((r) => r.id)).toEqual(["new"]);
    expect(result.current.loading).toBe(false);
  });
});

describe("usePolledCursorHistory filters", () => {
  it("sends filters, reports the total and reloads only when they change", async () => {
    const listPage = vi.fn(async (query?: CursorPaginationQuery) => ({
      ...page([row(String(query?.filters?.q ?? "all"), 10)], null),
      total: query?.filters?.q ? 1 : 9,
    }));
    const { result, rerender } = renderHook(
      ({ q }: { q: string }) =>
        usePolledCursorHistory({
          listPage,
          isRecordActive: () => false,
          loadErrorMessage: "load failed",
          loadMoreErrorMessage: "more failed",
          filters: { q: q || undefined },
        }),
      { initialProps: { q: "" } },
    );

    await waitFor(() => expect(result.current.total).toBe(9));
    expect(listPage).toHaveBeenLastCalledWith({
      limit: 25,
      cursor: null,
      filters: {},
    });

    // A fresh but equal filters object does not reload.
    rerender({ q: "" });
    expect(listPage).toHaveBeenCalledTimes(1);

    rerender({ q: "fox" });
    await waitFor(() => expect(result.current.total).toBe(1));
    expect(result.current.records.map((r) => r.id)).toEqual(["fox"]);
    expect(listPage).toHaveBeenLastCalledWith({
      limit: 25,
      cursor: null,
      filters: { q: "fox" },
    });
  });
});
