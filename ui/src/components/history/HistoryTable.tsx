import type { KeyboardEvent, MouseEvent, ReactNode } from "react";
import { AlertTriangle, Loader2 } from "lucide-react";

import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";

export interface HistoryColumn<T> {
  id: string;
  /** Column heading; pass a visually hidden label for icon columns. */
  header: ReactNode;
  /** Applied to the heading and every cell: widths and alignment. */
  className?: string;
  /** Applied to body cells only: text colour, size, numerals. */
  cellClassName?: string;
  cell: (row: T) => ReactNode;
}

interface HistoryTableProps<T> {
  rows: T[];
  columns: HistoryColumn<T>[];
  getRowId: (row: T) => string;
  /** Accessible name of the row's open action: "Open {label}". */
  getRowLabel: (row: T) => string;
  onOpenRow: (row: T) => void;
  /** Trailing cell: quick actions and the row menu. */
  rowActions?: (row: T) => ReactNode;
  rowClassName?: (row: T) => string | undefined;
  loading: boolean;
  error: string | null;
  onRetry?: () => void;
  /** Shown when there are no rows and no filter is active. */
  empty: { title: string; description: string; action?: ReactNode };
  /** True when search or a filter narrows the list. */
  filtered?: boolean;
  /** Rows matching the filters, if the server reports it. */
  total: number | null;
  /** Plural noun for the footer count ("transcripts"). */
  noun: string;
  loadMore?: { canLoadMore: boolean; loading: boolean; onLoadMore: () => void };
}

/** Clicks inside portaled menus bubble through React; only rows count. */
function isRowActivation(event: MouseEvent | KeyboardEvent): boolean {
  const target = event.target as Node;
  if (!event.currentTarget.contains(target)) {
    return false;
  }
  return !(target instanceof Element && target.closest("[data-row-action]"));
}

/**
 * History list shared by the record routes: clickable rows that open the
 * record, a trailing actions cell, loading/error/empty states, and a footer
 * with the count and "Load more".
 */
export function HistoryTable<T>({
  rows,
  columns,
  getRowId,
  getRowLabel,
  onOpenRow,
  rowActions,
  rowClassName,
  loading,
  error,
  onRetry,
  empty,
  filtered = false,
  total,
  noun,
  loadMore,
}: HistoryTableProps<T>) {
  if (loading && rows.length === 0) {
    return (
      <div className="flex min-h-64 items-center justify-center rounded-[14px] border border-(--border-subtle) bg-(--surface-panel) text-sm text-(--text-muted)">
        <Loader2 className="mr-2 h-4 w-4 animate-spin" />
        Loading {noun}…
      </div>
    );
  }

  if (error && rows.length === 0) {
    return (
      <div
        role="alert"
        className="flex items-start justify-between gap-3 rounded-[14px] border border-(--danger-border) bg-(--danger-bg) p-4 text-sm text-(--danger-text)"
      >
        <div className="flex items-start gap-3">
          <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0" />
          <p>{error}</p>
        </div>
        {onRetry ? (
          <Button type="button" variant="outline" size="sm" onClick={onRetry}>
            Retry
          </Button>
        ) : null}
      </div>
    );
  }

  if (rows.length === 0) {
    return (
      <div className="rounded-[14px] border border-(--border-subtle) bg-(--surface-panel) px-6 py-12 text-center">
        <h3 className="text-base font-semibold text-(--text-primary)">
          {filtered ? `No matching ${noun}` : empty.title}
        </h3>
        <p className="mx-auto mt-1.5 max-w-md text-sm text-(--text-muted)">
          {filtered
            ? "Try a different search or filter."
            : empty.description}
        </p>
        {!filtered && empty.action ? (
          <div className="mt-4 flex justify-center">{empty.action}</div>
        ) : null}
      </div>
    );
  }

  return (
    <div
      className={cn(
        "overflow-hidden rounded-[14px] border border-(--border-subtle) bg-(--surface-panel) transition-opacity",
        loading && "opacity-60",
      )}
      aria-busy={loading || undefined}
    >
      <div className="overflow-x-auto">
        <table className="w-full min-w-lg border-separate border-spacing-0 text-[13.5px]">
          <thead>
            <tr>
              {columns.map((column) => (
                <th
                  key={column.id}
                  scope="col"
                  className={cn(
                    "border-b border-(--border-subtle) px-3.5 py-2.5 text-left text-xs font-medium text-(--text-subtle)",
                    column.className,
                  )}
                >
                  {column.header}
                </th>
              ))}
              {rowActions ? (
                <th
                  scope="col"
                  className="w-[1%] border-b border-(--border-subtle) px-3.5 py-2.5"
                >
                  <span className="sr-only">Actions</span>
                </th>
              ) : null}
            </tr>
          </thead>
          <tbody>
            {rows.map((row) => (
              <tr
                key={getRowId(row)}
                tabIndex={0}
                aria-label={`Open ${getRowLabel(row)}`}
                className={cn(
                  "group cursor-pointer outline-hidden transition-colors hover:bg-(--bg-surface-1) focus-visible:bg-(--bg-surface-1) [&:last-child>td]:border-b-0",
                  rowClassName?.(row),
                )}
                onClick={(event) => {
                  if (isRowActivation(event)) {
                    onOpenRow(row);
                  }
                }}
                onKeyDown={(event) => {
                  if (
                    (event.key === "Enter" || event.key === " ") &&
                    isRowActivation(event)
                  ) {
                    event.preventDefault();
                    onOpenRow(row);
                  }
                }}
              >
                {columns.map((column) => (
                  <td
                    key={column.id}
                    className={cn(
                      "border-b border-(--border-subtle) px-3.5 py-2.5 align-middle",
                      column.className,
                      column.cellClassName,
                    )}
                  >
                    {column.cell(row)}
                  </td>
                ))}
                {rowActions ? (
                  <td className="border-b border-(--border-subtle) px-2.5 py-2 align-middle">
                    <div
                      data-row-action
                      className="flex items-center justify-end gap-0.5"
                    >
                      {rowActions(row)}
                    </div>
                  </td>
                ) : null}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <div className="flex flex-wrap items-center justify-between gap-2 border-t border-(--border-subtle) px-3.5 py-2.5 text-[12.5px] text-(--text-subtle)">
        <span>
          {total !== null && total >= rows.length
            ? `Showing ${rows.length} of ${total} ${noun}`
            : `Showing ${rows.length} ${noun}`}
        </span>
        {error ? (
          <span role="alert" className="text-(--danger-text)">
            {error}
          </span>
        ) : null}
        {loadMore?.canLoadMore ? (
          <Button
            type="button"
            variant="outline"
            size="sm"
            className="h-7 gap-2 px-2.5 text-[12.5px]"
            onClick={loadMore.onLoadMore}
            disabled={loadMore.loading}
          >
            {loadMore.loading ? (
              <Loader2 className="h-3.5 w-3.5 animate-spin" />
            ) : null}
            Load more
          </Button>
        ) : null}
      </div>
    </div>
  );
}
