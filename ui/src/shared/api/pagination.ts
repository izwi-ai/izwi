export interface CursorPaginationMeta {
  next_cursor: string | null;
  has_more: boolean;
  limit: number;
}

export type ListFilterValue = string | number | boolean | null | undefined;

/**
 * Extra list query parameters (q, status, …). Empty strings, false, null and
 * undefined are left out, so "no filter" never reaches the server.
 */
export type ListFilters = Record<string, ListFilterValue>;

export interface CursorPaginationQuery {
  limit?: number;
  cursor?: string | null;
  filters?: ListFilters;
}

export interface CursorPageResult<T> {
  items: T[];
  pagination: CursorPaginationMeta;
  /** Rows matching the filters, ignoring paging, when the server reports it. */
  total?: number | null;
}

export function buildCursorQueryString(
  query?: CursorPaginationQuery,
): string {
  if (!query) {
    return "";
  }

  const params = new URLSearchParams();
  if (query.limit && Number.isFinite(query.limit) && query.limit > 0) {
    params.set("limit", String(Math.floor(query.limit)));
  }
  if (query.cursor && query.cursor.trim().length > 0) {
    params.set("cursor", query.cursor.trim());
  }
  for (const [key, value] of Object.entries(query.filters ?? {})) {
    if (value === null || value === undefined || value === false) {
      continue;
    }
    const serialized = String(value).trim();
    if (serialized.length > 0) {
      params.set(key, serialized);
    }
  }

  const serialized = params.toString();
  return serialized ? `?${serialized}` : "";
}

export function normalizeCursorPaginationMeta(
  value: Partial<CursorPaginationMeta> | null | undefined,
  fallbackLimit: number,
): CursorPaginationMeta {
  return {
    next_cursor: value?.next_cursor ?? null,
    has_more: Boolean(value?.has_more),
    limit:
      typeof value?.limit === "number" && Number.isFinite(value.limit)
        ? Math.max(1, Math.floor(value.limit))
        : Math.max(1, Math.floor(fallbackLimit)),
  };
}

/** The `total` a list response reports, or null when it has none. */
export function readListTotal(payload: { total?: unknown }): number | null {
  return typeof payload.total === "number" && Number.isFinite(payload.total)
    ? payload.total
    : null;
}
