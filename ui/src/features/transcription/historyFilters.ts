import type { SegmentOption } from "@/components/history/HistoryToolbar";
import type { ListFilters } from "@/shared/api/pagination";

export type TranscriptionListFilter = "all" | "active" | "speakers" | "failed";

export const TRANSCRIPTION_FILTER_OPTIONS: SegmentOption<TranscriptionListFilter>[] =
  [
    { id: "all", label: "All" },
    { id: "active", label: "In progress" },
    { id: "speakers", label: "With speakers" },
    { id: "failed", label: "Failed" },
  ];

export function parseTranscriptionFilter(
  value: string | null,
): TranscriptionListFilter {
  return TRANSCRIPTION_FILTER_OPTIONS.some((option) => option.id === value)
    ? (value as TranscriptionListFilter)
    : "all";
}

/** Query parameters for GET /transcription/jobs. */
export function transcriptionListFilters(
  search: string,
  filter: TranscriptionListFilter,
): ListFilters {
  return {
    q: search.trim() || undefined,
    status:
      filter === "active"
        ? "pending,processing"
        : filter === "failed"
          ? "failed"
          : undefined,
    with_speakers: filter === "speakers" || undefined,
  };
}
