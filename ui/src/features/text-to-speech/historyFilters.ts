import { startOfTodayMs } from "@/components/history/format";
import type { SegmentOption } from "@/components/history/HistoryToolbar";
import type { ListFilters } from "@/shared/api/pagination";

export type SpeechListFilter = "all" | "today" | "builtin" | "saved";

export const SPEECH_FILTER_OPTIONS: SegmentOption<SpeechListFilter>[] = [
  { id: "all", label: "All" },
  { id: "today", label: "Today" },
  { id: "builtin", label: "Built-in" },
  { id: "saved", label: "My voices" },
];

export function parseSpeechFilter(value: string | null): SpeechListFilter {
  return SPEECH_FILTER_OPTIONS.some((option) => option.id === value)
    ? (value as SpeechListFilter)
    : "all";
}

/** Query parameters for GET /text-to-speech/records. */
export function speechListFilters(
  search: string,
  filter: SpeechListFilter,
  now = new Date(),
): ListFilters {
  return {
    q: search.trim() || undefined,
    voice_kind: filter === "builtin" || filter === "saved" ? filter : undefined,
    created_after: filter === "today" ? startOfTodayMs(now) : undefined,
  };
}
