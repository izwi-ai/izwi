import type { DiarizationSummaryStatus } from "@/api";

export function normalizeDiarizationSummaryStatus(
  status: string | null | undefined,
  summaryText?: string | null,
  summaryError?: string | null,
): DiarizationSummaryStatus {
  if (
    status === "not_requested" ||
    status === "pending" ||
    status === "ready" ||
    status === "failed"
  ) {
    return status;
  }
  if ((summaryText ?? "").trim().length > 0) {
    return "ready";
  }
  if ((summaryError ?? "").trim().length > 0) {
    return "failed";
  }
  return "not_requested";
}

export function diarizationSummaryStatusLabel(
  status: DiarizationSummaryStatus,
): string {
  switch (status) {
    case "pending":
      return "Summary pending";
    case "ready":
      return "Summary ready";
    case "failed":
      return "Summary failed";
    case "not_requested":
    default:
      return "Summary not requested";
  }
}

export function buildDiarizationSummaryPreview(
  summaryText: string | null | undefined,
  maxChars = 200,
): string | null {
  if (!summaryText) {
    return null;
  }
  const normalized = summaryText.trim().replace(/\s+/g, " ");
  if (!normalized) {
    return null;
  }
  if (normalized.length <= maxChars) {
    return normalized;
  }
  return `${normalized.slice(0, maxChars)}...`;
}
