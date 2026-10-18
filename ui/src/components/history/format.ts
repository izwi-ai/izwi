const DAY_MS = 24 * 60 * 60 * 1000;

function startOfDay(date: Date): number {
  return new Date(date.getFullYear(), date.getMonth(), date.getDate()).getTime();
}

/** "Today, 13:15", "Yesterday, 09:02", "14 Oct", "28 Aug 2025". */
export function formatHistoryDate(timestampMs: number, now = new Date()): string {
  const value = new Date(timestampMs);
  if (!Number.isFinite(timestampMs) || Number.isNaN(value.getTime())) {
    return "—";
  }
  const time = value.toLocaleTimeString([], {
    hour: "2-digit",
    minute: "2-digit",
  });
  const dayDelta = Math.round((startOfDay(now) - startOfDay(value)) / DAY_MS);
  if (dayDelta === 0) {
    return `Today, ${time}`;
  }
  if (dayDelta === 1) {
    return `Yesterday, ${time}`;
  }
  return value.toLocaleDateString([], {
    day: "numeric",
    month: "short",
    ...(value.getFullYear() === now.getFullYear() ? {} : { year: "numeric" }),
  });
}

/** Clock-style length: "0:04", "25:56", "1:02:10"; "—" when unknown. */
export function formatClockDuration(seconds: number | null | undefined): string {
  if (seconds == null || !Number.isFinite(seconds) || seconds < 0) {
    return "—";
  }
  const total = Math.round(seconds);
  const hours = Math.floor(total / 3600);
  const minutes = Math.floor((total % 3600) / 60);
  const secs = String(total % 60).padStart(2, "0");
  return hours > 0
    ? `${hours}:${String(minutes).padStart(2, "0")}:${secs}`
    : `${minutes}:${secs}`;
}

/** Midnight today in Unix milliseconds, for "Today" filters. */
export function startOfTodayMs(now = new Date()): number {
  return startOfDay(now);
}
