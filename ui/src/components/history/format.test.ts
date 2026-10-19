import { describe, expect, it } from "vitest";

import { formatClockDuration, formatHistoryDate } from "@/components/history/format";

describe("history formatters", () => {
  const now = new Date(2026, 9, 18, 15, 0);

  it("names today and yesterday and drops the year only for this year", () => {
    expect(formatHistoryDate(new Date(2026, 9, 18, 13, 15).getTime(), now)).toMatch(
      /^Today, /,
    );
    expect(formatHistoryDate(new Date(2026, 9, 17, 9, 2).getTime(), now)).toMatch(
      /^Yesterday, /,
    );
    expect(formatHistoryDate(new Date(2026, 9, 14).getTime(), now)).not.toMatch(
      /2026/,
    );
    expect(formatHistoryDate(new Date(2025, 7, 28).getTime(), now)).toMatch(/2025/);
  });

  it("formats lengths as a clock", () => {
    expect(formatClockDuration(4.2)).toBe("0:04");
    expect(formatClockDuration(1556)).toBe("25:56");
    expect(formatClockDuration(3730)).toBe("1:02:10");
    expect(formatClockDuration(null)).toBe("—");
  });
});
