import { describe, expect, it } from "vitest";

import { buildCursorQueryString, readListTotal } from "@/shared/api/pagination";

describe("buildCursorQueryString", () => {
  it("adds set filters and leaves out empty ones", () => {
    expect(
      buildCursorQueryString({
        limit: 25,
        cursor: "abc",
        filters: {
          q: " fox ",
          status: "pending,processing",
          with_speakers: true,
          voice_kind: "",
          created_after: undefined,
          other: false,
        },
      }),
    ).toBe("?limit=25&cursor=abc&q=fox&status=pending%2Cprocessing&with_speakers=true");
  });
});

describe("readListTotal", () => {
  it("reads a numeric total only", () => {
    expect(readListTotal({ total: 63 })).toBe(63);
    expect(readListTotal({})).toBeNull();
    expect(readListTotal({ total: "63" })).toBeNull();
  });
});
