import { describe, expect, it } from "vitest";

import { parseKeywords } from "./keywords";

describe("parseKeywords", () => {
  it("splits on lines and commas, collapses whitespace and drops duplicates", () => {
    expect(parseKeywords("Izwi,  Granite\n\nIBM   Research, Izwi\n")).toEqual([
      "Izwi",
      "Granite",
      "IBM Research",
    ]);
    expect(parseKeywords("  \n , ")).toEqual([]);
  });
});
