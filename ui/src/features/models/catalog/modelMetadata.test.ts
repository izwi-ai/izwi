import { describe, expect, it } from "vitest";

import { getModelDisplayName, resolveModelDetail } from "./modelMetadata";

describe("model display metadata", () => {
  it("prefers the curated short name", () => {
    expect(
      getModelDisplayName("Qwen3-TTS-12Hz-0.6B-CustomVoice", "Qwen3-TTS 0.6B CustomVoice"),
    ).toBe("Qwen3 0.6B CustomVoice");
  });

  it("falls back to the API display name, then the raw id", () => {
    expect(getModelDisplayName("New-Model-1B", "New Model 1B")).toBe("New Model 1B");
    expect(getModelDisplayName("New-Model-1B")).toBe("New-Model-1B");
  });

  it("builds details for models the curated table does not list", () => {
    const detail = resolveModelDetail({
      variant: "New-Chat-1B",
      display_name: "New Chat 1B",
      task: "chat",
    });
    expect(detail.shortName).toBe("New Chat 1B");
    expect(detail.category).toBe("chat");

    expect(
      resolveModelDetail({ variant: "New-Aligner", task: "forced_align" }).category,
    ).toBe("asr");
  });
});
