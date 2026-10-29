import { describe, expect, it } from "vitest";

import {
  getModelDisplayName,
  getModelProviderLabel,
  resolveModelDetail,
} from "./modelMetadata";

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

  it("lists both LFM3 Audio variants as Liquid AI speech models", () => {
    expect(getModelDisplayName("LFM3-Audio-2B-GGUF", "LFM3 Audio 2B GGUF")).toBe(
      "LFM3 Audio 2B",
    );
    expect(
      getModelDisplayName("LFM3-Audio-2B-Q8_0-GGUF", "LFM3 Audio 2B Q8_0 GGUF"),
    ).toBe("LFM3 Audio 2B Q8_0");
    for (const [variant, size] of [
      ["LFM3-Audio-2B-GGUF", "1.46 GB"],
      ["LFM3-Audio-2B-Q8_0-GGUF", "2.20 GB"],
    ]) {
      const detail = resolveModelDetail({ variant, task: "audio_chat" });
      expect(detail.size).toBe(size);
      expect(detail.categories).toEqual(["asr", "tts"]);
      expect(detail.capabilities).toEqual(
        expect.arrayContaining(["Speech-to-speech", "Transcription", "Text to Speech"]),
      );
      expect(getModelProviderLabel(variant)).toBe("Liquid AI");
    }
    expect(getModelProviderLabel("LFM2.5-Audio-1.5B-GGUF")).toBe("Liquid AI");
    expect(getModelProviderLabel("LFM3-1.7B-GGUF")).toBe("Liquid AI");
  });
});
