import { describe, expect, it } from "vitest";

import { VIEW_CONFIGS, getSpeakerProfilesForVariant } from "@/types";

describe("VIEW_CONFIGS.chat.modelFilter", () => {
  it("includes the shipped Qwen3.5 chat variants", () => {
    expect(VIEW_CONFIGS.chat.modelFilter("Qwen3.5-0.8B")).toBe(true);
    expect(VIEW_CONFIGS.chat.modelFilter("Qwen3.5-2B")).toBe(true);
    expect(VIEW_CONFIGS.chat.modelFilter("Qwen3.5-4B")).toBe(true);
    expect(VIEW_CONFIGS.chat.modelFilter("Qwen3.5-9B")).toBe(true);
  });

  it("continues to reject non-chat variants", () => {
    expect(VIEW_CONFIGS.chat.modelFilter("Qwen3-ASR-0.6B")).toBe(false);
    expect(VIEW_CONFIGS.chat.modelFilter("Qwen3-TTS-12Hz-0.6B-Base")).toBe(
      false,
    );
  });
});

describe("speech route model filters", () => {
  it("includes lfm25 audio on transcription routes", () => {
    expect(VIEW_CONFIGS.transcription.modelFilter("LFM2.5-Audio-1.5B-GGUF")).toBe(
      true,
    );
  });

  it("keeps lfm25 audio off the standalone tts route filter", () => {
    expect(VIEW_CONFIGS["custom-voice"].modelFilter("LFM2.5-Audio-1.5B-GGUF")).toBe(
      false,
    );
  });
});

describe("getSpeakerProfilesForVariant", () => {
  it("maps lfm25 audio to its built-in speaker presets", () => {
    expect(
      getSpeakerProfilesForVariant("LFM2.5-Audio-1.5B-GGUF").map(
        (speaker) => speaker.id,
      ),
    ).toEqual(["US Female", "US Male", "UK Female", "UK Male"]);
  });
});
