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
    expect(VIEW_CONFIGS.chat.modelFilter("Parakeet-TDT-0.6B-v3")).toBe(false);
    expect(VIEW_CONFIGS.chat.modelFilter("Qwen3-TTS-12Hz-0.6B-Base")).toBe(
      false,
    );
  });
});

describe("speech route model filters", () => {
  it("includes qwen3 asr gguf variants on transcription routes", () => {
    expect(VIEW_CONFIGS.transcription.modelFilter("Qwen3-ASR-0.6B-GGUF")).toBe(
      true,
    );
    expect(VIEW_CONFIGS.transcription.modelFilter("Qwen3-ASR-1.7B-GGUF")).toBe(
      true,
    );
  });

  it("includes vibevoice asr on transcription routes", () => {
    expect(VIEW_CONFIGS.transcription.modelFilter("VibeVoice-ASR")).toBe(true);
    expect(
      VIEW_CONFIGS.transcription.modelFilter("microsoft/VibeVoice-ASR"),
    ).toBe(true);
  });

  it("includes nemotron asr on transcription routes", () => {
    expect(
      VIEW_CONFIGS.transcription.modelFilter(
        "Nemotron-3.5-ASR-Streaming-0.6B",
      ),
    ).toBe(true);
    expect(
      VIEW_CONFIGS.transcription.modelFilter(
        "nvidia/nemotron-3.5-asr-streaming-0.6b",
      ),
    ).toBe(true);
  });

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

  it("includes voxtral tts on the standalone tts route filter", () => {
    expect(VIEW_CONFIGS["custom-voice"].modelFilter("Voxtral-4B-TTS-2603")).toBe(
      true,
    );
  });

  it("keeps voxtral tts off transcription routes", () => {
    expect(VIEW_CONFIGS.transcription.modelFilter("Voxtral-4B-TTS-2603")).toBe(
      false,
    );
    expect(
      VIEW_CONFIGS.transcription.modelFilter("mistralai/Voxtral-4B-TTS-2603"),
    ).toBe(false);
    expect(
      VIEW_CONFIGS.transcription.modelFilter(
        "mistralai/Voxtral-Mini-4B-Realtime-2602",
      ),
    ).toBe(true);
  });

  it("routes vibevoice tts to voice cloning only", () => {
    expect(VIEW_CONFIGS["voice-clone"].modelFilter("VibeVoice-1.5B")).toBe(
      true,
    );
    expect(VIEW_CONFIGS["custom-voice"].modelFilter("VibeVoice-1.5B")).toBe(
      false,
    );
    expect(VIEW_CONFIGS.transcription.modelFilter("VibeVoice-1.5B")).toBe(
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

  it("maps voxtral tts to the official preset voices", () => {
    expect(
      getSpeakerProfilesForVariant("mistralai/Voxtral-4B-TTS-2603").map(
        (speaker) => speaker.id,
      ),
    ).toEqual([
      "casual_female",
      "casual_male",
      "cheerful_female",
      "neutral_female",
      "neutral_male",
      "pt_male",
      "pt_female",
      "nl_male",
      "nl_female",
      "it_male",
      "it_female",
      "fr_male",
      "fr_female",
      "es_male",
      "es_female",
      "de_male",
      "de_female",
      "ar_male",
      "hi_male",
      "hi_female",
    ]);
  });

  it("does not surface builtin speaker presets for vibevoice tts", () => {
    expect(getSpeakerProfilesForVariant("microsoft/VibeVoice-1.5B")).toEqual(
      [],
    );
  });
});
