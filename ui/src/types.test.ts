import { describe, expect, it } from "vitest";

import {
  VIEW_CONFIGS,
  getSpeakerProfilesForVariant,
  isLfm25AudioVariant,
  isLfm3AudioVariant,
} from "@/types";

const LFM3_AUDIO_VARIANTS = ["LFM3-Audio-2B-GGUF", "LFM3-Audio-2B-Q8_0-GGUF"];

describe("VIEW_CONFIGS.chat.modelFilter", () => {
  it("includes the shipped Qwen3.5 chat variants", () => {
    expect(VIEW_CONFIGS.chat.modelFilter("Qwen3.5-0.8B")).toBe(true);
    expect(VIEW_CONFIGS.chat.modelFilter("Qwen3.5-2B")).toBe(true);
    expect(VIEW_CONFIGS.chat.modelFilter("Qwen3.5-4B")).toBe(true);
    expect(VIEW_CONFIGS.chat.modelFilter("Qwen3.5-9B")).toBe(true);
  });

  it("includes Qwen3.8 text chat", () => {
    expect(VIEW_CONFIGS.chat.modelFilter("Qwen3.8-27B-FP8")).toBe(true);
  });

  it("leaves LFM3 Audio off chat, though it still transcribes and speaks", () => {
    for (const variant of LFM3_AUDIO_VARIANTS) {
      expect(isLfm3AudioVariant(variant)).toBe(true);
      expect(VIEW_CONFIGS.chat.modelFilter(variant)).toBe(false);
      expect(VIEW_CONFIGS.transcription.modelFilter(variant)).toBe(true);
      expect(VIEW_CONFIGS["custom-voice"].modelFilter(variant)).toBe(true);
    }
    expect(VIEW_CONFIGS.chat.modelFilter("LFM3-1.7B-GGUF")).toBe(true);
    expect(VIEW_CONFIGS.chat.modelFilter("LFM3-1.7B-Q8_0-GGUF")).toBe(true);
  });

  it("continues to reject non-chat variants", () => {
    expect(VIEW_CONFIGS.chat.modelFilter("Parakeet-TDT-0.6B-v3")).toBe(false);
    expect(VIEW_CONFIGS.chat.modelFilter("Qwen3-TTS-12Hz-0.6B-Base")).toBe(
      false,
    );
    expect(VIEW_CONFIGS.chat.modelFilter("LFM2.5-Audio-1.5B-GGUF")).toBe(false);
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

  it("includes nemotron 3.5 transcribe on transcription routes", () => {
    expect(
      VIEW_CONFIGS.transcription.modelFilter(
        "Nemotron-3.5-Transcribe-30B-A4B-FP8",
      ),
    ).toBe(true);
    expect(
      VIEW_CONFIGS.transcription.modelFilter(
        "nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B-FP8",
      ),
    ).toBe(true);
    // Only this checkpoint: an unsupported quantization and the diarizer
    // stay off, and the streaming ASR keeps its own entry.
    expect(
      VIEW_CONFIGS.transcription.modelFilter(
        "NVIDIA-Nemotron-3.5-Transcribe-30B-A4B-NVFP4",
      ),
    ).toBe(false);
    expect(
      VIEW_CONFIGS.transcription.modelFilter("Nemotron-3-Diarization"),
    ).toBe(false);
    expect(
      VIEW_CONFIGS.transcription.modelFilter(
        "Nemotron-3.5-ASR-Streaming-0.6B",
      ),
    ).toBe(true);
  });

  it("includes granite speech on regular transcription routes", () => {
    expect(
      VIEW_CONFIGS.transcription.modelFilter("Granite-Speech-4.1-2B"),
    ).toBe(true);
    expect(
      VIEW_CONFIGS.transcription.modelFilter("ibm-granite/granite-speech-4.1-2b"),
    ).toBe(true);
    expect(
      VIEW_CONFIGS.transcription.modelFilter("Granite-Speech-4.1-2B-Plus"),
    ).toBe(true);
    expect(
      VIEW_CONFIGS.transcription.modelFilter(
        "ibm-granite/granite-speech-4.1-2b-plus",
      ),
    ).toBe(true);
    expect(
      VIEW_CONFIGS.transcription.modelFilter("Granite-Speech-5.0-2B"),
    ).toBe(true);
    expect(
      VIEW_CONFIGS.transcription.modelFilter("ibm-granite/granite-speech-5.0-2b"),
    ).toBe(true);
    expect(
      VIEW_CONFIGS.transcription.modelFilter("Granite-Speech-5.0-2B-GGUF"),
    ).toBe(true);
    expect(
      VIEW_CONFIGS.transcription.modelFilter(
        "ibm-granite/granite-speech-5.0-2b-GGUF",
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

  it("puts lfm3 audio on both the transcription and tts route filters", () => {
    for (const variant of LFM3_AUDIO_VARIANTS) {
      expect(VIEW_CONFIGS.transcription.modelFilter(variant)).toBe(true);
      expect(VIEW_CONFIGS["custom-voice"].modelFilter(variant)).toBe(true);
      expect(VIEW_CONFIGS["voice-clone"].modelFilter(variant)).toBe(false);
      expect(VIEW_CONFIGS["voice-design"].modelFilter(variant)).toBe(false);
    }
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

describe("audio model ids", () => {
  it("tells LFM3 Audio from LFM2.5 Audio and LFM3 chat", () => {
    for (const variant of [
      ...LFM3_AUDIO_VARIANTS,
      "lfm3-audio-2b-q8_0-gguf",
      " LiquidAI/LFM3-Audio-2B-GGUF ",
      "LiquidAI/LFM3-Audio-2B",
    ]) {
      expect(isLfm3AudioVariant(variant)).toBe(true);
      expect(isLfm25AudioVariant(variant)).toBe(false);
    }
    expect(isLfm3AudioVariant("LFM2.5-Audio-1.5B-GGUF")).toBe(false);
    expect(isLfm3AudioVariant("LFM3-1.7B-GGUF")).toBe(false);
    expect(isLfm25AudioVariant("LFM2.5-Audio-1.5B-GGUF")).toBe(true);
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

  it("maps lfm3 audio to its two multilingual voices", () => {
    for (const variant of [...LFM3_AUDIO_VARIANTS, "LiquidAI/LFM3-Audio-2B"]) {
      expect(
        getSpeakerProfilesForVariant(variant).map(({ id, name, language }) => ({
          id,
          name,
          language,
        })),
      ).toEqual([
        { id: "female", name: "Female", language: "Multilingual" },
        { id: "male", name: "Male", language: "Multilingual" },
      ]);
    }
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

describe("Kokoro speakers", () => {
  it("lists only the 41 voices the server speaks", () => {
    // Japanese (j*) and Mandarin (z*) voices have no G2P yet: the server
    // hides them and refuses them with a 400.
    const ids = getSpeakerProfilesForVariant("Kokoro-82M").map(
      (speaker) => speaker.id,
    );
    expect(ids).toHaveLength(41);
    expect(ids.filter((id) => /^[jz]/.test(id))).toEqual([]);
  });
});
