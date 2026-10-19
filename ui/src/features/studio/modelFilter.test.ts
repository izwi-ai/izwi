import { describe, expect, it } from "vitest";

import type { SpeechModelCapabilities } from "@/api";

import { isStudioRenderableModel } from "./modelFilter";

const capabilities = (
  builtIn: boolean,
  reference: boolean,
  description: boolean,
): SpeechModelCapabilities => ({
  supports_builtin_voices: builtIn,
  built_in_voice_count: builtIn ? 1 : 0,
  supports_reference_voice: reference,
  supports_voice_description: description,
  supports_streaming: true,
  supports_speed_control: false,
  supports_auto_long_form: false,
});

describe("isStudioRenderableModel", () => {
  it("offers models with built-in or saved voices", () => {
    expect(
      isStudioRenderableModel({
        variant: "Kokoro-82M",
        speech_capabilities: capabilities(true, false, false),
      }),
    ).toBe(true);
    expect(
      isStudioRenderableModel({
        variant: "Qwen3-TTS-12Hz-1.7B-Base",
        speech_capabilities: capabilities(false, true, false),
      }),
    ).toBe(true);
  });

  it("excludes description-only models the server would reject", () => {
    expect(
      isStudioRenderableModel({
        variant: "Qwen3-TTS-12Hz-1.7B-VoiceDesign",
        speech_capabilities: capabilities(false, false, true),
      }),
    ).toBe(false);
  });

  it("excludes tokenizers and non-speech models", () => {
    expect(
      isStudioRenderableModel({
        variant: "Qwen3-TTS-Tokenizer-12Hz",
        speech_capabilities: capabilities(true, false, false),
      }),
    ).toBe(false);
    expect(
      isStudioRenderableModel({ variant: "Qwen3.5-4B", speech_capabilities: null }),
    ).toBe(false);
    expect(isStudioRenderableModel(undefined)).toBe(false);
  });
});
