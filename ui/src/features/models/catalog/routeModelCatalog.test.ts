import { describe, expect, it } from "vitest";

import {
  CHAT_PREFERRED_MODELS,
  DIARIZATION_PREFERRED_ASR_MODELS,
  DIARIZATION_PREFERRED_MODELS,
  DIARIZATION_PREFERRED_SUMMARY_MODELS,
  TRANSCRIPTION_PREFERRED_MODELS,
  getChatRouteModelLabel,
  isThinkingChatModel,
  resolvePreferredRouteModel,
} from "./routeModelCatalog";

describe("route model catalog", () => {
  it("prioritizes Qwen3-8B as the default chat pick", () => {
    const selected = resolvePreferredRouteModel({
      models: [
        { variant: "Qwen3-8B-GGUF", status: "downloaded" },
        { variant: "Qwen3-4B-GGUF", status: "ready" },
        { variant: "Gemma-3-4b-it", status: "downloaded" },
      ],
      selectedModel: null,
      preferredVariants: CHAT_PREFERRED_MODELS,
      preferAnyPreferredBeforeReadyAny: true,
    });

    expect(selected).toBe("Qwen3-4B-GGUF");
  });

  it("keeps an explicitly selected model when it is present", () => {
    const selected = resolvePreferredRouteModel({
      models: [
        { variant: "Qwen3-1.7B-GGUF", status: "downloaded" },
        { variant: "Qwen3-4B-GGUF", status: "ready" },
      ],
      selectedModel: "Qwen3-1.7B-GGUF",
      preferredVariants: CHAT_PREFERRED_MODELS,
      preferAnyPreferredBeforeReadyAny: true,
    });

    expect(selected).toBe("Qwen3-1.7B-GGUF");
  });

  it("picks a ready Qwen3.5 model before an unloaded older preference", () => {
    const selected = resolvePreferredRouteModel({
      models: [
        { variant: "Qwen3-8B-GGUF", status: "downloaded" },
        { variant: "Qwen3.5-4B", status: "ready" },
      ],
      selectedModel: null,
      preferredVariants: CHAT_PREFERRED_MODELS,
      preferAnyPreferredBeforeReadyAny: true,
    });

    expect(selected).toBe("Qwen3.5-4B");
  });

  it("keeps diarization defaults anchored to the preferred pipeline variants", () => {
    const selected = resolvePreferredRouteModel({
      models: [
        { variant: "diar_streaming_sortformer_4spk-v2.1", status: "downloaded" },
        { variant: "diar_general_sortformer", status: "ready" },
      ],
      selectedModel: null,
      preferredVariants: DIARIZATION_PREFERRED_MODELS,
      preferAnyPreferredBeforeReadyAny: true,
    });

    expect(selected).toBe("diar_streaming_sortformer_4spk-v2.1");
  });

  it("falls back to the ready diarization summary model when it is available", () => {
    const selected = resolvePreferredRouteModel({
      models: [
        { variant: "SomeOtherLLM", status: "downloaded" },
        { variant: "Qwen3.5-4B", status: "ready" },
        { variant: "Parakeet-TDT-0.6B-v3", status: "ready" },
      ],
      selectedModel: null,
      preferredVariants: DIARIZATION_PREFERRED_SUMMARY_MODELS,
      preferAnyPreferredBeforeReadyAny: true,
    });

    expect(selected).toBe("Qwen3.5-4B");
  });

  it("prefers the diarization ASR pipeline variant over a non-preferred ready model", () => {
    const selected = resolvePreferredRouteModel({
      models: [
        { variant: "Whisper-Large-v3-Turbo", status: "downloaded" },
        { variant: "Parakeet-TDT-0.6B-v3", status: "ready" },
      ],
      selectedModel: null,
      preferredVariants: DIARIZATION_PREFERRED_ASR_MODELS,
      preferAnyPreferredBeforeReadyAny: true,
    });

    expect(selected).toBe("Whisper-Large-v3-Turbo");
  });

  it("keeps Nemotron discoverable without making it the default transcription pick", () => {
    expect(TRANSCRIPTION_PREFERRED_MODELS).toContain(
      "Nemotron-3.5-ASR-Streaming-0.6B",
    );
    expect(TRANSCRIPTION_PREFERRED_MODELS[0]).toBe("Qwen3-ASR-0.6B-GGUF");
  });

  it("treats Qwen3.5 models as thinking-capable chat models", () => {
    expect(isThinkingChatModel("Qwen3.5-4B")).toBe(true);
    expect(isThinkingChatModel("Parakeet-TDT-0.6B-v3")).toBe(false);
  });

  it("uses Qwen chat-route labels without injecting chat into the model name", () => {
    expect(getChatRouteModelLabel("Qwen3-0.6B-GGUF")).toBe(
      "Qwen3 0.6B GGUF (Q8_0)",
    );
    expect(getChatRouteModelLabel("Qwen3-8B-GGUF")).toBe(
      "Qwen3 8B GGUF (Q4_K_M)",
    );
    expect(getChatRouteModelLabel("Qwen3.5-4B")).toBe(
      "Qwen3.5 4B GGUF (Q4_K_M)",
    );
  });
});
