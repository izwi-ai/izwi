import { describe, expect, it } from "vitest";

import {
  CHAT_PREFERRED_MODELS,
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

  it("treats Qwen3.5 models as thinking-capable chat models", () => {
    expect(isThinkingChatModel("Qwen3.5-4B")).toBe(true);
    expect(isThinkingChatModel("Parakeet-TDT-0.6B-v3")).toBe(false);
  });
});
