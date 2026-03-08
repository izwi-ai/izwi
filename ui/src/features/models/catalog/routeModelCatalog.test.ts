import { describe, expect, it } from "vitest";

import {
  CHAT_PREFERRED_MODELS,
  resolvePreferredRouteModel,
} from "./routeModelCatalog";

describe("route model catalog", () => {
  it("prioritizes Qwen3.5-4B ahead of legacy Qwen3-8B defaults", () => {
    const selected = resolvePreferredRouteModel({
      models: [
        { variant: "Qwen3-8B-GGUF", status: "downloaded" },
        { variant: "Qwen3.5-4B", status: "ready" },
        { variant: "Qwen3.5-2B", status: "downloaded" },
      ],
      selectedModel: null,
      preferredVariants: CHAT_PREFERRED_MODELS,
      preferAnyPreferredBeforeReadyAny: true,
    });

    expect(selected).toBe("Qwen3.5-4B");
  });

  it("keeps an explicitly selected model when it is present", () => {
    const selected = resolvePreferredRouteModel({
      models: [
        { variant: "Qwen3.5-2B", status: "downloaded" },
        { variant: "Qwen3.5-4B", status: "ready" },
      ],
      selectedModel: "Qwen3.5-2B",
      preferredVariants: CHAT_PREFERRED_MODELS,
      preferAnyPreferredBeforeReadyAny: true,
    });

    expect(selected).toBe("Qwen3.5-2B");
  });
});
