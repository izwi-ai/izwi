import { describe, expect, it } from "vitest";

import type { ModelInfo } from "@/api";
import {
  resolveDiarizationRouteModel,
} from "@/features/speech-text/modelFilters";

function model(variant: string, status: ModelInfo["status"] = "ready"): ModelInfo {
  return {
    variant,
    status,
    local_path: `/models/${variant}`,
    size_bytes: null,
    download_progress: null,
    error_message: null,
  };
}

describe("resolveDiarizationRouteModel", () => {
  const preferredVariants = ["diar_streaming_sortformer_4spk-v2.1"] as const;

  it("honors a selected diarization variant that is present in the catalog", () => {
    const resolved = resolveDiarizationRouteModel({
      models: [model("diar_streaming_sortformer_4spk-v2.1", "downloaded")],
      selectedModel: "diar_streaming_sortformer_4spk-v2.1",
      preferredVariants,
    });

    expect(resolved).toBe("diar_streaming_sortformer_4spk-v2.1");
  });

  it("surfaces a selected diarization variant that vanished from the catalog", () => {
    const resolved = resolveDiarizationRouteModel({
      models: [model("diar_streaming_sortformer_4spk-v2.1")],
      selectedModel: "Nemotron-3-Diarization",
      preferredVariants,
    });

    expect(resolved).toBe("Nemotron-3-Diarization");
  });

  it("falls back to the preferred list for a non-diarization selection", () => {
    const resolved = resolveDiarizationRouteModel({
      models: [model("diar_streaming_sortformer_4spk-v2.1", "downloaded")],
      selectedModel: "Qwen3.5-4B",
      preferredVariants,
    });

    expect(resolved).toBe("diar_streaming_sortformer_4spk-v2.1");
  });

  it("falls back to the preferred list when nothing is selected", () => {
    const resolved = resolveDiarizationRouteModel({
      models: [
        model("diar_streaming_sortformer_4spk-v2.1", "downloaded"),
        model("Nemotron-3-Diarization"),
      ],
      selectedModel: null,
      preferredVariants,
    });

    expect(resolved).toBe("diar_streaming_sortformer_4spk-v2.1");
  });
});
