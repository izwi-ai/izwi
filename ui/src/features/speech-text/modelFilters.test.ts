import { describe, expect, it } from "vitest";

import type { ModelInfo } from "@/api";
import {
  isDiarizationPipelineAsrModel,
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

  it("falls back to a ready diarization model when nothing is selected", () => {
    const resolved = resolveDiarizationRouteModel({
      models: [
        model("diar_streaming_sortformer_4spk-v2.1", "downloaded"),
        model("Nemotron-3-Diarization"),
      ],
      selectedModel: null,
      preferredVariants,
    });

    expect(resolved).toBe("Nemotron-3-Diarization");
  });

  it("resolves a loaded diarization model over a not-loaded preferred default", () => {
    // The user's scenario: the pipeline models were loaded (the aligner last,
    // so the global selection is not a diarization variant) and the preferred
    // default v2.1 was never loaded — the ready Nemotron must win.
    const resolved = resolveDiarizationRouteModel({
      models: [
        model("diar_streaming_sortformer_4spk-v2.1", "downloaded"),
        model("Nemotron-3-Diarization"),
      ],
      selectedModel: "Qwen3-ForcedAligner-0.6B",
      preferredVariants,
    });

    expect(resolved).toBe("Nemotron-3-Diarization");
  });

  it("keeps the preferred default when no diarization model is ready", () => {
    const resolved = resolveDiarizationRouteModel({
      models: [model("diar_streaming_sortformer_4spk-v2.1", "downloaded")],
      selectedModel: null,
      preferredVariants,
    });

    expect(resolved).toBe("diar_streaming_sortformer_4spk-v2.1");
  });

  it("prefers a ready preferred model over other ready models", () => {
    const resolved = resolveDiarizationRouteModel({
      models: [model("diar_streaming_sortformer_4spk-v2.1"), model("Nemotron-3-Diarization")],
      selectedModel: null,
      preferredVariants,
    });

    expect(resolved).toBe("diar_streaming_sortformer_4spk-v2.1");
  });
});

describe("isDiarizationPipelineAsrModel", () => {
  function withCapabilities(
    variant: string,
    capabilities: Partial<NonNullable<ModelInfo["route_capabilities"]>>,
  ): ModelInfo {
    return {
      ...model(variant),
      route_capabilities: {
        openai_chat_completions: false,
        openai_responses: false,
        openai_audio_speech: false,
        openai_audio_transcriptions: false,
        speech_to_text_jobs: false,
        speech_to_text_realtime: false,
        diarization_records: false,
        text_to_speech_records: false,
        voice_design_records: false,
        voice_clone_records: false,
        saved_voice_reuse: false,
        studio_projects: false,
        voice_realtime_text_model: false,
        voice_realtime_modular_asr: false,
        voice_realtime_modular_tts: false,
        voice_realtime_unified: false,
        forced_alignment: false,
        tokenizer: false,
        ...capabilities,
      },
    };
  }

  it("offers ASR models that run speech-to-text jobs", () => {
    expect(
      isDiarizationPipelineAsrModel(
        withCapabilities("Parakeet-TDT-0.6B-v3", {
          openai_audio_transcriptions: true,
          speech_to_text_jobs: true,
        }),
      ),
    ).toBe(true);
  });

  it("hides audio-chat models the pipeline cannot run", () => {
    expect(
      isDiarizationPipelineAsrModel(
        withCapabilities("LFM2.5-Audio-1.5B-GGUF", {
          openai_audio_transcriptions: true,
          voice_realtime_unified: true,
        }),
      ),
    ).toBe(false);
  });
});
