import { describe, expect, it } from "vitest";

import {
  CHAT_PREFERRED_MODELS,
  DIARIZATION_PREFERRED_ASR_MODELS,
  DIARIZATION_PREFERRED_MODELS,
  DIARIZATION_PREFERRED_SUMMARY_MODELS,
  SPEAKER_ATTRIBUTED_ASR_PREFERRED_MODELS,
  TRANSCRIPTION_PREFERRED_MODELS,
  VOICE_CLONING_PREFERRED_MODELS,
  diarizationSpeakerUpperBound,
  getChatRouteModelLabel,
  resolvePreferredRouteModel,
} from "./routeModelCatalog";
import { getModelProviderLabel, MODEL_DETAILS } from "./modelMetadata";

describe("route model catalog", () => {
  it("keeps Qwen3.8 discoverable without making the 27B model the default", () => {
    expect(CHAT_PREFERRED_MODELS).toContain("Qwen3.8-27B-FP8");
    expect(CHAT_PREFERRED_MODELS[0]).toBe("Qwen3.5-4B");
  });

  it("describes Qwen3.8 as a text-only Qwen chat model", () => {
    expect(getModelProviderLabel("Qwen3.8-27B-FP8")).toBe("Qwen");
    expect(MODEL_DETAILS["Qwen3.8-27B-FP8"]?.category).toBe("chat");
    expect(MODEL_DETAILS["Qwen3.8-27B-FP8"]?.capabilities).not.toContain(
      "Multimodal",
    );
    expect(MODEL_DETAILS["Qwen3.8-27B-FP8"]?.capabilities).toContain(
      "Reasoning Effort",
    );
    expect(MODEL_DETAILS["Qwen3.8-27B-FP8"]?.description).not.toContain(
      "Qwen3.5",
    );
  });

  it("prefers a downloaded higher preference only when no preferred model is ready", () => {
    const selected = resolvePreferredRouteModel({
      models: [
        { variant: "Qwen3.5-9B", status: "downloaded" },
        { variant: "Qwen3.5-4B", status: "ready" },
        { variant: "Gemma-3-4b-it", status: "downloaded" },
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
        { variant: "Qwen3.5-9B", status: "downloaded" },
        { variant: "Qwen3.5-4B", status: "ready" },
      ],
      selectedModel: "Qwen3.5-9B",
      preferredVariants: CHAT_PREFERRED_MODELS,
      preferAnyPreferredBeforeReadyAny: true,
    });

    expect(selected).toBe("Qwen3.5-9B");
  });

  it("picks a ready Qwen3.5 model before an unloaded older preference", () => {
    const selected = resolvePreferredRouteModel({
      models: [
        { variant: "Qwen3.8-27B-FP8", status: "downloaded" },
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

  it("bounds the speaker draft by the selected checkpoint's channel count", () => {
    expect(diarizationSpeakerUpperBound("diar_streaming_sortformer_4spk-v2.1")).toBe(4);
    expect(diarizationSpeakerUpperBound("Nemotron-3-Diarization")).toBe(8);
    expect(diarizationSpeakerUpperBound(null)).toBe(4);
    expect(diarizationSpeakerUpperBound(undefined)).toBe(4);
    // The Nemotron-3 metadata row carries the 8-speaker capability pin.
    expect(MODEL_DETAILS["Nemotron-3-Diarization"].capabilities).toContain(
      "Up to 8 speakers",
    );
    expect(MODEL_DETAILS["Nemotron-3-Diarization"] && getModelProviderLabel("Nemotron-3-Diarization")).toBe(
      "NVIDIA",
    );
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

  it("keeps larger ASR models discoverable without making them the default transcription pick", () => {
    expect(TRANSCRIPTION_PREFERRED_MODELS).toContain(
      "Nemotron-3.5-ASR-Streaming-0.6B",
    );
    expect(TRANSCRIPTION_PREFERRED_MODELS).not.toContain(
      "Granite-Speech-4.1-2B-Plus",
    );
    expect(TRANSCRIPTION_PREFERRED_MODELS[0]).toBe("Qwen3-ASR-0.6B-GGUF");
  });

  it("keeps Fish S2 discoverable in the voice cloning route", () => {
    expect(VOICE_CLONING_PREFERRED_MODELS[0]).toBe("FishAudio-S2-Pro");
  });

  it("keeps Granite as the preferred speaker-attributed ASR model", () => {
    expect(SPEAKER_ATTRIBUTED_ASR_PREFERRED_MODELS).toEqual([
      "Granite-Speech-4.1-2B-Plus",
    ]);
    expect(DIARIZATION_PREFERRED_MODELS).not.toContain(
      "Granite-Speech-4.1-2B-Plus",
    );
  });

  it("uses Qwen chat-route labels without injecting chat into the model name", () => {
    expect(getChatRouteModelLabel("Qwen3.5-4B")).toBe(
      "Qwen3.5 4B GGUF (Q4_K_M)",
    );
    expect(getChatRouteModelLabel("Qwen3.5-9B")).toBe(
      "Qwen3.5 9B GGUF (Q4_K_M)",
    );
    expect(getChatRouteModelLabel("Qwen3.8-27B-FP8")).toBe(
      "Qwen3.8 27B (FP8)",
    );
  });
});
