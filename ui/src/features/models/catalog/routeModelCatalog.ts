import type { ModelInfo } from "@/api";

type RouteModelLike = Pick<ModelInfo, "variant" | "status">;

export const TEXT_TO_SPEECH_PREFERRED_MODELS = [
  "Kokoro-82M",
  "Qwen3-TTS-12Hz-0.6B-CustomVoice",
  "Qwen3-TTS-12Hz-1.7B-CustomVoice",
] as const;

export const VOICE_CLONING_PREFERRED_MODELS = [
  "FishAudio-S2-Pro",
  "Qwen3-TTS-12Hz-0.6B-Base",
  "Qwen3-TTS-12Hz-1.7B-Base",
] as const;

export const VOICE_DESIGN_PREFERRED_MODELS = [
  "Qwen3-TTS-12Hz-1.7B-VoiceDesign",
] as const;

/**
 * Variants whose support was removed, with the supported counterpart a
 * stored selection is normalized to on read. The mlx 4-bit Qwen3-TTS
 * conversions never reach end of speech and produce unintelligible audio;
 * the mlx 4-bit forced aligner saved download size but no memory.
 */
export const REMOVED_MODEL_REPLACEMENTS: Readonly<Record<string, string>> = {
  "Qwen3-TTS-12Hz-0.6B-Base-4bit": "Qwen3-TTS-12Hz-0.6B-Base",
  "Qwen3-TTS-12Hz-0.6B-CustomVoice-4bit": "Qwen3-TTS-12Hz-0.6B-CustomVoice",
  "Qwen3-TTS-12Hz-1.7B-Base-4bit": "Qwen3-TTS-12Hz-1.7B-Base",
  "Qwen3-TTS-12Hz-1.7B-CustomVoice-4bit": "Qwen3-TTS-12Hz-1.7B-CustomVoice",
  "Qwen3-TTS-12Hz-1.7B-VoiceDesign-4bit": "Qwen3-TTS-12Hz-1.7B-VoiceDesign",
  "Qwen3-ForcedAligner-0.6B-4bit": "Qwen3-ForcedAligner-0.6B",
};

/** A stored model id with a removed variant replaced by its counterpart. */
export function normalizeStoredModelId(variant: string): string {
  return REMOVED_MODEL_REPLACEMENTS[variant] ?? variant;
}

export const TRANSCRIPTION_PREFERRED_MODELS = [
  "Qwen3-ASR-0.6B-GGUF",
  "Qwen3-ASR-1.7B-GGUF",
  "Parakeet-TDT-0.6B-v3",
  "Whisper-Large-v3-Turbo",
  "Nemotron-3.5-ASR-Streaming-0.6B",
  "Granite-Speech-5.0-2B",
] as const;

export const DIARIZATION_PREFERRED_MODELS = [
  "diar_streaming_sortformer_4spk-v2.1",
] as const;

/**
 * Highest speaker count a diarization model can resolve: v2.1 exposes four
 * channels, Nemotron-3-Diarization eight (its head is fixed at 8 and extra
 * channels emit near-zero probabilities when fewer speakers talk), and
 * Nemotron 3.5 Transcribe, diarizing alone, is capped at eight by the server.
 */
export function diarizationSpeakerUpperBound(
  variant: string | null | undefined,
): number {
  return variant === "Nemotron-3-Diarization" ||
    variant === "Nemotron-3.5-Transcribe-30B-A4B-FP8"
    ? 8
    : 4;
}

/**
 * Granite stays the default; Nemotron 3.5 Transcribe (about 32 GB of
 * memory) is the pick when it is loaded and Granite is not.
 */
export const SPEAKER_ATTRIBUTED_ASR_PREFERRED_MODELS = [
  "Granite-Speech-4.1-2B-Plus",
  "Nemotron-3.5-Transcribe-30B-A4B-FP8",
] as const;

export const DIARIZATION_PREFERRED_ASR_MODELS = [
  "Whisper-Large-v3-Turbo",
] as const;

export const DIARIZATION_PREFERRED_ALIGNER_MODELS = [
  "Qwen3-ForcedAligner-0.6B",
] as const;

export const DIARIZATION_PREFERRED_SUMMARY_MODELS = ["Qwen3.5-4B"] as const;

export const CHAT_PREFERRED_MODELS = [
  "Qwen3.5-4B",
  "Qwen3.5-9B",
  "Qwen3.8-27B-FP8",
  "Qwen3.5-2B",
  "Qwen3.5-0.8B",
] as const;

export function getModelStatusLabel(status: ModelInfo["status"]): string {
  switch (status) {
    case "ready":
      return "Loaded";
    case "loading":
      return "Loading";
    case "downloading":
      return "Downloading";
    case "downloaded":
      return "Downloaded";
    case "not_downloaded":
      return "Not downloaded";
    case "error":
      return "Error";
    default:
      return status;
  }
}

export function resolvePreferredRouteModel(options: {
  models: RouteModelLike[];
  selectedModel: string | null;
  preferredVariants: readonly string[];
  preferAnyPreferredBeforeReadyAny?: boolean;
}): string | null {
  const {
    models,
    selectedModel,
    preferredVariants,
    preferAnyPreferredBeforeReadyAny = false,
  } = options;

  if (selectedModel && models.some((model) => model.variant === selectedModel)) {
    return selectedModel;
  }

  const findPreferred = (requireReady: boolean) => {
    for (const variant of preferredVariants) {
      const match = models.find(
        (model) =>
          model.variant === variant &&
          (!requireReady || model.status === "ready"),
      );
      if (match) {
        return match.variant;
      }
    }

    return null;
  };

  const readyPreferred = findPreferred(true);
  if (readyPreferred) {
    return readyPreferred;
  }

  if (preferAnyPreferredBeforeReadyAny) {
    const preferred = findPreferred(false);
    if (preferred) {
      return preferred;
    }
  }

  const readyModel = models.find((model) => model.status === "ready");
  if (readyModel) {
    return readyModel.variant;
  }

  const preferred = findPreferred(false);
  if (preferred) {
    return preferred;
  }

  return models[0]?.variant ?? null;
}

export function getChatRouteModelLabel(variant: string): string {
  if (variant === "Qwen3.5-0.8B") {
    return "Qwen3.5 0.8B GGUF (Q4_K_M)";
  }
  if (variant === "Qwen3.5-2B") {
    return "Qwen3.5 2B GGUF (Q4_K_M)";
  }
  if (variant === "Qwen3.5-4B") {
    return "Qwen3.5 4B GGUF (Q4_K_M)";
  }
  if (variant === "Qwen3.5-9B") {
    return "Qwen3.5 9B GGUF (Q4_K_M)";
  }
  if (variant === "Qwen3.8-27B-FP8") {
    return "Qwen3.8 27B (FP8)";
  }
  if (variant === "Qwen3.6-35B-A3B-FP8") {
    return "Qwen3.6 35B-A3B (FP8 MoE)";
  }
  if (variant === "LFM2.5-1.2B-Instruct-GGUF") {
    return "LFM2.5 1.2B Instruct GGUF (Q4_K_M)";
  }
  if (variant === "LFM2.5-1.2B-Thinking-GGUF") {
    return "LFM2.5 1.2B Thinking GGUF (Q4_K_M)";
  }
  if (variant === "LFM3-1.7B-GGUF") {
    return "LFM3 1.7B GGUF (QAD Q4_0)";
  }
  if (variant === "LFM3-1.7B-Q8_0-GGUF") {
    return "LFM3 1.7B GGUF (Q8_0)";
  }
  if (variant === "Gemma-3-4b-it") {
    return "Gemma 3 4B Instruct";
  }
  return variant;
}
