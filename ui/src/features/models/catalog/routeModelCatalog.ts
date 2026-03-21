import type { ModelInfo } from "@/api";
import { withQwen3Prefix } from "@/utils/modelDisplay";

type RouteModelLike = Pick<ModelInfo, "variant" | "status">;

export const TEXT_TO_SPEECH_PREFERRED_MODELS = [
  "Kokoro-82M",
  "Qwen3-TTS-12Hz-1.7B-CustomVoice-4bit",
  "Qwen3-TTS-12Hz-0.6B-CustomVoice-4bit",
  "Qwen3-TTS-12Hz-0.6B-CustomVoice-8bit",
  "Qwen3-TTS-12Hz-0.6B-CustomVoice-bf16",
  "Qwen3-TTS-12Hz-0.6B-CustomVoice",
  "Qwen3-TTS-12Hz-1.7B-CustomVoice",
] as const;

export const VOICE_CLONING_PREFERRED_MODELS = [
  "Qwen3-TTS-12Hz-0.6B-Base-4bit",
  "Qwen3-TTS-12Hz-0.6B-Base-8bit",
  "Qwen3-TTS-12Hz-0.6B-Base-bf16",
  "Qwen3-TTS-12Hz-0.6B-Base",
  "Qwen3-TTS-12Hz-1.7B-Base-4bit",
  "Qwen3-TTS-12Hz-1.7B-Base",
] as const;

export const VOICE_DESIGN_PREFERRED_MODELS = [
  "Qwen3-TTS-12Hz-1.7B-VoiceDesign-4bit",
  "Qwen3-TTS-12Hz-1.7B-VoiceDesign-8bit",
  "Qwen3-TTS-12Hz-1.7B-VoiceDesign-bf16",
  "Qwen3-TTS-12Hz-1.7B-VoiceDesign",
] as const;

export const TRANSCRIPTION_PREFERRED_MODELS = [
  "Parakeet-TDT-0.6B-v3",
  "Whisper-Large-v3-Turbo",
] as const;

export const CHAT_PREFERRED_MODELS = [
  "Qwen3-8B-GGUF",
  "Qwen3-4B-GGUF",
  "Qwen3.5-9B",
  "Qwen3.5-4B",
  "Qwen3-1.7B-GGUF",
  "Qwen3.5-2B",
  "Qwen3-0.6B-GGUF",
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
  if (variant === "Qwen3-0.6B-GGUF") {
    return withQwen3Prefix("0.6B GGUF (Q8_0)", variant);
  }
  if (variant === "Qwen3-1.7B-GGUF") {
    return withQwen3Prefix("1.7B GGUF (Q8_0)", variant);
  }
  if (variant === "Qwen3-4B-GGUF") {
    return withQwen3Prefix("4B GGUF (Q4_K_M)", variant);
  }
  if (variant === "Qwen3-8B-GGUF") {
    return withQwen3Prefix("8B GGUF (Q4_K_M)", variant);
  }
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
  if (variant === "LFM2.5-1.2B-Instruct-GGUF") {
    return "LFM2.5 1.2B Instruct GGUF (Q4_K_M)";
  }
  if (variant === "LFM2.5-1.2B-Thinking-GGUF") {
    return "LFM2.5 1.2B Thinking GGUF (Q4_K_M)";
  }
  if (variant === "Gemma-3-1b-it") {
    return "Gemma 3 1B Instruct";
  }
  if (variant === "Gemma-3-4b-it") {
    return "Gemma 3 4B Instruct";
  }
  return variant;
}

export function isThinkingChatModel(variant: string): boolean {
  const normalized = variant.trim().toLowerCase();
  const isQwenThinkingFamily =
    (normalized.startsWith("qwen3-") || normalized.startsWith("qwen3.5-")) &&
    !normalized.includes("-asr-") &&
    !normalized.includes("-tts-") &&
    !normalized.includes("forcedaligner");

  const isLfmThinkingVariant = normalized === "lfm2.5-1.2b-thinking-gguf";

  return isQwenThinkingFamily || isLfmThinkingVariant;
}
