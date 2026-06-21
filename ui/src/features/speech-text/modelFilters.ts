import type { ModelInfo } from "@/api";

export const GRANITE_SPEECH_PLUS_VARIANT = "Granite-Speech-4.1-2B-Plus";

export function filterAndSortModels(
  models: ModelInfo[],
  matchesVariant: (variant: string) => boolean,
): ModelInfo[] {
  return models
    .filter((model) => matchesVariant(model.variant))
    .sort((left, right) => left.variant.localeCompare(right.variant));
}

export function isTranscriptionAlignerVariant(variant: string): boolean {
  return variant === "Qwen3-ForcedAligner-0.6B";
}

export function isTranscriptionSummaryVariant(variant: string): boolean {
  return variant === "Qwen3.5-4B";
}

export function isDiarizationVariant(variant: string): boolean {
  const normalized = variant.toLowerCase();
  return normalized.includes("sortformer") || normalized.includes("diar");
}

export function isSpeakerAttributedAsrVariant(variant: string): boolean {
  return variant === GRANITE_SPEECH_PLUS_VARIANT;
}

export function isDiarizationPipelineAsrVariant(variant: string): boolean {
  return variant === "Whisper-Large-v3-Turbo";
}

export function isDiarizationPipelineAlignerVariant(variant: string): boolean {
  return variant === "Qwen3-ForcedAligner-0.6B";
}

export function isDiarizationPipelineLlmVariant(variant: string): boolean {
  return variant === "Qwen3.5-4B";
}

export function collectManagedModels(options: {
  availableModels: ModelInfo[];
  managedVariants: Array<string | null | undefined>;
}): ModelInfo[] {
  const variants = Array.from(
    new Set(
      options.managedVariants.filter(
        (variant): variant is string => typeof variant === "string" && variant.length > 0,
      ),
    ),
  );

  return variants
    .map(
      (variant) =>
        options.availableModels.find((model) => model.variant === variant) ?? null,
    )
    .filter((model): model is ModelInfo => model !== null);
}
