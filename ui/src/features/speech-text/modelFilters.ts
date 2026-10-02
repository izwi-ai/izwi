import type { ModelInfo } from "@/api";
import { resolvePreferredRouteModel } from "@/features/models/catalog/routeModelCatalog";

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

/**
 * Models the speech pipelines rely on as a stack (diarization checkpoint +
 * ASR + forced aligner + refiner/summary LLM). Chat-route model switches
 * must never evict these: unloading one silently degrades the pipeline the
 * same way a too-small residency budget does.
 */
export function isSpeechPipelineManagedVariant(variant: string): boolean {
  return (
    isDiarizationVariant(variant) ||
    isDiarizationPipelineAsrVariant(variant) ||
    isDiarizationPipelineAlignerVariant(variant) ||
    isDiarizationPipelineLlmVariant(variant)
  );
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

/**
 * Resolves the diarization model a route should use. A user-selected
 * diarization variant that has vanished from the catalog surfaces as-is (the
 * readiness gate reports it as missing) instead of silently re-resolving to
 * the route's preferred model — the user's pick is the source of truth.
 *
 * When falling back (no selection, or a non-diarization one such as the last
 * loaded pipeline model), a READY diarization model wins over a not-loaded
 * preferred default: having loaded the stack should be enough to run.
 */
export function resolveDiarizationRouteModel(options: {
  models: ModelInfo[];
  selectedModel: string | null;
  preferredVariants: readonly string[];
}): string | null {
  const { models, selectedModel, preferredVariants } = options;
  if (
    selectedModel != null &&
    isDiarizationVariant(selectedModel) &&
    !models.some((model) => model.variant === selectedModel)
  ) {
    return selectedModel;
  }
  return resolvePreferredRouteModel({
    models,
    selectedModel,
    preferredVariants,
    preferAnyPreferredBeforeReadyAny: false,
  });
}
