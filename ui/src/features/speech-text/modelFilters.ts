import type { ModelInfo } from "@/api";
import {
  DIARIZATION_PREFERRED_SUMMARY_MODELS,
  resolvePreferredRouteModel,
} from "@/features/models/catalog/routeModelCatalog";

export const GRANITE_SPEECH_PLUS_VARIANT = "Granite-Speech-4.1-2B-Plus";

export function filterAndSortModels(
  models: ModelInfo[],
  matches: (model: ModelInfo) => boolean,
): ModelInfo[] {
  return models
    .filter(matches)
    .sort((left, right) => left.variant.localeCompare(right.variant));
}

type RouteCapabilityModel = Pick<ModelInfo, "route_capabilities">;

function hasCapability(
  model: RouteCapabilityModel,
  flag: keyof NonNullable<ModelInfo["route_capabilities"]>,
): boolean {
  return model.route_capabilities?.[flag] === true;
}

/**
 * Pipeline roles are selected by SERVER-REPORTED CAPABILITIES, never by model
 * id: any ASR model, aligner, diarization checkpoint, or chat model the
 * catalog marks with the matching route capability is eligible for the role.
 */
export function isDiarizationModel(model: ModelInfo): boolean {
  return hasCapability(model, "diarization_records");
}

export function isDiarizationPipelineAsrModel(model: ModelInfo): boolean {
  // Granite-Speech serves the dedicated speaker-attributed pipeline (its
  // transcripts are speaker turns, not plain text); the server's job kinds
  // keep the two apart and the pipeline groups mirror that split.
  if (model.variant === GRANITE_SPEECH_PLUS_VARIANT) {
    return false;
  }
  return hasCapability(model, "openai_audio_transcriptions");
}

export function isDiarizationPipelineAlignerModel(model: ModelInfo): boolean {
  return hasCapability(model, "forced_alignment");
}

export function isDiarizationPipelineLlmModel(model: ModelInfo): boolean {
  return hasCapability(model, "openai_chat_completions");
}

export function isTranscriptionAlignerModel(model: ModelInfo): boolean {
  return hasCapability(model, "forced_alignment");
}

export function isTranscriptionSummaryModel(model: ModelInfo): boolean {
  return hasCapability(model, "openai_chat_completions");
}

/**
 * Name-based display heuristic for a selected id that is no longer in the
 * catalog (no model object exists to read capabilities from). Selection
 * itself never consults this.
 */
export function isDiarizationVariant(variant: string): boolean {
  const normalized = variant.toLowerCase();
  return normalized.includes("sortformer") || normalized.includes("diar");
}

export function isSpeakerAttributedAsrVariant(variant: string): boolean {
  // Speaker-attributed ASR is a Granite-only pipeline today; the server
  // rejects other models for that job kind.
  return variant === GRANITE_SPEECH_PLUS_VARIANT;
}

/**
 * Models the speech pipelines rely on as a stack (diarization checkpoint +
 * ASR + forced aligner + refiner/summary LLM). Chat-route model switches
 * must never evict these: unloading one silently degrades the pipeline the
 * same way a too-small residency budget does.
 */
export function isSpeechPipelineManagedModel(model: ModelInfo): boolean {
  if (
    hasCapability(model, "diarization_records") ||
    hasCapability(model, "forced_alignment") ||
    hasCapability(model, "openai_audio_transcriptions")
  ) {
    return true;
  }
  // The refiner/summary LLM is a chat model: capabilities cannot separate it
  // from switchable chat models, so the preferred stack member stays
  // protected by preference (the same effective set as the previous
  // id-based rule).
  return (
    hasCapability(model, "openai_chat_completions") &&
    (DIARIZATION_PREFERRED_SUMMARY_MODELS as readonly string[]).includes(
      model.variant,
    )
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
