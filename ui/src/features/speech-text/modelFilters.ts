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
  // The pipeline transcribes as a speech-to-text job: audio-chat models
  // serve /audio/transcriptions but cannot run one, and the server rejects
  // them as the pipeline ASR.
  return (
    hasCapability(model, "openai_audio_transcriptions") &&
    hasCapability(model, "speech_to_text_jobs")
  );
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

export function isSpeakerAttributedAsrVariant(variant: string): boolean {
  // Speaker-attributed ASR is a Granite-only pipeline today; the server
  // rejects other models for that job kind.
  return variant === GRANITE_SPEECH_PLUS_VARIANT;
}

/**
 * Models the speech pipelines rely on as a stack (diarization checkpoint +
 * ASR + forced aligner + summary LLM). Chat-route model switches
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
  // The summary LLM is a chat model: capabilities cannot separate it
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

/** One diarization-stack role (diarizer, ASR or aligner), resolved. */
export interface DiarizationStackRole {
  /** The model the role uses, or null when no model is eligible. */
  variant: string | null;
  ready: boolean;
  /** The user's pick left the catalog; the role counts as not loaded. */
  missing: boolean;
}

/**
 * Resolves one diarization-stack role. The user's pick is the source of
 * truth while it is eligible: a pick that vanished from the catalog surfaces
 * as missing (never silently swapped), and one no longer eligible for the
 * role is ignored. Without a usable pick, a loaded model wins over a
 * not-loaded default; the preferred list orders defaults and never gates
 * readiness: loaded preferred, then any loaded eligible model, then the
 * preferred default, then the first eligible model.
 */
export function resolveDiarizationStackRole(options: {
  /** The whole catalog, to tell a deleted pick from an ineligible one. */
  models: ModelInfo[];
  /** The models eligible for this role. */
  eligibleModels: ModelInfo[];
  pick: string | null;
  preferredVariants: readonly string[];
}): DiarizationStackRole {
  const { models, eligibleModels, pick, preferredVariants } = options;
  if (pick) {
    const picked = eligibleModels.find((model) => model.variant === pick);
    if (picked) {
      return { variant: pick, ready: picked.status === "ready", missing: false };
    }
    if (!models.some((model) => model.variant === pick)) {
      return { variant: pick, ready: false, missing: true };
    }
  }
  const variant = resolvePreferredRouteModel({
    models: eligibleModels,
    selectedModel: null,
    preferredVariants,
    preferAnyPreferredBeforeReadyAny: false,
  });
  return {
    variant,
    ready:
      variant != null &&
      eligibleModels.some(
        (model) => model.variant === variant && model.status === "ready",
      ),
    missing: false,
  };
}
