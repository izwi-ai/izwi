import type { ModelInfo } from "@/api";
import {
  DIARIZATION_PREFERRED_SUMMARY_MODELS,
  resolvePreferredRouteModel,
} from "@/features/models/catalog/routeModelCatalog";

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
  // A speaker-attributed model (Granite Speech 4.1 Plus) belongs here too:
  // the pipeline asks it for a plain transcript, and the diarizer supplies
  // the speakers. The pipeline transcribes as a speech-to-text job:
  // audio-chat models serve /audio/transcriptions but cannot run one, and
  // the server rejects them as the pipeline ASR.
  return (
    hasCapability(model, "openai_audio_transcriptions") &&
    hasCapability(model, "speech_to_text_jobs")
  );
}

export function isDiarizationPipelineAlignerModel(model: ModelInfo): boolean {
  return hasCapability(model, "forced_alignment");
}

/** A text chat model: the server's own text generations (summaries, the
 * diarization pipeline's LLM) run on one. LFM3-Audio serves chat completions
 * too, but only as an audio chat model, so it is not one. */
function isTextChatModel(model: ModelInfo): boolean {
  return (
    hasCapability(model, "openai_chat_completions") &&
    hasCapability(model, "voice_realtime_text_model")
  );
}

export function isDiarizationPipelineLlmModel(model: ModelInfo): boolean {
  return isTextChatModel(model);
}

export function isTranscriptionAlignerModel(model: ModelInfo): boolean {
  return hasCapability(model, "forced_alignment");
}

export function isTranscriptionSummaryModel(model: ModelInfo): boolean {
  return isTextChatModel(model);
}

export function isSpeakerAttributedAsrModel(model: RouteCapabilityModel): boolean {
  return hasCapability(model, "speaker_attributed_asr");
}

/**
 * Display-only: the installed Granite Speech 5.0 model, if any, so the
 * speaker-attributed ASR picker can say why it is not offered there. 5.0 has
 * no speaker task; its speakers come from a diarization run with it as the
 * ASR model.
 */
export function installedGraniteSpeech5Model(models: ModelInfo[]): ModelInfo | null {
  return (
    models.find(
      (model) =>
        model.variant.startsWith("Granite-Speech-5.") &&
        !isSpeakerAttributedAsrModel(model) &&
        (model.status === "downloaded" ||
          model.status === "loading" ||
          model.status === "ready"),
    ) ?? null
  );
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
