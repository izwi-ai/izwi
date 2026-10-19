import type {
  ModelInfo,
  SavedVoiceSummary,
  StudioProjectRecord,
  StudioProjectSegmentRecord,
  StudioProjectVoiceMode,
} from "@/api";
import {
  TEXT_TO_SPEECH_PREFERRED_MODELS,
  resolvePreferredRouteModel,
} from "@/features/models/catalog/routeModelCatalog";
import { getModelDisplayName } from "@/features/models/catalog/modelMetadata";
import { isStudioRenderableModel } from "@/features/studio/modelFilter";
import { getSpeakerProfilesForVariant } from "@/types";

/** A project's (or a paragraph's) voice: a model and one of its voices. */
export interface StudioVoice {
  modelId: string;
  voiceMode: StudioProjectVoiceMode;
  speaker: string | null;
  savedVoiceId: string | null;
}

/** One voice a model can render with, as the pickers list it. */
export interface StudioVoiceOption {
  key: string;
  name: string;
  detail: string;
  voiceMode: StudioProjectVoiceMode;
  speaker: string | null;
  savedVoiceId: string | null;
}

/** Downloaded (or loaded): the server can render with it now. */
export function isModelOnDevice(model: ModelInfo | undefined): boolean {
  return (
    !!model &&
    model.enabled !== false &&
    ["downloaded", "loading", "ready"].includes(model.status)
  );
}

export function studioModels(models: ModelInfo[]): ModelInfo[] {
  return models.filter(isStudioRenderableModel);
}

export function modelName(modelId: string | null, models: ModelInfo[]): string {
  if (!modelId) return "No model";
  const model = models.find((candidate) => candidate.variant === modelId);
  return getModelDisplayName(modelId, model?.display_name);
}

export function voiceOptions(
  model: ModelInfo | undefined,
  savedVoices: SavedVoiceSummary[],
): StudioVoiceOption[] {
  if (!model) return [];
  const capabilities = model.speech_capabilities;
  const builtIn = capabilities?.supports_builtin_voices
    ? getSpeakerProfilesForVariant(model.variant).map((speaker) => ({
        key: `built_in:${speaker.id}`,
        name: speaker.name,
        detail: speaker.language,
        voiceMode: "built_in" as const,
        speaker: speaker.id,
        savedVoiceId: null,
      }))
    : [];
  const saved = capabilities?.supports_reference_voice
    ? savedVoices.map((voice) => ({
        key: `saved:${voice.id}`,
        name: voice.name,
        detail: "My voice",
        voiceMode: "saved" as const,
        speaker: null,
        savedVoiceId: voice.id,
      }))
    : [];
  return [...builtIn, ...saved];
}

export function voiceOptionKey(voice: Pick<StudioVoice, "voiceMode" | "speaker" | "savedVoiceId">): string {
  return voice.voiceMode === "saved"
    ? `saved:${voice.savedVoiceId ?? ""}`
    : `built_in:${voice.speaker ?? ""}`;
}

/** A voice's name: the speaker's display name or the saved voice's. */
export function voiceName(
  voice: Pick<StudioVoice, "modelId" | "voiceMode" | "speaker" | "savedVoiceId">,
  savedVoices: SavedVoiceSummary[],
): string {
  if (voice.voiceMode === "saved") {
    return (
      savedVoices.find((saved) => saved.id === voice.savedVoiceId)?.name ??
      "Saved voice"
    );
  }
  if (!voice.speaker) return "No voice";
  return (
    getSpeakerProfilesForVariant(voice.modelId).find(
      (speaker) => speaker.id === voice.speaker,
    )?.name ?? voice.speaker
  );
}

export function projectVoice(project: StudioProjectRecord): StudioVoice {
  return {
    modelId: project.model_id ?? "",
    voiceMode: project.voice_mode,
    speaker: project.speaker,
    savedVoiceId: project.saved_voice_id,
  };
}

/** The paragraph's own voice, or null when it follows the project. */
export function paragraphVoice(
  project: StudioProjectRecord,
  segment: StudioProjectSegmentRecord,
): StudioVoice | null {
  if (!segment.model_id && !segment.voice_mode && !segment.speaker && !segment.saved_voice_id) {
    return null;
  }
  const voiceMode = segment.voice_mode ?? project.voice_mode;
  return {
    modelId: segment.model_id ?? project.model_id ?? "",
    voiceMode,
    speaker: voiceMode === "built_in" ? (segment.speaker ?? project.speaker) : null,
    savedVoiceId:
      voiceMode === "saved" ? (segment.saved_voice_id ?? project.saved_voice_id) : null,
  };
}

/**
 * The voice a new project starts with: the preferred text-to-speech model
 * that is on this device, else the preferred one to download, with its
 * first built-in voice, or the first saved voice.
 */
export function defaultStudioVoice(
  models: ModelInfo[],
  savedVoices: SavedVoiceSummary[],
): StudioVoice | null {
  const candidates = studioModels(models);
  const onDevice = candidates.filter(isModelOnDevice);
  const pool = onDevice.length > 0 ? onDevice : candidates;
  const modelId = resolvePreferredRouteModel({
    models: pool,
    selectedModel: null,
    preferredVariants: TEXT_TO_SPEECH_PREFERRED_MODELS,
  });
  const model = pool.find((candidate) => candidate.variant === modelId);
  const first = voiceOptions(model, savedVoices)[0];
  if (!model || !first) return null;
  return {
    modelId: model.variant,
    voiceMode: first.voiceMode,
    speaker: first.speaker,
    savedVoiceId: first.savedVoiceId,
  };
}

/** Keep the voice when the new model has it, else its first voice. */
export function voiceForModel(
  current: StudioVoice | null,
  model: ModelInfo | undefined,
  savedVoices: SavedVoiceSummary[],
): StudioVoice | null {
  if (!model) return null;
  const options = voiceOptions(model, savedVoices);
  const kept = current
    ? options.find((option) => option.key === voiceOptionKey(current))
    : undefined;
  const chosen = kept ?? options[0];
  if (!chosen) return null;
  return {
    modelId: model.variant,
    voiceMode: chosen.voiceMode,
    speaker: chosen.speaker,
    savedVoiceId: chosen.savedVoiceId,
  };
}

const NEEDS_RENDER = new Set(["not_rendered", "stale", "failed"]);

export function needsRender(segment: StudioProjectSegmentRecord): boolean {
  return NEEDS_RENDER.has(segment.render_state);
}

export function isPending(segment: StudioProjectSegmentRecord): boolean {
  return segment.render_state === "queued" || segment.render_state === "rendering";
}

/** What the editor's main action does now. */
export type StudioMainAction =
  | { kind: "rendering"; current: number; total: number; jobId: string }
  | { kind: "render"; count: number; label: string }
  | { kind: "export" };

export function mainAction(project: StudioProjectRecord): StudioMainAction {
  const job = project.active_render_job;
  if (job) {
    const pendingIds = new Set(
      project.segments.filter(isPending).map((segment) => segment.id),
    );
    const total = job.queued_segment_ids.length;
    const remaining = job.queued_segment_ids.filter((id) => pendingIds.has(id)).length;
    return {
      kind: "rendering",
      current: Math.min(total, total - remaining + 1),
      total,
      jobId: job.id,
    };
  }
  const waiting = project.segments.filter(needsRender);
  if (waiting.length === 0) return { kind: "export" };
  const fresh = project.segments.every((segment) => segment.render_state === "not_rendered");
  const noun = fresh
    ? waiting.length === 1
      ? "paragraph"
      : "paragraphs"
    : waiting.length === 1
      ? "change"
      : "changes";
  return { kind: "render", count: waiting.length, label: `Render ${waiting.length} ${noun}` };
}

export function projectDurationSecs(project: StudioProjectRecord): number {
  return project.segments.reduce(
    (total, segment) => total + (segment.audio_duration_secs ?? 0),
    0,
  );
}

/** "about 6 min" from characters, at an average narration pace. */
export function estimatedLength(chars: number): string {
  const minutes = chars / 900;
  if (chars === 0) return "";
  if (minutes < 1) return "under a minute";
  return `about ${Math.round(minutes)} min`;
}

export function scriptFilename(name: string, extension: string): string {
  const slug = name
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "");
  return `${slug || "studio-project"}.${extension}`;
}
