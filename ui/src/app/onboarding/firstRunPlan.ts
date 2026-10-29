import type { ModelInfo } from "@/api";
import type { ModelRouteKey } from "@/app/providers/ModelCatalogProvider";
import {
  getModelDisplayName,
  getModelProviderLabel,
} from "@/features/models/catalog/modelMetadata";
import {
  CHAT_PREFERRED_MODELS,
  DIARIZATION_PREFERRED_ALIGNER_MODELS,
  DIARIZATION_PREFERRED_ASR_MODELS,
  DIARIZATION_PREFERRED_MODELS,
  TEXT_TO_SPEECH_PREFERRED_MODELS,
  TRANSCRIPTION_PREFERRED_MODELS,
  VOICE_CLONING_PREFERRED_MODELS,
  VOICE_DESIGN_PREFERRED_MODELS,
  diarizationSpeakerUpperBound,
} from "@/features/models/catalog/routeModelCatalog";
import {
  isVoiceEngineReady,
  resolveCompactVoiceModel,
  voiceEngineModels,
} from "@/features/voice/engine";
import type { HealthResponse } from "@/shared/api/health";
import { VIEW_CONFIGS, type ViewMode } from "@/types";

const GIB = 1024 ** 3;

export type TaskId =
  | "talk"
  | "chat"
  | "transcribe"
  | "read"
  | "voices"
  | "speakers";

interface TaskSpec {
  id: TaskId;
  title: string;
  /** The row on "You're all set". */
  doneTitle: string;
  path: string;
  /** Candidates for each model the task needs, in the page's order. */
  slots: (models: ModelInfo[]) => readonly (readonly string[])[];
  checkedByDefault: boolean;
}

const TASKS: readonly TaskSpec[] = [
  {
    // The Voice page's Standard engine.
    id: "talk",
    title: "Talk to Izwi",
    doneTitle: "Talk to Izwi",
    path: "/voice",
    slots: (models) =>
      voiceEngineModels("standard", models).map((row) => [row.variant]),
    checkedByDefault: true,
  },
  {
    id: "chat",
    title: "Chat",
    doneTitle: "Chat",
    path: "/chat",
    slots: () => [CHAT_PREFERRED_MODELS],
    checkedByDefault: true,
  },
  {
    id: "transcribe",
    title: "Transcribe recordings",
    doneTitle: "Transcribe a recording",
    path: "/transcription",
    slots: () => [TRANSCRIPTION_PREFERRED_MODELS],
    checkedByDefault: true,
  },
  {
    id: "read",
    title: "Read text aloud",
    doneTitle: "Read text aloud",
    path: "/text-to-speech",
    slots: () => [TEXT_TO_SPEECH_PREFERRED_MODELS],
    checkedByDefault: true,
  },
  {
    id: "voices",
    title: "Clone and design voices",
    doneTitle: "Clone or design a voice",
    path: "/voices",
    slots: () => [VOICE_CLONING_PREFERRED_MODELS, VOICE_DESIGN_PREFERRED_MODELS],
    checkedByDefault: false,
  },
  {
    id: "speakers",
    title: "Label who's speaking",
    doneTitle: "Label who's speaking",
    path: "/transcription",
    slots: () => [
      DIARIZATION_PREFERRED_MODELS,
      DIARIZATION_PREFERRED_ASR_MODELS,
      DIARIZATION_PREFERRED_ALIGNER_MODELS,
    ],
    checkedByDefault: false,
  },
];

export const DEFAULT_UNCHECKED_TASKS: ReadonlySet<TaskId> = new Set(
  TASKS.filter((task) => !task.checkedByDefault).map((task) => task.id),
);

export function isOnDisk(model: ModelInfo): boolean {
  return (
    model.status === "downloaded" ||
    model.status === "ready" ||
    model.status === "loading"
  );
}

export function needsDownload(model: ModelInfo): boolean {
  return model.status === "not_downloaded" || model.status === "error";
}

export function fitsDevice(model: ModelInfo): boolean {
  return model.device_fit !== "too_large";
}

/** Download size: the catalog's estimate for a model not on disk yet. */
export function downloadBytes(model: ModelInfo): number {
  return model.estimated_size_bytes ?? model.size_bytes ?? 0;
}

function memoryBytes(model: ModelInfo): number {
  return model.estimated_memory_bytes ?? (model.memory_required_gb ?? 0) * GIB;
}

export function modelName(model: ModelInfo): string {
  return getModelDisplayName(model.variant, model.display_name);
}

export interface TaskRow {
  id: TaskId;
  title: string;
  doneTitle: string;
  path: string;
  /** One model per slot, models shared with an earlier task reused. */
  models: ModelInfo[];
  /** The model this device can't run, when a slot has none that fits. */
  tooLarge: ModelInfo | null;
}

/**
 * The tasks first run offers, each with the models its page prefers that fit
 * this device. A model an earlier task already uses is reused, so Talk's
 * speech recognition also transcribes. Tasks whose models aren't in the
 * catalog are left out.
 */
export function buildTaskRows(models: ModelInfo[]): TaskRow[] {
  const catalog = new Map(
    models
      .filter((model) => model.enabled !== false)
      .map((model) => [model.variant, model]),
  );
  const used = new Set<string>();
  const rows: TaskRow[] = [];

  for (const task of TASKS) {
    const picks: ModelInfo[] = [];
    let tooLarge: ModelInfo | null = null;
    let missing = false;
    for (const candidates of task.slots(models)) {
      const listed = candidates
        .map((variant) => catalog.get(variant))
        .filter((model): model is ModelInfo => Boolean(model));
      if (listed.length === 0) {
        missing = true;
        break;
      }
      const fitting = listed.filter(fitsDevice);
      const pick =
        fitting.find((model) => used.has(model.variant)) ?? fitting[0];
      if (pick) {
        picks.push(pick);
      } else {
        tooLarge ??= listed.reduce((smallest, model) =>
          memoryBytes(model) < memoryBytes(smallest) ? model : smallest,
        );
      }
    }
    if (missing) {
      continue;
    }
    if (!tooLarge) {
      picks.forEach((model) => used.add(model.variant));
    }
    rows.push({
      id: task.id,
      title: task.title,
      doneTitle: task.doneTitle,
      path: task.path,
      models: tooLarge ? [] : picks,
      tooLarge,
    });
  }
  return rows;
}

/** "Parakeet v3 · Qwen3.5 4B · Kokoro 82M", with the voice count for reading aloud. */
export function taskDetail(row: TaskRow): string {
  const names = row.models.map(modelName);
  const voices = row.models[0]?.speech_capabilities?.built_in_voice_count;
  if (row.id === "read" && voices) {
    names.push(`${voices} voices`);
  }
  return names.join(" · ");
}

export type SizeLabel =
  | { kind: "bytes"; bytes: number }
  | { kind: "included" }
  | { kind: "downloaded" }
  | { kind: "downloading" };

/** Distinct models across rows, in order. */
export function uniqueModels(groups: ModelInfo[][]): ModelInfo[] {
  const seen = new Map<string, ModelInfo>();
  groups.flat().forEach((model) => {
    if (!seen.has(model.variant)) {
      seen.set(model.variant, model);
    }
  });
  return [...seen.values()];
}

/**
 * What each task adds to the download. A checked task counts the models
 * no checked task above it already counts ("Included" when that's none); an
 * unchecked one, what checking it would add.
 */
export function taskSizeLabels(
  rows: TaskRow[],
  checked: ReadonlySet<TaskId>,
): Map<TaskId, SizeLabel> {
  const allChecked = new Set(
    rows
      .filter((row) => checked.has(row.id))
      .flatMap((row) => row.models.map((model) => model.variant)),
  );
  const countedAbove = new Set<string>();
  const labels = new Map<TaskId, SizeLabel>();

  for (const row of rows) {
    const isChecked = checked.has(row.id);
    const fetch = row.models.filter(needsDownload);
    if (fetch.length === 0) {
      labels.set(row.id, {
        kind: row.models.some((model) => model.status === "downloading")
          ? "downloading"
          : "downloaded",
      });
    } else {
      const counted = isChecked ? countedAbove : allChecked;
      const bytes = fetch
        .filter((model) => !counted.has(model.variant))
        .reduce((total, model) => total + downloadBytes(model), 0);
      labels.set(
        row.id,
        bytes > 0 ? { kind: "bytes", bytes } : { kind: "included" },
      );
    }
    if (isChecked) {
      row.models.forEach((model) => countedAbove.add(model.variant));
    }
  }
  return labels;
}

export interface DownloadPlan {
  /** Models still to download, each once. */
  fetch: ModelInfo[];
  bytes: number;
  /** Free disk the downloads need; null while a model doesn't say. */
  diskBytes: number | null;
}

export function downloadPlan(chosen: ModelInfo[]): DownloadPlan {
  const fetch = uniqueModels([chosen]).filter(needsDownload);
  const diskKnown = fetch.every(
    (model) => typeof model.disk_required_bytes === "number",
  );
  return {
    fetch,
    bytes: fetch.reduce((total, model) => total + downloadBytes(model), 0),
    diskBytes: diskKnown
      ? fetch.reduce((total, model) => total + (model.disk_required_bytes ?? 0), 0)
      : null,
  };
}

export type ModelGroupId =
  | "asr"
  | "chat"
  | "tts"
  | "clone"
  | "design"
  | "speakers"
  | "align"
  | "voice";

export const MODEL_GROUPS: ReadonlyArray<{
  id: ModelGroupId;
  title: string;
  /** What a model of this group does, on the Setting up rows. */
  role: string;
}> = [
  { id: "asr", title: "Speech to text", role: "Speech recognition" },
  { id: "chat", title: "Chat", role: "Replies and chat" },
  { id: "tts", title: "Text to speech", role: "Speech" },
  { id: "clone", title: "Voice cloning", role: "Voice cloning" },
  { id: "design", title: "Voice design", role: "Voice design" },
  { id: "speakers", title: "Speaker labels", role: "Speaker labels" },
  { id: "align", title: "Word timestamps", role: "Word timestamps" },
  { id: "voice", title: "Voice conversations", role: "Listens and speaks" },
];

/** The model's task, from the catalog, else from what it can serve. */
function modelTask(model: ModelInfo): string | null {
  if (model.task) {
    return model.task;
  }
  const routes = model.route_capabilities;
  if (!routes || routes.tokenizer) return null;
  if (routes.voice_realtime_unified) return "audio_chat";
  if (routes.openai_chat_completions) return "chat";
  if (routes.diarization_records) return "diarization";
  if (routes.forced_alignment) return "forced_align";
  if (routes.openai_audio_transcriptions) return "asr";
  if (routes.openai_audio_speech) return "tts";
  return null;
}

/** Which "Choose models" group a model is listed in, if any. */
export function modelGroup(model: ModelInfo): ModelGroupId | null {
  switch (modelTask(model)) {
    case "asr":
      return "asr";
    case "chat":
      return "chat";
    case "diarization":
      return "speakers";
    case "forced_align":
      return "align";
    case "audio_chat":
      return "voice";
    case "tts": {
      const speech = model.speech_capabilities;
      if (speech?.supports_reference_voice) return "clone";
      if (speech?.supports_builtin_voices) return "tts";
      if (speech?.supports_voice_description) return "design";
      return "tts";
    }
    default:
      return null;
  }
}

export function modelRole(model: ModelInfo): string {
  const group = modelGroup(model);
  return MODEL_GROUPS.find((entry) => entry.id === group)?.role ?? "Model";
}

/** What sets the model apart, then its provider: "41 built-in voices · hexgrad". */
export function modelDescriptor(model: ModelInfo): string {
  const parts: string[] = [];
  const speech = model.speech_capabilities;
  switch (modelGroup(model)) {
    case "tts":
    case "voice": {
      const voices = speech?.built_in_voice_count;
      const lead = modelGroup(model) === "voice" ? "Listens and speaks" : null;
      const voiceText = voices
        ? `${voices} built-in voice${voices === 1 ? "" : "s"}`
        : null;
      const style = speech?.supports_voice_description ? "style direction" : null;
      const text = [lead, voiceText, style].filter(Boolean).join(", ");
      if (text) parts.push(text.charAt(0).toUpperCase() + text.slice(1));
      break;
    }
    case "clone":
      parts.push("Clones from a short recording");
      break;
    case "design":
      parts.push("Designs voices from a description");
      break;
    case "chat": {
      const features: string[] = [];
      if (model.modalities?.includes("image_input")) features.push("reads images");
      const chat = model.chat_capabilities;
      if (chat?.supports_thinking) {
        features.push(
          chat.thinking_toggleable === false
            ? "thinks before replying"
            : "can think before replying",
        );
      }
      const text = features.join(", ");
      if (text) parts.push(text.charAt(0).toUpperCase() + text.slice(1));
      break;
    }
    case "speakers":
      parts.push(`Up to ${diarizationSpeakerUpperBound(model.variant)} speakers`);
      break;
    case "asr":
      if (model.modalities?.includes("speaker_labels")) {
        parts.push("Transcripts with speakers");
      }
      break;
    default:
      break;
  }
  const provider = getModelProviderLabel(model.variant);
  if (provider !== "Other") {
    parts.push(provider);
  }
  return parts.join(" · ");
}

export interface ModelGroup {
  id: ModelGroupId;
  title: string;
  models: ModelInfo[];
}

/**
 * The catalog by group: recommended models first, then smallest first, and
 * models too large for this device last.
 */
export function groupModels(
  models: ModelInfo[],
  recommended: ReadonlySet<string>,
): ModelGroup[] {
  const rank = (model: ModelInfo) =>
    (fitsDevice(model) ? 0 : 2) + (recommended.has(model.variant) ? 0 : 1);
  return MODEL_GROUPS.map((group) => ({
    id: group.id,
    title: group.title,
    models: models
      .filter((model) => model.enabled !== false && modelGroup(model) === group.id)
      .sort(
        (left, right) =>
          rank(left) - rank(right) || downloadBytes(left) - downloadBytes(right),
      ),
  })).filter((group) => group.models.length > 0);
}

const PAGE_GROUPS: ReadonlyArray<[ModelRouteKey, ModelGroupId, ViewMode]> = [
  ["transcription", "asr", "transcription"],
  ["chat", "chat", "chat"],
  ["text-to-speech", "tts", "custom-voice"],
];

/**
 * The model each page should start on: the first newly downloaded model it
 * can use. Pages otherwise pick their own first preference, which may not be
 * the one first run downloaded.
 */
export function pageChoices(
  downloads: ModelInfo[],
): Array<[ModelRouteKey, string]> {
  const choices: Array<[ModelRouteKey, string]> = [];
  for (const [route, group, view] of PAGE_GROUPS) {
    const model = downloads.find(
      (candidate) =>
        modelGroup(candidate) === group &&
        VIEW_CONFIGS[view].modelFilter(candidate.variant),
    );
    if (model) {
      choices.push([route, model.variant]);
    }
  }
  return choices;
}

/** Tasks this device can do now: each has a model of every kind it needs on disk. */
export function readyTasks(rows: TaskRow[], models: ModelInfo[]): TaskRow[] {
  const onDisk = models.filter(isOnDisk);
  const has = (group: ModelGroupId) =>
    onDisk.some((model) => modelGroup(model) === group);
  const ready: Record<TaskId, boolean> = {
    talk:
      isVoiceEngineReady(voiceEngineModels("standard", models)) ||
      isVoiceEngineReady(
        voiceEngineModels(
          "compact",
          models,
          resolveCompactVoiceModel(null, models),
        ),
      ),
    chat: has("chat"),
    transcribe: has("asr"),
    read: has("tts"),
    voices: has("clone") || has("design"),
    // Granite Speech Plus labels speakers itself; otherwise the pipeline.
    speakers:
      onDisk.some(
        (model) =>
          modelGroup(model) === "asr" &&
          model.modalities?.includes("speaker_labels"),
      ) ||
      (has("speakers") && has("asr") && has("align")),
  };
  return rows.filter((row) => ready[row.id]);
}

const BACKEND_LABELS: Record<string, string> = {
  metal: "Apple Metal",
  cuda: "NVIDIA CUDA",
  cpu: "CPU",
};

/** "Picked for 16 GB of memory · Apple Metal", once health says. */
export function hardwareLabel(health: HealthResponse | null): string | null {
  const total = health?.runtime.memory.system_total_bytes;
  if (!total) {
    return null;
  }
  const backend = health.runtime.selected_backend;
  const backendLabel = BACKEND_LABELS[backend] ?? backend;
  return [`Picked for ${Math.round(total / GIB)} GB of memory`, backendLabel]
    .filter(Boolean)
    .join(" · ");
}
