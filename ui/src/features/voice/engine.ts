import type { ModelInfo } from "@/api";
import {
  MODULAR_STACK_VARIANTS,
  formatModelVariantLabel,
  isUsableVoiceModelStatus,
} from "@/features/voice/realtime/support";
import type { VoiceEngineId } from "@/features/voice/useVoicePreferences";
import type { VoiceSessionEngine } from "@/features/voice/useVoiceSession";
import { getSpeakerProfilesForVariant, type SpeakerProfile } from "@/types";

export const COMPACT_VOICE_MODEL = "LFM2.5-Audio-1.5B-GGUF";

/** Kokoro's most natural English voice; other engines start on their first. */
export const PREFERRED_SPEAKER = "af_heart";

export interface VoiceEngineModel {
  /** What the model does, in the user's words. */
  role: string;
  variant: string;
  label: string;
  model: ModelInfo | null;
}

export const VOICE_ENGINES: Record<
  VoiceEngineId,
  { label: string; description: string }
> = {
  standard: {
    label: "Standard",
    description:
      "Separate models for listening, thinking and speaking. Best answers and the full voice library.",
  },
  compact: {
    label: "Compact",
    description:
      "One model (LFM2.5 Audio) listens and speaks. Quicker to set up, fewer voices.",
  },
};

export function voiceEngineModels(
  engine: VoiceEngineId,
  models: ModelInfo[],
): VoiceEngineModel[] {
  const entry = (role: string, variant: string): VoiceEngineModel => ({
    role,
    variant,
    label: formatModelVariantLabel(variant),
    model: models.find((model) => model.variant === variant) ?? null,
  });
  if (engine === "compact") {
    return [entry("Listens, thinks and speaks", COMPACT_VOICE_MODEL)];
  }
  return [
    entry("Speech recognition", MODULAR_STACK_VARIANTS.asr),
    entry("Assistant", MODULAR_STACK_VARIANTS.text),
    entry("Voice", MODULAR_STACK_VARIANTS.tts),
  ];
}

/** Whether a conversation can start: every model is on disk. */
export function isVoiceEngineReady(rows: VoiceEngineModel[]): boolean {
  return rows.every(
    (row) => !!row.model && isUsableVoiceModelStatus(row.model.status),
  );
}

export function voiceSessionEngine(
  engine: VoiceEngineId,
): VoiceSessionEngine {
  return engine === "compact"
    ? { mode: "unified", s2s: COMPACT_VOICE_MODEL }
    : {
        mode: "modular",
        asr: MODULAR_STACK_VARIANTS.asr,
        text: MODULAR_STACK_VARIANTS.text,
        tts: MODULAR_STACK_VARIANTS.tts,
      };
}

/** The model whose built-in voices the assistant speaks with. */
export function voiceEngineSpeakerVariant(engine: VoiceEngineId): string {
  return engine === "compact" ? COMPACT_VOICE_MODEL : MODULAR_STACK_VARIANTS.tts;
}

export function voiceEngineSpeakers(engine: VoiceEngineId): SpeakerProfile[] {
  return getSpeakerProfilesForVariant(voiceEngineSpeakerVariant(engine));
}

/** The stored voice when this engine has it, else its default. */
export function resolveVoiceSpeaker(
  speakers: SpeakerProfile[],
  stored: string | null,
): SpeakerProfile | null {
  return (
    speakers.find((speaker) => speaker.id === stored) ??
    speakers.find((speaker) => speaker.id === PREFERRED_SPEAKER) ??
    speakers[0] ??
    null
  );
}

/** "American English · Female" for Kokoro voices, else the language. */
export function voiceSpeakerMeta(speaker: SpeakerProfile): string {
  const kokoro = /^(.*) (Female|Male) voice \(Kokoro\)$/.exec(
    speaker.description,
  );
  return kokoro ? `${kokoro[1]} · ${kokoro[2]}` : speaker.language;
}

/** Bytes still to download for these models (estimated when not on disk). */
export function voiceEngineDownloadBytes(rows: VoiceEngineModel[]): number {
  return rows.reduce((total, row) => {
    const model = row.model;
    if (!model || isUsableVoiceModelStatus(model.status)) return total;
    return total + (model.estimated_size_bytes ?? model.size_bytes ?? 0);
  }, 0);
}

export function voiceEngineTotalBytes(rows: VoiceEngineModel[]): number {
  return rows.reduce(
    (total, row) =>
      total + (row.model?.size_bytes ?? row.model?.estimated_size_bytes ?? 0),
    0,
  );
}
