import {
  AlignLeft,
  AudioLines,
  Box,
  FileText,
  MessageSquare,
  Mic,
  Users,
  type LucideIcon,
} from "lucide-react";

import type { ModelInfo } from "@/api";
import { resolveModelDetail } from "@/features/models/catalog/modelMetadata";

export type ModelTaskGroupId =
  | "audio_chat"
  | "asr"
  | "tts"
  | "chat"
  | "diarization"
  | "forced_align"
  | "other";

export interface ModelTaskGroup {
  id: ModelTaskGroupId;
  label: string;
  icon: LucideIcon;
  /** Pages that use these models. */
  usedBy?: string;
}

export const MODEL_TASK_GROUPS: ModelTaskGroup[] = [
  { id: "audio_chat", label: "Voice & audio chat", icon: AudioLines, usedBy: "Voice" },
  {
    id: "asr",
    label: "Speech to text",
    icon: FileText,
    usedBy: "Transcription, Voice",
  },
  {
    id: "tts",
    label: "Text to speech",
    icon: Mic,
    usedBy: "Text to Speech, Studio, Voices",
  },
  { id: "chat", label: "Chat", icon: MessageSquare, usedBy: "Chat, Voice" },
  {
    id: "diarization",
    label: "Speaker diarization",
    icon: Users,
    usedBy: "Transcription",
  },
  {
    id: "forced_align",
    label: "Alignment",
    icon: AlignLeft,
    usedBy: "Transcription timestamps",
  },
  { id: "other", label: "Other", icon: Box },
];

const KNOWN_TASKS = new Set<string>(
  MODEL_TASK_GROUPS.map((group) => group.id),
);

/** The group a model lists under: its API task, else its curated category. */
export function modelTaskGroupId(model: ModelInfo): ModelTaskGroupId {
  if (model.task && KNOWN_TASKS.has(model.task)) {
    return model.task as ModelTaskGroupId;
  }
  const category = resolveModelDetail(model).category;
  return KNOWN_TASKS.has(category) ? (category as ModelTaskGroupId) : "other";
}
