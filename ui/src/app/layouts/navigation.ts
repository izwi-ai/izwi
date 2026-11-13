import {
  AudioLines,
  Box,
  FileAudio,
  FileText,
  Keyboard,
  Library,
  MessageSquare,
  Mic,
  Settings,
  type LucideIcon,
} from "lucide-react";
import { isTauri } from "@tauri-apps/api/core";

export interface NavItem {
  id: string;
  label: string;
  icon: LucideIcon;
  path: string;
}

export interface NavGroup {
  label: string;
  items: NavItem[];
}

/** System-wide dictation exists only in the desktop app. */
const DICTATION_NAV_ITEMS: NavItem[] = isTauri()
  ? [{ id: "dictation", label: "Dictation", icon: Keyboard, path: "/dictation" }]
  : [];

export const NAV_GROUPS: NavGroup[] = [
  {
    label: "Converse",
    items: [
      { id: "voice", label: "Voice", icon: AudioLines, path: "/voice" },
      { id: "chat", label: "Chat", icon: MessageSquare, path: "/chat" },
    ],
  },
  {
    label: "Audio",
    items: [
      {
        id: "transcription",
        label: "Transcription",
        icon: FileText,
        path: "/transcription",
      },
      ...DICTATION_NAV_ITEMS,
      {
        id: "text-to-speech",
        label: "Text to Speech",
        icon: Mic,
        path: "/text-to-speech",
      },
      { id: "studio", label: "Studio", icon: FileAudio, path: "/studio" },
    ],
  },
  {
    label: "Library",
    items: [
      { id: "voices", label: "Voices", icon: Library, path: "/voices" },
      { id: "models", label: "Models", icon: Box, path: "/models" },
    ],
  },
];

export const SETTINGS_NAV_ITEM: NavItem = {
  id: "settings",
  label: "Settings",
  icon: Settings,
  path: "/settings",
};
