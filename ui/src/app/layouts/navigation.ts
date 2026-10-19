import {
  AudioLines,
  Box,
  FileAudio,
  FileText,
  Library,
  MessageSquare,
  Mic,
  Settings,
  Users,
  Wand2,
  type LucideIcon,
} from "lucide-react";

import { VOICE_STUDIO_ENABLED } from "@/shared/config/runtime";

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

// Voice Studio folds cloning and design into Voices; without it they keep
// their own pages.
const LEGACY_VOICE_ITEMS: NavItem[] = VOICE_STUDIO_ENABLED
  ? []
  : [
      {
        id: "voice-cloning",
        label: "Voice Cloning",
        icon: Users,
        path: "/voice-cloning",
      },
      {
        id: "voice-design",
        label: "Voice Design",
        icon: Wand2,
        path: "/voice-design",
      },
    ];

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
      ...LEGACY_VOICE_ITEMS,
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
