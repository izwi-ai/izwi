import {
  AudioLines,
  Box,
  FileAudio,
  FileText,
  Library,
  MessageSquare,
  Mic,
  Settings,
  type LucideIcon,
} from "lucide-react";

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
