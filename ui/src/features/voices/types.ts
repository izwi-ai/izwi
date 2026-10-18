import type { LucideIcon } from "lucide-react";

import type { RowMenuItem } from "@/components/history/RowMenu";

export interface VoiceLibraryItem {
  id: string;
  name: string;
  /** Line under the name: "Designed · 20 May 2026", "Built-in · Kokoro". */
  meta?: string;
  /** Badge in the card footer: "Cloned", "Designed", "Built-in". */
  kindLabel: string;
  description?: string;
  previewUrl?: string | null;
  previewLoading?: boolean;
  /** Generates a preview when there is none yet (built-in voices). */
  onRequestPreview?: () => void;
  primaryAction?: { label: string; icon: LucideIcon; onClick: () => void };
  menuItems?: RowMenuItem[];
}
