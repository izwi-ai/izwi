import { Check } from "lucide-react";
import type { SpeechTextCreationMode } from "@/features/speech-text/creationMode";

interface SpeechTextModeSwitchProps {
  selectedMode: SpeechTextCreationMode;
  onSelectMode?: (mode: SpeechTextCreationMode) => void;
}

export function SpeechTextModeSwitch({
  selectedMode,
  onSelectMode,
}: SpeechTextModeSwitchProps) {
  if (!onSelectMode) {
    return null;
  }

  return (
    <div className="mt-3 grid gap-2 sm:grid-cols-2">
      <label className="flex items-start justify-between gap-3 rounded-2xl border border-[var(--border-muted)] bg-[var(--bg-surface-1)] p-3.5">
        <div className="min-w-0">
          <div className="text-sm font-semibold text-[var(--text-primary)]">
            Transcription
          </div>
        </div>
        <div className="relative mt-0.5 shrink-0">
          <input
            id="speech-text-mode-transcription"
            name="speech-text-mode"
            type="radio"
            checked={selectedMode === "transcription"}
            onChange={() => onSelectMode("transcription")}
            className="peer sr-only"
          />
          <span className="flex h-5 w-5 items-center justify-center rounded-md border border-[var(--border-strong)] bg-[var(--bg-surface-0)] text-white transition peer-checked:border-[var(--status-info-text)] peer-checked:bg-[var(--status-info-text)] peer-focus-visible:outline-none peer-focus-visible:ring-2 peer-focus-visible:ring-ring/45 peer-focus-visible:ring-offset-2 peer-focus-visible:ring-offset-background">
            <Check className="h-3.5 w-3.5 opacity-0 transition peer-checked:opacity-100" />
          </span>
        </div>
      </label>
      <label className="flex items-start justify-between gap-3 rounded-2xl border border-[var(--border-muted)] bg-[var(--bg-surface-1)] p-3.5">
        <div className="min-w-0">
          <div className="text-sm font-semibold text-[var(--text-primary)]">
            Diarization
          </div>
        </div>
        <div className="relative mt-0.5 shrink-0">
          <input
            id="speech-text-mode-diarization"
            name="speech-text-mode"
            type="radio"
            checked={selectedMode === "diarization"}
            onChange={() => onSelectMode("diarization")}
            className="peer sr-only"
          />
          <span className="flex h-5 w-5 items-center justify-center rounded-md border border-[var(--border-strong)] bg-[var(--bg-surface-0)] text-white transition peer-checked:border-[var(--status-info-text)] peer-checked:bg-[var(--status-info-text)] peer-focus-visible:outline-none peer-focus-visible:ring-2 peer-focus-visible:ring-ring/45 peer-focus-visible:ring-offset-2 peer-focus-visible:ring-offset-background">
            <Check className="h-3.5 w-3.5 opacity-0 transition peer-checked:opacity-100" />
          </span>
        </div>
      </label>
    </div>
  );
}
