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

  const options: Array<{
    value: SpeechTextCreationMode;
    label: string;
    id: string;
  }> = [
    {
      value: "transcription",
      label: "Transcription",
      id: "speech-text-mode-transcription",
    },
    {
      value: "speaker_attributed_asr",
      label: "Speaker Attributed ASR",
      id: "speech-text-mode-speaker-attributed-asr",
    },
    {
      value: "diarization",
      label: "Diarization",
      id: "speech-text-mode-diarization",
    },
  ];

  return (
    <div className="mt-3 grid gap-2 lg:grid-cols-3">
      {options.map((option) => (
        <label
          key={option.value}
          className="flex cursor-pointer items-start justify-between gap-3 rounded-2xl border border-[var(--border-muted)] bg-[var(--bg-surface-1)] p-3.5 transition-colors has-[:checked]:border-[var(--text-muted)]"
        >
          <div className="min-w-0">
            <div className="text-sm font-semibold text-[var(--text-primary)]">
              {option.label}
            </div>
          </div>
          <div className="relative mt-0.5 shrink-0">
            <input
              id={option.id}
              name="speech-text-mode"
              type="radio"
              checked={selectedMode === option.value}
              onChange={() => onSelectMode(option.value)}
              className="peer sr-only"
            />
            {/* The dot is a child of the peer's sibling, so it is targeted
                from the sibling: peer-checked:[&>span]. */}
            <span className="flex h-5 w-5 items-center justify-center rounded-full border border-[var(--border-strong)] bg-[var(--bg-surface-0)] transition peer-checked:border-[var(--accent-solid)] peer-checked:[&>span]:scale-100 peer-focus-visible:outline-none peer-focus-visible:ring-2 peer-focus-visible:ring-ring/45 peer-focus-visible:ring-offset-2 peer-focus-visible:ring-offset-background">
              <span className="h-2.5 w-2.5 scale-0 rounded-full bg-[var(--accent-solid)] transition-transform" />
            </span>
          </div>
        </label>
      ))}
    </div>
  );
}
