import { SegmentedFilter } from "@/components/history/HistoryToolbar";
import type { SpeechTextCreationMode } from "@/features/speech-text/creationMode";

interface SpeechTextModeSwitchProps {
  selectedMode: SpeechTextCreationMode;
  onSelectMode?: (mode: SpeechTextCreationMode) => void;
  /** Locked while an upload is under way. */
  disabled?: boolean;
}

const MODE_OPTIONS: Array<{ id: SpeechTextCreationMode; label: string }> = [
  { id: "transcription", label: "Transcription" },
  { id: "speaker_attributed_asr", label: "Speaker Attributed ASR" },
  { id: "diarization", label: "Diarization" },
];

export function SpeechTextModeSwitch({
  selectedMode,
  onSelectMode,
  disabled = false,
}: SpeechTextModeSwitchProps) {
  if (!onSelectMode) {
    return null;
  }

  return (
    <div className="mt-3">
      <SegmentedFilter
        label="Transcript type"
        options={MODE_OPTIONS}
        value={selectedMode}
        onChange={onSelectMode}
        disabled={disabled}
      />
    </div>
  );
}
