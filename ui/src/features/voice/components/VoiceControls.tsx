import { Captions, Mic, MicOff, X } from "lucide-react";

import { cn } from "@/lib/utils";

const roundButton =
  "flex h-[52px] w-[52px] items-center justify-center rounded-full border transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[var(--border-strong)] focus-visible:ring-offset-2 focus-visible:ring-offset-[var(--bg-surface-0)]";

/** Mute, End and the transcript toggle, shown while a conversation runs. */
export function VoiceControls({
  isMicMuted,
  isTranscriptOpen,
  onToggleMic,
  onEnd,
  onToggleTranscript,
}: {
  isMicMuted: boolean;
  isTranscriptOpen: boolean;
  onToggleMic: () => void;
  onEnd: () => void;
  onToggleTranscript: () => void;
}) {
  return (
    <div className="flex items-center justify-center gap-4">
      <button
        type="button"
        onClick={onToggleMic}
        aria-label={isMicMuted ? "Unmute microphone" : "Mute microphone"}
        aria-pressed={isMicMuted}
        title={isMicMuted ? "Unmute (M)" : "Mute (M)"}
        className={cn(
          roundButton,
          isMicMuted
            ? "border-[var(--border-strong)] bg-[var(--accent-solid)] text-[var(--text-on-accent)]"
            : "border-[var(--border-default)] bg-[var(--bg-surface-2)] text-[var(--text-primary)] hover:bg-[var(--bg-surface-3)]",
        )}
      >
        {isMicMuted ? <MicOff className="h-5 w-5" /> : <Mic className="h-5 w-5" />}
      </button>
      <button
        type="button"
        onClick={onEnd}
        title="End (Space)"
        className="flex h-[52px] items-center gap-2 rounded-full bg-red-600 px-6 text-[15px] font-medium text-white transition-colors hover:bg-red-700 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-red-400 focus-visible:ring-offset-2 focus-visible:ring-offset-[var(--bg-surface-0)]"
      >
        <X className="h-[18px] w-[18px]" strokeWidth={2.25} />
        End
      </button>
      <button
        type="button"
        onClick={onToggleTranscript}
        aria-label={isTranscriptOpen ? "Hide transcript" : "Show transcript"}
        aria-pressed={isTranscriptOpen}
        className={cn(
          roundButton,
          isTranscriptOpen
            ? "border-[var(--border-strong)] bg-[var(--accent-solid)] text-[var(--text-on-accent)]"
            : "border-[var(--border-default)] bg-[var(--bg-surface-2)] text-[var(--text-primary)] hover:bg-[var(--bg-surface-3)]",
        )}
      >
        <Captions className="h-5 w-5" />
      </button>
    </div>
  );
}
