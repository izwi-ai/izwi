import { AudioPlayer } from "@/components/audio/AudioPlayer";
import { useAudioPlayback } from "@/components/audio/useAudioPlayback";

/** Self-contained player for one recording (a record page's audio). */
export function RecordAudioPlayer({
  src,
  durationHint,
  className,
  compact = false,
}: {
  src: string;
  durationHint?: number | null;
  className?: string;
  /** No skip buttons and no preloading: for players repeated in a list. */
  compact?: boolean;
}) {
  const playback = useAudioPlayback(src, {
    durationHint: durationHint ?? 0,
    preload: compact ? "none" : "metadata",
  });
  return (
    <>
      <audio {...playback.audioProps} className="hidden" />
      <AudioPlayer
        playback={playback}
        className={className}
        showSkip={!compact}
        data-testid="record-audio-player"
      />
    </>
  );
}
