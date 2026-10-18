import { AudioPlayer } from "@/components/audio/AudioPlayer";
import { useAudioPlayback } from "@/components/audio/useAudioPlayback";

/** Self-contained player for one recording (a record page's audio). */
export function RecordAudioPlayer({
  src,
  durationHint,
  className,
}: {
  src: string;
  durationHint?: number | null;
  className?: string;
}) {
  const playback = useAudioPlayback(src, {
    durationHint: durationHint ?? 0,
  });
  return (
    <>
      <audio {...playback.audioProps} className="hidden" />
      <AudioPlayer
        playback={playback}
        className={className}
        data-testid="record-audio-player"
      />
    </>
  );
}
