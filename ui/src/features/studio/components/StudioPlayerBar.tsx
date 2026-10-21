import type { KeyboardEvent, MouseEvent } from "react";
import { Pause, Play, SkipBack, SkipForward } from "lucide-react";

import { formatClockDuration } from "@/components/history/format";
import type { ProjectPlayer } from "@/features/studio/hooks/useProjectPlayer";

const SEEK_STEP = 0.02;

/** Plays the whole project, paragraph after paragraph. */
export function StudioPlayerBar({
  player,
  currentNumber,
  paragraphCount,
}: {
  player: ProjectPlayer;
  currentNumber: number | null;
  paragraphCount: number;
}) {
  const fraction = player.total > 0 ? player.position / player.total : 0;
  const seekFromPointer = (event: MouseEvent<HTMLDivElement>) => {
    const box = event.currentTarget.getBoundingClientRect();
    player.seek((event.clientX - box.left) / box.width);
  };
  const seekFromKeys = (event: KeyboardEvent<HTMLDivElement>) => {
    if (event.key === "ArrowRight") player.seek(fraction + SEEK_STEP);
    else if (event.key === "ArrowLeft") player.seek(fraction - SEEK_STEP);
    else return;
    event.preventDefault();
  };
  const control =
    "flex h-8 w-8 items-center justify-center rounded-full text-(--text-muted) transition-colors hover:text-(--text-primary) disabled:opacity-40";

  return (
    <footer
      aria-label="Player"
      className="sticky bottom-0 z-10 -mx-4 mt-6 flex flex-wrap items-center gap-x-4 gap-y-2 border-t border-(--border-subtle) bg-(--surface-canvas)/95 px-4 py-3 backdrop-blur-sm sm:-mx-6 sm:px-6"
    >
      <div className="flex items-center gap-1">
        <button type="button" aria-label="Previous paragraph" onClick={player.previous} disabled={!player.hasAudio} className={control}>
          <SkipBack className="h-4 w-4 fill-current" />
        </button>
        <button
          type="button"
          aria-label={player.playing ? "Pause" : "Play"}
          onClick={player.toggle}
          disabled={!player.hasAudio}
          className="flex h-9 w-9 items-center justify-center rounded-full bg-(--accent-solid) text-(--text-on-accent) transition-opacity hover:opacity-90 disabled:opacity-40"
        >
          {player.playing ? (
            <Pause className="h-3.5 w-3.5 fill-current" />
          ) : (
            <Play className="ml-0.5 h-3.5 w-3.5 fill-current" />
          )}
        </button>
        <button type="button" aria-label="Next paragraph" onClick={player.next} disabled={!player.hasAudio} className={control}>
          <SkipForward className="h-4 w-4 fill-current" />
        </button>
      </div>
      <span className="text-xs tabular-nums text-(--text-muted)">
        {formatClockDuration(player.position)}
      </span>
      <div
        role="slider"
        tabIndex={player.hasAudio ? 0 : -1}
        aria-label="Position in project"
        aria-valuemin={0}
        aria-valuemax={Math.round(player.total)}
        aria-valuenow={Math.round(player.position)}
        aria-valuetext={`${formatClockDuration(player.position)} of ${formatClockDuration(player.total)}`}
        onClick={player.hasAudio ? seekFromPointer : undefined}
        onKeyDown={seekFromKeys}
        className="relative h-4 min-w-32 flex-1 cursor-pointer outline-hidden focus-visible:ring-2 focus-visible:ring-(--border-strong)"
      >
        <div className="absolute inset-x-0 top-1/2 h-1 -translate-y-1/2 rounded-full bg-(--surface-raised)">
          <div className="h-full rounded-full bg-(--text-primary)" style={{ width: `${fraction * 100}%` }} />
        </div>
        {player.marks.map((mark) => (
          <span
            key={mark}
            aria-hidden="true"
            className="absolute top-1/2 h-2 w-px -translate-y-1/2 bg-(--border-strong)"
            style={{ left: `${mark * 100}%` }}
          />
        ))}
      </div>
      <span className="text-xs tabular-nums text-(--text-subtle)">
        {formatClockDuration(player.total)}
      </span>
      <span className="hidden text-xs text-(--text-subtle) md:inline">
        {currentNumber
          ? `Paragraph ${currentNumber} of ${paragraphCount}`
          : `${paragraphCount} ${paragraphCount === 1 ? "paragraph" : "paragraphs"}`}
      </span>
      <button
        type="button"
        onClick={player.cycleSpeed}
        aria-label={`Playback speed ${player.speed}×`}
        className="h-7 rounded-full border border-(--border-subtle) px-2.5 text-xs tabular-nums text-(--text-muted) transition-colors hover:text-(--text-primary)"
      >
        {player.speed}×
      </button>
    </footer>
  );
}
