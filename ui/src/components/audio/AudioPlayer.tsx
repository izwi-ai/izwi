import type { ReactNode } from "react";
import { Pause, Play, SkipBack, SkipForward } from "lucide-react";

import type { AudioPlayback } from "@/components/audio/useAudioPlayback";
import { formatClockDuration } from "@/components/history/format";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";

export const PLAYBACK_SPEEDS = [0.75, 1, 1.25, 1.5, 2];

/** Elapsed time ("0:02"), floored like a running clock. Total lengths use
 * formatClockDuration, which rounds, so a 2.85 s take reads 0:03 in the
 * player, its facts and its list row alike. */
export function formatClockTime(totalSeconds: number): string {
  if (!Number.isFinite(totalSeconds) || totalSeconds < 0) {
    return "0:00";
  }
  const minutes = Math.floor(totalSeconds / 60);
  const seconds = Math.floor(totalSeconds % 60);
  return `${minutes}:${seconds.toString().padStart(2, "0")}`;
}

/** A span drawn behind the seek bar (transcript segment, speaker turn). */
export interface TimelineSegment {
  start: number;
  end: number;
  active?: boolean;
  /** Fill when active / inactive; defaults to the accent. */
  solid?: string;
  soft?: string;
}

const DEFAULT_SOLID = "hsl(189 76% 42%)";
const DEFAULT_SOFT = "hsla(189 76% 42% / 0.16)";

const RANGE_CLASS =
  "relative z-10 h-7 w-full cursor-pointer appearance-none bg-transparent accent-[var(--text-primary)] focus-visible:outline-none disabled:cursor-not-allowed disabled:opacity-45 [&::-moz-range-progress]:bg-transparent [&::-moz-range-thumb]:h-3.5 [&::-moz-range-thumb]:w-3.5 [&::-moz-range-thumb]:rounded-full [&::-moz-range-thumb]:border-none [&::-moz-range-thumb]:bg-[var(--text-primary)] [&::-moz-range-thumb]:shadow-md [&::-moz-range-track]:h-1.5 [&::-moz-range-track]:rounded-full [&::-moz-range-track]:bg-transparent [&::-webkit-slider-runnable-track]:h-1.5 [&::-webkit-slider-runnable-track]:rounded-full [&::-webkit-slider-runnable-track]:bg-transparent [&::-webkit-slider-thumb]:-mt-1 [&::-webkit-slider-thumb]:h-3.5 [&::-webkit-slider-thumb]:w-3.5 [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:rounded-full [&::-webkit-slider-thumb]:border-none [&::-webkit-slider-thumb]:bg-[var(--text-primary)] [&::-webkit-slider-thumb]:shadow-md";

interface AudioPlayerProps {
  playback: AudioPlayback;
  /** Spans behind the seek bar; without them the bar shows progress. */
  segments?: TimelineSegment[];
  /** Right-hand label (active speaker, segment time). */
  trailing?: ReactNode;
  /** Skip buttons for long recordings. */
  showSkip?: boolean;
  className?: string;
  "data-testid"?: string;
}

/**
 * Transport bar for one audio element: play/pause, optional 10-second
 * skips, time, a seek bar (over timeline segments when given) and speed.
 * Render the element itself with `<audio {...playback.audioProps} hidden />`.
 */
export function AudioPlayer({
  playback,
  segments,
  trailing,
  showSkip = true,
  className,
  "data-testid": testId,
}: AudioPlayerProps) {
  const { src, currentTime, duration, isPlaying, playbackRate, error } =
    playback;
  const disabled = !src;
  const progress = duration > 0 ? Math.min(currentTime / duration, 1) : 0;

  const skipButton = (delta: number) => (
    <Button
      type="button"
      variant="outline"
      size="icon"
      className="h-8 w-8 rounded-full border-[var(--border-muted)] bg-[var(--bg-surface-1)] hover:bg-[var(--bg-surface-2)]"
      onClick={() => playback.skip(delta)}
      disabled={disabled}
      title={delta < 0 ? "Rewind 10 seconds" : "Forward 10 seconds"}
      aria-label={delta < 0 ? "Rewind 10 seconds" : "Forward 10 seconds"}
    >
      {delta < 0 ? (
        <SkipBack className="h-4 w-4" />
      ) : (
        <SkipForward className="h-4 w-4" />
      )}
    </Button>
  );

  return (
    <div data-testid={testId} className={cn("flex flex-col gap-2.5", className)}>
      <div className="flex items-center gap-3">
        <div className="flex items-center gap-2">
          {showSkip ? skipButton(-10) : null}
          <Button
            type="button"
            size="icon"
            className="h-9 w-9 rounded-full bg-[var(--text-primary)] text-[var(--bg-surface-0)] shadow-md hover:bg-[var(--text-secondary)]"
            onClick={() => void playback.toggle()}
            disabled={disabled}
            aria-label={isPlaying ? "Pause" : "Play"}
          >
            {isPlaying ? (
              <Pause className="h-4 w-4" fill="currentColor" />
            ) : (
              <Play className="ml-0.5 h-4 w-4" fill="currentColor" />
            )}
          </Button>
          {showSkip ? skipButton(10) : null}
        </div>

        <div className="font-mono text-[13px] font-medium tabular-nums tracking-tight text-[var(--text-primary)]">
          {formatClockTime(currentTime)}{" "}
          <span className="font-normal text-[var(--text-muted)]">
            / {formatClockDuration(duration)}
          </span>
        </div>

        <div className="relative mx-1.5 flex h-9 flex-1 items-center">
          <div className="pointer-events-none absolute inset-x-0 top-1/2 h-7 -translate-y-1/2 overflow-hidden rounded bg-[color-mix(in_srgb,var(--bg-surface-2)_50%,transparent)]">
            {segments && duration > 0 ? (
              segments.map((segment, index) => {
                const left = (segment.start / duration) * 100;
                const width =
                  segment.end > segment.start
                    ? ((segment.end - segment.start) / duration) * 100
                    : 100;
                return (
                  <div
                    key={index}
                    className="absolute bottom-0 top-0 border-r border-[var(--bg-surface-0)] transition-all duration-200"
                    style={{
                      left: `${Math.max(left, 0)}%`,
                      width: `${Math.max(width, 0.5)}%`,
                      backgroundColor: segment.active
                        ? (segment.solid ?? DEFAULT_SOLID)
                        : (segment.soft ?? DEFAULT_SOFT),
                      opacity: segment.active ? 1 : 0.75,
                    }}
                  />
                );
              })
            ) : (
              <div
                className="absolute inset-y-0 left-0 bg-[var(--sidebar-link-active-bg)]"
                style={{ width: `${progress * 100}%` }}
              />
            )}
          </div>
          <input
            type="range"
            min={0}
            max={duration || 0}
            step={0.05}
            value={Math.min(currentTime, duration || 0)}
            onChange={(event) => playback.seek(Number(event.target.value))}
            aria-label="Seek audio timeline"
            disabled={disabled || duration <= 0}
            className={RANGE_CLASS}
          />
        </div>

        <div className="flex items-center gap-2">
          {trailing !== undefined ? (
            <div className="min-w-[68px] truncate text-right text-[13px] font-semibold text-[var(--text-primary)]">
              {trailing}
            </div>
          ) : null}
          <select
            aria-label="Playback speed"
            className="h-7 rounded-full border border-[var(--border-muted)] bg-[var(--bg-surface-1)] px-2.5 text-[11px] font-medium text-[var(--text-primary)] outline-none"
            value={playbackRate}
            onChange={(event) =>
              playback.setPlaybackRate(Number(event.target.value))
            }
          >
            {PLAYBACK_SPEEDS.map((rate) => (
              <option key={rate} value={rate}>
                {rate}x
              </option>
            ))}
          </select>
        </div>
      </div>

      {error ? (
        <div className="rounded-lg border border-[var(--danger-border)] bg-[var(--danger-bg)] px-3 py-2 text-xs text-[var(--danger-text)]">
          {error}
        </div>
      ) : null}
    </div>
  );
}
