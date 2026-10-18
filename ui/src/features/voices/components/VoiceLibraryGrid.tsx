import { useEffect, useRef, useState } from "react";
import { Loader2, Pause, Play } from "lucide-react";

import { formatClockTime } from "@/components/audio/AudioPlayer";
import { useAudioPlayback } from "@/components/audio/useAudioPlayback";
import { RowBadge } from "@/components/history/RowBadge";
import { RowMenu } from "@/components/history/RowMenu";
import { Button } from "@/components/ui/button";
import { StatePanel } from "@/components/ui/state-panel";
import { type VoiceLibraryItem } from "@/features/voices/types";
import { cn } from "@/lib/utils";

function initials(name: string): string {
  return (
    name
      .split(/[\s–-]+/)
      .filter(Boolean)
      .map((part) => part[0])
      .join("")
      .slice(0, 2)
      .toUpperCase() || "?"
  );
}

const RANGE_CLASS =
  "relative z-10 m-0 h-5 w-full cursor-pointer appearance-none bg-transparent focus-visible:outline-none disabled:cursor-default [&::-moz-range-thumb]:h-2.5 [&::-moz-range-thumb]:w-2.5 [&::-moz-range-thumb]:rounded-full [&::-moz-range-thumb]:border-none [&::-moz-range-thumb]:bg-[var(--text-primary)] [&::-moz-range-track]:bg-transparent [&::-webkit-slider-runnable-track]:bg-transparent [&::-webkit-slider-thumb]:h-2.5 [&::-webkit-slider-thumb]:w-2.5 [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:rounded-full [&::-webkit-slider-thumb]:bg-[var(--text-primary)] disabled:[&::-webkit-slider-thumb]:opacity-0";

function VoicePreview({
  item,
  activeId,
  onActivate,
}: {
  item: VoiceLibraryItem;
  activeId: string | null;
  onActivate: (id: string | null) => void;
}) {
  const playback = useAudioPlayback(item.previewUrl, {
    loadErrorMessage: "Preview unavailable.",
    playErrorMessage: "Preview unavailable.",
  });
  const { isPlaying, pause, toggle } = playback;
  // A play press that asked for a preview plays it once it arrives.
  const playWhenReadyRef = useRef(false);

  // Starting another card stops this one. A card claims "active" when its
  // play button is pressed, before playback starts.
  useEffect(() => {
    if (activeId !== null && activeId !== item.id && isPlaying) {
      pause();
    }
  }, [activeId, isPlaying, item.id, pause]);

  useEffect(() => {
    if (item.previewUrl && playWhenReadyRef.current) {
      playWhenReadyRef.current = false;
      void toggle();
    }
  }, [item.previewUrl, toggle]);

  const canRequest = !item.previewUrl && Boolean(item.onRequestPreview);
  const disabled = item.previewLoading || (!item.previewUrl && !canRequest);

  return (
    <div className="flex items-center gap-2.5" data-testid={`voice-preview-${item.id}`}>
      <audio {...playback.audioProps} className="hidden" />
      <button
        type="button"
        disabled={disabled}
        onClick={() => {
          if (!isPlaying) {
            onActivate(item.id);
          }
          if (item.previewUrl) {
            void toggle();
          } else if (item.onRequestPreview) {
            playWhenReadyRef.current = true;
            item.onRequestPreview();
          }
        }}
        aria-label={`${isPlaying ? "Pause" : "Play"} preview for ${item.name}`}
        className="inline-flex h-[30px] w-[30px] shrink-0 items-center justify-center rounded-full border border-[var(--border-default)] bg-[var(--surface-panel-alt)] text-[var(--text-primary)] transition-colors hover:border-[var(--border-strong)] disabled:cursor-not-allowed disabled:opacity-40"
      >
        {item.previewLoading ? (
          <Loader2 className="h-3 w-3 animate-spin" />
        ) : isPlaying ? (
          <Pause className="h-3 w-3" fill="currentColor" />
        ) : (
          <Play className="ml-0.5 h-3 w-3" fill="currentColor" />
        )}
      </button>
      <div className="relative flex h-5 flex-1 items-center">
        <div className="pointer-events-none absolute inset-x-0 top-1/2 h-1 -translate-y-1/2 rounded-full bg-[var(--surface-raised)]" />
        <div
          className="pointer-events-none absolute left-0 top-1/2 h-1 -translate-y-1/2 rounded-full bg-[var(--text-muted)]"
          style={{
            width: `${playback.duration > 0 ? Math.min(playback.currentTime / playback.duration, 1) * 100 : 0}%`,
          }}
        />
        <input
          type="range"
          min={0}
          max={playback.duration || 0}
          step={0.05}
          value={Math.min(playback.currentTime, playback.duration || 0)}
          onChange={(event) => playback.seek(Number(event.target.value))}
          disabled={!item.previewUrl || playback.duration <= 0}
          aria-label={`Seek preview for ${item.name}`}
          className={RANGE_CLASS}
        />
      </div>
      <span className="w-9 text-right font-mono text-[11px] tabular-nums text-[var(--text-subtle)]">
        {item.previewLoading
          ? "…"
          : item.previewUrl
            ? formatClockTime(playback.duration)
            : "—"}
      </span>
    </div>
  );
}

interface VoiceLibraryGridProps {
  items: VoiceLibraryItem[];
  emptyTitle: string;
  emptyDescription: string;
  className?: string;
}

/** Voice cards: identity, notes, a preview player and actions. */
export function VoiceLibraryGrid({
  items,
  emptyTitle,
  emptyDescription,
  className,
}: VoiceLibraryGridProps) {
  const [activeId, setActiveId] = useState<string | null>(null);

  if (items.length === 0) {
    return (
      <StatePanel
        title={emptyTitle}
        description={emptyDescription}
        align="center"
        dashed
        className={className}
      />
    );
  }

  return (
    <ul
      aria-label="Voices"
      className={cn(
        "grid gap-3 sm:grid-cols-2 xl:grid-cols-3",
        className,
      )}
    >
      {items.map((item) => (
        <li
          key={item.id}
          data-testid={`voice-card-${item.id}`}
          className="flex min-w-0 flex-col gap-3 rounded-[14px] border border-[var(--border-subtle)] bg-[var(--surface-panel)] p-3.5"
        >
          <div className="flex items-start gap-3">
            <span
              aria-hidden="true"
              className="inline-flex h-10 w-10 shrink-0 items-center justify-center rounded-[10px] border border-[var(--speaker-avatar-border)] bg-[var(--speaker-avatar-bg)] text-[13px] font-semibold text-[var(--speaker-avatar-text)]"
            >
              {initials(item.name)}
            </span>
            <div className="min-w-0 flex-1">
              <h3 className="truncate text-[14px] font-semibold text-[var(--text-primary)]">
                {item.name}
              </h3>
              {item.meta ? (
                <p className="truncate text-xs text-[var(--text-subtle)]">
                  {item.meta}
                </p>
              ) : null}
            </div>
            {item.menuItems && item.menuItems.length > 0 ? (
              <RowMenu label={item.name} items={item.menuItems} />
            ) : null}
          </div>

          <p className="line-clamp-2 min-h-[2.5rem] text-[13px] leading-5 text-[var(--text-muted)]">
            {item.description || "No notes saved for this voice."}
          </p>

          <VoicePreview item={item} activeId={activeId} onActivate={setActiveId} />

          <div className="flex items-center gap-2 border-t border-[var(--border-subtle)] pt-3">
            {item.primaryAction ? (
              <Button
                type="button"
                variant="outline"
                size="sm"
                className="h-8 gap-1.5 px-2.5 text-[13px]"
                onClick={item.primaryAction.onClick}
              >
                <item.primaryAction.icon className="h-4 w-4" />
                {item.primaryAction.label}
              </Button>
            ) : null}
            <RowBadge className="ml-auto">{item.kindLabel}</RowBadge>
          </div>
        </li>
      ))}
    </ul>
  );
}
