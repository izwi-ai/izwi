import {
  useEffect,
  useId,
  useRef,
  useState,
  type KeyboardEvent,
} from "react";
import { Loader2, Pause, Play } from "lucide-react";

import { StatePanel } from "@/components/ui/state-panel";
import { type VoiceLibraryItem } from "@/features/voices/types";
import { cn } from "@/lib/utils";

interface VoiceLibraryTableProps {
  items: VoiceLibraryItem[];
  emptyTitle: string;
  emptyDescription: string;
  className?: string;
  compact?: boolean;
}

interface TablePreviewPlayerProps {
  item: VoiceLibraryItem;
  activePreviewId: string | null;
  onActivePreviewChange: (id: string | null) => void;
}

const SEEKBAR_CLASS =
  "relative z-10 m-0 h-5 w-full cursor-pointer appearance-none bg-transparent align-middle accent-[var(--text-primary)] focus-visible:outline-none disabled:cursor-not-allowed disabled:opacity-45 [&::-moz-range-progress]:bg-transparent [&::-moz-range-thumb]:h-3 [&::-moz-range-thumb]:w-3 [&::-moz-range-thumb]:rounded-full [&::-moz-range-thumb]:border-none [&::-moz-range-thumb]:bg-[var(--text-primary)] [&::-moz-range-thumb]:shadow-sm [&::-moz-range-track]:h-1 [&::-moz-range-track]:rounded-full [&::-moz-range-track]:bg-transparent [&::-webkit-slider-runnable-track]:h-1 [&::-webkit-slider-runnable-track]:rounded-full [&::-webkit-slider-runnable-track]:bg-transparent [&::-webkit-slider-thumb]:-mt-1 [&::-webkit-slider-thumb]:h-3 [&::-webkit-slider-thumb]:w-3 [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:rounded-full [&::-webkit-slider-thumb]:border-none [&::-webkit-slider-thumb]:bg-[var(--text-primary)] [&::-webkit-slider-thumb]:shadow-sm";

function formatClockTime(value: number): string {
  if (!Number.isFinite(value) || value < 0) {
    return "00:00";
  }

  const rounded = Math.floor(value);
  const minutes = Math.floor(rounded / 60);
  const seconds = rounded % 60;

  return `${String(minutes).padStart(2, "0")}:${String(seconds).padStart(2, "0")}`;
}

function resolveAudioDuration(audio: HTMLAudioElement): number {
  const candidates: number[] = [];
  if (Number.isFinite(audio.duration) && audio.duration > 0) {
    candidates.push(audio.duration);
  }
  if (audio.seekable.length > 0) {
    const seekableEnd = audio.seekable.end(audio.seekable.length - 1);
    if (Number.isFinite(seekableEnd) && seekableEnd > 0) {
      candidates.push(seekableEnd);
    }
  }
  if (audio.buffered.length > 0) {
    const bufferedEnd = audio.buffered.end(audio.buffered.length - 1);
    if (Number.isFinite(bufferedEnd) && bufferedEnd > 0) {
      candidates.push(bufferedEnd);
    }
  }
  return candidates.length > 0 ? Math.max(...candidates) : 0;
}

function TablePreviewPlayer({
  item,
  activePreviewId,
  onActivePreviewChange,
}: TablePreviewPlayerProps) {
  const seekId = useId();
  const audioRef = useRef<HTMLAudioElement>(null);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);

  const hasPreview = Boolean(item.previewUrl);
  const isPlayable = hasPreview && !item.previewLoading;
  const seekMax = duration > 0 ? duration : Math.max(currentTime + 0.25, 1);
  const progress = seekMax > 0 ? Math.min(currentTime / seekMax, 1) : 0;

  useEffect(() => {
    const audio = audioRef.current;
    if (!audio) {
      return;
    }

    const handleLoadedMetadata = () => {
      setDuration(resolveAudioDuration(audio));
    };
    const handleTimeUpdate = () => {
      setCurrentTime(audio.currentTime);
      setDuration(resolveAudioDuration(audio));
    };
    const handleDurationChange = () => {
      setDuration(resolveAudioDuration(audio));
    };
    const handlePlay = () => {
      setIsPlaying(true);
      onActivePreviewChange(item.id);
    };
    const handlePause = () => {
      setIsPlaying(false);
      if (activePreviewId === item.id) {
        onActivePreviewChange(null);
      }
    };
    const handleEnded = () => {
      setIsPlaying(false);
      setCurrentTime(0);
      if (activePreviewId === item.id) {
        onActivePreviewChange(null);
      }
    };

    audio.addEventListener("loadedmetadata", handleLoadedMetadata);
    audio.addEventListener("loadeddata", handleLoadedMetadata);
    audio.addEventListener("canplay", handleLoadedMetadata);
    audio.addEventListener("timeupdate", handleTimeUpdate);
    audio.addEventListener("durationchange", handleDurationChange);
    audio.addEventListener("play", handlePlay);
    audio.addEventListener("pause", handlePause);
    audio.addEventListener("ended", handleEnded);

    return () => {
      audio.removeEventListener("loadedmetadata", handleLoadedMetadata);
      audio.removeEventListener("loadeddata", handleLoadedMetadata);
      audio.removeEventListener("canplay", handleLoadedMetadata);
      audio.removeEventListener("timeupdate", handleTimeUpdate);
      audio.removeEventListener("durationchange", handleDurationChange);
      audio.removeEventListener("play", handlePlay);
      audio.removeEventListener("pause", handlePause);
      audio.removeEventListener("ended", handleEnded);
    };
  }, [activePreviewId, item.id, onActivePreviewChange]);

  useEffect(() => {
    const audio = audioRef.current;
    if (!audio) {
      return;
    }

    audio.pause();
    audio.currentTime = 0;
    setCurrentTime(0);
    setDuration(resolveAudioDuration(audio));
    setIsPlaying(false);
  }, [item.previewUrl]);

  useEffect(() => {
    const audio = audioRef.current;
    if (!audio || activePreviewId === item.id) {
      return;
    }

    if (!audio.paused) {
      audio.pause();
    }
  }, [activePreviewId, item.id]);

  const togglePlayback = async () => {
    const audio = audioRef.current;
    if (!audio || !isPlayable) {
      return;
    }

    if (!audio.paused) {
      audio.pause();
      return;
    }

    try {
      onActivePreviewChange(item.id);
      await audio.play();
    } catch {
      onActivePreviewChange(null);
    }
  };

  const seek = (nextTime: number) => {
    const audio = audioRef.current;
    if (!audio || !hasPreview) {
      return;
    }

    audio.currentTime = nextTime;
    setCurrentTime(nextTime);
  };

  if (item.previewLoading) {
    return (
      <div className="inline-flex items-center gap-2 text-xs text-[var(--text-muted)]">
        <Loader2 className="h-3.5 w-3.5 animate-spin" />
        {item.previewMessage || "Generating preview..."}
      </div>
    );
  }

  if (!item.previewUrl) {
    return (
      <p className="text-xs leading-5 text-[var(--text-muted)]">
        {item.previewMessage || "No preview available yet."}
      </p>
    );
  }

  return (
    <div
      className="rounded-[0.9rem] border border-[var(--border-muted)] px-2 py-2"
      style={{
        backgroundImage:
          "linear-gradient(180deg, var(--bg-surface-0), var(--bg-surface-1))",
      }}
      onClick={(event) => event.stopPropagation()}
      data-testid={`voice-preview-${item.id}`}
    >
      <audio
        ref={audioRef}
        src={item.previewUrl}
        preload="metadata"
        className="hidden"
      />

      <div className="flex items-center gap-2">
        <button
          type="button"
          onClick={() => void togglePlayback()}
          disabled={!isPlayable}
          className={cn(
            "flex h-7 w-7 shrink-0 items-center justify-center rounded-full border border-[var(--border-muted)] bg-[var(--bg-surface-1)] text-[var(--text-primary)] transition-colors",
            isPlayable
              ? "hover:border-[var(--border-strong)] hover:bg-[var(--bg-surface-2)]"
              : "cursor-not-allowed text-[var(--text-muted)] opacity-80",
          )}
          aria-label={
            isPlaying
              ? `Pause preview for ${item.name}`
              : `Play preview for ${item.name}`
          }
        >
          {isPlaying ? (
            <Pause className="h-3 w-3 fill-current" />
          ) : (
            <Play className="ml-0.5 h-3 w-3 fill-current" />
          )}
        </button>

        <label htmlFor={seekId} className="sr-only">
          Seek preview for {item.name}
        </label>
        <span className="min-w-[2.3rem] font-mono text-[10px] font-medium tabular-nums text-[var(--text-muted)]">
          {formatClockTime(currentTime)}
        </span>
        <div className="group relative flex h-5 flex-1 items-center">
          <div className="pointer-events-none absolute inset-x-0 top-1/2 h-1 -translate-y-1/2 rounded-full bg-[var(--bg-surface-3)]" />
          <div
            className="pointer-events-none absolute left-0 top-1/2 h-1 -translate-y-1/2 rounded-full bg-[var(--text-primary)]"
            style={{ width: `${Math.max(progress * 100, 0)}%` }}
          />
          <input
            id={seekId}
            type="range"
            min={0}
            max={seekMax}
            step={0.05}
            value={Math.min(currentTime, seekMax)}
            onChange={(event) => seek(Number(event.target.value))}
            disabled={!hasPreview}
            className={SEEKBAR_CLASS}
          />
        </div>
        <span className="min-w-[2.3rem] text-right font-mono text-[10px] font-medium tabular-nums text-[var(--text-muted)]">
          {duration > 0 ? formatClockTime(duration) : "00:00"}
        </span>
      </div>
    </div>
  );
}

export function VoiceLibraryTable({
  items,
  emptyTitle,
  emptyDescription,
  className,
  compact = false,
}: VoiceLibraryTableProps) {
  const [activePreviewId, setActivePreviewId] = useState<string | null>(null);

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
    <div
      className={cn(
        "overflow-hidden rounded-2xl border border-[var(--border-muted)] bg-[var(--bg-surface-0)]",
        compact && "rounded-xl",
        className,
      )}
    >
      <div className="overflow-hidden 2xl:overflow-x-auto">
        <table
          className={cn(
            "block w-full border-collapse text-sm 2xl:table",
            compact ? "2xl:min-w-[64rem]" : "2xl:min-w-[72rem]",
          )}
        >
          <thead className="sticky top-0 z-[1] hidden bg-[var(--bg-surface-1)] text-left text-xs uppercase tracking-[0.14em] text-[var(--text-muted)] 2xl:table-header-group">
            <tr>
              <th className="whitespace-nowrap px-4 py-2.5 font-semibold">Voice</th>
              <th className="whitespace-nowrap px-4 py-2.5 font-semibold">Type</th>
              <th className="whitespace-nowrap px-4 py-2.5 font-semibold">Notes</th>
              <th className="whitespace-nowrap px-4 py-2.5 font-semibold">Preview</th>
              <th className="whitespace-nowrap px-4 py-2.5 font-semibold text-right">Actions</th>
            </tr>
          </thead>
          <tbody className="block 2xl:table-row-group">
            {items.map((item) => {
              const handleRowKeyDown = (event: KeyboardEvent<HTMLTableRowElement>) => {
                if (!item.onSelect) {
                  return;
                }
                if (event.key === "Enter" || event.key === " ") {
                  event.preventDefault();
                  item.onSelect();
                }
              };

              return (
                <tr
                  key={item.id}
                  tabIndex={item.onSelect ? 0 : undefined}
                  onClick={item.onSelect}
                  onKeyDown={handleRowKeyDown}
                  data-testid={`voice-row-${item.id}`}
                  className={cn(
                    "grid grid-cols-1 gap-x-5 gap-y-3 border-t border-[var(--border-muted)] px-4 py-4 align-top transition-colors hover:bg-[var(--bg-surface-1)]/60 sm:grid-cols-[minmax(0,1fr)_auto] 2xl:table-row 2xl:px-0 2xl:py-0",
                    item.onSelect &&
                      "cursor-pointer focus-visible:bg-[var(--bg-surface-1)] focus-visible:outline-none",
                  )}
                >
                  <td className="block min-w-0 sm:col-start-1 sm:row-start-1 2xl:table-cell 2xl:px-4 2xl:py-3.5">
                    <div className="break-words font-semibold text-[var(--text-primary)]">
                      {item.name}
                    </div>
                    {item.secondaryLabel ? (
                      <p className="mt-1 text-xs text-[var(--text-muted)]">
                        {item.secondaryLabel}
                      </p>
                    ) : null}
                  </td>
                  <td className="block sm:col-start-1 sm:row-start-2 2xl:table-cell 2xl:px-4 2xl:py-3.5">
                    <span className="mb-1 block text-[10px] font-semibold uppercase tracking-[0.14em] text-[var(--text-subtle)] 2xl:hidden">
                      Type
                    </span>
                    <span className="text-xs font-semibold uppercase tracking-[0.12em] text-[var(--text-secondary)]">
                      {item.categoryLabel}
                    </span>
                  </td>
                  <td className="block min-w-0 sm:col-span-2 sm:row-start-3 2xl:table-cell 2xl:px-4 2xl:py-3.5 text-[var(--text-secondary)]">
                    <span className="mb-1 block text-[10px] font-semibold uppercase tracking-[0.14em] text-[var(--text-subtle)] 2xl:hidden">
                      Notes
                    </span>
                    <p className="line-clamp-2">
                      {item.description ||
                        "No reference notes were saved for this voice yet."}
                    </p>
                  </td>
                  <td className="block min-w-0 sm:col-span-2 sm:row-start-4 2xl:table-cell 2xl:px-4 2xl:py-3.5">
                    <span className="mb-1 block text-[10px] font-semibold uppercase tracking-[0.14em] text-[var(--text-subtle)] 2xl:hidden">
                      Preview
                    </span>
                    <TablePreviewPlayer
                      item={item}
                      activePreviewId={activePreviewId}
                      onActivePreviewChange={setActivePreviewId}
                    />
                  </td>
                  <td className="block sm:col-start-2 sm:row-start-1 2xl:table-cell 2xl:px-4 2xl:py-3.5">
                    <div
                      className="flex flex-wrap justify-start gap-2 sm:justify-end"
                      onClick={(event) => event.stopPropagation()}
                      role="group"
                      aria-label={`Actions for ${item.name}`}
                      data-testid={`voice-actions-${item.id}`}
                    >
                      {item.actions}
                    </div>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}
