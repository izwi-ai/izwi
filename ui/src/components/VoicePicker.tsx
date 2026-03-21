import {
  useEffect,
  useId,
  useMemo,
  useRef,
  useState,
  type KeyboardEvent,
  type ReactNode,
} from "react";
import {
  Loader2,
  Music4,
  Pause,
  Play,
} from "lucide-react";

import { cn } from "@/lib/utils";
import { Card } from "@/components/ui/card";
import { StatePanel } from "@/components/ui/state-panel";
import { StatusBadge } from "@/components/ui/status-badge";
import {
  VOICE_ROUTE_BODY_COPY_CLASS,
  VOICE_ROUTE_META_COPY_CLASS,
  VOICE_ROUTE_SECTION_LABEL_CLASS,
} from "@/components/voiceRouteTypography";

export interface VoicePickerItem {
  id: string;
  name: string;
  categoryLabel: string;
  description?: string;
  meta?: string[];
  previewUrl?: string | null;
  previewMessage?: string | null;
  previewLoading?: boolean;
  selected?: boolean;
  onSelect?: () => void;
  actions?: ReactNode;
}

interface VoicePickerProps {
  items: VoicePickerItem[];
  emptyTitle: string;
  emptyDescription: string;
  className?: string;
}

interface VoicePreviewPlayerProps {
  item: VoicePickerItem;
  activePreviewId: string | null;
  onActivePreviewChange: (id: string | null) => void;
}

const SEEKBAR_CLASS =
  "relative z-10 h-5 w-full cursor-pointer appearance-none bg-transparent accent-[var(--text-primary)] focus-visible:outline-none disabled:cursor-not-allowed disabled:opacity-45 [&::-moz-range-progress]:bg-transparent [&::-moz-range-thumb]:h-3 [&::-moz-range-thumb]:w-3 [&::-moz-range-thumb]:rounded-full [&::-moz-range-thumb]:border-none [&::-moz-range-thumb]:bg-[var(--text-primary)] [&::-moz-range-thumb]:shadow-sm [&::-moz-range-track]:h-1 [&::-moz-range-track]:rounded-full [&::-moz-range-track]:bg-transparent [&::-webkit-slider-runnable-track]:h-1 [&::-webkit-slider-runnable-track]:rounded-full [&::-webkit-slider-runnable-track]:bg-transparent [&::-webkit-slider-thumb]:-mt-1 [&::-webkit-slider-thumb]:h-3 [&::-webkit-slider-thumb]:w-3 [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:rounded-full [&::-webkit-slider-thumb]:border-none [&::-webkit-slider-thumb]:bg-[var(--text-primary)] [&::-webkit-slider-thumb]:shadow-sm";

function formatClockTime(value: number): string {
  if (!Number.isFinite(value) || value < 0) {
    return "00:00";
  }

  const rounded = Math.floor(value);
  const minutes = Math.floor(rounded / 60);
  const seconds = rounded % 60;

  return `${String(minutes).padStart(2, "0")}:${String(seconds).padStart(2, "0")}`;
}

function VoicePreviewPlayer({
  item,
  activePreviewId,
  onActivePreviewChange,
}: VoicePreviewPlayerProps) {
  const audioRef = useRef<HTMLAudioElement>(null);
  const seekId = useId();
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [audioError, setAudioError] = useState<string | null>(null);

  const hasPreview = Boolean(item.previewUrl);
  const isPlayable = hasPreview && !item.previewLoading;
  const progress = duration > 0 ? Math.min(currentTime / duration, 1) : 0;
  const hint = audioError ?? item.previewMessage ?? null;

  useEffect(() => {
    const audio = audioRef.current;
    if (!audio) {
      return;
    }

    const handleLoadedMetadata = () => {
      setDuration(Number.isFinite(audio.duration) ? audio.duration : 0);
      setAudioError(null);
    };
    const handleTimeUpdate = () => setCurrentTime(audio.currentTime);
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
    const handleError = () => {
      setAudioError("Preview unavailable for this voice.");
      setIsPlaying(false);
      setCurrentTime(0);
      setDuration(0);
      if (activePreviewId === item.id) {
        onActivePreviewChange(null);
      }
    };

    audio.addEventListener("loadedmetadata", handleLoadedMetadata);
    audio.addEventListener("timeupdate", handleTimeUpdate);
    audio.addEventListener("play", handlePlay);
    audio.addEventListener("pause", handlePause);
    audio.addEventListener("ended", handleEnded);
    audio.addEventListener("error", handleError);

    return () => {
      audio.removeEventListener("loadedmetadata", handleLoadedMetadata);
      audio.removeEventListener("timeupdate", handleTimeUpdate);
      audio.removeEventListener("play", handlePlay);
      audio.removeEventListener("pause", handlePause);
      audio.removeEventListener("ended", handleEnded);
      audio.removeEventListener("error", handleError);
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
    setDuration(0);
    setIsPlaying(false);
    setAudioError(null);
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
      setAudioError("Preview unavailable for this voice.");
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

  return (
    <div
      className="rounded-[0.9rem] border border-[var(--border-muted)] px-2 py-2"
      style={{
        backgroundImage:
          "radial-gradient(circle at top right, var(--accent-soft), transparent 42%), linear-gradient(180deg, var(--bg-surface-0), var(--bg-surface-1))",
      }}
      onClick={(event) => event.stopPropagation()}
      data-testid={`voice-preview-${item.id}`}
    >
      {item.previewUrl ? (
        <audio
          ref={audioRef}
          src={item.previewUrl}
          preload="metadata"
          className="hidden"
        />
      ) : null}

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
          {item.previewLoading ? (
            <Loader2 className="h-3 w-3 animate-spin" />
          ) : isPlaying ? (
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
        <div className="group relative flex-1">
          <div className="pointer-events-none absolute inset-x-0 top-1/2 h-1 -translate-y-1/2 rounded-full bg-[var(--bg-surface-3)]" />
          <div
            className="pointer-events-none absolute left-0 top-1/2 h-1 -translate-y-1/2 rounded-full bg-[var(--text-primary)]"
            style={{ width: `${Math.max(progress * 100, 0)}%` }}
          />
          <input
            id={seekId}
            type="range"
            min={0}
            max={duration || 0}
            step={0.05}
            value={Math.min(currentTime, duration || 0)}
            onChange={(event) => seek(Number(event.target.value))}
            disabled={!hasPreview || duration <= 0}
            className={SEEKBAR_CLASS}
          />
        </div>
        <span className="min-w-[2.3rem] text-right font-mono text-[10px] font-medium tabular-nums text-[var(--text-muted)]">
          {duration > 0 ? formatClockTime(duration) : "00:00"}
        </span>
      </div>

      {hint ? (
        <p className="mt-1.5 text-[11px] leading-4 text-[var(--text-muted)]">
          {hint}
        </p>
      ) : null}
    </div>
  );
}

function VoiceCard({
  item,
  activePreviewId,
  onActivePreviewChange,
}: {
  item: VoicePickerItem;
  activePreviewId: string | null;
  onActivePreviewChange: (id: string | null) => void;
}) {
  const titleMeta = useMemo(() => item.meta ?? [], [item.meta]);

  const handleKeyDown = (event: KeyboardEvent<HTMLDivElement>) => {
    if (!item.onSelect) {
      return;
    }

    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      item.onSelect();
    }
  };

  return (
    <Card
      key={item.id}
      className={cn(
        "relative h-full overflow-hidden rounded-[1rem] border-[var(--border-muted)] p-0 transition-[border-color,transform,box-shadow] duration-200",
        item.selected &&
          "border-[var(--border-strong)] shadow-[var(--shadow-soft)]",
        item.onSelect &&
          "cursor-pointer hover:-translate-y-[1px] hover:border-[var(--border-strong)] hover:shadow-[var(--shadow-raised)]",
      )}
      style={{
        backgroundImage:
          "radial-gradient(circle at top left, var(--accent-soft), transparent 32%), linear-gradient(180deg, var(--bg-surface-1), var(--bg-surface-1))",
      }}
      role={item.onSelect ? "button" : undefined}
      tabIndex={item.onSelect ? 0 : undefined}
      onClick={item.onSelect}
      onKeyDown={handleKeyDown}
      data-testid={`voice-card-${item.id}`}
    >
      <div className="absolute inset-x-3.5 top-0 h-px bg-[var(--accent-strong)]" />
      <div className="relative flex h-full flex-col gap-2.5 p-3 sm:p-3.5">
        <div className="min-w-0">
          <div className="flex flex-wrap items-center gap-2">
            <StatusBadge>{item.categoryLabel}</StatusBadge>
            {item.selected ? (
              <StatusBadge tone="info">Selected</StatusBadge>
            ) : null}
          </div>
          <h3 className="mt-1 line-clamp-1 text-lg font-semibold leading-tight tracking-tight text-[var(--text-primary)]">
            {item.name}
          </h3>
          {titleMeta.length > 0 ? (
            <div className="mt-1 flex flex-wrap gap-1.5">
              {titleMeta.map((meta) => (
                <span
                  key={`${item.id}-${meta}`}
                  className={cn(
                    VOICE_ROUTE_META_COPY_CLASS,
                    "rounded-full border border-[var(--border-muted)] bg-[var(--bg-surface-0)]/70 px-2 py-0.5 text-[10px] font-semibold",
                  )}
                >
                  {meta}
                </span>
              ))}
            </div>
          ) : null}
        </div>

        <div>
          <div className={cn(VOICE_ROUTE_SECTION_LABEL_CLASS, "mb-1")}>
            Voice Notes
          </div>
          <p
            className={cn(
              VOICE_ROUTE_BODY_COPY_CLASS,
              "line-clamp-2 text-sm leading-5 text-[var(--text-secondary)]",
              !item.description && "text-[var(--text-muted)]",
            )}
          >
            {item.description || "No reference notes were saved for this voice yet."}
          </p>
        </div>

        <div>
          <div className={cn(VOICE_ROUTE_SECTION_LABEL_CLASS, "mb-1")}>
            Preview
          </div>
          <VoicePreviewPlayer
            item={item}
            activePreviewId={activePreviewId}
            onActivePreviewChange={onActivePreviewChange}
          />
        </div>

        {item.actions ? (
          <div
            className="mt-auto grid gap-1.5 [&>*]:w-full [&>*]:justify-center"
            onClick={(event) => event.stopPropagation()}
          >
            {item.actions}
          </div>
        ) : null}
      </div>
    </Card>
  );
}

export function VoicePicker({
  items,
  emptyTitle,
  emptyDescription,
  className,
}: VoicePickerProps) {
  const [activePreviewId, setActivePreviewId] = useState<string | null>(null);

  if (items.length === 0) {
    return (
      <StatePanel
        title={emptyTitle}
        description={emptyDescription}
        icon={Music4}
        align="center"
        dashed
        className={className}
      />
    );
  }

  return (
    <div
      className={cn(
        "grid grid-cols-1 gap-2.5 lg:grid-cols-2 2xl:grid-cols-3",
        className,
      )}
    >
      {items.map((item) => (
        <VoiceCard
          key={item.id}
          item={item}
          activePreviewId={activePreviewId}
          onActivePreviewChange={setActivePreviewId}
        />
      ))}
    </div>
  );
}
