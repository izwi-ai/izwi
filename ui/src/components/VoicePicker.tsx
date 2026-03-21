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

const WAVEFORM_BARS = [
  0.18, 0.28, 0.52, 0.82, 0.6, 0.32, 0.25, 0.44, 0.72, 0.92, 0.66, 0.3, 0.22,
  0.5, 0.84, 0.62, 0.34, 0.26, 0.48, 0.86, 0.68, 0.38, 0.24,
] as const;

const SEEKBAR_CLASS =
  "relative z-10 h-7 w-full cursor-pointer appearance-none bg-transparent accent-[var(--text-primary)] focus-visible:outline-none disabled:cursor-not-allowed disabled:opacity-45 [&::-moz-range-progress]:bg-transparent [&::-moz-range-thumb]:h-3.5 [&::-moz-range-thumb]:w-3.5 [&::-moz-range-thumb]:rounded-full [&::-moz-range-thumb]:border-none [&::-moz-range-thumb]:bg-[var(--text-primary)] [&::-moz-range-thumb]:shadow-md [&::-moz-range-track]:h-1.5 [&::-moz-range-track]:rounded-full [&::-moz-range-track]:bg-transparent [&::-webkit-slider-runnable-track]:h-1.5 [&::-webkit-slider-runnable-track]:rounded-full [&::-webkit-slider-runnable-track]:bg-transparent [&::-webkit-slider-thumb]:-mt-1 [&::-webkit-slider-thumb]:h-3.5 [&::-webkit-slider-thumb]:w-3.5 [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:rounded-full [&::-webkit-slider-thumb]:border-none [&::-webkit-slider-thumb]:bg-[var(--text-primary)] [&::-webkit-slider-thumb]:shadow-md";

function itemInitial(name: string): string {
  return name.trim().charAt(0).toUpperCase() || "V";
}

function formatClockTime(value: number): string {
  if (!Number.isFinite(value) || value < 0) {
    return "00:00";
  }

  const rounded = Math.floor(value);
  const minutes = Math.floor(rounded / 60);
  const seconds = rounded % 60;

  return `${String(minutes).padStart(2, "0")}:${String(seconds).padStart(2, "0")}`;
}

function VoiceGlyph({ name }: { name: string }) {
  return (
    <div
      className="relative flex h-16 w-16 shrink-0 items-center justify-center rounded-full border border-[var(--border-muted)] shadow-[var(--shadow-soft)]"
      style={{
        backgroundImage:
          "radial-gradient(circle at 28% 24%, var(--accent-soft), transparent 38%), linear-gradient(145deg, var(--bg-surface-2), var(--bg-surface-3))",
      }}
    >
      <div className="absolute inset-[1px] rounded-full bg-[var(--accent-soft)]" />
      <div className="relative flex items-end gap-0.5">
        {[0.28, 0.52, 0.9, 1, 0.66, 0.38, 0.6, 0.44].map((height, index) => (
          <span
            key={`${name}-glyph-${index}`}
            className="w-1 rounded-full bg-[var(--text-primary)]/80"
            style={{ height: `${Math.max(height * 1.35, 0.35)}rem` }}
          />
        ))}
      </div>
      <div className="absolute -bottom-1 -right-1 flex h-6 w-6 items-center justify-center rounded-full border border-[var(--border-muted)] bg-[var(--bg-surface-0)] text-[10px] font-semibold text-[var(--text-primary)] shadow-sm">
        {itemInitial(name)}
      </div>
    </div>
  );
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
  const highlightedBarCount = Math.round(progress * WAVEFORM_BARS.length);

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
      className="rounded-[1.3rem] border border-[var(--border-muted)] px-3.5 py-3.5"
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

      <div className="flex items-center gap-4">
        <button
          type="button"
          onClick={() => void togglePlayback()}
          disabled={!isPlayable}
          className={cn(
            "flex h-11 w-11 shrink-0 items-center justify-center rounded-full border border-[var(--border-muted)] bg-[var(--bg-surface-1)] text-[var(--text-primary)] transition-colors",
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
            <Loader2 className="h-5 w-5 animate-spin" />
          ) : isPlaying ? (
            <Pause className="h-5 w-5 fill-current" />
          ) : (
            <Play className="ml-0.5 h-5 w-5 fill-current" />
          )}
        </button>

        <div className="min-w-0 flex-1">
          <div className="flex h-12 items-center gap-1">
            {WAVEFORM_BARS.map((barHeight, index) => {
              const isHighlighted = index < highlightedBarCount;
              const isAnimated = isPlaying && (index + 2) % 3 === 0;

              return (
                <span
                  key={`${item.id}-wave-${index}`}
                  className={cn(
                    "w-1 rounded-full transition-[opacity,transform,background-color] duration-200",
                    isPlayable ? "opacity-100" : "opacity-40",
                    isHighlighted
                      ? "bg-[var(--text-primary)]"
                      : "bg-[var(--text-muted)]/45",
                    isAnimated && "translate-y-[-1px]",
                  )}
                  style={{
                    height: `${Math.max(barHeight * 2.15, 0.35)}rem`,
                  }}
                />
              );
            })}
          </div>

          {hint ? (
            <p className="mt-1 text-xs leading-5 text-[var(--text-muted)]">{hint}</p>
          ) : null}
        </div>
      </div>

      <div className="mt-3.5 flex items-center gap-3">
        <label
          htmlFor={seekId}
          className="sr-only"
        >
          Seek preview for {item.name}
        </label>
        <span className="min-w-[3rem] font-mono text-[12px] font-medium tabular-nums text-[var(--text-muted)]">
          {formatClockTime(currentTime)}
        </span>
        <div className="group relative flex-1">
          <div className="pointer-events-none absolute inset-x-0 top-1/2 h-1.5 -translate-y-1/2 rounded-full bg-[var(--bg-surface-3)]" />
          <div
            className="pointer-events-none absolute left-0 top-1/2 h-1.5 -translate-y-1/2 rounded-full bg-[var(--text-primary)]"
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
        <span className="min-w-[3rem] text-right font-mono text-[12px] font-medium tabular-nums text-[var(--text-muted)]">
          {duration > 0 ? formatClockTime(duration) : "00:00"}
        </span>
      </div>
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
        "relative h-full overflow-hidden rounded-[1.55rem] border-[var(--border-muted)] p-0 transition-[border-color,transform,box-shadow] duration-200",
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
      <div className="absolute inset-x-6 top-0 h-px bg-[var(--accent-strong)]" />
      <div className="relative flex h-full flex-col gap-5 p-5 sm:p-6">
        <div className="flex items-start gap-4">
          <VoiceGlyph name={item.name} />
          <div className="min-w-0 flex-1 pt-0.5">
            <div className="flex flex-wrap items-center gap-2">
              <StatusBadge>{item.categoryLabel}</StatusBadge>
              {item.selected ? (
                <StatusBadge tone="info">Selected</StatusBadge>
              ) : null}
            </div>

            <h3 className="mt-3 text-[1.45rem] font-semibold leading-[1.1] tracking-tight text-[var(--text-primary)] sm:text-[1.6rem]">
              {item.name}
            </h3>

            {titleMeta.length > 0 ? (
              <div className="mt-3 flex flex-wrap gap-1.5">
                {titleMeta.map((meta) => (
                  <span
                    key={`${item.id}-${meta}`}
                    className={cn(
                      VOICE_ROUTE_META_COPY_CLASS,
                      "rounded-full border border-[var(--border-muted)] bg-[var(--bg-surface-0)]/70 px-2.5 py-1 text-[10px] font-semibold",
                    )}
                  >
                    {meta}
                  </span>
                ))}
              </div>
            ) : null}
          </div>
        </div>

        <div>
          <div className={cn(VOICE_ROUTE_SECTION_LABEL_CLASS, "mb-2")}>
            Voice Notes
          </div>
          <p
            className={cn(
              VOICE_ROUTE_BODY_COPY_CLASS,
              "line-clamp-4 text-[15px] leading-7 text-[var(--text-secondary)]",
              !item.description && "text-[var(--text-muted)]",
            )}
          >
            {item.description || "No reference notes were saved for this voice yet."}
          </p>
        </div>

        <div>
          <div className={cn(VOICE_ROUTE_SECTION_LABEL_CLASS, "mb-3")}>
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
            className="mt-auto grid gap-3 [&>*]:w-full [&>*]:justify-center"
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
        "grid grid-cols-1 gap-4 lg:grid-cols-2 2xl:grid-cols-3",
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
