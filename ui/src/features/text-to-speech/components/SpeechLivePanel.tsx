import { Headphones, Loader2, Play, Square } from "lucide-react";

import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { cn } from "@/lib/utils";

export interface SpeechLiveProgress {
  completedSegments: number;
  totalSegments: number;
}

interface SpeechLivePanelProps {
  /** `generating`: streaming, no audio yet; `playing`: audio is playing as
   * it is generated; `available`: the take is still rendering elsewhere and
   * can be listened to from its start. */
  state: "generating" | "playing" | "available";
  progress?: SpeechLiveProgress | null;
  onStop?: () => void;
  onListen?: () => void;
  className?: string;
}

function progressLabel(progress: SpeechLiveProgress): string {
  const { completedSegments, totalSegments } = progress;
  return `${completedSegments} of ${totalSegments} ${totalSegments === 1 ? "section" : "sections"} generated`;
}

/** A take that is generating now: what is happening, how far it is, and how
 * to stop it or listen. It sits with the page's other status, below the
 * header, so it never covers the header's title and actions. */
export function SpeechLivePanel({
  state,
  progress = null,
  onStop,
  onListen,
  className,
}: SpeechLivePanelProps) {
  const title =
    state === "playing"
      ? "Playing generated speech"
      : state === "generating"
        ? "Waiting for first audio"
        : "Still generating";
  const detail =
    state === "playing"
      ? "Audio plays as it is generated. The full take is saved when it finishes."
      : state === "generating"
        ? "Playback starts as soon as the first audio is ready."
        : "Listen from the beginning while the rest is generated.";
  const fraction =
    progress && progress.totalSegments > 0
      ? Math.min(1, progress.completedSegments / progress.totalSegments)
      : null;

  return (
    <Card
      role="status"
      aria-live="polite"
      className={cn(
        "rounded-[14px] border-(--status-info-border) bg-(--status-info-bg) p-4",
        className,
      )}
    >
      <div className="flex flex-col gap-3 sm:flex-row sm:items-center">
        <div className="flex min-w-0 flex-1 items-start gap-3">
          <span
            aria-hidden="true"
            className="mt-0.5 flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-(--bg-surface-2) text-(--status-info-text)"
          >
            {state === "playing" ? (
              <Headphones className="h-4 w-4" />
            ) : (
              <Loader2 className="h-4 w-4 animate-spin" />
            )}
          </span>
          <div className="min-w-0 flex-1">
            <p className="text-sm font-medium text-(--text-primary)">{title}</p>
            <p className="mt-0.5 text-[13px] text-(--text-muted)">{detail}</p>
          </div>
        </div>
        <div className="flex shrink-0 items-center gap-2 pl-11 sm:pl-0">
          {state === "available" && onListen ? (
            <Button
              type="button"
              variant="outline"
              size="sm"
              className="h-9 gap-2"
              onClick={onListen}
            >
              <Play className="h-4 w-4" />
              Listen from beginning
            </Button>
          ) : null}
          {state !== "available" && onStop ? (
            <Button
              type="button"
              variant="outline"
              size="sm"
              className="h-9 gap-2"
              onClick={onStop}
              aria-label="Stop generation and playback"
            >
              <Square className="h-3.5 w-3.5" />
              Stop
            </Button>
          ) : null}
        </div>
      </div>
      {progress ? (
        <div className="mt-3 pl-11">
          <div
            className="h-1.5 w-full overflow-hidden rounded-full bg-(--bg-surface-3)"
            role="progressbar"
            aria-label="Sections generated"
            aria-valuemin={0}
            aria-valuemax={progress.totalSegments}
            aria-valuenow={progress.completedSegments}
          >
            <div
              className="h-full rounded-full bg-(--status-info-text) transition-[width] duration-500"
              style={{ width: `${(fraction ?? 0) * 100}%` }}
            />
          </div>
          <p className="mt-1.5 text-xs text-(--text-subtle)">{progressLabel(progress)}</p>
        </div>
      ) : null}
    </Card>
  );
}
