import { useEffect, useRef, useState } from "react";
import { ChevronRight, Lightbulb } from "lucide-react";

import { cn } from "@/lib/utils";

/** "6s", "1m 12s": how long a reply spent reasoning. */
export function formatReasoningDuration(ms: number): string {
  const seconds = Math.max(1, Math.round(ms / 1000));
  if (seconds < 60) {
    return `${seconds}s`;
  }
  const minutes = Math.floor(seconds / 60);
  const rest = seconds % 60;
  return rest > 0 ? `${minutes}m ${rest}s` : `${minutes}m`;
}

function useElapsedSeconds(startedAt: number | null, running: boolean) {
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    if (!running || startedAt === null) {
      return;
    }
    setNow(Date.now());
    const timer = window.setInterval(() => setNow(Date.now()), 1000);
    return () => window.clearInterval(timer);
  }, [running, startedAt]);
  return startedAt === null
    ? null
    : Math.max(0, Math.floor((now - startedAt) / 1000));
}

interface ReasoningBlockProps {
  thinking: string;
  /** The model is reasoning now: show the live label and tail. */
  active: boolean;
  /** When the turn started, for the live timer and the final duration. */
  startedAt: number | null;
  /** Measured reasoning time; unknown for replies from an earlier session. */
  durationMs: number | null;
  expanded: boolean;
  onToggle: () => void;
  /** Called once when reasoning ends, with how long it took. */
  onFinished?: (durationMs: number) => void;
}

/**
 * One quiet row for a reply's reasoning. While the model reasons it reads
 * "Thinking · 4s" over a faded tail of the latest lines; afterwards it folds
 * to "Thought for 6s" and opens inline on click.
 */
export function ReasoningBlock({
  thinking,
  active,
  startedAt,
  durationMs,
  expanded,
  onToggle,
  onFinished,
}: ReasoningBlockProps) {
  const elapsed = useElapsedSeconds(startedAt, active);
  const activeStartRef = useRef<number | null>(null);
  const onFinishedRef = useRef(onFinished);
  useEffect(() => {
    onFinishedRef.current = onFinished;
  });

  useEffect(() => {
    if (active) {
      activeStartRef.current = startedAt;
      return;
    }
    const start = activeStartRef.current;
    activeStartRef.current = null;
    if (start !== null) {
      onFinishedRef.current?.(Date.now() - start);
    }
  }, [active, startedAt]);

  const label = active
    ? `Thinking${elapsed !== null ? ` · ${elapsed}s` : ""}`
    : durationMs !== null
      ? `Thought for ${formatReasoningDuration(durationMs)}`
      : "Thought process";

  return (
    <div className="flex flex-col gap-2">
      <button
        type="button"
        onClick={onToggle}
        aria-expanded={expanded}
        className="-ml-0.5 inline-flex h-[26px] items-center gap-1.5 self-start rounded-md pr-2 text-[13px] leading-none text-(--text-subtle) transition-colors hover:text-(--text-primary) focus-visible:outline-hidden focus-visible:ring-2 focus-visible:ring-(--border-strong)"
      >
        <Lightbulb className="h-3.5 w-3.5" aria-hidden="true" />
        <span className={cn(active && "chat-shimmer")}>{label}</span>
        <ChevronRight
          aria-hidden="true"
          className={cn(
            "h-3.5 w-3.5 transition-transform",
            expanded && "rotate-90",
          )}
        />
      </button>
      {expanded ? (
        <div className="whitespace-pre-wrap border-l-2 border-(--border-default) pl-3.5 text-[13.5px] leading-[1.65] text-(--text-muted)">
          {thinking}
        </div>
      ) : active && thinking ? (
        <div
          aria-hidden="true"
          className="chat-reasoning-tail flex max-h-[4.8em] flex-col justify-end overflow-hidden border-l-2 border-(--border-default) pl-3.5 text-[13px] leading-[1.6] text-(--text-subtle)"
        >
          <div className="whitespace-pre-wrap">{thinking}</div>
        </div>
      ) : null}
    </div>
  );
}
