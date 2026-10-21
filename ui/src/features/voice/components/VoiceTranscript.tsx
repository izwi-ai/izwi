import { useEffect, useRef } from "react";
import { X } from "lucide-react";

import type { TranscriptEntry } from "@/features/voice/realtime/support";

/** The conversation so far, as plain text, beside the stage. */
export function VoiceTranscript({
  entries,
  isAssistantLive,
  onClose,
}: {
  entries: TranscriptEntry[];
  /** The last line is the reply still arriving. */
  isAssistantLive: boolean;
  onClose: () => void;
}) {
  const endRef = useRef<HTMLLIElement | null>(null);
  useEffect(() => {
    endRef.current?.scrollIntoView({ block: "end" });
  }, [entries]);

  return (
    <aside
      aria-label="Transcript"
      className="flex min-h-0 w-full flex-col border-t border-(--border-muted) bg-(--bg-surface-1) lg:w-[340px] lg:shrink-0 lg:border-l lg:border-t-0"
    >
      <div className="flex h-16 shrink-0 items-center justify-between border-b border-(--border-muted) pl-5 pr-3">
        <h2 className="text-sm font-medium text-(--text-primary)">
          Transcript
        </h2>
        <button
          type="button"
          onClick={onClose}
          aria-label="Close transcript"
          className="flex h-10 w-10 items-center justify-center rounded-[10px] text-(--text-muted) transition-colors hover:bg-(--bg-surface-2) hover:text-(--text-primary)"
        >
          <X className="h-4 w-4" />
        </button>
      </div>
      <ol
        aria-live="polite"
        className="flex min-h-0 flex-1 flex-col gap-5 overflow-y-auto p-5"
      >
        {entries.length === 0 ? (
          <li className="text-sm text-(--text-subtle)">
            What you and Izwi say appears here.
          </li>
        ) : (
          entries.map((entry, index) => {
            const isLive =
              isAssistantLive &&
              entry.role === "assistant" &&
              index === entries.length - 1;
            return (
              <li key={entry.id} className="flex flex-col gap-1">
                <span className="text-xs text-(--text-subtle)">
                  {entry.role === "user" ? "You" : "Izwi"}
                </span>
                <p
                  className={
                    entry.role === "user"
                      ? "whitespace-pre-wrap text-sm leading-relaxed text-(--text-muted)"
                      : "whitespace-pre-wrap text-sm leading-relaxed text-(--text-primary)"
                  }
                >
                  {entry.text}
                  {isLive && (
                    <span
                      aria-hidden="true"
                      className="ml-0.5 inline-block h-[1em] w-0.5 translate-y-0.5 bg-(--text-primary)"
                    />
                  )}
                </p>
              </li>
            );
          })
        )}
        <li ref={endRef} aria-hidden="true" />
      </ol>
    </aside>
  );
}
