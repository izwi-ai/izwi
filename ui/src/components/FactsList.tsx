import type { ReactNode } from "react";

import { cn } from "@/lib/utils";

export interface Fact {
  label: string;
  value: ReactNode;
}

/** A panel of label/value rows (a record's overview, generation stats). */
export function FactsList({
  facts,
  label,
  className,
}: {
  facts: Fact[];
  /** Accessible name of the list. */
  label: string;
  className?: string;
}) {
  return (
    <dl
      aria-label={label}
      className={cn(
        "rounded-[14px] border border-(--border-subtle) bg-(--surface-panel) px-4 py-1",
        className,
      )}
    >
      {facts.map((fact) => (
        <div
          key={fact.label}
          className="flex items-baseline justify-between gap-3 border-b border-(--border-subtle) py-2.5 text-[13.5px] last:border-b-0"
        >
          <dt className="shrink-0 whitespace-nowrap text-(--text-muted)">
            {fact.label}
          </dt>
          <dd className="min-w-0 truncate text-right text-(--text-primary)">
            {fact.value}
          </dd>
        </div>
      ))}
    </dl>
  );
}

/** "1.9 s · 9.5× realtime" from processing time and real-time factor. */
export function formatProcessingFact(
  processingTimeMs: number | null | undefined,
  rtf: number | null | undefined,
): string | null {
  if (
    processingTimeMs == null ||
    !Number.isFinite(processingTimeMs) ||
    processingTimeMs <= 0
  ) {
    return null;
  }
  const seconds = processingTimeMs / 1000;
  const time =
    seconds >= 10 ? `${seconds.toFixed(0)} s` : `${seconds.toFixed(1)} s`;
  if (rtf == null || !Number.isFinite(rtf) || rtf <= 0) {
    return time;
  }
  const speed = 1 / rtf;
  return `${time} · ${speed >= 10 ? speed.toFixed(0) : speed.toFixed(1)}× realtime`;
}
