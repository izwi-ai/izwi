import type { ReactNode } from "react";
import { Check } from "lucide-react";

import type { ModelInfo } from "@/api";
import type { ModelDownloadProgressMap } from "@/features/models/downloadProgress";
import { cn } from "@/lib/utils";

export type SetupRowTone = "done" | "active" | "muted" | "error";

/** A download's whole percent: the live stream, else the last listed value. */
export function downloadPercent(
  model: ModelInfo,
  progress: ModelDownloadProgressMap,
): number {
  return Math.round(
    progress[model.variant]?.percent ?? model.download_progress ?? 0,
  );
}

/**
 * One model being set up: what it's for, which model, and how far along it
 * is. Voice's setup and first run list these in a `<ul>` with a top border.
 */
export function SetupRow({
  title,
  model,
  status,
  tone,
  percent,
  action,
  detail,
}: {
  title: string;
  model: string;
  status: string;
  tone: SetupRowTone;
  /** Shows a progress bar under the row. */
  percent?: number;
  /** Next to the status, e.g. Retry. */
  action?: ReactNode;
  /** A line under the row, e.g. why a download failed. */
  detail?: ReactNode;
}) {
  const statusLabel = (
    <span
      className={cn(
        "flex items-center gap-1.5 whitespace-nowrap text-[13px]",
        tone === "done" && "text-(--status-positive-text)",
        tone === "active" && "text-(--text-primary)",
        tone === "muted" && "text-(--text-muted)",
        tone === "error" && "text-(--danger-text)",
      )}
    >
      {tone === "done" && (
        <Check className="h-[15px] w-[15px]" strokeWidth={2.5} />
      )}
      {status}
    </span>
  );

  return (
    <li
      className={cn(
        "flex flex-col border-b border-(--border-muted) py-3.5",
        percent !== undefined ? "gap-2.5" : "gap-1.5",
      )}
    >
      <div className="grid grid-cols-[minmax(0,1fr)_auto] items-center gap-4">
        <div className="flex min-w-0 flex-col gap-0.5">
          <span className="text-sm font-medium text-(--text-primary)">
            {title}
          </span>
          <span className="truncate text-[13px] text-(--text-muted)">
            {model}
          </span>
        </div>
        {action ? (
          <span className="flex items-center gap-3 whitespace-nowrap">
            {statusLabel}
            {action}
          </span>
        ) : (
          statusLabel
        )}
      </div>
      {percent !== undefined && (
        <div
          role="progressbar"
          aria-label={`${title} download`}
          aria-valuenow={percent}
          aria-valuemin={0}
          aria-valuemax={100}
          className="h-1 overflow-hidden rounded-full bg-(--bg-surface-3)"
        >
          <div
            className="h-full rounded-full bg-(--accent-solid) transition-[width] duration-300"
            style={{ width: `${percent}%` }}
          />
        </div>
      )}
      {detail ? (
        <p className="text-[12.5px] text-(--text-subtle)">{detail}</p>
      ) : null}
    </li>
  );
}
