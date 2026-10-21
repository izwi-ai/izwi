import type { ReactNode } from "react";

import { cn } from "@/lib/utils";

export type RowBadgeTone = "neutral" | "info" | "positive" | "warning" | "danger";

const toneClasses: Record<RowBadgeTone, string> = {
  neutral: "border-(--border-default) text-(--text-muted)",
  info: "border-(--status-info-border) bg-(--status-info-bg) text-(--status-info-text)",
  positive:
    "border-(--status-positive-border) bg-(--status-positive-bg) text-(--status-positive-text)",
  warning:
    "border-(--status-warning-border) bg-(--status-warning-bg) text-(--status-warning-text)",
  danger: "border-(--danger-border) bg-(--danger-bg) text-(--danger-text)",
};

/** Compact sentence-case badge for table cells ("3 speakers", "Failed"). */
export function RowBadge({
  tone = "neutral",
  dot = false,
  icon,
  children,
  className,
}: {
  tone?: RowBadgeTone;
  dot?: boolean;
  icon?: ReactNode;
  children: ReactNode;
  className?: string;
}) {
  return (
    <span
      className={cn(
        "inline-flex h-[22px] items-center gap-1.5 whitespace-nowrap rounded-full border px-2 text-[11.5px] font-medium",
        toneClasses[tone],
        className,
      )}
    >
      {dot ? (
        <span aria-hidden="true" className="h-1.5 w-1.5 rounded-full bg-current" />
      ) : null}
      {icon}
      {children}
    </span>
  );
}
