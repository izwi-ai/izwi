import type { ReactNode } from "react";

import { cn } from "@/lib/utils";

export type RowBadgeTone = "neutral" | "info" | "positive" | "warning" | "danger";

const toneClasses: Record<RowBadgeTone, string> = {
  neutral: "border-[var(--border-default)] text-[var(--text-muted)]",
  info: "border-[var(--status-info-border)] bg-[var(--status-info-bg)] text-[var(--status-info-text)]",
  positive:
    "border-[var(--status-positive-border)] bg-[var(--status-positive-bg)] text-[var(--status-positive-text)]",
  warning:
    "border-[var(--status-warning-border)] bg-[var(--status-warning-bg)] text-[var(--status-warning-text)]",
  danger: "border-[var(--danger-border)] bg-[var(--danger-bg)] text-[var(--danger-text)]",
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
