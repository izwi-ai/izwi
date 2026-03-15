import type { HTMLAttributes } from "react";

import { cn } from "@/lib/utils";

type StatusTone = "neutral" | "info" | "success" | "warning" | "danger";
type StatusEmphasis = "subtle" | "solid";

const subtleToneClasses: Record<StatusTone, string> = {
  neutral:
    "border-border/80 bg-muted/60 text-muted-foreground",
  info:
    "border-[var(--status-info-border)] bg-[var(--status-info-bg)] text-[var(--status-info-text)]",
  success:
    "border-[var(--status-positive-border)] bg-[var(--status-positive-bg)] text-[var(--status-positive-text)]",
  warning:
    "border-[var(--status-warning-border)] bg-[var(--status-warning-bg)] text-[var(--status-warning-text)]",
  danger:
    "border-[var(--danger-border)] bg-[var(--danger-bg)] text-[var(--danger-text)]",
};

const solidToneClasses: Record<StatusTone, string> = {
  neutral:
    "border-border bg-foreground text-background",
  info:
    "border-[var(--status-info-text)] bg-[var(--status-info-text)] text-[var(--bg-surface-0)]",
  success:
    "border-[var(--status-positive-solid)] bg-[var(--status-positive-solid)] text-[var(--bg-surface-0)]",
  warning:
    "border-[var(--status-warning-text)] bg-[var(--status-warning-text)] text-[var(--bg-surface-0)]",
  danger:
    "border-[var(--danger-text)] bg-[var(--danger-text)] text-[var(--bg-surface-0)]",
};

interface StatusBadgeProps extends HTMLAttributes<HTMLSpanElement> {
  tone?: StatusTone;
  emphasis?: StatusEmphasis;
}

export function StatusBadge({
  className,
  tone = "neutral",
  emphasis = "subtle",
  ...props
}: StatusBadgeProps) {
  return (
    <span
      className={cn(
        "inline-flex items-center rounded-full border px-2.5 py-1 text-[10px] font-semibold uppercase tracking-[0.18em]",
        emphasis === "solid"
          ? solidToneClasses[tone]
          : subtleToneClasses[tone],
        className,
      )}
      {...props}
    />
  );
}
