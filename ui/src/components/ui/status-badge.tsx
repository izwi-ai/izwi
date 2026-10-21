import type { HTMLAttributes } from "react";

import { cn } from "@/lib/utils";

type StatusTone = "neutral" | "info" | "success" | "warning" | "danger";
type StatusEmphasis = "subtle" | "solid";

const subtleToneClasses: Record<StatusTone, string> = {
  neutral:
    "border-border/80 bg-muted/60 text-muted-foreground",
  info:
    "border-(--status-info-border) bg-(--status-info-bg) text-(--status-info-text)",
  success:
    "border-(--status-positive-border) bg-(--status-positive-bg) text-(--status-positive-text)",
  warning:
    "border-(--status-warning-border) bg-(--status-warning-bg) text-(--status-warning-text)",
  danger:
    "border-(--danger-border) bg-(--danger-bg) text-(--danger-text)",
};

const solidToneClasses: Record<StatusTone, string> = {
  neutral:
    "border-border bg-foreground text-background",
  info:
    "border-(--status-info-text) bg-(--status-info-text) text-(--bg-surface-0)",
  success:
    "border-(--status-positive-solid) bg-(--status-positive-solid) text-(--bg-surface-0)",
  warning:
    "border-(--status-warning-text) bg-(--status-warning-text) text-(--bg-surface-0)",
  danger:
    "border-(--danger-text) bg-(--danger-text) text-(--bg-surface-0)",
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
