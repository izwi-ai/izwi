import type { ReactNode } from "react";
import type { LucideIcon } from "lucide-react";

import { cn } from "@/lib/utils";
import { StatusBadge } from "@/components/ui/status-badge";

type StatePanelTone = "neutral" | "info" | "success" | "warning" | "danger";

const toneClasses: Record<StatePanelTone, string> = {
  neutral: "border-border/80 bg-card/80 text-card-foreground",
  info:
    "border-[var(--status-info-border)] bg-[var(--status-info-bg)]/70 text-[var(--status-info-text)]",
  success:
    "border-[var(--status-positive-border)] bg-[var(--status-positive-bg)]/70 text-[var(--status-positive-text)]",
  warning:
    "border-[var(--status-warning-border)] bg-[var(--status-warning-bg)]/70 text-[var(--status-warning-text)]",
  danger:
    "border-[var(--danger-border)] bg-[var(--danger-bg)]/75 text-[var(--danger-text)]",
};

interface StatePanelProps {
  title: ReactNode;
  description?: ReactNode;
  icon?: LucideIcon;
  tone?: StatePanelTone;
  dashed?: boolean;
  align?: "start" | "center";
  eyebrow?: string;
  actions?: ReactNode;
  className?: string;
}

export function StatePanel({
  title,
  description,
  icon: Icon,
  tone = "neutral",
  dashed = false,
  align = "start",
  eyebrow,
  actions,
  className,
}: StatePanelProps) {
  return (
    <div
      className={cn(
        "app-state-panel flex gap-4 p-5",
        align === "center" && "flex-col items-center text-center",
        dashed && "app-state-panel-dashed",
        toneClasses[tone],
        className,
      )}
    >
      {Icon ? (
        <div
          className={cn(
            "flex h-12 w-12 shrink-0 items-center justify-center rounded-[var(--radius-lg)] border border-border/70 bg-background/55",
            align === "center" && "mx-auto",
          )}
        >
          <Icon className="h-5 w-5" />
        </div>
      ) : null}

      <div className={cn("min-w-0 flex-1", align === "center" && "max-w-md")}>
        {eyebrow ? (
          <StatusBadge
            tone={tone === "neutral" ? "neutral" : tone}
            className={cn("mb-3", align === "center" && "mx-auto")}
          >
            {eyebrow}
          </StatusBadge>
        ) : null}
        <div className="text-sm font-semibold tracking-tight text-[var(--text-primary)]">
          {title}
        </div>
        {description ? (
          <div className="mt-1.5 text-sm leading-6 text-[var(--text-secondary)]">
            {description}
          </div>
        ) : null}
        {actions ? (
          <div
            className={cn(
              "mt-4 flex flex-wrap gap-2",
              align === "center" && "justify-center",
            )}
          >
            {actions}
          </div>
        ) : null}
      </div>
    </div>
  );
}
