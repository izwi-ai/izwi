import type { HTMLAttributes, ReactNode } from "react";
import type { LucideIcon } from "lucide-react";

import { cn } from "@/lib/utils";

export function WorkspaceFrame({
  className,
  ...props
}: HTMLAttributes<HTMLDivElement>) {
  return <div className={cn("app-workspace-frame p-4 sm:p-5", className)} {...props} />;
}

export function WorkspacePanel({
  className,
  ...props
}: HTMLAttributes<HTMLDivElement>) {
  return <div className={cn("app-surface p-4 sm:p-5", className)} {...props} />;
}

interface WorkspaceHeaderProps
  extends Omit<HTMLAttributes<HTMLDivElement>, "title"> {
  icon?: LucideIcon;
  title: ReactNode;
  description?: ReactNode;
  actions?: ReactNode;
}

export function WorkspaceHeader({
  icon: Icon,
  title,
  description,
  actions,
  className,
  ...props
}: WorkspaceHeaderProps) {
  return (
    <div
      className={cn(
        "flex flex-col gap-4 border-b border-border/70 pb-4 sm:flex-row sm:items-start sm:justify-between",
        className,
      )}
      {...props}
    >
      <div className="min-w-0 flex-1">
        <div className="flex items-start gap-3">
          {Icon ? (
            <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-[var(--radius-md)] border border-border/70 bg-background/55 shadow-[var(--shadow-soft)]">
              <Icon className="h-4.5 w-4.5 text-muted-foreground" />
            </div>
          ) : null}
          <div className="min-w-0 flex-1">
            <h2 className="text-base font-semibold tracking-tight text-[var(--text-primary)]">
              {title}
            </h2>
            {description ? (
              <p className="mt-1 text-sm leading-6 text-[var(--text-secondary)]">
                {description}
              </p>
            ) : null}
          </div>
        </div>
      </div>
      {actions ? <div className="flex shrink-0 items-center gap-3">{actions}</div> : null}
    </div>
  );
}

export function WorkspaceSectionLabel({
  className,
  ...props
}: HTMLAttributes<HTMLDivElement>) {
  return <div className={cn("app-section-label", className)} {...props} />;
}
