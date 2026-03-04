import type { ReactNode } from "react";
import { cn } from "@/lib/utils";

interface PageShellProps {
  children: ReactNode;
  className?: string;
}

interface PageHeaderProps {
  title: ReactNode;
  description?: ReactNode;
  actions?: ReactNode;
  className?: string;
  titleClassName?: string;
}

export function PageShell({ children, className }: PageShellProps) {
  return (
    <div className={cn("w-full max-w-[1460px] mx-auto", className)}>
      {children}
    </div>
  );
}

export function PageHeader({
  title,
  description,
  actions,
  className,
  titleClassName,
}: PageHeaderProps) {
  return (
    <div
      className={cn(
        "mb-8 flex flex-col gap-4 sm:flex-row sm:items-start sm:justify-between",
        className,
      )}
    >
      <div className="min-w-0 flex-1">
        <h1
          className={cn(
            "app-page-title text-[var(--text-primary)] tracking-tight",
            titleClassName,
          )}
        >
          {title}
        </h1>
        {description ? (
          <p className="mt-1.5 app-page-description text-[var(--text-muted)] max-w-3xl leading-relaxed">
            {description}
          </p>
        ) : null}
      </div>
      {actions ? (
        <div className="flex shrink-0 items-center gap-3 pt-1">{actions}</div>
      ) : null}
    </div>
  );
}
