import type { ReactNode } from "react";
import { Loader2, type LucideIcon } from "lucide-react";

import { cn } from "@/lib/utils";

const focusRing =
  "focus-visible:outline-hidden focus-visible:ring-2 focus-visible:ring-(--border-strong) focus-visible:ring-offset-2 focus-visible:ring-offset-(--bg-surface-0)";

/** Pill buttons for a PageState's actions; also fit a react-router Link. */
export const pageStateButtonClass = {
  primary: cn(
    "inline-flex h-10 items-center justify-center rounded-full bg-(--accent-solid) px-5 text-sm font-medium text-(--text-on-accent) transition-opacity hover:opacity-90 disabled:opacity-50",
    focusRing,
  ),
  secondary: cn(
    "inline-flex h-10 items-center justify-center rounded-full border border-(--border-default) px-4 text-sm text-(--text-primary) transition-colors hover:bg-(--bg-surface-2) disabled:opacity-50",
    focusRing,
  ),
};

interface PageStateProps {
  title: ReactNode;
  description?: ReactNode;
  icon?: LucideIcon;
  tone?: "neutral" | "danger";
  align?: "center" | "start";
  actions?: ReactNode;
  /** Extra content under the actions, e.g. error details. */
  children?: ReactNode;
  role?: "alert" | "status";
  className?: string;
}

/**
 * A whole page that is a message (page not found, page error): a title, one
 * line of explanation and what to do next, centred in the page. StatePanel is
 * the boxed version for a state inside a page.
 */
export function PageState({
  title,
  description,
  icon: Icon,
  tone = "neutral",
  align = "start",
  actions,
  children,
  role,
  className,
}: PageStateProps) {
  const centered = align === "center";
  return (
    <div
      role={role}
      className={cn(
        "flex flex-1 items-center justify-center py-12",
        className,
      )}
    >
      <div
        className={cn(
          "flex w-full max-w-[460px] flex-col gap-6",
          centered && "items-center text-center",
        )}
      >
        <div className={cn("flex flex-col gap-2", centered && "items-center")}>
          {Icon ? (
            <span
              className={cn(
                "mb-2 flex h-9 w-9 items-center justify-center rounded-full",
                tone === "danger"
                  ? "bg-(--danger-bg) text-(--danger-text)"
                  : "bg-(--accent-soft) text-(--text-primary)",
              )}
            >
              <Icon aria-hidden="true" className="h-[18px] w-[18px]" />
            </span>
          ) : null}
          <h1 className="text-2xl font-medium tracking-[-0.02em] text-(--text-primary)">
            {title}
          </h1>
          {description ? (
            <p className="text-sm leading-relaxed text-(--text-muted)">
              {description}
            </p>
          ) : null}
        </div>
        {actions ? (
          <div
            className={cn(
              "flex flex-wrap items-center gap-2.5",
              centered && "justify-center",
            )}
          >
            {actions}
          </div>
        ) : null}
        {children}
      </div>
    </div>
  );
}

/** Stand-in for a page's content while it loads. */
export function PageLoading({
  label = "Loading…",
  className,
}: {
  label?: string;
  className?: string;
}) {
  return (
    <div
      role="status"
      className={cn(
        "flex h-[420px] max-h-[70dvh] min-h-48 items-center justify-center rounded-[14px] border border-(--border-subtle) bg-(--surface-panel)",
        className,
      )}
    >
      <span className="flex items-center gap-2 text-[13px] text-(--text-muted)">
        <Loader2 aria-hidden="true" className="h-4 w-4 animate-spin" />
        {label}
      </span>
    </div>
  );
}
