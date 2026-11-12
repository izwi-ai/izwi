import type { ReactNode } from "react";

import { cn } from "@/lib/utils";

/** A titled group of settings rows in one bordered panel. */
export function SettingsSection({
  id,
  title,
  description,
  children,
}: {
  id: string;
  title: string;
  description?: ReactNode;
  children: ReactNode;
}) {
  const headingId = `${id}-heading`;
  return (
    <section id={id} aria-labelledby={headingId} className="scroll-mt-6">
      <h2
        id={headingId}
        className="text-[15px] font-semibold text-(--text-primary)"
      >
        {title}
      </h2>
      {description ? (
        <p className="mt-0.5 text-[13px] text-(--text-muted)">
          {description}
        </p>
      ) : null}
      <div className="mt-3 rounded-[14px] border border-(--border-subtle) bg-(--surface-panel) px-4">
        {children}
      </div>
    </section>
  );
}

/** One setting: its name and explanation, a control on the right, and
 * optional content underneath. */
export function SettingsRow({
  title,
  description,
  action,
  children,
  className,
}: {
  title?: ReactNode;
  description?: ReactNode;
  action?: ReactNode;
  children?: ReactNode;
  className?: string;
}) {
  const hasHeaderCopy = Boolean(title || description);

  return (
    <div
      className={cn(
        "border-b border-(--border-subtle) py-3.5 last:border-b-0",
        className,
      )}
    >
      {hasHeaderCopy || action ? (
        <div
          className={cn(
            "flex flex-col md:flex-row md:items-center md:justify-between",
            hasHeaderCopy && action ? "gap-3" : "gap-0",
          )}
        >
          {hasHeaderCopy ? (
            <div className="min-w-0 max-w-2xl">
              {title ? (
                <div className="text-[13.5px] font-medium text-(--text-primary)">
                  {title}
                </div>
              ) : null}
              {description ? (
                <div
                  className={cn(
                    "text-[12.5px] leading-5 text-(--text-subtle)",
                    title ? "mt-0.5" : undefined,
                  )}
                >
                  {description}
                </div>
              ) : null}
            </div>
          ) : null}
          {action ? (
            // ml-auto keeps action-only rows right-aligned; justify-between
            // has no effect with a single child.
            <div className="shrink-0 md:ml-auto">{action}</div>
          ) : null}
        </div>
      ) : null}
      {children ? (
        <div
          className={cn(
            hasHeaderCopy ? "mt-3" : action ? "mt-1.5" : "mt-0",
          )}
        >
          {children}
        </div>
      ) : null}
    </div>
  );
}
