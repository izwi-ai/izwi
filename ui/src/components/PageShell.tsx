import type { ReactNode } from "react";
import { Link } from "react-router";
import { ChevronRight } from "lucide-react";
import { cn } from "@/lib/utils";

interface PageShellProps {
  children: ReactNode;
  className?: string;
}

export interface Breadcrumb {
  label: ReactNode;
  /** Omit for the current page (the last crumb). */
  to?: string;
}

interface PageHeaderProps {
  title: ReactNode;
  /** Trail above the title on nested pages, ending with the current page. */
  breadcrumbs?: Breadcrumb[];
  description?: ReactNode;
  actions?: ReactNode;
  className?: string;
  titleClassName?: string;
}

export function PageShell({ children, className }: PageShellProps) {
  return (
    <section
      className={cn(
        "flow-root w-full max-w-[1460px] mx-auto flex-1",
        className,
      )}
    >
      {children}
    </section>
  );
}

export function PageHeader({
  title,
  breadcrumbs,
  description,
  actions,
  className,
  titleClassName,
}: PageHeaderProps) {
  return (
    <header
      className={cn(
        "mb-8 flex flex-col gap-4 sm:flex-row sm:items-start sm:justify-between",
        className,
      )}
    >
      <div className="min-w-0 flex-1">
        {breadcrumbs && breadcrumbs.length > 0 ? (
          <nav aria-label="Breadcrumb" className="mb-2.5">
            <ol className="flex min-w-0 items-center gap-1.5 text-[13px] text-(--text-subtle)">
              {breadcrumbs.map((crumb, index) => (
                <li
                  key={index}
                  className="flex min-w-0 items-center gap-1.5"
                >
                  {index > 0 ? (
                    <ChevronRight
                      aria-hidden="true"
                      className="h-3.5 w-3.5 shrink-0"
                    />
                  ) : null}
                  {crumb.to ? (
                    <Link
                      to={crumb.to}
                      className="shrink-0 text-(--text-muted) transition-colors hover:text-(--text-primary)"
                    >
                      {crumb.label}
                    </Link>
                  ) : (
                    <span aria-current="page" className="truncate">
                      {crumb.label}
                    </span>
                  )}
                </li>
              ))}
            </ol>
          </nav>
        ) : null}
        <h1
          className={cn(
            "app-page-title truncate text-(--text-primary) tracking-tight",
            titleClassName,
          )}
        >
          {title}
        </h1>
        {description ? (
          <p className="mt-1.5 app-page-description text-(--text-muted) max-w-3xl leading-relaxed">
            {description}
          </p>
        ) : null}
      </div>
      {actions ? (
        <div className="flex shrink-0 items-center gap-3 pt-1">{actions}</div>
      ) : null}
    </header>
  );
}
