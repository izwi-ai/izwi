import type { ReactNode } from "react";
import { cn } from "@/lib/utils";

interface StudioWorkspaceScaffoldProps {
  overview: ReactNode;
  statsRail?: ReactNode;
  editor: ReactNode;
  actionRail?: ReactNode;
  utilities?: ReactNode;
}

export function StudioWorkspaceScaffold({
  overview,
  statsRail,
  editor,
  actionRail,
  utilities,
}: StudioWorkspaceScaffoldProps) {
  const hasBothRails = Boolean(statsRail && actionRail);

  return (
    <div className="space-y-6 pb-8">
      <section data-testid="studio-library-pane" className="space-y-2">
        {overview}
      </section>

      <div
        className={cn(
          "grid gap-6",
          statsRail || actionRail
            ? "xl:grid-cols-[minmax(0,1fr)_minmax(18rem,20rem)]"
            : "grid-cols-1",
          hasBothRails &&
            "min-[1800px]:grid-cols-[240px_minmax(0,1fr)_360px]",
        )}
      >
        <section
          data-testid="studio-editor-pane"
          className={cn(
            "order-1 min-w-0",
            (statsRail || actionRail) && "xl:col-start-1 xl:row-start-1",
            hasBothRails && "min-[1800px]:col-start-2",
          )}
        >
          {editor}
        </section>

        {statsRail || actionRail ? (
          <div
            data-testid="studio-secondary-pane"
            role="complementary"
            aria-label="Project status and configuration"
            className={cn(
              "order-2 space-y-5 xl:col-start-2 xl:row-start-1 xl:sticky xl:top-4 xl:max-h-[calc(100dvh-2rem)] xl:self-start xl:overflow-y-auto xl:overscroll-contain xl:pr-1",
              hasBothRails &&
                "min-[1800px]:contents min-[1800px]:max-h-none min-[1800px]:space-y-0 min-[1800px]:overflow-visible min-[1800px]:pr-0",
            )}
          >
            {statsRail ? (
              <section
                data-testid="studio-stats-pane"
                aria-label="Project status"
                className={cn(
                  "rounded-2xl border border-[var(--border-muted)] bg-[var(--bg-surface-0)] p-4 shadow-sm",
                  hasBothRails &&
                    "min-[1800px]:sticky min-[1800px]:top-4 min-[1800px]:col-start-1 min-[1800px]:row-start-1 min-[1800px]:self-start min-[1800px]:rounded-none min-[1800px]:border-0 min-[1800px]:bg-transparent min-[1800px]:p-0 min-[1800px]:shadow-none",
                )}
              >
                {statsRail}
              </section>
            ) : null}

            {actionRail ? (
              <section
                data-testid="studio-delivery-pane"
                aria-label="Project configuration"
                className={cn(
                  "space-y-5 rounded-2xl border border-[var(--border-muted)] bg-[var(--bg-surface-0)] p-4 shadow-sm",
                  hasBothRails &&
                    "min-[1800px]:sticky min-[1800px]:top-4 min-[1800px]:col-start-3 min-[1800px]:row-start-1 min-[1800px]:self-start min-[1800px]:rounded-none min-[1800px]:border-0 min-[1800px]:bg-transparent min-[1800px]:p-0 min-[1800px]:shadow-none",
                )}
              >
                {actionRail}
              </section>
            ) : null}
          </div>
        ) : null}
      </div>

      {utilities ? (
        <section data-testid="studio-utilities-pane" className="space-y-5">
          {utilities}
        </section>
      ) : null}
    </div>
  );
}
