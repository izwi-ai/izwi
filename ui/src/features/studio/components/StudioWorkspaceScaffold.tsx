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
  return (
    <div className="space-y-6 pb-8">
      <section data-testid="studio-library-pane" className="space-y-2">
        {overview}
      </section>

      <div
        className={cn(
          "grid gap-6",
          statsRail
            ? "xl:grid-cols-[220px_minmax(0,1fr)_340px] 2xl:grid-cols-[240px_minmax(0,1fr)_360px]"
            : "xl:grid-cols-[minmax(0,1fr)_340px] 2xl:grid-cols-[minmax(0,1fr)_360px]",
        )}
      >
        {statsRail ? (
          <aside
            data-testid="studio-stats-pane"
            className="order-2 xl:order-1 xl:sticky xl:top-4 xl:self-start"
          >
            {statsRail}
          </aside>
        ) : null}

        <section
          data-testid="studio-editor-pane"
          className={cn(
            "min-w-0",
            statsRail ? "order-1 xl:order-2" : "order-1",
          )}
        >
          {editor}
        </section>

        {actionRail ? (
          <aside
            data-testid="studio-delivery-pane"
            className={cn(
              "space-y-5 xl:sticky xl:top-4 xl:self-start",
              statsRail ? "order-3 xl:order-3" : "order-2 xl:order-2",
            )}
          >
            {actionRail}
          </aside>
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
