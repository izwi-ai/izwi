import type { ReactNode } from "react";
import { Card } from "@/components/ui/card";

interface StudioWorkspaceScaffoldProps {
  overview: ReactNode;
  editor: ReactNode;
  actionRail?: ReactNode;
  utilities?: ReactNode;
}

export function StudioWorkspaceScaffold({
  overview,
  editor,
  actionRail,
  utilities,
}: StudioWorkspaceScaffoldProps) {
  return (
    <div className="space-y-5">
      <Card
        data-testid="studio-library-pane"
        className="rounded-2xl border-[var(--border-muted)] bg-[var(--bg-surface-0)] p-5 shadow-none sm:p-6"
      >
        {overview}
      </Card>

      <div className="grid gap-5 xl:grid-cols-[minmax(0,1fr)_340px] 2xl:grid-cols-[minmax(0,1fr)_360px]">
        <Card
          data-testid="studio-editor-pane"
          className="order-2 rounded-2xl border-[var(--border-muted)] bg-[var(--bg-surface-0)] p-5 shadow-none sm:p-6 xl:order-1"
        >
          {editor}
        </Card>

        {actionRail ? (
          <aside
            data-testid="studio-delivery-pane"
            className="order-1 space-y-5 xl:order-2 xl:sticky xl:top-4 xl:self-start"
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
