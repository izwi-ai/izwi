import type { ReactNode } from "react";

interface StudioWorkspaceScaffoldProps {
  library: ReactNode;
  editor: ReactNode;
  delivery: ReactNode;
}

export function StudioWorkspaceScaffold({
  library,
  editor,
  delivery,
}: StudioWorkspaceScaffoldProps) {
  return (
    <div className="space-y-5">
      <section
        data-testid="studio-library-pane"
        className="rounded-2xl border border-[var(--border-muted)] bg-[var(--bg-surface-0)] p-5 sm:p-6"
      >
        {library}
      </section>

      <div className="grid gap-5 xl:grid-cols-[minmax(0,1fr)_360px]">
        <section
          data-testid="studio-editor-pane"
          className="rounded-2xl border border-[var(--border-muted)] bg-[var(--bg-surface-0)] p-5 sm:p-6"
        >
          {editor}
        </section>

        <aside
          data-testid="studio-delivery-pane"
          className="space-y-5"
        >
          {delivery}
        </aside>
      </div>
    </div>
  );
}
