import type { ReactNode } from "react";
import { Card } from "@/components/ui/card";

interface StudioActionRailProps {
  workflowSummary: string;
  primaryActions: ReactNode;
  exportSettings: ReactNode;
  queuePanel: ReactNode;
}

export function StudioActionRail({
  workflowSummary,
  primaryActions,
  exportSettings,
  queuePanel,
}: StudioActionRailProps) {
  return (
    <div className="space-y-4 xl:space-y-5">
      <Card className="rounded-2xl border-[var(--border-muted)] bg-[var(--bg-surface-0)] p-4 shadow-none sm:p-5">
        <div className="text-[11px] font-semibold uppercase tracking-wider text-[var(--text-muted)]">
          Render Flow
        </div>
        <div className="mt-2 text-sm font-semibold text-[var(--text-primary)]">
          {workflowSummary}
        </div>
        <div className="mt-2 text-sm leading-relaxed text-[var(--text-secondary)]">
          Run renders as edits settle, then export the merged output once the
          project is current.
        </div>
        <div className="mt-4 grid gap-2">{primaryActions}</div>
      </Card>

      <Card className="rounded-2xl border-[var(--border-muted)] bg-[var(--bg-surface-0)] p-4 shadow-none sm:p-5">
        <div className="text-[11px] font-semibold uppercase tracking-wider text-[var(--text-muted)]">
          Export Options
        </div>
        <div className="mt-4">{exportSettings}</div>
      </Card>

      {queuePanel}
    </div>
  );
}
