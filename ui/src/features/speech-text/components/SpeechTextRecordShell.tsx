import { type ReactNode } from "react";
import { ArrowLeft } from "lucide-react";

import { Button } from "@/components/ui/button";

interface SpeechTextRecordShellProps {
  title: string;
  metadata?: ReactNode;
  actions?: ReactNode;
  alerts?: ReactNode;
  onBack?: () => void;
  backLabel?: string;
  children: ReactNode;
}

export function SpeechTextRecordShell({
  title,
  metadata,
  actions,
  alerts,
  onBack,
  backLabel = "Back",
  children,
}: SpeechTextRecordShellProps) {
  return (
    <div className="space-y-5">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div className="min-w-0">
          {onBack ? (
            <Button
              type="button"
              variant="outline"
              size="sm"
              className="mb-4 h-10 gap-2 rounded-full border-[var(--border-muted)] bg-[var(--bg-surface-0)] px-4 text-sm font-medium text-[var(--text-secondary)] shadow-sm hover:bg-[var(--bg-surface-1)]"
              onClick={onBack}
            >
              <ArrowLeft className="h-4 w-4" />
              {backLabel}
            </Button>
          ) : null}
          <h2 className="truncate text-2xl font-semibold tracking-tight text-[var(--text-primary)]">
            {title}
          </h2>
          {metadata ? (
            <div className="mt-2 flex flex-wrap items-center gap-2 text-xs text-[var(--text-muted)]">
              {metadata}
            </div>
          ) : null}
        </div>

        {actions ? (
          <div className="flex flex-wrap items-center justify-end gap-2">{actions}</div>
        ) : null}
      </div>

      {alerts}

      {children}
    </div>
  );
}
