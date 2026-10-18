import { type ReactNode } from "react";

interface SpeechTextRecordShellProps {
  metadata?: ReactNode;
  actions?: ReactNode;
  alerts?: ReactNode;
  children: ReactNode;
}

/**
 * Body of a transcription or diarization record. The route's PageHeader
 * carries the title and the breadcrumb back to the list.
 */
export function SpeechTextRecordShell({
  metadata,
  actions,
  alerts,
  children,
}: SpeechTextRecordShellProps) {
  return (
    <div className="space-y-5">
      {metadata || actions ? (
        <div className="flex flex-wrap items-center justify-between gap-3">
          {metadata ? (
            <div className="flex min-w-0 flex-wrap items-center gap-2 text-xs text-[var(--text-muted)]">
              {metadata}
            </div>
          ) : (
            <span />
          )}
          {actions ? (
            <div className="flex flex-wrap items-center justify-end gap-2">
              {actions}
            </div>
          ) : null}
        </div>
      ) : null}

      {alerts}

      {children}
    </div>
  );
}
