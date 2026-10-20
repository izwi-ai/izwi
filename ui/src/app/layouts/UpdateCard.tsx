import { useId, useState } from "react";
import { Download, X } from "lucide-react";

import { useAppUpdates } from "@/app/providers/AppUpdateProvider";
import { NotificationSlot } from "@/app/providers/NotificationProvider";
import { cn } from "@/lib/utils";

const focusRing =
  "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[var(--border-strong)]";
const primaryButton = cn(
  "inline-flex h-8 items-center rounded-full bg-[var(--accent-solid)] px-3.5 text-[13px] font-medium text-[var(--text-on-accent)] transition-opacity hover:opacity-90 disabled:opacity-50",
  focusRing,
);
const secondaryButton = cn(
  "inline-flex h-8 items-center rounded-full border border-[var(--border-default)] px-3.5 text-[13px] text-[var(--text-primary)] transition-colors hover:bg-[var(--bg-surface-2)]",
  focusRing,
);

/**
 * The desktop update card at the bottom of the notification stack: an update
 * on offer, its download, then "Restart now" once it is installed.
 */
export function UpdateCard() {
  const {
    availableUpdate,
    status,
    downloadedVersion,
    isPromptOpen,
    progressPercent,
    errorMessage,
    dismissPrompt,
    installUpdate,
    restartToApply,
  } = useAppUpdates();
  const titleId = useId();
  const notesId = useId();
  const [showNotes, setShowNotes] = useState(false);

  if (!isPromptOpen || !availableUpdate) {
    return null;
  }

  const { version, notes, platformBehavior } = availableUpdate;
  const ready = downloadedVersion === version;
  const downloading = status === "downloading";
  const title = ready
    ? `Izwi ${version} is ready to install`
    : downloading
      ? `Downloading Izwi ${version}`
      : `Izwi ${version} is available`;
  const summary = ready
    ? "Restart to finish. Your history and models stay as they are."
    : platformBehavior.appExitsDuringInstall
      ? "Izwi closes while it installs. Your history and models stay as they are."
      : "Download it now and restart when you’re ready.";

  return (
    <NotificationSlot>
      <section
        aria-labelledby={titleId}
        className="pointer-events-auto flex flex-col gap-3.5 rounded-[14px] border border-[var(--border-default)] bg-[var(--surface-panel)] p-4 text-[var(--text-primary)] shadow-[var(--shadow-raised)]"
      >
        <div className="grid grid-cols-[auto_minmax(0,1fr)_auto] items-start gap-3">
          <span className="flex h-8 w-8 items-center justify-center rounded-full bg-[var(--accent-soft)]">
            <Download aria-hidden="true" className="h-4 w-4" />
          </span>
          <div className="flex min-w-0 flex-col gap-0.5">
            <h2 id={titleId} className="text-[13.5px] font-medium leading-5">
              {title}
            </h2>
            {downloading ? (
              <div className="mt-1.5 flex items-center gap-2.5">
                <div
                  role="progressbar"
                  aria-label="Update download"
                  aria-valuemin={0}
                  aria-valuemax={100}
                  aria-valuenow={progressPercent ?? undefined}
                  className="h-[3px] flex-1 overflow-hidden rounded-full bg-[var(--bg-surface-3)]"
                >
                  <div
                    className={cn(
                      "h-full rounded-full bg-[var(--accent-solid)] transition-[width] duration-200",
                      progressPercent === null && "w-1/3 animate-pulse",
                    )}
                    style={
                      progressPercent !== null
                        ? { width: `${progressPercent}%` }
                        : undefined
                    }
                  />
                </div>
                <span className="text-[12.5px] tabular-nums text-[var(--text-muted)]">
                  {progressPercent !== null ? `${progressPercent}%` : "Starting…"}
                </span>
              </div>
            ) : (
              <p className="text-[12.5px] leading-[1.45] text-[var(--text-muted)]">
                {summary}
              </p>
            )}
            {errorMessage && !ready ? (
              <p className="text-[12.5px] leading-[1.45] text-[var(--danger-text)]">
                {errorMessage}
              </p>
            ) : null}
            {notes ? (
              <button
                type="button"
                aria-expanded={showNotes}
                aria-controls={showNotes ? notesId : undefined}
                onClick={() => setShowNotes((shown) => !shown)}
                className={cn(
                  "mt-1 w-fit rounded-sm text-[12.5px] underline decoration-[var(--border-strong)] underline-offset-[3px] transition-colors hover:decoration-[var(--text-primary)]",
                  focusRing,
                )}
              >
                What’s new
              </button>
            ) : null}
          </div>
          <button
            type="button"
            onClick={dismissPrompt}
            aria-label="Dismiss update"
            className={cn(
              "flex h-6 w-6 items-center justify-center rounded-md text-[var(--text-subtle)] transition-colors hover:bg-[var(--accent-soft)] hover:text-[var(--text-primary)]",
              focusRing,
            )}
          >
            <X aria-hidden="true" className="h-3.5 w-3.5" />
          </button>
        </div>

        {notes && showNotes ? (
          <p
            id={notesId}
            className="max-h-32 overflow-y-auto whitespace-pre-line rounded-[10px] border border-[var(--border-subtle)] bg-[var(--surface-panel-alt)] px-3 py-2 text-[12.5px] leading-[1.5] text-[var(--text-muted)]"
          >
            {notes}
          </p>
        ) : null}

        <div className="flex justify-end gap-2">
          <button type="button" onClick={dismissPrompt} className={secondaryButton}>
            Later
          </button>
          {ready ? (
            <button
              type="button"
              onClick={() => void restartToApply()}
              className={primaryButton}
            >
              Restart now
            </button>
          ) : downloading ? null : (
            <button
              type="button"
              disabled={status === "checking"}
              onClick={() => void installUpdate()}
              className={primaryButton}
            >
              Install
            </button>
          )}
        </div>
      </section>
    </NotificationSlot>
  );
}
