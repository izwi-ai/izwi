import type { KeyboardEvent, ReactNode } from "react";
import { Loader2 } from "lucide-react";

import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogTitle,
} from "@/components/ui/dialog";
import { isMacPlatform } from "@/lib/platform";
import { cn } from "@/lib/utils";

interface CreationDialogProps {
  open: boolean;
  onClose: () => void;
  title: ReactNode;
  description?: ReactNode;
  /** Extra header content under the description (mode switches). */
  header?: ReactNode;
  children: ReactNode;
  /** Primary action; omit for dialogs that start work another way. */
  onSubmit?: () => void;
  submitLabel?: ReactNode;
  submitIcon?: ReactNode;
  submitDisabled?: boolean;
  submitting?: boolean;
  /** Left side of the footer (counts, hints). */
  footerStart?: ReactNode;
  /** Ignore Escape and outside clicks, e.g. while a nested dialog is open. */
  blockDismiss?: boolean;
  /** False while a nested non-modal dialog (model manager) is on top. */
  modal?: boolean;
  className?: string;
}

/**
 * Frame for "New …" dialogs: a fixed header, a scrolling body and a
 * footer with Cancel and the primary action, which Cmd/Ctrl+Enter also
 * triggers.
 */
export function CreationDialog({
  open,
  onClose,
  title,
  description,
  header,
  children,
  onSubmit,
  submitLabel,
  submitIcon,
  submitDisabled = false,
  submitting = false,
  footerStart,
  blockDismiss = false,
  modal = true,
  className,
}: CreationDialogProps) {
  const canSubmit = Boolean(onSubmit) && !submitDisabled && !submitting;
  const mac = isMacPlatform();

  const handleKeyDown = (event: KeyboardEvent<HTMLDivElement>) => {
    if (
      event.key === "Enter" &&
      (event.metaKey || event.ctrlKey) &&
      !event.nativeEvent.isComposing
    ) {
      event.preventDefault();
      if (canSubmit) {
        onSubmit?.();
      }
    }
  };

  const preventWhenBlocked = (event: Event) => {
    if (blockDismiss) {
      event.preventDefault();
    }
  };

  return (
    <Dialog
      open={open}
      modal={modal}
      onOpenChange={(nextOpen) => {
        if (!nextOpen && !blockDismiss) {
          onClose();
        }
      }}
    >
      <DialogContent
        className={cn(
          "flex flex-col gap-0 overflow-hidden border-[var(--border-strong)] bg-[var(--bg-surface-0)] p-0",
          className,
        )}
        onKeyDown={handleKeyDown}
        onEscapeKeyDown={preventWhenBlocked}
        onPointerDownOutside={preventWhenBlocked}
        onInteractOutside={preventWhenBlocked}
      >
        <div className="shrink-0 border-b border-[var(--border-muted)] px-5 py-5 pr-14 sm:px-6 sm:pr-14">
          <DialogTitle className="text-xl font-semibold tracking-tight text-[var(--text-primary)]">
            {title}
          </DialogTitle>
          {description ? (
            <DialogDescription className="mt-1 max-w-3xl text-[13px] leading-5 text-[var(--text-muted)]">
              {description}
            </DialogDescription>
          ) : null}
          {header}
        </div>

        <div className="min-h-0 flex-1 overflow-y-auto overscroll-contain">
          {children}
        </div>

        <div className="flex shrink-0 flex-wrap items-center justify-between gap-3 border-t border-[var(--border-muted)] bg-[var(--bg-surface-0)] px-5 py-3.5 sm:px-6">
          <div className="min-w-0 text-xs text-[var(--text-muted)]">
            {footerStart}
          </div>
          <div className="flex items-center gap-2">
            <Button
              type="button"
              variant="ghost"
              size="sm"
              className="h-9"
              onClick={onClose}
              disabled={blockDismiss}
            >
              Cancel
            </Button>
            {onSubmit ? (
              <Button
                type="button"
                size="sm"
                className="h-9 gap-2"
                onClick={onSubmit}
                disabled={!canSubmit}
                aria-keyshortcuts={mac ? "Meta+Enter" : "Control+Enter"}
              >
                {submitting ? (
                  <Loader2 className="h-4 w-4 animate-spin" />
                ) : (
                  submitIcon
                )}
                {submitLabel}
                <kbd
                  aria-hidden="true"
                  className="ml-0.5 font-sans text-[11px] opacity-60"
                >
                  {mac ? "⌘↵" : "Ctrl ↵"}
                </kbd>
              </Button>
            ) : null}
          </div>
        </div>
      </DialogContent>
    </Dialog>
  );
}
