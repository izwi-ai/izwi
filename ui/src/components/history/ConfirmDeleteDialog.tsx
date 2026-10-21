import { AlertTriangle, Loader2 } from "lucide-react";

import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogTitle,
} from "@/components/ui/dialog";

interface ConfirmDeleteDialogProps {
  /** What is being deleted ("transcription"); null closes the dialog. */
  noun: string | null;
  /** The item's name, shown under the description. */
  itemLabel?: string;
  description: string;
  pending: boolean;
  error: string | null;
  onConfirm: () => void;
  onCancel: () => void;
}

export function ConfirmDeleteDialog({
  noun,
  itemLabel,
  description,
  pending,
  error,
  onConfirm,
  onCancel,
}: ConfirmDeleteDialogProps) {
  return (
    <Dialog
      open={noun !== null}
      onOpenChange={(open) => {
        if (!open && !pending) {
          onCancel();
        }
      }}
    >
      <DialogContent className="max-w-md border-(--border-strong) bg-(--bg-surface-1) p-5">
        <div className="flex items-start gap-3">
          <div className="mt-0.5 rounded-full border border-(--danger-border) bg-(--danger-bg) p-2 text-(--danger-text)">
            <AlertTriangle className="h-4 w-4" />
          </div>
          <div className="min-w-0 flex-1">
            <DialogTitle className="text-sm font-semibold text-(--text-primary)">
              Delete {noun}?
            </DialogTitle>
            <DialogDescription className="mt-1 text-sm text-(--text-muted)">
              {description}
            </DialogDescription>
            {itemLabel ? (
              <p className="mt-2 truncate text-xs text-(--text-subtle)">
                {itemLabel}
              </p>
            ) : null}
          </div>
        </div>

        {error ? (
          <div className="mt-4 rounded-md border border-(--danger-border) bg-(--danger-bg) px-3 py-2 text-xs text-(--danger-text)">
            {error}
          </div>
        ) : null}

        <div className="mt-5 flex items-center justify-end gap-2">
          <Button
            type="button"
            variant="outline"
            size="sm"
            className="h-8"
            onClick={onCancel}
            disabled={pending}
          >
            Cancel
          </Button>
          <Button
            type="button"
            variant="destructive"
            size="sm"
            className="h-8 gap-1.5"
            onClick={onConfirm}
            disabled={pending}
          >
            {pending ? (
              <>
                <Loader2 className="h-3.5 w-3.5 animate-spin" />
                Deleting
              </>
            ) : (
              `Delete ${noun ?? ""}`.trim()
            )}
          </Button>
        </div>
      </DialogContent>
    </Dialog>
  );
}
