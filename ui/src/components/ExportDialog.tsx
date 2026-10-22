import { useEffect, useState, type ReactNode } from "react";
import { AlertTriangle } from "lucide-react";

import {
  CREATION_DIALOG_CONTENT_CLASS,
  CreationDialogFrame,
} from "@/components/CreationDialog";
import { CreationOption } from "@/components/creation/CreationParts";
import {
  SegmentedFilter,
  type SegmentOption,
} from "@/components/history/HistoryToolbar";
import { Dialog, DialogContent, DialogTrigger } from "@/components/ui/dialog";
import { cn } from "@/lib/utils";
import { savesWithDialog } from "@/shared/saveFile";

export const EXPORT_DIALOG_CLASS = "max-w-[460px]";

/** Footer note for export dialogs: the desktop app asks where to save. */
export function exportFooterNote(): string | null {
  return savesWithDialog() ? "You choose where it's saved." : null;
}

/** The bordered list of export settings, one row each. */
export function ExportSettings({ children }: { children: ReactNode }) {
  return (
    <div className="divide-y divide-(--border-subtle) overflow-hidden rounded-[12px] border border-(--border-subtle) bg-(--surface-panel)">
      {children}
    </div>
  );
}

/** A setting chosen from a few options, with a line about the choice. */
export function ExportChoiceRow<T extends string>({
  label,
  hint,
  options,
  value,
  onChange,
  disabled = false,
}: {
  label: string;
  hint?: ReactNode;
  options: SegmentOption<T>[];
  value: T;
  onChange: (value: T) => void;
  disabled?: boolean;
}) {
  return (
    <div className="flex flex-col gap-1 px-4 py-3">
      <div className="flex flex-wrap items-center justify-between gap-x-4 gap-y-2">
        <span className="text-[13.5px] font-medium text-(--text-primary)">
          {label}
        </span>
        <SegmentedFilter
          label={label}
          options={options}
          value={value}
          onChange={onChange}
          disabled={disabled}
        />
      </div>
      {hint ? (
        <p className="text-xs leading-4 text-(--text-subtle)">{hint}</p>
      ) : null}
    </div>
  );
}

/** A setting that is on or off. */
export function ExportSwitchRow(props: Parameters<typeof CreationOption>[0]) {
  return (
    <div className="px-4 py-3">
      <CreationOption {...props} />
    </div>
  );
}

export function ExportError({ message }: { message: string }) {
  return (
    <div
      role="alert"
      className="flex items-start gap-2 rounded-[10px] border border-(--danger-border) bg-(--danger-bg) p-2.5 text-[13px] text-(--danger-text)"
    >
      <AlertTriangle aria-hidden="true" className="mt-0.5 h-4 w-4 shrink-0" />
      <span>{message}</span>
    </div>
  );
}

export interface ExportFormatOption<F extends string> extends SegmentOption<F> {
  description: string;
}

interface ExportDialogProps<F extends string> {
  title: ReactNode;
  open: boolean;
  onOpenChange: (open: boolean) => void;
  /** Opens the dialog when clicked. */
  trigger?: ReactNode;
  formats: ExportFormatOption<F>[];
  format: F;
  onFormatChange: (format: F) => void;
  /** Rows under the format: speaker names, details. */
  children?: ReactNode;
  /** Builds and saves the file; false when the save dialog was cancelled. */
  onExport: () => Promise<boolean>;
  disabled?: boolean;
}

/** Export a transcript as a file: a format, a few options, then Export…. */
export function ExportDialog<F extends string>({
  title,
  open,
  onOpenChange,
  trigger,
  formats,
  format,
  onFormatChange,
  children,
  onExport,
  disabled = false,
}: ExportDialogProps<F>) {
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (open) {
      setError(null);
    }
  }, [open]);

  const exportFile = async () => {
    if (saving) {
      return;
    }
    setSaving(true);
    setError(null);
    try {
      // A cancelled save dialog leaves this one open to try again.
      if (await onExport()) {
        onOpenChange(false);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "The file couldn't be saved.");
    } finally {
      setSaving(false);
    }
  };

  const selected = formats.find((option) => option.id === format);

  return (
    <Dialog
      open={open}
      onOpenChange={(next) => {
        if (!saving) {
          onOpenChange(next);
        }
      }}
    >
      {trigger ? <DialogTrigger asChild>{trigger}</DialogTrigger> : null}
      <DialogContent
        className={cn(CREATION_DIALOG_CONTENT_CLASS, EXPORT_DIALOG_CLASS)}
      >
        <CreationDialogFrame
          title={title}
          onClose={() => onOpenChange(false)}
          onSubmit={() => void exportFile()}
          submitLabel="Export…"
          submitDisabled={disabled}
          submitting={saving}
          cancelDisabled={saving}
          footerStart={exportFooterNote()}
        >
          <div className="flex flex-col gap-3 px-5 py-4 sm:px-6">
            <ExportSettings>
              <ExportChoiceRow
                label="Format"
                hint={selected?.description}
                options={formats}
                value={format}
                onChange={onFormatChange}
                disabled={saving}
              />
              {children}
            </ExportSettings>
            {error ? <ExportError message={error} /> : null}
          </div>
        </CreationDialogFrame>
      </DialogContent>
    </Dialog>
  );
}
