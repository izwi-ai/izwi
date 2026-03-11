import { useMemo, useState } from "react";
import { Download } from "lucide-react";

import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Switch } from "@/components/ui/switch";
import {
  buildTranscriptionExport,
  type ExportableTranscriptionRecord,
  type TranscriptionExportFormat,
} from "@/utils/transcriptionExport";

interface TranscriptionExportDialogProps {
  record: ExportableTranscriptionRecord | null;
  children: React.ReactNode;
}

const FORMAT_OPTIONS: Array<{
  value: TranscriptionExportFormat;
  label: string;
  description: string;
}> = [
  {
    value: "txt",
    label: "TXT transcript",
    description: "Readable transcript text with timestamps when available.",
  },
  {
    value: "json",
    label: "JSON transcript",
    description: "Structured transcript entries for downstream workflows.",
  },
  {
    value: "srt",
    label: "SRT captions",
    description: "Subtitle file generated from transcript timing when available.",
  },
  {
    value: "vtt",
    label: "VTT captions",
    description: "Web subtitle file suitable for HTML5 video workflows.",
  },
];

function downloadTextFile(
  content: string,
  filename: string,
  mimeType: string,
): void {
  const blob = new Blob([content], { type: mimeType });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = filename;
  anchor.click();
  URL.revokeObjectURL(url);
}

export function TranscriptionExportDialog({
  record,
  children,
}: TranscriptionExportDialogProps) {
  const [open, setOpen] = useState(false);
  const [format, setFormat] = useState<TranscriptionExportFormat>("txt");
  const [includeMetadata, setIncludeMetadata] = useState(false);

  const selectedOption = useMemo(
    () => FORMAT_OPTIONS.find((option) => option.value === format) ?? FORMAT_OPTIONS[0],
    [format],
  );
  const canIncludeMetadata = format === "txt" || format === "json";

  function handleExport(): void {
    if (!record) {
      return;
    }

    const payload = buildTranscriptionExport(record, format, {
      includeMetadata: canIncludeMetadata ? includeMetadata : false,
    });

    downloadTextFile(payload.content, payload.filename, payload.mimeType);
    setOpen(false);
  }

  return (
    <Dialog open={open} onOpenChange={setOpen}>
      <DialogTrigger asChild>{children}</DialogTrigger>
      <DialogContent className="max-w-lg border-[var(--border-strong)] bg-[var(--bg-surface-0)]">
        <DialogHeader>
          <DialogTitle className="text-base text-[var(--text-primary)]">
            Export transcription
          </DialogTitle>
          <DialogDescription className="text-[var(--text-muted)]">
            Export transcript text or captions using the same formats available in
            diarization.
          </DialogDescription>
        </DialogHeader>

        <div className="space-y-4">
          <div className="space-y-2">
            <Label
              htmlFor="transcription-export-format"
              className="text-xs uppercase tracking-wider text-[var(--text-subtle)]"
            >
              Format
            </Label>
            <Select
              value={format}
              onValueChange={(value) => setFormat(value as TranscriptionExportFormat)}
            >
              <SelectTrigger
                id="transcription-export-format"
                className="border-[var(--border-muted)] bg-[var(--bg-surface-1)]"
              >
                <SelectValue placeholder="Choose an export format" />
              </SelectTrigger>
              <SelectContent>
                {FORMAT_OPTIONS.map((option) => (
                  <SelectItem key={option.value} value={option.value}>
                    {option.label}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
            <p className="text-sm text-[var(--text-muted)]">
              {selectedOption.description}
            </p>
          </div>

          <div className="flex items-center justify-between gap-3 rounded-xl border border-[var(--border-muted)] bg-[var(--bg-surface-1)] px-4 py-3">
            <div>
              <div className="text-sm font-medium text-[var(--text-primary)]">
                Include metadata
              </div>
              <div className="text-xs text-[var(--text-muted)]">
                Add file, model, language, and duration details to TXT or JSON
                exports.
              </div>
            </div>
            <Switch
              checked={canIncludeMetadata && includeMetadata}
              onCheckedChange={setIncludeMetadata}
              disabled={!canIncludeMetadata}
              aria-label="Include metadata"
            />
          </div>
        </div>

        <DialogFooter className="gap-2">
          <Button
            type="button"
            variant="outline"
            size="sm"
            className="h-9 border-[var(--border-muted)] bg-[var(--bg-surface-1)]"
            onClick={() => setOpen(false)}
          >
            Cancel
          </Button>
          <Button
            type="button"
            size="sm"
            className="h-9 gap-1.5"
            onClick={handleExport}
            disabled={!record}
          >
            <Download className="h-3.5 w-3.5" />
            Export file
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
