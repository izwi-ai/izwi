import { useEffect, useMemo, useState } from "react";

import {
  ExportDialog,
  ExportSwitchRow,
  type ExportFormatOption,
} from "@/components/ExportDialog";
import { saveFile } from "@/shared/saveFile";
import {
  availableTranscriptionExportFormats,
  buildTranscriptionExport,
  type ExportableTranscriptionRecord,
  type TranscriptionExportFormat,
} from "@/utils/transcriptionExport";

interface TranscriptionExportDialogProps {
  record: ExportableTranscriptionRecord | null;
  children?: React.ReactNode;
  open?: boolean;
  onOpenChange?: (open: boolean) => void;
}

const FORMAT_OPTIONS: ExportFormatOption<TranscriptionExportFormat>[] = [
  {
    id: "txt",
    label: "Text",
    description: "A readable transcript, with timestamps when there are any.",
  },
  {
    id: "json",
    label: "JSON",
    description: "Structured transcript entries for other tools.",
  },
  {
    id: "srt",
    label: "SRT",
    description: "Captions for video editors and players.",
  },
  {
    id: "vtt",
    label: "VTT",
    description: "Captions for video on the web.",
  },
];

export function TranscriptionExportDialog({
  record,
  children,
  open: controlledOpen,
  onOpenChange: controlledOnOpenChange,
}: TranscriptionExportDialogProps) {
  const [uncontrolledOpen, setUncontrolledOpen] = useState(false);
  const [format, setFormat] = useState<TranscriptionExportFormat>("txt");
  const [includeMetadata, setIncludeMetadata] = useState(false);
  const open = controlledOpen ?? uncontrolledOpen;
  const setOpen = controlledOnOpenChange ?? setUncontrolledOpen;

  const formatOptions = useMemo(() => {
    const available = record
      ? availableTranscriptionExportFormats(record)
      : ["txt", "json"];
    return FORMAT_OPTIONS.filter((option) => available.includes(option.id));
  }, [record]);
  useEffect(() => {
    if (!formatOptions.some((option) => option.id === format)) {
      setFormat("txt");
    }
  }, [format, formatOptions]);
  const canIncludeMetadata = format === "txt" || format === "json";

  async function handleExport(): Promise<boolean> {
    if (!record) {
      return false;
    }
    const payload = buildTranscriptionExport(record, format, {
      includeMetadata: canIncludeMetadata && includeMetadata,
    });
    return saveFile(payload.filename, payload.content, payload.mimeType);
  }

  return (
    <ExportDialog
      title="Export transcript"
      open={open}
      onOpenChange={setOpen}
      trigger={children}
      formats={formatOptions}
      format={format}
      onFormatChange={setFormat}
      onExport={handleExport}
      disabled={!record}
    >
      <ExportSwitchRow
        title="Details"
        description={
          canIncludeMetadata
            ? "File name, model, language and length"
            : "Only in Text and JSON files"
        }
        checked={canIncludeMetadata && includeMetadata}
        onCheckedChange={setIncludeMetadata}
        disabled={!canIncludeMetadata}
      />
    </ExportDialog>
  );
}
