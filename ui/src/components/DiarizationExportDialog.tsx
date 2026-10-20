import { useState } from "react";

import {
  ExportDialog,
  ExportSwitchRow,
  type ExportFormatOption,
} from "@/components/ExportDialog";
import { saveFile } from "@/shared/saveFile";
import type { DiarizationRecord } from "../api";
import {
  buildDiarizationExport,
  type DiarizationExportFormat,
} from "../utils/diarizationExport";

type ExportableDiarizationRecord = Pick<
  DiarizationRecord,
  | "id"
  | "created_at"
  | "model_id"
  | "speaker_count"
  | "corrected_speaker_count"
  | "duration_secs"
  | "audio_filename"
  | "speaker_name_overrides"
  | "utterances"
  | "transcript"
  | "raw_transcript"
>;

interface DiarizationExportDialogProps {
  record: ExportableDiarizationRecord | null;
  children?: React.ReactNode;
  open?: boolean;
  onOpenChange?: (open: boolean) => void;
}

const FORMAT_OPTIONS: ExportFormatOption<DiarizationExportFormat>[] = [
  {
    id: "txt",
    label: "Text",
    description: "A readable transcript, each line marked with its speaker.",
  },
  {
    id: "json",
    label: "JSON",
    description: "Structured transcript entries for other tools.",
  },
  {
    id: "srt",
    label: "SRT",
    description: "Captions with the speaker in each cue.",
  },
  {
    id: "vtt",
    label: "VTT",
    description: "Captions for video on the web, with the speaker in each cue.",
  },
];

function hasSpeakerNames(record: ExportableDiarizationRecord | null): boolean {
  return Object.values(record?.speaker_name_overrides ?? {}).some(
    (name) => String(name).trim().length > 0,
  );
}

export function DiarizationExportDialog({
  record,
  children,
  open: controlledOpen,
  onOpenChange: controlledOnOpenChange,
}: DiarizationExportDialogProps) {
  const [uncontrolledOpen, setUncontrolledOpen] = useState(false);
  const [format, setFormat] = useState<DiarizationExportFormat>("txt");
  const [useSpeakerNames, setUseSpeakerNames] = useState(true);
  const [includeMetadata, setIncludeMetadata] = useState(false);
  const open = controlledOpen ?? uncontrolledOpen;
  const setOpen = controlledOnOpenChange ?? setUncontrolledOpen;

  const namedSpeakers = hasSpeakerNames(record);
  const canIncludeMetadata = format === "txt" || format === "json";

  async function handleExport(): Promise<boolean> {
    if (!record) {
      return false;
    }
    // Without the names, the export uses the labels diarization gave.
    const source =
      namedSpeakers && !useSpeakerNames
        ? {
            ...record,
            speaker_name_overrides: {},
            corrected_speaker_count: record.speaker_count,
          }
        : record;
    const payload = buildDiarizationExport(source, format, {
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
      formats={FORMAT_OPTIONS}
      format={format}
      onFormatChange={setFormat}
      onExport={handleExport}
      disabled={!record}
    >
      {namedSpeakers ? (
        <ExportSwitchRow
          title="Speaker names"
          description="Use the names you gave each speaker"
          checked={useSpeakerNames}
          onCheckedChange={setUseSpeakerNames}
        />
      ) : null}
      <ExportSwitchRow
        title="Details"
        description={
          canIncludeMetadata
            ? "File name, model, speakers and length"
            : "Only in Text and JSON files"
        }
        checked={canIncludeMetadata && includeMetadata}
        onCheckedChange={setIncludeMetadata}
        disabled={!canIncludeMetadata}
      />
    </ExportDialog>
  );
}
