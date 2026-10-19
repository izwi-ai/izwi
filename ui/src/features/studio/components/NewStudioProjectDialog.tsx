import { useEffect, useRef, useState, type ChangeEvent } from "react";
import { AlertTriangle, Upload } from "lucide-react";

import { api, type ModelInfo, type SavedVoiceSummary, type StudioProjectRecord } from "@/api";
import { CreationDialog } from "@/components/CreationDialog";
import { CreationBody, CreationField } from "@/components/creation/CreationParts";
import { Input } from "@/components/ui/input";
import {
  StudioVoicePicker,
  type StudioModelDownloads,
} from "@/features/studio/components/StudioVoicePicker";
import { defaultStudioVoice, estimatedLength, type StudioVoice } from "@/features/studio/studio";

const IMPORT_ACCEPT = ".txt,.md,.html,.htm,text/plain,text/markdown,text/html";

/** Script text from an imported file: HTML loses its markup. */
function importedScript(filename: string, text: string): string {
  if (/\.html?$/i.test(filename) && typeof DOMParser !== "undefined") {
    const document = new DOMParser().parseFromString(text, "text/html");
    return document.body?.textContent?.trim() || text;
  }
  return text;
}

interface NewStudioProjectDialogProps {
  open: boolean;
  onClose: () => void;
  models: ModelInfo[];
  savedVoices: SavedVoiceSummary[];
  downloads: StudioModelDownloads;
  onError: (message: string) => void;
  onCreated: (project: StudioProjectRecord) => void;
}

export function NewStudioProjectDialog({
  open,
  onClose,
  models,
  savedVoices,
  downloads,
  onError,
  onCreated,
}: NewStudioProjectDialogProps) {
  const [name, setName] = useState("");
  const [script, setScript] = useState("");
  const [filename, setFilename] = useState<string | null>(null);
  const [voice, setVoice] = useState<StudioVoice | null>(null);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const fileInput = useRef<HTMLInputElement>(null);

  useEffect(() => {
    if (!open) return;
    setName("");
    setScript("");
    setFilename(null);
    setError(null);
    setSubmitting(false);
  }, [open]);

  // Pick a voice once the models and saved voices are known.
  useEffect(() => {
    if (open && !voice) setVoice(defaultStudioVoice(models, savedVoices));
  }, [models, open, savedVoices, voice]);

  const importFile = async (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    event.target.value = "";
    if (!file) return;
    try {
      setScript(importedScript(file.name, await file.text()).trim());
      setFilename(file.name);
      if (!name.trim()) setName(file.name.replace(/\.[^.]+$/, ""));
      setError(null);
    } catch {
      setError(`Couldn't read ${file.name}.`);
    }
  };

  const submit = async () => {
    if (!script.trim()) {
      setError("Paste a script or import a file first.");
      return;
    }
    if (!voice) {
      setError("Choose a voice for the project.");
      return;
    }
    setSubmitting(true);
    setError(null);
    try {
      const project = await api.createStudioProject({
        name: name.trim() || undefined,
        source_filename: filename ?? undefined,
        source_text: script,
        model_id: voice.modelId,
        voice_mode: voice.voiceMode,
        speaker: voice.speaker ?? undefined,
        saved_voice_id: voice.savedVoiceId ?? undefined,
      });
      onCreated(project);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Couldn't create the project.");
      setSubmitting(false);
    }
  };

  const chars = Array.from(script).length;
  const length = estimatedLength(chars);

  return (
    <CreationDialog
      open={open}
      onClose={onClose}
      className="max-w-[52rem]"
      title="New project"
      onSubmit={() => void submit()}
      submitLabel="Create project"
      submitting={submitting}
      submitDisabled={!script.trim() || !voice}
      footerStart={chars > 0 ? `${chars.toLocaleString()} characters${length ? ` · ${length}` : ""}` : null}
    >
      <CreationBody
        main={
          <>
            <CreationField label="Name" htmlFor="studio-project-name">
              <Input
                id="studio-project-name"
                value={name}
                onChange={(event) => setName(event.target.value)}
                placeholder="Untitled project"
                disabled={submitting}
              />
            </CreationField>
            <div className="flex items-baseline justify-between">
              <label
                htmlFor="studio-project-script"
                className="text-[12.5px] font-medium text-[var(--text-muted)]"
              >
                Script
              </label>
              <button
                type="button"
                onClick={() => fileInput.current?.click()}
                disabled={submitting}
                className="inline-flex items-center gap-1.5 text-[12.5px] text-[var(--text-muted)] transition-colors hover:text-[var(--text-primary)]"
              >
                <Upload className="h-3.5 w-3.5" />
                Import a file
              </button>
              <input
                ref={fileInput}
                type="file"
                accept={IMPORT_ACCEPT}
                className="hidden"
                onChange={(event) => void importFile(event)}
              />
            </div>
            <textarea
              id="studio-project-script"
              value={script}
              onChange={(event) => setScript(event.target.value)}
              disabled={submitting}
              rows={12}
              autoFocus
              placeholder="Paste a chapter, an article or a podcast script…"
              className="min-h-[17rem] w-full flex-1 resize-y rounded-[12px] border border-[var(--border-default)] bg-[var(--surface-panel)] px-4 py-3 text-[14.5px] leading-6 text-[var(--text-primary)] outline-none transition-colors placeholder:text-[var(--text-subtle)] focus:border-[var(--border-strong)]"
            />
          </>
        }
        side={
          <>
            <CreationField label="Voice">
              <StudioVoicePicker
                models={models}
                savedVoices={savedVoices}
                value={voice}
                onChange={setVoice}
                onError={onError}
                downloads={downloads}
                disabled={submitting}
              />
            </CreationField>
            {error ? (
              <div
                role="alert"
                className="flex items-start gap-2 rounded-[10px] border border-[var(--danger-border)] bg-[var(--danger-bg)] p-2.5 text-[13px] text-[var(--danger-text)]"
              >
                <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0" />
                <span>{error}</span>
              </div>
            ) : null}
          </>
        }
      />
    </CreationDialog>
  );
}
