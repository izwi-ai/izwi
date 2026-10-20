import { useCallback, useEffect, useRef, useState } from "react";
import { Loader2, RefreshCw, Sparkles } from "lucide-react";

import { api, type ModelInfo, type SpeechHistoryRecord } from "@/api";
import { useAudioPreview } from "@/components/audio/useAudioPreview";
import { CreationDialog } from "@/components/CreationDialog";
import { CreationBody, CreationField } from "@/components/creation/CreationParts";
import { formatClockDuration } from "@/components/history/format";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { cn } from "@/lib/utils";
import { LANGUAGES } from "@/types";
import { blobToBase64Payload } from "@/utils/audioBase64";
import {
  FIELD_LABEL_CLASS,
  FormError,
  PlayButton,
  TEXTAREA_CLASS,
  TEXT_INPUT_CLASS,
  VoiceModelRow,
  VoiceSavedBody,
} from "./NewVoiceParts";

export const PREVIEW_LINE_MAX_CHARS = 300;
const DEFAULT_PREVIEW_LINE =
  "Hello, this is Izwi. Here's how I sound reading a short paragraph aloud.";
const DETECT_LANGUAGE = "Auto";
// Each option nudges the description a different way so the three differ.
const OPTION_STYLES = [
  "",
  " Make the delivery slightly warmer, more human, and more conversational.",
  " Make the delivery slightly more polished, authoritative, and presentation-ready.",
];
const EXAMPLES = [
  {
    name: "Calm narrator",
    description:
      "A warm, calm narrator in her forties with a light British accent. Unhurried and clear.",
  },
  {
    name: "Storyteller",
    description:
      "A gentle, warm male voice with a storytelling quality, made for audiobooks and bedtime stories.",
  },
  {
    name: "Upbeat host",
    description:
      "A young, energetic host, bright and enthusiastic, for adverts and podcasts.",
  },
  {
    name: "Documentary",
    description:
      "A mature, thoughtful voice with a slow, measured delivery for documentaries.",
  },
];

interface DesignOption {
  number: number;
  record: SpeechHistoryRecord;
}

interface DesignVoiceDialogProps {
  open: boolean;
  onClose: () => void;
  model: string | null;
  modelInfo?: ModelInfo | null;
  modelReady: boolean;
  isModelManagerOpen?: boolean;
  onOpenModelManager: () => void;
  /** Generate was asked for without a loaded VoiceDesign model. */
  onModelRequired: () => void;
  onSaved: (voiceId: string) => void;
  onUseInTextToSpeech?: (voiceId: string) => void;
}

export function DesignVoiceDialog({
  open,
  onClose,
  model,
  modelInfo = null,
  modelReady,
  isModelManagerOpen = false,
  onOpenModelManager,
  onModelRequired,
  onSaved,
  onUseInTextToSpeech,
}: DesignVoiceDialogProps) {
  const [description, setDescription] = useState("");
  const [examplesOpen, setExamplesOpen] = useState(false);
  const [previewLine, setPreviewLine] = useState(DEFAULT_PREVIEW_LINE);
  const [options, setOptions] = useState<DesignOption[]>([]);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [generating, setGenerating] = useState(false);
  const [name, setName] = useState("");
  const [language, setLanguage] = useState(DETECT_LANGUAGE);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [saved, setSaved] = useState<{ id: string; name: string } | null>(null);

  const generationRef = useRef<AbortController | null>(null);
  // Generated options not yet deleted or saved from: what closing discards.
  const unsavedIdsRef = useRef(new Set<string>());
  const preview = useAudioPreview();

  const deleteOptions = useCallback((ids: string[]) => {
    for (const id of ids) {
      unsavedIdsRef.current.delete(id);
      void api.deleteVoiceDesignRecord(id).catch(() => {});
    }
  }, []);

  // Stops a generation and deletes every option the user did not save.
  const discardAll = useCallback(() => {
    generationRef.current?.abort();
    generationRef.current = null;
    deleteOptions([...unsavedIdsRef.current]);
  }, [deleteOptions]);

  const stopPreview = preview.stop;
  useEffect(() => {
    if (open) {
      return;
    }
    discardAll();
    stopPreview();
    setDescription("");
    setExamplesOpen(false);
    setPreviewLine(DEFAULT_PREVIEW_LINE);
    setOptions([]);
    setSelectedId(null);
    setGenerating(false);
    setName("");
    setLanguage(DETECT_LANGUAGE);
    setSaving(false);
    setError(null);
    setSaved(null);
  }, [discardAll, open, stopPreview]);

  useEffect(() => discardAll, [discardAll]);

  const generate = async () => {
    if (!model || !modelReady) {
      onModelRequired();
      return;
    }
    const voiceDescription = description.trim();
    const text = previewLine.trim();
    if (!voiceDescription || !text) {
      return;
    }

    discardAll();
    const controller = new AbortController();
    generationRef.current = controller;
    preview.stop();
    setOptions([]);
    setSelectedId(null);
    setError(null);
    setGenerating(true);
    try {
      for (const [index, style] of OPTION_STYLES.entries()) {
        const record = await api.createVoiceDesignRecord(
          {
            text,
            model_id: model,
            language: language === DETECT_LANGUAGE ? undefined : language,
            max_tokens: 0,
            voice_description: `${voiceDescription}${style}`,
          },
          controller.signal,
        );
        // Stopped after the server had already made it: delete it.
        if (controller.signal.aborted) {
          deleteOptions([record.id]);
          return;
        }
        unsavedIdsRef.current.add(record.id);
        setOptions((current) => [...current, { number: index + 1, record }]);
      }
    } catch (err) {
      if (!controller.signal.aborted) {
        setError(err instanceof Error ? err.message : "Couldn't design the voice.");
      }
    } finally {
      if (generationRef.current === controller) {
        generationRef.current = null;
        setGenerating(false);
      }
    }
  };

  const picked = options.find((option) => option.record.id === selectedId) ?? null;
  const canSave = Boolean(picked && name.trim()) && !generating && !saving;

  const save = async () => {
    if (!picked || !canSave) {
      return;
    }
    const { record } = picked;
    preview.stop();
    setSaving(true);
    setError(null);
    // The saved voice keeps a link to its option, so closing must not delete it.
    unsavedIdsRef.current.delete(record.id);
    try {
      const response = await fetch(api.voiceDesignRecordAudioUrl(record.id));
      if (!response.ok) {
        throw new Error(`Couldn't load the option's audio (${response.status}).`);
      }
      const audio = await response.blob();
      const voice = await api.createSavedVoice({
        name: name.trim(),
        reference_text: record.input_text,
        audio_base64: await blobToBase64Payload(audio),
        audio_mime_type: record.audio_mime_type || audio.type || "audio/wav",
        audio_filename: record.audio_filename || `voice-design-${record.id}.wav`,
        source_route_kind: "voice_design",
        source_record_id: record.id,
      });
      deleteOptions([...unsavedIdsRef.current]);
      setSaved({ id: voice.id, name: voice.name || name.trim() });
      onSaved(voice.id);
    } catch (err) {
      unsavedIdsRef.current.add(record.id);
      setError(err instanceof Error ? err.message : "Couldn't save the voice.");
    } finally {
      setSaving(false);
    }
  };

  if (saved) {
    return (
      <CreationDialog
        open={open}
        onClose={onClose}
        className="max-w-[32rem]"
        title="Voice saved"
        onSubmit={
          onUseInTextToSpeech
            ? () => {
                onUseInTextToSpeech(saved.id);
                onClose();
              }
            : undefined
        }
        submitLabel="Use in Text to Speech"
      >
        <VoiceSavedBody name={saved.name} />
      </CreationDialog>
    );
  }

  const canGenerate = Boolean(description.trim() && previewLine.trim()) && !saving;
  const pendingCount = generating ? OPTION_STYLES.length - options.length : 0;

  return (
    <CreationDialog
      open={open}
      onClose={onClose}
      modal={!isModelManagerOpen}
      blockDismiss={isModelManagerOpen || saving}
      className="max-w-[52rem]"
      title="Design a voice"
      onSubmit={() => void save()}
      submitLabel={picked ? `Save option ${picked.number}` : "Save option"}
      submitDisabled={!canSave}
      submitting={saving}
      footerStart="The options you don't pick are discarded."
    >
      <CreationBody
        main={
          <>
            {/* The description comes before "Examples" in the DOM so the
                dialog focuses it first (autoFocus would stop the dialog
                returning focus to "New voice" on close); the grid draws
                "Examples" beside the label. */}
            <div className="grid grid-cols-[minmax(0,1fr)_auto] items-center gap-x-3 gap-y-1.5">
              <label htmlFor="design-voice-description" className={FIELD_LABEL_CLASS}>
                Describe the voice
              </label>
              <textarea
                id="design-voice-description"
                value={description}
                onChange={(event) => setDescription(event.target.value)}
                rows={2}
                disabled={saving}
                placeholder="Who is speaking, and how? Age, tone, accent, pace…"
                className={cn(TEXTAREA_CLASS, "col-span-2 row-start-2")}
              />
              <button
                type="button"
                aria-expanded={examplesOpen}
                aria-controls="design-voice-examples"
                onClick={() => setExamplesOpen((current) => !current)}
                className="col-start-2 row-start-1 text-[12.5px] text-[var(--text-muted)] underline-offset-[3px] hover:text-[var(--text-primary)] hover:underline"
              >
                Examples
              </button>
              {examplesOpen ? (
                <div id="design-voice-examples" className="col-span-2 flex flex-wrap gap-1.5">
                  {EXAMPLES.map((example) => (
                    <button
                      key={example.name}
                      type="button"
                      onClick={() => {
                        setDescription(example.description);
                        setExamplesOpen(false);
                      }}
                      disabled={saving}
                      className="h-7 rounded-full border border-[var(--border-subtle)] bg-[var(--surface-panel)] px-2.5 text-[12.5px] text-[var(--text-muted)] transition-colors hover:border-[var(--border-strong)] hover:text-[var(--text-primary)]"
                    >
                      {example.name}
                    </button>
                  ))}
                </div>
              ) : null}
            </div>

            <div className="flex flex-col gap-1.5">
              <div className="flex items-center justify-between gap-3">
                <label htmlFor="design-voice-preview-line" className={FIELD_LABEL_CLASS}>
                  Preview line
                </label>
                <span
                  id="design-voice-preview-count"
                  className="text-xs tabular-nums text-[var(--text-subtle)]"
                >
                  {previewLine.length} / {PREVIEW_LINE_MAX_CHARS}
                </span>
              </div>
              <textarea
                id="design-voice-preview-line"
                value={previewLine}
                onChange={(event) =>
                  setPreviewLine(event.target.value.slice(0, PREVIEW_LINE_MAX_CHARS))
                }
                maxLength={PREVIEW_LINE_MAX_CHARS}
                rows={2}
                disabled={saving}
                aria-describedby="design-voice-preview-count"
                className={TEXTAREA_CLASS}
              />
            </div>

            <div className="flex flex-col gap-1.5">
              <div className="flex min-h-7 items-center justify-between gap-3">
                <span id="design-voice-options-label" className={FIELD_LABEL_CLASS}>
                  Pick one
                </span>
                {options.length > 0 || generating ? (
                  <Button
                    type="button"
                    variant="outline"
                    size="sm"
                    className="h-7 gap-1.5 px-2.5 text-[12.5px]"
                    onClick={() => void generate()}
                    disabled={generating || !canGenerate}
                  >
                    {generating ? (
                      <Loader2 aria-hidden="true" className="animate-spin" />
                    ) : (
                      <RefreshCw aria-hidden="true" />
                    )}
                    {generating ? "Generating…" : "Generate again"}
                  </Button>
                ) : null}
              </div>
              {options.length === 0 && !generating ? (
                <div className="flex flex-col items-center gap-3 rounded-[12px] border border-dashed border-[var(--border-strong)] bg-[var(--surface-panel)] px-4 py-5 text-center">
                  <span className="text-[13px] text-[var(--text-subtle)]">
                    Describe the voice, then compare three takes on it.
                  </span>
                  <Button
                    type="button"
                    variant="outline"
                    size="sm"
                    className="h-8 gap-1.5"
                    onClick={() => void generate()}
                    disabled={!canGenerate}
                  >
                    <Sparkles aria-hidden="true" />
                    Generate options
                  </Button>
                </div>
              ) : (
                <div
                  role="radiogroup"
                  aria-labelledby="design-voice-options-label"
                  className="overflow-hidden rounded-[12px] border border-[var(--border-subtle)] bg-[var(--surface-panel)]"
                >
                  {options.map(({ number, record }) => {
                    const selected = record.id === selectedId;
                    return (
                      <label
                        key={record.id}
                        className={cn(
                          "grid cursor-pointer grid-cols-[auto_auto_minmax(0,1fr)_auto] items-center gap-3 border-b border-[var(--border-subtle)] px-3 py-2.5 last:border-b-0",
                          selected && "bg-[var(--accent-soft)]",
                        )}
                      >
                        <input
                          type="radio"
                          name="design-voice-option"
                          value={record.id}
                          checked={selected}
                          onChange={() => setSelectedId(record.id)}
                          disabled={saving}
                          aria-label={`Option ${number}`}
                          className="h-4 w-4 cursor-pointer appearance-none rounded-full border border-[var(--border-strong)] checked:border-[5px] checked:border-[var(--accent-solid)] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring/45"
                        />
                        <PlayButton
                          small
                          playing={preview.playingId === record.id}
                          label={`option ${number}`}
                          onClick={() =>
                            preview.toggle(record.id, api.voiceDesignRecordAudioUrl(record.id))
                          }
                        />
                        <span
                          className={cn(
                            "text-[13.5px] text-[var(--text-primary)]",
                            selected && "font-medium",
                          )}
                        >
                          Option {number}
                        </span>
                        <span className="text-[12px] tabular-nums text-[var(--text-subtle)]">
                          {formatClockDuration(record.audio_duration_secs)}
                        </span>
                      </label>
                    );
                  })}
                  {Array.from({ length: pendingCount }, (_, index) => (
                    <div
                      key={`pending-${index}`}
                      className="flex items-center gap-3 border-b border-[var(--border-subtle)] px-3 py-3 text-[13px] text-[var(--text-subtle)] last:border-b-0"
                    >
                      {index === 0 ? (
                        <Loader2 aria-hidden="true" className="h-4 w-4 animate-spin" />
                      ) : (
                        <span aria-hidden="true" className="h-4 w-4 rounded-full border border-dashed border-[var(--border-strong)]" />
                      )}
                      Option {options.length + index + 1}
                      {index === 0 ? " · generating…" : ""}
                    </div>
                  ))}
                </div>
              )}
            </div>
          </>
        }
        side={
          <>
            <CreationField label="Name" htmlFor="design-voice-name">
              <Input
                id="design-voice-name"
                value={name}
                onChange={(event) => setName(event.target.value)}
                placeholder="e.g. Calm British narrator"
                disabled={saving}
                className={TEXT_INPUT_CLASS}
              />
            </CreationField>

            <CreationField label="Language">
              <Select value={language} onValueChange={setLanguage} disabled={generating || saving}>
                <SelectTrigger
                  aria-label="Language"
                  className="h-9 w-full rounded-[10px] border-[var(--border-default)] bg-[var(--surface-panel)] text-[13.5px]"
                >
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {LANGUAGES.map((entry) => (
                    <SelectItem key={entry.id} value={entry.id}>
                      {entry.id === DETECT_LANGUAGE ? "Detect from text" : entry.name}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </CreationField>

            <VoiceModelRow
              model={model}
              modelInfo={modelInfo}
              ready={modelReady}
              onChange={onOpenModelManager}
              disabled={saving}
            />

            <FormError message={error} />
          </>
        }
      />
    </CreationDialog>
  );
}
