import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { AlertTriangle, Check, Loader2, Upload } from "lucide-react";

import { api, type TranscriptionRecord } from "@/api";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogTitle,
} from "@/components/ui/dialog";
import { StatusBadge } from "@/components/ui/status-badge";
import { LANGUAGE_OPTIONS } from "@/features/transcription/playground/support";
import {
  prepareSpeechTextUploadBlob,
  resolveSourceAudioFilename,
  resolveSpeechTextUploadFilename,
} from "@/shared/audioUpload";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import type { SpeechTextCreationMode } from "@/features/speech-text/creationMode";
import { SpeechTextModeSwitch } from "@/features/speech-text/components/SpeechTextModeSwitch";

interface NewTranscriptionModalProps {
  isOpen: boolean;
  onClose: () => void;
  blockOutsideDismiss?: boolean;
  renderInDialog?: boolean;
  selectedMode?: SpeechTextCreationMode;
  onSelectMode?: (mode: SpeechTextCreationMode) => void;
  selectedModel: string | null;
  selectedModelReady: boolean;
  selectedModelStatus?: string | null;
  timestampAlignerModelId: string | null;
  timestampAlignerReady: boolean;
  timestampAlignerModelStatus?: string | null;
  onOpenModelManager: () => void;
  onModelRequired: () => void;
  onTimestampAlignerRequired: () => void;
  onCreated: (record: TranscriptionRecord) => Promise<void> | void;
  onStreamingStart?: () => void;
  onStreamingDelta?: (delta: string) => void;
  onStreamingFinal?: (record: TranscriptionRecord) => void;
  onStreamingError?: (message: string) => void;
  onStreamingDone?: () => void;
}

interface SubmitAudioOptions {
  filename?: string;
}

const ASR_MODEL_REQUIRED_ERROR =
  "Select and load an ASR model before creating a transcription.";
const ALIGNER_MODEL_REQUIRED_ERROR =
  "Load the timestamp aligner model to include timestamps.";

export function NewTranscriptionModal({
  isOpen,
  onClose,
  blockOutsideDismiss = false,
  renderInDialog = true,
  selectedMode = "transcription",
  onSelectMode,
  selectedModel,
  selectedModelReady,
  selectedModelStatus = null,
  timestampAlignerModelId,
  timestampAlignerReady,
  timestampAlignerModelStatus = null,
  onOpenModelManager,
  onModelRequired,
  onTimestampAlignerRequired,
  onCreated,
  onStreamingStart,
  onStreamingDelta,
  onStreamingFinal,
  onStreamingError,
  onStreamingDone,
}: NewTranscriptionModalProps) {
  const fileInputRef = useRef<HTMLInputElement | null>(null);

  const [selectedLanguage, setSelectedLanguage] = useState("English");
  const [includeTimestamps, setIncludeTimestamps] = useState(false);
  const [streamingEnabled, setStreamingEnabled] = useState(true);
  const [isDraggingFile, setIsDraggingFile] = useState(false);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const requireReadyModel = useCallback(() => {
    if (!selectedModel || !selectedModelReady) {
      onModelRequired();
      setError(ASR_MODEL_REQUIRED_ERROR);
      return false;
    }
    return true;
  }, [onModelRequired, selectedModel, selectedModelReady]);

  const handleIncludeTimestampsChange = useCallback(
    (nextValue: boolean) => {
      if (!nextValue) {
        setIncludeTimestamps(false);
        return;
      }

      if (timestampAlignerModelId && timestampAlignerReady) {
        setIncludeTimestamps(true);
        setStreamingEnabled(false);
        return;
      }

      onTimestampAlignerRequired();
      setError(ALIGNER_MODEL_REQUIRED_ERROR);
    },
    [
      onTimestampAlignerRequired,
      timestampAlignerModelId,
      timestampAlignerReady,
    ],
  );

  const handleStreamingEnabledChange = useCallback((nextValue: boolean) => {
    setStreamingEnabled(nextValue);
    if (nextValue) {
      setIncludeTimestamps(false);
    }
  }, []);

  const requireTimestampAligner = useCallback(() => {
    if (!includeTimestamps) {
      return true;
    }
    if (timestampAlignerModelId && timestampAlignerReady) {
      return true;
    }
    onTimestampAlignerRequired();
    setError(ALIGNER_MODEL_REQUIRED_ERROR);
    return false;
  }, [
    includeTimestamps,
    onTimestampAlignerRequired,
    timestampAlignerModelId,
    timestampAlignerReady,
  ]);

  const submitAudio = useCallback(
    async (audioBlob: Blob, options: SubmitAudioOptions = {}) => {
      if (!requireReadyModel() || !requireTimestampAligner()) {
        return;
      }

      setIsSubmitting(true);
      setIsDraggingFile(false);
      setError(null);

      try {
        const sourceFileName =
          options.filename?.trim() || resolveSourceAudioFilename(audioBlob);
        const uploadedBlob = await prepareSpeechTextUploadBlob(audioBlob, 16000);
        const uploadFilename = resolveSpeechTextUploadFilename({
          sourceFileName,
          sourceBlob: audioBlob,
          uploadedBlob,
        });
        const request = {
          audio_file: uploadedBlob,
          audio_filename: uploadFilename,
          model_id: selectedModel || undefined,
          aligner_model_id: includeTimestamps
            ? timestampAlignerModelId || undefined
            : undefined,
          language: selectedLanguage,
          include_timestamps: includeTimestamps,
        };
        const record = streamingEnabled
          ? await new Promise<TranscriptionRecord>((resolve, reject) => {
              let settled = false;

              api.createTranscriptionRecordStream(request, {
                onStart: () => {
                  onStreamingStart?.();
                },
                onCreated: (createdRecord) => {
                  if (settled) {
                    return;
                  }
                  settled = true;
                  resolve(createdRecord);
                },
                onDelta: (delta) => {
                  onStreamingDelta?.(delta);
                },
                onFinal: (finalRecord) => {
                  onStreamingFinal?.(finalRecord);
                },
                onError: (message) => {
                  onStreamingError?.(message);
                  if (settled) {
                    return;
                  }
                  settled = true;
                  reject(new Error(message));
                },
                onDone: () => {
                  onStreamingDone?.();
                  if (settled) {
                    return;
                  }
                  settled = true;
                  reject(
                    new Error(
                      "Transcription started but no record was returned.",
                    ),
                  );
                },
              });
            })
          : await api.createTranscriptionRecord(request);
        await onCreated(record);
        onClose();
      } catch (err) {
        setError(
          err instanceof Error ? err.message : "Failed to create transcription.",
        );
      } finally {
        setIsSubmitting(false);
      }
    },
    [
      includeTimestamps,
      onClose,
      onCreated,
      onStreamingDelta,
      onStreamingDone,
      onStreamingError,
      onStreamingFinal,
      onStreamingStart,
      requireReadyModel,
      requireTimestampAligner,
      selectedLanguage,
      selectedModel,
      streamingEnabled,
      timestampAlignerModelId,
    ],
  );

  const handleSelectedFile = useCallback(
    async (file: File | null | undefined) => {
      if (!file) {
        return;
      }

      await submitAudio(file, {
        filename: file.name,
      });
    },
    [submitAudio],
  );

  const handleFileUpload = useCallback(
    async (event: React.ChangeEvent<HTMLInputElement>) => {
      const file = event.target.files?.[0];
      await handleSelectedFile(file);
      event.target.value = "";
    },
    [handleSelectedFile],
  );

  const handleFileDrop = useCallback(
    async (event: React.DragEvent<HTMLButtonElement>) => {
      event.preventDefault();
      setIsDraggingFile(false);

      const file = event.dataTransfer.files?.[0];
      await handleSelectedFile(file);
    },
    [handleSelectedFile],
  );

  const alignerReadyForUse =
    !includeTimestamps || (!!timestampAlignerModelId && timestampAlignerReady);
  const transcriptionStackReady = selectedModelReady && alignerReadyForUse;
  const isSelectedModelLoading =
    selectedModelStatus === "loading" || selectedModelStatus === "downloading";
  const isAlignerModelLoading =
    includeTimestamps &&
    (timestampAlignerModelStatus === "loading" ||
      timestampAlignerModelStatus === "downloading");
  const readinessStackLoading = isSelectedModelLoading || isAlignerModelLoading;
  const readinessStatusLabel = transcriptionStackReady
    ? "READY"
    : readinessStackLoading
      ? "LOADING"
      : "NOT LOADED";
  const readinessTone = transcriptionStackReady ? "success" : "warning";
  const readinessActionLabel = !selectedModelReady
    ? "Open ASR models"
    : includeTimestamps && !alignerReadyForUse
      ? "Open aligner models"
      : "Open models";
  const readinessActionVariant = "default";
  const readinessActionHandler = onOpenModelManager;
  const loadedModels = useMemo(() => {
    const models: string[] = [];
    if (selectedModelReady && selectedModel) {
      models.push(selectedModel);
    }
    if (timestampAlignerReady && timestampAlignerModelId) {
      const alreadyListed = models.includes(timestampAlignerModelId);
      if (!alreadyListed) {
        models.push(timestampAlignerModelId);
      }
    }
    return models;
  }, [
    selectedModel,
    selectedModelReady,
    timestampAlignerModelId,
    timestampAlignerReady,
  ]);

  useEffect(() => {
    setError((current) => {
      if (!current) {
        return current;
      }
      if (current === ASR_MODEL_REQUIRED_ERROR && selectedModelReady) {
        return null;
      }
      if (
        current === ALIGNER_MODEL_REQUIRED_ERROR &&
        (!includeTimestamps || (!!timestampAlignerModelId && timestampAlignerReady))
      ) {
        return null;
      }
      return current;
    });
  }, [
    includeTimestamps,
    selectedModelReady,
    timestampAlignerModelId,
    timestampAlignerReady,
  ]);

  const modalBody = (
    <>
      <div className="border-b border-[var(--border-muted)] bg-[var(--bg-surface-0)] px-5 py-5 sm:px-6">
        <DialogTitle className="text-xl font-semibold tracking-tight text-[var(--text-primary)]">
          New transcript
        </DialogTitle>
        <DialogDescription className="mt-1 max-w-3xl text-[13px] leading-5 text-[var(--text-muted)]">
          Upload a recording, choose streaming, and open the job page once the
          upload is accepted.
        </DialogDescription>
        <SpeechTextModeSwitch
          selectedMode={selectedMode}
          onSelectMode={onSelectMode}
        />
      </div>

      <div className="grid lg:grid-cols-[minmax(0,1.08fr),minmax(18rem,0.84fr)]">
        <div className="border-b border-[var(--border-muted)] px-5 py-5 sm:px-6 lg:border-b-0 lg:border-r">
          <div className="mb-4 flex items-start justify-between gap-4">
            <div>
              <div className="text-[11px] font-semibold uppercase tracking-[0.16em] text-[var(--text-muted)]">
                Audio source
              </div>
              <h3 className="mt-1.5 text-xl font-semibold tracking-tight text-[var(--text-primary)]">
                Bring in a recording
              </h3>
              <p className="mt-1.5 max-w-lg text-[13px] leading-5 text-[var(--text-muted)]">
                Choose a saved clip from your device and move straight into the dedicated transcript workspace.
              </p>
            </div>
            {isSubmitting ? (
              <div className="inline-flex items-center gap-2 rounded-full border border-[var(--border-muted)] bg-[var(--bg-surface-1)] px-3 py-1.5 text-xs font-medium text-[var(--text-muted)]">
                <Loader2 className="h-3.5 w-3.5 animate-spin" />
                Creating job
              </div>
            ) : null}
          </div>

            <button
              type="button"
              onClick={() => fileInputRef.current?.click()}
              onDragOver={(event) => {
                event.preventDefault();
                if (!isSubmitting) {
                  setIsDraggingFile(true);
                }
              }}
              onDragLeave={() => setIsDraggingFile(false)}
              onDrop={(event) => void handleFileDrop(event)}
              disabled={isSubmitting}
              aria-label="Upload audio file"
              className={`group flex min-h-[13.5rem] w-full flex-col items-center justify-center rounded-[24px] border border-dashed p-5 text-center transition-colors disabled:cursor-not-allowed disabled:opacity-60 ${
                isDraggingFile
                  ? "border-[var(--status-info-text)] bg-[var(--bg-surface-2)]"
                  : "border-[var(--border-strong)] bg-[var(--bg-surface-1)] hover:bg-[var(--bg-surface-2)]"
              }`}
            >
              <div className="flex h-16 w-16 items-center justify-center rounded-full border border-[var(--border-muted)] bg-[var(--bg-surface-0)]">
                <Upload className="h-5.5 w-5.5 text-[var(--text-primary)]" />
              </div>

              <div className="mt-5 space-y-2">
                <div>
                  <div className="text-base font-semibold text-[var(--text-primary)]">
                    Upload audio
                  </div>
                  <div className="mt-1.5 max-w-[18rem] text-[13px] leading-5 text-[var(--text-muted)]">
                    {isDraggingFile
                      ? "Drop the file here to create a transcript job."
                      : "Click to choose a recording or drag it here to upload."}
                  </div>
                </div>

                <div className="text-[11px] font-medium uppercase tracking-[0.14em] text-[var(--text-muted)]">
                  WAV, MP3, M4A, AAC
                </div>
              </div>

              <input
                ref={fileInputRef}
                type="file"
                accept="audio/*"
                className="hidden"
                onChange={(event) => void handleFileUpload(event)}
              />
            </button>

            <div className="mt-3">
              <div className="flex rounded-2xl border border-[var(--border-muted)] bg-[var(--bg-surface-1)] px-3.5 py-3 text-[13px] leading-5 text-[var(--text-muted)]">
                The job opens on its own page as soon as the upload is accepted.
              </div>
            </div>
          </div>

        <div className="px-6 py-5 sm:px-6">
          <div className="space-y-2.5">
              <div className="rounded-2xl border border-[var(--border-muted)] bg-[var(--bg-surface-1)] p-3.5">
                <div className="mb-1.5 text-[11px] font-semibold uppercase tracking-[0.14em] text-[var(--text-muted)]">
                  Language
                </div>
                <Select
                  value={selectedLanguage}
                  onValueChange={setSelectedLanguage}
                  disabled={isSubmitting}
                >
                  <SelectTrigger className="h-10 w-full rounded-2xl border-[var(--border-muted)] bg-[var(--bg-surface-0)] text-sm">
                    <SelectValue placeholder="Language" />
                  </SelectTrigger>
                  <SelectContent>
                    {LANGUAGE_OPTIONS.map((language) => (
                      <SelectItem key={language} value={language}>
                        {language}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>

              <label className="flex items-start justify-between gap-3 rounded-2xl border border-[var(--border-muted)] bg-[var(--bg-surface-1)] p-3.5">
                <div className="min-w-0">
                  <div className="text-sm font-semibold text-[var(--text-primary)]">
                    Include timestamps
                  </div>
                  <div className="mt-0.5 text-[13px] leading-5 text-[var(--text-muted)]">
                    Add word and segment timing when the aligner path is ready.
                  </div>
                </div>
                <div className="relative mt-0.5 shrink-0">
                  <input
                    type="checkbox"
                    checked={includeTimestamps}
                    onChange={(event) =>
                      handleIncludeTimestampsChange(event.target.checked)
                    }
                    className="peer sr-only"
                    disabled={isSubmitting}
                  />
                  <span className="flex h-5 w-5 items-center justify-center rounded-md border border-[var(--border-strong)] bg-[var(--bg-surface-0)] text-white transition peer-checked:border-[var(--status-info-text)] peer-checked:bg-[var(--status-info-text)] peer-focus-visible:outline-none peer-focus-visible:ring-2 peer-focus-visible:ring-ring/45 peer-focus-visible:ring-offset-2 peer-focus-visible:ring-offset-background peer-disabled:opacity-50">
                    <Check className="h-3.5 w-3.5 opacity-0 transition peer-checked:opacity-100" />
                  </span>
                </div>
              </label>

              <label className="flex items-start justify-between gap-3 rounded-2xl border border-[var(--border-muted)] bg-[var(--bg-surface-1)] p-3.5">
                <div className="min-w-0">
                  <div className="text-sm font-semibold text-[var(--text-primary)]">
                    Stream results
                  </div>
                  <div className="mt-0.5 text-[13px] leading-5 text-[var(--text-muted)]">
                    Start the job with live transcript updates instead of waiting
                    for a single final response.
                  </div>
                </div>
                <div className="relative mt-0.5 shrink-0">
                  <input
                    type="checkbox"
                    checked={streamingEnabled}
                    onChange={(event) =>
                      handleStreamingEnabledChange(event.target.checked)
                    }
                    className="peer sr-only"
                    disabled={isSubmitting}
                  />
                  <span className="flex h-5 w-5 items-center justify-center rounded-md border border-[var(--border-strong)] bg-[var(--bg-surface-0)] text-white transition peer-checked:border-[var(--status-info-text)] peer-checked:bg-[var(--status-info-text)] peer-focus-visible:outline-none peer-focus-visible:ring-2 peer-focus-visible:ring-ring/45 peer-focus-visible:ring-offset-2 peer-focus-visible:ring-offset-background peer-disabled:opacity-50">
                    <Check className="h-3.5 w-3.5 opacity-0 transition peer-checked:opacity-100" />
                  </span>
                </div>
              </label>

              <div className="rounded-2xl border border-[var(--border-muted)] bg-[var(--bg-surface-1)] p-3.5">
                <div className="text-[11px] font-semibold uppercase tracking-[0.14em] text-[var(--text-muted)]">
                  Model readiness
                </div>

                <div className="mt-2.5 rounded-2xl border border-[var(--border-muted)] bg-[var(--bg-surface-0)] p-3">
                  <div className="flex items-center justify-between gap-3">
                    <span className="text-xs text-[var(--text-muted)]">
                      Transcription model
                    </span>
                    <StatusBadge tone={readinessTone}>
                      {readinessStatusLabel}
                    </StatusBadge>
                  </div>

                  {loadedModels.length > 0 ? (
                    <div className="mt-3 rounded-lg border border-[var(--border-muted)] bg-[var(--bg-surface-1)] px-2.5 py-2">
                      <div className="space-y-1">
                        {loadedModels.map((model) => (
                          <div
                            key={model}
                            className="truncate text-xs text-[var(--text-primary)]"
                            title={model}
                          >
                            {model}
                          </div>
                        ))}
                      </div>
                    </div>
                  ) : null}

                  <Button
                    type="button"
                    variant={readinessActionVariant}
                    size="sm"
                    className="mt-3 h-9 w-full gap-2"
                    onClick={readinessActionHandler}
                    disabled={isSubmitting}
                  >
                    {readinessActionLabel}
                  </Button>
                </div>
              </div>

              {error ? (
                <div className="rounded-2xl border border-[var(--danger-border)] bg-[var(--danger-bg)] px-3.5 py-2.5 text-[13px] leading-5 text-[var(--danger-text)]">
                  <div className="flex items-start gap-2">
                    <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0" />
                    <p>{error}</p>
                  </div>
                </div>
              ) : null}
            </div>

          <div className="mt-4 flex justify-end">
            <Button
              type="button"
              variant="ghost"
              onClick={onClose}
              disabled={isSubmitting}
            >
              Cancel
            </Button>
          </div>
        </div>
      </div>
    </>
  );

  if (!renderInDialog) {
    return modalBody;
  }

  return (
    <Dialog open={isOpen} onOpenChange={(open) => !open && onClose()}>
      <DialogContent
        className="max-w-[46rem] overflow-hidden border-[var(--border-strong)] bg-[var(--bg-surface-0)] p-0"
        onEscapeKeyDown={(event) => {
          if (blockOutsideDismiss) {
            event.preventDefault();
          }
        }}
        onPointerDownOutside={(event) => {
          if (blockOutsideDismiss) {
            event.preventDefault();
          }
        }}
        onInteractOutside={(event) => {
          if (blockOutsideDismiss) {
            event.preventDefault();
          }
        }}
      >
        {modalBody}
      </DialogContent>
    </Dialog>
  );
}
