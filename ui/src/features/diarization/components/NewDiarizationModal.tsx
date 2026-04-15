import { useCallback, useRef, useState } from "react";
import {
  AlertTriangle,
  Loader2,
  Mic,
  Square,
  Upload,
} from "lucide-react";

import { api, type DiarizationRecord } from "@/api";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogTitle,
} from "@/components/ui/dialog";
import { StatusBadge } from "@/components/ui/status-badge";
import {
  clampIntegerDraft,
  formatDraftValue,
  resolveDiarizationUploadFilename,
  resolveSourceAudioFilename,
  transcodeToWav,
} from "@/features/diarization/audioUpload";
import { SpeechTextModeSwitch } from "@/features/speech-text/components/SpeechTextModeSwitch";
import type { SpeechTextCreationMode } from "@/features/speech-text/creationMode";

interface NewDiarizationModalProps {
  isOpen: boolean;
  onClose: () => void;
  renderInDialog?: boolean;
  selectedMode?: SpeechTextCreationMode;
  onSelectMode?: (mode: SpeechTextCreationMode) => void;
  selectedModel: string | null;
  selectedModelReady: boolean;
  pipelineAsrModelId?: string | null;
  pipelineAlignerModelId?: string | null;
  pipelineLlmModelId?: string | null;
  pipelineModelsReady?: boolean;
  onModelRequired: () => void;
  onPipelineModelsRequired: () => void;
  managedModelCount?: number;
  readyManagedModelCount?: number;
  canLoadAnyManagedModels?: boolean;
  canUnloadAnyManagedModels?: boolean;
  isManagedModelActionBusy?: boolean;
  onLoadAllManagedModels: () => void;
  onUnloadAllManagedModels: () => void;
  onCreated: (record: DiarizationRecord) => Promise<void> | void;
}

export function NewDiarizationModal({
  isOpen,
  onClose,
  renderInDialog = true,
  selectedMode = "diarization",
  onSelectMode,
  selectedModel,
  selectedModelReady,
  pipelineAsrModelId = null,
  pipelineAlignerModelId = null,
  pipelineLlmModelId = null,
  pipelineModelsReady = true,
  onModelRequired,
  onPipelineModelsRequired,
  managedModelCount = 0,
  readyManagedModelCount = 0,
  canLoadAnyManagedModels = false,
  canUnloadAnyManagedModels = false,
  isManagedModelActionBusy = false,
  onLoadAllManagedModels,
  onUnloadAllManagedModels,
  onCreated,
}: NewDiarizationModalProps) {
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioChunksRef = useRef<Blob[]>([]);

  const [minSpeakers, setMinSpeakers] = useState("1");
  const [maxSpeakers, setMaxSpeakers] = useState("4");
  const [minSpeechMs, setMinSpeechMs] = useState("240");
  const [minSilenceMs, setMinSilenceMs] = useState("200");
  const [isDraggingFile, setIsDraggingFile] = useState(false);
  const [isRecording, setIsRecording] = useState(false);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const requireReadyModel = useCallback(() => {
    if (!selectedModel || !selectedModelReady) {
      onModelRequired();
      setError("Select and load a diarization model before creating a run.");
      return false;
    }
    return true;
  }, [onModelRequired, selectedModel, selectedModelReady]);

  const requireReadyPipelineModels = useCallback(() => {
    if (
      !pipelineAsrModelId ||
      !pipelineAlignerModelId ||
      !pipelineModelsReady
    ) {
      onPipelineModelsRequired();
      setError("Load ASR and forced aligner models before diarization.");
      return false;
    }
    return true;
  }, [
    onPipelineModelsRequired,
    pipelineAlignerModelId,
    pipelineAsrModelId,
    pipelineModelsReady,
  ]);

  const normalizeSettings = useCallback(() => {
    let nextMinSpeakers = clampIntegerDraft(minSpeakers, 1, 1, 4);
    const nextMaxSpeakers = clampIntegerDraft(maxSpeakers, 4, 1, 4);
    const nextMinSpeechMs = clampIntegerDraft(minSpeechMs, 240, 40, 5000);
    const nextMinSilenceMs = clampIntegerDraft(minSilenceMs, 200, 40, 5000);

    if (nextMinSpeakers > nextMaxSpeakers) {
      nextMinSpeakers = nextMaxSpeakers;
    }

    setMinSpeakers(formatDraftValue(nextMinSpeakers));
    setMaxSpeakers(formatDraftValue(nextMaxSpeakers));
    setMinSpeechMs(formatDraftValue(nextMinSpeechMs));
    setMinSilenceMs(formatDraftValue(nextMinSilenceMs));

    return {
      minSpeakers: nextMinSpeakers,
      maxSpeakers: nextMaxSpeakers,
      minSpeechMs: nextMinSpeechMs,
      minSilenceMs: nextMinSilenceMs,
    };
  }, [maxSpeakers, minSilenceMs, minSpeakers, minSpeechMs]);

  const submitAudio = useCallback(
    async (audioBlob: Blob) => {
      if (!requireReadyModel() || !requireReadyPipelineModels()) {
        return;
      }

      const captureSettings = normalizeSettings();

      setIsSubmitting(true);
      setIsDraggingFile(false);
      setError(null);

      try {
        const sourceFileName = resolveSourceAudioFilename(audioBlob);
        const uploadedBlob = await transcodeToWav(
          audioBlob,
          16000,
          sourceFileName,
        ).catch(() => audioBlob);
        const uploadFilename = resolveDiarizationUploadFilename({
          sourceFileName,
          sourceBlob: audioBlob,
          uploadedBlob,
        });

        const record = await api.createDiarizationRecord({
          audio_file: uploadedBlob,
          audio_filename: uploadFilename,
          model_id: selectedModel || undefined,
          asr_model_id: pipelineAsrModelId || undefined,
          aligner_model_id: pipelineAlignerModelId || undefined,
          llm_model_id: pipelineLlmModelId || undefined,
          min_speakers: captureSettings.minSpeakers,
          max_speakers: captureSettings.maxSpeakers,
          min_speech_duration_ms: captureSettings.minSpeechMs,
          min_silence_duration_ms: captureSettings.minSilenceMs,
        });

        onClose();
        await Promise.resolve(onCreated(record));
      } catch (err) {
        setError(
          err instanceof Error ? err.message : "Failed to create diarization.",
        );
      } finally {
        setIsSubmitting(false);
      }
    },
    [
      normalizeSettings,
      onClose,
      onCreated,
      pipelineAlignerModelId,
      pipelineAsrModelId,
      pipelineLlmModelId,
      requireReadyModel,
      requireReadyPipelineModels,
      selectedModel,
    ],
  );

  const handleSelectedFile = useCallback(
    async (file: File | null | undefined) => {
      if (!file) {
        return;
      }
      await submitAudio(file);
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

  const handleStartRecording = useCallback(async () => {
    if (!requireReadyModel() || !requireReadyPipelineModels()) {
      return;
    }

    setError(null);

    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      let mediaRecorder: MediaRecorder | null = null;
      const mimeCandidates = [
        "audio/webm;codecs=opus",
        "audio/webm",
        "audio/mp4",
      ];
      for (const mimeType of mimeCandidates) {
        if (MediaRecorder.isTypeSupported(mimeType)) {
          mediaRecorder = new MediaRecorder(stream, { mimeType });
          break;
        }
      }
      if (!mediaRecorder) {
        mediaRecorder = new MediaRecorder(stream);
      }
      mediaRecorderRef.current = mediaRecorder;
      audioChunksRef.current = [];

      mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          audioChunksRef.current.push(event.data);
        }
      };

      mediaRecorder.onstop = async () => {
        const audioBlob = new Blob(audioChunksRef.current, {
          type: mediaRecorder?.mimeType || "audio/webm",
        });
        stream.getTracks().forEach((track) => track.stop());
        await submitAudio(audioBlob);
      };

      mediaRecorder.start();
      setIsRecording(true);
    } catch {
      setError("Could not access microphone. Please grant permission.");
    }
  }, [requireReadyModel, requireReadyPipelineModels, submitAudio]);

  const handleStopRecording = useCallback(() => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop();
      setIsRecording(false);
    }
  }, [isRecording]);

  const allManagedModelsReady =
    managedModelCount > 0 && readyManagedModelCount === managedModelCount;
  const readinessStatusLabel = allManagedModelsReady
    ? "READY"
    : isManagedModelActionBusy
      ? "LOADING"
      : "NOT LOADED";
  const readinessTone = allManagedModelsReady ? "success" : "warning";
  const readinessActionIsUnload = allManagedModelsReady;
  const readinessActionLabel = isManagedModelActionBusy
    ? "Loading models..."
    : readinessActionIsUnload
      ? "Unload Models"
      : "Load Models";
  const readinessActionClass = readinessActionIsUnload
    ? "mt-3 h-9 w-full gap-2 border-[var(--danger-border)] bg-[var(--danger-bg)] text-[var(--danger-text)] hover:bg-[var(--danger-bg-hover)] hover:text-[var(--danger-text)]"
    : "mt-3 h-9 w-full gap-2";
  const canRunReadinessAction = readinessActionIsUnload
    ? canUnloadAnyManagedModels
    : canLoadAnyManagedModels;

  const modalBody = (
    <>
      <div className="border-b border-[var(--border-muted)] bg-[var(--bg-surface-0)] px-5 py-5 sm:px-6">
        <DialogTitle className="text-xl font-semibold tracking-tight text-[var(--text-primary)]">
          New diarization
        </DialogTitle>
        <DialogDescription className="mt-1 max-w-3xl text-[13px] leading-5 text-[var(--text-muted)]">
          Upload a recording or mic capture, then open the job page once the
          upload is accepted.
        </DialogDescription>
        <SpeechTextModeSwitch
          selectedMode={selectedMode}
          onSelectMode={onSelectMode}
        />
      </div>

      <div className="grid lg:grid-cols-[minmax(0,1.1fr),minmax(19rem,0.9fr)]">
        <div className="border-b border-[var(--border-muted)] px-5 py-5 sm:px-6 lg:border-b-0 lg:border-r">
          <div className="mb-4 flex items-start justify-between gap-4">
            <div>
              <div className="text-[11px] font-semibold uppercase tracking-[0.16em] text-[var(--text-muted)]">
                Audio source
              </div>
              <h3 className="mt-1.5 text-xl font-semibold tracking-tight text-[var(--text-primary)]">
                Choose how to start
              </h3>
              <p className="mt-1.5 max-w-lg text-[13px] leading-5 text-[var(--text-muted)]">
                Bring in a saved clip or capture a fresh conversation, then
                move straight into the dedicated diarization record page.
              </p>
            </div>
            {isSubmitting ? (
              <div className="inline-flex items-center gap-2 rounded-full border border-[var(--border-muted)] bg-[var(--bg-surface-1)] px-3 py-1.5 text-xs font-medium text-[var(--text-muted)]">
                <Loader2 className="h-3.5 w-3.5 animate-spin" />
                Creating run
              </div>
            ) : null}
          </div>

            <div className="grid gap-3 sm:grid-cols-2">
              <button
                type="button"
                onClick={() => fileInputRef.current?.click()}
                onDragOver={(event) => {
                  event.preventDefault();
                  if (!isSubmitting && !isRecording) {
                    setIsDraggingFile(true);
                  }
                }}
                onDragLeave={() => setIsDraggingFile(false)}
                onDrop={(event) => void handleFileDrop(event)}
                disabled={isSubmitting || isRecording}
                aria-label="Upload audio file"
                className={`group flex min-h-[14rem] flex-col items-center justify-center rounded-[24px] border border-dashed p-5 text-center transition-colors disabled:cursor-not-allowed disabled:opacity-60 ${
                  isDraggingFile
                    ? "border-[var(--status-info-text)] bg-[var(--bg-surface-2)]"
                    : "border-[var(--border-strong)] bg-[var(--bg-surface-1)] hover:bg-[var(--bg-surface-2)]"
                }`}
              >
                <div className="flex h-16 w-16 items-center justify-center rounded-full border border-[var(--border-muted)] bg-[var(--bg-surface-0)]">
                  <Upload className="h-5.5 w-5.5 text-[var(--text-primary)]" />
                </div>

                <div className="mt-5 space-y-2">
                  <div className="text-base font-semibold text-[var(--text-primary)]">
                    Upload audio
                  </div>
                  <div className="max-w-[18rem] text-[13px] leading-5 text-[var(--text-muted)]">
                    {isDraggingFile
                      ? "Drop the file here to create a diarization run."
                      : "Click to choose a recording or drag it here to upload."}
                  </div>
                </div>

                <div className="mt-3 text-[11px] font-medium uppercase tracking-[0.14em] text-[var(--text-muted)]">
                  WAV, MP3, M4A, AAC
                </div>

                <input
                  ref={fileInputRef}
                  type="file"
                  accept="audio/*"
                  className="hidden"
                  onChange={(event) => void handleFileUpload(event)}
                />
              </button>

              <button
                type="button"
                onClick={() => {
                  if (isRecording) {
                    handleStopRecording();
                  } else {
                    void handleStartRecording();
                  }
                }}
                disabled={isSubmitting}
                className={`group flex min-h-[14rem] flex-col items-center justify-center rounded-[24px] border p-5 text-center transition-colors disabled:cursor-not-allowed disabled:opacity-60 ${
                  isRecording
                    ? "border-[var(--status-warning-border)] bg-[var(--status-warning-bg)] text-[var(--status-warning-text)]"
                    : "border-[var(--border-strong)] bg-[var(--bg-surface-1)] hover:bg-[var(--bg-surface-2)]"
                }`}
              >
                <div
                  className={`flex h-16 w-16 items-center justify-center rounded-full ${
                    isRecording
                      ? "bg-[var(--status-warning-text)] text-white"
                      : "border border-[var(--border-muted)] bg-[var(--bg-surface-0)] text-[var(--text-primary)]"
                  }`}
                >
                  {isRecording ? (
                    <Square className="h-5.5 w-5.5 fill-current" />
                  ) : (
                    <Mic className="h-5.5 w-5.5" />
                  )}
                </div>

                <div className="mt-5 space-y-2">
                  <div className="text-base font-semibold">
                    {isRecording ? "Stop recording" : "Record audio"}
                  </div>
                  <div className="max-w-[18rem] text-[13px] leading-5 text-[var(--text-muted)]">
                    {isRecording
                      ? "Stop capture to submit the recording for diarization."
                      : "Use your microphone to capture a fresh conversation."}
                  </div>
                </div>
              </button>
            </div>

            <div className="mt-3 rounded-2xl border border-[var(--border-muted)] bg-[var(--bg-surface-1)] px-3.5 py-3 text-[13px] leading-5 text-[var(--text-muted)]">
              The run opens on its own page as soon as the upload is accepted.
            </div>
          </div>

        <div className="px-6 py-5 sm:px-6">
          <div className="space-y-2.5">
              <div className="rounded-2xl border border-[var(--border-muted)] bg-[var(--bg-surface-1)] p-3.5">
                <div className="mb-2 text-[11px] font-semibold uppercase tracking-[0.14em] text-[var(--text-muted)]">
                  Speaker range
                </div>
                <div className="grid grid-cols-2 gap-3">
                  <label className="space-y-2 text-xs font-medium text-[var(--text-muted)]">
                    <span>Min speakers</span>
                    <input
                      aria-label="Min speakers"
                      type="text"
                      inputMode="numeric"
                      value={minSpeakers}
                      onChange={(event) => setMinSpeakers(event.target.value)}
                      onBlur={normalizeSettings}
                      className="flex h-10 w-full rounded-lg border border-[var(--border-muted)] bg-[var(--bg-surface-0)] px-3 py-1 text-sm text-[var(--text-primary)]"
                    />
                  </label>
                  <label className="space-y-2 text-xs font-medium text-[var(--text-muted)]">
                    <span>Max speakers</span>
                    <input
                      aria-label="Max speakers"
                      type="text"
                      inputMode="numeric"
                      value={maxSpeakers}
                      onChange={(event) => setMaxSpeakers(event.target.value)}
                      onBlur={normalizeSettings}
                      className="flex h-10 w-full rounded-lg border border-[var(--border-muted)] bg-[var(--bg-surface-0)] px-3 py-1 text-sm text-[var(--text-primary)]"
                    />
                  </label>
                </div>
              </div>

              <div className="rounded-2xl border border-[var(--border-muted)] bg-[var(--bg-surface-1)] p-3.5">
                <div className="mb-2 text-[11px] font-semibold uppercase tracking-[0.14em] text-[var(--text-muted)]">
                  Timing windows
                </div>
                <div className="grid grid-cols-2 gap-3">
                  <label className="space-y-2 text-xs font-medium text-[var(--text-muted)]">
                    <span>Min speech (ms)</span>
                    <input
                      aria-label="Min speech (ms)"
                      type="text"
                      inputMode="numeric"
                      value={minSpeechMs}
                      onChange={(event) => setMinSpeechMs(event.target.value)}
                      onBlur={normalizeSettings}
                      className="flex h-10 w-full rounded-lg border border-[var(--border-muted)] bg-[var(--bg-surface-0)] px-3 py-1 text-sm text-[var(--text-primary)]"
                    />
                  </label>
                  <label className="space-y-2 text-xs font-medium text-[var(--text-muted)]">
                    <span>Min silence (ms)</span>
                    <input
                      aria-label="Min silence (ms)"
                      type="text"
                      inputMode="numeric"
                      value={minSilenceMs}
                      onChange={(event) =>
                        setMinSilenceMs(event.target.value)
                      }
                      onBlur={normalizeSettings}
                      className="flex h-10 w-full rounded-lg border border-[var(--border-muted)] bg-[var(--bg-surface-0)] px-3 py-1 text-sm text-[var(--text-primary)]"
                    />
                  </label>
                </div>
              </div>

              <div className="rounded-2xl border border-[var(--border-muted)] bg-[var(--bg-surface-1)] p-3.5">
                <div className="text-[11px] font-semibold uppercase tracking-[0.14em] text-[var(--text-muted)]">
                  Model readiness
                </div>

                <div className="mt-2.5 rounded-2xl border border-[var(--border-muted)] bg-[var(--bg-surface-0)] p-3">
                  <div className="flex items-center justify-between gap-3">
                    <span className="text-xs text-[var(--text-muted)]">
                      Diarization stack
                    </span>
                    <StatusBadge tone={readinessTone}>
                      {readinessStatusLabel}
                    </StatusBadge>
                  </div>

                  <Button
                    type="button"
                    variant={readinessActionIsUnload ? "outline" : "default"}
                    size="sm"
                    className={readinessActionClass}
                    onClick={
                      readinessActionIsUnload
                        ? onUnloadAllManagedModels
                        : onLoadAllManagedModels
                    }
                    disabled={
                      isSubmitting ||
                      isRecording ||
                      isManagedModelActionBusy ||
                      !canRunReadinessAction
                    }
                  >
                    {isManagedModelActionBusy ? (
                      <Loader2 className="h-4 w-4 animate-spin" />
                    ) : null}
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
              disabled={isSubmitting || isRecording}
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
    <Dialog
      open={isOpen}
      onOpenChange={(open) => {
        if (!open && !isSubmitting && !isRecording) {
          onClose();
        }
      }}
    >
      <DialogContent className="max-w-[52rem] overflow-hidden border-[var(--border-strong)] bg-[var(--bg-surface-0)] p-0">
        {modalBody}
      </DialogContent>
    </Dialog>
  );
}
