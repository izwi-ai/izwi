import { useCallback, useEffect, useRef, useState } from "react";
import {
  AlertTriangle,
  Mic,
  Square,
  Upload,
} from "lucide-react";

import { api, type DiarizationRecord, type ModelInfo } from "@/api";
import {
  CreationDialog,
  CreationDialogFrame,
} from "@/components/CreationDialog";
import {
  CreationBody,
  CreationField,
  ModelRow,
  ModelRowAction,
} from "@/components/creation/CreationParts";
import { cn } from "@/lib/utils";
import {
  clampIntegerDraft,
  formatDraftValue,
} from "@/features/diarization/audioUpload";
import {
  prepareSpeechTextUploadBlob,
  resolveSourceAudioFilename,
  resolveSpeechTextUploadFilename,
} from "@/shared/audioUpload";
import type { UploadProgressInfo } from "@/shared/api/audio";
import { isAbortError } from "@/shared/api/http";
import { AUDIO_FILE_ACCEPT, AUDIO_FILE_HINT } from "@/shared/audioFileAccept";
import { diarizationSpeakerUpperBound, getModelStatusLabel } from "@/features/models/catalog/routeModelCatalog";
import { SpeechTextModeSwitch } from "@/features/speech-text/components/SpeechTextModeSwitch";
import { SpeechTextUploadProgress } from "@/features/speech-text/components/SpeechTextUploadProgress";
import {
  AUTO_LANGUAGE,
  GenerateSummaryOption,
  SpeechTextLanguageField,
  languageHint,
} from "@/features/speech-text/components/SpeechTextOptions";
import type { SpeechTextCreationMode } from "@/features/speech-text/creationMode";
import {
  clampUploadPercent,
  createSpeechTextUploadState,
  createUploadAbortError,
  type SpeechTextUploadState,
} from "@/features/speech-text/uploadProgress";

interface NewDiarizationModalProps {
  isOpen: boolean;
  onClose: () => void;
  renderInDialog?: boolean;
  selectedMode?: SpeechTextCreationMode;
  onSelectMode?: (mode: SpeechTextCreationMode) => void;
  pipelineMode?: "classic" | "granite";
  selectedModel: string | null;
  selectedModelReady: boolean;
  diarizationModels: ModelInfo[];
  onSelectModel: (variant: string) => void;
  pipelineAsrModelId?: string | null;
  pipelineAlignerModelId?: string | null;
  pipelineLlmModelId?: string | null;
  pipelineModelsReady?: boolean;
  /** Whether the summary model is loaded; it joins the stack only when a
   * summary is requested. */
  summaryModelReady?: boolean;
  onModelRequired: () => void;
  onPipelineModelsRequired: () => void;
  /** The stack without the summary model. */
  managedModelCount?: number;
  readyManagedModelCount?: number;
  canUnloadAnyManagedModels?: boolean;
  isManagedModelActionBusy?: boolean;
  onOpenModelManager: () => void;
  onUnloadAllManagedModels: () => void;
  onCreated: (record: DiarizationRecord) => Promise<void> | void;
}

interface SubmitAudioOptions {
  filename?: string;
}

interface ActiveRecording {
  generation: number;
  recorder: MediaRecorder;
  stream: MediaStream;
  chunks: Blob[];
  submitOnStop: boolean;
}

function stopMediaStream(stream: MediaStream): void {
  stream.getTracks().forEach((track) => track.stop());
}

const DIARIZATION_MODEL_REQUIRED_ERROR =
  "Select and load a diarization model before creating a run.";
const CLASSIC_PIPELINE_REQUIRED_ERROR =
  "Load ASR and forced aligner models before diarization.";

export function NewDiarizationModal({
  isOpen,
  onClose,
  renderInDialog = true,
  selectedMode = "diarization",
  onSelectMode,
  pipelineMode = "classic",
  selectedModel,
  selectedModelReady,
  diarizationModels,
  onSelectModel,
  pipelineAsrModelId = null,
  pipelineAlignerModelId = null,
  pipelineLlmModelId = null,
  pipelineModelsReady = true,
  summaryModelReady = false,
  onModelRequired,
  onPipelineModelsRequired,
  managedModelCount = 0,
  readyManagedModelCount = 0,
  canUnloadAnyManagedModels = false,
  isManagedModelActionBusy = false,
  onOpenModelManager,
  onUnloadAllManagedModels,
  onCreated,
}: NewDiarizationModalProps) {
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const activeRecordingRef = useRef<ActiveRecording | null>(null);
  const recordingGenerationRef = useRef(0);
  const uploadAbortRef = useRef<AbortController | null>(null);

  const [minSpeakers, setMinSpeakers] = useState("1");
  const [maxSpeakers, setMaxSpeakers] = useState("4");
  const [minSpeechMs, setMinSpeechMs] = useState("240");
  const [minSilenceMs, setMinSilenceMs] = useState("200");
  const [selectedLanguage, setSelectedLanguage] = useState(AUTO_LANGUAGE);
  const [generateSummary, setGenerateSummary] = useState(false);
  const [isDraggingFile, setIsDraggingFile] = useState(false);
  const [isRecording, setIsRecording] = useState(false);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [uploadState, setUploadState] =
    useState<SpeechTextUploadState | null>(null);
  const isGranitePipeline = pipelineMode === "granite";

  const discardActiveRecording = useCallback(() => {
    recordingGenerationRef.current += 1;
    const activeRecording = activeRecordingRef.current;
    if (!activeRecording) {
      return;
    }

    activeRecordingRef.current = null;
    activeRecording.submitOnStop = false;
    activeRecording.recorder.ondataavailable = null;
    activeRecording.recorder.onstop = null;
    try {
      if (activeRecording.recorder.state !== "inactive") {
        activeRecording.recorder.stop();
      }
    } catch {
      // The stream still needs to be released if the recorder is already closed.
    }
    stopMediaStream(activeRecording.stream);
  }, []);

  useEffect(() => {
    if (!isOpen) {
      discardActiveRecording();
      setIsRecording(false);
    }
  }, [discardActiveRecording, isOpen]);

  useEffect(
    () => () => {
      discardActiveRecording();
    },
    [discardActiveRecording],
  );

  const updateUploadProgress = useCallback((progress: UploadProgressInfo) => {
    setUploadState((current) => {
      if (!current || current.phase === "accepted" || current.phase === "opening") {
        return current;
      }

      return {
        ...current,
        phase: "uploading",
        loadedBytes: progress.loadedBytes,
        totalBytes: progress.totalBytes ?? current.totalBytes,
        percent:
          progress.percent === null
            ? null
            : clampUploadPercent(progress.percent),
      };
    });
  }, []);

  const handleCancelUpload = useCallback(() => {
    discardActiveRecording();
    setIsRecording(false);
    uploadAbortRef.current?.abort();
    uploadAbortRef.current = null;
    setUploadState((current) =>
      current
        ? {
            ...current,
            phase: "cancelled",
            errorMessage: undefined,
          }
        : current,
    );
    setIsSubmitting(false);
  }, [discardActiveRecording]);

  const requireReadyModel = useCallback(() => {
    if (!selectedModel || !selectedModelReady) {
      onModelRequired();
      setError(DIARIZATION_MODEL_REQUIRED_ERROR);
      return false;
    }
    return true;
  }, [onModelRequired, selectedModel, selectedModelReady]);

  const requireReadyPipelineModels = useCallback(() => {
    if (isGranitePipeline) {
      return true;
    }
    if (
      !pipelineAsrModelId ||
      !pipelineAlignerModelId ||
      !pipelineModelsReady
    ) {
      onPipelineModelsRequired();
      setError(CLASSIC_PIPELINE_REQUIRED_ERROR);
      return false;
    }
    return true;
  }, [
    isGranitePipeline,
    onPipelineModelsRequired,
    pipelineAlignerModelId,
    pipelineAsrModelId,
    pipelineModelsReady,
  ]);

  useEffect(() => {
    setError((current) => {
      if (!current) {
        return current;
      }
      if (current === DIARIZATION_MODEL_REQUIRED_ERROR && selectedModelReady) {
        return null;
      }
      if (
        current === CLASSIC_PIPELINE_REQUIRED_ERROR &&
        (isGranitePipeline || pipelineModelsReady)
      ) {
        return null;
      }
      return current;
    });
  }, [isGranitePipeline, pipelineModelsReady, selectedModelReady]);

  const normalizeSettings = useCallback(() => {
    const speakerUpperBound = diarizationSpeakerUpperBound(selectedModel);
    let nextMinSpeakers = clampIntegerDraft(minSpeakers, 1, 1, speakerUpperBound);
    const nextMaxSpeakers = clampIntegerDraft(
      maxSpeakers,
      speakerUpperBound,
      1,
      speakerUpperBound,
    );
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
  }, [maxSpeakers, minSilenceMs, minSpeakers, minSpeechMs, selectedModel]);

  const submitAudio = useCallback(
    async (audioBlob: Blob, options: SubmitAudioOptions = {}) => {
      if (!requireReadyModel() || !requireReadyPipelineModels()) {
        return;
      }

      const captureSettings = normalizeSettings();

      setIsSubmitting(true);
      setIsDraggingFile(false);
      setError(null);
      uploadAbortRef.current = null;

      try {
        const uploadController = new AbortController();
        uploadAbortRef.current = uploadController;
        const fallbackSourceFileName = resolveSpeechTextUploadFilename({
          sourceBlob: audioBlob,
          uploadedBlob: audioBlob,
        });
        const sourceFileName =
          options.filename?.trim() ||
          resolveSourceAudioFilename(audioBlob) ||
          fallbackSourceFileName;
        setUploadState(
          createSpeechTextUploadState(audioBlob, sourceFileName, "preparing"),
        );
        const uploadedBlob = await prepareSpeechTextUploadBlob(
          audioBlob,
          16000,
        );
        if (uploadController.signal.aborted) {
          throw createUploadAbortError();
        }
        const uploadFilename = resolveSpeechTextUploadFilename({
          sourceFileName,
          sourceBlob: audioBlob,
          uploadedBlob,
        });
        setUploadState(
          createSpeechTextUploadState(uploadedBlob, uploadFilename, "uploading"),
        );

        const record = await api.createDiarizationRecord(
          isGranitePipeline
            ? {
                audio_file: uploadedBlob,
                audio_filename: uploadFilename,
                model_id: selectedModel || undefined,
                llm_model_id: pipelineLlmModelId || undefined,
                generate_summary: generateSummary,
                language: languageHint(selectedLanguage),
              }
            : {
                audio_file: uploadedBlob,
                audio_filename: uploadFilename,
                model_id: selectedModel || undefined,
                asr_model_id: pipelineAsrModelId || undefined,
                aligner_model_id: pipelineAlignerModelId || undefined,
                llm_model_id: pipelineLlmModelId || undefined,
                generate_summary: generateSummary,
                language: languageHint(selectedLanguage),
                min_speakers: captureSettings.minSpeakers,
                max_speakers: captureSettings.maxSpeakers,
                min_speech_duration_ms: captureSettings.minSpeechMs,
                min_silence_duration_ms: captureSettings.minSilenceMs,
              },
          {
            signal: uploadController.signal,
            onUploadProgress: updateUploadProgress,
          },
        );

        setUploadState((current) =>
          current
            ? {
                ...current,
                phase: "accepted",
                loadedBytes: current.totalBytes ?? current.loadedBytes,
                percent: 100,
              }
            : current,
        );
        setUploadState((current) =>
          current
            ? {
                ...current,
                phase: "opening",
              }
            : current,
        );

        onClose();
        await Promise.resolve(onCreated(record));
      } catch (err) {
        if (isAbortError(err)) {
          setUploadState((current) =>
            current
              ? {
                  ...current,
                  phase: "cancelled",
                  errorMessage: undefined,
                }
              : current,
          );
        } else {
          const message =
            err instanceof Error ? err.message : "Failed to create diarization.";
          setError(message);
          setUploadState((current) =>
            current
              ? {
                  ...current,
                  phase: "failed",
                  errorMessage: message,
                }
              : current,
          );
        }
      } finally {
        uploadAbortRef.current = null;
        setIsSubmitting(false);
      }
    },
    [
      generateSummary,
      normalizeSettings,
      onClose,
      onCreated,
      isGranitePipeline,
      pipelineAlignerModelId,
      pipelineAsrModelId,
      pipelineLlmModelId,
      requireReadyModel,
      requireReadyPipelineModels,
      selectedLanguage,
      selectedModel,
      updateUploadProgress,
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

  const handleStartRecording = useCallback(async () => {
    if (!requireReadyModel() || !requireReadyPipelineModels()) {
      return;
    }

    setError(null);
    const recordingGeneration = ++recordingGenerationRef.current;
    let stream: MediaStream | null = null;

    try {
      stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      if (recordingGeneration !== recordingGenerationRef.current) {
        stopMediaStream(stream);
        return;
      }

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

      const activeRecording: ActiveRecording = {
        generation: recordingGeneration,
        recorder: mediaRecorder,
        stream,
        chunks: [],
        submitOnStop: true,
      };
      activeRecordingRef.current = activeRecording;

      mediaRecorder.ondataavailable = (event) => {
        if (
          activeRecordingRef.current === activeRecording &&
          event.data.size > 0
        ) {
          activeRecording.chunks.push(event.data);
        }
      };

      mediaRecorder.onstop = async () => {
        if (
          activeRecordingRef.current !== activeRecording ||
          !activeRecording.submitOnStop
        ) {
          return;
        }

        activeRecordingRef.current = null;
        mediaRecorder.ondataavailable = null;
        mediaRecorder.onstop = null;
        stopMediaStream(activeRecording.stream);
        setIsRecording(false);
        const audioBlob = new Blob(activeRecording.chunks, {
          type: mediaRecorder.mimeType || "audio/webm",
        });
        await submitAudio(audioBlob);
      };

      mediaRecorder.start();
      setIsRecording(true);
    } catch {
      const shouldReportError =
        recordingGeneration === recordingGenerationRef.current;
      const activeRecording = activeRecordingRef.current;
      if (activeRecording?.generation === recordingGeneration) {
        discardActiveRecording();
      } else if (stream) {
        stopMediaStream(stream);
      }
      if (shouldReportError) {
        setError("Could not access microphone. Please grant permission.");
      }
    }
  }, [
    discardActiveRecording,
    requireReadyModel,
    requireReadyPipelineModels,
    submitAudio,
  ]);

  const handleStopRecording = useCallback(() => {
    const activeRecording = activeRecordingRef.current;
    if (activeRecording && isRecording) {
      try {
        activeRecording.recorder.stop();
      } catch {
        discardActiveRecording();
        setError("Could not finish the microphone recording. Please try again.");
      }
      setIsRecording(false);
    }
  }, [discardActiveRecording, isRecording]);

  const countsSummaryModel = generateSummary && pipelineLlmModelId != null;
  const stackModelCount = managedModelCount + (countsSummaryModel ? 1 : 0);
  const readyStackModelCount =
    readyManagedModelCount + (countsSummaryModel && summaryModelReady ? 1 : 0);
  const allManagedModelsReady =
    stackModelCount > 0 && readyStackModelCount === stackModelCount;
  const readinessStatusLabel = allManagedModelsReady
    ? "Loaded"
    : isManagedModelActionBusy
      ? "Loading"
      : "Not loaded";
  const readinessTone = allManagedModelsReady ? "success" : "warning";
  const readinessLabel = "Diarization stack";
  const selectedModelInCatalog =
    selectedModel != null &&
    diarizationModels.some((model) => model.variant === selectedModel);
  const pickerOptions = selectedModel != null && !selectedModelInCatalog
    ? [
        { variant: selectedModel, label: `${selectedModel} (not in catalog)` },
        ...diarizationModels.map((model) => ({
          variant: model.variant,
          label: `${model.variant} — ${getModelStatusLabel(model.status)}`,
        })),
      ]
    : diarizationModels.map((model) => ({
        variant: model.variant,
        label: `${model.variant} — ${getModelStatusLabel(model.status)}`,
      }));
  const submissionStatusLabel =
    uploadState?.phase === "preparing"
      ? "Preparing"
      : uploadState?.phase === "uploading"
        ? "Uploading"
        : uploadState?.phase === "accepted" || uploadState?.phase === "opening"
          ? "Opening run"
          : "Creating run";

  const modalBody = (
    <CreationBody
      main={
        <>
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
              className={cn(
                "flex min-h-[11rem] flex-col items-center justify-center gap-2.5 rounded-[12px] border p-5 text-center transition-colors disabled:cursor-not-allowed disabled:opacity-60 border-dashed",
                isDraggingFile
                  ? "border-[var(--text-primary)] bg-[var(--surface-raised)]"
                  : "border-[var(--border-strong)] bg-[var(--surface-panel)] hover:bg-[var(--surface-raised)]",
              )}
            >
              <span className="flex h-11 w-11 items-center justify-center rounded-full border border-[var(--border-default)] bg-[var(--surface-panel-alt)]">
                <Upload className="h-[18px] w-[18px] text-[var(--text-primary)]" />
              </span>
              <span className="text-[14px] font-medium text-[var(--text-primary)]">
                {isDraggingFile ? "Drop to start the run" : "Upload a recording"}
              </span>
              <span className="text-xs text-[var(--text-subtle)]">
                Drop a file or browse · {AUDIO_FILE_HINT}
              </span>
              <input
                ref={fileInputRef}
                type="file"
                accept={AUDIO_FILE_ACCEPT}
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
              className={cn(
                "flex min-h-[11rem] flex-col items-center justify-center gap-2.5 rounded-[12px] border p-5 text-center transition-colors disabled:cursor-not-allowed disabled:opacity-60",
                isRecording
                  ? "border-[var(--status-warning-border)] bg-[var(--status-warning-bg)] text-[var(--status-warning-text)]"
                  : "border-[var(--border-default)] bg-[var(--surface-panel)] hover:bg-[var(--surface-raised)]",
              )}
            >
              <span
                className={cn(
                  "flex h-11 w-11 items-center justify-center rounded-full",
                  isRecording
                    ? "bg-[var(--status-warning-text)] text-white"
                    : "border border-[var(--border-default)] bg-[var(--surface-panel-alt)] text-[var(--text-primary)]",
                )}
              >
                {isRecording ? (
                  <Square className="h-[18px] w-[18px] fill-current" />
                ) : (
                  <Mic className="h-[18px] w-[18px]" />
                )}
              </span>
              <span className="text-[14px] font-medium">
                {isRecording ? "Stop recording" : "Record audio"}
              </span>
              <span className="text-xs text-[var(--text-subtle)]">
                {isRecording
                  ? "Stop to submit the recording."
                  : "Capture a conversation with your microphone."}
              </span>
            </button>
          </div>

          {uploadState ? (
            <SpeechTextUploadProgress
              {...uploadState}
              canCancel={
                isSubmitting &&
                (uploadState.phase === "preparing" ||
                  uploadState.phase === "uploading")
              }
              onCancel={handleCancelUpload}
            />
          ) : null}
        </>
      }
      side={
        <>
          <CreationField label="Diarization model" htmlFor="diarization-model">
            <select
              id="diarization-model"
              aria-label="Diarization model"
              value={selectedModel ?? ""}
              onChange={(event) => onSelectModel(event.target.value)}
              disabled={isSubmitting || isRecording}
              className="flex h-9 w-full rounded-[10px] border border-[var(--border-default)] bg-[var(--surface-panel)] px-3 text-[13.5px] text-[var(--text-primary)]"
            >
              {pickerOptions.map((option) => (
                <option key={option.variant} value={option.variant}>
                  {option.label}
                </option>
              ))}
            </select>
          </CreationField>

          <ModelRow
            name={readinessLabel}
            status={
              isManagedModelActionBusy
                ? "busy"
                : readinessTone === "success"
                  ? "ready"
                  : "not_ready"
            }
            statusLabel={readinessStatusLabel}
            actions={
              <>
                {allManagedModelsReady ? (
                  <ModelRowAction
                    onClick={onUnloadAllManagedModels}
                    disabled={
                      isSubmitting ||
                      isRecording ||
                      isManagedModelActionBusy ||
                      !canUnloadAnyManagedModels
                    }
                    busy={isManagedModelActionBusy}
                    aria-label="Unload Models"
                  >
                    Unload
                  </ModelRowAction>
                ) : null}
                <ModelRowAction
                  onClick={onOpenModelManager}
                  disabled={isSubmitting || isRecording}
                  aria-label="Change diarization models"
                >
                  Change
                </ModelRowAction>
              </>
            }
          />

          <SpeechTextLanguageField
            value={selectedLanguage}
            onValueChange={setSelectedLanguage}
            disabled={isSubmitting || isRecording}
          />

          {!isGranitePipeline ? (
            <>
              <CreationField label="Speakers">
                <div className="grid grid-cols-2 gap-2">
                <label className="flex flex-col gap-1 text-xs text-[var(--text-subtle)]">
                  Min speakers
                  <input
                    aria-label="Min speakers"
                    type="text"
                    inputMode="numeric"
                    value={minSpeakers}
                    onChange={(event) => setMinSpeakers(event.target.value)}
                    onBlur={normalizeSettings}
                    className="flex h-9 w-full rounded-[10px] border border-[var(--border-default)] bg-[var(--surface-panel)] px-3 text-[13.5px] text-[var(--text-primary)]"
                  />
                </label>
                <label className="flex flex-col gap-1 text-xs text-[var(--text-subtle)]">
                  Max speakers
                  <input
                    aria-label="Max speakers"
                    type="text"
                    inputMode="numeric"
                    value={maxSpeakers}
                    onChange={(event) => setMaxSpeakers(event.target.value)}
                    onBlur={normalizeSettings}
                    className="flex h-9 w-full rounded-[10px] border border-[var(--border-default)] bg-[var(--surface-panel)] px-3 text-[13.5px] text-[var(--text-primary)]"
                  />
                </label>
                </div>
              </CreationField>
              <CreationField label="Timing">
                <div className="grid grid-cols-2 gap-2">
                <label className="flex flex-col gap-1 text-xs text-[var(--text-subtle)]">
                  Min speech (ms)
                  <input
                    aria-label="Min speech (ms)"
                    type="text"
                    inputMode="numeric"
                    value={minSpeechMs}
                    onChange={(event) => setMinSpeechMs(event.target.value)}
                    onBlur={normalizeSettings}
                    className="flex h-9 w-full rounded-[10px] border border-[var(--border-default)] bg-[var(--surface-panel)] px-3 text-[13.5px] text-[var(--text-primary)]"
                  />
                </label>
                <label className="flex flex-col gap-1 text-xs text-[var(--text-subtle)]">
                  Min silence (ms)
                  <input
                    aria-label="Min silence (ms)"
                    type="text"
                    inputMode="numeric"
                    value={minSilenceMs}
                    onChange={(event) => setMinSilenceMs(event.target.value)}
                    onBlur={normalizeSettings}
                    className="flex h-9 w-full rounded-[10px] border border-[var(--border-default)] bg-[var(--surface-panel)] px-3 text-[13.5px] text-[var(--text-primary)]"
                  />
                </label>
                </div>
              </CreationField>
            </>
          ) : null}

          <GenerateSummaryOption
            checked={generateSummary}
            onCheckedChange={setGenerateSummary}
            disabled={isSubmitting || isRecording}
          />

          {error ? (
            <div
              role="alert"
              className="flex items-start gap-2 rounded-[10px] border border-[var(--danger-border)] bg-[var(--danger-bg)] p-2.5 text-[13px] text-[var(--danger-text)]"
            >
              <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0" />
              <p>{error}</p>
            </div>
          ) : null}
        </>
      }
    />
  );
  const busy = isSubmitting || isRecording;
  const frameProps = {
    title: "New diarization",
    description:
      "Upload a recording or mic capture, then open the job page once the upload is accepted.",
    header: (
      <SpeechTextModeSwitch
        selectedMode={selectedMode}
        onSelectMode={onSelectMode}
      />
    ),
    onClose,
    footerStart: isSubmitting
      ? `${submissionStatusLabel}…`
      : isRecording
        ? "Recording…"
        : "The run starts when you choose or record audio.",
  };

  if (!renderInDialog) {
    return (
      <CreationDialogFrame {...frameProps} cancelDisabled={busy}>
        {modalBody}
      </CreationDialogFrame>
    );
  }

  return (
    <CreationDialog
      {...frameProps}
      open={isOpen}
      blockDismiss={busy}
      className="max-w-[52rem]"
    >
      {modalBody}
    </CreationDialog>
  );
}
