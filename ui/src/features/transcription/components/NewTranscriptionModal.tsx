import { useCallback, useEffect, useRef, useState } from "react";
import { AlertTriangle, Upload } from "lucide-react";

import {
  api,
  type TranscriptionProcessingProgress,
  type TranscriptionRecord,
} from "@/api";
import {
  CreationDialog,
  CreationDialogFrame,
} from "@/components/CreationDialog";
import {
  CreationBody,
  CreationField,
  CreationOption,
  ModelRow,
  ModelRowAction,
} from "@/components/creation/CreationParts";
import { getModelDisplayName } from "@/features/models/catalog/modelMetadata";
import { cn } from "@/lib/utils";
import {
  prepareSpeechTextUploadBlob,
  resolveSourceAudioFilename,
  resolveSpeechTextUploadFilename,
} from "@/shared/audioUpload";
import type { UploadProgressInfo } from "@/shared/api/audio";
import { isAbortError } from "@/shared/api/http";
import { AUDIO_FILE_ACCEPT, AUDIO_FILE_HINT } from "@/shared/audioFileAccept";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Textarea } from "@/components/ui/textarea";
import { parseKeywords } from "@/features/speech-text/keywords";
import type { SpeechTextCreationMode } from "@/features/speech-text/creationMode";
import { SpeechTextModeSwitch } from "@/features/speech-text/components/SpeechTextModeSwitch";
import { SpeechTextUploadProgress } from "@/features/speech-text/components/SpeechTextUploadProgress";
import {
  AUTO_LANGUAGE,
  GenerateSummaryOption,
  SpeechTextLanguageField,
  languageHint,
} from "@/features/speech-text/components/SpeechTextOptions";
import {
  clampUploadPercent,
  createSpeechTextUploadState,
  createUploadAbortError,
  type SpeechTextUploadState,
} from "@/features/speech-text/uploadProgress";

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
  /** The selected model translates speech on request. */
  selectedModelSupportsTranslation?: boolean;
  /** The selected model accepts a keyword list. */
  selectedModelSupportsKeywords?: boolean;
  /** The selected model times words itself: timestamps need no aligner. */
  selectedModelHasNativeWordTimestamps?: boolean;
  /** Shown under the SAA model: why an installed model is not offered. */
  speakerAttributionHint?: string | null;
  timestampAlignerModelId: string | null;
  timestampAlignerReady: boolean;
  timestampAlignerModelStatus?: string | null;
  onOpenModelManager: () => void;
  onModelRequired: () => void;
  onTimestampAlignerRequired: () => void;
  onCreated: (record: TranscriptionRecord) => Promise<void> | void;
  /** Told while an upload is under way, so a host dialog stays open. */
  onBusyChange?: (busy: boolean) => void;
  onStreamingStart?: () => void;
  onStreamingDelta?: (delta: string) => void;
  onStreamingProgress?: (progress: TranscriptionProcessingProgress) => void;
  onStreamingFinal?: (record: TranscriptionRecord) => void;
  onStreamingError?: (message: string, recordId: string | null) => void;
  onStreamingDone?: (recordId: string | null) => void;
}

interface SubmitAudioOptions {
  filename?: string;
}

const ASR_MODEL_REQUIRED_ERROR =
  "Select and load an ASR model before creating a transcription.";
const SAA_MODEL_REQUIRED_ERROR =
  "Select and load a speaker-attributed ASR model (Granite Speech 4.1 Plus or Nemotron 3.5 Transcribe) before creating speaker-attributed ASR.";
const ALIGNER_MODEL_REQUIRED_ERROR =
  "Load the timestamp aligner model to include timestamps.";
/** Leave the speech in its own language. */
const NO_TRANSLATION = "Off";
/** Languages a translation can target (the server checks the pair). */
const TRANSLATION_TARGETS = [
  "English",
  "French",
  "German",
  "Spanish",
  "Portuguese",
  "Japanese",
  "Italian",
  "Chinese",
] as const;

const SPEAKER_EXPECTATION_OPTIONS = [
  { value: "auto", label: "Auto" },
  { value: "2", label: "2+" },
  { value: "3", label: "3+" },
  { value: "4", label: "4+" },
] as const;

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
  selectedModelSupportsTranslation = false,
  selectedModelSupportsKeywords = false,
  selectedModelHasNativeWordTimestamps = false,
  speakerAttributionHint = null,
  timestampAlignerModelId,
  timestampAlignerReady,
  timestampAlignerModelStatus = null,
  onOpenModelManager,
  onModelRequired,
  onTimestampAlignerRequired,
  onCreated,
  onBusyChange,
  onStreamingStart,
  onStreamingDelta,
  onStreamingProgress,
  onStreamingFinal,
  onStreamingError,
  onStreamingDone,
}: NewTranscriptionModalProps) {
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const uploadAbortRef = useRef<AbortController | null>(null);

  const [selectedLanguage, setSelectedLanguage] = useState(AUTO_LANGUAGE);
  const [translationTarget, setTranslationTarget] = useState(NO_TRANSLATION);
  const [keywordsText, setKeywordsText] = useState("");
  const [includeTimestamps, setIncludeTimestamps] = useState(false);
  const [streamingEnabled, setStreamingEnabled] = useState(true);
  const [generateSummary, setGenerateSummary] = useState(false);
  const [speakerExpectation, setSpeakerExpectation] = useState("2");
  const [isDraggingFile, setIsDraggingFile] = useState(false);
  const [isSubmitting, setIsSubmitting] = useState(false);
  useEffect(() => {
    onBusyChange?.(isSubmitting);
  }, [isSubmitting, onBusyChange]);
  // A dialog that goes away mid-upload stops the upload.
  useEffect(
    () => () => {
      uploadAbortRef.current?.abort();
      onBusyChange?.(false);
    },
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [],
  );
  const [error, setError] = useState<string | null>(null);
  const [uploadState, setUploadState] =
    useState<SpeechTextUploadState | null>(null);
  const isSpeakerAttributedAsrMode = selectedMode === "speaker_attributed_asr";
  const modelRequiredError = isSpeakerAttributedAsrMode
    ? SAA_MODEL_REQUIRED_ERROR
    : ASR_MODEL_REQUIRED_ERROR;
  const modalTitle = isSpeakerAttributedAsrMode
    ? "New speaker-attributed ASR"
    : "New transcript";

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
  }, []);

  // A model that times words itself needs no aligner for timestamps.
  const timestampsNeedAligner = !selectedModelHasNativeWordTimestamps;

  const requireReadyModel = useCallback(() => {
    if (!selectedModel || !selectedModelReady) {
      onModelRequired();
      setError(modelRequiredError);
      return false;
    }
    return true;
  }, [modelRequiredError, onModelRequired, selectedModel, selectedModelReady]);

  const handleIncludeTimestampsChange = useCallback(
    (nextValue: boolean) => {
      if (!nextValue) {
        setIncludeTimestamps(false);
        return;
      }

      if (
        !timestampsNeedAligner ||
        (timestampAlignerModelId && timestampAlignerReady)
      ) {
        setIncludeTimestamps(true);
        return;
      }

      onTimestampAlignerRequired();
      setError(ALIGNER_MODEL_REQUIRED_ERROR);
    },
    [
      onTimestampAlignerRequired,
      timestampAlignerModelId,
      timestampAlignerReady,
      timestampsNeedAligner,
    ],
  );

  const handleStreamingEnabledChange = useCallback((nextValue: boolean) => {
    setStreamingEnabled(nextValue);
  }, []);

  const translating =
    selectedModelSupportsTranslation && translationTarget !== NO_TRANSLATION;
  const streamingUnavailableReason = includeTimestamps
    ? "Streaming is unavailable while timestamps are enabled. Turn timestamps off to stream."
    : null;
  // A translation's words are not the audio's, so neither the aligner nor
  // the model can time them; the server refuses the combination.
  const timestampsUnavailableReason = translating
    ? "Timestamps are unavailable for a translation. Turn translation off to add timestamps."
    : streamingEnabled
      ? "Timestamps are unavailable while streaming is enabled. Turn streaming off to add timestamps."
      : null;
  const translationUnavailableReason = includeTimestamps
    ? "Translation is unavailable while timestamps are enabled. Turn timestamps off to translate."
    : null;

  const requireTimestampAligner = useCallback(() => {
    if (isSpeakerAttributedAsrMode) {
      return true;
    }
    if (!includeTimestamps || !timestampsNeedAligner) {
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
    isSpeakerAttributedAsrMode,
    onTimestampAlignerRequired,
    timestampAlignerModelId,
    timestampAlignerReady,
    timestampsNeedAligner,
  ]);

  useEffect(() => {
    if (!isSpeakerAttributedAsrMode) {
      return;
    }
    setIncludeTimestamps(false);
    setStreamingEnabled(false);
  }, [isSpeakerAttributedAsrMode]);

  const submitAudio = useCallback(
    async (audioBlob: Blob, options: SubmitAudioOptions = {}) => {
      if (!requireReadyModel() || !requireTimestampAligner()) {
        return;
      }

      setIsSubmitting(true);
      setIsDraggingFile(false);
      setError(null);
      uploadAbortRef.current = null;

      try {
        const uploadController = new AbortController();
        uploadAbortRef.current = uploadController;
        const sourceFileName =
          options.filename?.trim() ||
          resolveSourceAudioFilename(audioBlob) ||
          "audio";
        setUploadState(
          createSpeechTextUploadState(audioBlob, sourceFileName, "preparing"),
        );
        const uploadedBlob = await prepareSpeechTextUploadBlob(audioBlob, 16000);
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
        const keywords = selectedModelSupportsKeywords
          ? parseKeywords(keywordsText)
          : [];
        const request = {
          audio_file: uploadedBlob,
          audio_filename: uploadFilename,
          model_id: selectedModel || undefined,
          aligner_model_id:
            includeTimestamps && timestampsNeedAligner
              ? timestampAlignerModelId || undefined
              : undefined,
          language: languageHint(selectedLanguage),
          target_language: translating ? translationTarget : undefined,
          keywords: keywords.length > 0 ? keywords : undefined,
          include_timestamps: includeTimestamps,
          generate_summary: generateSummary,
        };
        let streamingCreatedHandled: Promise<void> | null = null;
        const record = isSpeakerAttributedAsrMode
          ? await api.createSpeakerAttributedAsrRecord(
              {
                audio_file: uploadedBlob,
                audio_filename: uploadFilename,
                model_id: selectedModel || undefined,
                language: languageHint(selectedLanguage),
                generate_summary: generateSummary,
                min_speakers:
                  speakerExpectation === "auto"
                    ? undefined
                    : Number(speakerExpectation),
              },
              {
                signal: uploadController.signal,
                onUploadProgress: updateUploadProgress,
              },
            )
          : streamingEnabled
            ? await new Promise<TranscriptionRecord>((resolve, reject) => {
              let settled = false;
              let createdRecordId: string | null = null;
              const handleCreated = (createdRecord: TranscriptionRecord) => {
                if (settled) {
                  return;
                }
                settled = true;
                createdRecordId = createdRecord.id;
                try {
                  streamingCreatedHandled = Promise.resolve(
                    onCreated(createdRecord),
                  );
                } catch (err) {
                  streamingCreatedHandled = Promise.reject(err);
                }
                resolve(createdRecord);
              };
              const rejectIfAborted = () => {
                if (settled) {
                  return;
                }
                settled = true;
                reject(createUploadAbortError());
              };

              const streamController = api.createTranscriptionRecordStream(request, {
                onUploadProgress: updateUploadProgress,
                onStart: () => {
                  onStreamingStart?.();
                },
                onCreated: handleCreated,
                onDelta: (delta) => {
                  onStreamingDelta?.(delta);
                },
                onProgress: (progress) => {
                  onStreamingProgress?.(progress);
                },
                onFinal: (finalRecord) => {
                  onStreamingFinal?.(finalRecord);
                },
                onError: (message) => {
                  onStreamingError?.(message, createdRecordId);
                  if (settled) {
                    return;
                  }
                  settled = true;
                  reject(new Error(message));
                },
                onDone: () => {
                  onStreamingDone?.(createdRecordId);
                  if (settled) {
                    return;
                  }
                  if (uploadAbortRef.current?.signal.aborted) {
                    settled = true;
                    reject(createUploadAbortError());
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
              uploadAbortRef.current = streamController;
              streamController.signal.addEventListener("abort", rejectIfAborted, {
                once: true,
              });
            })
            : await (() => {
                return api.createTranscriptionRecord(request, {
                signal: uploadController.signal,
                onUploadProgress: updateUploadProgress,
              });
            })();
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
        if (streamingCreatedHandled) {
          await streamingCreatedHandled;
        } else {
          await onCreated(record);
        }
        onClose();
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
            err instanceof Error
              ? err.message
              : "Failed to create transcription.";
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
      includeTimestamps,
      isSpeakerAttributedAsrMode,
      generateSummary,
      keywordsText,
      selectedModelSupportsKeywords,
      translating,
      translationTarget,
      onClose,
      onCreated,
      onStreamingDelta,
      onStreamingDone,
      onStreamingError,
      onStreamingFinal,
      onStreamingProgress,
      onStreamingStart,
      requireReadyModel,
      requireTimestampAligner,
      selectedLanguage,
      selectedModel,
      speakerExpectation,
      streamingEnabled,
      timestampAlignerModelId,
      timestampsNeedAligner,
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

  const isSelectedModelLoading =
    selectedModelStatus === "loading" || selectedModelStatus === "downloading";
  const isAlignerModelLoading =
    !isSpeakerAttributedAsrMode &&
    includeTimestamps &&
    timestampsNeedAligner &&
    (timestampAlignerModelStatus === "loading" ||
      timestampAlignerModelStatus === "downloading");
  const readinessActionHandler = onOpenModelManager;

  useEffect(() => {
    setError((current) => {
      if (!current) {
        return current;
      }
      if (
        (current === ASR_MODEL_REQUIRED_ERROR ||
          current === SAA_MODEL_REQUIRED_ERROR) &&
        selectedModelReady
      ) {
        return null;
      }
      if (
        current === ALIGNER_MODEL_REQUIRED_ERROR &&
        (!includeTimestamps ||
          !timestampsNeedAligner ||
          (!!timestampAlignerModelId && timestampAlignerReady))
      ) {
        return null;
      }
      return current;
    });
  }, [
    includeTimestamps,
    isSpeakerAttributedAsrMode,
    selectedModelReady,
    timestampAlignerModelId,
    timestampAlignerReady,
    timestampsNeedAligner,
  ]);

  const submissionStatusLabel =
    uploadState?.phase === "preparing"
      ? "Preparing"
      : uploadState?.phase === "uploading"
        ? "Uploading"
        : uploadState?.phase === "accepted" || uploadState?.phase === "opening"
          ? "Opening job"
          : "Creating job";

  const modelRowStatus = (ready: boolean, loading: boolean, modelId: string | null) =>
    !modelId ? "missing" : ready ? "ready" : loading ? "busy" : "not_ready";
  const modelRowLabel = (ready: boolean, loading: boolean, modelId: string | null) =>
    !modelId
      ? "Choose one in Models"
      : ready
        ? "Loaded"
        : loading
          ? "Loading…"
          : "Not loaded";
  const modalBody = (
    <CreationBody
      main={
        <>
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
            className={cn(
              "group flex min-h-52 w-full flex-col items-center justify-center gap-3 rounded-[12px] border border-dashed p-5 text-center transition-colors disabled:cursor-not-allowed disabled:opacity-60",
              isDraggingFile
                ? "border-(--text-primary) bg-(--surface-raised)"
                : "border-(--border-strong) bg-(--surface-panel) hover:bg-(--surface-raised)",
            )}
          >
            <span className="flex h-11 w-11 items-center justify-center rounded-full border border-(--border-default) bg-(--surface-panel-alt)">
              <Upload className="h-[18px] w-[18px] text-(--text-primary)" />
            </span>
            <span className="text-[14px] font-medium text-(--text-primary)">
              {isDraggingFile
                ? "Drop to start the transcript"
                : "Drop a recording here or browse"}
            </span>
            <span className="text-xs text-(--text-subtle)">
              {AUDIO_FILE_HINT}
            </span>
            <input
              ref={fileInputRef}
              type="file"
              accept={AUDIO_FILE_ACCEPT}
              className="hidden"
              onChange={(event) => void handleFileUpload(event)}
            />
          </button>

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
          <SpeechTextLanguageField
            value={selectedLanguage}
            onValueChange={setSelectedLanguage}
            disabled={isSubmitting}
          />

          {isSpeakerAttributedAsrMode ? (
            <CreationField label="Expected speakers">
              <Select
                value={speakerExpectation}
                onValueChange={setSpeakerExpectation}
                disabled={isSubmitting}
              >
                <SelectTrigger aria-label="Expected speakers" className="h-9 w-full rounded-[10px] border-(--border-default) bg-(--surface-panel) text-[13.5px]">
                  <SelectValue placeholder="Expected speakers" />
                </SelectTrigger>
                <SelectContent>
                  {SPEAKER_EXPECTATION_OPTIONS.map((option) => (
                    <SelectItem key={option.value} value={option.value}>
                      {option.label}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </CreationField>
          ) : null}

          <CreationField
            label={isSpeakerAttributedAsrMode ? "SAA model" : "Transcription model"}
            hint={isSpeakerAttributedAsrMode ? speakerAttributionHint : null}
          >
            <ModelRow
              name={
                selectedModel
                  ? getModelDisplayName(selectedModel)
                  : "No model selected"
              }
              status={modelRowStatus(
                selectedModelReady,
                isSelectedModelLoading,
                selectedModel,
              )}
              statusLabel={modelRowLabel(
                selectedModelReady,
                isSelectedModelLoading,
                selectedModel,
              )}
              actions={
                <ModelRowAction
                  onClick={readinessActionHandler}
                  disabled={isSubmitting}
                  aria-label={
                    isSpeakerAttributedAsrMode
                      ? "Change SAA model"
                      : "Change transcription model"
                  }
                >
                  Change
                </ModelRowAction>
              }
            />
          </CreationField>

          {!isSpeakerAttributedAsrMode && selectedModelSupportsTranslation ? (
            <CreationField
              label="Translate to"
              hint={translationUnavailableReason}
            >
              <Select
                value={translationTarget}
                onValueChange={setTranslationTarget}
                disabled={isSubmitting || includeTimestamps}
              >
                <SelectTrigger
                  aria-label="Translate to"
                  className="h-9 w-full rounded-[10px] border-(--border-default) bg-(--surface-panel) text-[13.5px]"
                >
                  <SelectValue placeholder="Translate to" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value={NO_TRANSLATION}>
                    Off (transcribe)
                  </SelectItem>
                  {TRANSLATION_TARGETS.map((language) => (
                    <SelectItem key={language} value={language}>
                      {language}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </CreationField>
          ) : null}

          {!isSpeakerAttributedAsrMode && selectedModelSupportsKeywords ? (
            <CreationField label="Keywords">
              <Textarea
                aria-label="Keywords"
                value={keywordsText}
                onChange={(event) => setKeywordsText(event.target.value)}
                disabled={isSubmitting}
                placeholder="Names, products, acronyms — one per line or comma-separated"
                className="min-h-[72px] text-[13.5px]"
              />
            </CreationField>
          ) : null}

          {!isSpeakerAttributedAsrMode ? (
            <>
              <CreationOption
                title="Include timestamps"
                description={
                  timestampsUnavailableReason ??
                  (timestampsNeedAligner
                    ? "Segment and word times for review and subtitles."
                    : "Segment and word times. This model times words itself, so no aligner is needed.")
                }
                checked={includeTimestamps}
                onCheckedChange={handleIncludeTimestampsChange}
                disabled={isSubmitting || streamingEnabled || translating}
              />
              {includeTimestamps && timestampsNeedAligner ? (
                <CreationField label="Timestamp aligner">
                  <ModelRow
                    name={
                      timestampAlignerModelId
                        ? getModelDisplayName(timestampAlignerModelId)
                        : "No aligner selected"
                    }
                    status={modelRowStatus(
                      timestampAlignerReady,
                      isAlignerModelLoading,
                      timestampAlignerModelId,
                    )}
                    statusLabel={modelRowLabel(
                      timestampAlignerReady,
                      isAlignerModelLoading,
                      timestampAlignerModelId,
                    )}
                    actions={
                      <ModelRowAction
                        onClick={readinessActionHandler}
                        disabled={isSubmitting}
                        aria-label="Change aligner model"
                      >
                        Change
                      </ModelRowAction>
                    }
                  />
                </CreationField>
              ) : null}
              <CreationOption
                title="Stream results"
                description={
                  streamingUnavailableReason ?? "See text as it is recognised."
                }
                checked={streamingEnabled}
                onCheckedChange={handleStreamingEnabledChange}
                disabled={isSubmitting || includeTimestamps}
              />
            </>
          ) : null}

          <GenerateSummaryOption
            checked={generateSummary}
            onCheckedChange={setGenerateSummary}
            disabled={isSubmitting}
          />

          {error ? (
            <div
              role="alert"
              className="flex items-start gap-2 rounded-[10px] border border-(--danger-border) bg-(--danger-bg) p-2.5 text-[13px] text-(--danger-text)"
            >
              <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0" />
              <p>{error}</p>
            </div>
          ) : null}
        </>
      }
    />
  );
  const frameProps = {
    title: modalTitle,
    description:
      "Upload a recording, choose streaming, and open the job page once the upload is accepted.",
    header: (
      <SpeechTextModeSwitch
        selectedMode={selectedMode}
        onSelectMode={onSelectMode}
        disabled={isSubmitting}
      />
    ),
    onClose,
    footerStart: isSubmitting
      ? `${submissionStatusLabel}…`
      : "The job starts when you choose a file.",
  };

  if (!renderInDialog) {
    return (
      <CreationDialogFrame {...frameProps} cancelDisabled={isSubmitting}>
        {modalBody}
      </CreationDialogFrame>
    );
  }

  return (
    <CreationDialog
      {...frameProps}
      open={isOpen}
      blockDismiss={blockOutsideDismiss || isSubmitting}
      className="max-w-184"
    >
      {modalBody}
    </CreationDialog>
  );
}
