import { useCallback, useEffect, useMemo, useState } from "react";
import { useNavigate, useParams } from "react-router-dom";
import {
  api,
  type DiarizationRecord,
  type ModelInfo,
  type SpeechTextJobSummary,
  type TranscriptionRecord,
} from "@/api";
import { PageHeader, PageShell } from "@/components/PageShell";
import { Button } from "@/components/ui/button";
import { VIEW_CONFIGS } from "@/types";
import { NewTranscriptionModal } from "@/features/transcription/components/NewTranscriptionModal";
import { NewDiarizationModal } from "@/features/diarization/components/NewDiarizationModal";
import { TranscriptionHistoryTable } from "@/features/transcription/components/TranscriptionHistoryTable";
import { TranscriptionRecordDetail } from "@/features/transcription/components/TranscriptionRecordDetail";
import {
  DIARIZATION_PREFERRED_ALIGNER_MODELS,
  DIARIZATION_PREFERRED_ASR_MODELS,
  DIARIZATION_PREFERRED_MODELS,
  DIARIZATION_PREFERRED_SUMMARY_MODELS,
  TRANSCRIPTION_PREFERRED_MODELS,
  resolvePreferredRouteModel,
} from "@/features/models/catalog/routeModelCatalog";
import { RouteModelModal } from "@/features/models/components/RouteModelModal";
import { useRouteModelSelection } from "@/features/models/hooks/useRouteModelSelection";
import type { SpeechTextCreationMode } from "@/features/speech-text/creationMode";
import { useTranscriptionHistory } from "@/features/transcription/hooks/useTranscriptionHistory";
import { useTranscriptionRecord } from "@/features/transcription/hooks/useTranscriptionRecord";
import { normalizeProcessingStatus } from "@/features/transcription/playground/support";
import { Settings2 } from "lucide-react";

const TRANSCRIPTION_PREFERRED_SUMMARY_MODELS = ["Qwen3.5-4B"] as const;

function isTranscriptionAlignerVariant(variant: string): boolean {
  return variant === "Qwen3-ForcedAligner-0.6B";
}

function isTranscriptionSummaryVariant(variant: string): boolean {
  return variant === "Qwen3.5-4B";
}

function isDiarizationVariant(variant: string): boolean {
  const normalized = variant.toLowerCase();
  return normalized.includes("sortformer") || normalized.includes("diar");
}

function isDiarizationPipelineAsrVariant(variant: string): boolean {
  return variant === "Whisper-Large-v3-Turbo";
}

function isDiarizationPipelineAlignerVariant(variant: string): boolean {
  return variant === "Qwen3-ForcedAligner-0.6B";
}

function isDiarizationPipelineLlmVariant(variant: string): boolean {
  return variant === "Qwen3.5-4B";
}

interface TranscriptionPageProps {
  models: ModelInfo[];
  selectedModel: string | null;
  loading: boolean;
  downloadProgress: Record<
    string,
    {
      percent: number;
      currentFile: string;
      status: string;
      downloadedBytes: number;
      totalBytes: number;
    }
  >;
  onDownload: (variant: string) => void;
  onCancelDownload?: (variant: string) => void;
  onLoad: (variant: string) => void;
  onUnload: (variant: string) => void;
  onDelete: (variant: string) => void;
  onSelect: (variant: string) => void;
  onError: (message: string) => void;
}

export function TranscriptionPage({
  models,
  selectedModel,
  loading,
  downloadProgress,
  onDownload,
  onCancelDownload,
  onLoad,
  onUnload,
  onDelete,
  onSelect,
  onError,
}: TranscriptionPageProps) {
  const { recordId } = useParams<{ recordId: string }>();
  const navigate = useNavigate();
  const [isNewTranscriptionModalOpen, setIsNewTranscriptionModalOpen] =
    useState(false);
  const [newSpeechTextMode, setNewSpeechTextMode] =
    useState<SpeechTextCreationMode>("transcription");
  const [recordActionError, setRecordActionError] = useState<string | null>(null);
  const [recordDeletePending, setRecordDeletePending] = useState(false);
  const [recordSummaryRefreshPending, setRecordSummaryRefreshPending] =
    useState(false);
  const [recordSummaryRefreshError, setRecordSummaryRefreshError] = useState<
    string | null
  >(null);
  const [streamingRecord, setStreamingRecord] =
    useState<TranscriptionRecord | null>(null);
  const viewConfig = VIEW_CONFIGS.transcription;
  const transcriptionAlignerModels = useMemo(
    () =>
      models
        .filter((model) => isTranscriptionAlignerVariant(model.variant))
        .sort((a, b) => a.variant.localeCompare(b.variant)),
    [models],
  );
  const transcriptionSummaryModels = useMemo(
    () =>
      models
        .filter((model) => isTranscriptionSummaryVariant(model.variant))
        .sort((a, b) => a.variant.localeCompare(b.variant)),
    [models],
  );
  const diarizationModels = useMemo(
    () =>
      models
        .filter((model) => isDiarizationVariant(model.variant))
        .sort((a, b) => a.variant.localeCompare(b.variant)),
    [models],
  );
  const diarizationAsrPipelineModels = useMemo(
    () =>
      models
        .filter((model) => isDiarizationPipelineAsrVariant(model.variant))
        .sort((a, b) => a.variant.localeCompare(b.variant)),
    [models],
  );
  const diarizationAlignerPipelineModels = useMemo(
    () =>
      models
        .filter((model) => isDiarizationPipelineAlignerVariant(model.variant))
        .sort((a, b) => a.variant.localeCompare(b.variant)),
    [models],
  );
  const diarizationLlmPipelineModels = useMemo(
    () =>
      models
        .filter((model) => isDiarizationPipelineLlmVariant(model.variant))
        .sort((a, b) => a.variant.localeCompare(b.variant)),
    [models],
  );
  const {
    routeModels: transcriptionModels,
    resolvedSelectedModel,
    selectedModelReady,
    isModelModalOpen,
    intentVariant,
    closeModelModal,
    openModelManager,
    requestModel,
  } = useRouteModelSelection({
    models,
    selectedModel,
    onSelect,
    modelFilter: viewConfig.modelFilter,
    resolveSelectedModel: (routeModels, currentModel) =>
      resolvePreferredRouteModel({
        models: routeModels,
        selectedModel: currentModel,
        preferredVariants: TRANSCRIPTION_PREFERRED_MODELS,
        preferAnyPreferredBeforeReadyAny: true,
      }),
  });
  const resolvedSummaryModel = useMemo(
    () =>
      resolvePreferredRouteModel({
        models: transcriptionSummaryModels,
        selectedModel: null,
        preferredVariants: TRANSCRIPTION_PREFERRED_SUMMARY_MODELS,
        preferAnyPreferredBeforeReadyAny: true,
      }),
    [transcriptionSummaryModels],
  );
  const summaryModelStatus = useMemo(() => {
    if (!resolvedSummaryModel) {
      return null;
    }
    return (
      transcriptionSummaryModels.find(
        (model) => model.variant === resolvedSummaryModel,
      )?.status ?? null
    );
  }, [resolvedSummaryModel, transcriptionSummaryModels]);
  const summaryModelReady =
    resolvedSummaryModel != null &&
    transcriptionSummaryModels.some(
      (model) =>
        model.variant === resolvedSummaryModel && model.status === "ready",
    );
  const summaryModelRequirementMessage = useMemo(() => {
    const modelName = resolvedSummaryModel || "Qwen3.5-4B";
    switch (summaryModelStatus) {
      case "downloaded":
        return `Load ${modelName} in Transcription Models to generate summaries.`;
      case "downloading":
        return `${modelName} is downloading. Wait for download to complete, then generate the summary.`;
      case "loading":
        return `${modelName} is loading. Wait for it to become ready, then generate the summary.`;
      case "not_downloaded":
      case "error":
      default:
        return `Download and load ${modelName} in Transcription Models to generate summaries.`;
    }
  }, [resolvedSummaryModel, summaryModelStatus]);
  const resolvedDiarizationModel = useMemo(
    () =>
      resolvePreferredRouteModel({
        models: diarizationModels,
        selectedModel,
        preferredVariants: DIARIZATION_PREFERRED_MODELS,
        preferAnyPreferredBeforeReadyAny: true,
      }),
    [diarizationModels, selectedModel],
  );
  const diarizationModelReady =
    resolvedDiarizationModel != null &&
    diarizationModels.some(
      (model) =>
        model.variant === resolvedDiarizationModel && model.status === "ready",
    );
  const resolvedDiarizationAsrModel = useMemo(
    () =>
      resolvePreferredRouteModel({
        models: diarizationAsrPipelineModels,
        selectedModel: null,
        preferredVariants: DIARIZATION_PREFERRED_ASR_MODELS,
        preferAnyPreferredBeforeReadyAny: true,
      }),
    [diarizationAsrPipelineModels],
  );
  const resolvedDiarizationAlignerModel = useMemo(
    () =>
      resolvePreferredRouteModel({
        models: diarizationAlignerPipelineModels,
        selectedModel: null,
        preferredVariants: DIARIZATION_PREFERRED_ALIGNER_MODELS,
        preferAnyPreferredBeforeReadyAny: true,
      }),
    [diarizationAlignerPipelineModels],
  );
  const resolvedDiarizationLlmModel = useMemo(
    () =>
      resolvePreferredRouteModel({
        models: diarizationLlmPipelineModels,
        selectedModel: null,
        preferredVariants: DIARIZATION_PREFERRED_SUMMARY_MODELS,
        preferAnyPreferredBeforeReadyAny: true,
      }),
    [diarizationLlmPipelineModels],
  );
  const diarizationAsrModelReady =
    resolvedDiarizationAsrModel != null &&
    diarizationAsrPipelineModels.some(
      (model) =>
        model.variant === resolvedDiarizationAsrModel && model.status === "ready",
    );
  const diarizationAlignerModelReady =
    resolvedDiarizationAlignerModel != null &&
    diarizationAlignerPipelineModels.some(
      (model) =>
        model.variant === resolvedDiarizationAlignerModel &&
        model.status === "ready",
    );
  const diarizationPipelineModelsReady =
    diarizationAsrModelReady && diarizationAlignerModelReady;
  const diarizationManagedModelVariants = useMemo(
    () =>
      Array.from(
        new Set(
          [
            resolvedDiarizationModel,
            resolvedDiarizationAsrModel,
            resolvedDiarizationAlignerModel,
            resolvedDiarizationLlmModel,
          ].filter((variant): variant is string => Boolean(variant)),
        ),
      ),
    [
      resolvedDiarizationAlignerModel,
      resolvedDiarizationAsrModel,
      resolvedDiarizationLlmModel,
      resolvedDiarizationModel,
    ],
  );
  const diarizationPipelineModels = useMemo(
    () => [
      ...diarizationModels,
      ...diarizationAsrPipelineModels,
      ...diarizationAlignerPipelineModels,
      ...diarizationLlmPipelineModels,
    ],
    [
      diarizationAlignerPipelineModels,
      diarizationAsrPipelineModels,
      diarizationLlmPipelineModels,
      diarizationModels,
    ],
  );
  const diarizationManagedModels = useMemo(
    () =>
      diarizationManagedModelVariants
        .map((variant) =>
          diarizationPipelineModels.find((model) => model.variant === variant) ??
          null,
        )
        .filter((model): model is ModelInfo => model !== null),
    [diarizationManagedModelVariants, diarizationPipelineModels],
  );
  const readyDiarizationManagedModelCount = diarizationManagedModels.filter(
    (model) => model.status === "ready",
  ).length;
  const canLoadAnyDiarizationManagedModels = diarizationManagedModels.some(
    (model) =>
      model.status === "downloaded" ||
      model.status === "not_downloaded" ||
      model.status === "error",
  );
  const canUnloadAnyDiarizationManagedModels = diarizationManagedModels.some(
    (model) => model.status === "ready",
  );
  const isDiarizationManagedModelActionBusy = diarizationManagedModels.some(
    (model) => model.status === "loading" || model.status === "downloading",
  );
  const modelSections = useMemo(
    () => [
      {
        key: "asr",
        title: "ASR Models",
        description: "Speech-to-text models available for transcription.",
        models: transcriptionModels,
      },
      {
        key: "aligner",
        title: "Timestamp Aligners",
        description:
          "Optional forced aligner models used when timestamped transcription is enabled.",
        models: transcriptionAlignerModels,
      },
      {
        key: "summary",
        title: "Summary Models",
        description:
          "Model used to generate AI summaries for completed transcriptions.",
        models: transcriptionSummaryModels,
      },
    ],
    [transcriptionAlignerModels, transcriptionModels, transcriptionSummaryModels],
  );
  const {
    records,
    loading: historyLoading,
    loadingMore: historyLoadingMore,
    error: historyError,
    hasMoreRecords,
    loadMoreRecords,
    refresh: refreshHistory,
  } = useTranscriptionHistory();
  const {
    record,
    loading: recordLoading,
    error: recordError,
    refresh: refreshRecord,
  } = useTranscriptionRecord(recordId);

  useEffect(() => {
    if (!recordId) {
      setStreamingRecord(null);
      return;
    }

    setStreamingRecord((current) =>
      current?.id === recordId ? current : null,
    );
  }, [recordId]);

  useEffect(() => {
    if (!record || !streamingRecord || record.id !== streamingRecord.id) {
      return;
    }

    const processingStatus = normalizeProcessingStatus(
      record.processing_status,
      record.processing_error,
    );
    if (processingStatus === "ready" || processingStatus === "failed") {
      setStreamingRecord(null);
    }
  }, [record, streamingRecord]);

  const handleOpenModels = useCallback(() => {
    openModelManager();
  }, [openModelManager]);
  const handleOpenNewTranscriptionModal = useCallback(() => {
    setNewSpeechTextMode("transcription");
    setIsNewTranscriptionModalOpen(true);
  }, []);
  const handleCloseNewTranscriptionModal = useCallback(() => {
    setIsNewTranscriptionModalOpen(false);
  }, []);
  const handleLoadAllDiarizationManagedModels = useCallback(() => {
    for (const model of diarizationManagedModels) {
      if (model.status === "downloaded") {
        onLoad(model.variant);
      } else if (
        model.status === "not_downloaded" ||
        model.status === "error"
      ) {
        onDownload(model.variant);
      }
    }
  }, [diarizationManagedModels, onDownload, onLoad]);
  const handleUnloadAllDiarizationManagedModels = useCallback(() => {
    for (const model of diarizationManagedModels) {
      if (model.status === "ready") {
        onUnload(model.variant);
      }
    }
  }, [diarizationManagedModels, onUnload]);

  const handleDetailDelete = useCallback(async () => {
    if (!recordId || recordDeletePending) {
      return;
    }

    setRecordDeletePending(true);
    setRecordActionError(null);
    try {
      await api.deleteTranscriptionRecord(recordId);
      await refreshHistory();
      navigate("/transcription", { replace: true });
    } catch (err) {
      setRecordActionError(
        err instanceof Error
          ? err.message
          : "Failed to delete transcription record.",
      );
    } finally {
      setRecordDeletePending(false);
    }
  }, [navigate, recordDeletePending, recordId, refreshHistory]);

  const handleHistoryDelete = useCallback(
    async (targetRecord: SpeechTextJobSummary) => {
      if (targetRecord.kind === "diarization") {
        await api.deleteDiarizationRecord(targetRecord.id);
      } else {
        await api.deleteTranscriptionRecord(targetRecord.id);
      }
      await refreshHistory();
    },
    [refreshHistory],
  );

  const handleDetailRegenerateSummary = useCallback(async () => {
    if (!recordId || recordSummaryRefreshPending) {
      return;
    }
    if (!summaryModelReady) {
      openModelManager();
      setRecordSummaryRefreshError(summaryModelRequirementMessage);
      onError(summaryModelRequirementMessage);
      return;
    }

    setRecordSummaryRefreshPending(true);
    setRecordSummaryRefreshError(null);
    setRecordActionError(null);

    try {
      await api.regenerateTranscriptionSummary(recordId);
      await refreshRecord();
    } catch (err) {
      setRecordSummaryRefreshError(
        err instanceof Error
          ? err.message
          : "Failed to regenerate transcription summary.",
      );
    } finally {
      setRecordSummaryRefreshPending(false);
    }
  }, [
    onError,
    openModelManager,
    recordId,
    recordSummaryRefreshPending,
    refreshRecord,
    summaryModelReady,
    summaryModelRequirementMessage,
  ]);
  const handleCreatedDiarizationRecord = useCallback(
    async (createdRecord: DiarizationRecord) => {
      await refreshHistory().catch(() => undefined);
      navigate(`/transcription/${createdRecord.id}?mode=diarization`);
    },
    [navigate, refreshHistory],
  );

  const detailAudioUrl = useMemo(
    () => (recordId ? api.transcriptionRecordAudioUrl(recordId) : null),
    [recordId],
  );
  const timestampAlignerReady = useMemo(
    () =>
      transcriptionAlignerModels.some((model) => model.status === "ready"),
    [transcriptionAlignerModels],
  );
  const resolvedAlignerModel = useMemo(
    () =>
      transcriptionAlignerModels.find((model) => model.status === "ready")
        ?.variant ??
      transcriptionAlignerModels[0]?.variant ??
      null,
    [transcriptionAlignerModels],
  );
  const detailDescription = useMemo(() => {
    const visibleRecord =
      streamingRecord && streamingRecord.id === recordId ? streamingRecord : record;

    if (!visibleRecord) {
      return "Inspect job status, transcript output, and summary state for this transcription record.";
    }

    const processingStatus = normalizeProcessingStatus(
      visibleRecord.processing_status,
      visibleRecord.processing_error,
    );
    switch (processingStatus) {
      case "pending":
        return "This transcription job is queued.";
      case "processing":
        return "This transcription job is actively processing.";
      case "failed":
        return "This transcription job failed during processing.";
      case "ready":
      default:
        return "Inspect job status, transcript output, and summary state for this transcription record.";
    }
  }, [record, recordId, streamingRecord]);
  const visibleRecord = useMemo(() => {
    if (!recordId) {
      return null;
    }
    if (!streamingRecord || streamingRecord.id !== recordId) {
      return record;
    }
    if (!record) {
      return streamingRecord;
    }

    return {
      ...record,
      processing_status: streamingRecord.processing_status,
      processing_error:
        streamingRecord.processing_error ?? record.processing_error,
      language: streamingRecord.language ?? record.language,
      duration_secs: record.duration_secs ?? streamingRecord.duration_secs,
      processing_time_ms:
        record.processing_time_ms || streamingRecord.processing_time_ms,
      rtf: record.rtf ?? streamingRecord.rtf,
      transcription:
        streamingRecord.transcription || record.transcription,
      segments:
        streamingRecord.segments.length > 0
          ? streamingRecord.segments
          : record.segments,
      words:
        streamingRecord.words.length > 0 ? streamingRecord.words : record.words,
    };
  }, [record, recordId, streamingRecord]);

  return (
    <PageShell className={recordId ? "pb-32 sm:pb-36" : undefined}>
      {recordId ? (
        <>
          <PageHeader
            title="Transcription Record"
            description={detailDescription}
            actions={
              <Button
                type="button"
                variant="outline"
                size="sm"
                className="h-9 gap-2"
                onClick={handleOpenModels}
              >
                <Settings2 className="h-4 w-4" />
                Models
              </Button>
            }
          />

          <TranscriptionRecordDetail
            record={visibleRecord}
            audioUrl={detailAudioUrl}
            loading={recordLoading}
            error={recordError}
            deleteError={recordActionError}
            summaryModelGuidance={summaryModelReady ? null : summaryModelRequirementMessage}
            onBack={() => navigate("/transcription")}
            onDelete={handleDetailDelete}
            onRegenerateSummary={() => void handleDetailRegenerateSummary()}
            deletePending={recordDeletePending}
            summaryRefreshPending={recordSummaryRefreshPending}
            summaryRefreshError={recordSummaryRefreshError}
          />
        </>
      ) : (
        <>
          <PageHeader
            title="Transcription"
            description="Monitor queued, processing, and completed transcription jobs in one operational history table."
            actions={
              <>
                <Button
                  type="button"
                  size="sm"
                  className="h-9 gap-2"
                  onClick={handleOpenNewTranscriptionModal}
                >
                  New transcript
                </Button>
                <Button
                  type="button"
                  variant="outline"
                  size="sm"
                  className="h-9 gap-2"
                  onClick={handleOpenModels}
                >
                  <Settings2 className="h-4 w-4" />
                  Models
                </Button>
              </>
            }
          />

          <TranscriptionHistoryTable
            records={records}
            loading={historyLoading}
            error={historyError}
            loadMore={{
              canLoadMore: hasMoreRecords,
              loading: historyLoadingMore,
              onLoadMore: () => {
                void loadMoreRecords();
              },
            }}
            onRefresh={() => void refreshHistory()}
            onDeleteRecord={handleHistoryDelete}
            onOpenRecord={(nextRecord) => {
              if (nextRecord.kind === "diarization") {
                navigate(`/transcription/${nextRecord.id}?mode=diarization`);
                return;
              }
              navigate(`/transcription/${nextRecord.id}`);
            }}
          />

          {newSpeechTextMode === "transcription" ? (
            <NewTranscriptionModal
              isOpen={isNewTranscriptionModalOpen}
              onClose={handleCloseNewTranscriptionModal}
              blockOutsideDismiss={isModelModalOpen}
              selectedMode={newSpeechTextMode}
              onSelectMode={setNewSpeechTextMode}
              selectedModel={resolvedSelectedModel}
              selectedModelReady={selectedModelReady}
              timestampAlignerModelId={resolvedAlignerModel}
              timestampAlignerReady={timestampAlignerReady}
              onModelRequired={() => {
                requestModel();
                onError("Select and load an ASR model to start transcribing.");
              }}
              onTimestampAlignerRequired={() => {
                openModelManager();
                onError("Load the timestamp aligner model to enable timestamps.");
              }}
              onCreated={async (createdRecord: TranscriptionRecord) => {
                setStreamingRecord(createdRecord);
                await refreshHistory().catch(() => undefined);
                navigate(`/transcription/${createdRecord.id}`);
              }}
              onStreamingStart={() => {
                setStreamingRecord((current) =>
                  current
                    ? {
                        ...current,
                        processing_status: "processing",
                      }
                    : current,
                );
              }}
              onStreamingDelta={(delta) => {
                setStreamingRecord((current) =>
                  current
                    ? {
                        ...current,
                        processing_status: "processing",
                        transcription: `${current.transcription}${delta}`,
                      }
                    : current,
                );
              }}
              onStreamingFinal={(finalRecord) => {
                setStreamingRecord(finalRecord);
              }}
              onStreamingError={() => {
                void refreshRecord();
              }}
            />
          ) : (
            <NewDiarizationModal
              isOpen={isNewTranscriptionModalOpen}
              onClose={handleCloseNewTranscriptionModal}
              selectedMode={newSpeechTextMode}
              onSelectMode={setNewSpeechTextMode}
              selectedModel={resolvedDiarizationModel}
              selectedModelReady={diarizationModelReady}
              pipelineAsrModelId={resolvedDiarizationAsrModel}
              pipelineAlignerModelId={resolvedDiarizationAlignerModel}
              pipelineLlmModelId={resolvedDiarizationLlmModel}
              pipelineModelsReady={diarizationPipelineModelsReady}
              onModelRequired={() => {
                onError("Select and load a diarization model to start.");
              }}
              onPipelineModelsRequired={() => {
                onError("Load ASR and forced aligner models before diarization.");
              }}
              managedModelCount={diarizationManagedModels.length}
              readyManagedModelCount={readyDiarizationManagedModelCount}
              canLoadAnyManagedModels={canLoadAnyDiarizationManagedModels}
              canUnloadAnyManagedModels={canUnloadAnyDiarizationManagedModels}
              isManagedModelActionBusy={isDiarizationManagedModelActionBusy}
              onLoadAllManagedModels={handleLoadAllDiarizationManagedModels}
              onUnloadAllManagedModels={handleUnloadAllDiarizationManagedModels}
              onCreated={handleCreatedDiarizationRecord}
            />
          )}
        </>
      )}

      <RouteModelModal
        isOpen={isModelModalOpen}
        onClose={closeModelModal}
        title="Transcription Models"
        description="Manage ASR models, the optional timestamp aligner, and the summary model for this route."
        models={[
          ...transcriptionModels,
          ...transcriptionAlignerModels,
          ...transcriptionSummaryModels,
        ]}
        loading={loading}
        selectedVariant={resolvedSelectedModel}
        intentVariant={intentVariant}
        downloadProgress={downloadProgress}
        onDownload={onDownload}
        onCancelDownload={onCancelDownload}
        onLoad={onLoad}
        onUnload={onUnload}
        onDelete={onDelete}
        onUseModel={onSelect}
        emptyMessage="No transcription models available for this route."
        sections={modelSections}
        canUseModel={(variant) =>
          transcriptionModels.some((model) => model.variant === variant)
        }
        zIndexClassName={
          isNewTranscriptionModalOpen ? "z-[70]" : "z-50"
        }
      />
    </PageShell>
  );
}
