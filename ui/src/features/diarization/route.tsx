import { useCallback, useEffect, useMemo, useState } from "react";
import { useNavigate, useParams } from "react-router-dom";
import {
  api,
  type DiarizationRecord,
  type DiarizationRecordRerunRequest,
  type ModelInfo,
} from "@/api";
import { PageHeader, PageShell } from "@/components/PageShell";
import { Button } from "@/components/ui/button";
import { DiarizationHistoryTable } from "@/features/diarization/components/DiarizationHistoryTable";
import { NewDiarizationModal } from "@/features/diarization/components/NewDiarizationModal";
import { DiarizationRecordDetail } from "@/features/diarization/components/DiarizationRecordDetail";
import {
  DIARIZATION_PREFERRED_ALIGNER_MODELS,
  DIARIZATION_PREFERRED_ASR_MODELS,
  DIARIZATION_PREFERRED_MODELS,
  DIARIZATION_PREFERRED_SUMMARY_MODELS,
  resolvePreferredRouteModel,
} from "@/features/models/catalog/routeModelCatalog";
import type { ModelDownloadProgressMap } from "@/features/models/downloadProgress";
import { RouteModelModal } from "@/features/models/components/RouteModelModal";
import {
  collectManagedModels,
  filterAndSortModels,
  isDiarizationPipelineAlignerVariant,
  isDiarizationPipelineAsrVariant,
  isDiarizationPipelineLlmVariant,
  isDiarizationVariant,
} from "@/features/speech-text/modelFilters";
import { useDiarizationHistory } from "@/features/diarization/hooks/useDiarizationHistory";
import { useDiarizationRecord } from "@/features/diarization/hooks/useDiarizationRecord";
import { summarizeDiarizationRecord } from "@/features/diarization/historySummary";
import { normalizeDiarizationProcessingStatus } from "@/utils/diarizationProcessing";
import { Settings2 } from "lucide-react";

interface DiarizationPageProps {
  models: ModelInfo[];
  selectedModel: string | null;
  loading: boolean;
  downloadProgress: ModelDownloadProgressMap;
  onDownload: (variant: string) => void;
  onCancelDownload?: (variant: string) => void;
  onLoad: (variant: string) => void;
  onUnload: (variant: string) => void;
  onDelete: (variant: string) => void;
  onSelect: (variant: string) => void;
  onError: (message: string) => void;
  routeBasePath?: string;
}

interface DiarizationModelGroup {
  key: string;
  title: string;
  description: string;
  models: ModelInfo[];
}

export function DiarizationPage({
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
  routeBasePath = "/diarization",
}: DiarizationPageProps) {
  const { recordId } = useParams<{ recordId: string }>();
  const navigate = useNavigate();
  const buildRoutePath = useCallback(
    (targetRecordId?: string) => {
      const basePath = targetRecordId
        ? `${routeBasePath}/${targetRecordId}`
        : routeBasePath;
      if (routeBasePath !== "/transcription") {
        return basePath;
      }
      const params = new URLSearchParams();
      params.set("mode", "diarization");
      return `${basePath}?${params.toString()}`;
    },
    [routeBasePath],
  );
  const [isNewDiarizationModalOpen, setIsNewDiarizationModalOpen] =
    useState(false);
  const [isModelModalOpen, setIsModelModalOpen] = useState(false);
  const [modalIntentModel, setModalIntentModel] = useState<string | null>(null);
  const [autoCloseOnIntentReady, setAutoCloseOnIntentReady] = useState(false);
  const [latestRecord, setLatestRecord] = useState<DiarizationRecord | null>(
    null,
  );

  const diarizationModels = useMemo(
    () => filterAndSortModels(models, isDiarizationVariant),
    [models],
  );

  const asrPipelineModels = useMemo(
    () => filterAndSortModels(models, isDiarizationPipelineAsrVariant),
    [models],
  );

  const alignerPipelineModels = useMemo(
    () => filterAndSortModels(models, isDiarizationPipelineAlignerVariant),
    [models],
  );

  const llmPipelineModels = useMemo(
    () => filterAndSortModels(models, isDiarizationPipelineLlmVariant),
    [models],
  );

  const pipelineModelGroups = useMemo<DiarizationModelGroup[]>(
    () => [
      {
        key: "diarization",
        title: "Diarization",
        description: "Speaker segmentation model used by this route.",
        models: diarizationModels,
      },
      {
        key: "asr",
        title: "ASR",
        description: "Transcript generation model in the diarization pipeline.",
        models: asrPipelineModels,
      },
      {
        key: "aligner",
        title: "Forced Aligner",
        description: "Word timing alignment model for speaker attribution.",
        models: alignerPipelineModels,
      },
      {
        key: "llm",
        title: "Refiner + Summary",
        description:
          "LLM used for transcript refinement and diarization summaries.",
        models: llmPipelineModels,
      },
    ],
    [
      asrPipelineModels,
      alignerPipelineModels,
      diarizationModels,
      llmPipelineModels,
    ],
  );

  const pipelineModels = useMemo(
    () => pipelineModelGroups.flatMap((group) => group.models),
    [pipelineModelGroups],
  );

  const resolvedSelectedModel = useMemo(
    () =>
      resolvePreferredRouteModel({
        models: diarizationModels,
        selectedModel,
        preferredVariants: DIARIZATION_PREFERRED_MODELS,
        preferAnyPreferredBeforeReadyAny: true,
      }),
    [diarizationModels, selectedModel],
  );

  const selectedModelInfo =
    diarizationModels.find(
      (model) => model.variant === resolvedSelectedModel,
    ) ?? null;
  const selectedModelReady = selectedModelInfo?.status === "ready";

  const resolvedAsrModel = useMemo(
    () =>
      resolvePreferredRouteModel({
        models: asrPipelineModels,
        selectedModel: null,
        preferredVariants: DIARIZATION_PREFERRED_ASR_MODELS,
        preferAnyPreferredBeforeReadyAny: true,
      }),
    [asrPipelineModels],
  );
  const resolvedAlignerModel = useMemo(
    () =>
      resolvePreferredRouteModel({
        models: alignerPipelineModels,
        selectedModel: null,
        preferredVariants: DIARIZATION_PREFERRED_ALIGNER_MODELS,
        preferAnyPreferredBeforeReadyAny: true,
      }),
    [alignerPipelineModels],
  );
  const resolvedLlmModel = useMemo(
    () =>
      resolvePreferredRouteModel({
        models: llmPipelineModels,
        selectedModel: null,
        preferredVariants: DIARIZATION_PREFERRED_SUMMARY_MODELS,
        preferAnyPreferredBeforeReadyAny: true,
      }),
    [llmPipelineModels],
  );
  const resolvedSummaryModel = resolvedLlmModel;
  const managedModels = useMemo(
    () =>
      collectManagedModels({
        availableModels: pipelineModels,
        managedVariants: [
          resolvedSelectedModel,
          resolvedAsrModel,
          resolvedAlignerModel,
          resolvedLlmModel,
        ],
      }),
    [
      pipelineModels,
      resolvedAlignerModel,
      resolvedAsrModel,
      resolvedLlmModel,
      resolvedSelectedModel,
    ],
  );

  const asrModelReady =
    resolvedAsrModel != null &&
    asrPipelineModels.some(
      (model) => model.variant === resolvedAsrModel && model.status === "ready",
    );
  const alignerModelReady =
    resolvedAlignerModel != null &&
    alignerPipelineModels.some(
      (model) =>
        model.variant === resolvedAlignerModel && model.status === "ready",
    );
  const llmModelReady =
    resolvedLlmModel != null &&
    llmPipelineModels.some(
      (model) => model.variant === resolvedLlmModel && model.status === "ready",
    );
  const summaryModelStatus = useMemo(() => {
    if (!resolvedSummaryModel) {
      return null;
    }
    return (
      llmPipelineModels.find(
        (model) => model.variant === resolvedSummaryModel,
      )?.status ?? null
    );
  }, [llmPipelineModels, resolvedSummaryModel]);
  const summaryModelReady = llmModelReady;
  const summaryModelRequirementMessage = useMemo(() => {
    const modelName = resolvedSummaryModel || "Qwen3.5-4B";
    switch (summaryModelStatus) {
      case "downloaded":
        return `Load ${modelName} in Diarization Models to generate summaries.`;
      case "downloading":
        return `${modelName} is downloading. Wait for download to complete, then generate summaries.`;
      case "loading":
        return `${modelName} is loading. Wait for it to become ready, then generate summaries.`;
      case "not_downloaded":
      case "error":
      default:
        return `Download and load ${modelName} in Diarization Models to generate summaries.`;
    }
  }, [resolvedSummaryModel, summaryModelStatus]);
  const pipelineModelsReady = asrModelReady && alignerModelReady;
  const readyManagedModelCount = managedModels.filter(
    (model) => model.status === "ready",
  ).length;
  const canLoadAnyManagedModels = managedModels.some(
    (model) =>
      model.status === "downloaded" ||
      model.status === "not_downloaded" ||
      model.status === "error",
  );
  const canUnloadAnyManagedModels = managedModels.some(
    (model) => model.status === "ready",
  );
  const isManagedModelActionBusy = managedModels.some(
    (model) => model.status === "loading" || model.status === "downloading",
  );

  const {
    records,
    loading: historyLoading,
    loadingMore: historyLoadingMore,
    error: historyError,
    hasMoreRecords,
    loadMoreRecords,
    refresh: refreshHistory,
  } = useDiarizationHistory();
  const {
    record,
    loading: recordLoading,
    error: recordError,
    refresh: refreshRecord,
  } = useDiarizationRecord(recordId);
  const detailAudioUrl = useMemo(
    () => (recordId ? api.diarizationRecordAudioUrl(recordId) : null),
    [recordId],
  );
  const visibleRecord = useMemo(() => {
    if (!recordId) {
      return null;
    }
    if (!record) {
      return latestRecord?.id === recordId ? latestRecord : null;
    }
    return record;
  }, [latestRecord, record, recordId]);
  const detailDescription = useMemo(() => {
    if (!visibleRecord) {
      return "Inspect transcript output, speaker corrections, and quality reruns for this saved diarization record.";
    }

    const processingStatus = normalizeDiarizationProcessingStatus(
      visibleRecord.processing_status,
      visibleRecord.processing_error,
    );
    switch (processingStatus) {
      case "pending":
        return "This diarization run is queued.";
      case "processing":
        return "This diarization run is actively processing.";
      case "failed":
        return "This diarization run failed during processing.";
      case "ready":
      default:
        return "Inspect transcript output, speaker corrections, and quality reruns for this saved diarization record.";
    }
  }, [visibleRecord]);
  const visibleHistoryRecords = useMemo(() => {
    if (!latestRecord) {
      return records;
    }
    if (records.some((recordSummary) => recordSummary.id === latestRecord.id)) {
      return records;
    }
    return [summarizeDiarizationRecord(latestRecord), ...records];
  }, [latestRecord, records]);

  useEffect(() => {
    if (!record || !latestRecord || record.id !== latestRecord.id) {
      return;
    }
    if (record === latestRecord) {
      return;
    }
    setLatestRecord(record);
  }, [latestRecord, record]);

  const closeModelModal = () => {
    setIsModelModalOpen(false);
    setAutoCloseOnIntentReady(false);
  };

  useEffect(() => {
    if (!isModelModalOpen || !modalIntentModel || !autoCloseOnIntentReady) {
      return;
    }
    const targetModel = pipelineModels.find(
      (model) => model.variant === modalIntentModel,
    );
    if (targetModel?.status === "ready") {
      closeModelModal();
    }
  }, [
    autoCloseOnIntentReady,
    pipelineModels,
    isModelModalOpen,
    modalIntentModel,
  ]);

  const openModelManager = () => {
    setModalIntentModel(null);
    setAutoCloseOnIntentReady(false);
    setIsModelModalOpen(true);
  };

  const openModelManagerForPipeline = () => {
    const missingPipelineVariant =
      (!asrModelReady && resolvedAsrModel) ||
      (!alignerModelReady && resolvedAlignerModel) ||
      resolvedSelectedModel;
    setModalIntentModel(missingPipelineVariant);
    setAutoCloseOnIntentReady(true);
    setIsModelModalOpen(true);
  };

  const handleOpenRecord = useCallback(
    (nextRecordId: string) => {
      navigate(buildRoutePath(nextRecordId));
    },
    [buildRoutePath, navigate],
  );

  const handleCloseRecord = useCallback(() => {
    navigate(buildRoutePath());
  }, [buildRoutePath, navigate]);
  const handleOpenNewDiarizationModal = useCallback(() => {
    setIsNewDiarizationModalOpen(true);
  }, []);
  const handleCloseNewDiarizationModal = useCallback(() => {
    setIsNewDiarizationModalOpen(false);
  }, []);

  const handleOpenModels = useCallback(() => {
    openModelManager();
  }, [openModelManager]);

  const handleLoadAllManagedModels = useCallback(() => {
    for (const model of managedModels) {
      if (model.status === "downloaded") {
        onLoad(model.variant);
      } else if (
        model.status === "not_downloaded" ||
        model.status === "error"
      ) {
        onDownload(model.variant);
      }
    }
  }, [managedModels, onDownload, onLoad]);

  const handleUnloadAllManagedModels = useCallback(() => {
    for (const model of managedModels) {
      if (model.status === "ready") {
        onUnload(model.variant);
      }
    }
  }, [managedModels, onUnload]);

  const handleCreatedRecord = useCallback(
    (createdRecord: DiarizationRecord) => {
      setLatestRecord(createdRecord);
      navigate(buildRoutePath(createdRecord.id));
      void refreshHistory().catch(() => undefined);
    },
    [buildRoutePath, navigate, refreshHistory],
  );

  const handleDeleteRecord = useCallback(
    async (targetRecordId: string) => {
      await api.deleteDiarizationRecord(targetRecordId);
      await refreshHistory();
      if (recordId === targetRecordId) {
        navigate(buildRoutePath(), { replace: true });
      }
      if (latestRecord?.id === targetRecordId) {
        setLatestRecord(null);
      }
    },
    [buildRoutePath, latestRecord?.id, navigate, recordId, refreshHistory],
  );

  const handleSaveSpeakerCorrections = useCallback(
    async (
      targetRecordId: string,
      speakerNameOverrides: Record<string, string>,
    ) => {
      await api.updateDiarizationRecord(targetRecordId, {
        speaker_name_overrides: speakerNameOverrides,
      });
      await Promise.all([
        refreshHistory(),
        recordId === targetRecordId ? refreshRecord() : Promise.resolve(),
      ]);
    },
    [recordId, refreshHistory, refreshRecord],
  );

  const handleRerunRecord = useCallback(
    async (
      targetRecordId: string,
      request: DiarizationRecordRerunRequest,
    ) => {
      const rerunRecord = await api.rerunDiarizationRecord(targetRecordId, request);
      await refreshHistory();
      if (latestRecord?.id === targetRecordId) {
        setLatestRecord(rerunRecord);
      }
      navigate(buildRoutePath(rerunRecord.id));
    },
    [buildRoutePath, latestRecord?.id, navigate, refreshHistory],
  );

  const handleCancelProcessing = useCallback(
    async (targetRecordId: string) => {
      const cancelledRecord = await api.cancelDiarizationRecord(targetRecordId);
      await Promise.all([
        refreshHistory(),
        recordId === targetRecordId ? refreshRecord() : Promise.resolve(),
      ]);
      if (latestRecord?.id === targetRecordId) {
        setLatestRecord(cancelledRecord);
      }
    },
    [latestRecord?.id, recordId, refreshHistory, refreshRecord],
  );

  const handleRegenerateSummary = useCallback(
    async (targetRecordId: string) => {
      if (!summaryModelReady) {
        openModelManager();
        onError(summaryModelRequirementMessage);
        throw new Error(summaryModelRequirementMessage);
      }
      await api.regenerateDiarizationSummary(targetRecordId);
      await Promise.all([
        refreshHistory(),
        recordId === targetRecordId ? refreshRecord() : Promise.resolve(),
      ]);
    },
    [
      onError,
      openModelManager,
      recordId,
      refreshHistory,
      refreshRecord,
      summaryModelReady,
      summaryModelRequirementMessage,
    ],
  );

  return (
    <PageShell className={recordId ? "pb-24 sm:pb-28" : undefined}>
      {recordId ? (
        <>
          <PageHeader
            title="Diarization Record"
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

          <DiarizationRecordDetail
            record={visibleRecord}
            audioUrl={detailAudioUrl}
            loading={recordLoading}
            error={recordError}
            summaryModelGuidance={
              summaryModelReady ? null : summaryModelRequirementMessage
            }
            onBack={handleCloseRecord}
            onDelete={handleDeleteRecord}
            onSaveSpeakerCorrections={handleSaveSpeakerCorrections}
            onRerun={handleRerunRecord}
            onCancelProcessing={handleCancelProcessing}
            onRegenerateSummary={handleRegenerateSummary}
          />
        </>
      ) : (
        <>
          <PageHeader
            title="Diarization"
            description="Monitor saved diarization runs in one history table and launch new speaker-separated transcripts from a focused creation modal."
            actions={
              <>
                <Button
                  type="button"
                  size="sm"
                  className="h-9 gap-2"
                  onClick={handleOpenNewDiarizationModal}
                >
                  New diarization
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

          <DiarizationHistoryTable
            records={visibleHistoryRecords}
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
            onOpenRecord={handleOpenRecord}
            onDeleteRecord={handleDeleteRecord}
          />

          <NewDiarizationModal
            isOpen={isNewDiarizationModalOpen}
            onClose={handleCloseNewDiarizationModal}
            selectedModel={resolvedSelectedModel}
            selectedModelReady={selectedModelReady}
            pipelineAsrModelId={resolvedAsrModel}
            pipelineAlignerModelId={resolvedAlignerModel}
            pipelineLlmModelId={resolvedLlmModel}
            pipelineModelsReady={pipelineModelsReady}
            onModelRequired={() => {
              setModalIntentModel(resolvedSelectedModel);
              setAutoCloseOnIntentReady(true);
              setIsModelModalOpen(true);
              onError("Select and load a diarization model to start.");
            }}
            onPipelineModelsRequired={() => {
              openModelManagerForPipeline();
              onError("Load ASR and forced aligner models before diarization.");
            }}
            managedModelCount={managedModels.length}
            readyManagedModelCount={readyManagedModelCount}
            canLoadAnyManagedModels={canLoadAnyManagedModels}
            canUnloadAnyManagedModels={canUnloadAnyManagedModels}
            isManagedModelActionBusy={isManagedModelActionBusy}
            onLoadAllManagedModels={handleLoadAllManagedModels}
            onUnloadAllManagedModels={handleUnloadAllManagedModels}
            onCreated={handleCreatedRecord}
          />
        </>
      )}

      <RouteModelModal
        isOpen={isModelModalOpen}
        onClose={closeModelModal}
        title="Diarization Models"
        description="Manage diarization pipeline models for /v1/diarizations."
        models={pipelineModels}
        sections={pipelineModelGroups}
        loading={loading}
        selectedVariant={null}
        intentVariant={modalIntentModel}
        selectionMode="manage"
        downloadProgress={downloadProgress}
        onDownload={onDownload}
        onCancelDownload={onCancelDownload}
        onLoad={onLoad}
        onUnload={onUnload}
        onDelete={onDelete}
        onUseModel={onSelect}
        emptyMessage="No diarization pipeline models available for this route."
      />
    </PageShell>
  );
}
