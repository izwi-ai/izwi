import { useCallback, useMemo, useState } from "react";
import { useNavigate, useParams } from "react-router";
import {
  api,
  type DiarizationRecordRerunRequest,
  type ModelInfo,
  type ModelResidencySummary,
} from "@/api";
import { PageShell } from "@/components/PageShell";
import { DiarizationRecordDetail } from "@/features/diarization/components/DiarizationRecordDetail";
import {
  DIARIZATION_PREFERRED_SUMMARY_MODELS,
  resolvePreferredRouteModel,
} from "@/features/models/catalog/routeModelCatalog";
import type { ModelDownloadProgressMap } from "@/features/models/downloadProgress";
import { RouteModelModal } from "@/features/models/components/RouteModelModal";
import {
  filterAndSortModels,
  isDiarizationPipelineAlignerModel,
  isDiarizationPipelineAsrModel,
  isDiarizationPipelineLlmModel,
  isDiarizationModel,
  isOneModelDiarizationModel,
  SINGLE_MODEL_DIARIZATION_DESCRIPTION,
} from "@/features/speech-text/modelFilters";
import { useDiarizationRecord } from "@/features/diarization/hooks/useDiarizationRecord";
import { Settings2 } from "lucide-react";

/** Diarization records are listed, and created, on the transcription page. */
const TRANSCRIPTION_LIST_PATH = "/transcription";

function diarizationRecordPath(recordId: string): string {
  return `${TRANSCRIPTION_LIST_PATH}/${recordId}?mode=diarization`;
}

interface DiarizationPageProps {
  models: ModelInfo[];
  loading: boolean;
  downloadProgress: ModelDownloadProgressMap;
  residencySummary?: ModelResidencySummary | null;
  onDownload: (variant: string) => void;
  onCancelDownload?: (variant: string) => void;
  onLoad: (variant: string) => void;
  onUnload: (variant: string) => void;
  onDelete: (variant: string) => void;
  onSelect: (variant: string) => void;
  onError: (message: string) => void;
}

interface DiarizationModelGroup {
  key: string;
  title: string;
  description: string;
  models: ModelInfo[];
}

/** A diarization record's detail page (`/transcription/:recordId?mode=diarization`). */
export function DiarizationPage({
  models,
  loading,
  downloadProgress,
  residencySummary,
  onDownload,
  onCancelDownload,
  onLoad,
  onUnload,
  onDelete,
  onSelect,
  onError,
}: DiarizationPageProps) {
  const { recordId } = useParams<{ recordId: string }>();
  const navigate = useNavigate();
  const [isModelModalOpen, setIsModelModalOpen] = useState(false);

  const diarizationModels = useMemo(
    () => filterAndSortModels(models, isDiarizationModel),
    [models],
  );

  const oneModelDiarizationModels = useMemo(
    () => filterAndSortModels(models, isOneModelDiarizationModel),
    [models],
  );

  const asrPipelineModels = useMemo(
    () => filterAndSortModels(models, isDiarizationPipelineAsrModel),
    [models],
  );

  const alignerPipelineModels = useMemo(
    () => filterAndSortModels(models, isDiarizationPipelineAlignerModel),
    [models],
  );

  const llmPipelineModels = useMemo(
    () => filterAndSortModels(models, isDiarizationPipelineLlmModel),
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
      ...(oneModelDiarizationModels.length > 0
        ? [
            {
              key: "single-model",
              title: "Single model",
              description: SINGLE_MODEL_DIARIZATION_DESCRIPTION,
              models: oneModelDiarizationModels,
            },
          ]
        : []),
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
        title: "Summary",
        description: "LLM that writes diarization summaries.",
        models: llmPipelineModels,
      },
    ],
    [
      asrPipelineModels,
      alignerPipelineModels,
      diarizationModels,
      llmPipelineModels,
      oneModelDiarizationModels,
    ],
  );

  const pipelineModels = useMemo(
    () => pipelineModelGroups.flatMap((group) => group.models),
    [pipelineModelGroups],
  );

  const resolvedSummaryModel = useMemo(
    () =>
      resolvePreferredRouteModel({
        models: llmPipelineModels,
        selectedModel: null,
        preferredVariants: DIARIZATION_PREFERRED_SUMMARY_MODELS,
        preferAnyPreferredBeforeReadyAny: true,
      }),
    [llmPipelineModels],
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
  const summaryModelReady = summaryModelStatus === "ready";
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

  const openModelManager = useCallback(() => {
    setIsModelModalOpen(true);
  }, []);
  const closeModelModal = useCallback(() => {
    setIsModelModalOpen(false);
  }, []);

  const handleDeleteRecord = useCallback(
    async (targetRecordId: string) => {
      await api.deleteDiarizationRecord(targetRecordId);
      if (recordId === targetRecordId) {
        navigate(TRANSCRIPTION_LIST_PATH, { replace: true });
      }
    },
    [navigate, recordId],
  );

  const handleSaveSpeakerCorrections = useCallback(
    async (
      targetRecordId: string,
      speakerNameOverrides: Record<string, string>,
    ) => {
      await api.updateDiarizationRecord(targetRecordId, {
        speaker_name_overrides: speakerNameOverrides,
      });
      if (recordId === targetRecordId) {
        await refreshRecord();
      }
    },
    [recordId, refreshRecord],
  );

  const handleRerunRecord = useCallback(
    async (
      targetRecordId: string,
      request: DiarizationRecordRerunRequest,
    ) => {
      const rerunRecord = await api.rerunDiarizationRecord(targetRecordId, request);
      navigate(diarizationRecordPath(rerunRecord.id));
    },
    [navigate],
  );

  const handleCancelProcessing = useCallback(
    async (targetRecordId: string) => {
      await api.cancelDiarizationRecord(targetRecordId);
      if (recordId === targetRecordId) {
        await refreshRecord();
      }
    },
    [recordId, refreshRecord],
  );

  const handleRegenerateSummary = useCallback(
    async (targetRecordId: string) => {
      if (!summaryModelReady) {
        openModelManager();
        onError(summaryModelRequirementMessage);
        throw new Error(summaryModelRequirementMessage);
      }
      await api.regenerateDiarizationSummary(targetRecordId);
      if (recordId === targetRecordId) {
        await refreshRecord();
      }
    },
    [
      onError,
      openModelManager,
      recordId,
      refreshRecord,
      summaryModelReady,
      summaryModelRequirementMessage,
    ],
  );

  return (
    <PageShell className="pb-24 sm:pb-28">
      <DiarizationRecordDetail
        parentCrumb={{ label: "Transcription", to: TRANSCRIPTION_LIST_PATH }}
        extraMenuItems={[
          {
            id: "models",
            label: "Models",
            icon: Settings2,
            onSelect: openModelManager,
          },
        ]}
        record={record}
        audioUrl={detailAudioUrl}
        loading={recordLoading}
        error={recordError}
        summaryModelGuidance={
          summaryModelReady ? null : summaryModelRequirementMessage
        }
        onDelete={handleDeleteRecord}
        onSaveSpeakerCorrections={handleSaveSpeakerCorrections}
        onRerun={handleRerunRecord}
        onCancelProcessing={handleCancelProcessing}
        onRegenerateSummary={handleRegenerateSummary}
      />

      <RouteModelModal
        isOpen={isModelModalOpen}
        onClose={closeModelModal}
        title="Diarization models"
        description="Manage diarization pipeline models for /v1/diarizations."
        models={pipelineModels}
        sections={pipelineModelGroups}
        loading={loading}
        selectedVariant={null}
        selectionMode="manage"
        downloadProgress={downloadProgress}
        onDownload={onDownload}
        onCancelDownload={onCancelDownload}
        onLoad={onLoad}
        onUnload={onUnload}
        onDelete={onDelete}
        onUseModel={onSelect}
        emptyMessage="No diarization pipeline models available for this route."
        residencySummary={residencySummary}
      />
    </PageShell>
  );
}
