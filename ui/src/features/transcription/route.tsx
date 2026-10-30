import { useCallback, useEffect, useMemo, useState } from "react";
import { useNavigate, useParams, useSearchParams } from "react-router";
import {
  api,
  type DiarizationRecord,
  type ModelInfo,
  type SpeechTextJobSummary,
  type TranscriptionRecord,
} from "@/api";
import { PageHeader, PageShell } from "@/components/PageShell";
import { Button } from "@/components/ui/button";
import { Dialog, DialogContent } from "@/components/ui/dialog";
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
  SPEAKER_ATTRIBUTED_ASR_PREFERRED_MODELS,
  TRANSCRIPTION_PREFERRED_MODELS,
  resolvePreferredRouteModel,
} from "@/features/models/catalog/routeModelCatalog";
import { getModelDisplayName } from "@/features/models/catalog/modelMetadata";
import type { ModelDownloadProgressMap } from "@/features/models/downloadProgress";
import { RouteModelModal } from "@/features/models/components/RouteModelModal";
import { useRouteModelSelection } from "@/features/models/hooks/useRouteModelSelection";
import type { SpeechTextCreationMode } from "@/features/speech-text/creationMode";
import {
  collectManagedModels,
  filterAndSortModels,
  isDiarizationPipelineAlignerModel,
  isDiarizationPipelineAsrModel,
  isDiarizationPipelineLlmModel,
  isDiarizationModel,
  installedGraniteSpeech5Model,
  isOneModelDiarizationModel,
  isSpeakerAttributedAsrModel,
  isTranscriptionAlignerModel,
  isTranscriptionSummaryModel,
  resolveDiarizationStackRole,
  SINGLE_MODEL_DIARIZATION_DESCRIPTION,
} from "@/features/speech-text/modelFilters";
import type { PipelineRoleKey } from "@/app/providers/ModelCatalogProvider";
import { useTranscriptionHistory } from "@/features/transcription/hooks/useTranscriptionHistory";
import { useTranscriptionRecord } from "@/features/transcription/hooks/useTranscriptionRecord";
import { useNewItemShortcut } from "@/hooks/useNewItemShortcut";
import {
  HistoryToolbar,
  SearchField,
  SegmentedFilter,
} from "@/components/history/HistoryToolbar";
import { NewItemButton } from "@/components/history/NewItemButton";
import { CREATION_DIALOG_CONTENT_CLASS } from "@/components/CreationDialog";
import { cn } from "@/lib/utils";
import {
  parseTranscriptionFilter,
  TRANSCRIPTION_FILTER_OPTIONS,
  transcriptionListFilters,
} from "@/features/transcription/historyFilters";
import { normalizeProcessingStatus } from "@/features/transcription/playground/support";
import { AnimatePresence, motion } from "framer-motion";
import { Settings2 } from "lucide-react";

const TRANSCRIPTION_PREFERRED_SUMMARY_MODELS = ["Qwen3.5-4B"] as const;
type ModelModalContext =
  | "transcription"
  | "speaker_attributed_asr"
  | "diarization";
type DiarizationPipelineMode = "classic" | "granite";

const noRolePick = (): string | null => null;

interface TranscriptionPageProps {
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
  /** The diarization stack's own picks per role (diarizer, ASR, aligner). */
  rolePickFor?: (role: PipelineRoleKey) => string | null;
  onPickRoleModel?: (role: PipelineRoleKey, variant: string | null) => void;
  onLoadForRole?: (variant: string, role: PipelineRoleKey | null) => void;
  speechTextMode?: "transcription" | "speaker_attributed_asr";
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
  rolePickFor = noRolePick,
  onPickRoleModel,
  onLoadForRole,
  speechTextMode = "transcription",
}: TranscriptionPageProps) {
  const { recordId } = useParams<{ recordId: string }>();
  const navigate = useNavigate();
  const [searchParams, setSearchParams] = useSearchParams();
  const [isNewTranscriptionModalOpen, setIsNewTranscriptionModalOpen] =
    useState(false);
  // An upload in the creation dialog keeps it open: closing would abort it.
  const [isCreationBusy, setIsCreationBusy] = useState(false);
  const [newSpeechTextMode, setNewSpeechTextMode] =
    useState<SpeechTextCreationMode>("transcription");
  const [recordActionError, setRecordActionError] = useState<string | null>(null);
  const [recordDeletePending, setRecordDeletePending] = useState(false);
  const [recordCancelPending, setRecordCancelPending] = useState(false);
  const [recordRetryPending, setRecordRetryPending] = useState(false);
  const [recordSummaryRefreshPending, setRecordSummaryRefreshPending] =
    useState(false);
  const [recordSummaryRefreshError, setRecordSummaryRefreshError] = useState<
    string | null
  >(null);
  const [streamingRecord, setStreamingRecord] =
    useState<TranscriptionRecord | null>(null);
  const [modelModalContext, setModelModalContext] =
    useState<ModelModalContext>("transcription");
  const viewConfig = VIEW_CONFIGS.transcription;
  const transcriptionAlignerModels = useMemo(
    () => filterAndSortModels(models, isTranscriptionAlignerModel),
    [models],
  );
  const transcriptionSummaryModels = useMemo(
    () => filterAndSortModels(models, isTranscriptionSummaryModel),
    [models],
  );
  const diarizationModels = useMemo(
    () => filterAndSortModels(models, isDiarizationModel),
    [models],
  );
  const oneModelDiarizationModels = useMemo(
    () => filterAndSortModels(models, isOneModelDiarizationModel),
    [models],
  );
  const speakerAttributedAsrModels = useMemo(
    () => filterAndSortModels(models, isSpeakerAttributedAsrModel),
    [models],
  );
  const diarizationAsrPipelineModels = useMemo(
    () => filterAndSortModels(models, isDiarizationPipelineAsrModel),
    [models],
  );
  const diarizationAlignerPipelineModels = useMemo(
    () => filterAndSortModels(models, isDiarizationPipelineAlignerModel),
    [models],
  );
  const diarizationLlmPipelineModels = useMemo(
    () => filterAndSortModels(models, isDiarizationPipelineLlmModel),
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
  const selectedTranscriptionModelStatus = useMemo(
    () =>
      transcriptionModels.find(
        (model) => model.variant === resolvedSelectedModel,
      )?.status ?? null,
    [resolvedSelectedModel, transcriptionModels],
  );
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
        return `Load ${modelName} in Transcription models to generate summaries.`;
      case "downloading":
        return `${modelName} is downloading. Wait for download to complete, then generate the summary.`;
      case "loading":
        return `${modelName} is loading. Wait for it to become ready, then generate the summary.`;
      case "not_downloaded":
      case "error":
      default:
        return `Download and load ${modelName} in Transcription models to generate summaries.`;
    }
  }, [resolvedSummaryModel, summaryModelStatus]);
  const resolvedSpeakerAttributedAsrModel = useMemo(
    () =>
      resolvePreferredRouteModel({
        models: speakerAttributedAsrModels,
        selectedModel,
        preferredVariants: SPEAKER_ATTRIBUTED_ASR_PREFERRED_MODELS,
        preferAnyPreferredBeforeReadyAny: true,
      }),
    [selectedModel, speakerAttributedAsrModels],
  );
  const speakerAttributedAsrModelReady =
    resolvedSpeakerAttributedAsrModel != null &&
    speakerAttributedAsrModels.some(
      (model) =>
        model.variant === resolvedSpeakerAttributedAsrModel &&
        model.status === "ready",
    );
  const selectedSpeakerAttributedAsrModelStatus = useMemo(
    () =>
      speakerAttributedAsrModels.find(
        (model) => model.variant === resolvedSpeakerAttributedAsrModel,
      )?.status ?? null,
    [resolvedSpeakerAttributedAsrModel, speakerAttributedAsrModels],
  );
  // The diarization stack is one diarizer, one ASR and one aligner: each the
  // user's pick for its role, else a loaded model, else the default. A model
  // that diarizes alone is the diarizer only when the user picks it, and
  // then the stack is that model by itself.
  const diarizerRole = useMemo(
    () =>
      resolveDiarizationStackRole({
        models,
        eligibleModels: diarizationModels,
        pickOnlyModels: oneModelDiarizationModels,
        pick: rolePickFor("diarization"),
        preferredVariants: DIARIZATION_PREFERRED_MODELS,
      }),
    [diarizationModels, models, oneModelDiarizationModels, rolePickFor],
  );
  const diarizationAsrRole = useMemo(
    () =>
      resolveDiarizationStackRole({
        models,
        eligibleModels: diarizationAsrPipelineModels,
        pick: rolePickFor("diarization-asr"),
        preferredVariants: DIARIZATION_PREFERRED_ASR_MODELS,
      }),
    [diarizationAsrPipelineModels, models, rolePickFor],
  );
  const diarizationAlignerRole = useMemo(
    () =>
      resolveDiarizationStackRole({
        models,
        eligibleModels: diarizationAlignerPipelineModels,
        pick: rolePickFor("diarization-aligner"),
        preferredVariants: DIARIZATION_PREFERRED_ALIGNER_MODELS,
      }),
    [diarizationAlignerPipelineModels, models, rolePickFor],
  );
  const resolvedDiarizationModel = diarizerRole.variant;
  const diarizationModelReady = diarizerRole.ready;
  const diarizesInOneModel = oneModelDiarizationModels.some(
    (model) => model.variant === resolvedDiarizationModel,
  );
  const resolvedDiarizationAsrModel = diarizationAsrRole.variant;
  const resolvedDiarizationAlignerModel = diarizationAlignerRole.variant;
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
  const diarizationPipelineMode: DiarizationPipelineMode = "classic";
  const diarizationPipelineModelsReady =
    diarizationAsrRole.ready && diarizationAlignerRole.ready;
  const baseDiarizationPipelineModels = useMemo(
    () => [
      ...diarizationModels,
      ...oneModelDiarizationModels,
      ...diarizationAsrPipelineModels,
      ...diarizationAlignerPipelineModels,
      ...diarizationLlmPipelineModels,
    ],
    [
      diarizationAlignerPipelineModels,
      diarizationAsrPipelineModels,
      diarizationLlmPipelineModels,
      diarizationModels,
      oneModelDiarizationModels,
    ],
  );
  const diarizationPipelineModels = baseDiarizationPipelineModels;
  const diarizationManagedModels = useMemo(
    () =>
      collectManagedModels({
        availableModels: diarizationPipelineModels,
        // A model that diarizes alone runs no ASR or aligner.
        managedVariants: diarizesInOneModel
          ? [resolvedDiarizationModel, resolvedDiarizationLlmModel]
          : [
              resolvedDiarizationModel,
              resolvedDiarizationAsrModel,
              resolvedDiarizationAlignerModel,
              resolvedDiarizationLlmModel,
            ],
      }),
    [
      diarizationPipelineModels,
      diarizesInOneModel,
      resolvedDiarizationAlignerModel,
      resolvedDiarizationAsrModel,
      resolvedDiarizationLlmModel,
      resolvedDiarizationModel,
    ],
  );
  const diarizationSummaryModelReady =
    resolvedDiarizationLlmModel != null &&
    diarizationLlmPipelineModels.some(
      (model) =>
        model.variant === resolvedDiarizationLlmModel && model.status === "ready",
    );
  const canUnloadAnyDiarizationManagedModels = diarizationManagedModels.some(
    (model) => model.status === "ready",
  );
  const isDiarizationManagedModelActionBusy = diarizationManagedModels.some(
    (model) => model.status === "loading" || model.status === "downloading",
  );
  const transcriptionModelSections = useMemo(
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
  const speakerAttributedAsrModelSections = useMemo(
    () => [
      {
        key: "speaker_attributed_asr",
        title: "Speaker Attributed ASR",
        description: "ASR models that write speaker-turn transcripts themselves.",
        models: speakerAttributedAsrModels,
      },
      {
        key: "summary",
        title: "Summary Models",
        description:
          "Model used to generate AI summaries for completed SAA transcripts.",
        models: transcriptionSummaryModels,
      },
    ],
    [speakerAttributedAsrModels, transcriptionSummaryModels],
  );
  const baseDiarizationModelSections = useMemo(
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
        models: diarizationAsrPipelineModels,
      },
      {
        key: "aligner",
        title: "Forced Aligner",
        description: "Word timing alignment model for speaker attribution.",
        models: diarizationAlignerPipelineModels,
      },
      {
        key: "llm",
        title: "Summary",
        description: "LLM that writes diarization summaries.",
        models: diarizationLlmPipelineModels,
      },
    ],
    [
      diarizationAlignerPipelineModels,
      diarizationAsrPipelineModels,
      diarizationLlmPipelineModels,
      diarizationModels,
      oneModelDiarizationModels,
    ],
  );
  const diarizationModelSections = baseDiarizationModelSections;
  // List search and filter live in the URL, so Back from a record keeps them.
  const historySearch = searchParams.get("q") ?? "";
  const historyFilter = parseTranscriptionFilter(searchParams.get("show"));
  const setHistoryParam = useCallback(
    (key: "q" | "show", value: string) => {
      const next = new URLSearchParams(searchParams);
      if (value && !(key === "show" && value === "all")) {
        next.set(key, value);
      } else {
        next.delete(key);
      }
      setSearchParams(next, { replace: true });
    },
    [searchParams, setSearchParams],
  );
  const {
    records,
    total: historyTotal,
    loading: historyLoading,
    loadingMore: historyLoadingMore,
    error: historyError,
    hasMoreRecords,
    loadMoreRecords,
    refresh: refreshHistory,
  } = useTranscriptionHistory(
    transcriptionListFilters(historySearch, historyFilter),
  );
  const {
    record,
    loading: recordLoading,
    error: recordError,
    refresh: refreshRecord,
  } = useTranscriptionRecord(recordId, {
    jobKind:
      speechTextMode === "speaker_attributed_asr"
        ? "speaker_attributed_asr"
        : "transcription",
  });

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

  const openTranscriptionModelManager = useCallback(() => {
    setModelModalContext("transcription");
    openModelManager();
  }, [openModelManager]);
  const openSpeakerAttributedAsrModelManager = useCallback(() => {
    setModelModalContext("speaker_attributed_asr");
    openModelManager();
  }, [openModelManager]);
  const openDiarizationModelManager = useCallback(() => {
    setModelModalContext("diarization");
    openModelManager();
  }, [openModelManager]);
  const handleOpenModels = useCallback(() => {
    if (speechTextMode === "speaker_attributed_asr") {
      openSpeakerAttributedAsrModelManager();
      return;
    }
    openTranscriptionModelManager();
  }, [
    openSpeakerAttributedAsrModelManager,
    openTranscriptionModelManager,
    speechTextMode,
  ]);
  const handleOpenNewTranscriptionModal = useCallback(() => {
    setNewSpeechTextMode("transcription");
    setIsNewTranscriptionModalOpen(true);
  }, []);
  useNewItemShortcut(handleOpenNewTranscriptionModal);
  const handleCloseNewTranscriptionModal = useCallback(() => {
    setIsNewTranscriptionModalOpen(false);
    if (searchParams.has("create")) {
      const nextSearchParams = new URLSearchParams(searchParams);
      nextSearchParams.delete("create");
      setSearchParams(nextSearchParams, { replace: true });
    }
  }, [searchParams, setSearchParams]);

  // ?create=transcription|diarization (the command palette, legacy
  // /diarization links) opens the creation modal in that mode.
  useEffect(() => {
    const requestedMode = searchParams.get("create");
    if (
      recordId ||
      (requestedMode !== "transcription" && requestedMode !== "diarization")
    ) {
      return;
    }

    setNewSpeechTextMode(requestedMode);
    setIsNewTranscriptionModalOpen(true);
  }, [recordId, searchParams]);
  // Loading from the diarization models dialog is choosing that model for
  // the role it fills (the summary model has no pick of its own).
  const handleLoadDiarizationStackModel = useCallback(
    (variant: string) => {
      if (!onLoadForRole) {
        onLoad(variant);
        return;
      }
      const model = models.find((candidate) => candidate.variant === variant);
      const role: PipelineRoleKey | null = !model
        ? null
        : isDiarizationModel(model) || isOneModelDiarizationModel(model)
          ? "diarization"
          : isDiarizationPipelineAlignerModel(model)
            ? "diarization-aligner"
            : isDiarizationPipelineAsrModel(model)
              ? "diarization-asr"
              : null;
      onLoadForRole(variant, role);
    },
    [models, onLoad, onLoadForRole],
  );
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
      if (speechTextMode === "speaker_attributed_asr") {
        await api.deleteSpeakerAttributedAsrRecord(recordId);
      } else {
        await api.deleteTranscriptionRecord(recordId);
      }
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
  }, [navigate, recordDeletePending, recordId, refreshHistory, speechTextMode]);

  const handleDetailCancel = useCallback(async () => {
    if (!recordId || recordCancelPending) {
      return;
    }
    setRecordCancelPending(true);
    setRecordActionError(null);
    try {
      await api.cancelTranscriptionRecord(recordId);
      await refreshRecord();
      await refreshHistory();
    } catch (err) {
      setRecordActionError(
        err instanceof Error ? err.message : "Failed to cancel this job.",
      );
    } finally {
      setRecordCancelPending(false);
    }
  }, [recordCancelPending, recordId, refreshHistory, refreshRecord]);

  /** A retry is a new record with the same settings: open it. */
  const handleDetailRetry = useCallback(async () => {
    if (!recordId || recordRetryPending) {
      return;
    }
    setRecordRetryPending(true);
    setRecordActionError(null);
    try {
      const retried = await api.retryTranscriptionRecord(recordId);
      await refreshHistory();
      navigate(
        retried.transcription_mode === "speaker_attributed_asr"
          ? `/transcription/${retried.id}?mode=speaker_attributed_asr`
          : `/transcription/${retried.id}`,
      );
    } catch (err) {
      setRecordActionError(
        err instanceof Error ? err.message : "Failed to retry this job.",
      );
    } finally {
      setRecordRetryPending(false);
    }
  }, [navigate, recordId, recordRetryPending, refreshHistory]);

  const handleHistoryDelete = useCallback(
    async (targetRecord: SpeechTextJobSummary) => {
      if (targetRecord.kind === "diarization") {
        await api.deleteDiarizationRecord(targetRecord.id);
      } else if (targetRecord.kind === "speaker_attributed_asr") {
        await api.deleteSpeakerAttributedAsrRecord(targetRecord.id);
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
      openTranscriptionModelManager();
      setRecordSummaryRefreshError(summaryModelRequirementMessage);
      onError(summaryModelRequirementMessage);
      return;
    }

    setRecordSummaryRefreshPending(true);
    setRecordSummaryRefreshError(null);
    setRecordActionError(null);

    try {
      if (speechTextMode === "speaker_attributed_asr") {
        await api.regenerateSpeakerAttributedAsrSummary(recordId);
      } else {
        await api.regenerateTranscriptionSummary(recordId);
      }
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
    openTranscriptionModelManager,
    recordId,
    recordSummaryRefreshPending,
    refreshRecord,
    speechTextMode,
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
    () =>
      recordId
        ? speechTextMode === "speaker_attributed_asr"
          ? api.speakerAttributedAsrRecordAudioUrl(recordId)
          : api.transcriptionRecordAudioUrl(recordId)
        : null,
    [recordId, speechTextMode],
  );
  const isDiarizationModelModal = modelModalContext === "diarization";
  const isSpeakerAttributedAsrModelModal =
    modelModalContext === "speaker_attributed_asr";
  const transcriptionModalModels = useMemo(
    () => [
      ...transcriptionModels,
      ...transcriptionAlignerModels,
      ...transcriptionSummaryModels,
    ],
    [transcriptionAlignerModels, transcriptionModels, transcriptionSummaryModels],
  );
  const speakerAttributedAsrModalModels = useMemo(
    () => [...speakerAttributedAsrModels, ...transcriptionSummaryModels],
    [speakerAttributedAsrModels, transcriptionSummaryModels],
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
  const selectedAlignerModelStatus = useMemo(
    () =>
      transcriptionAlignerModels.find(
        (model) => model.variant === resolvedAlignerModel,
      )?.status ?? null,
    [resolvedAlignerModel, transcriptionAlignerModels],
  );
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
      processing_progress:
        streamingRecord.processing_progress ?? record.processing_progress,
      rtf: record.rtf ?? streamingRecord.rtf,
      transcription_mode:
        streamingRecord.transcription_mode ?? record.transcription_mode,
      transcription:
        streamingRecord.transcription || record.transcription,
      segments:
        streamingRecord.segments.length > 0
          ? streamingRecord.segments
          : record.segments,
      words:
        streamingRecord.words.length > 0 ? streamingRecord.words : record.words,
      speaker_attributed_text:
        streamingRecord.speaker_attributed_text ??
        record.speaker_attributed_text,
      speaker_turns:
        (streamingRecord.speaker_turns?.length ?? 0) > 0
          ? streamingRecord.speaker_turns
          : record.speaker_turns,
      saa_status: streamingRecord.saa_status ?? record.saa_status,
      saa_warnings: streamingRecord.saa_warnings ?? record.saa_warnings,
    };
  }, [record, recordId, streamingRecord]);
  const isNewSpeakerAttributedAsrMode =
    newSpeechTextMode === "speaker_attributed_asr";
  const installedGraniteSpeech5 = useMemo(
    () => installedGraniteSpeech5Model(models),
    [models],
  );
  // Granite Speech 5.0 is not a speaker-attributed model; say where its
  // speakers come from instead of leaving users to wonder why it is missing.
  const speakerAttributionHint = installedGraniteSpeech5
    ? `${getModelDisplayName(installedGraniteSpeech5.variant, installedGraniteSpeech5.display_name)} has no speaker attribution. For speakers with it, use Diarization and pick it as the ASR model.`
    : null;
  const reconcileStreamedTranscription = useCallback(
    async (createdRecordId: string | null) => {
      if (!createdRecordId) {
        await refreshRecord().catch(() => undefined);
        return;
      }

      try {
        const persistedRecord = isNewSpeakerAttributedAsrMode
          ? await api.getSpeakerAttributedAsrRecord(createdRecordId)
          : await api.getTranscriptionRecord(createdRecordId);
        setStreamingRecord(persistedRecord);
      } catch {
        await refreshRecord().catch(() => undefined);
      } finally {
        await refreshHistory().catch(() => undefined);
      }
    },
    [isNewSpeakerAttributedAsrMode, refreshHistory, refreshRecord],
  );
  const newTranscriptionSelectedModel = isNewSpeakerAttributedAsrMode
    ? resolvedSpeakerAttributedAsrModel
    : resolvedSelectedModel;
  const newTranscriptionSelectedModelReady = isNewSpeakerAttributedAsrMode
    ? speakerAttributedAsrModelReady
    : selectedModelReady;
  const newTranscriptionSelectedModelStatus = isNewSpeakerAttributedAsrMode
    ? selectedSpeakerAttributedAsrModelStatus
    : selectedTranscriptionModelStatus;
  const newTranscriptionRouteCapabilities = models.find(
    (model) => model.variant === newTranscriptionSelectedModel,
  )?.route_capabilities;
  const openNewTranscriptionModelManager = isNewSpeakerAttributedAsrMode
    ? openSpeakerAttributedAsrModelManager
    : openTranscriptionModelManager;

  return (
    <PageShell className={recordId ? "pb-32 sm:pb-36" : undefined}>
      {recordId ? (
        <>
          <TranscriptionRecordDetail
            parentCrumb={{ label: "Transcription", to: "/transcription" }}
            extraMenuItems={[
              {
                id: "models",
                label: "Models",
                icon: Settings2,
                onSelect: handleOpenModels,
              },
            ]}
            record={visibleRecord}
            audioUrl={detailAudioUrl}
            loading={recordLoading}
            error={recordError}
            deleteError={recordActionError}
            summaryModelGuidance={summaryModelReady ? null : summaryModelRequirementMessage}
            onDelete={handleDetailDelete}
            onCancel={() => void handleDetailCancel()}
            cancelPending={recordCancelPending}
            onRetry={() => void handleDetailRetry()}
            retryPending={recordRetryPending}
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
            description="Transcribe recordings and files locally. Nothing leaves this device."
            actions={
              <>
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
                <NewItemButton
                  label="New transcript"
                  onClick={handleOpenNewTranscriptionModal}
                />
              </>
            }
          />

          <HistoryToolbar>
            <SearchField
              value={historySearch}
              onChange={(value) => setHistoryParam("q", value)}
              placeholder="Search transcripts…"
            />
            <SegmentedFilter
              label="Show"
              options={TRANSCRIPTION_FILTER_OPTIONS}
              value={historyFilter}
              onChange={(value) => setHistoryParam("show", value)}
              selectedCount={historyLoading ? null : historyTotal}
            />
          </HistoryToolbar>

          <TranscriptionHistoryTable
            records={records}
            total={historyTotal}
            filtered={Boolean(historySearch.trim()) || historyFilter !== "all"}
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
              if (nextRecord.kind === "speaker_attributed_asr") {
                navigate(
                  `/transcription/${nextRecord.id}?mode=speaker_attributed_asr`,
                );
                return;
              }
              navigate(`/transcription/${nextRecord.id}`);
            }}
          />

          <Dialog
            open={isNewTranscriptionModalOpen}
            modal={!isModelModalOpen}
            onOpenChange={(open) => {
              if (!open && !isCreationBusy) {
                handleCloseNewTranscriptionModal();
              }
            }}
          >
            <DialogContent
              className={cn(CREATION_DIALOG_CONTENT_CLASS, "max-w-208")}
              onEscapeKeyDown={(event) => {
                if (isModelModalOpen || isCreationBusy) {
                  event.preventDefault();
                }
              }}
              onPointerDownOutside={(event) => {
                if (isModelModalOpen || isCreationBusy) {
                  event.preventDefault();
                }
              }}
              onInteractOutside={(event) => {
                if (isModelModalOpen || isCreationBusy) {
                  event.preventDefault();
                }
              }}
            >
              <AnimatePresence initial={false} mode="wait">
                <motion.div
                  key={newSpeechTextMode}
                  className="flex min-h-0 flex-1 flex-col"
                  initial={{ opacity: 0, y: 8 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -6 }}
                  transition={{ duration: 0.18, ease: "easeOut" }}
                >
                  {newSpeechTextMode !== "diarization" ? (
                    <NewTranscriptionModal
                      isOpen={isNewTranscriptionModalOpen}
                      onClose={handleCloseNewTranscriptionModal}
                      renderInDialog={false}
                      selectedMode={newSpeechTextMode}
                      onSelectMode={setNewSpeechTextMode}
                      selectedModel={newTranscriptionSelectedModel}
                      selectedModelReady={newTranscriptionSelectedModelReady}
                      selectedModelStatus={newTranscriptionSelectedModelStatus}
                      selectedModelSupportsTranslation={
                        newTranscriptionRouteCapabilities?.speech_translation ===
                        true
                      }
                      selectedModelSupportsKeywords={
                        newTranscriptionRouteCapabilities?.keyword_biasing === true
                      }
                      selectedModelHasNativeWordTimestamps={
                        newTranscriptionRouteCapabilities?.native_word_timestamps ===
                        true
                      }
                      speakerAttributionHint={speakerAttributionHint}
                      timestampAlignerModelId={resolvedAlignerModel}
                      timestampAlignerReady={timestampAlignerReady}
                      timestampAlignerModelStatus={selectedAlignerModelStatus}
                      onOpenModelManager={openNewTranscriptionModelManager}
                      onModelRequired={() => {
                        if (isNewSpeakerAttributedAsrMode) {
                          openSpeakerAttributedAsrModelManager();
                          onError(
                            "Select and load a speaker-attributed ASR model to start SAA.",
                          );
                          return;
                        }
                        setModelModalContext("transcription");
                        requestModel();
                        onError(
                          "Select and load an ASR model to start transcribing.",
                        );
                      }}
                      onTimestampAlignerRequired={() => {
                        openTranscriptionModelManager();
                        onError("Load the timestamp aligner model to enable timestamps.");
                      }}
                      onBusyChange={setIsCreationBusy}
                      onCreated={async (createdRecord: TranscriptionRecord) => {
                        setStreamingRecord(createdRecord);
                        await refreshHistory().catch(() => undefined);
                        navigate(
                          isNewSpeakerAttributedAsrMode
                            ? `/transcription/${createdRecord.id}?mode=speaker_attributed_asr`
                            : `/transcription/${createdRecord.id}`,
                        );
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
                      onStreamingProgress={(progress) => {
                        setStreamingRecord((current) =>
                          current
                            ? {
                                ...current,
                                processing_status: "processing",
                                processing_progress: progress,
                              }
                            : current,
                        );
                      }}
                      onStreamingFinal={(finalRecord) => {
                        setStreamingRecord(finalRecord);
                      }}
                      onStreamingError={(message, createdRecordId) => {
                        setStreamingRecord((current) =>
                          current && current.id === createdRecordId
                            ? {
                                ...current,
                                processing_status: "failed",
                                processing_error: message,
                              }
                            : current,
                        );
                      }}
                      onStreamingDone={(createdRecordId) => {
                        void reconcileStreamedTranscription(createdRecordId);
                      }}
                    />
                  ) : (
                    <NewDiarizationModal
                      isOpen={isNewTranscriptionModalOpen}
                      onClose={handleCloseNewTranscriptionModal}
                      renderInDialog={false}
                      selectedMode={newSpeechTextMode}
                      onSelectMode={setNewSpeechTextMode}
                      pipelineMode={diarizationPipelineMode}
                      selectedModel={resolvedDiarizationModel}
                      selectedModelReady={diarizationModelReady}
                      diarizationModels={diarizationModels}
                      singleModelDiarizationModels={oneModelDiarizationModels}
                      onSelectModel={(variant) =>
                        onPickRoleModel?.("diarization", variant)
                      }
                      asrModels={diarizationAsrPipelineModels}
                      alignerModels={diarizationAlignerPipelineModels}
                      onSelectAsrModel={
                        onPickRoleModel
                          ? (variant) => onPickRoleModel("diarization-asr", variant)
                          : undefined
                      }
                      onSelectAlignerModel={
                        onPickRoleModel
                          ? (variant) =>
                              onPickRoleModel("diarization-aligner", variant)
                          : undefined
                      }
                      pipelineAsrModelId={resolvedDiarizationAsrModel}
                      pipelineAlignerModelId={resolvedDiarizationAlignerModel}
                      pipelineLlmModelId={resolvedDiarizationLlmModel}
                      pipelineModelsReady={diarizationPipelineModelsReady}
                      summaryModelReady={diarizationSummaryModelReady}
                      onModelRequired={() => {
                        openDiarizationModelManager();
                        onError("Select and load a diarization model to start.");
                      }}
                      onPipelineModelsRequired={() => {
                        openDiarizationModelManager();
                        onError(
                          "Load ASR and forced aligner models before diarization.",
                        );
                      }}
                      canUnloadAnyManagedModels={canUnloadAnyDiarizationManagedModels}
                      isManagedModelActionBusy={isDiarizationManagedModelActionBusy}
                      onOpenModelManager={openDiarizationModelManager}
                      onUnloadAllManagedModels={handleUnloadAllDiarizationManagedModels}
                      onBusyChange={setIsCreationBusy}
                      onCreated={handleCreatedDiarizationRecord}
                    />
                  )}
                </motion.div>
              </AnimatePresence>
            </DialogContent>
          </Dialog>
        </>
      )}

      <RouteModelModal
        isOpen={isModelModalOpen}
        onClose={closeModelModal}
        title={
          isDiarizationModelModal
            ? "Diarization models"
            : isSpeakerAttributedAsrModelModal
              ? "Speaker-attributed transcription models"
              : "Transcription models"
        }
        description={
          isDiarizationModelModal
            ? "Manage diarization pipeline models for /v1/diarizations."
            : isSpeakerAttributedAsrModelModal
              ? "Manage speaker-attributed ASR and summary models."
              : "Manage ASR models, the optional timestamp aligner, and the summary model for this route."
        }
        models={
          isDiarizationModelModal
            ? diarizationPipelineModels
            : isSpeakerAttributedAsrModelModal
              ? speakerAttributedAsrModalModels
              : transcriptionModalModels
        }
        loading={loading}
        selectedVariant={
          isDiarizationModelModal
            ? null
            : isSpeakerAttributedAsrModelModal
              ? resolvedSpeakerAttributedAsrModel
              : resolvedSelectedModel
        }
        intentVariant={
          isDiarizationModelModal || isSpeakerAttributedAsrModelModal
            ? null
            : intentVariant
        }
        downloadProgress={downloadProgress}
        onDownload={onDownload}
        onCancelDownload={onCancelDownload}
        onLoad={isDiarizationModelModal ? handleLoadDiarizationStackModel : onLoad}
        onUnload={onUnload}
        onDelete={onDelete}
        onUseModel={onSelect}
        emptyMessage={
          isDiarizationModelModal
            ? "No diarization pipeline models available for this route."
            : isSpeakerAttributedAsrModelModal
              ? "No speaker-attributed ASR models available for this route."
              : "No transcription models available for this route."
        }
        sections={
          isDiarizationModelModal
            ? diarizationModelSections
            : isSpeakerAttributedAsrModelModal
              ? speakerAttributedAsrModelSections
              : transcriptionModelSections
        }
        canUseModel={
          isDiarizationModelModal
            ? undefined
            : isSpeakerAttributedAsrModelModal
              ? (variant) =>
                  speakerAttributedAsrModels.some(
                    (model) => model.variant === variant,
                  )
              : (variant) =>
                  transcriptionModels.some((model) => model.variant === variant)
        }
        selectionMode={isDiarizationModelModal ? "manage" : "route"}
        zIndexClassName={
          isNewTranscriptionModalOpen ? "z-70" : "z-50"
        }
      />
    </PageShell>
  );
}
