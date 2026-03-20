import { useEffect, useMemo, useRef, useState } from "react";
import type { ModelInfo } from "@/api";
import { DiarizationPlayground } from "@/components/DiarizationPlayground";
import { PageHeader, PageShell } from "@/components/PageShell";
import { RouteModelModal } from "@/features/models/components/RouteModelModal";

interface DiarizationPageProps {
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

interface DiarizationModelGroup {
  key: string;
  title: string;
  description: string;
  models: ModelInfo[];
}

function resolvePreferredVariant(
  availableModels: ModelInfo[],
  preferredOrder: string[],
): string | null {
  for (const variant of preferredOrder) {
    const readyPreferred = availableModels.find(
      (model) => model.variant === variant && model.status === "ready",
    );
    if (readyPreferred) {
      return readyPreferred.variant;
    }
  }

  const readyModel = availableModels.find((model) => model.status === "ready");
  if (readyModel) {
    return readyModel.variant;
  }

  for (const variant of preferredOrder) {
    const preferred = availableModels.find((model) => model.variant === variant);
    if (preferred) {
      return preferred.variant;
    }
  }

  return availableModels[0]?.variant ?? null;
}

function isDiarizationVariant(variant: string): boolean {
  const normalized = variant.toLowerCase();
  return normalized.includes("sortformer") || normalized.includes("diar");
}

function isPipelineAsrVariant(variant: string): boolean {
  return variant.startsWith("Parakeet-TDT-");
}

function isPipelineAlignerVariant(variant: string): boolean {
  return variant === "Qwen3-ForcedAligner-0.6B";
}

function isPipelineLlmVariant(variant: string): boolean {
  return variant === "Qwen3-1.7B-GGUF";
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
}: DiarizationPageProps) {
  const [historyActionContainer, setHistoryActionContainer] =
    useState<HTMLDivElement | null>(null);
  const [isModelModalOpen, setIsModelModalOpen] = useState(false);
  const [modalIntentModel, setModalIntentModel] = useState<string | null>(null);
  const [autoCloseOnIntentReady, setAutoCloseOnIntentReady] = useState(false);
  const [isPipelineLoadAllRequested, setIsPipelineLoadAllRequested] =
    useState(false);
  const loadAllDownloadRequestedRef = useRef(new Set<string>());
  const loadAllLoadRequestedRef = useRef(new Set<string>());

  const diarizationModels = useMemo(
    () =>
      models
        .filter((model) => isDiarizationVariant(model.variant))
        .sort((a, b) => a.variant.localeCompare(b.variant)),
    [models],
  );

  const asrPipelineModels = useMemo(
    () =>
      models
        .filter((model) => isPipelineAsrVariant(model.variant))
        .sort((a, b) => a.variant.localeCompare(b.variant)),
    [models],
  );

  const alignerPipelineModels = useMemo(
    () =>
      models
        .filter((model) => isPipelineAlignerVariant(model.variant))
        .sort((a, b) => a.variant.localeCompare(b.variant)),
    [models],
  );

  const llmPipelineModels = useMemo(
    () =>
      models
        .filter((model) => isPipelineLlmVariant(model.variant))
        .sort((a, b) => a.variant.localeCompare(b.variant)),
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
        title: "Transcript Refiner",
        description: "LLM used to polish final diarized transcript output.",
        models: llmPipelineModels,
      },
    ],
    [asrPipelineModels, alignerPipelineModels, diarizationModels, llmPipelineModels],
  );

  const pipelineModels = useMemo(
    () => pipelineModelGroups.flatMap((group) => group.models),
    [pipelineModelGroups],
  );

  const preferredDiarizationModelOrder = ["diar_streaming_sortformer_4spk-v2.1"];
  const preferredAsrModelOrder = ["Parakeet-TDT-0.6B-v3", "Parakeet-TDT-0.6B-v2"];
  const preferredAlignerModelOrder = ["Qwen3-ForcedAligner-0.6B"];
  const preferredLlmModelOrder = ["Qwen3-1.7B-GGUF"];

  const resolvedSelectedModel =
    selectedModel &&
    diarizationModels.some((model) => model.variant === selectedModel)
      ? selectedModel
      : resolvePreferredVariant(diarizationModels, preferredDiarizationModelOrder);

  const selectedModelInfo =
    diarizationModels.find(
      (model) => model.variant === resolvedSelectedModel,
    ) ?? null;
  const selectedModelReady = selectedModelInfo?.status === "ready";

  const resolvedAsrModel = resolvePreferredVariant(
    asrPipelineModels,
    preferredAsrModelOrder,
  );
  const resolvedAlignerModel = resolvePreferredVariant(
    alignerPipelineModels,
    preferredAlignerModelOrder,
  );
  const resolvedLlmModel = resolvePreferredVariant(
    llmPipelineModels,
    preferredLlmModelOrder,
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
  const pipelineModelsReady = asrModelReady && alignerModelReady;

  const targetPipelineVariants = useMemo(
    () =>
      [
        resolvedSelectedModel,
        resolvedAsrModel,
        resolvedAlignerModel,
        resolvedLlmModel,
      ].filter((variant): variant is string => !!variant),
    [
      resolvedAlignerModel,
      resolvedAsrModel,
      resolvedLlmModel,
      resolvedSelectedModel,
    ],
  );

  const targetPipelineModels = useMemo(
    () =>
      targetPipelineVariants
        .map(
          (variant) =>
            pipelineModels.find((model) => model.variant === variant) ?? null,
        )
        .filter((model): model is ModelInfo => model !== null),
    [pipelineModels, targetPipelineVariants],
  );

  const pipelineAllLoaded =
    targetPipelineModels.length > 0 &&
    targetPipelineModels.every((model) => model.status === "ready");

  const pipelineLoadAllBusy =
    isPipelineLoadAllRequested ||
    targetPipelineModels.some(
      (model) => model.status === "downloading" || model.status === "loading",
    );

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
    setModalIntentModel(resolvedSelectedModel);
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

  const handleToggleLoadAllPipeline = () => {
    if (pipelineAllLoaded) {
      for (const model of targetPipelineModels) {
        if (model.status === "ready") {
          onUnload(model.variant);
        }
      }
      return;
    }

    loadAllDownloadRequestedRef.current.clear();
    loadAllLoadRequestedRef.current.clear();
    setIsPipelineLoadAllRequested(true);
  };

  useEffect(() => {
    if (!isPipelineLoadAllRequested) {
      return;
    }

    if (targetPipelineModels.length === 0) {
      setIsPipelineLoadAllRequested(false);
      return;
    }

    let allReady = true;
    let encounteredError = false;

    for (const model of targetPipelineModels) {
      if (model.status === "ready") {
        continue;
      }

      allReady = false;

      if (
        model.status === "error" &&
        loadAllDownloadRequestedRef.current.has(model.variant)
      ) {
        encounteredError = true;
      }

      if (
        (model.status === "not_downloaded" || model.status === "error") &&
        !loadAllDownloadRequestedRef.current.has(model.variant)
      ) {
        loadAllDownloadRequestedRef.current.add(model.variant);
        onDownload(model.variant);
      }

      if (
        model.status === "downloaded" &&
        !loadAllLoadRequestedRef.current.has(model.variant)
      ) {
        loadAllLoadRequestedRef.current.add(model.variant);
        onLoad(model.variant);
      }
    }

    if (allReady || encounteredError) {
      setIsPipelineLoadAllRequested(false);
      loadAllDownloadRequestedRef.current.clear();
      loadAllLoadRequestedRef.current.clear();
    }
  }, [isPipelineLoadAllRequested, onDownload, onLoad, targetPipelineModels]);

  return (
    <PageShell>
      <PageHeader
        title="Diarization"
        description="Separate speakers from audio streams and review timestamped transcript segments."
        actions={
          <div
            ref={setHistoryActionContainer}
            data-testid="page-header-history-slot"
            className="flex min-h-9 items-center"
          />
        }
      />

      <DiarizationPlayground
        selectedModel={resolvedSelectedModel}
        selectedModelReady={selectedModelReady}
        onOpenModelManager={openModelManager}
        onTogglePipelineLoadAll={handleToggleLoadAllPipeline}
        pipelineAllLoaded={pipelineAllLoaded}
        pipelineLoadAllBusy={pipelineLoadAllBusy}
        onModelRequired={() => {
          setModalIntentModel(resolvedSelectedModel);
          setAutoCloseOnIntentReady(true);
          setIsModelModalOpen(true);
          onError("Select and load a diarization model to start.");
        }}
        pipelineAsrModelId={resolvedAsrModel}
        pipelineAlignerModelId={resolvedAlignerModel}
        pipelineLlmModelId={resolvedLlmModel}
        pipelineLlmModelReady={llmModelReady}
        pipelineModelsReady={pipelineModelsReady}
        historyActionContainer={historyActionContainer}
        onPipelineModelsRequired={() => {
          openModelManagerForPipeline();
          onError("Load ASR and forced aligner models before diarization.");
        }}
      />

      <RouteModelModal
        isOpen={isModelModalOpen}
        onClose={closeModelModal}
        title="Diarization Models"
        description="Manage pipeline models for /v1/diarizations."
        models={pipelineModels}
        sections={pipelineModelGroups}
        canUseModel={isDiarizationVariant}
        loading={loading}
        selectedVariant={resolvedSelectedModel}
        intentVariant={modalIntentModel}
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
