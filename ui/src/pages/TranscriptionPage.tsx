import { useEffect, useMemo, useState } from "react";
import { ModelInfo } from "../api";
import { PageHeader, PageShell } from "../components/PageShell";
import { RouteModelModal } from "../components/RouteModelModal";
import { TranscriptionPlayground } from "../components/TranscriptionPlayground";
import { VIEW_CONFIGS } from "../types";

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

function getStatusLabel(status: ModelInfo["status"]): string {
  switch (status) {
    case "ready":
      return "Loaded";
    case "loading":
      return "Loading";
    case "downloading":
      return "Downloading";
    case "downloaded":
      return "Downloaded";
    case "not_downloaded":
      return "Not downloaded";
    case "error":
      return "Error";
    default:
      return status;
  }
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
  const viewConfig = VIEW_CONFIGS.transcription;
  const [isModelModalOpen, setIsModelModalOpen] = useState(false);
  const [modalIntentModel, setModalIntentModel] = useState<string | null>(null);
  const [autoCloseOnIntentReady, setAutoCloseOnIntentReady] = useState(false);

  const transcriptionModels = useMemo(
    () =>
      models
        .filter((model) => !model.variant.includes("Tokenizer"))
        .filter((model) => viewConfig.modelFilter(model.variant))
        .sort((a, b) => a.variant.localeCompare(b.variant)),
    [models, viewConfig],
  );

  const preferredModelOrder = [
    "Parakeet-TDT-0.6B-v3",
    "Parakeet-TDT-0.6B-v2",
    "Whisper-Large-v3-Turbo",
    "LFM2.5-Audio-1.5B-4bit",
    "LFM2.5-Audio-1.5B",
    "Qwen3-ASR-0.6B",
    "Qwen3-ASR-1.7B-4bit",
    "Qwen3-ASR-1.7B",
  ];

  const resolvedSelectedModel = (() => {
    if (
      selectedModel &&
      transcriptionModels.some((model) => model.variant === selectedModel)
    ) {
      return selectedModel;
    }

    for (const variant of preferredModelOrder) {
      const preferred = transcriptionModels.find(
        (model) => model.variant === variant,
      );
      if (preferred) {
        return preferred.variant;
      }
    }

    const readyModel = transcriptionModels.find(
      (model) => model.status === "ready",
    );
    if (readyModel) {
      return readyModel.variant;
    }

    return transcriptionModels[0]?.variant ?? null;
  })();

  const selectedModelInfo =
    transcriptionModels.find(
      (model) => model.variant === resolvedSelectedModel,
    ) ?? null;
  const selectedModelReady = selectedModelInfo?.status === "ready";

  const closeModelModal = () => {
    setIsModelModalOpen(false);
    setAutoCloseOnIntentReady(false);
  };

  useEffect(() => {
    if (!isModelModalOpen || !modalIntentModel || !autoCloseOnIntentReady) {
      return;
    }
    const targetModel = transcriptionModels.find(
      (model) => model.variant === modalIntentModel,
    );
    if (targetModel?.status === "ready") {
      closeModelModal();
    }
  }, [
    autoCloseOnIntentReady,
    isModelModalOpen,
    modalIntentModel,
    transcriptionModels,
  ]);

  const openModelManager = () => {
    setModalIntentModel(resolvedSelectedModel);
    setAutoCloseOnIntentReady(false);
    setIsModelModalOpen(true);
  };

  const modelOptions = transcriptionModels.map((model) => ({
    value: model.variant,
    label: model.variant,
    statusLabel: getStatusLabel(model.status),
    isReady: model.status === "ready",
  }));

  const handleModelSelect = (variant: string) => {
    const model = transcriptionModels.find(
      (entry) => entry.variant === variant,
    );
    if (!model) {
      return;
    }

    onSelect(variant);

    if (model.status !== "ready") {
      setModalIntentModel(variant);
      setAutoCloseOnIntentReady(true);
      setIsModelModalOpen(true);
    }
  };

  return (
    <PageShell>
      <PageHeader
        title="Transcription"
        description="Capture audio, transcribe live, and browse saved transcription history."
      />

      <TranscriptionPlayground
        selectedModel={resolvedSelectedModel}
        selectedModelReady={selectedModelReady}
        modelOptions={modelOptions}
        onSelectModel={handleModelSelect}
        onOpenModelManager={openModelManager}
        onModelRequired={() => {
          setModalIntentModel(resolvedSelectedModel);
          setAutoCloseOnIntentReady(true);
          setIsModelModalOpen(true);
          onError("Select and load an ASR model to start transcribing.");
        }}
      />

      <RouteModelModal
        isOpen={isModelModalOpen}
        onClose={closeModelModal}
        title="Transcription Models"
        description="Manage ASR models for this route."
        models={transcriptionModels}
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
        emptyMessage="No transcription models available for this route."
      />
    </PageShell>
  );
}
