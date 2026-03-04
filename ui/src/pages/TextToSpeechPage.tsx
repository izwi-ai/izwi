import { useEffect, useMemo, useState } from "react";
import { ModelInfo } from "../api";
import { CustomVoicePlayground } from "../components/CustomVoicePlayground";
import { PageHeader, PageShell } from "../components/PageShell";
import { RouteModelModal } from "../components/RouteModelModal";
import { VIEW_CONFIGS } from "../types";

interface TextToSpeechPageProps {
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

export function TextToSpeechPage({
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
}: TextToSpeechPageProps) {
  const viewConfig = VIEW_CONFIGS["custom-voice"];
  const [isModelModalOpen, setIsModelModalOpen] = useState(false);
  const [modalIntentModel, setModalIntentModel] = useState<string | null>(null);
  const [autoCloseOnIntentReady, setAutoCloseOnIntentReady] = useState(false);

  const routeModels = useMemo(
    () =>
      models
        .filter((model) => !model.variant.includes("Tokenizer"))
        .filter((model) => viewConfig.modelFilter(model.variant))
        .sort((a, b) => a.variant.localeCompare(b.variant)),
    [models, viewConfig],
  );

  const preferredModelOrder = [
    "Kokoro-82M",
    "LFM2.5-Audio-1.5B-4bit",
    "LFM2.5-Audio-1.5B",
    "Qwen3-TTS-12Hz-1.7B-CustomVoice-4bit",
    "Qwen3-TTS-12Hz-0.6B-CustomVoice-4bit",
    "Qwen3-TTS-12Hz-0.6B-CustomVoice-8bit",
    "Qwen3-TTS-12Hz-0.6B-CustomVoice-bf16",
    "Qwen3-TTS-12Hz-0.6B-CustomVoice",
    "Qwen3-TTS-12Hz-1.7B-CustomVoice",
  ];

  const resolvedSelectedModel = (() => {
    if (selectedModel && routeModels.some((m) => m.variant === selectedModel)) {
      return selectedModel;
    }

    for (const variant of preferredModelOrder) {
      const readyPreferred = routeModels.find(
        (model) => model.variant === variant && model.status === "ready",
      );
      if (readyPreferred) {
        return readyPreferred.variant;
      }
    }

    const readyModel = routeModels.find((model) => model.status === "ready");
    if (readyModel) {
      return readyModel.variant;
    }

    for (const variant of preferredModelOrder) {
      const preferred = routeModels.find((model) => model.variant === variant);
      if (preferred) {
        return preferred.variant;
      }
    }

    return routeModels[0]?.variant ?? null;
  })();

  const selectedModelInfo =
    routeModels.find((model) => model.variant === resolvedSelectedModel) ??
    null;
  const selectedModelReady = selectedModelInfo?.status === "ready";

  useEffect(() => {
    if (!isModelModalOpen || !modalIntentModel || !autoCloseOnIntentReady) {
      return;
    }
    const targetModel = routeModels.find(
      (model) => model.variant === modalIntentModel,
    );
    if (targetModel?.status === "ready") {
      setIsModelModalOpen(false);
      setAutoCloseOnIntentReady(false);
    }
  }, [autoCloseOnIntentReady, isModelModalOpen, modalIntentModel, routeModels]);

  const closeModelModal = () => {
    setIsModelModalOpen(false);
    setAutoCloseOnIntentReady(false);
  };

  const openModelManager = () => {
    setModalIntentModel(resolvedSelectedModel);
    setAutoCloseOnIntentReady(false);
    setIsModelModalOpen(true);
  };

  const getStatusLabel = (status: ModelInfo["status"]): string => {
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
  };

  const modelOptions = routeModels.map((model) => ({
    value: model.variant,
    label: model.variant,
    statusLabel: getStatusLabel(model.status),
    isReady: model.status === "ready",
  }));

  const handleModelSelect = (variant: string) => {
    const model = routeModels.find((m) => m.variant === variant);
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
        title="Text to Speech"
        description="Generate natural speech from text with local voice models and reusable presets."
      />

      <CustomVoicePlayground
        selectedModel={resolvedSelectedModel}
        selectedModelReady={selectedModelReady}
        modelOptions={modelOptions}
        onSelectModel={handleModelSelect}
        onOpenModelManager={openModelManager}
        onModelRequired={() => {
          setModalIntentModel(resolvedSelectedModel);
          setAutoCloseOnIntentReady(true);
          setIsModelModalOpen(true);
          onError(
            "Select and load a CustomVoice, Kokoro, or LFM2.5 model to generate speech.",
          );
        }}
      />

      <RouteModelModal
        isOpen={isModelModalOpen}
        onClose={closeModelModal}
        title="Text-to-Speech Models"
        description="Manage CustomVoice, Kokoro, and LFM2.5 models for this route."
        models={routeModels}
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
        emptyMessage={viewConfig.emptyStateDescription}
      />
    </PageShell>
  );
}
