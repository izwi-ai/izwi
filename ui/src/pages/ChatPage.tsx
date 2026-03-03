import { useEffect, useMemo, useState } from "react";
import { ModelInfo } from "../api";
import { ChatPlayground } from "../components/ChatPlayground";
import { PageHeader, PageShell } from "../components/PageShell";
import { RouteModelModal } from "../components/RouteModelModal";
import { VIEW_CONFIGS } from "../types";
import { withQwen3Prefix } from "../utils/modelDisplay";

interface ChatPageProps {
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

function getChatModelName(variant: string): string {
  if (variant === "Qwen3-0.6B-GGUF") {
    return withQwen3Prefix("Chat 0.6B GGUF", variant);
  }
  if (variant === "Qwen3-1.7B-GGUF") {
    return withQwen3Prefix("Chat 1.7B GGUF", variant);
  }
  if (variant === "Qwen3-4B-GGUF") {
    return withQwen3Prefix("Chat 4B GGUF (Q4_K_M)", variant);
  }
  if (variant === "Qwen3-8B-GGUF") {
    return withQwen3Prefix("Chat 8B GGUF (Q4_K_M)", variant);
  }
  if (variant === "Qwen3.5-0.8B") {
    return withQwen3Prefix("Chat 0.8B GGUF (Q4_K_M)", variant);
  }
  if (variant === "Qwen3.5-2B") {
    return withQwen3Prefix("Chat 2B GGUF (Q4_K_M)", variant);
  }
  if (variant === "Qwen3.5-4B") {
    return withQwen3Prefix("Chat 4B GGUF (Q4_K_M)", variant);
  }
  if (variant === "Qwen3.5-9B") {
    return withQwen3Prefix("Chat 9B GGUF (Q4_K_M)", variant);
  }
  if (variant === "Gemma-3-1b-it") {
    return "Gemma 3 1B Instruct";
  }
  if (variant === "Gemma-3-4b-it") {
    return "Gemma 3 4B Instruct";
  }
  return variant;
}

function isThinkingChatModel(variant: string): boolean {
  const normalized = variant.trim().toLowerCase();
  return (
    (normalized.startsWith("qwen3-") || normalized.startsWith("qwen3.5-")) &&
    !normalized.includes("-asr-") &&
    !normalized.includes("-tts-") &&
    !normalized.includes("forcedaligner")
  );
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

const DEFAULT_CHAT_MODEL_VARIANT = "Qwen3-8B-GGUF";

export function ChatPage({
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
}: ChatPageProps) {
  const viewConfig = VIEW_CONFIGS.chat;
  const [isModelModalOpen, setIsModelModalOpen] = useState(false);
  const [modalIntentModel, setModalIntentModel] = useState<string | null>(null);
  const [autoCloseOnIntentReady, setAutoCloseOnIntentReady] = useState(false);

  const chatModels = useMemo(
    () =>
      models
        .filter((model) => viewConfig.modelFilter(model.variant))
        .sort((a, b) => a.variant.localeCompare(b.variant)),
    [models, viewConfig],
  );

  const resolvedSelectedModel = (() => {
    if (selectedModel && viewConfig.modelFilter(selectedModel)) {
      return selectedModel;
    }
    const preferredModel = chatModels.find(
      (model) => model.variant === DEFAULT_CHAT_MODEL_VARIANT,
    );
    if (preferredModel) {
      return preferredModel.variant;
    }
    const readyModel = chatModels.find((model) => model.status === "ready");
    if (readyModel) {
      return readyModel.variant;
    }
    return chatModels[0]?.variant ?? null;
  })();

  const selectedModelInfo =
    chatModels.find((model) => model.variant === resolvedSelectedModel) ?? null;
  const selectedModelReady = selectedModelInfo?.status === "ready";

  const closeModelModal = () => {
    setIsModelModalOpen(false);
    setAutoCloseOnIntentReady(false);
  };

  useEffect(() => {
    if (!isModelModalOpen || !modalIntentModel || !autoCloseOnIntentReady) {
      return;
    }
    const targetModel = chatModels.find(
      (model) => model.variant === modalIntentModel,
    );
    if (targetModel?.status === "ready") {
      closeModelModal();
    }
  }, [autoCloseOnIntentReady, chatModels, isModelModalOpen, modalIntentModel]);

  const handleModelSelect = (variant: string) => {
    const model = chatModels.find((entry) => entry.variant === variant);
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

  const openModelManager = () => {
    setModalIntentModel(resolvedSelectedModel);
    setAutoCloseOnIntentReady(false);
    setIsModelModalOpen(true);
  };

  const modelOptions = chatModels.map((model) => ({
    value: model.variant,
    label: getChatModelName(model.variant),
    statusLabel: getStatusLabel(model.status),
    isReady: model.status === "ready",
  }));

  return (
    <PageShell>
      <PageHeader
        title="Chat"
        description="Run local text conversations with selectable reasoning-capable models."
      />
      <ChatPlayground
        selectedModel={resolvedSelectedModel}
        selectedModelReady={selectedModelReady}
        modelLabel={
          selectedModelInfo ? getChatModelName(selectedModelInfo.variant) : null
        }
        supportsThinking={
          resolvedSelectedModel
            ? isThinkingChatModel(resolvedSelectedModel)
            : false
        }
        modelOptions={modelOptions}
        onSelectModel={handleModelSelect}
        onOpenModelManager={openModelManager}
        onModelRequired={() => {
          setModalIntentModel(resolvedSelectedModel);
          setAutoCloseOnIntentReady(true);
          setIsModelModalOpen(true);
          onError("Select a model and load it to start chatting.");
        }}
      />

      <RouteModelModal
        isOpen={isModelModalOpen}
        onClose={closeModelModal}
        title="Chat Models"
        description="Select, download, load, and unload chat models."
        models={chatModels}
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
        getModelLabel={getChatModelName}
        emptyMessage="No chat models available for this route."
      />
    </PageShell>
  );
}
