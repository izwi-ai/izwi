import type { ModelInfo } from "@/api";
import { PageHeader, PageShell } from "@/components/PageShell";
import { VIEW_CONFIGS } from "@/types";
import { ChatPlayground } from "@/features/chat/components/ChatPlayground";
import {
  CHAT_PREFERRED_MODELS,
  getChatRouteModelLabel,
  isThinkingChatModel,
  resolvePreferredRouteModel,
} from "@/features/models/catalog/routeModelCatalog";
import { RouteModelModal } from "@/features/models/components/RouteModelModal";
import { useRouteModelSelection } from "@/features/models/hooks/useRouteModelSelection";

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
  const {
    routeModels: chatModels,
    resolvedSelectedModel,
    selectedModelInfo,
    selectedModelReady,
    isModelModalOpen,
    intentVariant,
    closeModelModal,
    openModelManager,
    requestModel,
    handleModelSelect,
    modelOptions,
  } = useRouteModelSelection({
    models,
    selectedModel,
    onSelect,
    modelFilter: viewConfig.modelFilter,
    getModelLabel: getChatRouteModelLabel,
    resolveSelectedModel: (routeModels, currentModel) =>
      resolvePreferredRouteModel({
        models: routeModels,
        selectedModel: currentModel,
        preferredVariants: CHAT_PREFERRED_MODELS,
        preferAnyPreferredBeforeReadyAny: true,
      }),
  });

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
          selectedModelInfo
            ? getChatRouteModelLabel(selectedModelInfo.variant)
            : null
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
          requestModel();
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
        intentVariant={intentVariant}
        downloadProgress={downloadProgress}
        onDownload={onDownload}
        onCancelDownload={onCancelDownload}
        onLoad={onLoad}
        onUnload={onUnload}
        onDelete={onDelete}
        onUseModel={onSelect}
        getModelLabel={getChatRouteModelLabel}
        emptyMessage="No chat models available for this route."
      />
    </PageShell>
  );
}
