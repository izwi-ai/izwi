import { useCallback, useMemo, useState } from "react";

import type { ModelInfo } from "@/api";
import { PageShell } from "@/components/PageShell";
import { getSpeakerProfilesForVariant, VIEW_CONFIGS } from "@/types";
import { ChatPlayground } from "@/features/chat/components/ChatPlayground";
import type { ModelDownloadProgressMap } from "@/features/models/downloadProgress";
import {
  CHAT_PREFERRED_MODELS,
  getChatRouteModelLabel,
  resolvePreferredRouteModel,
  TEXT_TO_SPEECH_PREFERRED_MODELS,
} from "@/features/models/catalog/routeModelCatalog";
import { RouteModelModal } from "@/features/models/components/RouteModelModal";
import { useRouteModelSelection } from "@/features/models/hooks/useRouteModelSelection";

interface ChatPageProps {
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

  const modelSupportsThinking = useCallback(
    (variant: string) => {
      const model = models.find((candidate) => candidate.variant === variant);
      return model
        ? Boolean(model.chat_capabilities?.supports_thinking)
        : undefined;
    },
    [models],
  );

  const modelChatCapabilities = useCallback(
    (variant: string) =>
      models.find((candidate) => candidate.variant === variant)
        ?.chat_capabilities,
    [models],
  );

  const modelSupportsImages = useCallback(
    (variant: string) => {
      const model = models.find((candidate) => candidate.variant === variant);
      return model?.modalities
        ? model.modalities.includes("image_input")
        : undefined;
    },
    [models],
  );

  // Read aloud uses a loaded Text to Speech model with a built-in voice:
  // the one picked in the speech model manager, else the preferred one.
  const speechModels = useMemo(
    () =>
      models.filter((model) =>
        VIEW_CONFIGS["custom-voice"].modelFilter(model.variant),
      ),
    [models],
  );
  const [chosenSpeechModel, setChosenSpeechModel] = useState<string | null>(
    null,
  );
  const [isSpeechModelModalOpen, setIsSpeechModelModalOpen] = useState(false);
  const readAloudVoice = useMemo(() => {
    const readyModels = speechModels.filter(
      (model) =>
        model.status === "ready" &&
        getSpeakerProfilesForVariant(model.variant).length > 0,
    );
    const modelId = resolvePreferredRouteModel({
      models: readyModels,
      selectedModel: chosenSpeechModel,
      preferredVariants: TEXT_TO_SPEECH_PREFERRED_MODELS,
    });
    if (!modelId) {
      return null;
    }
    const speakers = getSpeakerProfilesForVariant(modelId);
    const speaker =
      speakers.find((profile) => profile.id === "Vivian")?.id ?? speakers[0].id;
    return { modelId, speaker };
  }, [chosenSpeechModel, speechModels]);

  return (
    <PageShell className="flex min-h-0 flex-col">
      <ChatPlayground
        selectedModel={resolvedSelectedModel}
        selectedModelReady={selectedModelReady}
        modelLabel={
          selectedModelInfo
            ? getChatRouteModelLabel(selectedModelInfo.variant)
            : null
        }
        supportsThinking={
          selectedModelInfo?.chat_capabilities?.supports_thinking ?? false
        }
        chatCapabilities={selectedModelInfo?.chat_capabilities ?? null}
        modelOptions={modelOptions}
        modelSupportsThinking={modelSupportsThinking}
        modelChatCapabilities={modelChatCapabilities}
        modelSupportsImages={modelSupportsImages}
        onSelectModel={handleModelSelect}
        onOpenModelManager={openModelManager}
        onModelRequired={() => {
          requestModel();
          onError("Select a model and load it to start chatting.");
        }}
        readAloudVoice={readAloudVoice}
        onOpenSpeechModels={() => setIsSpeechModelModalOpen(true)}
      />

      <RouteModelModal
        isOpen={isSpeechModelModalOpen}
        onClose={() => setIsSpeechModelModalOpen(false)}
        title="Speech Models"
        description="Load a Text to Speech model to read replies aloud."
        models={speechModels}
        loading={loading}
        selectedVariant={readAloudVoice?.modelId ?? null}
        downloadProgress={downloadProgress}
        onDownload={onDownload}
        onCancelDownload={onCancelDownload}
        onLoad={onLoad}
        onUnload={onUnload}
        onDelete={onDelete}
        onUseModel={(variant) => {
          setChosenSpeechModel(variant);
          setIsSpeechModelModalOpen(false);
        }}
        emptyMessage={VIEW_CONFIGS["custom-voice"].emptyStateDescription}
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
