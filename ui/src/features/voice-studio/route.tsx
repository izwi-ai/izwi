import { useEffect, useState } from "react";
import { Settings2 } from "lucide-react";
import { useNavigate, useSearchParams } from "react-router-dom";
import type { ModelInfo } from "@/api";
import { PageHeader, PageShell } from "@/components/PageShell";
import { Button } from "@/components/ui/button";
import type { ModelDownloadProgressMap } from "@/features/models/downloadProgress";
import {
  VOICE_CLONING_PREFERRED_MODELS,
  VOICE_DESIGN_PREFERRED_MODELS,
  resolvePreferredRouteModel,
} from "@/features/models/catalog/routeModelCatalog";
import { RouteModelModal } from "@/features/models/components/RouteModelModal";
import { useRouteModelSelection } from "@/features/models/hooks/useRouteModelSelection";
import { useNewItemShortcut } from "@/hooks/useNewItemShortcut";
import { CloneVoiceDialog } from "@/features/voice-studio/components/CloneVoiceDialog";
import { DesignVoiceDialog } from "@/features/voice-studio/components/DesignVoiceDialog";
import {
  NewVoiceMenu,
  type NewVoiceKind,
} from "@/features/voice-studio/components/NewVoiceMenu";
import { VoicesPage } from "@/features/voices/route";
import { VIEW_CONFIGS } from "@/types";

interface VoiceStudioPageProps {
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

export function VoiceStudioPage({
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
}: VoiceStudioPageProps) {
  const navigate = useNavigate();
  const [searchParams, setSearchParams] = useSearchParams();
  const [isNewVoiceMenuOpen, setIsNewVoiceMenuOpen] = useState(false);
  const [newVoiceDialog, setNewVoiceDialog] = useState<NewVoiceKind | null>(
    null,
  );
  useNewItemShortcut(() => setIsNewVoiceMenuOpen(true));

  // /voice-cloning and /voice-design redirect here with ?tab=clone|design:
  // open that dialog, then drop the parameter so a refresh or Back does not
  // reopen it.
  const requestedTab = searchParams.get("tab");
  useEffect(() => {
    if (requestedTab !== "clone" && requestedTab !== "design") {
      return;
    }
    setNewVoiceDialog(requestedTab);
    const next = new URLSearchParams(searchParams);
    next.delete("tab");
    setSearchParams(next, { replace: true });
  }, [requestedTab, searchParams, setSearchParams]);
  const [voicesRefreshKey, setVoicesRefreshKey] = useState(0);
  const [voicesModelManagerRequestKey, setVoicesModelManagerRequestKey] =
    useState(0);
  const designViewConfig = VIEW_CONFIGS["voice-design"];
  const {
    routeModels: designRouteModels,
    resolvedSelectedModel: designResolvedModel,
    selectedModelInfo: designModelInfo,
    selectedModelReady: designModelReady,
    isModelModalOpen: isDesignModelModalOpen,
    intentVariant: designIntentVariant,
    closeModelModal: closeDesignModelModal,
    openModelManager: openDesignModelManager,
    requestModel: requestDesignModel,
  } = useRouteModelSelection({
    models,
    selectedModel,
    onSelect,
    modelFilter: designViewConfig.modelFilter,
    resolveSelectedModel: (routeModels, currentModel) =>
      resolvePreferredRouteModel({
        models: routeModels,
        selectedModel: currentModel,
        preferredVariants: VOICE_DESIGN_PREFERRED_MODELS,
      }),
  });

  // The models that speak with a saved voice: the clone dialog shows the one
  // a new voice will use, and "Use in Text to Speech" opens with it.
  const {
    routeModels: savedVoiceRouteModels,
    resolvedSelectedModel: savedVoiceModel,
    selectedModelInfo: savedVoiceModelInfo,
    selectedModelReady: savedVoiceModelReady,
    isModelModalOpen: isSavedVoiceModelModalOpen,
    intentVariant: savedVoiceIntentVariant,
    closeModelModal: closeSavedVoiceModelModal,
    openModelManager: openSavedVoiceModelManager,
    handleModelSelect: handleSavedVoiceModelSelect,
  } = useRouteModelSelection({
    models,
    selectedModel,
    onSelect,
    modelFilter: (variant) =>
      models.some(
        (model) =>
          model.variant === variant &&
          model.speech_capabilities?.supports_reference_voice === true,
      ),
    resolveSelectedModel: (routeModels, currentModel) =>
      resolvePreferredRouteModel({
        models: routeModels,
        selectedModel: currentModel,
        preferredVariants: VOICE_CLONING_PREFERRED_MODELS,
      }),
  });

  const handleVoiceSaved = () => {
    setVoicesRefreshKey((current) => current + 1);
  };
  const openVoiceInTextToSpeech = (voiceId: string) => {
    const params = new URLSearchParams();
    params.set("voiceId", voiceId);
    if (savedVoiceModel) {
      params.set("model", savedVoiceModel);
    }
    navigate(`/text-to-speech?${params.toString()}`);
  };
  const closeNewVoiceDialog = () => setNewVoiceDialog(null);

  return (
    <PageShell>
      <PageHeader
        title="Voices"
        description="Saved and built-in voices you can use across Text to Speech, Studio and Voice."
        actions={
          <>
            <Button
              type="button"
              variant="outline"
              size="sm"
              className="h-9 gap-2"
              onClick={() =>
                setVoicesModelManagerRequestKey((current) => current + 1)
              }
            >
              <Settings2 className="h-4 w-4" />
              Models
            </Button>
            <NewVoiceMenu
              open={isNewVoiceMenuOpen}
              onOpenChange={setIsNewVoiceMenuOpen}
              onSelect={setNewVoiceDialog}
            />
          </>
        }
      />

      <div className="pb-4 sm:pb-5">
        <VoicesPage
          models={models}
          selectedModel={selectedModel}
          loading={loading}
          downloadProgress={downloadProgress}
          onDownload={onDownload}
          onCancelDownload={onCancelDownload}
          onLoad={onLoad}
          onUnload={onUnload}
          onDelete={onDelete}
          onSelect={onSelect}
          onError={onError}
          embedded
          refreshKey={voicesRefreshKey}
          openModelManagerRequestKey={voicesModelManagerRequestKey}
        />
      </div>

      <CloneVoiceDialog
        open={newVoiceDialog === "clone"}
        onClose={closeNewVoiceDialog}
        model={savedVoiceModel}
        modelInfo={savedVoiceModelInfo}
        modelReady={savedVoiceModelReady}
        isModelManagerOpen={isSavedVoiceModelModalOpen}
        onOpenModelManager={openSavedVoiceModelManager}
        onSaved={handleVoiceSaved}
        onUseInTextToSpeech={openVoiceInTextToSpeech}
      />

      <DesignVoiceDialog
        open={newVoiceDialog === "design"}
        onClose={closeNewVoiceDialog}
        model={designResolvedModel}
        modelInfo={designModelInfo}
        modelReady={designModelReady}
        isModelManagerOpen={isDesignModelModalOpen}
        onOpenModelManager={openDesignModelManager}
        onModelRequired={() => {
          requestDesignModel();
          onError("Select and load a VoiceDesign model to continue.");
        }}
        onSaved={handleVoiceSaved}
        onUseInTextToSpeech={openVoiceInTextToSpeech}
      />

      <RouteModelModal
        isOpen={isSavedVoiceModelModalOpen}
        onClose={closeSavedVoiceModelModal}
        title="Voice Cloning Models"
        description="Manage the models that speak with saved and cloned voices."
        models={savedVoiceRouteModels}
        loading={loading}
        selectedVariant={savedVoiceModel}
        intentVariant={savedVoiceIntentVariant}
        downloadProgress={downloadProgress}
        onDownload={onDownload}
        onCancelDownload={onCancelDownload}
        onLoad={onLoad}
        onUnload={onUnload}
        onDelete={onDelete}
        onUseModel={handleSavedVoiceModelSelect}
        emptyMessage={VIEW_CONFIGS["voice-clone"].emptyStateDescription}
        zIndexClassName="z-[70]"
      />

      <RouteModelModal
        isOpen={isDesignModelModalOpen}
        onClose={closeDesignModelModal}
        title="Voice Design Models"
        description="Manage VoiceDesign models for voice creation."
        models={designRouteModels}
        loading={loading}
        selectedVariant={designResolvedModel}
        intentVariant={designIntentVariant}
        downloadProgress={downloadProgress}
        onDownload={onDownload}
        onCancelDownload={onCancelDownload}
        onLoad={onLoad}
        onUnload={onUnload}
        onDelete={onDelete}
        onUseModel={onSelect}
        emptyMessage={designViewConfig.emptyStateDescription}
        zIndexClassName="z-[70]"
      />
    </PageShell>
  );
}
