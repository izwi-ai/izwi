import { useEffect, useMemo, useState } from "react";
import { Settings2 } from "lucide-react";
import { useNavigate, useSearchParams } from "react-router-dom";
import type { ModelInfo } from "@/api";
import { PageHeader, PageShell } from "@/components/PageShell";
import { Button } from "@/components/ui/button";
import { VoiceCreationModal } from "@/components/VoiceCreationModal";
import type { ModelDownloadProgressMap } from "@/features/models/downloadProgress";
import {
  TEXT_TO_SPEECH_PREFERRED_MODELS,
  VOICE_DESIGN_PREFERRED_MODELS,
  resolvePreferredRouteModel,
} from "@/features/models/catalog/routeModelCatalog";
import { RouteModelModal } from "@/features/models/components/RouteModelModal";
import { useRouteModelSelection } from "@/features/models/hooks/useRouteModelSelection";
import { useNewItemShortcut } from "@/hooks/useNewItemShortcut";
import { NewItemButton } from "@/components/history/NewItemButton";
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
  const [isCreationModalOpen, setIsCreationModalOpen] = useState(false);
  const [creationStep, setCreationStep] = useState<
    "choice" | "clone" | "design"
  >("choice");

  // /voice-cloning and /voice-design redirect here with ?tab=clone|design:
  // open the creation modal on that step, then drop the parameter so a
  // refresh or Back does not reopen it.
  const openNewVoice = () => {
    setCreationStep("choice");
    setIsCreationModalOpen(true);
  };
  useNewItemShortcut(openNewVoice);

  const requestedTab = searchParams.get("tab");
  useEffect(() => {
    if (requestedTab !== "clone" && requestedTab !== "design") {
      return;
    }
    setCreationStep(requestedTab);
    setIsCreationModalOpen(true);
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
    selectedModelReady: designModelReady,
    isModelModalOpen: isDesignModelModalOpen,
    intentVariant: designIntentVariant,
    closeModelModal: closeDesignModelModal,
    openModelManager: openDesignModelManager,
    requestModel: requestDesignModel,
    handleModelSelect: handleDesignModelSelect,
    modelOptions: designModelOptions,
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

  const preferredSavedVoiceModel = useMemo(() => {
    const savedVoiceModels = models.filter(
      (model) =>
        !model.variant.includes("Tokenizer") &&
        model.speech_capabilities?.supports_reference_voice === true,
    );
    if (savedVoiceModels.length === 0) {
      return null;
    }

    const selectedSavedVoiceModel =
      selectedModel &&
      savedVoiceModels.some((model) => model.variant === selectedModel)
        ? selectedModel
        : null;

    return resolvePreferredRouteModel({
      models: savedVoiceModels,
      selectedModel: selectedSavedVoiceModel,
      preferredVariants: TEXT_TO_SPEECH_PREFERRED_MODELS,
    });
  }, [models, selectedModel]);

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
            <NewItemButton label="New voice" onClick={openNewVoice} />
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

      <VoiceCreationModal
        open={isCreationModalOpen}
        onOpenChange={setIsCreationModalOpen}
        initialStep={creationStep}
        onVoiceCreated={() => {
          setVoicesRefreshKey((current) => current + 1);
        }}
        onUseSavedVoiceInTts={(voiceId) => {
          const params = new URLSearchParams();
          params.set("voiceId", voiceId);
          if (preferredSavedVoiceModel) {
            params.set("model", preferredSavedVoiceModel);
          }
          navigate(`/text-to-speech?${params.toString()}`);
        }}
        designModel={designResolvedModel}
        designModelReady={designModelReady}
        designModelOptions={designModelOptions}
        onSelectDesignModel={handleDesignModelSelect}
        onOpenDesignModelManager={openDesignModelManager}
        onDesignModelRequired={() => {
          requestDesignModel();
          onError("Select and load a VoiceDesign model to continue.");
        }}
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
