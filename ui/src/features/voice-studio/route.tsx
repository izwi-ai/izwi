import { useState } from "react";
import { Plus } from "lucide-react";
import { useNavigate } from "react-router-dom";
import type { ModelInfo } from "@/api";
import { PageHeader, PageShell } from "@/components/PageShell";
import { Button } from "@/components/ui/button";
import { VoiceCreationModal } from "@/components/VoiceCreationModal";
import {
  VOICE_DESIGN_PREFERRED_MODELS,
  resolvePreferredRouteModel,
} from "@/features/models/catalog/routeModelCatalog";
import { RouteModelModal } from "@/features/models/components/RouteModelModal";
import { useRouteModelSelection } from "@/features/models/hooks/useRouteModelSelection";
import { VoicesPage } from "@/features/voices/route";
import { VIEW_CONFIGS } from "@/types";

interface VoiceStudioPageProps {
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
  const [isCreationModalOpen, setIsCreationModalOpen] = useState(false);
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

  return (
    <PageShell>
      <PageHeader
        title="Voice Studio"
        description="Manage saved and built-in voices, then create new cloned or designed voices from one place."
        actions={
          <Button
            onClick={() => setIsCreationModalOpen(true)}
            className="h-9 rounded-[var(--radius-pill)] px-4 text-sm"
          >
            <Plus className="h-4 w-4" />
            New Voice
          </Button>
        }
      />

      <div className="mt-5 pb-4 sm:pb-5">
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
          onAddNewVoice={() => setIsCreationModalOpen(true)}
        />
      </div>

      <VoiceCreationModal
        open={isCreationModalOpen}
        onOpenChange={setIsCreationModalOpen}
        onUseSavedVoiceInTts={(voiceId) =>
          navigate(`/text-to-speech?voiceId=${encodeURIComponent(voiceId)}`)
        }
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
