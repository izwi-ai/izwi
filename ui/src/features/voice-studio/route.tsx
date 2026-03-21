import { useMemo, useState } from "react";
import { useNavigate, useSearchParams } from "react-router-dom";
import type { ModelInfo } from "@/api";
import { VoiceCaptureWorkspace } from "@/components/VoiceCaptureWorkspace";
import { VoiceDesignWorkspace } from "@/components/VoiceDesignWorkspace";
import { PageHeader, PageShell } from "@/components/PageShell";
import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { VoicesPage } from "@/features/voices/route";
import {
  VOICE_DESIGN_PREFERRED_MODELS,
  resolvePreferredRouteModel,
} from "@/features/models/catalog/routeModelCatalog";
import { RouteModelModal } from "@/features/models/components/RouteModelModal";
import { useRouteModelSelection } from "@/features/models/hooks/useRouteModelSelection";
import { VIEW_CONFIGS } from "@/types";

type VoiceStudioTab = "library" | "clone" | "design";

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

function resolveTab(value: string | null): VoiceStudioTab {
  if (value === "clone" || value === "design" || value === "library") {
    return value;
  }
  return "library";
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
  const [searchParams, setSearchParams] = useSearchParams();
  const navigate = useNavigate();
  const activeTab = resolveTab(searchParams.get("tab"));
  const [designHistoryActionContainer, setDesignHistoryActionContainer] =
    useState<HTMLDivElement | null>(null);

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

  const setStudioTab = (nextTab: VoiceStudioTab) => {
    const nextSearchParams = new URLSearchParams(searchParams);
    nextSearchParams.set("tab", nextTab);
    setSearchParams(nextSearchParams, { replace: true });
  };

  const description = useMemo(() => {
    if (activeTab === "clone") {
      return "Capture voice references with transcript and save reusable voice profiles for Text to Speech.";
    }
    if (activeTab === "design") {
      return "Create new voices from natural-language prompts, compare candidates, and save your best option.";
    }
    return "Manage saved and built-in voices, then move seamlessly into cloning and voice design workflows.";
  }, [activeTab]);

  const headerActions =
    activeTab === "design" ? (
      <div
        ref={setDesignHistoryActionContainer}
        data-testid="page-header-history-slot"
        className="flex min-h-9 items-center"
      />
    ) : undefined;

  return (
    <PageShell>
      <PageHeader title="Voice Studio" description={description} actions={headerActions} />

      <Tabs
        value={activeTab}
        onValueChange={(value) => setStudioTab(resolveTab(value))}
        className="w-full"
      >
        <TabsList className="grid h-10 w-full max-w-[22rem] grid-cols-3 overflow-hidden rounded-[var(--radius-pill)] border-[var(--border-strong)] bg-[var(--bg-surface-2)] p-[2px] shadow-none">
          <TabsTrigger
            value="library"
            className="h-full rounded-[var(--radius-pill)] px-3 text-[13px] font-semibold text-[var(--text-muted)] data-[state=active]:bg-[var(--bg-surface-1)] data-[state=active]:text-[var(--text-primary)] data-[state=active]:shadow-none"
          >
            Library
          </TabsTrigger>
          <TabsTrigger
            value="clone"
            className="h-full rounded-[var(--radius-pill)] px-3 text-[13px] font-semibold text-[var(--text-muted)] data-[state=active]:bg-[var(--bg-surface-1)] data-[state=active]:text-[var(--text-primary)] data-[state=active]:shadow-none"
          >
            Clone
          </TabsTrigger>
          <TabsTrigger
            value="design"
            className="h-full rounded-[var(--radius-pill)] px-3 text-[13px] font-semibold text-[var(--text-muted)] data-[state=active]:bg-[var(--bg-surface-1)] data-[state=active]:text-[var(--text-primary)] data-[state=active]:shadow-none"
          >
            Design
          </TabsTrigger>
        </TabsList>
      </Tabs>

      <div className="mt-5 pb-4 sm:pb-5">
        {activeTab === "library" ? (
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
            onAddNewVoice={() => setStudioTab("design")}
          />
        ) : null}

        {activeTab === "clone" ? (
          <VoiceCaptureWorkspace
            onUseInTts={(voiceId) =>
              navigate(`/text-to-speech?voiceId=${encodeURIComponent(voiceId)}`)
            }
          />
        ) : null}

        {activeTab === "design" ? (
          <VoiceDesignWorkspace
            selectedModel={designResolvedModel}
            selectedModelReady={designModelReady}
            modelOptions={designModelOptions}
            onSelectModel={handleDesignModelSelect}
            onOpenModelManager={openDesignModelManager}
            onModelRequired={() => {
              requestDesignModel();
              onError("Select and load a VoiceDesign model to continue.");
            }}
            onUseInTts={(voiceId) =>
              navigate(`/text-to-speech?voiceId=${encodeURIComponent(voiceId)}`)
            }
            historyActionContainer={designHistoryActionContainer}
          />
        ) : null}
      </div>

      <RouteModelModal
        isOpen={isDesignModelModalOpen}
        onClose={closeDesignModelModal}
        title="Voice Design Models"
        description="Manage VoiceDesign models for this route."
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
      />
    </PageShell>
  );
}
