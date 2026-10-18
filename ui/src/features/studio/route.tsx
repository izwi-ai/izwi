import { useCallback, useEffect, useRef, useState } from "react";
import { useNavigate, useParams, useSearchParams } from "react-router-dom";
import { StudioWorkspace } from "@/components/StudioWorkspace";
import { usePageTitle } from "@/app/router/DocumentTitle";
import { PageHeader, PageShell } from "@/components/PageShell";
import {
  TEXT_TO_SPEECH_PREFERRED_MODELS,
  resolvePreferredRouteModel,
} from "@/features/models/catalog/routeModelCatalog";
import { RouteModelModal } from "@/features/models/components/RouteModelModal";
import { useRouteModelSelection } from "@/features/models/hooks/useRouteModelSelection";
import type { SharedPageProps } from "@/app/router/types";
import { isStudioRenderableModel } from "@/features/studio/modelFilter";

export function StudioPage({
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
}: SharedPageProps) {
  const navigate = useNavigate();
  const { projectId } = useParams<{ projectId?: string }>();
  const [searchParams] = useSearchParams();
  const modelQuery = searchParams.get("model");
  const [headerActionContainer, setHeaderActionContainer] =
    useState<HTMLDivElement | null>(null);
  const appliedQueryModelRef = useRef(false);
  const [projectTitle, setProjectTitle] = useState<string | null>(null);
  usePageTitle(projectId ? projectTitle : null);
  const handleNavigateProject = useCallback(
    (nextProjectId: string | null) => {
      const query = modelQuery ? `?model=${encodeURIComponent(modelQuery)}` : "";
      navigate(nextProjectId ? `/studio/${nextProjectId}${query}` : `/studio${query}`);
    },
    [modelQuery, navigate],
  );
  const {
    routeModels,
    resolvedSelectedModel,
    selectedModelInfo,
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
    modelFilter: (variant) =>
      isStudioRenderableModel(models.find((model) => model.variant === variant)),
    resolveSelectedModel: (availableRouteModels, currentModel) =>
      resolvePreferredRouteModel({
        models: availableRouteModels,
        selectedModel: currentModel,
        preferredVariants: TEXT_TO_SPEECH_PREFERRED_MODELS,
      }),
  });

  useEffect(() => {
    if (appliedQueryModelRef.current || routeModels.length === 0) {
      return;
    }

    const requestedModel = searchParams.get("model");
    if (
      requestedModel &&
      routeModels.some((model) => model.variant === requestedModel)
    ) {
      onSelect(requestedModel);
    }

    appliedQueryModelRef.current = true;
  }, [onSelect, routeModels, searchParams]);

  return (
    <PageShell>
      <PageHeader
        title={projectId ? (projectTitle ?? "Studio project") : "Studio"}
        breadcrumbs={
          projectId
            ? [
                { label: "Studio", to: "/studio" },
                { label: projectTitle ?? "Project" },
              ]
            : undefined
        }
        description={
          projectId
            ? undefined
            : "Turn long scripts into narrated audio, segment by segment."
        }
        actions={
          <div
            ref={setHeaderActionContainer}
            data-testid="page-header-history-slot"
            className="flex min-h-9 items-center"
          />
        }
      />

      <StudioWorkspace
        activeProjectId={projectId ?? null}
        selectedModel={resolvedSelectedModel}
        selectedModelInfo={selectedModelInfo}
        availableModels={routeModels}
        modelOptions={modelOptions}
        headerActionContainer={headerActionContainer}
        onNavigateProject={handleNavigateProject}
        onProjectTitleChange={setProjectTitle}
        onSelectModel={handleModelSelect}
        onOpenModelManager={openModelManager}
        onModelRequired={() => {
          requestModel();
          onError(
            "Select and load a compatible TTS model with built-in or saved-voice support to use Studio.",
          );
        }}
        onError={onError}
      />

      <RouteModelModal
        isOpen={isModelModalOpen}
        onClose={closeModelModal}
        title="Text-to-Speech Models"
        description="Manage built-in voice, saved-voice, and voice-direction TTS models for this route."
        models={routeModels}
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
        emptyMessage="Load a compatible TTS model with built-in voices, saved voices, or voice-direction prompts to generate speech."
      />
    </PageShell>
  );
}
