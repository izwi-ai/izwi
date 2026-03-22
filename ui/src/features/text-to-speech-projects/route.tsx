import { useEffect, useRef, useState } from "react";
import { useSearchParams } from "react-router-dom";
import { TextToSpeechProjectsWorkspace } from "@/components/TextToSpeechProjectsWorkspace";
import { PageHeader, PageShell } from "@/components/PageShell";
import {
  TEXT_TO_SPEECH_PREFERRED_MODELS,
  resolvePreferredRouteModel,
} from "@/features/models/catalog/routeModelCatalog";
import { RouteModelModal } from "@/features/models/components/RouteModelModal";
import { useRouteModelSelection } from "@/features/models/hooks/useRouteModelSelection";
import type { SharedPageProps } from "@/app/router/types";

export function TextToSpeechProjectsPage({
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
  const [searchParams] = useSearchParams();
  const [headerActionContainer, setHeaderActionContainer] =
    useState<HTMLDivElement | null>(null);
  const appliedQueryModelRef = useRef(false);
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
    modelFilter: (variant) => {
      const match = models.find((model) => model.variant === variant);
      const capabilities = match?.speech_capabilities;
      return Boolean(
        capabilities &&
          !variant.includes("Tokenizer") &&
          (capabilities.supports_builtin_voices ||
            capabilities.supports_reference_voice ||
            capabilities.supports_voice_description),
      );
    },
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
        title="Studio"
        description="Manage long-form text-to-speech scripts, render reusable segments, and export merged narration outputs."
        actions={
          <div
            ref={setHeaderActionContainer}
            data-testid="page-header-history-slot"
            className="flex min-h-9 items-center"
          />
        }
      />

      <TextToSpeechProjectsWorkspace
        selectedModel={resolvedSelectedModel}
        selectedModelInfo={selectedModelInfo}
        availableModels={routeModels}
        modelOptions={modelOptions}
        headerActionContainer={headerActionContainer}
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
