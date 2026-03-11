import { useMemo, useState } from "react";
import type { ModelInfo } from "@/api";
import { PageHeader, PageShell } from "@/components/PageShell";
import { VIEW_CONFIGS } from "@/types";
import { TranscriptionPlayground } from "@/features/transcription/components/TranscriptionPlayground";
import {
  TRANSCRIPTION_PREFERRED_MODELS,
  resolvePreferredRouteModel,
} from "@/features/models/catalog/routeModelCatalog";
import { RouteModelModal } from "@/features/models/components/RouteModelModal";
import { useRouteModelSelection } from "@/features/models/hooks/useRouteModelSelection";

const TRANSCRIPTION_PREFERRED_ALIGNERS = ["Qwen3-ForcedAligner-0.6B"] as const;

function isTranscriptionAlignerVariant(variant: string): boolean {
  return variant === "Qwen3-ForcedAligner-0.6B";
}

interface TranscriptionPageProps {
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

export function TranscriptionPage({
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
}: TranscriptionPageProps) {
  const [historyActionContainer, setHistoryActionContainer] =
    useState<HTMLDivElement | null>(null);
  const viewConfig = VIEW_CONFIGS.transcription;
  const transcriptionAlignerModels = useMemo(
    () =>
      models
        .filter((model) => isTranscriptionAlignerVariant(model.variant))
        .sort((a, b) => a.variant.localeCompare(b.variant)),
    [models],
  );
  const {
    routeModels: transcriptionModels,
    resolvedSelectedModel,
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
    resolveSelectedModel: (routeModels, currentModel) =>
      resolvePreferredRouteModel({
        models: routeModels,
        selectedModel: currentModel,
        preferredVariants: TRANSCRIPTION_PREFERRED_MODELS,
        preferAnyPreferredBeforeReadyAny: true,
      }),
  });
  const resolvedAlignerModel = useMemo(
    () =>
      resolvePreferredRouteModel({
        models: transcriptionAlignerModels,
        selectedModel: null,
        preferredVariants: TRANSCRIPTION_PREFERRED_ALIGNERS,
        preferAnyPreferredBeforeReadyAny: true,
      }),
    [transcriptionAlignerModels],
  );
  const timestampAlignerReady =
    resolvedAlignerModel != null &&
    transcriptionAlignerModels.some(
      (model) =>
        model.variant === resolvedAlignerModel && model.status === "ready",
    );
  const modelSections = useMemo(
    () => [
      {
        key: "asr",
        title: "ASR Models",
        description: "Speech-to-text models available for transcription.",
        models: transcriptionModels,
      },
      {
        key: "aligner",
        title: "Timestamp Aligners",
        description:
          "Optional forced aligner models used when timestamped transcription is enabled.",
        models: transcriptionAlignerModels,
      },
    ],
    [transcriptionAlignerModels, transcriptionModels],
  );

  return (
    <PageShell>
      <PageHeader
        title="Transcription"
        description="Capture audio, transcribe live, add optional timestamps, and browse saved transcription history."
        actions={
          <div
            ref={setHistoryActionContainer}
            data-testid="page-header-history-slot"
            className="flex min-h-9 items-center"
          />
        }
      />

      <TranscriptionPlayground
        selectedModel={resolvedSelectedModel}
        selectedModelReady={selectedModelReady}
        modelOptions={modelOptions}
        onSelectModel={handleModelSelect}
        onOpenModelManager={openModelManager}
        onModelRequired={() => {
          requestModel();
          onError("Select and load an ASR model to start transcribing.");
        }}
        timestampAlignerModelId={resolvedAlignerModel}
        timestampAlignerReady={timestampAlignerReady}
        onTimestampAlignerRequired={() => {
          openModelManager();
          onError("Load the timestamp aligner model to enable timestamps.");
        }}
        historyActionContainer={historyActionContainer}
      />

      <RouteModelModal
        isOpen={isModelModalOpen}
        onClose={closeModelModal}
        title="Transcription Models"
        description="Manage ASR models and the optional timestamp aligner for this route."
        models={[...transcriptionModels, ...transcriptionAlignerModels]}
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
        emptyMessage="No transcription models available for this route."
        sections={modelSections}
        canUseModel={(variant) =>
          transcriptionModels.some((model) => model.variant === variant)
        }
      />
    </PageShell>
  );
}
