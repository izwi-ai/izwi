import { useEffect, useMemo, useState } from "react";
import type { ModelInfo } from "@/api";
import { getModelStatusLabel } from "@/features/models/catalog/routeModelCatalog";

interface UseRouteModelSelectionOptions {
  models: ModelInfo[];
  selectedModel: string | null;
  onSelect: (variant: string) => void;
  modelFilter: (variant: string) => boolean;
  resolveSelectedModel: (
    routeModels: ModelInfo[],
    selectedModel: string | null,
  ) => string | null;
  getModelLabel?: (variant: string) => string;
}

export function useRouteModelSelection({
  models,
  selectedModel,
  onSelect,
  modelFilter,
  resolveSelectedModel,
  getModelLabel = (variant) => variant,
}: UseRouteModelSelectionOptions) {
  const [isModelModalOpen, setIsModelModalOpen] = useState(false);
  const [intentVariant, setIntentVariant] = useState<string | null>(null);
  const [autoCloseOnIntentReady, setAutoCloseOnIntentReady] = useState(false);

  const routeModels = useMemo(
    () =>
      models
        .filter((model) => !model.variant.includes("Tokenizer"))
        .filter((model) => modelFilter(model.variant))
        .sort((a, b) => a.variant.localeCompare(b.variant)),
    [modelFilter, models],
  );

  const resolvedSelectedModel = useMemo(
    () => resolveSelectedModel(routeModels, selectedModel),
    [resolveSelectedModel, routeModels, selectedModel],
  );

  const selectedModelInfo = useMemo(
    () =>
      routeModels.find((model) => model.variant === resolvedSelectedModel) ?? null,
    [resolvedSelectedModel, routeModels],
  );

  useEffect(() => {
    if (!isModelModalOpen || !intentVariant || !autoCloseOnIntentReady) {
      return;
    }

    const targetModel = routeModels.find(
      (model) => model.variant === intentVariant,
    );
    if (targetModel?.status === "ready") {
      setIsModelModalOpen(false);
      setAutoCloseOnIntentReady(false);
    }
  }, [autoCloseOnIntentReady, intentVariant, isModelModalOpen, routeModels]);

  const modelOptions = useMemo(
    () =>
      routeModels.map((model) => ({
        value: model.variant,
        label: getModelLabel(model.variant),
        statusLabel: getModelStatusLabel(model.status),
        isReady: model.status === "ready",
      })),
    [getModelLabel, routeModels],
  );

  const closeModelModal = () => {
    setIsModelModalOpen(false);
    setAutoCloseOnIntentReady(false);
  };

  const openModelManager = () => {
    setIntentVariant(resolvedSelectedModel);
    setAutoCloseOnIntentReady(false);
    setIsModelModalOpen(true);
  };

  const requestModel = (variant: string | null = resolvedSelectedModel) => {
    setIntentVariant(variant);
    setAutoCloseOnIntentReady(true);
    setIsModelModalOpen(true);
  };

  const handleModelSelect = (variant: string) => {
    const model = routeModels.find((entry) => entry.variant === variant);
    if (!model) {
      return;
    }

    onSelect(variant);

    if (model.status !== "ready") {
      requestModel(variant);
    }
  };

  return {
    routeModels,
    resolvedSelectedModel,
    selectedModelInfo,
    selectedModelReady: selectedModelInfo?.status === "ready",
    isModelModalOpen,
    intentVariant,
    closeModelModal,
    openModelManager,
    requestModel,
    handleModelSelect,
    modelOptions,
  };
}
