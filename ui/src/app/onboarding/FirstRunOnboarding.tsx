import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  AudioLines,
  CheckCircle2,
  FileText,
  Loader2,
  MessageSquare,
  Sparkles,
} from "lucide-react";

import { api, type ModelInfo } from "@/api";
import { setAnalyticsEnabled } from "@/app/analytics/client";
import {
  trackAnalyticsConsentChanged,
  trackOnboardingCompleted,
  trackOnboardingViewed,
} from "@/app/analytics/events";
import { useModelCatalog } from "@/app/providers/ModelCatalogProvider";
import { useNotifications } from "@/app/providers/NotificationProvider";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogTitle,
} from "@/components/ui/dialog";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Switch } from "@/components/ui/switch";
import {
  MODEL_DETAILS,
  type ModelDetail,
} from "@/features/models/catalog/modelMetadata";
import { getModelStatusLabel } from "@/features/models/catalog/routeModelCatalog";
import { cn } from "@/lib/utils";
import { APP_ICON_URL } from "@/shared/config/runtime";

const QUICK_SETUP_VARIANTS = [
  "Parakeet-TDT-0.6B-v3",
  "Kokoro-82M",
  "Qwen3-1.7B-GGUF",
] as const;

const MANUAL_DOWNLOAD_VARIANTS = new Set(["Gemma-3-1b-it"]);

const FEATURE_ITEMS = [
  {
    icon: AudioLines,
    title: "Realtime voice sessions",
    description: "Low-latency speech input and playback for live interaction.",
  },
  {
    icon: MessageSquare,
    title: "Local chat models",
    description: "Run Qwen and other chat models with full local control.",
  },
  {
    icon: FileText,
    title: "Transcription and diarization",
    description:
      "One shared workspace for speech-to-text, timestamps, and speaker separation.",
  },
  {
    icon: Sparkles,
    title: "Voice Studio",
    description: "Clone, design, and manage reusable voices in one workspace.",
  },
];

const STEP_COPY = [
  {
    title: "Welcome to Izwi",
    description: "Local voice and chat tooling. Three short steps to get set.",
  },
  {
    title: "Set up your models",
    description:
      "Start with a recommended starter pack or pick exactly what you want.",
  },
  {
    title: "All setup",
    description:
      "You can always adjust models later from the Models workspace.",
  },
];

const GETTING_STARTED_ITEMS = [
  "Review the feature list",
  "Choose quick or custom setup",
  "Start using Izwi",
] as const;

type SetupMode = "quick" | "custom";
type OnboardingCategoryKey =
  | "chat"
  | "text-to-speech"
  | "voice-cloning"
  | "voice-design"
  | "transcription"
  | "diarization";

const CATEGORY_CONFIGS: Array<{
  key: OnboardingCategoryKey;
  title: string;
  description: string;
}> = [
  {
    key: "chat",
    title: "Chat",
    description: "Text models for chat and voice sessions.",
  },
  {
    key: "text-to-speech",
    title: "Text to Speech",
    description: "Built-in voices for fast speech synthesis.",
  },
  {
    key: "voice-cloning",
    title: "Voice Studio: Cloning",
    description: "Base models for reference voice cloning workflows.",
  },
  {
    key: "voice-design",
    title: "Voice Studio: Design",
    description: "Models for text-prompt voice design workflows.",
  },
  {
    key: "transcription",
    title: "Transcription",
    description: "ASR models for speech-to-text.",
  },
  {
    key: "diarization",
    title: "Diarization",
    description: "Speaker segmentation models.",
  },
];

function parseSize(sizeStr: string): number {
  const match = sizeStr.match(/^([\d.]+)\s*(GB|MB|KB|B)?$/i);
  if (!match) return 0;
  const value = parseFloat(match[1]);
  const unit = (match[2] || "B").toUpperCase();
  const multipliers: Record<string, number> = {
    B: 1,
    KB: 1024,
    MB: 1024 * 1024,
    GB: 1024 * 1024 * 1024,
  };
  return value * (multipliers[unit] || 1);
}

function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  if (bytes < 1024 * 1024 * 1024)
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  return `${(bytes / (1024 * 1024 * 1024)).toFixed(2)} GB`;
}

function resolveModelSizeBytes(model: ModelInfo): number {
  if (model.size_bytes && model.size_bytes > 0) {
    return model.size_bytes;
  }
  const metadata = MODEL_DETAILS[model.variant];
  return metadata ? parseSize(metadata.size) : 0;
}

function resolveModelSizeLabel(model: ModelInfo): string {
  const bytes = resolveModelSizeBytes(model);
  return bytes > 0 ? formatBytes(bytes) : "Size unknown";
}

function isDownloadable(model: ModelInfo): boolean {
  return model.status === "not_downloaded" || model.status === "error";
}

function resolveCategory(
  model: ModelInfo,
  metadata?: ModelDetail,
): OnboardingCategoryKey | null {
  if (!metadata) return null;

  if (metadata.category === "chat") {
    return "chat";
  }

  if (metadata.category === "tts") {
    if (model.variant.includes("VoiceDesign")) return "voice-design";
    if (model.variant.includes("Base")) return "voice-cloning";
    return "text-to-speech";
  }

  if (metadata.category === "asr") {
    const hasDiarization = metadata.capabilities.some(
      (capability) => capability.toLowerCase() === "diarization",
    );
    if (hasDiarization || model.variant.includes("diar_")) {
      return "diarization";
    }
    return "transcription";
  }

  return null;
}

export function FirstRunOnboarding() {
  const { models, loading, downloadModel } = useModelCatalog();
  const { notify } = useNotifications();
  const [isOpen, setIsOpen] = useState(false);
  const [step, setStep] = useState(0);
  const [setupMode, setSetupMode] = useState<SetupMode>("quick");
  const [analyticsOptIn, setAnalyticsOptIn] = useState(false);
  const [selectedVariants, setSelectedVariants] = useState<Set<string>>(
    new Set(),
  );
  const [isApplying, setIsApplying] = useState(false);
  const hasCompletedRef = useRef(false);
  const hasTrackedOnboardingViewRef = useRef(false);

  const stepCopy = STEP_COPY[Math.min(step, STEP_COPY.length - 1)];

  const modelLookup = useMemo(() => {
    const map = new Map<string, ModelInfo>();
    models.forEach((model) => {
      map.set(model.variant, model);
    });
    return map;
  }, [models]);

  const quickModels = useMemo(() => {
    return QUICK_SETUP_VARIANTS.map((variant) => modelLookup.get(variant)).filter(
      (model): model is ModelInfo => Boolean(model),
    );
  }, [modelLookup]);

  const quickDownloadTargets = useMemo(() => {
    return quickModels.filter(
      (model) =>
        isDownloadable(model) && !MANUAL_DOWNLOAD_VARIANTS.has(model.variant),
    );
  }, [quickModels]);

  const quickTotalBytes = useMemo(() => {
    return quickModels.reduce((total, model) => {
      return total + resolveModelSizeBytes(model);
    }, 0);
  }, [quickModels]);

  const categorizedModels = useMemo(() => {
    const buckets: Record<OnboardingCategoryKey, ModelInfo[]> = {
      chat: [],
      "text-to-speech": [],
      "voice-cloning": [],
      "voice-design": [],
      transcription: [],
      diarization: [],
    };

    models
      .filter((model) => !model.variant.includes("Tokenizer"))
      .forEach((model) => {
        const metadata = MODEL_DETAILS[model.variant];
        const category = resolveCategory(model, metadata);
        if (!category) {
          return;
        }
        buckets[category].push(model);
      });

    Object.values(buckets).forEach((bucket) => {
      bucket.sort((left, right) => {
        return resolveModelSizeBytes(left) - resolveModelSizeBytes(right);
      });
    });

    return buckets;
  }, [models]);

  const selectedModels = useMemo(() => {
    return Array.from(selectedVariants)
      .map((variant) => modelLookup.get(variant))
      .filter((model): model is ModelInfo => Boolean(model));
  }, [modelLookup, selectedVariants]);

  const selectedDownloadTargets = useMemo(() => {
    return selectedModels.filter(
      (model) =>
        isDownloadable(model) && !MANUAL_DOWNLOAD_VARIANTS.has(model.variant),
    );
  }, [selectedModels]);

  const selectedTotalBytes = useMemo(() => {
    return selectedModels.reduce((total, model) => {
      return total + resolveModelSizeBytes(model);
    }, 0);
  }, [selectedModels]);

  const selectedManualTargets = useMemo(() => {
    return selectedModels.filter((model) =>
      MANUAL_DOWNLOAD_VARIANTS.has(model.variant),
    );
  }, [selectedModels]);

  const activeModelCount =
    setupMode === "quick" ? quickModels.length : selectedModels.length;
  const activeDownloadCount =
    setupMode === "quick"
      ? quickDownloadTargets.length
      : selectedDownloadTargets.length;
  const activeTotalBytes =
    setupMode === "quick" ? quickTotalBytes : selectedTotalBytes;

  useEffect(() => {
    let active = true;

    api
      .getOnboardingState()
      .then((state) => {
        if (!active) {
          return;
        }
        if (!state.completed) {
          setStep(0);
          setSetupMode("quick");
          setAnalyticsOptIn(Boolean(state.analytics_opt_in));
          setAnalyticsEnabled(Boolean(state.analytics_opt_in));
          if (!hasTrackedOnboardingViewRef.current) {
            hasTrackedOnboardingViewRef.current = true;
            void trackOnboardingViewed();
          }
          setIsOpen(true);
        } else {
          setAnalyticsEnabled(Boolean(state.analytics_opt_in));
          setIsOpen(false);
        }
      })
      .catch((err) => {
        console.error("Failed to load onboarding state:", err);
      });

    return () => {
      active = false;
    };
  }, []);

  const markCompleted = useCallback(async () => {
    if (hasCompletedRef.current) {
      return;
    }
    try {
      await api.completeOnboarding();
      hasCompletedRef.current = true;
    } catch (err) {
      console.error("Failed to persist onboarding completion:", err);
      notify({
        title: "Could not save onboarding status",
        description: "We will retry later. You can continue using Izwi.",
        tone: "warning",
      });
    }
  }, [notify]);

  const persistAnalyticsPreference = useCallback(async () => {
    try {
      await api.updateAnalyticsPreference({ opt_in: analyticsOptIn });
      setAnalyticsEnabled(analyticsOptIn);
      await trackAnalyticsConsentChanged(
        analyticsOptIn ? "opted_in" : "opted_out",
        "onboarding",
      );
    } catch (err) {
      console.error("Failed to persist analytics preference:", err);
      notify({
        title: "Could not save analytics preference",
        description:
          "Your app setup will continue. You can retry later in Settings.",
        tone: "warning",
      });
    }
  }, [analyticsOptIn, notify]);

  const handleSkip = useCallback(async () => {
    await persistAnalyticsPreference();
    await markCompleted();
    await trackOnboardingCompleted("skip");
    setIsOpen(false);
  }, [markCompleted, persistAnalyticsPreference]);

  const toggleVariant = useCallback((variant: string) => {
    setSelectedVariants((prev) => {
      const next = new Set(prev);
      if (next.has(variant)) {
        next.delete(variant);
      } else {
        next.add(variant);
      }
      return next;
    });
  }, []);

  const handleApplySetup = useCallback(async () => {
    if (isApplying) {
      return;
    }
    setIsApplying(true);

    const chosenModels = setupMode === "quick" ? quickModels : selectedModels;
    const downloadTargets = chosenModels.filter(
      (model) =>
        isDownloadable(model) && !MANUAL_DOWNLOAD_VARIANTS.has(model.variant),
    );

    try {
      if (downloadTargets.length > 0) {
        await Promise.all(
          downloadTargets.map((model) => downloadModel(model.variant)),
        );
      }
    } catch (err) {
      console.error("Onboarding download failed:", err);
      notify({
        title: "Some downloads did not start",
        description: "You can retry from the Models page at any time.",
        tone: "warning",
      });
    } finally {
      await persistAnalyticsPreference();
      await markCompleted();
      await trackOnboardingCompleted(setupMode);
      setStep(2);
      setIsApplying(false);
    }
  }, [
    downloadModel,
    isApplying,
    markCompleted,
    notify,
    quickModels,
    selectedModels,
    setupMode,
    persistAnalyticsPreference,
  ]);

  const handleBack = useCallback(() => {
    setStep((current) => Math.max(0, current - 1));
  }, []);

  const handleNext = useCallback(() => {
    if (step === 0) {
      setStep(1);
      return;
    }
    if (step === 1) {
      void handleApplySetup();
      return;
    }
    if (step === 2) {
      setIsOpen(false);
    }
  }, [handleApplySetup, step]);

  if (!isOpen) {
    return null;
  }

  return (
    <Dialog open={isOpen}>
      <DialogContent
        className="max-w-[min(94vw,980px)] p-0 [&>button]:hidden"
        onEscapeKeyDown={(event) => event.preventDefault()}
        onPointerDownOutside={(event) => event.preventDefault()}
        onInteractOutside={(event) => event.preventDefault()}
      >
        <div className="flex flex-col">
          <div className="overflow-hidden border-b border-border/70 bg-[var(--bg-surface-1)] px-6 py-4 sm:px-8 sm:py-5">
            <div className="flex flex-col gap-3">
              <div className="flex flex-wrap items-center justify-between gap-4">
                <div className="flex flex-wrap items-center gap-3">
                  <div className="relative h-10 w-10 overflow-hidden rounded-[var(--radius-md)] border border-border/70 bg-[var(--bg-surface-2)] shadow-[var(--shadow-soft)]">
                    <img
                      src={APP_ICON_URL}
                      alt="Izwi logo"
                      className="h-full w-full object-cover p-1 brightness-110 contrast-110"
                    />
                  </div>
                  <DialogTitle className="text-xl font-semibold tracking-tight">
                    {stepCopy.title}
                  </DialogTitle>
                </div>
                <div className="text-xs text-[var(--text-muted)]">
                  Step {step + 1} of 3
                </div>
              </div>
              <DialogDescription className="text-sm text-[var(--text-muted)]">
                {stepCopy.description}
              </DialogDescription>
            </div>
          </div>

          <div className="space-y-6 px-6 py-6 sm:px-8 sm:py-8">
            {step === 0 ? (
              <div className="grid gap-6 lg:grid-cols-[1.2fr_0.8fr]">
                <div className="space-y-4">
                  <div>
                    <div className="text-lg font-semibold tracking-tight">
                      Core capabilities
                    </div>
                    <div className="mt-1 text-sm text-[var(--text-muted)]">
                      Everything below is available locally with full control.
                    </div>
                  </div>
                  <div className="grid gap-2.5">
                    {FEATURE_ITEMS.map((feature) => {
                      const Icon = feature.icon;
                      return (
                        <div
                          key={feature.title}
                          className="flex items-start gap-3 rounded-[var(--radius-md)] border border-border/70 bg-[var(--bg-surface-2)]/75 px-3.5 py-3"
                        >
                          <div className="mt-0.5 flex h-9 w-9 shrink-0 items-center justify-center rounded-[var(--radius-sm)] border border-border/70 bg-[var(--bg-surface-3)]">
                            <Icon className="h-4 w-4 text-[var(--text-primary)]" />
                          </div>
                          <div className="min-w-0">
                            <div className="text-base font-semibold leading-5 text-[var(--text-primary)]">
                              {feature.title}
                            </div>
                            <div className="mt-1 text-sm text-[var(--text-muted)]">
                              {feature.description}
                            </div>
                          </div>
                        </div>
                      );
                    })}
                  </div>
                </div>

                <div className="border-t border-border/70 pt-6 lg:border-l lg:border-t-0 lg:pl-8">
                  <div className="space-y-5">
                    <div>
                      <div className="text-lg font-semibold tracking-tight">
                        Getting started
                      </div>
                      <div className="mt-1 text-sm text-[var(--text-muted)]">
                        Setup takes a minute and can be changed later.
                      </div>
                    </div>

                    <div className="space-y-2.5">
                      {GETTING_STARTED_ITEMS.map((item, index) => (
                        <div key={item} className="flex items-center gap-2.5 text-sm">
                          <CheckCircle2
                            className={cn(
                              "h-4 w-4 shrink-0",
                              index === 0
                                ? "text-[var(--text-primary)]"
                                : "text-[var(--text-subtle)]",
                            )}
                          />
                          <span
                            className={cn(
                              index === 0
                                ? "text-[var(--text-primary)]"
                                : "text-[var(--text-secondary)]",
                            )}
                          >
                            {item}
                          </span>
                        </div>
                      ))}
                    </div>

                    <div className="border-t border-border/70 pt-4">
                      <div className="flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-between">
                        <span className="whitespace-nowrap text-base font-semibold leading-5 text-[var(--text-primary)]">
                          Share anonymous usage data
                        </span>
                        <div className="flex items-center gap-2 self-start sm:self-auto">
                          <span className="text-xs font-semibold uppercase tracking-[0.2em] text-[var(--text-muted)]">
                            {analyticsOptIn ? "On" : "Off"}
                          </span>
                          <Switch
                            checked={analyticsOptIn}
                            onCheckedChange={setAnalyticsOptIn}
                            aria-label="Share anonymous usage data"
                            className="border-border/70 data-[state=checked]:bg-[var(--checkbox-checked-bg)] data-[state=unchecked]:bg-[var(--bg-surface-3)]"
                          />
                        </div>
                      </div>
                      <div className="mt-2 text-sm leading-6 text-[var(--text-muted)]">
                        Help us improve Izwi with anonymous feature and model
                        usage metrics. We never collect prompts, transcripts,
                        audio, or personal data.
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            ) : null}

            {step === 1 ? (
              <div className="grid gap-6 lg:grid-cols-[1.2fr_0.8fr]">
                <div className="space-y-4">
                  <div>
                    <div className="text-lg font-semibold tracking-tight">
                      Choose model setup
                    </div>
                    <div className="mt-1 text-sm text-[var(--text-muted)]">
                      Start with a recommended pack or handpick models by
                      category.
                    </div>
                  </div>

                  <div className="inline-flex items-center gap-2 rounded-[var(--radius-pill)] border border-border/70 bg-[var(--bg-surface-2)]/70 p-1">
                    <Button
                      type="button"
                      size="sm"
                      className="h-8 px-3"
                      variant={setupMode === "quick" ? "default" : "ghost"}
                      onClick={() => setSetupMode("quick")}
                    >
                      Quick setup
                    </Button>
                    <Button
                      type="button"
                      size="sm"
                      className="h-8 px-3"
                      variant={setupMode === "custom" ? "default" : "ghost"}
                      onClick={() => setSetupMode("custom")}
                    >
                      Custom setup
                    </Button>
                  </div>

                  {setupMode === "quick" ? (
                    <div className="space-y-3">
                      <div className="flex flex-wrap items-center justify-between gap-2">
                        <div className="flex items-center gap-2 text-base font-semibold leading-5 text-[var(--text-primary)]">
                          <Sparkles className="h-4 w-4 text-[var(--text-primary)]" />
                          Recommended starter pack
                        </div>
                        <div className="text-xs uppercase tracking-[0.2em] text-[var(--text-subtle)]">
                          {quickTotalBytes > 0
                            ? `~${formatBytes(quickTotalBytes)}`
                            : "Size varies"}
                        </div>
                      </div>

                      <div className="grid gap-2.5">
                        {quickModels.map((model) => {
                          const details = MODEL_DETAILS[model.variant];
                          return (
                            <div
                              key={model.variant}
                              className="grid grid-cols-[minmax(0,1fr)_auto] items-start gap-3 rounded-[var(--radius-md)] border border-border/70 bg-[var(--bg-surface-2)]/75 px-3.5 py-3"
                            >
                              <div className="min-w-0">
                                <div className="text-base font-semibold leading-5 text-[var(--text-primary)]">
                                  {details?.shortName ?? model.variant}
                                </div>
                                <div className="mt-1 text-sm text-[var(--text-muted)]">
                                  {details?.description ?? "Recommended model"}
                                </div>
                              </div>
                              <div className="whitespace-nowrap text-right text-xs text-[var(--text-muted)]">
                                {resolveModelSizeLabel(model)}
                              </div>
                            </div>
                          );
                        })}
                      </div>

                      {quickModels.length === 0 ? (
                        <div className="rounded-[var(--radius-md)] border border-dashed border-border/70 bg-[var(--bg-surface-2)]/60 px-3 py-4 text-sm text-[var(--text-muted)]">
                          No recommended models are available yet.
                        </div>
                      ) : null}

                      <div className="text-sm text-[var(--text-muted)]">
                        Downloads run in the background. You can manage models
                        at any time from Models.
                      </div>

                      {quickDownloadTargets.length === 0 &&
                      quickModels.length > 0 ? (
                        <div className="flex items-center gap-2 text-sm text-[var(--status-positive-text)]">
                          <CheckCircle2 className="h-4 w-4" />
                          Starter models already downloaded.
                        </div>
                      ) : null}
                    </div>
                  ) : (
                    <div className="space-y-3">
                      <div className="flex flex-wrap items-center justify-between gap-3">
                        <div>
                          <div className="text-base font-semibold leading-5">
                            Choose your models
                          </div>
                          <div className="mt-1 text-sm text-[var(--text-muted)]">
                            Select as many as you want. You can change later.
                          </div>
                        </div>
                        <div className="text-xs uppercase tracking-[0.2em] text-[var(--text-subtle)]">
                          {selectedVariants.size} selected
                        </div>
                      </div>

                      <ScrollArea className="h-[min(46vh,360px)] pr-2">
                        <div className="space-y-5">
                          {CATEGORY_CONFIGS.map((category) => {
                            const modelsForCategory = categorizedModels[category.key];
                            if (!modelsForCategory || modelsForCategory.length === 0) {
                              return null;
                            }
                            return (
                              <div key={category.key} className="space-y-3">
                                <div>
                                  <div className="text-xs uppercase tracking-[0.2em] text-[var(--text-subtle)]">
                                    {category.title}
                                  </div>
                                  <div className="text-sm text-[var(--text-muted)]">
                                    {category.description}
                                  </div>
                                </div>
                                <div className="grid gap-2">
                                  {modelsForCategory.map((model) => {
                                    const metadata = MODEL_DETAILS[model.variant];
                                    const statusLabel = getModelStatusLabel(
                                      model.status,
                                    );
                                    const isManual = MANUAL_DOWNLOAD_VARIANTS.has(
                                      model.variant,
                                    );
                                    const isDisabled = isManual || !isDownloadable(model);
                                    return (
                                      <label
                                        key={model.variant}
                                        className={cn(
                                          "flex gap-3 rounded-[var(--radius-md)] border border-border/70 bg-[var(--bg-surface-2)]/75 p-3",
                                          isDisabled && "opacity-60",
                                        )}
                                      >
                                        <input
                                          type="checkbox"
                                          className="app-checkbox mt-1 h-4 w-4"
                                          checked={selectedVariants.has(model.variant)}
                                          onChange={() => toggleVariant(model.variant)}
                                          disabled={isDisabled}
                                        />
                                        <div className="min-w-0 flex-1">
                                          <div className="grid grid-cols-[minmax(0,1fr)_auto] items-start gap-2">
                                            <div className="text-base font-semibold leading-5 text-[var(--text-primary)]">
                                              {metadata?.shortName ?? model.variant}
                                            </div>
                                            <div className="whitespace-nowrap text-right text-xs text-[var(--text-muted)]">
                                              {resolveModelSizeLabel(model)}
                                            </div>
                                          </div>
                                          <div className="mt-1 text-sm text-[var(--text-muted)]">
                                            {metadata?.description ?? "Model option"}
                                          </div>
                                          <div className="mt-2 text-xs uppercase tracking-[0.2em] text-[var(--text-subtle)]">
                                            {isManual
                                              ? "Manual download"
                                              : statusLabel}
                                          </div>
                                        </div>
                                      </label>
                                    );
                                  })}
                                </div>
                              </div>
                            );
                          })}
                        </div>
                      </ScrollArea>

                      {loading ? (
                        <div className="flex items-center gap-2 text-sm text-[var(--text-muted)]">
                          <Loader2 className="h-4 w-4 animate-spin" />
                          Loading model catalog...
                        </div>
                      ) : null}
                    </div>
                  )}
                </div>

                <div className="border-t border-border/70 pt-6 lg:border-l lg:border-t-0 lg:pl-8">
                  <div className="space-y-5">
                    <div>
                      <div className="text-lg font-semibold tracking-tight">
                        Setup summary
                      </div>
                      <div className="mt-1 text-sm text-[var(--text-muted)]">
                        Review your selection before starting downloads.
                      </div>
                    </div>

                    <div className="grid gap-2">
                      <div className="flex items-center justify-between rounded-[var(--radius-sm)] border border-border/70 bg-[var(--bg-surface-2)]/70 px-3 py-2 text-sm">
                        <span className="text-[var(--text-muted)]">Mode</span>
                        <span className="font-semibold text-[var(--text-primary)]">
                          {setupMode === "quick" ? "Quick setup" : "Custom setup"}
                        </span>
                      </div>
                      <div className="flex items-center justify-between rounded-[var(--radius-sm)] border border-border/70 bg-[var(--bg-surface-2)]/70 px-3 py-2 text-sm">
                        <span className="text-[var(--text-muted)]">Models</span>
                        <span className="font-semibold text-[var(--text-primary)]">
                          {activeModelCount}
                        </span>
                      </div>
                      <div className="flex items-center justify-between rounded-[var(--radius-sm)] border border-border/70 bg-[var(--bg-surface-2)]/70 px-3 py-2 text-sm">
                        <span className="text-[var(--text-muted)]">
                          To download
                        </span>
                        <span className="font-semibold text-[var(--text-primary)]">
                          {activeDownloadCount}
                        </span>
                      </div>
                      <div className="flex items-center justify-between rounded-[var(--radius-sm)] border border-border/70 bg-[var(--bg-surface-2)]/70 px-3 py-2 text-sm">
                        <span className="text-[var(--text-muted)]">
                          Estimated size
                        </span>
                        <span className="font-semibold text-[var(--text-primary)]">
                          {activeTotalBytes > 0
                            ? formatBytes(activeTotalBytes)
                            : "Size varies"}
                        </span>
                      </div>
                    </div>

                    <div className="border-t border-border/70 pt-4">
                      <div className="space-y-2.5 text-sm">
                        <div className="flex items-center gap-2.5 text-[var(--text-secondary)]">
                          <CheckCircle2 className="h-4 w-4 text-[var(--text-primary)]" />
                          Review your model selections
                        </div>
                        <div className="flex items-center gap-2.5 text-[var(--text-secondary)]">
                          <CheckCircle2 className="h-4 w-4 text-[var(--text-subtle)]" />
                          Start downloads in the background
                        </div>
                        <div className="flex items-center gap-2.5 text-[var(--text-secondary)]">
                          <CheckCircle2 className="h-4 w-4 text-[var(--text-subtle)]" />
                          Continue to the app while setup completes
                        </div>
                      </div>
                    </div>

                    {setupMode === "custom" && selectedVariants.size === 0 ? (
                      <div className="rounded-[var(--radius-sm)] border border-border/70 bg-[var(--bg-surface-2)]/70 px-3 py-2 text-sm text-[var(--text-muted)]">
                        Select at least one model to queue downloads from custom
                        setup.
                      </div>
                    ) : null}

                    {setupMode === "custom" && selectedManualTargets.length > 0 ? (
                      <div className="rounded-[var(--radius-sm)] border border-border/70 bg-[var(--bg-surface-2)]/70 px-3 py-2 text-sm text-[var(--text-muted)]">
                        {selectedManualTargets.length} selected model
                        {selectedManualTargets.length === 1 ? "" : "s"} require
                        manual download from the Models page.
                      </div>
                    ) : null}
                  </div>
                </div>
              </div>
            ) : null}

            {step === 2 ? (
              <div className="mx-auto w-full max-w-2xl space-y-6">
                <div className="flex items-start gap-3">
                  <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-full border border-[var(--status-positive-border)] bg-[var(--status-positive-bg)]">
                    <CheckCircle2 className="h-5 w-5 text-[var(--status-positive-solid)]" />
                  </div>
                  <div>
                    <div className="text-lg font-semibold tracking-tight text-[var(--text-primary)]">
                      Setup complete
                    </div>
                    <div className="mt-1 text-sm text-[var(--text-muted)]">
                      You are ready to use Izwi. Downloads continue in the
                      background.
                    </div>
                  </div>
                </div>

                <div className="grid gap-2.5">
                  <div className="flex items-center gap-2.5 text-sm text-[var(--text-secondary)]">
                    <CheckCircle2 className="h-4 w-4 text-[var(--text-primary)]" />
                    Continue to the app now
                  </div>
                  <div className="flex items-center gap-2.5 text-sm text-[var(--text-secondary)]">
                    <CheckCircle2 className="h-4 w-4 text-[var(--text-subtle)]" />
                    Track download progress in Models
                  </div>
                  <div className="flex items-center gap-2.5 text-sm text-[var(--text-secondary)]">
                    <CheckCircle2 className="h-4 w-4 text-[var(--text-subtle)]" />
                    Update model choices any time later
                  </div>
                </div>

                <div className="border-t border-border/70 pt-4 text-sm text-[var(--text-muted)]">
                  Open the Models workspace later to pause downloads, add more
                  models, or retry any manual installs.
                </div>
              </div>
            ) : null}
          </div>

          <div className="flex flex-wrap items-center justify-between gap-3 border-t border-border/70 bg-[var(--bg-surface-1)] px-6 py-4 sm:px-8">
            <Button type="button" variant="ghost" onClick={handleSkip}>
              Skip setup
            </Button>
            <div className="flex items-center gap-2">
              <Button
                type="button"
                variant="outline"
                onClick={handleBack}
                disabled={step === 0 || isApplying}
              >
                Back
              </Button>
              <Button type="button" onClick={handleNext} disabled={isApplying}>
                {step === 2 ? "Go to app" : "Next"}
              </Button>
            </div>
          </div>
        </div>
      </DialogContent>
    </Dialog>
  );
}
