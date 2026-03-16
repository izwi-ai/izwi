import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  AudioLines,
  CheckCircle2,
  FileText,
  Loader2,
  MessageSquare,
  Sparkles,
  Users,
  Wand2,
} from "lucide-react";

import { api, type ModelInfo } from "@/api";
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
    description: "High-quality speech-to-text with timestamps and speakers.",
  },
  {
    icon: Users,
    title: "Voice cloning library",
    description: "Capture and reuse voices with reference audio.",
  },
  {
    icon: Wand2,
    title: "Voice design tools",
    description: "Create custom voices from natural-language prompts.",
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
    title: "Voice Cloning",
    description: "Base models for reference voice cloning.",
  },
  {
    key: "voice-design",
    title: "Voice Design",
    description: "Create voices from text descriptions.",
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
  const [selectedVariants, setSelectedVariants] = useState<Set<string>>(
    new Set(),
  );
  const [isApplying, setIsApplying] = useState(false);
  const hasCompletedRef = useRef(false);

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
          setIsOpen(true);
        } else {
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

  const handleSkip = useCallback(async () => {
    await markCompleted();
    setIsOpen(false);
  }, [markCompleted]);

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
      await markCompleted();
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
          <div className="relative overflow-hidden border-b border-border/70 bg-[var(--bg-surface-1)] px-6 py-4 sm:px-8 sm:py-5">
            <div className="pointer-events-none absolute inset-0 bg-[radial-gradient(circle_at_top,_rgba(255,255,255,0.06),_transparent_60%)]" />
            <div className="relative z-10 flex flex-col gap-3">
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
              <div className="grid gap-6 lg:grid-cols-[1.15fr_0.85fr]">
                <div className="space-y-4">
                  <div className="text-xs uppercase tracking-[0.3em] text-[var(--text-subtle)]">
                    Core capabilities
                  </div>
                  <div className="grid gap-3">
                    {FEATURE_ITEMS.map((feature) => {
                      const Icon = feature.icon;
                      return (
                        <div
                          key={feature.title}
                          className="flex items-start gap-3 rounded-[var(--radius-md)] border border-border/70 bg-[var(--bg-surface-2)]/70 p-3"
                        >
                          <div className="mt-1 flex h-9 w-9 shrink-0 items-center justify-center rounded-[var(--radius-sm)] border border-border/70 bg-[var(--bg-surface-3)]">
                            <Icon className="h-4 w-4 text-[var(--text-primary)]" />
                          </div>
                          <div>
                            <div className="text-sm font-semibold">
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
                <div className="flex h-full flex-col justify-between gap-4 rounded-[var(--radius-lg)] border border-border/70 bg-[var(--bg-surface-2)]/80 p-5">
                  <div className="space-y-3">
                    <div className="text-sm font-semibold">
                      What happens next
                    </div>
                    <div className="text-sm text-[var(--text-muted)]">
                      Pick a starter pack or curate your own models. You can
                      adjust everything later.
                    </div>
                  </div>
                  <div className="space-y-2 text-sm">
                    {[
                      "1. Review the feature list",
                      "2. Choose quick or custom setup",
                      "3. Start using Izwi",
                    ].map((item) => (
                      <div
                        key={item}
                        className="rounded-[var(--radius-sm)] border border-border/70 bg-[var(--bg-surface-3)]/70 px-3 py-2 text-[var(--text-secondary)]"
                      >
                        {item}
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            ) : null}

            {step === 1 ? (
              <div className="space-y-6">
                <div className="flex flex-wrap gap-2">
                  <Button
                    type="button"
                    variant={setupMode === "quick" ? "default" : "outline"}
                    onClick={() => setSetupMode("quick")}
                  >
                    Quick setup
                  </Button>
                  <Button
                    type="button"
                    variant={setupMode === "custom" ? "default" : "outline"}
                    onClick={() => setSetupMode("custom")}
                  >
                    Custom setup
                  </Button>
                </div>

                {setupMode === "quick" ? (
                  <div className="rounded-[var(--radius-lg)] border border-border/70 bg-[var(--bg-surface-2)]/80 p-5">
                    <div className="flex flex-wrap items-center justify-between gap-3">
                      <div className="flex items-center gap-2">
                        <Sparkles className="h-4 w-4 text-[var(--text-primary)]" />
                        <div className="text-sm font-semibold">
                          Recommended starter pack
                        </div>
                      </div>
                      <div className="text-xs uppercase tracking-[0.28em] text-[var(--text-subtle)]">
                        {quickTotalBytes > 0
                          ? `~${formatBytes(quickTotalBytes)}`
                          : "Size varies"}
                      </div>
                    </div>
                    <div className="mt-4 grid gap-2">
                      {quickModels.map((model) => {
                        const details = MODEL_DETAILS[model.variant];
                        return (
                          <div
                            key={model.variant}
                            className="flex flex-wrap items-center justify-between gap-3 rounded-[var(--radius-md)] border border-border/70 bg-[var(--bg-surface-3)]/70 px-3 py-2 text-sm"
                          >
                            <div>
                              <div className="font-semibold">
                                {details?.shortName ?? model.variant}
                              </div>
                              <div className="text-xs text-[var(--text-muted)]">
                                {details?.description ?? "Recommended model"}
                              </div>
                            </div>
                            <div className="text-xs text-[var(--text-muted)]">
                              {resolveModelSizeLabel(model)}
                            </div>
                          </div>
                        );
                      })}
                      {quickModels.length === 0 ? (
                        <div className="rounded-[var(--radius-md)] border border-dashed border-border/70 bg-[var(--bg-surface-3)]/60 px-3 py-4 text-sm text-[var(--text-muted)]">
                          No recommended models are available yet.
                        </div>
                      ) : null}
                    </div>
                    <div className="mt-4 text-sm text-[var(--text-muted)]">
                      Downloads run in the background. You can manage models at
                      any time from Models.
                    </div>
                    {quickDownloadTargets.length === 0 &&
                    quickModels.length > 0 ? (
                      <div className="mt-3 flex items-center gap-2 text-xs text-[var(--status-positive-text)]">
                        <CheckCircle2 className="h-4 w-4" />
                        Starter models already downloaded.
                      </div>
                    ) : null}
                  </div>
                ) : (
                  <div className="rounded-[var(--radius-lg)] border border-border/70 bg-[var(--bg-surface-2)]/80 p-5">
                    <div className="flex flex-wrap items-center justify-between gap-3">
                      <div>
                        <div className="text-sm font-semibold">
                          Choose your models
                        </div>
                        <div className="text-sm text-[var(--text-muted)]">
                          Select as many as you want. You can change later.
                        </div>
                      </div>
                      <div className="text-xs uppercase tracking-[0.28em] text-[var(--text-subtle)]">
                        {selectedVariants.size} selected
                      </div>
                    </div>
                    <div className="mt-4">
                      <ScrollArea className="h-[min(46vh,360px)] pr-2">
                        <div className="space-y-5">
                          {CATEGORY_CONFIGS.map((category) => {
                            const modelsForCategory =
                              categorizedModels[category.key];
                            if (!modelsForCategory || modelsForCategory.length === 0) {
                              return null;
                            }
                            return (
                              <div key={category.key} className="space-y-3">
                                <div>
                                  <div className="text-xs uppercase tracking-[0.3em] text-[var(--text-subtle)]">
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
                                          "flex gap-3 rounded-[var(--radius-md)] border border-border/70 bg-[var(--bg-surface-3)]/70 p-3",
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
                                          <div className="flex flex-wrap items-center justify-between gap-2">
                                            <div className="text-sm font-semibold">
                                              {metadata?.shortName ?? model.variant}
                                            </div>
                                            <div className="text-xs text-[var(--text-muted)]">
                                              {resolveModelSizeLabel(model)}
                                            </div>
                                          </div>
                                          <div className="mt-1 text-sm text-[var(--text-muted)]">
                                            {metadata?.description ?? "Model option"}
                                          </div>
                                          <div className="mt-2 text-xs uppercase tracking-[0.28em] text-[var(--text-subtle)]">
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
                    </div>
                    {loading ? (
                      <div className="mt-3 flex items-center gap-2 text-sm text-[var(--text-muted)]">
                        <Loader2 className="h-4 w-4 animate-spin" />
                        Loading model catalog...
                      </div>
                    ) : null}
                  </div>
                )}
              </div>
            ) : null}

            {step === 2 ? (
              <div className="flex flex-col items-center gap-4 rounded-[var(--radius-lg)] border border-border/70 bg-[var(--bg-surface-2)]/80 p-6 text-center">
                <div className="flex h-12 w-12 items-center justify-center rounded-full border border-[var(--status-positive-border)] bg-[var(--status-positive-bg)]">
                  <CheckCircle2 className="h-6 w-6 text-[var(--status-positive-solid)]" />
                </div>
                <div className="space-y-2">
                  <div className="text-lg font-semibold">
                    All setup, go to app
                  </div>
                  <div className="text-sm text-[var(--text-muted)]">
                    Downloads continue in the background. Open the Models page
                    to track progress or make changes later.
                  </div>
                </div>
                <div className="flex flex-wrap justify-center gap-2 text-xs uppercase tracking-[0.28em] text-[var(--text-subtle)]">
                  <span className="rounded-full border border-border/70 px-3 py-1">
                    Local models
                  </span>
                  <span className="rounded-full border border-border/70 px-3 py-1">
                    Background downloads
                  </span>
                  <span className="rounded-full border border-border/70 px-3 py-1">
                    Ready to explore
                  </span>
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
