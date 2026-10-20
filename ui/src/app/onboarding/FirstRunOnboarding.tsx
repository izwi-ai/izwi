import {
  useCallback,
  useEffect,
  useId,
  useMemo,
  useRef,
  useState,
  type ReactNode,
} from "react";
import { Link } from "react-router-dom";
import { Check, ChevronRight, Cpu, Loader2 } from "lucide-react";

import { api, type ModelInfo, type ModelStorage } from "@/api";
import { setAnalyticsEnabled } from "@/app/analytics/client";
import {
  trackAnalyticsConsentChanged,
  trackOnboardingCompleted,
  trackOnboardingViewed,
} from "@/app/analytics/events";
import { NAV_GROUPS } from "@/app/layouts/navigation";
import { EngineOffline } from "@/app/onboarding/EngineOffline";
import {
  BackButton,
  OnboardingFrame,
  OnboardingStage,
  PrimaryButton,
  StageHeading,
  deviceNoun,
  linkButtonClass,
} from "@/app/onboarding/OnboardingFrame";
import {
  DEFAULT_UNCHECKED_TASKS,
  buildTaskRows,
  downloadBytes,
  downloadPlan,
  fitsDevice,
  groupModels,
  hardwareLabel,
  isOnDisk,
  modelDescriptor,
  modelName,
  modelRole,
  pageChoices,
  readyTasks,
  taskDetail,
  taskSizeLabels,
  uniqueModels,
  type DownloadPlan,
  type SizeLabel,
  type TaskId,
} from "@/app/onboarding/firstRunPlan";
import { useModelCatalog } from "@/app/providers/ModelCatalogProvider";
import { useNotifications } from "@/app/providers/NotificationProvider";
import { SetupRow, downloadPercent, type SetupRowTone } from "@/components/SetupRow";
import { Switch } from "@/components/ui/switch";
import {
  formatModelBytes,
  neededMemoryLabel,
} from "@/features/models/components/ModelListRow";
import { cn } from "@/lib/utils";
import { APP_ICON_URL } from "@/shared/config/runtime";
import { useRuntimeHealth } from "@/shared/hooks/useRuntimeHealth";

type Step = "welcome" | "setup" | "choose" | "progress" | "done";

const STEP_BARS: Record<Step, 1 | 2 | 3> = {
  welcome: 1,
  setup: 2,
  choose: 2,
  progress: 3,
  done: 3,
};

function sizeText(label: SizeLabel | undefined): string {
  switch (label?.kind) {
    case "bytes":
      return formatModelBytes(label.bytes);
    case "included":
      return "Included";
    case "downloading":
      return "Downloading";
    case "downloaded":
      return "Downloaded";
    default:
      return "";
  }
}

/** A checkbox row: a task, or one model on "Choose models". */
function ChoiceRow({
  title,
  detail,
  trailing,
  checked,
  disabled = false,
  dimmed = false,
  onToggle,
}: {
  title: string;
  detail: string;
  trailing: string;
  checked: boolean;
  disabled?: boolean;
  /** Can't be used on this device. */
  dimmed?: boolean;
  onToggle: () => void;
}) {
  const id = useId();
  return (
    <li className="grid grid-cols-[auto_minmax(0,1fr)_auto] items-center gap-3.5 border-b border-[var(--border-muted)] py-3 [@media(max-height:760px)]:py-2">
      <input
        id={id}
        type="checkbox"
        checked={checked}
        disabled={disabled}
        onChange={onToggle}
        aria-labelledby={`${id}-title`}
        aria-describedby={`${id}-detail`}
        className={cn(
          "app-checkbox h-4 w-4",
          disabled && "cursor-not-allowed",
          dimmed && "opacity-40",
        )}
      />
      <label
        htmlFor={id}
        className={cn(
          "flex min-w-0 flex-col gap-0.5",
          disabled ? "cursor-default" : "cursor-pointer",
        )}
      >
        <span
          id={`${id}-title`}
          className={cn(
            "text-sm font-medium",
            dimmed ? "text-[var(--text-subtle)]" : "text-[var(--text-primary)]",
          )}
        >
          {title}
        </span>
        <span
          id={`${id}-detail`}
          className={cn(
            "truncate text-[13px]",
            dimmed ? "text-[var(--text-subtle)]" : "text-[var(--text-muted)]",
          )}
        >
          {detail}
        </span>
      </label>
      <span
        className={cn(
          "whitespace-nowrap text-[13px] tabular-nums",
          dimmed ? "text-[var(--text-subtle)]" : "text-[var(--text-muted)]",
        )}
      >
        {trailing}
      </span>
    </li>
  );
}

function LoadingModels() {
  return (
    <div
      role="status"
      className="flex items-center gap-2 border-y border-[var(--border-muted)] py-6 text-[13px] text-[var(--text-muted)]"
    >
      <Loader2 aria-hidden="true" className="h-4 w-4 animate-spin" />
      Loading models…
    </div>
  );
}

/** "Needs 8.8 GB of free disk space.", once both numbers are known. */
function diskLine(
  plan: DownloadPlan,
  storage: ModelStorage | null,
): { text: string; short: boolean } | null {
  const free = storage?.free_bytes;
  if (plan.fetch.length === 0 || plan.diskBytes === null || free == null) {
    return null;
  }
  const needs = `Needs ${formatModelBytes(plan.diskBytes)} of free disk space.`;
  return plan.diskBytes > free
    ? { text: `${needs} ${formatModelBytes(free)} is free.`, short: true }
    : { text: needs, short: false };
}

function progressStatus(
  model: ModelInfo,
  percent: number,
  cancelled: boolean,
): { text: string; tone: SetupRowTone; percent?: number; failed?: boolean } {
  if (isOnDisk(model)) return { text: "Ready", tone: "done" };
  if (model.status === "downloading") {
    return { text: `Downloading · ${percent}%`, tone: "active", percent };
  }
  if (cancelled && model.status === "not_downloaded") {
    return { text: "Cancelled", tone: "muted", failed: true };
  }
  return { text: "Couldn't download", tone: "error", failed: true };
}

/**
 * First run, in the whole window: Welcome, Set up Izwi for this Mac (tasks,
 * or Choose models), Setting up and You're all set. Setup counts as done when
 * downloads start or on Skip; there's no going back after that.
 */
export function FirstRunOnboarding({
  initialAnalyticsOptIn,
  onComplete,
  onExit,
}: {
  initialAnalyticsOptIn: boolean;
  /** Records that first run is done; safe to call more than once. */
  onComplete: () => void;
  /** Leaves first run for the app, optionally on a page. */
  onExit: (path?: string) => void;
}) {
  const {
    models,
    loading,
    catalogError,
    downloadProgress,
    downloadModel,
    cancelModelDownload,
    refreshModels,
    selectModel,
  } = useModelCatalog();
  const { notify } = useNotifications();
  const healthState = useRuntimeHealth();
  const device = deviceNoun();

  const [step, setStep] = useState<Step>("welcome");
  const [analyticsOptIn, setAnalyticsOptIn] = useState(initialAnalyticsOptIn);
  const [busy, setBusy] = useState<"start" | "skip" | null>(null);
  const [uncheckedTasks, setUncheckedTasks] = useState<ReadonlySet<TaskId>>(
    DEFAULT_UNCHECKED_TASKS,
  );
  const [pickedModels, setPickedModels] = useState<ReadonlySet<string> | null>(
    null,
  );
  const [started, setStarted] = useState<string[]>([]);
  const [cancelled, setCancelled] = useState<ReadonlySet<string>>(new Set());
  const [storage, setStorage] = useState<ModelStorage | null>(null);
  const savedOptInRef = useRef(initialAnalyticsOptIn);
  const trackedViewRef = useRef(false);
  const trackedCompletionRef = useRef(false);
  const headingRef = useRef<HTMLHeadingElement>(null);
  const firstStepRef = useRef(true);

  const trackViewed = useCallback(() => {
    if (!trackedViewRef.current) {
      trackedViewRef.current = true;
      void trackOnboardingViewed();
    }
  }, []);

  // With usage data already on (setup run again), the view counts now; on a
  // first run it counts once the person turns it on.
  useEffect(() => {
    if (initialAnalyticsOptIn) {
      trackViewed();
    }
  }, [initialAnalyticsOptIn, trackViewed]);

  useEffect(() => {
    if (firstStepRef.current) {
      firstStepRef.current = false;
      return;
    }
    headingRef.current?.focus();
  }, [step]);

  const choosing = step === "setup" || step === "choose";
  useEffect(() => {
    if (!choosing) {
      return;
    }
    let active = true;
    api
      .getModelStorage()
      .then((next) => {
        if (active) setStorage(next);
      })
      .catch(() => {
        // The disk line stays hidden.
      });
    return () => {
      active = false;
    };
  }, [choosing]);

  const modelsByVariant = useMemo(
    () => new Map(models.map((model) => [model.variant, model])),
    [models],
  );
  const taskRows = useMemo(() => buildTaskRows(models), [models]);
  const checkedTasks = useMemo(
    () =>
      new Set(
        taskRows
          .filter((row) => !row.tooLarge && !uncheckedTasks.has(row.id))
          .map((row) => row.id),
      ),
    [taskRows, uncheckedTasks],
  );
  const taskModels = useMemo(
    () =>
      uniqueModels(
        taskRows.filter((row) => checkedTasks.has(row.id)).map((row) => row.models),
      ),
    [checkedTasks, taskRows],
  );
  const taskSizes = useMemo(
    () => taskSizeLabels(taskRows, checkedTasks),
    [checkedTasks, taskRows],
  );
  const taskPlan = useMemo(() => downloadPlan(taskModels), [taskModels]);

  const recommended = useMemo(
    () =>
      new Set(
        taskRows
          .filter((row) => !DEFAULT_UNCHECKED_TASKS.has(row.id))
          .flatMap((row) => row.models.map((model) => model.variant)),
      ),
    [taskRows],
  );
  const modelGroups = useMemo(
    () => groupModels(models, recommended),
    [models, recommended],
  );
  const chosenModels = useMemo(
    () =>
      [...(pickedModels ?? [])]
        .map((variant) => modelsByVariant.get(variant))
        .filter((model): model is ModelInfo => Boolean(model)),
    [modelsByVariant, pickedModels],
  );
  const choosePlan = useMemo(() => downloadPlan(chosenModels), [chosenModels]);

  const catalogUnavailable = models.length === 0;
  const engineDown = catalogUnavailable && !loading && Boolean(catalogError);
  const health = healthState.status === "ready" ? healthState.health : null;
  const hardware = hardwareLabel(health);

  const trackCompleted = useCallback(
    (mode: "quick" | "custom" | "skip") => {
      if (!trackedCompletionRef.current) {
        trackedCompletionRef.current = true;
        void trackOnboardingCompleted(mode);
      }
    },
    [],
  );

  const saveAnalytics = useCallback(async () => {
    if (analyticsOptIn === savedOptInRef.current) {
      return;
    }
    try {
      await api.updateAnalyticsPreference({ opt_in: analyticsOptIn });
      savedOptInRef.current = analyticsOptIn;
      setAnalyticsEnabled(analyticsOptIn);
      if (analyticsOptIn) {
        trackViewed();
      }
      void trackAnalyticsConsentChanged(
        analyticsOptIn ? "opted_in" : "opted_out",
        "onboarding",
      );
    } catch (error) {
      console.error("Failed to save the usage data choice:", error);
      notify({
        title: "Couldn't save your usage data choice",
        description: "You can change it any time in Settings.",
        tone: "warning",
      });
    }
  }, [analyticsOptIn, notify, trackViewed]);

  const getStarted = async () => {
    setBusy("start");
    await saveAnalytics();
    setBusy(null);
    setStep("setup");
  };

  const skip = async () => {
    setBusy("skip");
    if (step === "welcome") {
      await saveAnalytics();
    }
    onComplete();
    trackCompleted("skip");
    onExit();
  };

  const start = (mode: "quick" | "custom", chosen: ModelInfo[]) => {
    const plan = downloadPlan(chosen);
    onComplete();
    trackCompleted(mode);
    pageChoices(plan.fetch).forEach(([route, variant]) =>
      selectModel(variant, route),
    );
    plan.fetch.forEach((model) => void downloadModel(model.variant));
    setStarted(uniqueModels([chosen]).map((model) => model.variant));
    setCancelled(new Set());
    setStep(chosen.every(isOnDisk) ? "done" : "progress");
  };

  const openChooser = () => {
    setPickedModels(
      (current) => current ?? new Set(taskModels.map((model) => model.variant)),
    );
    setStep("choose");
  };

  const toggleTask = (id: TaskId) =>
    setUncheckedTasks((current) => {
      const next = new Set(current);
      if (!next.delete(id)) next.add(id);
      return next;
    });

  const toggleModel = (variant: string) =>
    setPickedModels((current) => {
      const next = new Set(current);
      if (!next.delete(variant)) next.add(variant);
      return next;
    });

  const startedModels = useMemo(
    () =>
      started
        .map((variant) => modelsByVariant.get(variant))
        .filter((model): model is ModelInfo => Boolean(model)),
    [modelsByVariant, started],
  );
  const downloading = startedModels.filter(
    (model) => model.status === "downloading",
  );

  useEffect(() => {
    if (
      step === "progress" &&
      startedModels.length > 0 &&
      startedModels.every(isOnDisk)
    ) {
      setStep("done");
    }
  }, [startedModels, step]);

  const cancelDownloads = () => {
    setCancelled(new Set(downloading.map((model) => model.variant)));
    downloading.forEach((model) => void cancelModelDownload(model.variant));
  };

  const retryDownload = (variant: string) => {
    setCancelled((current) => {
      const next = new Set(current);
      next.delete(variant);
      return next;
    });
    void downloadModel(variant);
  };

  let stage: ReactNode;
  if (engineDown && step !== "welcome") {
    stage = (
      <EngineOffline
        detail={catalogError}
        onRetry={() => refreshModels()}
      />
    );
  } else if (step === "welcome") {
    stage = (
      <OnboardingStage>
        <img src={APP_ICON_URL} alt="" className="h-12 w-12 rounded-[12px]" />
        <StageHeading
          headingRef={headingRef}
          title="Welcome to Izwi"
          description={`Talk, transcribe, read aloud and chat on this ${device}. What you say and type stays on it.`}
        />
        <div className="grid grid-cols-[minmax(0,1fr)_auto] items-center gap-4 border-y border-[var(--border-muted)] py-3.5">
          <div className="flex min-w-0 flex-col gap-0.5">
            <label htmlFor="first-run-usage-data" className="text-sm font-medium">
              Share anonymous usage data
            </label>
            <span
              id="first-run-usage-data-detail"
              className="text-[13px] text-[var(--text-muted)]"
            >
              Which features and models you use. Never your audio, text or files.
            </span>
          </div>
          <Switch
            id="first-run-usage-data"
            checked={analyticsOptIn}
            onCheckedChange={setAnalyticsOptIn}
            aria-describedby="first-run-usage-data-detail"
          />
        </div>
        <PrimaryButton
          autoFocus
          busy={busy === "start"}
          onClick={() => void getStarted()}
        >
          Get started
        </PrimaryButton>
      </OnboardingStage>
    );
  } else if (step === "setup") {
    const disk = diskLine(taskPlan, storage);
    stage = (
      <OnboardingStage width="wide">
        <StageHeading
          headingRef={headingRef}
          before={<BackButton onClick={() => setStep("welcome")} />}
          title={`Set up Izwi for this ${device}`}
          description="Pick what to use first. You can add more any time in Models."
          after={
            hardware ? (
              <p className="flex items-center gap-1.5 text-[12.5px] text-[var(--text-subtle)]">
                <Cpu aria-hidden="true" className="h-3.5 w-3.5" />
                {hardware}
              </p>
            ) : null
          }
        />
        {catalogUnavailable ? (
          <LoadingModels />
        ) : (
          <ul aria-label="Tasks" className="border-t border-[var(--border-muted)]">
            {taskRows.map((row) => (
              <ChoiceRow
                key={row.id}
                title={row.title}
                detail={
                  (row.tooLarge ? neededMemoryLabel(row.tooLarge) : null) ??
                  taskDetail(row)
                }
                trailing={row.tooLarge ? "" : sizeText(taskSizes.get(row.id))}
                checked={checkedTasks.has(row.id)}
                disabled={Boolean(row.tooLarge)}
                dimmed={Boolean(row.tooLarge)}
                onToggle={() => toggleTask(row.id)}
              />
            ))}
          </ul>
        )}
        <div className="flex flex-col gap-3.5">
          <PrimaryButton
            disabled={checkedTasks.size === 0}
            onClick={() => start("quick", taskModels)}
          >
            {taskPlan.fetch.length > 0
              ? `Download and set up · ${formatModelBytes(taskPlan.bytes)}`
              : "Set up"}
          </PrimaryButton>
          <p className="text-[13px] text-[var(--text-muted)]">
            {disk ? (
              <span className={cn(disk.short && "text-[var(--danger-text)]")}>
                {disk.text}{" "}
              </span>
            ) : null}
            <button
              type="button"
              className={linkButtonClass}
              disabled={catalogUnavailable}
              onClick={openChooser}
            >
              Choose models myself
            </button>
          </p>
        </div>
      </OnboardingStage>
    );
  } else if (step === "choose") {
    const disk = diskLine(choosePlan, storage);
    const fetchCount = choosePlan.fetch.length;
    stage = (
      <div className="mx-auto flex min-h-0 w-full max-w-[520px] flex-1 flex-col gap-5 pb-8 pt-4">
        <StageHeading
          headingRef={headingRef}
          before={
            <BackButton onClick={() => setStep("setup")}>
              Recommended setup
            </BackButton>
          }
          title="Choose models"
          description={`Models that need more memory than this ${device} has can't be picked.`}
        />
        {catalogUnavailable ? (
          <LoadingModels />
        ) : (
          <div className="min-h-0 overflow-y-auto pr-2">
            {modelGroups.map((group, index) => (
              <section key={group.id} aria-labelledby={`first-run-group-${group.id}`}>
                <h2
                  id={`first-run-group-${group.id}`}
                  className={cn(
                    "pb-1.5 text-[13px] font-medium text-[var(--text-muted)]",
                    index > 0 && "pt-6",
                  )}
                >
                  {group.title}
                </h2>
                <ul className="border-t border-[var(--border-muted)]">
                  {group.models.map((model) => {
                    const onDisk = isOnDisk(model);
                    const settled = onDisk || model.status === "downloading";
                    const tooLarge = !settled && !fitsDevice(model);
                    return (
                      <ChoiceRow
                        key={model.variant}
                        title={modelName(model)}
                        detail={
                          (tooLarge ? neededMemoryLabel(model) : null) ??
                          modelDescriptor(model)
                        }
                        trailing={
                          onDisk
                            ? "Downloaded"
                            : settled
                              ? "Downloading"
                              : formatModelBytes(downloadBytes(model))
                        }
                        checked={settled || (pickedModels?.has(model.variant) ?? false)}
                        disabled={settled || tooLarge}
                        dimmed={tooLarge}
                        onToggle={() => toggleModel(model.variant)}
                      />
                    );
                  })}
                </ul>
              </section>
            ))}
          </div>
        )}
        <div className="flex shrink-0 flex-col gap-3 border-t border-[var(--border-muted)] pt-4">
          <PrimaryButton
            disabled={chosenModels.length === 0}
            onClick={() => start("custom", chosenModels)}
          >
            {fetchCount > 0
              ? `Download ${fetchCount} model${fetchCount === 1 ? "" : "s"} · ${formatModelBytes(choosePlan.bytes)}`
              : "Set up"}
          </PrimaryButton>
          {disk ? (
            <p
              className={cn(
                "text-center text-[13px]",
                disk.short ? "text-[var(--danger-text)]" : "text-[var(--text-muted)]",
              )}
            >
              {disk.text}
            </p>
          ) : null}
        </div>
      </div>
    );
  } else if (step === "progress") {
    stage = (
      <OnboardingStage>
        <StageHeading
          headingRef={headingRef}
          title="Setting up Izwi"
          description="You can start using Izwi now. Downloads keep going in the background."
        />
        <ul aria-label="Downloads" className="border-t border-[var(--border-muted)]">
          {startedModels.map((model) => {
            const status = progressStatus(
              model,
              downloadPercent(model, downloadProgress),
              cancelled.has(model.variant),
            );
            const name = modelName(model);
            return (
              <SetupRow
                key={model.variant}
                title={modelRole(model)}
                model={name}
                status={status.text}
                tone={status.tone}
                percent={status.percent}
                action={
                  status.failed ? (
                    <button
                      type="button"
                      className={cn(linkButtonClass, "text-[13px]")}
                      aria-label={`Retry ${name}`}
                      onClick={() => retryDownload(model.variant)}
                    >
                      Retry
                    </button>
                  ) : null
                }
                detail={
                  model.status === "error" ? model.error_message : null
                }
              />
            );
          })}
        </ul>
        <div className="flex flex-col gap-3.5">
          <PrimaryButton onClick={() => onExit()}>Start using Izwi</PrimaryButton>
          <p className="text-[13px] text-[var(--text-muted)]">
            Progress also shows in Models.
            {downloading.length > 0 ? (
              <>
                {" "}
                <button
                  type="button"
                  className={linkButtonClass}
                  onClick={cancelDownloads}
                >
                  Cancel downloads
                </button>
              </>
            ) : null}
          </p>
        </div>
      </OnboardingStage>
    );
  } else {
    const destinations = readyTasks(taskRows, models).filter(
      (row, index, rows) =>
        rows.findIndex((other) => other.path === row.path) === index,
    );
    stage = (
      <OnboardingStage>
        <StageHeading
          headingRef={headingRef}
          before={
            <span className="mb-2 flex h-9 w-9 items-center justify-center rounded-full bg-[var(--status-positive-bg)] text-[var(--status-positive-text)]">
              <Check aria-hidden="true" className="h-[18px] w-[18px]" strokeWidth={2.5} />
            </span>
          }
          title="You're all set"
          description={
            destinations.length > 0
              ? `Everything below is ready on this ${device}.`
              : undefined
          }
        />
        {destinations.length > 0 ? (
          <nav aria-label="Start with" className="border-t border-[var(--border-muted)]">
            {destinations.map((row) => {
              const Icon = NAV_GROUPS.flatMap((group) => group.items).find(
                (item) => item.path === row.path,
              )?.icon;
              return (
                <Link
                  key={row.id}
                  to={row.path}
                  onClick={(event) => {
                    event.preventDefault();
                    onExit(row.path);
                  }}
                  className="grid grid-cols-[auto_minmax(0,1fr)_auto] items-center gap-3 border-b border-[var(--border-muted)] py-3.5 text-[var(--text-primary)] transition-colors hover:bg-[var(--sidebar-link-hover-bg)] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[var(--border-strong)]"
                >
                  {Icon ? (
                    <Icon aria-hidden="true" className="h-4 w-4 text-[var(--text-muted)]" />
                  ) : (
                    <span />
                  )}
                  <span className="text-sm font-medium">{row.doneTitle}</span>
                  <ChevronRight
                    aria-hidden="true"
                    className="h-4 w-4 text-[var(--text-subtle)]"
                  />
                </Link>
              );
            })}
          </nav>
        ) : null}
        <PrimaryButton onClick={() => onExit()}>Open Izwi</PrimaryButton>
      </OnboardingStage>
    );
  }

  const canSkip = step === "welcome" || step === "setup" || step === "choose";
  return (
    <OnboardingFrame
      step={STEP_BARS[step]}
      onSkip={canSkip ? () => void skip() : undefined}
      skipping={busy === "skip"}
    >
      {stage}
    </OnboardingFrame>
  );
}
