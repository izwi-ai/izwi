import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useLayoutEffect,
  useMemo,
  useRef,
  useState,
  type ReactNode,
} from "react";
import {
  api,
  type ModelInfo,
  type ModelResidencySummary,
  type RemovedModelInfo,
} from "@/api";
import {
  trackModelDownloadCompleted,
  trackModelDownloadStarted,
  trackModelLoaded,
} from "@/app/analytics/events";
import { Download } from "lucide-react";
import { useNotifications } from "@/app/providers/NotificationProvider";
import { getModelDisplayName } from "@/features/models/catalog/modelMetadata";
import { normalizeStoredModelId } from "@/features/models/catalog/routeModelCatalog";
import type { ModelDownloadProgressMap } from "@/features/models/downloadProgress";
import { VIEW_CONFIGS } from "@/types";
import { isSpeechPipelineManagedModel } from "@/features/speech-text/modelFilters";

/** Pages that keep their own model choice. */
export type ModelRouteKey =
  | "chat"
  | "text-to-speech"
  | "studio"
  | "voices"
  | "transcription";

interface ModelCatalogContextValue {
  models: ModelInfo[];
  /** The model chosen last, on any page. */
  selectedModel: string | null;
  /** A page's own choice, else the model chosen last. */
  selectedModelFor: (route: ModelRouteKey) => string | null;
  loading: boolean;
  error: string | null;
  catalogError: string | null;
  downloadProgress: ModelDownloadProgressMap;
  readyModelsCount: number;
  residencySummary: ModelResidencySummary | null;
  removedModels: RemovedModelInfo[];
  selectModel: (variant: string | null, route?: ModelRouteKey) => void;
  reportError: (message: string) => void;
  clearError: () => void;
  refreshModels: () => Promise<boolean>;
  downloadModel: (variant: string) => Promise<void>;
  cancelModelDownload: (variant: string) => Promise<void>;
  /** Loads a model; with `route`, it becomes that page's choice. */
  loadModel: (variant: string, options?: { route?: ModelRouteKey }) => Promise<void>;
  unloadModel: (variant: string) => Promise<void>;
  /** "Keep loaded" on or off for a loaded model. */
  setModelPinned: (variant: string, pinned: boolean) => Promise<void>;
  deleteModel: (variant: string) => Promise<void>;
}

const ModelCatalogContext = createContext<ModelCatalogContextValue | null>(null);

/** Several downloads at once show as one notification. */
const DOWNLOADS_GROUP = {
  key: "model-downloads",
  title: (count: number) => `Downloading ${count} models`,
};

const downloadNotificationId = (variant: string) => `download:${variant}`;

function formatDownloadedBytes(downloaded: number, total: number): string | undefined {
  if (!(total > 0)) {
    return undefined;
  }
  const gigabytes = (bytes: number) => (bytes / 1e9).toFixed(1);
  return total >= 1e9
    ? `${gigabytes(downloaded)} of ${gigabytes(total)} GB`
    : `${Math.round(downloaded / 1e6)} of ${Math.round(total / 1e6)} MB`;
}

/** The model list also refreshes on this interval and on focus, so state
 * changed elsewhere (another window, the CLI, an idle unload) shows up. */
const BACKGROUND_REFRESH_INTERVAL_MS = 15_000;

/**
 * Download progress arrives as a stream of SSE messages, and every state
 * update re-renders the shell and the active page. Updates are batched to at
 * most one flush per interval; terminal statuses flush immediately.
 */
const PROGRESS_FLUSH_INTERVAL_MS = 250;

function modelActionError(err: unknown, fallback: string): string {
  return err instanceof Error && err.message.trim() ? err.message : fallback;
}

const USER_SELECTED_MODEL_STORAGE_KEY = "izwi.modelCatalog.userSelectedModel";

function readPersistedUserSelectedModel(): string | null {
  try {
    const stored = window.localStorage.getItem(USER_SELECTED_MODEL_STORAGE_KEY);
    // A selection saved before its variant was removed reads back as the
    // supported counterpart.
    return stored === null ? null : normalizeStoredModelId(stored);
  } catch {
    return null;
  }
}

function persistUserSelectedModel(variant: string | null): void {
  try {
    if (variant === null) {
      window.localStorage.removeItem(USER_SELECTED_MODEL_STORAGE_KEY);
    } else {
      window.localStorage.setItem(USER_SELECTED_MODEL_STORAGE_KEY, variant);
    }
  } catch {
    // Persistence is best-effort; the in-memory selection still works.
  }
}

const ROUTE_SELECTIONS_STORAGE_KEY = "izwi.modelCatalog.routeSelections";

type RouteSelections = Partial<Record<ModelRouteKey, string>>;

function readPersistedRouteSelections(): RouteSelections {
  try {
    const stored = window.localStorage.getItem(ROUTE_SELECTIONS_STORAGE_KEY);
    const parsed: unknown = stored ? JSON.parse(stored) : null;
    if (!parsed || typeof parsed !== "object") {
      return {};
    }
    return Object.fromEntries(
      Object.entries(parsed as Record<string, unknown>)
        .filter((entry): entry is [string, string] => typeof entry[1] === "string")
        .map(([route, variant]) => [route, normalizeStoredModelId(variant)]),
    );
  } catch {
    return {};
  }
}

function persistRouteSelections(selections: RouteSelections): void {
  try {
    window.localStorage.setItem(
      ROUTE_SELECTIONS_STORAGE_KEY,
      JSON.stringify(selections),
    );
  } catch {
    // Persistence is best-effort.
  }
}

interface ModelCatalogProviderProps {
  children: ReactNode;
}

export function ModelCatalogProvider({
  children,
}: ModelCatalogProviderProps) {
  const { notify, dismiss } = useNotifications();
  // Last whole percent shown per download: progress arrives many times a
  // second, the notification changes once per percent.
  const notifiedPercentRef = useRef<Record<string, number>>({});
  // Retry from a failed download's notification; set once downloadModel
  // exists.
  const retryDownloadRef = useRef<(variant: string) => void>(() => {});
  // The progress stream's watchdog and reconnect timers open it again; set
  // once connectDownloadStream exists, which can't name itself.
  const reconnectDownloadStreamRef = useRef<(variant: string) => void>(() => {});
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [residencySummary, setResidencySummary] =
    useState<ModelResidencySummary | null>(null);
  const [removedModels, setRemovedModels] = useState<RemovedModelInfo[]>([]);
  const [selectedModel, setSelectedModelState] = useState<string | null>(() =>
    readPersistedUserSelectedModel(),
  );
  // Tracks whether the current selection was made by the user (explicit
  // select or load) versus auto-picked as a fallback. Only user choices are
  // persisted, and only user choices survive a vanished catalog variant.
  const userSelectedModelRef = useRef(selectedModel !== null);
  const [routeSelections, setRouteSelections] = useState<RouteSelections>(() =>
    readPersistedRouteSelections(),
  );
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [catalogError, setCatalogError] = useState<string | null>(null);
  const [downloadProgress, setDownloadProgress] = useState<ModelDownloadProgressMap>(
    {},
  );

  const pollingRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const activeDownloadsRef = useRef<Set<string>>(new Set());
  const activeModelLoadsRef = useRef<Set<string>>(new Set());
  const cancelledModelLoadsRef = useRef<Set<string>>(new Set());
  const loadModelAbortControllersRef = useRef<Map<string, AbortController>>(
    new Map(),
  );
  const eventSourcesRef = useRef<Record<string, EventSource>>({});
  const reconnectTimersRef = useRef<Record<string, ReturnType<typeof setTimeout>>>(
    {},
  );
  const streamWatchdogTimersRef = useRef<
    Record<string, ReturnType<typeof setInterval>>
  >({});
  const lastProgressAtRef = useRef<Record<string, number>>({});
  const suppressReconnectRef = useRef<Set<string>>(new Set());
  const lastDownloadTerminalStateRef = useRef<Record<string, string>>({});
  // Streams that reported progress in this session. A stream's first
  // message can be a stale "cancelled" snapshot of a download that ended
  // before the app opened; that is not news.
  const streamedProgressRef = useRef<Set<string>>(new Set());

  // Notifications name models by display name. Read the catalog through a
  // ref so the label lookup (and the stream callbacks that use it) stay
  // stable across catalog refreshes.
  const modelsRef = useRef(models);
  useLayoutEffect(() => {
    modelsRef.current = models;
  });
  const getModelLabel = useCallback(
    (variant: string) =>
      getModelDisplayName(
        variant,
        modelsRef.current.find((model) => model.variant === variant)
          ?.display_name,
      ),
    [],
  );

  const adoptUserSelectedModel = useCallback((variant: string | null) => {
    userSelectedModelRef.current = variant !== null;
    persistUserSelectedModel(variant);
    setSelectedModelState(variant);
  }, []);

  const adoptRouteSelection = useCallback(
    (route: ModelRouteKey, variant: string | null) => {
      setRouteSelections((current) => {
        const next = { ...current };
        if (variant === null) {
          delete next[route];
        } else {
          next[route] = variant;
        }
        persistRouteSelections(next);
        return next;
      });
    },
    [],
  );

  const selectModel = useCallback(
    (variant: string | null, route?: ModelRouteKey) => {
      adoptUserSelectedModel(variant);
      if (route) {
        adoptRouteSelection(route, variant);
      }
    },
    [adoptRouteSelection, adoptUserSelectedModel],
  );

  const selectedModelFor = useCallback(
    (route: ModelRouteKey) => routeSelections[route] ?? selectedModel,
    [routeSelections, selectedModel],
  );

  const reportError = useCallback((message: string) => {
    setError(message);
    notify({
      title: "Action failed",
      description: message,
      tone: "danger",
    });
  }, [notify]);

  const clearError = useCallback(() => {
    setError(null);
  }, []);

  const refreshModels = useCallback(async (): Promise<boolean> => {
    try {
      const response = await api.listModels();
      const mergedModels = response.models
        .map((model) =>
          activeModelLoadsRef.current.has(model.variant)
            ? { ...model, status: "loading" as const }
            : model,
        );

      const downloadingVariants = new Set(
        mergedModels
          .filter((model) => model.status === "downloading")
          .map((model) => model.variant),
      );
      suppressReconnectRef.current.forEach((variant) => {
        if (!downloadingVariants.has(variant)) {
          suppressReconnectRef.current.delete(variant);
        }
      });

      setModels(mergedModels);
      setResidencySummary(response.residency ?? null);
      setRemovedModels(response.removed_models ?? []);
      setCatalogError(null);
      setSelectedModelState((current) => {
        if (
          current &&
          mergedModels.some((model) => model.variant === current)
        ) {
          return current;
        }
        if (current && userSelectedModelRef.current) {
          // The user explicitly picked this variant. If it disappeared from
          // the catalog (removed upstream), keep surfacing the stale
          // selection instead of silently swapping to whatever is ready.
          return current;
        }

        const readyModel = mergedModels.find((model) => model.status === "ready");
        return readyModel?.variant ?? null;
      });
      return true;
    } catch (err) {
      console.error("Failed to load models:", err);
      setCatalogError(
        modelActionError(
          err,
          "Izwi could not reach the local model service. Please try again.",
        ),
      );
      return false;
    }
  }, []);

  const pendingProgressRef = useRef<ModelDownloadProgressMap>({});
  const progressFlushTimerRef = useRef<ReturnType<typeof setTimeout> | null>(
    null,
  );

  const flushDownloadProgress = useCallback(() => {
    if (progressFlushTimerRef.current) {
      clearTimeout(progressFlushTimerRef.current);
      progressFlushTimerRef.current = null;
    }
    const pending = pendingProgressRef.current;
    if (Object.keys(pending).length === 0) {
      return;
    }
    pendingProgressRef.current = {};
    setDownloadProgress((prev) => ({ ...prev, ...pending }));
  }, []);

  const queueDownloadProgress = useCallback(
    (
      variant: string,
      entry: ModelDownloadProgressMap[string],
      flushNow: boolean,
    ) => {
      pendingProgressRef.current = {
        ...pendingProgressRef.current,
        [variant]: entry,
      };
      if (flushNow) {
        flushDownloadProgress();
      } else if (!progressFlushTimerRef.current) {
        progressFlushTimerRef.current = setTimeout(
          flushDownloadProgress,
          PROGRESS_FLUSH_INTERVAL_MS,
        );
      }
    },
    [flushDownloadProgress],
  );

  const clearDownloadProgress = useCallback((variant: string) => {
    const { [variant]: _pending, ...remaining } = pendingProgressRef.current;
    pendingProgressRef.current = remaining;
    setDownloadProgress((prev) => {
      const { [variant]: _removed, ...rest } = prev;
      return rest;
    });
  }, []);

  const clearReconnectTimer = useCallback((variant: string) => {
    const timer = reconnectTimersRef.current[variant];
    if (timer) {
      clearTimeout(timer);
      delete reconnectTimersRef.current[variant];
    }
  }, []);

  const closeDownloadStream = useCallback(
    (variant: string) => {
      clearReconnectTimer(variant);

      const watchdogTimer = streamWatchdogTimersRef.current[variant];
      if (watchdogTimer) {
        clearInterval(watchdogTimer);
        delete streamWatchdogTimersRef.current[variant];
      }
      delete lastProgressAtRef.current[variant];

      const eventSource = eventSourcesRef.current[variant];
      if (eventSource) {
        eventSource.close();
        delete eventSourcesRef.current[variant];
      }
    },
    [clearReconnectTimer],
  );

  const connectDownloadStream = useCallback(
    (variant: string) => {
      clearReconnectTimer(variant);
      closeDownloadStream(variant);
      suppressReconnectRef.current.delete(variant);
      activeDownloadsRef.current.add(variant);

      const eventSource = new EventSource(
        `${api.baseUrl}/admin/models/${variant}/download/progress`,
      );
      eventSourcesRef.current[variant] = eventSource;
      lastProgressAtRef.current[variant] = Date.now();

      const existingWatchdog = streamWatchdogTimersRef.current[variant];
      if (existingWatchdog) {
        clearInterval(existingWatchdog);
      }
      streamWatchdogTimersRef.current[variant] = setInterval(async () => {
        if (suppressReconnectRef.current.has(variant)) {
          return;
        }

        const lastProgressAt = lastProgressAtRef.current[variant] ?? 0;
        if (Date.now() - lastProgressAt < 8000) {
          return;
        }

        closeDownloadStream(variant);
        try {
          const model = await api.getModelInfo(variant);
          if (model.status === "downloading") {
            reconnectDownloadStreamRef.current(variant);
          }
        } catch (watchdogErr) {
          console.error(
            `Stream watchdog check failed for ${variant}:`,
            watchdogErr,
          );
        }
      }, 4000);

      eventSource.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          lastProgressAtRef.current[variant] = Date.now();
          const isTerminal =
            data.status === "completed" ||
            data.status === "error" ||
            data.status === "cancelled";
          const sawProgress = streamedProgressRef.current.has(variant);
          if (!isTerminal) {
            streamedProgressRef.current.add(variant);
          }
          queueDownloadProgress(
            variant,
            {
              percent: data.percent,
              currentFile: data.current_file,
              status: data.status,
              downloadedBytes: data.downloaded_bytes,
              totalBytes: data.total_bytes,
            },
            isTerminal,
          );
          const percent = Math.floor(Number(data.percent) || 0);
          if (!isTerminal && notifiedPercentRef.current[variant] !== percent) {
            notifiedPercentRef.current[variant] = percent;
            notify({
              id: downloadNotificationId(variant),
              title: `Downloading ${getModelLabel(variant)}`,
              icon: Download,
              progress: percent,
              detail: formatDownloadedBytes(
                Number(data.downloaded_bytes) || 0,
                Number(data.total_bytes) || 0,
              ),
              group: DOWNLOADS_GROUP,
            });
          }

          if (isTerminal) {
            const previousTerminalState =
              lastDownloadTerminalStateRef.current[variant];
            if (previousTerminalState !== data.status) {
              lastDownloadTerminalStateRef.current[variant] = data.status;
              const id = downloadNotificationId(variant);
              if (data.status === "completed") {
                void trackModelDownloadCompleted(variant);
                notify({
                  id,
                  title: `${getModelLabel(variant)} downloaded`,
                  description: "Ready to load.",
                  tone: "success",
                });
              } else if (data.status === "cancelled") {
                // A stale snapshot from before the app opened is not news.
                if (sawProgress) {
                  notify({
                    id,
                    title: `${getModelLabel(variant)} download cancelled`,
                    tone: "info",
                  });
                } else {
                  dismiss(id);
                }
              } else if (data.status === "error") {
                notify({
                  id,
                  title: `${getModelLabel(variant)} didn't download`,
                  description:
                    typeof data.error === "string" && data.error.trim()
                      ? data.error
                      : "Try the download again.",
                  tone: "danger",
                  action: {
                    label: "Retry",
                    onClick: () => retryDownloadRef.current(variant),
                  },
                });
              }
            }
            delete notifiedPercentRef.current[variant];
            streamedProgressRef.current.delete(variant);
            closeDownloadStream(variant);
            activeDownloadsRef.current.delete(variant);
            suppressReconnectRef.current.delete(variant);
            // Until the refresh lands the list still says "downloading",
            // which would register the download again.
            const settled = {
              completed: "downloaded",
              error: "error",
              cancelled: "not_downloaded",
            } as const;
            setModels((previous) =>
              previous.map((model) =>
                model.variant === variant
                  ? {
                      ...model,
                      status: settled[data.status as keyof typeof settled],
                      error_message:
                        data.status === "error" && typeof data.error === "string"
                          ? data.error
                          : model.error_message,
                    }
                  : model,
              ),
            );

            void refreshModels();

            setTimeout(() => {
              clearDownloadProgress(variant);
            }, 3000);
          }
        } catch (err) {
          console.error("Failed to parse progress event:", err);
        }
      };

      eventSource.onerror = (err) => {
        closeDownloadStream(variant);

        if (suppressReconnectRef.current.has(variant)) {
          return;
        }
        if (reconnectTimersRef.current[variant]) {
          return;
        }

        reconnectTimersRef.current[variant] = setTimeout(async () => {
          delete reconnectTimersRef.current[variant];

          if (suppressReconnectRef.current.has(variant)) {
            return;
          }

          try {
            const model = await api.getModelInfo(variant);
            if (model.status === "downloading") {
              console.warn(
                `Download progress stream disconnected for ${variant}; reconnecting`,
                err,
              );
              reconnectDownloadStreamRef.current(variant);
              return;
            }
          } catch (reconnectErr) {
            console.error(
              `Reconnect check failed for ${variant}:`,
              reconnectErr,
            );
          }

          activeDownloadsRef.current.delete(variant);
          clearDownloadProgress(variant);
          await refreshModels();
        }, 1500);
      };
    },
    [
      clearDownloadProgress,
      clearReconnectTimer,
      closeDownloadStream,
      dismiss,
      getModelLabel,
      notify,
      queueDownloadProgress,
      refreshModels,
    ],
  );

  useEffect(() => {
    reconnectDownloadStreamRef.current = connectDownloadStream;
  }, [connectDownloadStream]);

  useEffect(() => {
    // No once-guard here: StrictMode's dev mount→cleanup→mount must be free to
    // start a second init run, because the first run exits at its next failure
    // check once cleaned up and can neither retry nor clear the spinner. The
    // supersession rules below are what keep that hand-off single-owner; in
    // production the effect runs once and nothing else changes.
    //
    // The first catalog load can race the local server's cold start (a slow
    // /admin/models call, a poison-monitor respawn). Retry with bounded
    // backoff so the spinner ends with either models or a clear error instead
    // of hanging forever on a single stalled request. A SUCCESSFUL fetch
    // always clears the spinner (the models are already in state, so that is
    // simply the truth); only failure retries respect supersession, because
    // StrictMode's mount→cleanup→mount can abandon the first run mid-flight
    // and a superseded failure must not spin forever.
    const MAX_ATTEMPTS = 3;
    const RETRY_DELAYS_MS = [750, 1_500];
    let active = true;
    let retryTimer: ReturnType<typeof setTimeout> | null = null;

    const init = async () => {
      setLoading(true);
      for (let attempt = 0; attempt < MAX_ATTEMPTS; attempt++) {
        const ok = await refreshModels();
        if (ok) {
          setLoading(false);
          return;
        }
        if (!active) {
          return;
        }
        const delay = RETRY_DELAYS_MS[Math.min(attempt, RETRY_DELAYS_MS.length - 1)];
        await new Promise((resolve) => {
          retryTimer = setTimeout(resolve, delay);
        });
      }
      if (active) {
        setLoading(false);
      }
    };

    void init();

    return () => {
      active = false;
      if (retryTimer !== null) {
        clearTimeout(retryTimer);
      }
    };
  }, [refreshModels]);

  useEffect(() => {
    const hasActiveOperations = models.some(
      (model) =>
        model.status === "downloading" || model.status === "loading",
    );

    if (!hasActiveOperations) {
      if (pollingRef.current) {
        clearInterval(pollingRef.current);
        pollingRef.current = null;
      }
      return;
    }

    if (!pollingRef.current) {
      pollingRef.current = setInterval(() => {
        void refreshModels();
      }, 3000);
    }

    return () => {
      if (pollingRef.current) {
        clearInterval(pollingRef.current);
        pollingRef.current = null;
      }
    };
  }, [models, refreshModels]);

  useEffect(() => {
    const downloading = new Set(
      models
        .filter((model) => model.status === "downloading")
        .map((model) => model.variant),
    );

    downloading.forEach((variant) => {
      if (suppressReconnectRef.current.has(variant)) {
        return;
      }
      activeDownloadsRef.current.add(variant);
      if (
        !eventSourcesRef.current[variant] &&
        !reconnectTimersRef.current[variant]
      ) {
        connectDownloadStream(variant);
      }
    });

    Object.keys(eventSourcesRef.current).forEach((variant) => {
      const streamStatus = downloadProgress[variant]?.status;
      const shouldKeepStream =
        downloading.has(variant) ||
        activeDownloadsRef.current.has(variant) ||
        streamStatus === "downloading";

      if (!shouldKeepStream) {
        closeDownloadStream(variant);
        activeDownloadsRef.current.delete(variant);
      }
    });
  }, [models, downloadProgress, connectDownloadStream, closeDownloadStream]);

  useEffect(() => {
    return () => {
      Object.values(eventSourcesRef.current).forEach((source) => source.close());
      eventSourcesRef.current = {};
      Object.values(reconnectTimersRef.current).forEach((timer) =>
        clearTimeout(timer),
      );
      reconnectTimersRef.current = {};
      Object.values(streamWatchdogTimersRef.current).forEach((timer) =>
        clearInterval(timer),
      );
      streamWatchdogTimersRef.current = {};
      lastProgressAtRef.current = {};
      if (progressFlushTimerRef.current) {
        clearTimeout(progressFlushTimerRef.current);
        progressFlushTimerRef.current = null;
      }
    };
  }, []);

  const downloadModel = useCallback(
    async (variant: string) => {
      try {
        if (activeDownloadsRef.current.has(variant)) {
          return;
        }
        suppressReconnectRef.current.delete(variant);
        clearReconnectTimer(variant);
        activeDownloadsRef.current.add(variant);
        delete lastDownloadTerminalStateRef.current[variant];

        setModels((prev) =>
          prev.map((model) =>
            model.variant === variant
              ? { ...model, status: "downloading" as const }
              : model,
          ),
        );

        const response = await api.downloadModel(variant);
        notify({
          id: downloadNotificationId(variant),
          title: `Downloading ${getModelLabel(variant)}`,
          icon: Download,
          progress: 0,
          group: DOWNLOADS_GROUP,
        });
        void trackModelDownloadStarted(variant);

        if (
          response.status === "started" ||
          response.status === "downloading"
        ) {
          await refreshModels();
        } else {
          activeDownloadsRef.current.delete(variant);
          await refreshModels();
        }
      } catch (err) {
        console.error("Download failed:", err);
        activeDownloadsRef.current.delete(variant);
        setError(
          err instanceof Error
            ? err.message
            : "Failed to download model. Please try again.",
        );
        notify({
          id: downloadNotificationId(variant),
          title: `${getModelLabel(variant)} didn't download`,
          description:
            err instanceof Error ? err.message : "Try the download again.",
          tone: "danger",
          action: {
            label: "Retry",
            onClick: () => retryDownloadRef.current(variant),
          },
        });

        closeDownloadStream(variant);

        await refreshModels();
      }
    },
    [
      clearReconnectTimer,
      closeDownloadStream,
      getModelLabel,
      notify,
      refreshModels,
    ],
  );

  useEffect(() => {
    retryDownloadRef.current = (variant) => void downloadModel(variant);
  }, [downloadModel]);

  const cancelModelDownload = useCallback(
    async (variant: string) => {
      try {
        suppressReconnectRef.current.add(variant);
        closeDownloadStream(variant);

        activeDownloadsRef.current.delete(variant);

        await api.cancelDownload(variant);
        lastDownloadTerminalStateRef.current[variant] = "cancelled";
        dismiss(downloadNotificationId(variant));
        delete notifiedPercentRef.current[variant];

        clearDownloadProgress(variant);

        setModels((prev) =>
          prev.map((model) =>
            model.variant === variant
              ? {
                  ...model,
                  status: "not_downloaded" as const,
                  download_progress: null,
                }
              : model,
          ),
        );

        await refreshModels();
      } catch (err) {
        suppressReconnectRef.current.delete(variant);
        console.error("Cancel failed:", err);
        setError(
          err instanceof Error ? err.message : "Failed to cancel download.",
        );
        notify({
          title: "Cancel failed",
          description:
            err instanceof Error
              ? err.message
              : `Izwi could not cancel ${getModelLabel(variant)}.`,
          tone: "danger",
        });
        await refreshModels();
      }
    },
    [
      clearDownloadProgress,
      closeDownloadStream,
      dismiss,
      getModelLabel,
      notify,
      refreshModels,
    ],
  );

  const loadModel = useCallback(
    async (variant: string, options?: { route?: ModelRouteKey }) => {
      if (activeModelLoadsRef.current.has(variant)) {
        return;
      }

      activeModelLoadsRef.current.add(variant);
      const abortController = new AbortController();
      loadModelAbortControllersRef.current.set(variant, abortController);

      try {
        const isChatTarget = VIEW_CONFIGS.chat.modelFilter(variant);
        // Chat stays single-active, but pinned models and speech-pipeline
        // stack members (diarization + ASR + aligner + summary LLM) are never
        // evicted by a chat switch: unloading one silently degrades the
        // pipeline. Unload them explicitly instead.
        const isEvictableChatModel = (model: ModelInfo) =>
          model.status === "ready" &&
          VIEW_CONFIGS.chat.modelFilter(model.variant) &&
          model.variant !== variant &&
          !model.pinned &&
          !isSpeechPipelineManagedModel(model);
        const loadedChatModels = isChatTarget
          ? models.filter(isEvictableChatModel)
          : [];

        for (const loadedModel of loadedChatModels) {
          await api.unloadModel(loadedModel.variant);
        }

        const demotedVariants = new Set(loadedChatModels.map((model) => model.variant));
        setModels((prev) =>
          prev.map((model) =>
            model.variant === variant
              ? { ...model, status: "loading" as const }
              : demotedVariants.has(model.variant)
                ? { ...model, status: "downloaded" as const }
                : model,
          ),
        );

        // The load POST blocks server-side until the weights are resident,
        // which for large models runs far past the default request timeout.
        // It therefore rides this controller's signal (no client timeout);
        // an explicit unload or delete aborts it.
        const response = await api.loadModel(variant, {
          signal: abortController.signal,
        });
        if (!cancelledModelLoadsRef.current.has(variant)) {
          // Loading a speech-pipeline stack member (diarization checkpoint,
          // ASR, aligner, summary LLM) is pipeline setup, not a model switch:
          // adopting it as the global selection overwrote — and since the
          // persistence change, permanently rewrote — whatever the user had
          // selected.
          const loadedModel = models.find((model) => model.variant === variant);
          if (!loadedModel || !isSpeechPipelineManagedModel(loadedModel)) {
            adoptUserSelectedModel(variant);
          }
          if (options?.route) {
            adoptRouteSelection(options.route, variant);
          }
          void trackModelLoaded(variant);
          notify(
            response?.warning
              ? {
                  title: `${getModelLabel(variant)} loaded`,
                  description: response.warning,
                  tone: "warning",
                }
              : {
                  title: "Model loaded",
                  description: `${getModelLabel(variant)} is now active.`,
                  tone: "success",
                },
          );
        }
      } catch (err) {
        if (cancelledModelLoadsRef.current.has(variant)) {
          return;
        }
        if (abortController.signal.aborted) {
          return;
        }
        console.error("Load failed:", err);
        const message = modelActionError(err, "Failed to load model. Please try again.");
        setError(message);
        notify({
          title: "Model load failed",
          description: message,
          tone: "danger",
        });
      } finally {
        loadModelAbortControllersRef.current.delete(variant);
        activeModelLoadsRef.current.delete(variant);
        await refreshModels();
      }
    },
    [
      adoptRouteSelection,
      adoptUserSelectedModel,
      getModelLabel,
      models,
      notify,
      refreshModels,
    ],
  );

  const unloadModel = useCallback(
    async (variant: string) => {
      const cancellingLoad =
        activeModelLoadsRef.current.has(variant) ||
        models.some(
          (model) => model.variant === variant && model.status === "loading",
        );
      if (cancellingLoad) {
        cancelledModelLoadsRef.current.add(variant);
        loadModelAbortControllersRef.current.get(variant)?.abort();
      }
      try {
        const response = await api.unloadModel(variant);
        await refreshModels();
        setSelectedModelState((current) =>
          current === variant ? null : current,
        );
        if (selectedModel === variant) {
          userSelectedModelRef.current = false;
          persistUserSelectedModel(null);
        }
        // The server stops jobs that were using the model, and says so.
        const stoppedJobs = /cancelling \d+/.test(response?.message ?? "");
        notify({
          title: cancellingLoad ? "Model load cancelled" : "Model unloaded",
          description: cancellingLoad
            ? `${getModelLabel(variant)} was stopped and unloaded.`
            : stoppedJobs
              ? response.message
              : `${getModelLabel(variant)} was unloaded from memory.`,
          tone: stoppedJobs ? "warning" : "info",
        });
      } catch (err) {
        console.error("Unload failed:", err);
        const message = modelActionError(
          err,
          "Failed to unload model. Please try again.",
        );
        setError(message);
        notify({
          title: "Model unload failed",
          description: message,
          tone: "danger",
        });
      } finally {
        cancelledModelLoadsRef.current.delete(variant);
      }
    },
    [getModelLabel, models, notify, refreshModels, selectedModel],
  );

  const deleteModel = useCallback(
    async (variant: string) => {
      try {
        suppressReconnectRef.current.add(variant);
        closeDownloadStream(variant);
        activeDownloadsRef.current.delete(variant);
        clearDownloadProgress(variant);
        loadModelAbortControllersRef.current.get(variant)?.abort();

        await api.deleteModel(variant);
        await refreshModels();
        setSelectedModelState((current) =>
          current === variant ? null : current,
        );
        if (selectedModel === variant) {
          userSelectedModelRef.current = false;
          persistUserSelectedModel(null);
        }
        notify({
          title: "Model deleted",
          description: `${getModelLabel(variant)} was removed from disk.`,
          tone: "info",
        });
      } catch (err) {
        suppressReconnectRef.current.delete(variant);
        console.error("Delete failed:", err);
        const message = modelActionError(
          err,
          `Izwi could not delete ${getModelLabel(variant)}.`,
        );
        setError(message);
        notify({
          title: "Delete failed",
          description: message,
          tone: "danger",
        });
        await refreshModels();
      }
    },
    [
      clearDownloadProgress,
      closeDownloadStream,
      getModelLabel,
      notify,
      refreshModels,
      selectedModel,
    ],
  );

  const setModelPinned = useCallback(
    async (variant: string, pinned: boolean) => {
      try {
        await (pinned ? api.pinModel(variant) : api.unpinModel(variant));
      } catch (err) {
        notify({
          title: pinned ? "Couldn't keep the model loaded" : "Couldn't change the model",
          description: modelActionError(err, getModelLabel(variant)),
          tone: "danger",
        });
      } finally {
        await refreshModels();
      }
    },
    [getModelLabel, notify, refreshModels],
  );

  // Background refresh: state can change without this window (an idle
  // unload, the CLI, another window). Skipped while the window is hidden.
  useEffect(() => {
    const refreshIfVisible = () => {
      if (document.visibilityState === "visible") {
        void refreshModels();
      }
    };
    const timer = window.setInterval(
      refreshIfVisible,
      BACKGROUND_REFRESH_INTERVAL_MS,
    );
    window.addEventListener("focus", refreshIfVisible);
    document.addEventListener("visibilitychange", refreshIfVisible);
    return () => {
      window.clearInterval(timer);
      window.removeEventListener("focus", refreshIfVisible);
      document.removeEventListener("visibilitychange", refreshIfVisible);
    };
  }, [refreshModels]);

  const value = useMemo<ModelCatalogContextValue>(
    () => ({
      models,
      selectedModel,
      selectedModelFor,
      loading,
      error,
      catalogError,
      downloadProgress,
      readyModelsCount: models.filter((model) => model.status === "ready").length,
      residencySummary,
      removedModels,
      selectModel,
      reportError,
      clearError,
      refreshModels,
      downloadModel,
      cancelModelDownload,
      loadModel,
      unloadModel,
      setModelPinned,
      deleteModel,
    }),
    [
      cancelModelDownload,
      clearError,
      deleteModel,
      downloadModel,
      downloadProgress,
      error,
      catalogError,
      loadModel,
      loading,
      models,
      refreshModels,
      removedModels,
      reportError,
      residencySummary,
      selectModel,
      selectedModel,
      selectedModelFor,
      setModelPinned,
      unloadModel,
    ],
  );

  return (
    <ModelCatalogContext.Provider value={value}>
      {children}
    </ModelCatalogContext.Provider>
  );
}

/** The catalog, or null outside the provider (isolated components). */
export function useOptionalModelCatalog() {
  return useContext(ModelCatalogContext);
}

export function useModelCatalog() {
  const context = useContext(ModelCatalogContext);

  if (!context) {
    throw new Error(
      "useModelCatalog must be used within ModelCatalogProvider",
    );
  }

  return context;
}
