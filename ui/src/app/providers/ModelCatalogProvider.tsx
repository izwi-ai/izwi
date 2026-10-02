import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useRef,
  useState,
  type ReactNode,
} from "react";
import { api, type ModelInfo, type ModelResidencySummary } from "@/api";
import {
  trackModelDownloadCompleted,
  trackModelDownloadStarted,
  trackModelLoaded,
} from "@/app/analytics/events";
import { useNotifications } from "@/app/providers/NotificationProvider";
import type { ModelDownloadProgressMap } from "@/features/models/downloadProgress";
import { VIEW_CONFIGS } from "@/types";
import { isSpeechPipelineManagedVariant } from "@/features/speech-text/modelFilters";

interface ModelCatalogContextValue {
  models: ModelInfo[];
  selectedModel: string | null;
  loading: boolean;
  error: string | null;
  catalogError: string | null;
  downloadProgress: ModelDownloadProgressMap;
  readyModelsCount: number;
  residencySummary: ModelResidencySummary | null;
  selectModel: (variant: string | null) => void;
  reportError: (message: string) => void;
  clearError: () => void;
  refreshModels: () => Promise<boolean>;
  downloadModel: (variant: string) => Promise<void>;
  cancelModelDownload: (variant: string) => Promise<void>;
  loadModel: (variant: string) => Promise<void>;
  unloadModel: (variant: string) => Promise<void>;
  deleteModel: (variant: string) => Promise<void>;
}

const ModelCatalogContext = createContext<ModelCatalogContextValue | null>(null);

function modelActionError(err: unknown, fallback: string): string {
  return err instanceof Error && err.message.trim() ? err.message : fallback;
}

const USER_SELECTED_MODEL_STORAGE_KEY = "izwi.modelCatalog.userSelectedModel";

function readPersistedUserSelectedModel(): string | null {
  try {
    return window.localStorage.getItem(USER_SELECTED_MODEL_STORAGE_KEY);
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

interface ModelCatalogProviderProps {
  children: ReactNode;
}

export function ModelCatalogProvider({
  children,
}: ModelCatalogProviderProps) {
  const { notify } = useNotifications();
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [residencySummary, setResidencySummary] =
    useState<ModelResidencySummary | null>(null);
  const [selectedModel, setSelectedModelState] = useState<string | null>(() =>
    readPersistedUserSelectedModel(),
  );
  // Tracks whether the current selection was made by the user (explicit
  // select or load) versus auto-picked as a fallback. Only user choices are
  // persisted, and only user choices survive a vanished catalog variant.
  const userSelectedModelRef = useRef(selectedModel !== null);
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
  const initializedRef = useRef(false);
  const lastDownloadTerminalStateRef = useRef<Record<string, string>>({});

  const getModelLabel = useCallback(
    (variant: string) =>
      models.find((model) => model.variant === variant)?.variant ?? variant,
    [models],
  );

  const adoptUserSelectedModel = useCallback((variant: string | null) => {
    userSelectedModelRef.current = variant !== null;
    persistUserSelectedModel(variant);
    setSelectedModelState(variant);
  }, []);

  const selectModel = useCallback(
    (variant: string | null) => {
      adoptUserSelectedModel(variant);
    },
    [adoptUserSelectedModel],
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

  const clearDownloadProgress = useCallback((variant: string) => {
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
            connectDownloadStream(variant);
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
          setDownloadProgress((prev) => ({
            ...prev,
            [variant]: {
              percent: data.percent,
              currentFile: data.current_file,
              status: data.status,
              downloadedBytes: data.downloaded_bytes,
              totalBytes: data.total_bytes,
            },
          }));

          if (
            data.status === "completed" ||
            data.status === "error" ||
            data.status === "cancelled"
          ) {
            const previousTerminalState =
              lastDownloadTerminalStateRef.current[variant];
            if (previousTerminalState !== data.status) {
              lastDownloadTerminalStateRef.current[variant] = data.status;
              if (data.status === "completed") {
                void trackModelDownloadCompleted(variant);
                notify({
                  title: "Model download complete",
                  description: `${getModelLabel(variant)} is ready to load.`,
                  tone: "success",
                });
              } else if (data.status === "cancelled") {
                notify({
                  title: "Model download cancelled",
                  description: `${getModelLabel(variant)} download was stopped.`,
                  tone: "info",
                });
              } else if (data.status === "error") {
                notify({
                  title: "Model download failed",
                  description: `Izwi could not finish downloading ${getModelLabel(variant)}.`,
                  tone: "danger",
                });
              }
            }
            closeDownloadStream(variant);
            activeDownloadsRef.current.delete(variant);
            suppressReconnectRef.current.delete(variant);

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
              connectDownloadStream(variant);
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
      refreshModels,
    ],
  );

  useEffect(() => {
    if (initializedRef.current) {
      return;
    }

    initializedRef.current = true;

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
          title: "Downloading model",
          description: `${getModelLabel(variant)} download started in the background.`,
          tone: "info",
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
          title: "Model download failed",
          description:
            err instanceof Error
              ? err.message
              : `Izwi could not download ${getModelLabel(variant)}.`,
          tone: "danger",
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

  const cancelModelDownload = useCallback(
    async (variant: string) => {
      try {
        suppressReconnectRef.current.add(variant);
        closeDownloadStream(variant);

        activeDownloadsRef.current.delete(variant);

        await api.cancelDownload(variant);
        lastDownloadTerminalStateRef.current[variant] = "cancelled";

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
      getModelLabel,
      notify,
      refreshModels,
    ],
  );

  const loadModel = useCallback(
    async (variant: string) => {
      if (activeModelLoadsRef.current.has(variant)) {
        return;
      }

      activeModelLoadsRef.current.add(variant);
      const abortController = new AbortController();
      loadModelAbortControllersRef.current.set(variant, abortController);

      try {
        const isChatTarget = VIEW_CONFIGS.chat.modelFilter(variant);
        // Chat stays single-active, but pinned models and speech-pipeline
        // stack members (diarization + ASR + aligner + refiner) are never
        // evicted by a chat switch: unloading one silently degrades the
        // pipeline. Unload them explicitly instead.
        const isEvictableChatModel = (model: ModelInfo) =>
          model.status === "ready" &&
          VIEW_CONFIGS.chat.modelFilter(model.variant) &&
          model.variant !== variant &&
          !model.pinned &&
          !isSpeechPipelineManagedVariant(model.variant);
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
        await api.loadModel(variant, { signal: abortController.signal });
        if (!cancelledModelLoadsRef.current.has(variant)) {
          // Loading a speech-pipeline stack member (diarization checkpoint,
          // ASR, aligner, refiner LLM) is pipeline setup, not a model switch:
          // adopting it as the global selection overwrote — and since the
          // persistence change, permanently rewrote — whatever the user had
          // selected.
          if (!isSpeechPipelineManagedVariant(variant)) {
            adoptUserSelectedModel(variant);
          }
          void trackModelLoaded(variant);
          notify({
            title: "Model loaded",
            description: `${getModelLabel(variant)} is now active.`,
            tone: "success",
          });
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
    [adoptUserSelectedModel, getModelLabel, models, notify, refreshModels],
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
        await api.unloadModel(variant);
        await refreshModels();
        setSelectedModelState((current) =>
          current === variant ? null : current,
        );
        if (selectedModel === variant) {
          userSelectedModelRef.current = false;
          persistUserSelectedModel(null);
        }
        notify({
          title: cancellingLoad ? "Model load cancelled" : "Model unloaded",
          description: cancellingLoad
            ? `${getModelLabel(variant)} was stopped and unloaded.`
            : `${getModelLabel(variant)} was unloaded from memory.`,
          tone: "info",
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
        setError("Failed to delete model. Please try again.");
        notify({
          title: "Delete failed",
          description: `Izwi could not delete ${getModelLabel(variant)}.`,
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

  const value = useMemo<ModelCatalogContextValue>(
    () => ({
      models,
      selectedModel,
      loading,
      error,
      catalogError,
      downloadProgress,
      readyModelsCount: models.filter((model) => model.status === "ready").length,
      residencySummary,
      selectModel,
      reportError,
      clearError,
      refreshModels,
      downloadModel,
      cancelModelDownload,
      loadModel,
      unloadModel,
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
      reportError,
      residencySummary,
      selectModel,
      selectedModel,
      unloadModel,
    ],
  );

  return (
    <ModelCatalogContext.Provider value={value}>
      {children}
    </ModelCatalogContext.Provider>
  );
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
