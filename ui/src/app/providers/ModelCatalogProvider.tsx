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
import { api, type ModelInfo } from "@/api";
import { useNotifications } from "@/app/providers/NotificationProvider";
import { VIEW_CONFIGS } from "@/types";
import type { DownloadProgressMap } from "@/app/router/types";

interface ModelCatalogContextValue {
  models: ModelInfo[];
  selectedModel: string | null;
  loading: boolean;
  error: string | null;
  downloadProgress: DownloadProgressMap;
  readyModelsCount: number;
  selectModel: (variant: string | null) => void;
  reportError: (message: string) => void;
  clearError: () => void;
  refreshModels: () => Promise<void>;
  downloadModel: (variant: string) => Promise<void>;
  cancelModelDownload: (variant: string) => Promise<void>;
  loadModel: (variant: string) => Promise<void>;
  unloadModel: (variant: string) => Promise<void>;
  deleteModel: (variant: string) => Promise<void>;
}

const ModelCatalogContext = createContext<ModelCatalogContextValue | null>(null);

interface ModelCatalogProviderProps {
  children: ReactNode;
}

export function ModelCatalogProvider({
  children,
}: ModelCatalogProviderProps) {
  const { notify } = useNotifications();
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [selectedModel, setSelectedModelState] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [downloadProgress, setDownloadProgress] = useState<DownloadProgressMap>(
    {},
  );

  const pollingRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const activeDownloadsRef = useRef<Set<string>>(new Set());
  const activeModelLoadsRef = useRef<Set<string>>(new Set());
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

  const selectModel = useCallback((variant: string | null) => {
    setSelectedModelState(variant);
  }, []);

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

  const refreshModels = useCallback(async () => {
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
      setSelectedModelState((current) => {
        if (
          current &&
          mergedModels.some((model) => model.variant === current)
        ) {
          return current;
        }

        const readyModel = mergedModels.find((model) => model.status === "ready");
        return readyModel?.variant ?? null;
      });
    } catch (err) {
      console.error("Failed to load models:", err);
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

    const init = async () => {
      setLoading(true);
      await refreshModels();
      setLoading(false);
    };

    void init();
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

      try {
        const isChatTarget = VIEW_CONFIGS.chat.modelFilter(variant);
        const loadedChatModels = isChatTarget
          ? models.filter(
              (model) =>
                model.status === "ready" &&
                VIEW_CONFIGS.chat.modelFilter(model.variant) &&
                model.variant !== variant,
            )
          : [];

        for (const loadedModel of loadedChatModels) {
          await api.unloadModel(loadedModel.variant);
        }

        setModels((prev) =>
          prev.map((model) =>
            model.variant === variant
              ? { ...model, status: "loading" as const }
              : isChatTarget &&
                  model.status === "ready" &&
                  VIEW_CONFIGS.chat.modelFilter(model.variant)
                ? { ...model, status: "downloaded" as const }
                : model,
          ),
        );

        await api.loadModel(variant);
        setSelectedModelState(variant);
        notify({
          title: "Model loaded",
          description: `${getModelLabel(variant)} is now active.`,
          tone: "success",
        });
      } catch (err) {
        console.error("Load failed:", err);
        setError("Failed to load model. Please try again.");
        notify({
          title: "Model load failed",
          description: `Izwi could not load ${getModelLabel(variant)}.`,
          tone: "danger",
        });
      } finally {
        activeModelLoadsRef.current.delete(variant);
        await refreshModels();
      }
    },
    [getModelLabel, models, notify, refreshModels],
  );

  const unloadModel = useCallback(
    async (variant: string) => {
      try {
        await api.unloadModel(variant);
        await refreshModels();
        setSelectedModelState((current) =>
          current === variant ? null : current,
        );
        notify({
          title: "Model unloaded",
          description: `${getModelLabel(variant)} was unloaded from memory.`,
          tone: "info",
        });
      } catch (err) {
        console.error("Unload failed:", err);
        setError("Failed to unload model. Please try again.");
        notify({
          title: "Model unload failed",
          description: `Izwi could not unload ${getModelLabel(variant)}.`,
          tone: "danger",
        });
      }
    },
    [getModelLabel, notify, refreshModels],
  );

  const deleteModel = useCallback(
    async (variant: string) => {
      try {
        suppressReconnectRef.current.add(variant);
        closeDownloadStream(variant);
        activeDownloadsRef.current.delete(variant);
        clearDownloadProgress(variant);

        await api.deleteModel(variant);
        await refreshModels();
        setSelectedModelState((current) =>
          current === variant ? null : current,
        );
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
    ],
  );

  const value = useMemo<ModelCatalogContextValue>(
    () => ({
      models,
      selectedModel,
      loading,
      error,
      downloadProgress,
      readyModelsCount: models.filter((model) => model.status === "ready").length,
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
      loadModel,
      loading,
      models,
      refreshModels,
      reportError,
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
