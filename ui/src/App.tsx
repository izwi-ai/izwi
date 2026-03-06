import { useState, useEffect, useCallback, useRef } from "react";
import { Navigate, Route, Routes } from "react-router-dom";
import { api, ModelInfo } from "./api";
import { useTheme } from "@/app/providers/ThemeProvider";
import { Layout } from "./components/Layout";
import { VIEW_CONFIGS } from "./types";
import {
  TextToSpeechPage,
  VoiceCloningPage,
  VoiceDesignPage,
  TranscriptionPage,
  DiarizationPage,
  ChatPage,
  VoicePage,
  MyModelsPage,
} from "./pages";

function App() {
  const { resolvedTheme, themePreference, setThemePreference } = useTheme();
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [selectedModel, setSelectedModel] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [downloadProgress, setDownloadProgress] = useState<
    Record<
      string,
      {
        percent: number;
        currentFile: string;
        status: string;
        downloadedBytes: number;
        totalBytes: number;
      }
    >
  >({});

  // Use ref to track polling state and active downloads
  const pollingRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const activeDownloadsRef = useRef<Set<string>>(new Set());
  const activeModelLoadsRef = useRef<Set<string>>(new Set());
  const eventSourcesRef = useRef<Record<string, EventSource>>({});
  const reconnectTimersRef = useRef<Record<string, ReturnType<typeof setTimeout>>>({});
  const streamWatchdogTimersRef = useRef<
    Record<string, ReturnType<typeof setInterval>>
  >({});
  const lastProgressAtRef = useRef<Record<string, number>>({});
  const suppressReconnectRef = useRef<Set<string>>(new Set());
  const initializedRef = useRef(false);

  const loadModels = useCallback(async () => {
    try {
      const response = await api.listModels();
      const mergedModels = response.models.map((model) =>
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

      // Auto-select first ready model
      const readyModel = mergedModels.find((m) => m.status === "ready");
      if (readyModel && !selectedModel) {
        setSelectedModel(readyModel.variant);
      }
    } catch (err) {
      console.error("Failed to load models:", err);
    }
  }, [selectedModel]);

  useEffect(() => {
    // Prevent duplicate calls from React StrictMode
    if (initializedRef.current) return;
    initializedRef.current = true;

    const init = async () => {
      setLoading(true);
      await loadModels();
      setLoading(false);
    };
    init();
  }, []);

  // Smart polling: only poll when there are active operations, use ref to prevent duplicates
  useEffect(() => {
    const hasActiveOperations = models.some(
      (m) => m.status === "downloading" || m.status === "loading",
    );

    // Clear existing polling if no active operations
    if (!hasActiveOperations) {
      if (pollingRef.current) {
        clearInterval(pollingRef.current);
        pollingRef.current = null;
      }
      return;
    }

    // Only start polling if not already polling
    if (!pollingRef.current) {
      pollingRef.current = setInterval(loadModels, 3000); // Increased to 3s to reduce server load
    }

    return () => {
      if (pollingRef.current) {
        clearInterval(pollingRef.current);
        pollingRef.current = null;
      }
    };
  }, [models, loadModels]);

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

  // Connect to SSE for real-time download progress
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
          console.error(`Stream watchdog check failed for ${variant}:`, watchdogErr);
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

          // Close connection when complete
          if (
            data.status === "completed" ||
            data.status === "error" ||
            data.status === "cancelled"
          ) {
            closeDownloadStream(variant);
            activeDownloadsRef.current.delete(variant);
            suppressReconnectRef.current.delete(variant);

            // Refresh models after completion
            void loadModels();

            // Clear progress after delay
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
            console.error(`Reconnect check failed for ${variant}:`, reconnectErr);
          }

          activeDownloadsRef.current.delete(variant);
          clearDownloadProgress(variant);
          await loadModels();
        }, 1500);
      };
    },
    [
      clearDownloadProgress,
      clearReconnectTimer,
      closeDownloadStream,
      loadModels,
    ],
  );

  // Keep stream subscriptions aligned to active downloading models,
  // including downloads that started before this page mounted.
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
      if (!eventSourcesRef.current[variant] && !reconnectTimersRef.current[variant]) {
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

  const handleDownload = async (variant: string) => {
    try {
      // Prevent duplicate downloads
      if (activeDownloadsRef.current.has(variant)) {
        return;
      }
      suppressReconnectRef.current.delete(variant);
      clearReconnectTimer(variant);
      activeDownloadsRef.current.add(variant);

      setModels((prev) =>
        prev.map((m) =>
          m.variant === variant ? { ...m, status: "downloading" as const } : m,
        ),
      );

      // Start download
      const response = await api.downloadModel(variant);

      if (response.status === "started" || response.status === "downloading") {
        // Let the model-status effect own stream subscription to avoid
        // duplicate EventSource open/close churn.
        await loadModels();
      } else {
        // Download already complete or not started
        activeDownloadsRef.current.delete(variant);
        await loadModels();
      }
    } catch (err: any) {
      console.error("Download failed:", err);
      activeDownloadsRef.current.delete(variant);
      setError(err.message || "Failed to download model. Please try again.");

      closeDownloadStream(variant);

      await loadModels();
    }
  };

  const handleCancelDownload = async (variant: string) => {
    try {
      suppressReconnectRef.current.add(variant);
      closeDownloadStream(variant);

      activeDownloadsRef.current.delete(variant);

      await api.cancelDownload(variant);

      // Update UI immediately
      clearDownloadProgress(variant);

      setModels((prev) =>
        prev.map((m) =>
          m.variant === variant
            ? {
                ...m,
                status: "not_downloaded" as const,
                download_progress: null,
              }
            : m,
        ),
      );

      await loadModels();
    } catch (err: any) {
      suppressReconnectRef.current.delete(variant);
      console.error("Cancel failed:", err);
      setError(err.message || "Failed to cancel download.");
      await loadModels();
    }
  };

  const handleLoad = async (variant: string) => {
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
        prev.map((m) =>
          m.variant === variant
            ? { ...m, status: "loading" as const }
            : isChatTarget &&
                m.status === "ready" &&
                VIEW_CONFIGS.chat.modelFilter(m.variant)
              ? { ...m, status: "downloaded" as const }
              : m,
        ),
      );

      await api.loadModel(variant);
      setSelectedModel(variant);
    } catch (err) {
      console.error("Load failed:", err);
      setError("Failed to load model. Please try again.");
    } finally {
      activeModelLoadsRef.current.delete(variant);
      // Refresh models after load completes or fails
      await loadModels();
    }
  };

  const handleUnload = async (variant: string) => {
    try {
      await api.unloadModel(variant);
      // Refresh models after unload
      await loadModels();
      if (selectedModel === variant) {
        setSelectedModel(null);
      }
    } catch (err) {
      console.error("Unload failed:", err);
      setError("Failed to unload model. Please try again.");
    }
  };

  const handleDelete = async (variant: string) => {
    try {
      suppressReconnectRef.current.add(variant);
      closeDownloadStream(variant);
      activeDownloadsRef.current.delete(variant);
      clearDownloadProgress(variant);

      await api.deleteModel(variant);
      // Refresh models after delete
      await loadModels();
      if (selectedModel === variant) {
        setSelectedModel(null);
      }
    } catch (err) {
      suppressReconnectRef.current.delete(variant);
      console.error("Delete failed:", err);
      setError("Failed to delete model. Please try again.");
      await loadModels();
    }
  };

  const readyModelsCount = models.filter((m) => m.status === "ready").length;

  const pageProps = {
    models,
    selectedModel,
    loading,
    downloadProgress,
    onDownload: handleDownload,
    onCancelDownload: handleCancelDownload,
    onLoad: handleLoad,
    onUnload: handleUnload,
    onDelete: handleDelete,
    onSelect: setSelectedModel,
    onError: setError,
    onRefresh: loadModels,
  };

  return (
    <Routes>
      <Route
        element={
          <Layout
            error={error}
            onErrorDismiss={() => setError(null)}
            readyModelsCount={readyModelsCount}
            resolvedTheme={resolvedTheme}
            themePreference={themePreference}
            onThemePreferenceChange={setThemePreference}
          />
        }
      >
        <Route
          path="/text-to-speech"
          element={<TextToSpeechPage {...pageProps} />}
        />
        <Route
          path="/voice-cloning"
          element={<VoiceCloningPage {...pageProps} />}
        />
        <Route
          path="/voice-design"
          element={<VoiceDesignPage {...pageProps} />}
        />
        <Route
          path="/transcription"
          element={<TranscriptionPage {...pageProps} />}
        />
        <Route path="/diarization" element={<DiarizationPage {...pageProps} />} />
        <Route path="/chat" element={<ChatPage {...pageProps} />} />
        <Route
          path="/voice"
          element={
            <VoicePage
              models={models}
              loading={loading}
              downloadProgress={downloadProgress}
              onDownload={handleDownload}
              onCancelDownload={handleCancelDownload}
              onLoad={handleLoad}
              onUnload={handleUnload}
              onDelete={handleDelete}
              onError={setError}
            />
          }
        />
        <Route
          path="/models"
          element={
            <MyModelsPage
              models={models}
              loading={loading}
              downloadProgress={downloadProgress}
              onDownload={handleDownload}
              onCancelDownload={handleCancelDownload}
              onLoad={handleLoad}
              onUnload={handleUnload}
              onDelete={handleDelete}
              onRefresh={loadModels}
            />
          }
        />
        <Route path="/my-models" element={<Navigate to="/models" replace />} />
        <Route path="/" element={<Navigate to="/voice" replace />} />
        <Route path="*" element={<Navigate to="/voice" replace />} />
      </Route>
    </Routes>
  );
}

export default App;
