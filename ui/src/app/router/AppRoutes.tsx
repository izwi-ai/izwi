import { lazy, Suspense, useEffect, useRef, type ReactNode } from "react";
import {
  Navigate,
  Route,
  Routes,
  useLocation,
} from "react-router-dom";
import { routeIdFromPathname, trackRouteViewed } from "@/app/analytics/events";
import { AppLayout } from "@/app/layouts/AppLayout";
import { useModelCatalog } from "@/app/providers/ModelCatalogProvider";
import { useTheme } from "@/app/providers/ThemeProvider";
import type {
  ModelsRouteProps,
  SharedPageProps,
  VoiceRouteProps,
} from "@/app/router/types";
import { VOICE_STUDIO_ENABLED } from "@/shared/config/runtime";

const TextToSpeechPage = lazy(async () => {
  const module = await import("@/features/text-to-speech/route");
  return { default: module.TextToSpeechPage };
});

const StudioPage = lazy(async () => {
  const module = await import("@/features/studio/route");
  return { default: module.StudioPage };
});

const VoiceCloningPage = lazy(async () => {
  const module = await import("@/features/voice-cloning/route");
  return { default: module.VoiceCloningPage };
});

const VoiceDesignPage = lazy(async () => {
  const module = await import("@/features/voice-design/route");
  return { default: module.VoiceDesignPage };
});

const VoicesPage = lazy(async () => {
  const module = await import("@/features/voices/route");
  return { default: module.VoicesPage };
});

const VoiceStudioPage = lazy(async () => {
  const module = await import("@/features/voice-studio/route");
  return { default: module.VoiceStudioPage };
});

const SpeechTextPage = lazy(async () => {
  const module = await import("@/features/speech-text/route");
  return { default: module.SpeechTextPage };
});

const ChatPage = lazy(async () => {
  const module = await import("@/features/chat/route");
  return { default: module.ChatPage };
});

const VoicePage = lazy(async () => {
  const module = await import("@/features/voice/route");
  return { default: module.VoicePage };
});

const MyModelsPage = lazy(async () => {
  const module = await import("@/features/models/route");
  return { default: module.MyModelsPage };
});

const SettingsPage = lazy(async () => {
  const module = await import("@/features/settings/route");
  return { default: module.SettingsPage };
});

function RouteLoadingFallback() {
  return (
    <div className="flex min-h-[16rem] items-center justify-center rounded-2xl border border-border/60 bg-card/60 text-sm text-muted-foreground shadow-sm">
      Loading route...
    </div>
  );
}

function withSuspense(children: ReactNode) {
  return <Suspense fallback={<RouteLoadingFallback />}>{children}</Suspense>;
}

export function AppRoutes() {
  const location = useLocation();
  const hasTrackedInitialRouteRef = useRef(false);
  const { resolvedTheme, themePreference, setThemePreference } = useTheme();
  const {
    models,
    selectedModel,
    loading,
    downloadProgress,
    readyModelsCount,
    selectModel,
    reportError,
    refreshModels,
    downloadModel,
    cancelModelDownload,
    loadModel,
    unloadModel,
    deleteModel,
  } = useModelCatalog();

  const pageProps: SharedPageProps = {
    models,
    selectedModel,
    loading,
    downloadProgress,
    onDownload: downloadModel,
    onCancelDownload: cancelModelDownload,
    onLoad: loadModel,
    onUnload: unloadModel,
    onDelete: deleteModel,
    onSelect: (variant) => selectModel(variant),
    onError: reportError,
    onRefresh: refreshModels,
  };

  const voicePageProps: VoiceRouteProps = {
    models,
    loading,
    downloadProgress,
    onDownload: downloadModel,
    onCancelDownload: cancelModelDownload,
    onLoad: loadModel,
    onUnload: unloadModel,
    onDelete: deleteModel,
    onError: reportError,
  };

  const modelsPageProps: ModelsRouteProps = {
    models,
    loading,
    downloadProgress,
    onDownload: downloadModel,
    onCancelDownload: cancelModelDownload,
    onLoad: loadModel,
    onUnload: unloadModel,
    onDelete: deleteModel,
    onRefresh: refreshModels,
  };
  const selectedModelLabel =
    models.find((model) => model.variant === selectedModel)?.variant ??
    selectedModel;

  useEffect(() => {
    if (!hasTrackedInitialRouteRef.current) {
      hasTrackedInitialRouteRef.current = true;
      return;
    }
    const routeId = routeIdFromPathname(location.pathname);
    if (!routeId) {
      return;
    }
    void trackRouteViewed(routeId);
  }, [location.pathname]);

  return (
    <Routes>
      <Route
        element={
          <AppLayout
            readyModelsCount={readyModelsCount}
            selectedModelLabel={selectedModelLabel}
            resolvedTheme={resolvedTheme}
            themePreference={themePreference}
            onThemePreferenceChange={setThemePreference}
          />
        }
      >
        <Route
          path="/text-to-speech"
          element={withSuspense(<TextToSpeechPage {...pageProps} />)}
        />
        <Route
          path="/text-to-speech/:recordId"
          element={withSuspense(<TextToSpeechPage {...pageProps} />)}
        />
        <Route
          path="/studio"
          element={withSuspense(<StudioPage {...pageProps} />)}
        />
        <Route
          path="/studio/:projectId"
          element={withSuspense(<StudioPage {...pageProps} />)}
        />
        {VOICE_STUDIO_ENABLED ? (
          <>
            <Route
              path="/voices"
              element={withSuspense(<VoiceStudioPage {...pageProps} />)}
            />
            <Route
              path="/voice-cloning"
              element={<Navigate to="/voices?tab=clone" replace />}
            />
            <Route
              path="/voice-design"
              element={<Navigate to="/voices?tab=design" replace />}
            />
          </>
        ) : (
          <>
            <Route
              path="/voice-cloning"
              element={withSuspense(<VoiceCloningPage {...pageProps} />)}
            />
            <Route
              path="/voice-design"
              element={withSuspense(<VoiceDesignPage {...pageProps} />)}
            />
            <Route
              path="/voices"
              element={withSuspense(<VoicesPage {...pageProps} />)}
            />
          </>
        )}
        <Route
          path="/transcription"
          element={withSuspense(<SpeechTextPage {...pageProps} />)}
        />
        <Route
          path="/transcription/:recordId"
          element={withSuspense(<SpeechTextPage {...pageProps} />)}
        />
        <Route path="/chat" element={withSuspense(<ChatPage {...pageProps} />)} />
        <Route
          path="/voice"
          element={withSuspense(<VoicePage {...voicePageProps} />)}
        />
        <Route
          path="/models"
          element={withSuspense(<MyModelsPage {...modelsPageProps} />)}
        />
        <Route path="/settings" element={withSuspense(<SettingsPage />)} />
        <Route path="/my-models" element={<Navigate to="/models" replace />} />
        <Route path="/" element={<Navigate to="/voice" replace />} />
        <Route path="*" element={<Navigate to="/voice" replace />} />
      </Route>
    </Routes>
  );
}
