import { lazy, Suspense, useEffect, useRef, type ReactNode } from "react";
import {
  Navigate,
  Route,
  Routes,
  useLocation,
  useParams,
} from "react-router-dom";
import { routeIdFromPathname, trackRouteViewed } from "@/app/analytics/events";
import { AppLayout } from "@/app/layouts/AppLayout";
import { useModelCatalog } from "@/app/providers/ModelCatalogProvider";
import { DocumentTitle } from "@/app/router/DocumentTitle";
import { NotFoundPage } from "@/app/router/NotFoundPage";
import { legacyDiarizationTarget } from "@/app/router/routePresentation";
import type {
  ModelsRouteProps,
  SharedPageProps,
  VoiceRouteProps,
} from "@/app/router/types";
import { getModelDisplayName } from "@/features/models/catalog/modelMetadata";

// Route code loads lazily, then preloads once the app is idle so later
// navigation does not wait on a chunk (react-router keeps the previous page
// on screen while a lazy route suspends inside a transition).
const ROUTE_MODULES = {
  TextToSpeechPage: () => import("@/features/text-to-speech/route"),
  StudioPage: () => import("@/features/studio/route"),
  VoiceStudioPage: () => import("@/features/voice-studio/route"),
  SpeechTextPage: () => import("@/features/speech-text/route"),
  ChatPage: () => import("@/features/chat/route"),
  VoicePage: () => import("@/features/voice/route"),
  MyModelsPage: () => import("@/features/models/route"),
  SettingsPage: () => import("@/features/settings/route"),
};

function preloadRouteModules() {
  for (const load of Object.values(ROUTE_MODULES)) {
    void load().catch(() => undefined);
  }
}

const TextToSpeechPage = lazy(async () => ({
  default: (await ROUTE_MODULES.TextToSpeechPage()).TextToSpeechPage,
}));

const StudioPage = lazy(async () => ({
  default: (await ROUTE_MODULES.StudioPage()).StudioPage,
}));

const VoiceStudioPage = lazy(async () => ({
  default: (await ROUTE_MODULES.VoiceStudioPage()).VoiceStudioPage,
}));

const SpeechTextPage = lazy(async () => ({
  default: (await ROUTE_MODULES.SpeechTextPage()).SpeechTextPage,
}));

const ChatPage = lazy(async () => ({
  default: (await ROUTE_MODULES.ChatPage()).ChatPage,
}));

const VoicePage = lazy(async () => ({
  default: (await ROUTE_MODULES.VoicePage()).VoicePage,
}));

const MyModelsPage = lazy(async () => ({
  default: (await ROUTE_MODULES.MyModelsPage()).MyModelsPage,
}));

const SettingsPage = lazy(async () => ({
  default: (await ROUTE_MODULES.SettingsPage()).SettingsPage,
}));

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

function LegacyDiarizationRedirect() {
  const { recordId } = useParams<{ recordId?: string }>();
  const location = useLocation();

  return (
    <Navigate
      to={legacyDiarizationTarget(recordId, location.search)}
      replace
    />
  );
}

export function AppRoutes() {
  const location = useLocation();
  const hasTrackedInitialRouteRef = useRef(false);

  useEffect(() => {
    const idle =
      typeof window.requestIdleCallback === "function"
        ? window.requestIdleCallback(preloadRouteModules, { timeout: 4000 })
        : window.setTimeout(preloadRouteModules, 1500);
    return () => {
      if (typeof window.cancelIdleCallback === "function") {
        window.cancelIdleCallback(idle);
      } else {
        window.clearTimeout(idle);
      }
    };
  }, []);
  const {
    models,
    selectedModel,
    loading,
    catalogError,
    downloadProgress,
    readyModelsCount,
    residencySummary,
    removedModels,
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
    residencySummary,
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
    removedModels,
    loading,
    downloadProgress,
    onDownload: downloadModel,
    onCancelDownload: cancelModelDownload,
    onLoad: loadModel,
    onUnload: unloadModel,
    onDelete: deleteModel,
    onRefresh: refreshModels,
  };
  const selectedModelInfo = models.find(
    (model) => model.variant === selectedModel,
  );
  const selectedModelLabel = selectedModel
    ? getModelDisplayName(selectedModel, selectedModelInfo?.display_name)
    : null;
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
    <>
      <DocumentTitle />
      <Routes>
        <Route
          element={
            <AppLayout
              readyModelsCount={readyModelsCount}
              selectedModelLabel={selectedModelLabel}
              catalogError={catalogError}
              onRetryModelCatalog={refreshModels}
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
          <Route
            path="/transcription"
            element={withSuspense(<SpeechTextPage {...pageProps} />)}
          />
          <Route
            path="/transcription/:recordId"
            element={withSuspense(<SpeechTextPage {...pageProps} />)}
          />
          <Route path="/diarization" element={<LegacyDiarizationRedirect />} />
          <Route
            path="/diarization/:recordId"
            element={<LegacyDiarizationRedirect />}
          />
          <Route
            path="/chat"
            element={withSuspense(<ChatPage {...pageProps} />)}
          />
          <Route
            path="/voice"
            element={withSuspense(<VoicePage {...voicePageProps} />)}
          />
          <Route
            path="/models"
            element={withSuspense(<MyModelsPage {...modelsPageProps} />)}
          />
          <Route
            path="/settings"
            element={withSuspense(<SettingsPage />)}
          />
          <Route path="/my-models" element={<Navigate to="/models" replace />} />
          <Route path="/" element={<Navigate to="/voice" replace />} />
          <Route path="*" element={<NotFoundPage />} />
        </Route>
      </Routes>
    </>
  );
}
