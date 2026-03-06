import { lazy, Suspense, type ReactNode } from "react";
import { Navigate, Route, Routes } from "react-router-dom";
import { AppLayout } from "@/app/layouts/AppLayout";
import type { AppRoutesProps } from "@/app/router/types";

const TextToSpeechPage = lazy(async () => {
  const module = await import("@/pages/TextToSpeechPage");
  return { default: module.TextToSpeechPage };
});

const VoiceCloningPage = lazy(async () => {
  const module = await import("@/pages/VoiceCloningPage");
  return { default: module.VoiceCloningPage };
});

const VoiceDesignPage = lazy(async () => {
  const module = await import("@/pages/VoiceDesignPage");
  return { default: module.VoiceDesignPage };
});

const TranscriptionPage = lazy(async () => {
  const module = await import("@/pages/TranscriptionPage");
  return { default: module.TranscriptionPage };
});

const DiarizationPage = lazy(async () => {
  const module = await import("@/pages/DiarizationPage");
  return { default: module.DiarizationPage };
});

const ChatPage = lazy(async () => {
  const module = await import("@/pages/ChatPage");
  return { default: module.ChatPage };
});

const VoicePage = lazy(async () => {
  const module = await import("@/pages/VoicePage");
  return { default: module.VoicePage };
});

const MyModelsPage = lazy(async () => {
  const module = await import("@/pages/MyModelsPage");
  return { default: module.MyModelsPage };
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

export function AppRoutes({
  error,
  onErrorDismiss,
  readyModelsCount,
  resolvedTheme,
  themePreference,
  onThemePreferenceChange,
  pageProps,
  voicePageProps,
  modelsPageProps,
}: AppRoutesProps) {
  return (
    <Routes>
      <Route
        element={
          <AppLayout
            error={error}
            onErrorDismiss={onErrorDismiss}
            readyModelsCount={readyModelsCount}
            resolvedTheme={resolvedTheme}
            themePreference={themePreference}
            onThemePreferenceChange={onThemePreferenceChange}
          />
        }
      >
        <Route
          path="/text-to-speech"
          element={withSuspense(<TextToSpeechPage {...pageProps} />)}
        />
        <Route
          path="/voice-cloning"
          element={withSuspense(<VoiceCloningPage {...pageProps} />)}
        />
        <Route
          path="/voice-design"
          element={withSuspense(<VoiceDesignPage {...pageProps} />)}
        />
        <Route
          path="/transcription"
          element={withSuspense(<TranscriptionPage {...pageProps} />)}
        />
        <Route
          path="/diarization"
          element={withSuspense(<DiarizationPage {...pageProps} />)}
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
        <Route path="/my-models" element={<Navigate to="/models" replace />} />
        <Route path="/" element={<Navigate to="/voice" replace />} />
        <Route path="*" element={<Navigate to="/voice" replace />} />
      </Route>
    </Routes>
  );
}
