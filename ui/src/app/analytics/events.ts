import { trackAnalyticsEvent } from "@/app/analytics/client";

export type RouteId =
  | "voice"
  | "chat"
  | "transcription"
  | "text_to_speech"
  | "studio"
  | "voices"
  | "models"
  | "settings";

export function routeIdFromPathname(pathname: string): RouteId | null {
  const firstSegment = pathname.split("/").filter(Boolean)[0];
  switch (firstSegment) {
    case "voice":
      return "voice";
    case "chat":
      return "chat";
    case "transcription":
      return "transcription";
    case "diarization":
      return "transcription";
    case "text-to-speech":
      return "text_to_speech";
    case "studio":
      return "studio";
    case "voices":
    case "voice-cloning":
    case "voice-design":
      return "voices";
    case "models":
    case "my-models":
      return "models";
    case "settings":
      return "settings";
    default:
      return null;
  }
}

export function modelCategoryFromVariant(
  modelVariant: string,
): "chat" | "asr" | "tts" | "other" {
  const normalized = modelVariant.toLowerCase();

  if (
    normalized.includes("qwen3-asr") ||
    normalized.includes("whisper") ||
    normalized.includes("parakeet") ||
    normalized.includes("voxtral") ||
    normalized.includes("diar_")
  ) {
    return "asr";
  }

  if (
    normalized.includes("tts") ||
    normalized.includes("kokoro") ||
    normalized.includes("voicedesign") ||
    normalized.includes("customvoice") ||
    normalized.includes("base")
  ) {
    return "tts";
  }

  if (
    normalized.includes("qwen3") ||
    normalized.includes("qwen3.5") ||
    normalized.includes("gemma") ||
    normalized.includes("lfm")
  ) {
    return "chat";
  }

  return "other";
}

export async function trackAppOpened() {
  await trackAnalyticsEvent("app_opened", {
    entrypoint: "desktop",
    release_channel: "stable",
  });
}

export async function trackRouteViewed(routeId: RouteId) {
  await trackAnalyticsEvent("route_viewed", {
    route_id: routeId,
  });
}

export async function trackOnboardingViewed() {
  await trackAnalyticsEvent("onboarding_viewed", {
    onboarding_version: "v1",
  });
}

export async function trackOnboardingCompleted(
  setupMode: "quick" | "custom" | "skip",
) {
  await trackAnalyticsEvent("onboarding_completed", {
    setup_mode: setupMode,
  });
}

export async function trackAnalyticsConsentChanged(
  state: "opted_in" | "opted_out",
  source: "onboarding" | "settings",
) {
  await trackAnalyticsEvent("analytics_consent_changed", {
    state,
    source,
  });
}

export async function trackThemePreferenceChanged(
  themePreference: "system" | "light" | "dark",
) {
  await trackAnalyticsEvent("theme_preference_changed", {
    theme_preference: themePreference,
  });
}

export async function trackModelDownloadStarted(modelVariant: string) {
  await trackAnalyticsEvent("model_download_started", {
    model_variant: modelVariant,
    model_category: modelCategoryFromVariant(modelVariant),
  });
}

export async function trackModelDownloadCompleted(modelVariant: string) {
  await trackAnalyticsEvent("model_download_completed", {
    model_variant: modelVariant,
    model_category: modelCategoryFromVariant(modelVariant),
  });
}

export async function trackModelLoaded(modelVariant: string) {
  await trackAnalyticsEvent("model_loaded", {
    model_variant: modelVariant,
    model_category: modelCategoryFromVariant(modelVariant),
  });
}

export async function trackUpdateCheckStarted(
  source: "manual" | "background",
) {
  await trackAnalyticsEvent("update_check_started", {
    source,
  });
}

export async function trackUpdateCheckCompleted(
  outcome: "update_available" | "no_update" | "failed",
  version?: string,
  reason?: string,
) {
  await trackAnalyticsEvent("update_check_completed", {
    outcome,
    ...(version ? { version } : {}),
    ...(reason ? { reason } : {}),
  });
}

export async function trackUpdateInstallStarted(version: string) {
  await trackAnalyticsEvent("update_install_started", {
    version,
  });
}

export async function trackUpdateInstallCompleted(version: string) {
  await trackAnalyticsEvent("update_install_completed", {
    version,
  });
}

export async function trackUpdateInstallFailed(reason: string) {
  await trackAnalyticsEvent("update_install_failed", {
    reason,
  });
}
