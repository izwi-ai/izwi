import { beforeEach, describe, expect, it, vi } from "vitest";

import { trackAnalyticsEvent } from "@/app/analytics/client";
import {
  modelCategoryFromVariant,
  routeIdFromPathname,
  trackAnalyticsConsentChanged,
  trackModelLoaded,
  trackRouteViewed,
  trackThemePreferenceChanged,
} from "@/app/analytics/events";

vi.mock("@/app/analytics/client", () => ({
  trackAnalyticsEvent: vi.fn(),
}));

const mockTrackAnalyticsEvent = vi.mocked(trackAnalyticsEvent);

describe("analytics event helpers", () => {
  beforeEach(() => {
    mockTrackAnalyticsEvent.mockReset();
  });

  it("maps route paths into tracked route ids", () => {
    expect(routeIdFromPathname("/voice")).toBe("voice");
    expect(routeIdFromPathname("/chat/abc")).toBe("chat");
    expect(routeIdFromPathname("/text-to-speech/record_1")).toBe(
      "text_to_speech",
    );
    expect(routeIdFromPathname("/diarization")).toBe("transcription");
    expect(routeIdFromPathname("/voice-cloning")).toBe("voices");
    expect(routeIdFromPathname("/my-models")).toBe("models");
    expect(routeIdFromPathname("/unknown")).toBeNull();
  });

  it("classifies model variants by analytics category", () => {
    expect(modelCategoryFromVariant("Qwen3.5-4B")).toBe("chat");
    expect(modelCategoryFromVariant("Qwen3-ASR-0.6B-GGUF")).toBe("asr");
    expect(modelCategoryFromVariant("Kokoro-82M")).toBe("tts");
    expect(modelCategoryFromVariant("UnmappedModel")).toBe("other");
  });

  it("emits route and model events with normalized payloads", async () => {
    await trackRouteViewed("settings");
    await trackModelLoaded("Qwen3.5-4B");

    expect(mockTrackAnalyticsEvent).toHaveBeenNthCalledWith(1, "route_viewed", {
      route_id: "settings",
    });
    expect(mockTrackAnalyticsEvent).toHaveBeenNthCalledWith(2, "model_loaded", {
      model_variant: "Qwen3.5-4B",
      model_category: "chat",
    });
  });

  it("emits consent and theme change events", async () => {
    await trackAnalyticsConsentChanged("opted_in", "settings");
    await trackThemePreferenceChanged("dark");

    expect(mockTrackAnalyticsEvent).toHaveBeenNthCalledWith(
      1,
      "analytics_consent_changed",
      {
        state: "opted_in",
        source: "settings",
      },
    );
    expect(mockTrackAnalyticsEvent).toHaveBeenNthCalledWith(
      2,
      "theme_preference_changed",
      {
        theme_preference: "dark",
      },
    );
  });
});
