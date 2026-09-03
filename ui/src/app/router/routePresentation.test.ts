import { describe, expect, it } from "vitest";

import {
  documentTitleForLocation,
  legacyDiarizationTarget,
} from "@/app/router/routePresentation";

describe("legacyDiarizationTarget", () => {
  it("opens the unified creation flow for a legacy collection link", () => {
    expect(legacyDiarizationTarget(null, "")).toBe(
      "/transcription?create=diarization",
    );
  });

  it("preserves compatible collection query state", () => {
    expect(legacyDiarizationTarget(undefined, "?model=sortformer&mode=legacy")).toBe(
      "/transcription?model=sortformer&create=diarization",
    );
  });

  it("opens a legacy detail link in diarization mode", () => {
    expect(legacyDiarizationTarget("record / one", "?model=sortformer")).toBe(
      "/transcription/record%20%2F%20one?model=sortformer&mode=diarization",
    );
  });
});

describe("documentTitleForLocation", () => {
  it.each([
    ["/voice", "", "Voice · Izwi"],
    ["/chat", "", "Chat · Izwi"],
    ["/transcription", "", "Transcription · Izwi"],
    ["/transcription", "?create=diarization", "New Diarization · Izwi"],
    ["/transcription/tx-1", "", "Transcription Record · Izwi"],
    ["/transcription/diar-1", "?mode=diarization", "Diarization Record · Izwi"],
    ["/text-to-speech", "", "Text to Speech · Izwi"],
    ["/text-to-speech/tts-1", "", "Text-to-Speech Record · Izwi"],
    ["/studio/project-1", "", "Studio Project · Izwi"],
    ["/models", "", "Models · Izwi"],
    ["/settings", "", "Settings · Izwi"],
    ["/missing", "", "Page Not Found · Izwi"],
  ])("maps %s%s to %s", (pathname, search, expected) => {
    expect(documentTitleForLocation(pathname, search)).toBe(expected);
  });
});
