import { describe, expect, it } from "vitest";

import type { SpeechTextJobKind } from "@/api";
import { presentSpeechTextHistoryStatus } from "@/features/transcription/historyStatus";

const kinds: SpeechTextJobKind[] = [
  "transcription",
  "speaker_attributed_asr",
  "diarization",
];

describe("presentSpeechTextHistoryStatus", () => {
  it.each(kinds)("normalizes every visible state for %s", (kind) => {
    expect(presentSpeechTextHistoryStatus("pending", null, kind)).toMatchObject({
      status: "queued",
      label: "Queued",
      tone: "neutral",
    });
    expect(
      presentSpeechTextHistoryStatus("processing", null, kind),
    ).toMatchObject({
      status: "processing",
      label: "Processing",
      tone: "info",
    });
    expect(presentSpeechTextHistoryStatus("ready", null, kind)).toMatchObject({
      status: "ready",
      label: "Ready",
      tone: "success",
    });
    expect(presentSpeechTextHistoryStatus("failed", null, kind)).toMatchObject({
      status: "failed",
      label: "Failed",
      tone: "danger",
    });
    expect(presentSpeechTextHistoryStatus("canceled", null, kind)).toMatchObject({
      status: "canceled",
      label: "Canceled",
      tone: "neutral",
    });
    expect(
      presentSpeechTextHistoryStatus("engine_specific_state", null, kind),
    ).toMatchObject({
      status: "unknown",
      label: "Status Unknown",
      tone: "neutral",
    });
  });

  it("treats a technical error as failure even when the backend status is stale", () => {
    expect(
      presentSpeechTextHistoryStatus(
        "processing",
        "CUDA allocation failed",
        "transcription",
      ),
    ).toMatchObject({ status: "failed", label: "Failed" });
  });
});
