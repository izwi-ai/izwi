import { describe, expect, it } from "vitest";
import {
  formatAudioDuration,
  formatClockTime,
  normalizeSummaryStatus,
  summaryStatusLabel,
  summarizeRecord,
} from "./support";

describe("transcription playground support", () => {
  it("summarizes records with a normalized preview and character count", () => {
    expect(
      summarizeRecord({
        id: "record-1",
        created_at: 123,
        model_id: "Parakeet-TDT-0.6B-v3",
        aligner_model_id: null,
        language: "English",
        processing_status: "ready",
        processing_error: null,
        duration_secs: 12.3,
        processing_time_ms: 45,
        rtf: 0.2,
        audio_mime_type: "audio/wav",
        audio_filename: "clip.wav",
        transcription: " Hello   world ",
        segments: [],
        words: [],
        summary_status: "ready",
        summary_model_id: "Qwen3.5-4B",
        summary_text: " Brief summary ",
        summary_error: null,
        summary_updated_at: 123,
      }),
    ).toMatchObject({
      transcription_preview: "Hello world",
      transcription_chars: 15,
      summary_status: "ready",
      summary_preview: "Brief summary",
      summary_chars: 15,
    });
  });

  it("normalizes summary status and labels", () => {
    expect(normalizeSummaryStatus(undefined, "summary", null)).toBe("ready");
    expect(normalizeSummaryStatus(undefined, null, "error")).toBe("failed");
    expect(summaryStatusLabel("pending")).toBe("Summary pending");
  });

  it("formats durations consistently", () => {
    expect(formatAudioDuration(12.34)).toBe("12.3s");
    expect(formatAudioDuration(61)).toBe("1m 1s");
    expect(formatClockTime(125)).toBe("2:05");
  });
});
