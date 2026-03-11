import { describe, expect, it } from "vitest";

import type { TranscriptionRecord } from "@/api";
import { buildTranscriptionExport } from "@/utils/transcriptionExport";

const record = {
  id: "transcription-export-1",
  created_at: Date.UTC(2026, 2, 10, 10, 0, 0),
  model_id: "Qwen3-ASR-0.6B",
  aligner_model_id: "Qwen3-ForcedAligner-0.6B",
  language: "English",
  duration_secs: 2.5,
  processing_time_ms: 120,
  rtf: 0.5,
  audio_mime_type: "audio/wav",
  audio_filename: "meeting.wav",
  transcription: "Hello there. Hi back.",
  segments: [
    {
      start: 0,
      end: 1.25,
      text: "Hello there.",
      word_start: 0,
      word_end: 1,
    },
    {
      start: 1.25,
      end: 2.5,
      text: "Hi back.",
      word_start: 2,
      word_end: 3,
    },
  ],
  words: [
    { word: "Hello", start: 0, end: 0.5 },
    { word: "there.", start: 0.55, end: 1.25 },
    { word: "Hi", start: 1.25, end: 1.65 },
    { word: "back.", start: 1.7, end: 2.5 },
  ],
} satisfies TranscriptionRecord;

describe("buildTranscriptionExport", () => {
  it("preserves transcript timing across export formats", () => {
    const txt = buildTranscriptionExport(record, "txt");
    const json = buildTranscriptionExport(record, "json");
    const srt = buildTranscriptionExport(record, "srt");
    const vtt = buildTranscriptionExport(record, "vtt");

    expect(txt.content).toContain("[00:00.000 - 00:01.250] Hello there.");
    expect(txt.filename).toBe("meeting.txt");

    expect(json.content).toContain('"text": "Hello there."');
    expect(json.content).toContain('"timed": true');
    expect(json.filename).toBe("meeting.json");

    expect(srt.content).toContain("00:00:00,000 --> 00:00:01,250");
    expect(srt.content).toContain("Hello there.");
    expect(srt.filename).toBe("meeting.srt");

    expect(vtt.content).toContain("WEBVTT");
    expect(vtt.content).toContain("00:00:00.000 --> 00:00:01.250");
    expect(vtt.content).toContain("Hi back.");
    expect(vtt.filename).toBe("meeting.vtt");
  });

  it("adds metadata to txt exports when requested", () => {
    const txt = buildTranscriptionExport(record, "txt", {
      includeMetadata: true,
    });

    expect(txt.content).toContain("File: meeting.wav");
    expect(txt.content).toContain("Model: Qwen3-ASR-0.6B");
    expect(txt.content).toContain(
      "Timestamps: Enabled (Qwen3-ForcedAligner-0.6B)",
    );
  });
});
