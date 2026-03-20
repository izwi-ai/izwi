import { describe, expect, it } from "vitest";

import type { DiarizationRecord } from "../api";
import { buildDiarizationExport } from "./diarizationExport";

const record = {
  id: "diar-export-1",
  created_at: Date.UTC(2026, 2, 10, 10, 0, 0),
  model_id: "diar_streaming_sortformer_4spk-v2.1",
  asr_model_id: "Parakeet-TDT-0.6B-v3",
  aligner_model_id: "Qwen3-ForcedAligner-0.6B",
  llm_model_id: null,
  min_speakers: 1,
  max_speakers: 4,
  min_speech_duration_ms: 240,
  min_silence_duration_ms: 200,
  enable_llm_refinement: false,
  processing_time_ms: 120,
  duration_secs: 6.4,
  rtf: 0.5,
  speaker_count: 2,
  corrected_speaker_count: 2,
  alignment_coverage: 1,
  unattributed_words: 0,
  llm_refined: false,
  asr_text: "Hello there. Hi back.",
  raw_transcript: "",
  transcript: "",
  segments: [],
  words: [],
  utterances: [
    {
      speaker: "SPEAKER_00",
      start: 0,
      end: 1.25,
      text: "Hello there.",
      word_start: 0,
      word_end: 1,
    },
    {
      speaker: "SPEAKER_01",
      start: 1.25,
      end: 2.5,
      text: "Hi back.",
      word_start: 2,
      word_end: 3,
    },
  ],
  speaker_name_overrides: {
    SPEAKER_00: "Alice",
    SPEAKER_01: "Bob",
  },
  audio_mime_type: "audio/wav",
  audio_filename: "meeting.wav",
} satisfies DiarizationRecord;

describe("buildDiarizationExport", () => {
  it("preserves corrected labels and timestamps across export formats", () => {
    const txt = buildDiarizationExport(record, "txt");
    const json = buildDiarizationExport(record, "json");
    const srt = buildDiarizationExport(record, "srt");
    const vtt = buildDiarizationExport(record, "vtt");

    expect(txt.content).toContain("Alice [0.00s - 1.25s]: Hello there.");
    expect(txt.filename).toBe("meeting.txt");

    expect(json.content).toContain('"speaker": "Alice"');
    expect(json.content).toContain('"speaker": "Bob"');
    expect(json.filename).toBe("meeting.json");

    expect(srt.content).toContain("00:00:00,000 --> 00:00:01,250");
    expect(srt.content).toContain("Alice: Hello there.");
    expect(srt.filename).toBe("meeting.srt");

    expect(vtt.content).toContain("WEBVTT");
    expect(vtt.content).toContain("00:00:00.000 --> 00:00:01.250");
    expect(vtt.content).toContain("Bob: Hi back.");
    expect(vtt.filename).toBe("meeting.vtt");
  });
});
