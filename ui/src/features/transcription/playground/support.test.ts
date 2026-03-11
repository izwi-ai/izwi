import { describe, expect, it } from "vitest";
import {
  buildTranscriptionRealtimeWebSocketUrl,
  encodeTranscriptionRealtimePcm16Frame,
  formatAudioDuration,
  formatClockTime,
  summarizeRecord,
} from "./support";

describe("transcription playground support", () => {
  it("builds the realtime websocket endpoint from an api base url", () => {
    expect(
      buildTranscriptionRealtimeWebSocketUrl("https://api.example.com/v1"),
    ).toBe("wss://api.example.com/v1/transcription/realtime/ws");
  });

  it("encodes realtime pcm16 frames with the expected header", () => {
    const frame = encodeTranscriptionRealtimePcm16Frame(
      new Uint8Array([1, 2, 3, 4]),
      16000,
      7,
    );
    const view = new DataView(frame.buffer);

    expect(String.fromCharCode(...frame.slice(0, 4))).toBe("ITRW");
    expect(view.getUint8(4)).toBe(1);
    expect(view.getUint8(5)).toBe(1);
    expect(view.getUint32(8, true)).toBe(16000);
    expect(view.getUint32(12, true)).toBe(7);
    expect(Array.from(frame.slice(16))).toEqual([1, 2, 3, 4]);
  });

  it("summarizes records with a normalized preview and character count", () => {
    expect(
      summarizeRecord({
        id: "record-1",
        created_at: 123,
        model_id: "Parakeet-TDT-0.6B-v3",
        aligner_model_id: null,
        language: "English",
        duration_secs: 12.3,
        processing_time_ms: 45,
        rtf: 0.2,
        audio_mime_type: "audio/wav",
        audio_filename: "clip.wav",
        transcription: " Hello   world ",
        segments: [],
        words: [],
      }),
    ).toMatchObject({
      transcription_preview: "Hello world",
      transcription_chars: 15,
    });
  });

  it("formats durations consistently", () => {
    expect(formatAudioDuration(12.34)).toBe("12.3s");
    expect(formatAudioDuration(61)).toBe("1m 1s");
    expect(formatClockTime(125)).toBe("2:05");
  });
});
