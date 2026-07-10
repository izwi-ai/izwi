import { describe, expect, it } from "vitest";
import {
  TRANSCRIPTION_REALTIME_PROTOCOL,
  TRANSCRIPTION_REALTIME_VERSION,
  type TranscriptionRealtimeV3ServerEnvelope,
  buildTranscriptionRealtimeV3SessionStart,
  buildTranscriptionRealtimeWebSocketUrl,
  encodeTranscriptionRealtimePcm16Frame,
  formatAudioDuration,
  formatClockTime,
  isTranscriptionRealtimeV3ServerEnvelope,
  isValidTranscriptionRealtimeV3Successor,
  normalizeSummaryStatus,
  summaryStatusLabel,
  summarizeRecord,
} from "./support";

describe("transcription playground support", () => {
  it("builds the realtime websocket endpoint from an api base url", () => {
    expect(
      buildTranscriptionRealtimeWebSocketUrl("https://api.example.com/v1"),
    ).toBe("wss://api.example.com/v1/speech-to-text/realtime/ws");
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

  it("serializes the opt-in v3 negotiation payload with a stable shape", () => {
    expect(
      JSON.stringify(
        buildTranscriptionRealtimeV3SessionStart(
          "Parakeet-TDT-0.6B-v3",
          "English",
        ),
      ),
    ).toBe(
      '{"type":"session_start","protocol":"transcription_realtime","version":3,"model_id":"Parakeet-TDT-0.6B-v3","language":"English"}',
    );
  });

  it("recognizes the typed SessionReady golden envelope", () => {
    const ready = typedEnvelope(1, 0, {
      type: "session_ready",
      data: {
        accepted_version: TRANSCRIPTION_REALTIME_VERSION,
        owner_instance_id: "process-42",
        resumable: false,
        resume_window_ms: 0,
      },
    });

    expect(isTranscriptionRealtimeV3ServerEnvelope(ready)).toBe(true);
    expect(JSON.stringify(ready)).toBe(
      '{"protocol":"transcription_realtime","version":3,"event_id":1,"sequence":0,"session_id":"session-1","connection_epoch":0,"timestamp_ms":1725000000123,"type":"session_ready","data":{"accepted_version":3,"owner_instance_id":"process-42","resumable":false,"resume_window_ms":0}}',
    );
  });

  it("rejects malformed or unknown v3 payloads", () => {
    const malformed = typedEnvelope(1, 0, {
      type: "transcript_partial",
      data: { text: 42, revision: 1 },
    } as never);
    const unknown = typedEnvelope(1, 0, {
      type: "future_event",
    } as never);

    expect(isTranscriptionRealtimeV3ServerEnvelope(malformed)).toBe(false);
    expect(isTranscriptionRealtimeV3ServerEnvelope(unknown)).toBe(false);
  });

  it("enforces strict v3 event sequencing", () => {
    const first = typedEnvelope(1, 0, {
      type: "session_started",
    });
    const next = typedEnvelope(2, 1, {
      type: "transcript_partial",
      data: { text: "hello", revision: 1, language: "en" },
    });
    const duplicate = typedEnvelope(3, 1, {
      type: "transcript_final",
      data: { text: "hello", revision: 2, language: "en" },
    });

    expect(isValidTranscriptionRealtimeV3Successor(null, first)).toBe(true);
    expect(isValidTranscriptionRealtimeV3Successor(first, next)).toBe(true);
    expect(isValidTranscriptionRealtimeV3Successor(next, duplicate)).toBe(
      false,
    );
  });

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

function typedEnvelope(
  eventId: number,
  sequence: number,
  payload: Pick<TranscriptionRealtimeV3ServerEnvelope, "type"> &
    Partial<TranscriptionRealtimeV3ServerEnvelope>,
): TranscriptionRealtimeV3ServerEnvelope {
  return {
    protocol: TRANSCRIPTION_REALTIME_PROTOCOL,
    version: TRANSCRIPTION_REALTIME_VERSION,
    event_id: eventId,
    sequence,
    session_id: "session-1",
    connection_epoch: 0,
    timestamp_ms: 1_725_000_000_123,
    ...payload,
  } as TranscriptionRealtimeV3ServerEnvelope;
}
