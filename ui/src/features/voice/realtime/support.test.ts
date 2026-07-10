import { describe, expect, it } from "vitest";
import voiceV2SessionReadyFixture from "./voice_v2_session_ready.fixture.json";
import {
  buildVoiceRealtimeV2SessionStart,
  buildVoiceRealtimeWebSocketUrl,
  encodeVoiceRealtimeClientPcm16Frame,
  formatModelVariantLabel,
  isAsrVariant,
  isValidVoiceRealtimeV2Successor,
  isVoiceRealtimeV2ServerEnvelope,
  isVoiceRealtimeServerEvent,
  isUnifiedAudioChatVariant,
  makeTranscriptEntryId,
  mergeSampleChunks,
  normalizeVoiceRealtimeV2Event,
  parseFinalAnswer,
  parseVoiceRealtimeAssistantAudioBinaryChunk,
  shouldStopVoiceRealtimePlayback,
  type VoiceRealtimeServerEvent,
  type VoiceRealtimeV2ServerEnvelope,
} from "./support";

describe("voice realtime support", () => {
  it("strips think tags from final answers", () => {
    expect(parseFinalAnswer("hello <think>hidden</think>world")).toBe(
      "hello world",
    );
    expect(parseFinalAnswer("<think>hidden")).toBe("");
  });

  it("formats known model labels predictably", () => {
    expect(formatModelVariantLabel("Parakeet-TDT-0.6B-v3")).toBe(
      "Parakeet 0.6B-v3",
    );
    expect(formatModelVariantLabel("Whisper-Large-v3-Turbo")).toBe(
      "Whisper Large v3 Turbo",
    );
    expect(formatModelVariantLabel("LFM2.5-Audio-1.5B-GGUF")).toBe(
      "LFM2.5 Audio 1.5B GGUF",
    );
  });

  it("detects the unified lfm25 audio variant", () => {
    expect(isUnifiedAudioChatVariant("LFM2.5-Audio-1.5B-GGUF")).toBe(true);
    expect(isUnifiedAudioChatVariant("Qwen3-1.7B-GGUF")).toBe(false);
  });

  it("keeps Voxtral out of modular voice ASR until native realtime support lands", () => {
    expect(isAsrVariant("Parakeet-TDT-0.6B-v3")).toBe(true);
    expect(isAsrVariant("Whisper-Large-v3-Turbo")).toBe(true);
    expect(isAsrVariant("Voxtral-Mini-4B-Realtime-2602")).toBe(false);
  });

  it("creates uuid transcript entry ids", () => {
    const id = makeTranscriptEntryId("user");
    expect(id).toMatch(
      /^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i,
    );
  });

  it("builds the voice realtime websocket endpoint from an api base url", () => {
    expect(buildVoiceRealtimeWebSocketUrl("http://localhost:3000/api")).toBe(
      "ws://localhost:3000/api/voice/realtime/ws",
    );
  });

  it("encodes client pcm frames with the expected header", () => {
    const frame = encodeVoiceRealtimeClientPcm16Frame(
      new Uint8Array([5, 6]),
      24000,
      9,
    );
    const view = new DataView(frame.buffer);

    expect(String.fromCharCode(...frame.slice(0, 4))).toBe("IVWS");
    expect(view.getUint8(4)).toBe(1);
    expect(view.getUint8(5)).toBe(1);
    expect(view.getUint32(8, true)).toBe(24000);
    expect(view.getUint32(12, true)).toBe(9);
    expect(Array.from(frame.slice(16))).toEqual([5, 6]);
  });

  it("parses assistant audio binary chunks", () => {
    const buffer = new ArrayBuffer(27);
    const bytes = new Uint8Array(buffer);
    bytes.set(Array.from("IVWS").map((char) => char.charCodeAt(0)));
    const view = new DataView(buffer);
    view.setUint8(4, 1);
    view.setUint8(5, 2);
    view.setUint16(6, 1, true);
    view.setBigUint64(8, 12n, true);
    view.setUint32(16, 3, true);
    view.setUint32(20, 24000, true);
    bytes.set([1, 2, 3], 24);

    expect(parseVoiceRealtimeAssistantAudioBinaryChunk(buffer)).toEqual({
      utteranceSeq: 12,
      sequence: 3,
      sampleRate: 24000,
      isFinal: true,
      pcm16Bytes: new Uint8Array([1, 2, 3]),
    });
  });

  it("merges sequential sample chunks", () => {
    expect(
      Array.from(
        mergeSampleChunks([new Float32Array([1, 2]), new Float32Array([3])]),
      ),
    ).toEqual([1, 2, 3]);
  });

  it("models voice websocket ownership as process-local and non-resumable", () => {
    const event = {
      type: "session_ready",
      protocol: "voice_realtime_v1",
      session_id: "transport-session",
      owner_instance_id: "process-123",
      connection_epoch: 0,
      resumable: false,
      resume_window_ms: 0,
    } satisfies VoiceRealtimeServerEvent;

    expect(event.owner_instance_id).toBe("process-123");
    expect(event.resumable).toBe(false);
    expect(event.resume_window_ms).toBe(0);
  });

  it("serializes the opt-in v2 session negotiation with a stable shape", () => {
    expect(
      JSON.stringify(buildVoiceRealtimeV2SessionStart("Be concise.")),
    ).toBe(
      '{"type":"session_start","protocol":"voice_realtime","version":2,"system_prompt":"Be concise."}',
    );
  });

  it("recognizes and normalizes the typed voice SessionReady golden envelope", () => {
    const ready = voiceV2Envelope(0, {
      type: "session_ready",
      data: {
        accepted_version: 2,
        owner_instance_id: "process-42",
        resumable: false,
        resume_window_ms: 0,
      },
    });

    expect(isVoiceRealtimeV2ServerEnvelope(ready)).toBe(true);
    expect(ready).toEqual(voiceV2SessionReadyFixture);
    expect(normalizeVoiceRealtimeV2Event(ready)).toEqual({
      type: "session_ready",
      protocol: "voice_realtime",
      session_id: "voice-session-1",
      owner_instance_id: "process-42",
      connection_epoch: 0,
      resumable: false,
      resume_window_ms: 0,
    });
  });

  it("enforces contiguous typed voice sequences and explicit cutoff epochs", () => {
    const first = voiceV2Envelope(0, { type: "session_started" });
    const next = voiceV2Envelope(1, { type: "session_started" });
    const cutoff = {
      ...voiceV2Envelope(0, { type: "session_started" }),
      event_id: 3,
      connection_epoch: 1,
    };

    expect(isValidVoiceRealtimeV2Successor(null, first)).toBe(true);
    expect(isValidVoiceRealtimeV2Successor(first, next)).toBe(true);
    expect(isValidVoiceRealtimeV2Successor(next, cutoff)).toBe(true);
    expect(isValidVoiceRealtimeV2Successor(next, { ...next, event_id: 3 })).toBe(
      false,
    );
  });

  it("rejects unknown and malformed voice server events", () => {
    expect(isVoiceRealtimeServerEvent({ type: "made_up" })).toBe(false);
    expect(
      isVoiceRealtimeServerEvent({
        type: "session_ready",
        protocol: "voice_realtime_v1",
      }),
    ).toBe(false);
    expect(
      isVoiceRealtimeServerEvent({
        type: "assistant_text_snapshot",
        utterance_id: "turn-1",
        utterance_seq: 1,
        text: "Hello",
      }),
    ).toBe(true);
  });

  it("cuts off scheduled playback immediately for barge-in and interruption", () => {
    expect(
      shouldStopVoiceRealtimePlayback(
        {
          type: "user_speech_start",
          utterance_id: "next",
          utterance_seq: 8,
        },
        7,
      ),
    ).toBe(true);
    expect(
      shouldStopVoiceRealtimePlayback(
        {
          type: "turn_interrupted",
          utterance_id: "active",
          utterance_seq: 7,
          reason: "barge_in",
        },
        7,
      ),
    ).toBe(true);
    expect(
      shouldStopVoiceRealtimePlayback(
        {
          type: "turn_done",
          utterance_id: "other",
          utterance_seq: 6,
          status: "interrupted",
        },
        7,
      ),
    ).toBe(false);
  });
});

function voiceV2Envelope(
  sequence: number,
  payload: Pick<VoiceRealtimeV2ServerEnvelope, "type"> &
    Partial<VoiceRealtimeV2ServerEnvelope>,
): VoiceRealtimeV2ServerEnvelope {
  return {
    protocol: "voice_realtime",
    version: 2,
    event_id: sequence + 1,
    sequence,
    session_id: "voice-session-1",
    connection_epoch: 0,
    timestamp_ms: 1_725_000_000_123,
    ...payload,
  } as VoiceRealtimeV2ServerEnvelope;
}
