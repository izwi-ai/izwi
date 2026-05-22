import { describe, expect, it } from "vitest";
import {
  buildVoiceRealtimeWebSocketUrl,
  encodeVoiceRealtimeClientPcm16Frame,
  formatModelVariantLabel,
  isAsrVariant,
  isUnifiedAudioChatVariant,
  makeTranscriptEntryId,
  mergeSampleChunks,
  parseFinalAnswer,
  parseVoiceRealtimeAssistantAudioBinaryChunk,
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
});
