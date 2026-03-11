import { afterEach, describe, expect, it, vi } from "vitest";

import { AudioApiClient, type DiarizationRecord } from "@/shared/api/audio";
import { ApiHttpClient } from "@/shared/api/http";

const updatedRecord = {
  id: "diar-1",
  created_at: 1,
  model_id: "diar_streaming_sortformer_4spk-v2.1",
  asr_model_id: "Qwen3-ASR-0.6B",
  aligner_model_id: "Qwen3-ForcedAligner-0.6B",
  llm_model_id: null,
  min_speakers: 1,
  max_speakers: 4,
  min_speech_duration_ms: 240,
  min_silence_duration_ms: 200,
  enable_llm_refinement: false,
  processing_time_ms: 120,
  duration_secs: 6,
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
  utterances: [],
  speaker_name_overrides: {
    SPEAKER_00: "Alice",
  },
  audio_mime_type: "audio/wav",
  audio_filename: "meeting.wav",
} satisfies DiarizationRecord;

describe("AudioApiClient.updateDiarizationRecord", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("retries with PUT when PATCH is rejected", async () => {
    const fetchMock = vi
      .fn()
      .mockResolvedValueOnce(
        new Response(
          JSON.stringify({
            error: { message: "Method Not Allowed" },
          }),
          {
            status: 405,
            headers: {
              "Content-Type": "application/json",
            },
          },
        ),
      )
      .mockResolvedValueOnce(
        new Response(JSON.stringify(updatedRecord), {
          status: 200,
          headers: {
            "Content-Type": "application/json",
          },
        }),
      );

    vi.stubGlobal("fetch", fetchMock);

    const client = new AudioApiClient(new ApiHttpClient("http://localhost/v1"));
    const result = await client.updateDiarizationRecord("diar-1", {
      speaker_name_overrides: {
        SPEAKER_00: "Alice",
      },
    });

    expect(result).toEqual(updatedRecord);
    expect(fetchMock).toHaveBeenCalledTimes(2);
    expect(fetchMock).toHaveBeenNthCalledWith(
      1,
      "http://localhost/v1/diarizations/diar-1",
      expect.objectContaining({ method: "PATCH" }),
    );
    expect(fetchMock).toHaveBeenNthCalledWith(
      2,
      "http://localhost/v1/diarizations/diar-1",
      expect.objectContaining({ method: "PUT" }),
    );
  });

  it("lists transcriptions through the canonical collection route", async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      new Response(JSON.stringify({ records: [] }), {
        status: 200,
        headers: {
          "Content-Type": "application/json",
        },
      }),
    );

    vi.stubGlobal("fetch", fetchMock);

    const client = new AudioApiClient(new ApiHttpClient("http://localhost/v1"));
    const result = await client.listTranscriptionRecords();

    expect(result).toEqual([]);
    expect(fetchMock).toHaveBeenCalledWith(
      "http://localhost/v1/transcriptions",
      expect.objectContaining({
        headers: expect.objectContaining({
          "Content-Type": "application/json",
        }),
      }),
    );
  });

  it("posts transcription records with optional timestamp settings", async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      new Response(
        JSON.stringify({
          id: "txr-1",
          created_at: 1,
          model_id: "Qwen3-ASR-0.6B",
          aligner_model_id: "Qwen3-ForcedAligner-0.6B",
          language: "English",
          duration_secs: 4,
          processing_time_ms: 120,
          rtf: 0.5,
          audio_mime_type: "audio/wav",
          audio_filename: "clip.wav",
          transcription: "Hello there.",
          segments: [],
          words: [],
        }),
        {
          status: 200,
          headers: {
            "Content-Type": "application/json",
          },
        },
      ),
    );

    vi.stubGlobal("fetch", fetchMock);

    const client = new AudioApiClient(new ApiHttpClient("http://localhost/v1"));
    await client.createTranscriptionRecord({
      audio_base64: "UklGRg==",
      model_id: "Qwen3-ASR-0.6B",
      aligner_model_id: "Qwen3-ForcedAligner-0.6B",
      language: "English",
      include_timestamps: true,
    });

    expect(fetchMock).toHaveBeenCalledWith(
      "http://localhost/v1/transcriptions",
      expect.objectContaining({
        method: "POST",
        body: JSON.stringify({
          audio_base64: "UklGRg==",
          model: "Qwen3-ASR-0.6B",
          aligner_model: "Qwen3-ForcedAligner-0.6B",
          language: "English",
          include_timestamps: true,
          stream: false,
        }),
      }),
    );
  });

  it("posts text-to-speech history to the canonical generation route", async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      new Response(
        JSON.stringify({
          id: "tts-1",
          created_at: 1,
          route_kind: "text_to_speech",
          model_id: "model",
          speaker: "Narrator",
          language: "en",
          input_text: "Hello",
          voice_description: null,
          reference_text: null,
          generation_time_ms: 10,
          audio_duration_secs: 1,
          rtf: 0.5,
          tokens_generated: 10,
          audio_mime_type: "audio/wav",
          audio_filename: "tts.wav",
        }),
        {
          status: 200,
          headers: {
            "Content-Type": "application/json",
          },
        },
      ),
    );

    vi.stubGlobal("fetch", fetchMock);

    const client = new AudioApiClient(new ApiHttpClient("http://localhost/v1"));
    await client.createTextToSpeechRecord({
      model_id: "model",
      text: "Hello",
    });

    expect(fetchMock).toHaveBeenCalledWith(
      "http://localhost/v1/text-to-speech-generations",
      expect.objectContaining({ method: "POST" }),
    );
  });

  it("posts diarization reruns to the canonical reruns resource", async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      new Response(JSON.stringify(updatedRecord), {
        status: 200,
        headers: {
          "Content-Type": "application/json",
        },
      }),
    );

    vi.stubGlobal("fetch", fetchMock);

    const client = new AudioApiClient(new ApiHttpClient("http://localhost/v1"));
    const result = await client.rerunDiarizationRecord("diar-1", {});

    expect(result).toEqual(updatedRecord);
    expect(fetchMock).toHaveBeenCalledWith(
      "http://localhost/v1/diarizations/diar-1/reruns",
      expect.objectContaining({ method: "POST" }),
    );
  });

  it("builds canonical voice clone audio urls", () => {
    const client = new AudioApiClient(new ApiHttpClient("http://localhost/v1"));

    expect(client.voiceCloningRecordAudioUrl("clone-1")).toBe(
      "http://localhost/v1/voice-clone-generations/clone-1/audio",
    );
  });
});
