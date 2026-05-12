import { afterEach, describe, expect, it, vi } from "vitest";

import { AudioApiClient, type DiarizationRecord } from "@/shared/api/audio";
import { ApiHttpClient } from "@/shared/api/http";

const updatedRecord = {
  id: "diar-1",
  created_at: 1,
  model_id: "diar_streaming_sortformer_4spk-v2.1",
  asr_model_id: "Parakeet-TDT-0.6B-v3",
  aligner_model_id: "Qwen3-ForcedAligner-0.6B",
  llm_model_id: null,
  processing_status: "ready" as const,
  processing_error: null,
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
  summary_status: "ready",
  summary_model_id: "Qwen3.5-4B",
  summary_text: "Speaker 00 greets and Speaker 01 responds.",
  summary_error: null,
  summary_updated_at: 1,
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
      "http://localhost/v1/transcriptions/jobs/diar-1?job_kind=diarization",
      expect.objectContaining({ method: "PATCH" }),
    );
    expect(fetchMock).toHaveBeenNthCalledWith(
      2,
      "http://localhost/v1/transcriptions/jobs/diar-1?job_kind=diarization",
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
      "http://localhost/v1/transcriptions/jobs?job_kind=transcription",
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
          model_id: "Parakeet-TDT-0.6B-v3",
          aligner_model_id: "Qwen3-ForcedAligner-0.6B",
          language: "English",
          processing_status: "pending",
          processing_error: null,
          duration_secs: null,
          processing_time_ms: 0,
          rtf: null,
          audio_mime_type: "audio/wav",
          audio_filename: "clip.wav",
          transcription: "",
          segments: [],
          words: [],
          summary_status: "not_requested",
        }),
        {
          status: 202,
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
      model_id: "Parakeet-TDT-0.6B-v3",
      aligner_model_id: "Qwen3-ForcedAligner-0.6B",
      language: "English",
      include_timestamps: true,
    });

    expect(fetchMock).toHaveBeenCalledWith(
      "http://localhost/v1/transcriptions/jobs?job_kind=transcription",
      expect.objectContaining({
        method: "POST",
        body: JSON.stringify({
          audio_base64: "UklGRg==",
          model: "Parakeet-TDT-0.6B-v3",
          aligner_model: "Qwen3-ForcedAligner-0.6B",
          language: "English",
          include_timestamps: true,
          stream: false,
        }),
      }),
    );
  });

  it("posts transcription summary regenerations to the canonical summary route", async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      new Response(
        JSON.stringify({
          id: "txr-1",
          created_at: 1,
          model_id: "Parakeet-TDT-0.6B-v3",
          aligner_model_id: null,
          language: "English",
          processing_status: "ready",
          processing_error: null,
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
    await client.regenerateTranscriptionSummary("txr-1");

    expect(fetchMock).toHaveBeenCalledWith(
      "http://localhost/v1/transcriptions/jobs/txr-1/summary/regenerate?job_kind=transcription",
      expect.objectContaining({
        method: "POST",
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
          saved_voice_id: "voice-1",
          speed: 1.1,
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
      saved_voice_id: "voice-1",
      speed: 1.1,
    });

    expect(fetchMock).toHaveBeenCalledWith(
      "http://localhost/v1/text-to-speech",
      expect.objectContaining({
        method: "POST",
        body: JSON.stringify({
          model_id: "model",
          text: "Hello",
          speaker: undefined,
          language: undefined,
          voice_description: undefined,
          reference_audio: undefined,
          reference_text: undefined,
          saved_voice_id: "voice-1",
          temperature: undefined,
          speed: 1.1,
          max_tokens: undefined,
          max_output_tokens: undefined,
          top_k: undefined,
          stream: false,
        }),
      }),
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
      "http://localhost/v1/transcriptions/jobs/diar-1/reruns?job_kind=diarization",
      expect.objectContaining({ method: "POST" }),
    );
  });

  it("posts diarization cancels to the canonical cancel route", async () => {
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
    const result = await client.cancelDiarizationRecord("diar-1");

    expect(result).toEqual(updatedRecord);
    expect(fetchMock).toHaveBeenCalledWith(
      "http://localhost/v1/transcriptions/jobs/diar-1/cancel?job_kind=diarization",
      expect.objectContaining({ method: "POST" }),
    );
  });

  it("posts diarization summary regenerations to the canonical summary route", async () => {
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
    const result = await client.regenerateDiarizationSummary("diar-1");

    expect(result).toEqual(updatedRecord);
    expect(fetchMock).toHaveBeenCalledWith(
      "http://localhost/v1/transcriptions/jobs/diar-1/summary/regenerate?job_kind=diarization",
      expect.objectContaining({ method: "POST" }),
    );
  });

  it("lists unified speech text jobs with pagination and kind filters", async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      new Response(JSON.stringify({ records: [], pagination: {} }), {
        status: 200,
        headers: {
          "Content-Type": "application/json",
        },
      }),
    );

    vi.stubGlobal("fetch", fetchMock);

    const client = new AudioApiClient(new ApiHttpClient("http://localhost/v1"));
    const page = await client.listSpeechTextJobPage({
      limit: 10,
      cursor: "cursor-1",
      job_kind: "diarization",
    });

    expect(page.items).toEqual([]);
    expect(fetchMock).toHaveBeenCalledWith(
      "http://localhost/v1/transcriptions/jobs?limit=10&cursor=cursor-1&job_kind=diarization",
      expect.objectContaining({
        headers: expect.objectContaining({
          "Content-Type": "application/json",
        }),
      }),
    );
  });

  it("creates unified speech text diarization jobs", async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      new Response(
        JSON.stringify({
          ...updatedRecord,
          kind: "diarization",
        }),
        {
          status: 202,
          headers: {
            "Content-Type": "application/json",
          },
        },
      ),
    );

    vi.stubGlobal("fetch", fetchMock);

    const client = new AudioApiClient(new ApiHttpClient("http://localhost/v1"));
    await client.createSpeechTextJob({
      kind: "diarization",
      audio_base64: "UklGRg==",
    });

    expect(fetchMock).toHaveBeenCalledWith(
      "http://localhost/v1/transcriptions/jobs?job_kind=diarization",
      expect.objectContaining({
        method: "POST",
        body: JSON.stringify({
          audio_base64: "UklGRg==",
          model: undefined,
          asr_model: undefined,
          aligner_model: undefined,
          llm_model: undefined,
          enable_llm_refinement: true,
          min_speakers: undefined,
          max_speakers: undefined,
          min_speech_duration_ms: undefined,
          min_silence_duration_ms: undefined,
        }),
      }),
    );
  });

  it("builds unified speech text audio urls with kind filters", () => {
    const client = new AudioApiClient(new ApiHttpClient("http://localhost/v1"));

    expect(client.speechTextJobAudioUrl("diar-1", { job_kind: "diarization" })).toBe(
      "http://localhost/v1/transcriptions/jobs/diar-1/audio?job_kind=diarization",
    );
  });

  it("builds canonical voice clone audio urls", () => {
    const client = new AudioApiClient(new ApiHttpClient("http://localhost/v1"));

    expect(client.voiceCloningRecordAudioUrl("clone-1")).toBe(
      "http://localhost/v1/voice-clones/clone-1/audio",
    );
  });

  it("posts Studio projects to the canonical projects collection", async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      new Response(
        JSON.stringify({
          id: "ttsp-1",
          created_at: 1,
          updated_at: 1,
          name: "Narration",
          source_filename: "script.txt",
          source_text: "Hello world.",
          model_id: "Qwen3-TTS-0.6B-CustomVoice",
          voice_mode: "built_in",
          speaker: "Vivian",
          saved_voice_id: null,
          speed: 1,
          segments: [],
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
    await client.createStudioProject({
      name: "Narration",
      source_filename: "script.txt",
      source_text: "Hello world.",
      model_id: "Qwen3-TTS-0.6B-CustomVoice",
      voice_mode: "built_in",
      speaker: "Vivian",
      speed: 1,
    });

    expect(fetchMock).toHaveBeenCalledWith(
      "http://localhost/v1/studio/projects",
      expect.objectContaining({
        method: "POST",
        body: JSON.stringify({
          name: "Narration",
          source_filename: "script.txt",
          source_text: "Hello world.",
          model_id: "Qwen3-TTS-0.6B-CustomVoice",
          voice_mode: "built_in",
          speaker: "Vivian",
          saved_voice_id: undefined,
          speed: 1,
        }),
      }),
    );
  });

  it("posts Studio project segment splits to the canonical split route", async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      new Response(
        JSON.stringify({
          id: "ttsp-1",
          created_at: 1,
          updated_at: 1,
          name: "Narration",
          source_filename: "script.txt",
          source_text: "Hello world.\n\nAnother sentence.",
          model_id: "Qwen3-TTS-0.6B-CustomVoice",
          voice_mode: "built_in",
          speaker: "Vivian",
          saved_voice_id: null,
          speed: 1,
          segments: [],
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
    await client.splitStudioProjectSegment("ttsp-1", "ttss-1", {
      before_text: "Hello world.",
      after_text: "Another sentence.",
    });

    expect(fetchMock).toHaveBeenCalledWith(
      "http://localhost/v1/studio/projects/ttsp-1/segments/ttss-1/split",
      expect.objectContaining({
        method: "POST",
        body: JSON.stringify({
          before_text: "Hello world.",
          after_text: "Another sentence.",
        }),
      }),
    );
  });

  it("posts Studio project segment inserts to the canonical segment collection", async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      new Response(
        JSON.stringify({
          id: "ttsp-1",
          segments: [],
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
    await client.createStudioProjectSegment("ttsp-1", {
      text: "A brand new segment.",
      after_segment_id: "ttss-1",
    });

    expect(fetchMock).toHaveBeenCalledWith(
      "http://localhost/v1/studio/projects/ttsp-1/segments",
      expect.objectContaining({
        method: "POST",
        body: JSON.stringify({
          text: "A brand new segment.",
          after_segment_id: "ttss-1",
        }),
      }),
    );
  });

  it("deletes Studio project segments through the canonical segment route", async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      new Response(
        JSON.stringify({
          id: "ttsp-1",
          created_at: 1,
          updated_at: 1,
          name: "Narration",
          source_filename: "script.txt",
          source_text: "Hello world.",
          model_id: "Qwen3-TTS-0.6B-CustomVoice",
          voice_mode: "built_in",
          speaker: "Vivian",
          saved_voice_id: null,
          speed: 1,
          segments: [],
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
    await client.deleteStudioProjectSegment("ttsp-1", "ttss-1");

    expect(fetchMock).toHaveBeenCalledWith(
      "http://localhost/v1/studio/projects/ttsp-1/segments/ttss-1",
      expect.objectContaining({
        method: "DELETE",
      }),
    );
  });

  it("builds canonical Studio project export urls", () => {
    const client = new AudioApiClient(new ApiHttpClient("http://localhost/v1"));

    expect(client.studioProjectAudioUrl("ttsp-1")).toBe(
      "http://localhost/v1/studio/projects/ttsp-1/audio",
    );
    expect(
      client.studioProjectAudioUrl("ttsp-1", {
        download: true,
        format: "raw_i16",
        segment_ids: ["a", "b"],
      }),
    ).toBe(
      "http://localhost/v1/studio/projects/ttsp-1/audio?download=true&format=raw_i16&segment_ids=a%2Cb",
    );
  });

  it("posts segment reorder operations to the canonical studio route", async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      new Response(
        JSON.stringify({
          id: "ttsp-1",
          segments: [],
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
    await client.reorderStudioProjectSegments("ttsp-1", {
      ordered_segment_ids: ["ttss-2", "ttss-1"],
    });

    expect(fetchMock).toHaveBeenCalledWith(
      "http://localhost/v1/studio/projects/ttsp-1/segments/reorder",
      expect.objectContaining({
        method: "PATCH",
        body: JSON.stringify({
          ordered_segment_ids: ["ttss-2", "ttss-1"],
        }),
      }),
    );
  });

  it("posts bulk segment deletion to the canonical studio route", async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      new Response(
        JSON.stringify({
          id: "ttsp-1",
          segments: [],
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
    await client.bulkDeleteStudioProjectSegments("ttsp-1", {
      segment_ids: ["ttss-1", "ttss-2"],
    });

    expect(fetchMock).toHaveBeenCalledWith(
      "http://localhost/v1/studio/projects/ttsp-1/segments/bulk-delete",
      expect.objectContaining({
        method: "POST",
        body: JSON.stringify({
          segment_ids: ["ttss-1", "ttss-2"],
        }),
      }),
    );
  });

  it("posts render-job updates to the canonical studio route", async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      new Response(
        JSON.stringify({
          id: "ttsj-1",
          project_id: "ttsp-1",
          created_at: 1,
          updated_at: 1,
          status: "queued",
          error_message: null,
          queued_segment_ids: [],
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
    await client.updateStudioProjectRenderJob("ttsp-1", "ttsj-1", {
      status: "failed",
      error_message: "boom",
    });

    expect(fetchMock).toHaveBeenCalledWith(
      "http://localhost/v1/studio/projects/ttsp-1/render-jobs/ttsj-1",
      expect.objectContaining({
        method: "PATCH",
        body: JSON.stringify({
          status: "failed",
          error_message: "boom",
        }),
      }),
    );
  });
});
