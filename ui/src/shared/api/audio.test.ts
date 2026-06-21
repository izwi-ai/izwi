import { afterEach, describe, expect, it, vi } from "vitest";

import {
  AudioApiClient,
  type DiarizationRecord,
  type TranscriptionRecord,
} from "@/shared/api/audio";
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

const createdTranscriptionRecord = {
  id: "txr-xhr-1",
  created_at: 1,
  model_id: "Parakeet-TDT-0.6B-v3",
  aligner_model_id: null,
  language: "English",
  processing_status: "pending" as const,
  processing_error: null,
  duration_secs: null,
  processing_time_ms: 0,
  rtf: null,
  audio_mime_type: "audio/wav",
  audio_filename: "clip.wav",
  transcription: "",
  segments: [],
  words: [],
  summary_status: "not_requested" as const,
  summary_model_id: null,
  summary_text: null,
  summary_error: null,
  summary_updated_at: null,
} satisfies TranscriptionRecord;

function sseResponse(events: Array<Record<string, unknown> | string>): Response {
  const encoder = new TextEncoder();
  return new Response(
    new ReadableStream({
      start(controller) {
        for (const event of events) {
          const payload =
            typeof event === "string" ? event : JSON.stringify(event);
          controller.enqueue(encoder.encode(`data: ${payload}\n\n`));
        }
        controller.close();
      },
    }),
    {
      status: 200,
      headers: {
        "Content-Type": "text/event-stream",
      },
    },
  );
}

type XhrProgressHandler = ((event: ProgressEvent) => void) | null;

class MockXMLHttpRequest {
  static instances: MockXMLHttpRequest[] = [];

  upload: { onprogress: XhrProgressHandler } = { onprogress: null };
  onload: XhrProgressHandler = null;
  onerror: XhrProgressHandler = null;
  onabort: XhrProgressHandler = null;
  onprogress: XhrProgressHandler = null;
  method = "";
  url = "";
  headers = new Map<string, string>();
  body: XMLHttpRequestBodyInit | null = null;
  status = 0;
  responseText = "";
  responseType: XMLHttpRequestResponseType = "";

  constructor() {
    MockXMLHttpRequest.instances.push(this);
  }

  open(method: string, url: string) {
    this.method = method;
    this.url = url;
  }

  setRequestHeader(name: string, value: string) {
    this.headers.set(name.toLowerCase(), value);
  }

  send(body: XMLHttpRequestBodyInit | null = null) {
    this.body = body;
  }

  abort() {
    this.onabort?.({} as ProgressEvent);
  }

  emitUploadProgress(
    loaded: number,
    total: number,
    lengthComputable = true,
  ) {
    this.upload.onprogress?.({
      lengthComputable,
      loaded,
      total,
    } as ProgressEvent);
  }

  appendResponse(chunk: string, status = 202) {
    this.status = status;
    this.responseText += chunk;
    this.onprogress?.({} as ProgressEvent);
  }

  complete(status = 202) {
    this.status = status;
    this.onload?.({} as ProgressEvent);
  }

  respond(status: number, responseText: string) {
    this.status = status;
    this.responseText = responseText;
    this.onload?.({} as ProgressEvent);
  }
}

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
      "http://localhost/v1/speech-to-text/jobs/diar-1?job_kind=diarization",
      expect.objectContaining({ method: "PATCH" }),
    );
    expect(fetchMock).toHaveBeenNthCalledWith(
      2,
      "http://localhost/v1/speech-to-text/jobs/diar-1?job_kind=diarization",
      expect.objectContaining({ method: "PUT" }),
    );
  });

  it("parses direct TTS stream audio event names", async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      sseResponse([
        {
          event: "audio.started",
          request_id: "req-1",
          sample_rate: 24000,
          audio_format: "pcm_i16",
        },
        {
          event: "audio.chunk",
          request_id: "req-1",
          sequence: 1,
          audio_base64: "AAAA",
          sample_count: 2,
          is_final: false,
        },
        {
          event: "audio.done",
          request_id: "req-1",
          tokens_generated: 12,
          generation_time_ms: 100,
          audio_duration_secs: 1,
          rtf: 0.1,
        },
      ]),
    );
    vi.stubGlobal("fetch", fetchMock);

    const client = new AudioApiClient(new ApiHttpClient("http://localhost/v1"));
    const onStart = vi.fn();
    const onChunk = vi.fn();
    const onFinal = vi.fn();
    await new Promise<void>((resolve) => {
      client.generateTTSStream(
        { model_id: "Kokoro-82M", text: "Hello" },
        { onStart, onChunk, onFinal, onDone: resolve },
      );
    });

    expect(onStart).toHaveBeenCalledWith({
      requestId: "req-1",
      sampleRate: 24000,
      audioFormat: "pcm_i16",
    });
    expect(onChunk).toHaveBeenCalledWith({
      requestId: "req-1",
      sequence: 1,
      audioBase64: "AAAA",
      sampleCount: 2,
      isFinal: false,
    });
    expect(onFinal).toHaveBeenCalledWith({
      generation_time_ms: 100,
      audio_duration_secs: 1,
      rtf: 0.1,
      tokens_generated: 12,
    });
  });

  it("parses direct ASR stream transcript type names", async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      sseResponse([
        { type: "transcript.text.delta", delta: "Hel" },
        { type: "transcript.text.delta", delta: "lo" },
        {
          type: "transcript.text.done",
          text: "Hello",
          language: "English",
          audio_duration_secs: 1.2,
        },
      ]),
    );
    vi.stubGlobal("fetch", fetchMock);

    const client = new AudioApiClient(new ApiHttpClient("http://localhost/v1"));
    const onDelta = vi.fn();
    const onPartial = vi.fn();
    const onFinal = vi.fn();
    await new Promise<void>((resolve) => {
      client.asrTranscribeStream(
        { audio_base64: "AAAA", model_id: "Parakeet-TDT-0.6B-v3" },
        { onDelta, onPartial, onFinal, onDone: resolve },
      );
    });

    expect(onDelta).toHaveBeenNthCalledWith(1, "Hel");
    expect(onDelta).toHaveBeenNthCalledWith(2, "lo");
    expect(onPartial).toHaveBeenNthCalledWith(1, "Hel");
    expect(onPartial).toHaveBeenNthCalledWith(2, "Hello");
    expect(onFinal).toHaveBeenCalledWith("Hello", "English", 1.2);
  });

  it("posts direct ASR prompt and timestamp controls", async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      new Response(
        JSON.stringify({
          text: "hello world",
          language: "en",
          duration: 1.2,
          processing_time_ms: 100,
          rtf: 0.1,
          words: [{ word: "hello", start: 0, end: 0.5 }],
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
    await client.asrTranscribe({
      audio_base64: "AAAA",
      model_id: "Granite-Speech-4.1-2B-Plus",
      language: "en",
      prompt: "keywords: izwi, granite",
      max_tokens: 64,
      timestamp_granularities: ["word"],
    });

    expect(fetchMock).toHaveBeenCalledWith(
      "http://localhost/v1/audio/transcriptions",
      expect.objectContaining({
        method: "POST",
        body: JSON.stringify({
          audio_base64: "AAAA",
          model: "Granite-Speech-4.1-2B-Plus",
          language: "en",
          prompt: "keywords: izwi, granite",
          max_tokens: 64,
          timestamp_granularities: ["word"],
          response_format: "verbose_json",
          stream: false,
        }),
      }),
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
      "http://localhost/v1/speech-to-text/jobs?job_kind=transcription",
      expect.objectContaining({
        headers: expect.objectContaining({
          "Content-Type": "application/json",
        }),
      }),
    );
  });

  it("posts transcription records with optional timestamp and summary settings", async () => {
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
      generate_summary: true,
    });

    expect(fetchMock).toHaveBeenCalledWith(
      "http://localhost/v1/speech-to-text/jobs?job_kind=transcription",
      expect.objectContaining({
        method: "POST",
        body: JSON.stringify({
          audio_base64: "UklGRg==",
          model: "Parakeet-TDT-0.6B-v3",
          aligner_model: "Qwen3-ForcedAligner-0.6B",
          language: "English",
          include_timestamps: true,
          generate_summary: true,
          stream: false,
        }),
      }),
    );
  });

  it("defaults transcription summary generation to false in JSON requests", async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      new Response(
        JSON.stringify({
          ...createdTranscriptionRecord,
          processing_status: "pending",
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
    });

    expect(fetchMock).toHaveBeenCalledWith(
      "http://localhost/v1/speech-to-text/jobs?job_kind=transcription",
      expect.objectContaining({
        method: "POST",
        body: JSON.stringify({
          audio_base64: "UklGRg==",
          model: "Parakeet-TDT-0.6B-v3",
          aligner_model: undefined,
          language: undefined,
          include_timestamps: false,
          generate_summary: false,
          stream: false,
        }),
      }),
    );
  });

  it("creates speaker-attributed ASR records through the SAA job kind", async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      new Response(
        JSON.stringify({
          ...createdTranscriptionRecord,
          kind: "speaker_attributed_asr",
          transcription_mode: "speaker_attributed_asr",
          model_id: "Granite-Speech-4.1-2B-Plus",
          speaker_attributed_text: null,
          speaker_turns: [],
          saa_status: "not_requested",
          saa_warnings: [],
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
    await client.createSpeakerAttributedAsrRecord({
      audio_base64: "UklGRg==",
      model_id: "Granite-Speech-4.1-2B-Plus",
      language: "English",
      generate_summary: true,
      min_speakers: 2,
    });

    expect(fetchMock).toHaveBeenCalledWith(
      "http://localhost/v1/speech-to-text/jobs?job_kind=speaker_attributed_asr",
      expect.objectContaining({
        method: "POST",
        body: JSON.stringify({
          audio_base64: "UklGRg==",
          model: "Granite-Speech-4.1-2B-Plus",
          language: "English",
          generate_summary: true,
          min_speakers: 2,
          max_speakers: undefined,
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
      "http://localhost/v1/speech-to-text/jobs/txr-1/summary/regenerate?job_kind=transcription",
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
      "http://localhost/v1/speech-to-text/jobs/diar-1/reruns?job_kind=diarization",
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
      "http://localhost/v1/speech-to-text/jobs/diar-1/cancel?job_kind=diarization",
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
      "http://localhost/v1/speech-to-text/jobs/diar-1/summary/regenerate?job_kind=diarization",
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
      "http://localhost/v1/speech-to-text/jobs?limit=10&cursor=cursor-1&job_kind=diarization",
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
      "http://localhost/v1/speech-to-text/jobs?job_kind=diarization",
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

  it("uses XHR for progress-enabled diarization file uploads", async () => {
    MockXMLHttpRequest.instances = [];
    vi.stubGlobal("XMLHttpRequest", MockXMLHttpRequest);

    const client = new AudioApiClient(new ApiHttpClient("http://localhost/v1"));
    const onUploadProgress = vi.fn();
    const file = new File(["0123456789"], "meeting.mp3", {
      type: "audio/mpeg",
    });

    const promise = client.createDiarizationRecord(
      {
        audio_file: file,
        audio_filename: file.name,
        model_id: "diar_streaming_sortformer_4spk-v2.1",
        asr_model_id: "Whisper-Large-v3-Turbo",
        aligner_model_id: "Qwen3-ForcedAligner-0.6B",
        min_speakers: 1,
        max_speakers: 4,
      },
      { onUploadProgress },
    );

    const xhr = MockXMLHttpRequest.instances[0]!;
    expect(xhr.method).toBe("POST");
    expect(xhr.url).toBe(
      "http://localhost/v1/speech-to-text/jobs?job_kind=diarization",
    );
    expect(xhr.headers.has("content-type")).toBe(false);
    expect(xhr.body).toBeInstanceOf(FormData);

    const form = xhr.body as FormData;
    const uploadedFile = form.get("file") as File;
    expect(uploadedFile.name).toBe("meeting.mp3");
    expect(uploadedFile.type).toBe("audio/mpeg");
    await expect(uploadedFile.text()).resolves.toBe("0123456789");
    expect(form.get("model")).toBe("diar_streaming_sortformer_4spk-v2.1");
    expect(form.get("asr_model")).toBe("Whisper-Large-v3-Turbo");
    expect(form.get("aligner_model")).toBe("Qwen3-ForcedAligner-0.6B");
    expect(form.get("enable_llm_refinement")).toBe("true");

    xhr.emitUploadProgress(5, 10);
    xhr.respond(
      202,
      JSON.stringify({
        ...updatedRecord,
        kind: "diarization",
      }),
    );

    await expect(promise).resolves.toEqual(updatedRecord);
    expect(onUploadProgress).toHaveBeenCalledWith({
      loadedBytes: 5,
      totalBytes: 10,
      percent: 50,
      lengthComputable: true,
    });
  });

  it("preserves JSON error messages from progress-enabled uploads", async () => {
    MockXMLHttpRequest.instances = [];
    vi.stubGlobal("XMLHttpRequest", MockXMLHttpRequest);

    const client = new AudioApiClient(new ApiHttpClient("http://localhost/v1"));
    const promise = client.createDiarizationRecord(
      {
        audio_file: new File(["audio"], "large.mp3", {
          type: "audio/mpeg",
        }),
        audio_filename: "large.mp3",
      },
      { onUploadProgress: vi.fn() },
    );

    MockXMLHttpRequest.instances[0]!.respond(
      413,
      JSON.stringify({
        error: {
          message: "Uploaded audio is too large for this server.",
        },
      }),
    );

    await expect(promise).rejects.toThrow(
      "Uploaded audio is too large for this server.",
    );
  });

  it("rejects oversized first-party audio uploads before creating requests", async () => {
    MockXMLHttpRequest.instances = [];
    vi.stubGlobal("XMLHttpRequest", MockXMLHttpRequest);

    const client = new AudioApiClient(new ApiHttpClient("http://localhost/v1"));
    const oversizedAudio = {
      size: 64 * 1024 * 1024 + 1,
    } as Blob;

    await expect(
      client.createDiarizationRecord({
        audio_file: oversizedAudio,
        audio_filename: "too-large.wav",
      }),
    ).rejects.toThrow("Uploaded audio is too large for this server.");

    const callbacks = {
      onUploadProgress: vi.fn(),
      onError: vi.fn(),
      onDone: vi.fn(),
    };
    expect(() =>
      client.createTranscriptionRecordStream(
        {
          audio_file: oversizedAudio,
          audio_filename: "too-large.wav",
        },
        callbacks,
      ),
    ).toThrow("Uploaded audio is too large for this server.");
    expect(MockXMLHttpRequest.instances).toHaveLength(0);
  });

  it("aborts progress-enabled upload requests", async () => {
    MockXMLHttpRequest.instances = [];
    vi.stubGlobal("XMLHttpRequest", MockXMLHttpRequest);

    const client = new AudioApiClient(new ApiHttpClient("http://localhost/v1"));
    const abortController = new AbortController();
    const promise = client.createDiarizationRecord(
      {
        audio_file: new File(["audio"], "meeting.wav", {
          type: "audio/wav",
        }),
        audio_filename: "meeting.wav",
      },
      {
        signal: abortController.signal,
        onUploadProgress: vi.fn(),
      },
    );

    abortController.abort();

    await expect(promise).rejects.toMatchObject({ name: "AbortError" });
  });

  it("streams transcription creation over XHR when upload progress is requested", async () => {
    MockXMLHttpRequest.instances = [];
    vi.stubGlobal("XMLHttpRequest", MockXMLHttpRequest);

    const client = new AudioApiClient(new ApiHttpClient("http://localhost/v1"));
    const onUploadProgress = vi.fn();
    const onCreated = vi.fn();
    const onDelta = vi.fn();
    const onProgress = vi.fn();
    const onDone = vi.fn();

    client.createTranscriptionRecordStream(
      {
        audio_file: new File(["audio"], "stream.wav", {
          type: "audio/wav",
        }),
        audio_filename: "stream.wav",
        model_id: "Parakeet-TDT-0.6B-v3",
        generate_summary: true,
      },
      {
        onUploadProgress,
        onCreated,
        onDelta,
        onProgress,
        onDone,
      },
    );

    const xhr = MockXMLHttpRequest.instances[0]!;
    expect(xhr.url).toBe(
      "http://localhost/v1/speech-to-text/jobs?job_kind=transcription",
    );
    expect(xhr.body).toBeInstanceOf(FormData);
    expect((xhr.body as FormData).get("generate_summary")).toBe("true");
    xhr.emitUploadProgress(4, 8);
    xhr.appendResponse(
      `data: ${JSON.stringify({
        event: "created",
        record: createdTranscriptionRecord,
      })}\n\n`,
    );
    xhr.appendResponse(
      `data: ${JSON.stringify({ event: "delta", delta: "Hello" })}\n\n`,
    );
    xhr.appendResponse(
      `data: ${JSON.stringify({
        event: "progress",
        progress: {
          phase: "chunk_finished",
          current_chunk: 1,
          total_chunks: 2,
          processed_audio_secs: 5,
          total_audio_secs: 10,
          percent: 50,
        },
      })}\n\n`,
    );
    xhr.appendResponse(`data: ${JSON.stringify({ event: "done" })}\n\n`);
    xhr.complete();

    expect(onUploadProgress).toHaveBeenCalledWith({
      loadedBytes: 4,
      totalBytes: 8,
      percent: 50,
      lengthComputable: true,
    });
    expect(onCreated).toHaveBeenCalledWith(createdTranscriptionRecord);
    expect(onDelta).toHaveBeenCalledWith("Hello");
    expect(onProgress).toHaveBeenCalledWith({
      phase: "chunk_finished",
      current_chunk: 1,
      total_chunks: 2,
      processed_audio_secs: 5,
      total_audio_secs: 10,
      percent: 50,
    });
    expect(onDone).toHaveBeenCalledTimes(1);
  });

  it("builds unified speech text audio urls with kind filters", () => {
    const client = new AudioApiClient(new ApiHttpClient("http://localhost/v1"));

    expect(client.speechTextJobAudioUrl("diar-1", { job_kind: "diarization" })).toBe(
      "http://localhost/v1/speech-to-text/jobs/diar-1/audio?job_kind=diarization",
    );
    expect(
      client.speakerAttributedAsrRecordAudioUrl("saa-1"),
    ).toBe(
      "http://localhost/v1/speech-to-text/jobs/saa-1/audio?job_kind=speaker_attributed_asr",
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
