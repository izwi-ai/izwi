const tauriServerUrl =
  typeof window !== "undefined"
    ? ((window as { __IZWI_SERVER_URL__?: string }).__IZWI_SERVER_URL__ ?? null)
    : null;
const API_BASE = tauriServerUrl
  ? `${tauriServerUrl.replace(/\/$/, "")}/v1`
  : "/v1";

// ============================================================================
// Types
// ============================================================================

export interface ModelInfo {
  variant: string;
  status:
    | "not_downloaded"
    | "downloading"
    | "downloaded"
    | "loading"
    | "ready"
    | "error";
  local_path: string | null;
  size_bytes: number | null;
  download_progress: number | null;
  error_message: string | null;
}

export interface ModelsResponse {
  models: ModelInfo[];
}

// ============================================================================
// Chat Types
// ============================================================================

export interface ChatMessage {
  role: "system" | "user" | "assistant";
  content: string;
}

export interface ChatCompletionRequest {
  model_id?: string;
  messages: ChatMessage[];
  max_tokens?: number;
}

export interface ChatCompletionResponse {
  model_id: string;
  message: ChatMessage;
  stats: {
    tokens_generated: number;
    generation_time_ms: number;
  };
}

export type ChatStreamEvent =
  | { event: "start"; model_id: string }
  | { event: "delta"; delta: string }
  | {
      event: "done";
      model_id: string;
      message: string;
      stats: {
        tokens_generated: number;
        generation_time_ms: number;
      };
    }
  | { event: "error"; error: string };

export interface ChatStreamCallbacks {
  onStart?: (modelId: string) => void;
  onDelta?: (delta: string) => void;
  onDone?: (
    message: string,
    stats: { tokens_generated: number; generation_time_ms: number },
  ) => void;
  onError?: (error: string) => void;
}

export interface ChatThread {
  id: string;
  title: string;
  model_id: string | null;
  created_at: number;
  updated_at: number;
  last_message_preview: string | null;
  message_count: number;
}

export interface ChatThreadMessageRecord {
  id: string;
  thread_id: string;
  role: "system" | "user" | "assistant";
  content: string;
  content_parts?: ChatThreadContentPart[] | null;
  created_at: number;
  tokens_generated: number | null;
  generation_time_ms: number | null;
}

export interface ChatThreadDetail {
  thread: ChatThread;
  messages: ChatThreadMessageRecord[];
}

export interface ChatThreadCreateRequest {
  title?: string;
  model_id?: string;
}

export interface ChatThreadUpdateRequest {
  title: string;
}

export interface ChatThreadContentPart {
  type:
    | "text"
    | "input_text"
    | "image_url"
    | "input_image"
    | "video"
    | "video_url"
    | "input_video";
  text?: string;
  input_text?: string;
  image_url?: unknown;
  input_image?: unknown;
  image?: unknown;
  video?: unknown;
  video_url?: unknown;
  input_video?: unknown;
}

export interface ChatThreadSendMessageRequest {
  model_id?: string;
  content: string;
  content_parts?: ChatThreadContentPart[];
  max_tokens?: number;
  system_prompt?: string;
  enable_thinking?: boolean;
}

export interface ChatThreadSendMessageResponse {
  thread_id: string;
  model_id: string;
  user_message: ChatThreadMessageRecord;
  assistant_message: ChatThreadMessageRecord;
  stats: {
    tokens_generated: number;
    generation_time_ms: number;
  };
}

type ChatThreadStreamEvent =
  | {
      event: "start";
      thread_id: string;
      model_id: string;
      user_message: ChatThreadMessageRecord;
    }
  | { event: "delta"; delta: string }
  | {
      event: "done";
      thread_id: string;
      model_id: string;
      assistant_message: ChatThreadMessageRecord;
      stats: {
        tokens_generated: number;
        generation_time_ms: number;
      };
    }
  | { event: "error"; error: string };

export interface ChatThreadStreamCallbacks {
  onStart?: (event: {
    threadId: string;
    modelId: string;
    userMessage: ChatThreadMessageRecord;
  }) => void;
  onDelta?: (delta: string) => void;
  onDone?: (event: {
    threadId: string;
    modelId: string;
    assistantMessage: ChatThreadMessageRecord;
    stats: {
      tokens_generated: number;
      generation_time_ms: number;
    };
  }) => void;
  onError?: (error: string) => void;
  onClose?: () => void;
}

// ============================================================================
// Agent API Types (minimal MVP)
// ============================================================================

export type AgentPlanningMode = "off" | "auto" | "on";

export interface AgentSessionCreateRequest {
  agent_id?: string;
  model_id?: string;
  system_prompt?: string;
  planning_mode?: AgentPlanningMode;
  title?: string;
}

export interface AgentSession {
  id: string;
  agent_id: string;
  thread_id: string;
  model_id: string;
  planning_mode: AgentPlanningMode;
  created_at: number;
  updated_at: number;
}

export type AgentEvent =
  | { event: "turn_started"; session_id: string; thread_id: string }
  | { event: "plan_created"; steps: string[] }
  | { event: "tool_call_started"; name: string }
  | { event: "tool_call_completed"; name: string; output: string }
  | { event: "assistant_message"; content: string }
  | { event: "turn_completed"; session_id: string; thread_id: string };

export interface AgentToolCallRecord {
  name: string;
  input_summary: string;
  output: string;
}

export interface AgentTurnRequest {
  input: string;
  model_id?: string;
  max_output_tokens?: number;
}

export interface AgentTurnResponse {
  session_id: string;
  thread_id: string;
  model_id: string;
  assistant_text: string;
  plan: { mode: AgentPlanningMode; steps: string[] } | null;
  tool_calls: AgentToolCallRecord[];
  events: AgentEvent[];
}

// ============================================================================
// Responses API Types
// ============================================================================

export interface ResponsesCreateRequest {
  model_id?: string;
  input: string;
  instructions?: string;
  max_output_tokens?: number;
  metadata?: Record<string, unknown>;
  store?: boolean;
}

export interface ResponsesObject {
  id: string;
  status: string;
  model: string;
  output_text: string;
  usage: {
    input_tokens: number;
    output_tokens: number;
    total_tokens: number;
  };
}

export interface ResponsesStreamCallbacks {
  onCreated?: (response: ResponsesObject) => void;
  onDelta?: (delta: string) => void;
  onCompleted?: (response: ResponsesObject) => void;
  onError?: (error: string) => void;
  onDone?: () => void;
}

// ============================================================================
// Unified TTS Types
// ============================================================================

export interface TTSRequest {
  text: string;
  model_id: string;
  language?: string;
  speaker?: string;
  voice_description?: string;
  reference_audio?: string;
  reference_text?: string;
  max_tokens?: number;
  format?: "wav" | "pcm" | "raw_f32" | "raw_i16";
  temperature?: number;
  speed?: number;
}

export interface TTSGenerationStats {
  generation_time_ms: number;
  audio_duration_secs: number;
  rtf: number;
  tokens_generated: number;
}

export interface TTSGenerateResult {
  audioBlob: Blob;
  stats: TTSGenerationStats | null;
}

export type TTSStreamEvent =
  | {
      event: "start";
      request_id: string;
      sample_rate: number;
      audio_format: "wav" | "pcm_i16" | "pcm_f32";
    }
  | {
      event: "chunk";
      request_id: string;
      sequence: number;
      audio_base64: string;
      sample_count: number;
      is_final: boolean;
    }
  | {
      event: "final";
      request_id: string;
      tokens_generated: number;
      generation_time_ms: number;
      audio_duration_secs: number;
      rtf: number;
    }
  | { event: "error"; request_id?: string; error: string }
  | { event: "done"; request_id?: string };

export interface TTSStreamCallbacks {
  onStart?: (event: {
    requestId: string;
    sampleRate: number;
    audioFormat: "wav" | "pcm_i16" | "pcm_f32";
  }) => void;
  onChunk?: (event: {
    requestId: string;
    sequence: number;
    audioBase64: string;
    sampleCount: number;
    isFinal: boolean;
  }) => void;
  onFinal?: (stats: TTSGenerationStats) => void;
  onError?: (error: string) => void;
  onDone?: () => void;
}

export type SpeechHistoryRoute =
  | "text-to-speech"
  | "voice-design"
  | "voice-cloning";

export interface SpeechHistoryRecordSummary {
  id: string;
  created_at: number;
  route_kind: "text_to_speech" | "voice_design" | "voice_cloning";
  model_id: string | null;
  speaker: string | null;
  language: string | null;
  input_preview: string;
  input_chars: number;
  generation_time_ms: number;
  audio_duration_secs: number | null;
  rtf: number | null;
  tokens_generated: number | null;
  audio_mime_type: string;
  audio_filename: string | null;
}

export interface SpeechHistoryRecord {
  id: string;
  created_at: number;
  route_kind: "text_to_speech" | "voice_design" | "voice_cloning";
  model_id: string | null;
  speaker: string | null;
  language: string | null;
  input_text: string;
  voice_description: string | null;
  reference_text: string | null;
  generation_time_ms: number;
  audio_duration_secs: number | null;
  rtf: number | null;
  tokens_generated: number | null;
  audio_mime_type: string;
  audio_filename: string | null;
}

export interface SpeechHistoryRecordCreateRequest {
  model_id: string;
  text: string;
  speaker?: string;
  language?: string;
  voice_description?: string;
  reference_audio?: string;
  reference_text?: string;
  temperature?: number;
  speed?: number;
  max_tokens?: number;
  max_output_tokens?: number;
  top_k?: number;
}

type SpeechHistoryRecordStreamEvent =
  | {
      event: "start";
      request_id: string;
      sample_rate: number;
      audio_format: "pcm_i16";
    }
  | {
      event: "chunk";
      request_id: string;
      sequence: number;
      audio_base64: string;
      sample_count: number;
    }
  | {
      event: "final";
      request_id: string;
      tokens_generated: number;
      generation_time_ms: number;
      audio_duration_secs: number;
      rtf: number;
      record: SpeechHistoryRecord;
    }
  | { event: "error"; request_id?: string; error: string }
  | { event: "done"; request_id?: string };

export interface SpeechHistoryRecordStreamCallbacks {
  onStart?: (event: {
    requestId: string;
    sampleRate: number;
    audioFormat: "pcm_i16";
  }) => void;
  onChunk?: (event: {
    requestId: string;
    sequence: number;
    audioBase64: string;
    sampleCount: number;
  }) => void;
  onFinal?: (event: {
    record: SpeechHistoryRecord;
    stats: TTSGenerationStats;
  }) => void;
  onError?: (error: string) => void;
  onDone?: () => void;
}

export type SavedVoiceSourceRouteKind = "voice_design" | "voice_cloning";

export interface SavedVoiceSummary {
  id: string;
  created_at: number;
  updated_at: number;
  name: string;
  reference_text_preview: string;
  reference_text_chars: number;
  audio_mime_type: string;
  audio_filename: string | null;
  source_route_kind: SavedVoiceSourceRouteKind | null;
  source_record_id: string | null;
}

export interface SavedVoice {
  id: string;
  created_at: number;
  updated_at: number;
  name: string;
  reference_text: string;
  audio_mime_type: string;
  audio_filename: string | null;
  source_route_kind: SavedVoiceSourceRouteKind | null;
  source_record_id: string | null;
}

export interface SavedVoiceCreateRequest {
  name: string;
  reference_text: string;
  audio_base64: string;
  audio_mime_type?: string;
  audio_filename?: string;
  source_route_kind?: SavedVoiceSourceRouteKind;
  source_record_id?: string;
}

// ============================================================================
// Unified STT (ASR) Types
// ============================================================================

export interface STTRequest {
  audio_base64: string;
  model_id?: string;
  language?: string;
}

export interface STTResponse {
  transcription: string;
  language: string | null;
}

export interface ASRTranscribeRequest {
  audio_base64?: string;
  audio_file?: Blob;
  audio_filename?: string;
  model_id?: string;
  language?: string;
}

export interface ASRTranscribeResponse {
  transcription: string;
  language: string | null;
  stats?: {
    processing_time_ms: number;
    audio_duration_secs: number | null;
    rtf: number | null;
  };
}

export interface TranscriptionRecordSummary {
  id: string;
  created_at: number;
  model_id: string | null;
  language: string | null;
  duration_secs: number | null;
  processing_time_ms: number;
  rtf: number | null;
  audio_mime_type: string;
  audio_filename: string | null;
  transcription_preview: string;
  transcription_chars: number;
}

export interface TranscriptionRecord {
  id: string;
  created_at: number;
  model_id: string | null;
  language: string | null;
  duration_secs: number | null;
  processing_time_ms: number;
  rtf: number | null;
  audio_mime_type: string;
  audio_filename: string | null;
  transcription: string;
}

export interface TranscriptionRecordCreateRequest {
  audio_base64?: string;
  audio_file?: Blob;
  audio_filename?: string;
  model_id?: string;
  language?: string;
}

type TranscriptionRecordStreamEvent =
  | { event: "start" }
  | { event: "delta"; delta: string }
  | { event: "final"; record: TranscriptionRecord }
  | { event: "error"; error: string }
  | { event: "done" };

export interface TranscriptionRecordStreamCallbacks {
  onStart?: () => void;
  onDelta?: (delta: string) => void;
  onFinal?: (record: TranscriptionRecord) => void;
  onError?: (error: string) => void;
  onDone?: () => void;
}

export interface DiarizationSegment {
  speaker: string;
  start: number;
  end: number;
  confidence?: number | null;
}

export interface DiarizationWord {
  word: string;
  speaker: string;
  start: number;
  end: number;
  speaker_confidence?: number | null;
  overlaps_segment: boolean;
}

export interface DiarizationUtterance {
  speaker: string;
  start: number;
  end: number;
  text: string;
  word_start: number;
  word_end: number;
}

export interface DiarizationRequest {
  audio_base64?: string;
  audio_file?: Blob;
  audio_filename?: string;
  model_id?: string;
  asr_model_id?: string;
  aligner_model_id?: string;
  llm_model_id?: string;
  enable_llm_refinement?: boolean;
  min_speakers?: number;
  max_speakers?: number;
  min_speech_duration_ms?: number;
  min_silence_duration_ms?: number;
}

export interface DiarizationResponse {
  segments: DiarizationSegment[];
  words: DiarizationWord[];
  utterances: DiarizationUtterance[];
  asr_text: string;
  raw_transcript: string;
  transcript: string;
  llm_refined: boolean;
  alignment_coverage: number;
  unattributed_words: number;
  speaker_count: number;
  duration: number;
  stats?: {
    processing_time_ms: number;
    rtf: number | null;
  };
}

export interface DiarizationRecordSummary {
  id: string;
  created_at: number;
  model_id: string | null;
  speaker_count: number;
  duration_secs: number | null;
  processing_time_ms: number;
  rtf: number | null;
  audio_mime_type: string;
  audio_filename: string | null;
  transcript_preview: string;
  transcript_chars: number;
}

export interface DiarizationRecord {
  id: string;
  created_at: number;
  model_id: string | null;
  asr_model_id: string | null;
  aligner_model_id: string | null;
  llm_model_id: string | null;
  min_speakers: number | null;
  max_speakers: number | null;
  min_speech_duration_ms: number | null;
  min_silence_duration_ms: number | null;
  enable_llm_refinement: boolean;
  processing_time_ms: number;
  duration_secs: number | null;
  rtf: number | null;
  speaker_count: number;
  alignment_coverage: number | null;
  unattributed_words: number;
  llm_refined: boolean;
  asr_text: string;
  raw_transcript: string;
  transcript: string;
  segments: DiarizationSegment[];
  words: DiarizationWord[];
  utterances: DiarizationUtterance[];
  audio_mime_type: string;
  audio_filename: string | null;
}

export interface DiarizationRecordCreateRequest {
  audio_base64?: string;
  audio_file?: Blob;
  audio_filename?: string;
  model_id?: string;
  asr_model_id?: string;
  aligner_model_id?: string;
  llm_model_id?: string;
  min_speakers?: number;
  max_speakers?: number;
  min_speech_duration_ms?: number;
  min_silence_duration_ms?: number;
  enable_llm_refinement?: boolean;
}

export type ASRStreamEvent =
  | { event: "start"; audio_duration_secs: number | null }
  | { event: "delta"; delta: string }
  | { event: "partial"; text: string; is_final: boolean }
  | {
      event: "final";
      text: string;
      language: string | null;
      audio_duration_secs: number | null;
    }
  | { event: "error"; error: string }
  | { event: "done" };

export interface ASRStreamCallbacks {
  onStart?: (audioDuration: number | null) => void;
  onDelta?: (delta: string) => void;
  onPartial?: (text: string) => void;
  onFinal?: (
    text: string,
    language: string | null,
    audioDuration: number | null,
  ) => void;
  onError?: (error: string) => void;
  onDone?: () => void;
}

export interface ASRStatusResponse {
  running: boolean;
  status: string;
  device: string | null;
  cached_models: string[];
}

export interface SpeechToSpeechRequest {
  audio_base64?: string;
  audio_file?: Blob;
  audio_filename?: string;
  model_id?: string;
  language?: string;
  system_prompt?: string;
  temperature?: number;
  top_k?: number;
}

export interface SpeechToSpeechResponse {
  text: string;
  transcription: string | null;
  audioBlob: Blob;
  sampleRate: number;
  generationTimeMs: number;
}

export type SpeechToSpeechStreamEvent =
  | { event: "start" }
  | { event: "delta"; delta: string }
  | {
      event: "final";
      text: string;
      transcription: string | null;
      audio_base64: string;
      sample_rate: number;
      generation_time_ms: number;
    }
  | { event: "error"; error: string }
  | { event: "done" };

export interface SpeechToSpeechStreamCallbacks {
  onStart?: () => void;
  onDelta?: (delta: string) => void;
  onFinal?: (result: SpeechToSpeechResponse) => void;
  onError?: (error: string) => void;
  onDone?: () => void;
}

interface OpenAiChatCompletion {
  id: string;
  model: string;
  choices: Array<{
    index: number;
    message: {
      role: "assistant" | "system" | "user";
      content: string;
    };
    finish_reason: string;
  }>;
  usage?: {
    completion_tokens?: number;
  };
  izwi_generation_time_ms?: number;
}

interface OpenAiChatChunk {
  id: string;
  model: string;
  choices: Array<{
    index: number;
    delta: {
      role?: "assistant" | "system" | "user";
      content?: string;
    };
    finish_reason: string | null;
  }>;
  usage?: {
    completion_tokens?: number;
  };
  izwi_generation_time_ms?: number;
}

interface OpenAiResponseObject {
  id: string;
  status: string;
  model: string;
  output?: Array<{
    type: string;
    role: string;
    content?: Array<{
      type: string;
      text?: string;
    }>;
  }>;
  usage?: {
    input_tokens?: number;
    output_tokens?: number;
    total_tokens?: number;
  };
}

type AsrResponseFormat = "json" | "verbose_json";
type DiarizationResponseFormat = "json" | "verbose_json" | "text";

class ApiClient {
  readonly baseUrl: string;

  constructor(baseUrl: string = API_BASE) {
    this.baseUrl = baseUrl;
  }

  private async request<T>(path: string, options?: RequestInit): Promise<T> {
    const response = await fetch(`${this.baseUrl}${path}`, {
      ...options,
      headers: {
        "Content-Type": "application/json",
        ...options?.headers,
      },
    });

    if (!response.ok) {
      const error = await response
        .json()
        .catch(() => ({ error: { message: "Request failed" } }));
      throw new Error(error.error?.message || "Request failed");
    }

    return response.json();
  }

  private mapResponseObject(payload: OpenAiResponseObject): ResponsesObject {
    const firstOutputText =
      payload.output?.[0]?.content
        ?.map((part) => part.text ?? "")
        .join("") ?? "";

    return {
      id: payload.id,
      status: payload.status,
      model: payload.model,
      output_text: firstOutputText,
      usage: {
        input_tokens: payload.usage?.input_tokens ?? 0,
        output_tokens: payload.usage?.output_tokens ?? 0,
        total_tokens: payload.usage?.total_tokens ?? 0,
      },
    };
  }

  // ========================================================================
  // Admin Model Management
  // ========================================================================

  async listModels(): Promise<ModelsResponse> {
    return this.request("/admin/models");
  }

  async getModelInfo(variant: string): Promise<ModelInfo> {
    return this.request(`/admin/models/${variant}`);
  }

  async downloadModel(
    variant: string,
  ): Promise<{ status: string; message: string }> {
    return this.request(`/admin/models/${variant}/download`, { method: "POST" });
  }

  async loadModel(
    variant: string,
  ): Promise<{ status: string; message: string }> {
    return this.request(`/admin/models/${variant}/load`, { method: "POST" });
  }

  async unloadModel(
    variant: string,
  ): Promise<{ status: string; message: string }> {
    return this.request(`/admin/models/${variant}/unload`, { method: "POST" });
  }

  async deleteModel(
    variant: string,
  ): Promise<{ status: string; message: string }> {
    return this.request(`/admin/models/${variant}`, { method: "DELETE" });
  }

  async cancelDownload(
    variant: string,
  ): Promise<{ status: string; message: string }> {
    return this.request(`/admin/models/${variant}/download/cancel`, {
      method: "POST",
    });
  }

  // ========================================================================
  // OpenAI-compatible TTS API
  // ========================================================================

  async generateTTS(request: TTSRequest): Promise<Blob> {
    const result = await this.generateTTSWithStats(request);
    return result.audioBlob;
  }

  async generateTTSWithStats(request: TTSRequest): Promise<TTSGenerateResult> {
    const response = await fetch(`${this.baseUrl}/audio/speech`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        model: request.model_id,
        input: request.text,
        language: request.language,
        voice: request.speaker,
        instructions: request.voice_description,
        reference_audio: request.reference_audio,
        reference_text: request.reference_text,
        max_tokens: request.max_tokens,
        temperature: request.temperature,
        speed: request.speed,
        response_format: request.format ?? "wav",
      }),
    });

    if (!response.ok) {
      const error = await response
        .json()
        .catch(() => ({ error: { message: "TTS generation failed" } }));
      throw new Error(error.error?.message || "TTS generation failed");
    }

    const generationTimeMs = response.headers.get("X-Generation-Time-Ms");
    const audioDurationSecs = response.headers.get("X-Audio-Duration-Secs");
    const rtf = response.headers.get("X-RTF");
    const tokensGenerated = response.headers.get("X-Tokens-Generated");

    const stats: TTSGenerationStats | null =
      generationTimeMs && audioDurationSecs && rtf && tokensGenerated
        ? {
            generation_time_ms: parseFloat(generationTimeMs),
            audio_duration_secs: parseFloat(audioDurationSecs),
            rtf: parseFloat(rtf),
            tokens_generated: parseInt(tokensGenerated, 10),
          }
        : null;

    const audioBlob = await response.blob();
    return { audioBlob, stats };
  }

  generateTTSStream(
    request: TTSRequest,
    callbacks: TTSStreamCallbacks,
  ): AbortController {
    const abortController = new AbortController();

    const startStream = async () => {
      try {
        const response = await fetch(`${this.baseUrl}/audio/speech`, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            model: request.model_id,
            input: request.text,
            language: request.language,
            voice: request.speaker,
            instructions: request.voice_description,
            reference_audio: request.reference_audio,
            reference_text: request.reference_text,
            max_tokens: request.max_tokens,
            temperature: request.temperature,
            speed: request.speed,
            response_format: request.format ?? "pcm",
            stream: true,
          }),
          signal: abortController.signal,
        });

        if (!response.ok) {
          const error = await response
            .json()
            .catch(() => ({ error: { message: "TTS streaming failed" } }));
          callbacks.onError?.(error.error?.message || "TTS streaming failed");
          callbacks.onDone?.();
          return;
        }

        const reader = response.body?.getReader();
        if (!reader) {
          callbacks.onError?.("No response body");
          callbacks.onDone?.();
          return;
        }

        const decoder = new TextDecoder();
        let buffer = "";

        while (true) {
          const { done, value } = await reader.read();
          if (done) break;

          buffer += decoder.decode(value, { stream: true });
          const lines = buffer.split("\n");
          buffer = lines.pop() || "";

          for (const line of lines) {
            if (!line.startsWith("data:")) continue;
            const data = line.slice(5).trim();
            if (!data) continue;

            try {
              const event = JSON.parse(data) as TTSStreamEvent;
              switch (event.event) {
                case "start":
                  callbacks.onStart?.({
                    requestId: event.request_id,
                    sampleRate: event.sample_rate,
                    audioFormat: event.audio_format,
                  });
                  break;
                case "chunk":
                  callbacks.onChunk?.({
                    requestId: event.request_id,
                    sequence: event.sequence,
                    audioBase64: event.audio_base64,
                    sampleCount: event.sample_count,
                    isFinal: event.is_final,
                  });
                  break;
                case "final":
                  callbacks.onFinal?.({
                    generation_time_ms: event.generation_time_ms,
                    audio_duration_secs: event.audio_duration_secs,
                    rtf: event.rtf,
                    tokens_generated: event.tokens_generated,
                  });
                  break;
                case "error":
                  callbacks.onError?.(event.error);
                  break;
                case "done":
                  callbacks.onDone?.();
                  return;
              }
            } catch {
              // Skip malformed payloads.
            }
          }
        }

        callbacks.onDone?.();
      } catch (error) {
        if ((error as Error).name !== "AbortError") {
          callbacks.onError?.(
            error instanceof Error ? error.message : "TTS stream error",
          );
        }
        callbacks.onDone?.();
      }
    };

    startStream();
    return abortController;
  }

  // ========================================================================
  // Speech History Routes
  // ========================================================================

  private speechHistoryRoutePrefix(route: SpeechHistoryRoute): string {
    return `/${route}`;
  }

  private buildSpeechHistoryRecordCreateBody(
    request: SpeechHistoryRecordCreateRequest,
    stream: boolean,
  ): Record<string, unknown> {
    return {
      model_id: request.model_id,
      text: request.text,
      speaker: request.speaker,
      language: request.language,
      voice_description: request.voice_description,
      reference_audio: request.reference_audio,
      reference_text: request.reference_text,
      temperature: request.temperature,
      speed: request.speed,
      max_tokens: request.max_tokens,
      max_output_tokens: request.max_output_tokens,
      top_k: request.top_k,
      stream,
    };
  }

  async listSpeechHistoryRecords(
    route: SpeechHistoryRoute,
  ): Promise<SpeechHistoryRecordSummary[]> {
    const payload = await this.request<{ records: SpeechHistoryRecordSummary[] }>(
      `${this.speechHistoryRoutePrefix(route)}/records`,
    );
    return payload.records ?? [];
  }

  async getSpeechHistoryRecord(
    route: SpeechHistoryRoute,
    recordId: string,
  ): Promise<SpeechHistoryRecord> {
    return this.request(
      `${this.speechHistoryRoutePrefix(route)}/records/${encodeURIComponent(recordId)}`,
    );
  }

  async createSpeechHistoryRecord(
    route: SpeechHistoryRoute,
    request: SpeechHistoryRecordCreateRequest,
  ): Promise<SpeechHistoryRecord> {
    const response = await fetch(
      `${this.baseUrl}${this.speechHistoryRoutePrefix(route)}/records`,
      {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(this.buildSpeechHistoryRecordCreateBody(request, false)),
      },
    );

    if (!response.ok) {
      const error = await response
        .json()
        .catch(() => ({ error: { message: "Speech generation failed" } }));
      throw new Error(error.error?.message || "Speech generation failed");
    }

    return response.json();
  }

  createSpeechHistoryRecordStream(
    route: SpeechHistoryRoute,
    request: SpeechHistoryRecordCreateRequest,
    callbacks: SpeechHistoryRecordStreamCallbacks,
  ): AbortController {
    const abortController = new AbortController();

    const startStream = async () => {
      try {
        const response = await fetch(
          `${this.baseUrl}${this.speechHistoryRoutePrefix(route)}/records`,
          {
            method: "POST",
            headers: {
              "Content-Type": "application/json",
            },
            body: JSON.stringify(
              this.buildSpeechHistoryRecordCreateBody(request, true),
            ),
            signal: abortController.signal,
          },
        );

        if (!response.ok) {
          const error = await response
            .json()
            .catch(() => ({ error: { message: "Speech streaming failed" } }));
          callbacks.onError?.(error.error?.message || "Speech streaming failed");
          callbacks.onDone?.();
          return;
        }

        const reader = response.body?.getReader();
        if (!reader) {
          callbacks.onError?.("No response body");
          callbacks.onDone?.();
          return;
        }

        const decoder = new TextDecoder();
        let buffer = "";

        while (true) {
          const { done, value } = await reader.read();
          if (done) break;

          buffer += decoder.decode(value, { stream: true });
          const lines = buffer.split("\n");
          buffer = lines.pop() || "";

          for (const line of lines) {
            if (!line.startsWith("data:")) continue;
            const data = line.slice(5).trim();
            if (!data) continue;

            try {
              const event = JSON.parse(data) as SpeechHistoryRecordStreamEvent;
              switch (event.event) {
                case "start":
                  callbacks.onStart?.({
                    requestId: event.request_id,
                    sampleRate: event.sample_rate,
                    audioFormat: event.audio_format,
                  });
                  break;
                case "chunk":
                  callbacks.onChunk?.({
                    requestId: event.request_id,
                    sequence: event.sequence,
                    audioBase64: event.audio_base64,
                    sampleCount: event.sample_count,
                  });
                  break;
                case "final":
                  callbacks.onFinal?.({
                    record: event.record,
                    stats: {
                      generation_time_ms: event.generation_time_ms,
                      audio_duration_secs: event.audio_duration_secs,
                      rtf: event.rtf,
                      tokens_generated: event.tokens_generated,
                    },
                  });
                  break;
                case "error":
                  callbacks.onError?.(event.error);
                  break;
                case "done":
                  callbacks.onDone?.();
                  return;
              }
            } catch {
              // Skip malformed SSE payloads.
            }
          }
        }

        callbacks.onDone?.();
      } catch (error) {
        if ((error as Error).name !== "AbortError") {
          callbacks.onError?.(
            error instanceof Error ? error.message : "Speech stream error",
          );
        }
        callbacks.onDone?.();
      }
    };

    startStream();
    return abortController;
  }

  speechHistoryRecordAudioUrl(
    route: SpeechHistoryRoute,
    recordId: string,
    options?: {
      download?: boolean;
    },
  ): string {
    const base = `${this.baseUrl}${this.speechHistoryRoutePrefix(route)}/records/${encodeURIComponent(recordId)}/audio`;
    if (options?.download) {
      return `${base}?download=true`;
    }
    return base;
  }

  async downloadAudioFile(url: string, suggestedFilename: string): Promise<void> {
    if (typeof window !== "undefined") {
      const internals = (
        window as unknown as {
          __TAURI_INTERNALS__?: {
            invoke?: (
              command: string,
              args?: Record<string, unknown>,
            ) => Promise<unknown>;
          };
        }
      ).__TAURI_INTERNALS__;

      if (
        typeof internals?.invoke === "function" &&
        /^https?:\/\//i.test(url)
      ) {
        await internals.invoke("download_audio_file", {
          url,
          suggested_filename: suggestedFilename,
        });
        return;
      }
    }

    const response = await fetch(url);
    if (!response.ok) {
      throw new Error(`Audio download failed (${response.status})`);
    }

    const blob = await response.blob();
    const objectUrl = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = objectUrl;
    anchor.download = suggestedFilename;
    anchor.style.display = "none";
    document.body.appendChild(anchor);
    anchor.click();
    anchor.remove();

    window.setTimeout(() => {
      URL.revokeObjectURL(objectUrl);
    }, 1000);
  }

  async saveAudioFile(url: string, suggestedFilename: string): Promise<void> {
    await this.downloadAudioFile(url, suggestedFilename);
  }

  async deleteSpeechHistoryRecord(
    route: SpeechHistoryRoute,
    recordId: string,
  ): Promise<{ id: string; deleted: boolean }> {
    return this.request(
      `${this.speechHistoryRoutePrefix(route)}/records/${encodeURIComponent(recordId)}`,
      { method: "DELETE" },
    );
  }

  async listTextToSpeechRecords(): Promise<SpeechHistoryRecordSummary[]> {
    return this.listSpeechHistoryRecords("text-to-speech");
  }

  async getTextToSpeechRecord(recordId: string): Promise<SpeechHistoryRecord> {
    return this.getSpeechHistoryRecord("text-to-speech", recordId);
  }

  async createTextToSpeechRecord(
    request: SpeechHistoryRecordCreateRequest,
  ): Promise<SpeechHistoryRecord> {
    return this.createSpeechHistoryRecord("text-to-speech", request);
  }

  createTextToSpeechRecordStream(
    request: SpeechHistoryRecordCreateRequest,
    callbacks: SpeechHistoryRecordStreamCallbacks,
  ): AbortController {
    return this.createSpeechHistoryRecordStream(
      "text-to-speech",
      request,
      callbacks,
    );
  }

  textToSpeechRecordAudioUrl(
    recordId: string,
    options?: {
      download?: boolean;
    },
  ): string {
    return this.speechHistoryRecordAudioUrl("text-to-speech", recordId, options);
  }

  async deleteTextToSpeechRecord(
    recordId: string,
  ): Promise<{ id: string; deleted: boolean }> {
    return this.deleteSpeechHistoryRecord("text-to-speech", recordId);
  }

  async listVoiceDesignRecords(): Promise<SpeechHistoryRecordSummary[]> {
    return this.listSpeechHistoryRecords("voice-design");
  }

  async getVoiceDesignRecord(recordId: string): Promise<SpeechHistoryRecord> {
    return this.getSpeechHistoryRecord("voice-design", recordId);
  }

  async createVoiceDesignRecord(
    request: SpeechHistoryRecordCreateRequest,
  ): Promise<SpeechHistoryRecord> {
    return this.createSpeechHistoryRecord("voice-design", request);
  }

  voiceDesignRecordAudioUrl(
    recordId: string,
    options?: {
      download?: boolean;
    },
  ): string {
    return this.speechHistoryRecordAudioUrl("voice-design", recordId, options);
  }

  async deleteVoiceDesignRecord(
    recordId: string,
  ): Promise<{ id: string; deleted: boolean }> {
    return this.deleteSpeechHistoryRecord("voice-design", recordId);
  }

  async listVoiceCloningRecords(): Promise<SpeechHistoryRecordSummary[]> {
    return this.listSpeechHistoryRecords("voice-cloning");
  }

  async getVoiceCloningRecord(recordId: string): Promise<SpeechHistoryRecord> {
    return this.getSpeechHistoryRecord("voice-cloning", recordId);
  }

  async createVoiceCloningRecord(
    request: SpeechHistoryRecordCreateRequest,
  ): Promise<SpeechHistoryRecord> {
    return this.createSpeechHistoryRecord("voice-cloning", request);
  }

  voiceCloningRecordAudioUrl(
    recordId: string,
    options?: {
      download?: boolean;
    },
  ): string {
    return this.speechHistoryRecordAudioUrl("voice-cloning", recordId, options);
  }

  async deleteVoiceCloningRecord(
    recordId: string,
  ): Promise<{ id: string; deleted: boolean }> {
    return this.deleteSpeechHistoryRecord("voice-cloning", recordId);
  }

  async listSavedVoices(): Promise<SavedVoiceSummary[]> {
    const payload = await this.request<{ voices: SavedVoiceSummary[] }>(
      "/voices",
    );
    return payload.voices ?? [];
  }

  async getSavedVoice(voiceId: string): Promise<SavedVoice> {
    return this.request(`/voices/${encodeURIComponent(voiceId)}`);
  }

  async createSavedVoice(request: SavedVoiceCreateRequest): Promise<SavedVoice> {
    const response = await fetch(`${this.baseUrl}/voices`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        name: request.name,
        reference_text: request.reference_text,
        audio_base64: request.audio_base64,
        audio_mime_type: request.audio_mime_type,
        audio_filename: request.audio_filename,
        source_route_kind: request.source_route_kind,
        source_record_id: request.source_record_id,
      }),
    });

    if (!response.ok) {
      const error = await response
        .json()
        .catch(() => ({ error: { message: "Failed to save voice" } }));
      throw new Error(error.error?.message || "Failed to save voice");
    }

    return response.json();
  }

  savedVoiceAudioUrl(
    voiceId: string,
    options?: {
      download?: boolean;
    },
  ): string {
    const base = `${this.baseUrl}/voices/${encodeURIComponent(voiceId)}/audio`;
    if (options?.download) {
      return `${base}?download=true`;
    }
    return base;
  }

  async deleteSavedVoice(
    voiceId: string,
  ): Promise<{ id: string; deleted: boolean }> {
    return this.request(`/voices/${encodeURIComponent(voiceId)}`, {
      method: "DELETE",
    });
  }

  // ========================================================================
  // OpenAI-compatible ASR API
  // ========================================================================

  async listTranscriptionRecords(): Promise<TranscriptionRecordSummary[]> {
    const payload = await this.request<{ records: TranscriptionRecordSummary[] }>(
      "/transcription/records",
    );
    return payload.records ?? [];
  }

  async getTranscriptionRecord(recordId: string): Promise<TranscriptionRecord> {
    return this.request(
      `/transcription/records/${encodeURIComponent(recordId)}`,
    );
  }

  async createTranscriptionRecord(
    request: TranscriptionRecordCreateRequest,
  ): Promise<TranscriptionRecord> {
    const response = await fetch(`${this.baseUrl}/transcription/records`, {
      ...this.buildTranscriptionRecordRequestInit(request, false),
    });

    if (!response.ok) {
      const error = await response
        .json()
        .catch(() => ({ error: { message: "Transcription failed" } }));
      throw new Error(error.error?.message || "Transcription failed");
    }

    return response.json();
  }

  createTranscriptionRecordStream(
    request: TranscriptionRecordCreateRequest,
    callbacks: TranscriptionRecordStreamCallbacks,
  ): AbortController {
    const abortController = new AbortController();

    const startStream = async () => {
      try {
        const response = await fetch(`${this.baseUrl}/transcription/records`, {
          ...this.buildTranscriptionRecordRequestInit(request, true),
          signal: abortController.signal,
        });

        if (!response.ok) {
          const error = await response.json().catch(() => ({
            error: { message: "Streaming transcription failed" },
          }));
          callbacks.onError?.(
            error.error?.message || "Streaming transcription failed",
          );
          callbacks.onDone?.();
          return;
        }

        const reader = response.body?.getReader();
        if (!reader) {
          callbacks.onError?.("No response body");
          callbacks.onDone?.();
          return;
        }

        const decoder = new TextDecoder();
        let buffer = "";

        while (true) {
          const { done, value } = await reader.read();
          if (done) break;

          buffer += decoder.decode(value, { stream: true });
          const lines = buffer.split("\n");
          buffer = lines.pop() || "";

          for (const line of lines) {
            if (!line.startsWith("data:")) continue;
            const data = line.slice(5).trim();
            if (!data) continue;

            try {
              const event = JSON.parse(data) as TranscriptionRecordStreamEvent;
              switch (event.event) {
                case "start":
                  callbacks.onStart?.();
                  break;
                case "delta":
                  callbacks.onDelta?.(event.delta);
                  break;
                case "final":
                  callbacks.onFinal?.(event.record);
                  break;
                case "error":
                  callbacks.onError?.(event.error);
                  break;
                case "done":
                  callbacks.onDone?.();
                  return;
              }
            } catch {
              // Skip malformed SSE payloads.
            }
          }
        }

        callbacks.onDone?.();
      } catch (error) {
        if ((error as Error).name !== "AbortError") {
          callbacks.onError?.(
            error instanceof Error ? error.message : "Transcription stream error",
          );
        }
        callbacks.onDone?.();
      }
    };

    startStream();
    return abortController;
  }

  transcriptionRecordAudioUrl(recordId: string): string {
    return `${this.baseUrl}/transcription/records/${encodeURIComponent(recordId)}/audio`;
  }

  async deleteTranscriptionRecord(
    recordId: string,
  ): Promise<{ id: string; deleted: boolean }> {
    return this.request(`/transcription/records/${encodeURIComponent(recordId)}`, {
      method: "DELETE",
    });
  }

  async asrStatus(): Promise<ASRStatusResponse> {
    // Legacy method retained for UI compatibility.
    return {
      running: false,
      status: "unknown",
      device: null,
      cached_models: [],
    };
  }

  async asrTranscribe(
    request: ASRTranscribeRequest,
  ): Promise<ASRTranscribeResponse> {
    const response = await fetch(
      `${this.baseUrl}/audio/transcriptions`,
      this.buildAsrRequestInit(request, "verbose_json", false),
    );

    if (!response.ok) {
      const error = await response
        .json()
        .catch(() => ({ error: { message: "Transcription failed" } }));
      throw new Error(error.error?.message || "Transcription failed");
    }

    const payload = await response.json();
    const transcription = payload.text ?? "";

    return {
      transcription,
      language: payload.language ?? null,
      stats:
        typeof payload.processing_time_ms === "number"
          ? {
              processing_time_ms: payload.processing_time_ms,
              audio_duration_secs:
                typeof payload.duration === "number" ? payload.duration : null,
              rtf: typeof payload.rtf === "number" ? payload.rtf : null,
            }
          : undefined,
    };
  }

  async listDiarizationRecords(): Promise<DiarizationRecordSummary[]> {
    const payload = await this.request<{ records: DiarizationRecordSummary[] }>(
      "/diarization/records",
    );
    return payload.records ?? [];
  }

  async getDiarizationRecord(recordId: string): Promise<DiarizationRecord> {
    return this.request(`/diarization/records/${encodeURIComponent(recordId)}`);
  }

  async createDiarizationRecord(
    request: DiarizationRecordCreateRequest,
  ): Promise<DiarizationRecord> {
    const response = await fetch(`${this.baseUrl}/diarization/records`, {
      ...this.buildDiarizationRecordRequestInit(request),
    });

    if (!response.ok) {
      const error = await response
        .json()
        .catch(() => ({ error: { message: "Diarization failed" } }));
      throw new Error(error.error?.message || "Diarization failed");
    }

    return response.json();
  }

  diarizationRecordAudioUrl(recordId: string): string {
    return `${this.baseUrl}/diarization/records/${encodeURIComponent(recordId)}/audio`;
  }

  async deleteDiarizationRecord(
    recordId: string,
  ): Promise<{ id: string; deleted: boolean }> {
    return this.request(`/diarization/records/${encodeURIComponent(recordId)}`, {
      method: "DELETE",
    });
  }

  async diarize(request: DiarizationRequest): Promise<DiarizationResponse> {
    const response = await fetch(
      `${this.baseUrl}/audio/diarizations`,
      this.buildDiarizationRequestInit(request, "verbose_json"),
    );

    if (!response.ok) {
      const error = await response
        .json()
        .catch(() => ({ error: { message: "Diarization failed" } }));
      throw new Error(error.error?.message || "Diarization failed");
    }

    const payload = await response.json();
    const rawSegments = Array.isArray(payload.segments) ? payload.segments : [];
    const rawWords = Array.isArray(payload.words) ? payload.words : [];
    const rawUtterances = Array.isArray(payload.utterances)
      ? payload.utterances
      : [];

    const segments: DiarizationSegment[] = rawSegments
      .map((segment: unknown): DiarizationSegment => {
        const raw =
          segment && typeof segment === "object"
            ? (segment as Record<string, unknown>)
            : {};
        return {
          speaker: String(raw.speaker ?? "SPEAKER_00"),
          start: Number(raw.start ?? 0),
          end: Number(raw.end ?? 0),
          confidence:
            typeof raw.confidence === "number" ? raw.confidence : null,
        };
      })
      .filter(
        (segment: DiarizationSegment) =>
          Number.isFinite(segment.start) &&
          Number.isFinite(segment.end) &&
          segment.end > segment.start,
      );

    const words: DiarizationWord[] = rawWords
      .map((word: unknown): DiarizationWord => {
        const raw =
          word && typeof word === "object"
            ? (word as Record<string, unknown>)
            : {};
        return {
          word: String(raw.word ?? ""),
          speaker: String(raw.speaker ?? "UNKNOWN"),
          start: Number(raw.start ?? 0),
          end: Number(raw.end ?? 0),
          speaker_confidence:
            typeof raw.speaker_confidence === "number"
              ? raw.speaker_confidence
              : null,
          overlaps_segment: Boolean(raw.overlaps_segment),
        };
      })
      .filter(
        (word: DiarizationWord) =>
          word.word.trim().length > 0 &&
          Number.isFinite(word.start) &&
          Number.isFinite(word.end) &&
          word.end > word.start,
      );

    const utterances: DiarizationUtterance[] = rawUtterances
      .map((utterance: unknown): DiarizationUtterance => {
        const raw =
          utterance && typeof utterance === "object"
            ? (utterance as Record<string, unknown>)
            : {};
        return {
          speaker: String(raw.speaker ?? "UNKNOWN"),
          start: Number(raw.start ?? 0),
          end: Number(raw.end ?? 0),
          text: String(raw.text ?? ""),
          word_start: Number(raw.word_start ?? 0),
          word_end: Number(raw.word_end ?? 0),
        };
      })
      .filter(
        (utterance: DiarizationUtterance) =>
          utterance.text.trim().length > 0 &&
          Number.isFinite(utterance.start) &&
          Number.isFinite(utterance.end) &&
          utterance.end > utterance.start,
      );

    return {
      segments,
      words,
      utterances,
      asr_text: typeof payload.asr_text === "string" ? payload.asr_text : "",
      raw_transcript:
        typeof payload.raw_transcript === "string" ? payload.raw_transcript : "",
      transcript:
        typeof payload.transcript === "string"
          ? payload.transcript
          : typeof payload.text === "string"
            ? payload.text
            : "",
      llm_refined: Boolean(payload.llm_refined),
      alignment_coverage:
        typeof payload.alignment_coverage === "number"
          ? payload.alignment_coverage
          : 0,
      unattributed_words:
        typeof payload.unattributed_words === "number"
          ? payload.unattributed_words
          : 0,
      speaker_count:
        typeof payload.speaker_count === "number"
          ? payload.speaker_count
          : new Set(segments.map((segment) => segment.speaker)).size,
      duration: typeof payload.duration === "number" ? payload.duration : 0,
      stats:
        typeof payload.processing_time_ms === "number"
          ? {
              processing_time_ms: payload.processing_time_ms,
              rtf: typeof payload.rtf === "number" ? payload.rtf : null,
            }
          : undefined,
    };
  }

  asrTranscribeStream(
    request: ASRTranscribeRequest,
    callbacks: ASRStreamCallbacks,
  ): AbortController {
    const abortController = new AbortController();

    const startStream = async () => {
      try {
        const response = await fetch(`${this.baseUrl}/audio/transcriptions`, {
          ...this.buildAsrRequestInit(request, "json", true),
          signal: abortController.signal,
        });

        if (!response.ok) {
          const error = await response.json().catch(() => ({
            error: { message: "Streaming transcription failed" },
          }));
          callbacks.onError?.(
            error.error?.message || "Streaming transcription failed",
          );
          callbacks.onDone?.();
          return;
        }

        const reader = response.body?.getReader();
        if (!reader) {
          callbacks.onError?.("No response body");
          callbacks.onDone?.();
          return;
        }

        const decoder = new TextDecoder();
        let buffer = "";
        let assembledText = "";

        while (true) {
          const { done, value } = await reader.read();
          if (done) break;

          buffer += decoder.decode(value, { stream: true });

          const lines = buffer.split("\n");
          buffer = lines.pop() || "";

          for (const line of lines) {
            if (!line.startsWith("data:")) continue;
            const data = line.slice(5).trim();
            if (!data) continue;

            if (data === "[DONE]") {
              callbacks.onDone?.();
              return;
            }

            try {
              const event = JSON.parse(data) as ASRStreamEvent;
              switch (event.event) {
                case "start":
                  callbacks.onStart?.(event.audio_duration_secs);
                  break;
                case "delta":
                  assembledText += event.delta;
                  callbacks.onDelta?.(event.delta);
                  callbacks.onPartial?.(assembledText);
                  break;
                case "partial":
                  if (event.text.startsWith(assembledText)) {
                    const delta = event.text.slice(assembledText.length);
                    if (delta) callbacks.onDelta?.(delta);
                  } else if (event.text !== assembledText) {
                    callbacks.onDelta?.(event.text);
                  }
                  assembledText = event.text;
                  callbacks.onPartial?.(event.text);
                  break;
                case "final":
                  assembledText = event.text;
                  callbacks.onFinal?.(
                    event.text,
                    event.language,
                    event.audio_duration_secs,
                  );
                  break;
                case "error":
                  callbacks.onError?.(event.error);
                  break;
                case "done":
                  callbacks.onDone?.();
                  return;
              }
            } catch {
              // Skip malformed payloads.
            }
          }
        }

        callbacks.onDone?.();
      } catch (error) {
        if ((error as Error).name !== "AbortError") {
          callbacks.onError?.(
            error instanceof Error ? error.message : "Stream error",
          );
        }
        callbacks.onDone?.();
      }
    };

    startStream();
    return abortController;
  }

  async speechToSpeech(
    request: SpeechToSpeechRequest,
  ): Promise<SpeechToSpeechResponse> {
    const response = await fetch(
      `${this.baseUrl}/audio/speech-to-speech`,
      this.buildSpeechToSpeechRequestInit(request, false),
    );

    if (!response.ok) {
      const error = await response
        .json()
        .catch(() => ({ error: { message: "Speech-to-speech failed" } }));
      throw new Error(error.error?.message || "Speech-to-speech failed");
    }

    const payload = await response.json();
    return {
      text: payload.text ?? "",
      transcription: payload.transcription ?? null,
      audioBlob: this.wavBlobFromBase64(payload.audio_base64 ?? ""),
      sampleRate: payload.sample_rate ?? 24000,
      generationTimeMs:
        typeof payload.generation_time_ms === "number"
          ? payload.generation_time_ms
          : 0,
    };
  }

  speechToSpeechStream(
    request: SpeechToSpeechRequest,
    callbacks: SpeechToSpeechStreamCallbacks,
  ): AbortController {
    const abortController = new AbortController();

    const startStream = async () => {
      try {
        const response = await fetch(`${this.baseUrl}/audio/speech-to-speech`, {
          ...this.buildSpeechToSpeechRequestInit(request, true),
          signal: abortController.signal,
        });

        if (!response.ok) {
          const error = await response
            .json()
            .catch(() => ({ error: { message: "Speech-to-speech streaming failed" } }));
          callbacks.onError?.(
            error.error?.message || "Speech-to-speech streaming failed",
          );
          callbacks.onDone?.();
          return;
        }

        const reader = response.body?.getReader();
        if (!reader) {
          callbacks.onError?.("No response body");
          callbacks.onDone?.();
          return;
        }

        const decoder = new TextDecoder();
        let buffer = "";
        let latestText = "";

        while (true) {
          const { done, value } = await reader.read();
          if (done) break;

          buffer += decoder.decode(value, { stream: true });
          const lines = buffer.split("\n");
          buffer = lines.pop() || "";

          for (const line of lines) {
            if (!line.startsWith("data:")) continue;
            const data = line.slice(5).trim();
            if (!data) continue;

            try {
              const event = JSON.parse(data) as SpeechToSpeechStreamEvent;
              switch (event.event) {
                case "start":
                  callbacks.onStart?.();
                  break;
                case "delta":
                  latestText += event.delta;
                  callbacks.onDelta?.(event.delta);
                  break;
                case "final":
                  callbacks.onFinal?.({
                    text: event.text || latestText,
                    transcription: event.transcription ?? null,
                    audioBlob: this.wavBlobFromBase64(event.audio_base64),
                    sampleRate: event.sample_rate,
                    generationTimeMs: event.generation_time_ms,
                  });
                  break;
                case "error":
                  callbacks.onError?.(event.error);
                  break;
                case "done":
                  callbacks.onDone?.();
                  return;
              }
            } catch {
              // Skip malformed payloads.
            }
          }
        }

        callbacks.onDone?.();
      } catch (error) {
        if ((error as Error).name !== "AbortError") {
          callbacks.onError?.(
            error instanceof Error ? error.message : "Speech-to-speech stream error",
          );
        }
        callbacks.onDone?.();
      }
    };

    startStream();
    return abortController;
  }

  private buildAsrRequestInit(
    request: ASRTranscribeRequest,
    responseFormat: AsrResponseFormat,
    stream: boolean,
  ): RequestInit {
    if (request.audio_file) {
      const form = new FormData();
      form.append(
        "file",
        request.audio_file,
        request.audio_filename || "audio.wav",
      );
      if (request.model_id) form.append("model", request.model_id);
      if (request.language) form.append("language", request.language);
      form.append("response_format", responseFormat);
      if (stream) form.append("stream", "true");
      return {
        method: "POST",
        body: form,
      };
    }

    if (!request.audio_base64) {
      throw new Error("Missing audio input");
    }

    return {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        audio_base64: request.audio_base64,
        model: request.model_id,
        language: request.language,
        response_format: responseFormat,
        stream,
      }),
    };
  }

  private buildTranscriptionRecordRequestInit(
    request: TranscriptionRecordCreateRequest,
    stream: boolean,
  ): RequestInit {
    if (request.audio_file) {
      const form = new FormData();
      form.append(
        "file",
        request.audio_file,
        request.audio_filename || "audio.wav",
      );
      if (request.model_id) form.append("model", request.model_id);
      if (request.language) form.append("language", request.language);
      if (stream) form.append("stream", "true");
      return {
        method: "POST",
        body: form,
      };
    }

    if (!request.audio_base64) {
      throw new Error("Missing audio input");
    }

    return {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        audio_base64: request.audio_base64,
        model: request.model_id,
        language: request.language,
        stream,
      }),
    };
  }

  private buildDiarizationRecordRequestInit(
    request: DiarizationRecordCreateRequest,
  ): RequestInit {
    const enableLlmRefinement = true;

    if (request.audio_file) {
      const form = new FormData();
      form.append(
        "file",
        request.audio_file,
        request.audio_filename || "audio.wav",
      );
      if (request.model_id) form.append("model", request.model_id);
      if (request.asr_model_id) form.append("asr_model", request.asr_model_id);
      if (request.aligner_model_id) {
        form.append("aligner_model", request.aligner_model_id);
      }
      if (request.llm_model_id) form.append("llm_model", request.llm_model_id);
      form.append("enable_llm_refinement", "true");
      if (typeof request.min_speakers === "number") {
        form.append("min_speakers", String(request.min_speakers));
      }
      if (typeof request.max_speakers === "number") {
        form.append("max_speakers", String(request.max_speakers));
      }
      if (typeof request.min_speech_duration_ms === "number") {
        form.append(
          "min_speech_duration_ms",
          String(request.min_speech_duration_ms),
        );
      }
      if (typeof request.min_silence_duration_ms === "number") {
        form.append(
          "min_silence_duration_ms",
          String(request.min_silence_duration_ms),
        );
      }

      return {
        method: "POST",
        body: form,
      };
    }

    if (!request.audio_base64) {
      throw new Error("Missing audio input");
    }

    return {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        audio_base64: request.audio_base64,
        model: request.model_id,
        asr_model: request.asr_model_id,
        aligner_model: request.aligner_model_id,
        llm_model: request.llm_model_id,
        enable_llm_refinement: enableLlmRefinement,
        min_speakers: request.min_speakers,
        max_speakers: request.max_speakers,
        min_speech_duration_ms: request.min_speech_duration_ms,
        min_silence_duration_ms: request.min_silence_duration_ms,
      }),
    };
  }

  private buildDiarizationRequestInit(
    request: DiarizationRequest,
    responseFormat: DiarizationResponseFormat,
  ): RequestInit {
    const enableLlmRefinement = true;

    if (request.audio_file) {
      const form = new FormData();
      form.append(
        "file",
        request.audio_file,
        request.audio_filename || "audio.wav",
      );
      if (request.model_id) form.append("model", request.model_id);
      if (request.asr_model_id) form.append("asr_model", request.asr_model_id);
      if (request.aligner_model_id) {
        form.append("aligner_model", request.aligner_model_id);
      }
      if (request.llm_model_id) form.append("llm_model", request.llm_model_id);
      form.append("enable_llm_refinement", "true");
      if (typeof request.min_speakers === "number") {
        form.append("min_speakers", String(request.min_speakers));
      }
      if (typeof request.max_speakers === "number") {
        form.append("max_speakers", String(request.max_speakers));
      }
      if (typeof request.min_speech_duration_ms === "number") {
        form.append(
          "min_speech_duration_ms",
          String(request.min_speech_duration_ms),
        );
      }
      if (typeof request.min_silence_duration_ms === "number") {
        form.append(
          "min_silence_duration_ms",
          String(request.min_silence_duration_ms),
        );
      }
      form.append("response_format", responseFormat);
      return {
        method: "POST",
        body: form,
      };
    }

    if (!request.audio_base64) {
      throw new Error("Missing audio input");
    }

    return {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        audio_base64: request.audio_base64,
        model: request.model_id,
        asr_model: request.asr_model_id,
        aligner_model: request.aligner_model_id,
        llm_model: request.llm_model_id,
        enable_llm_refinement: enableLlmRefinement,
        min_speakers: request.min_speakers,
        max_speakers: request.max_speakers,
        min_speech_duration_ms: request.min_speech_duration_ms,
        min_silence_duration_ms: request.min_silence_duration_ms,
        response_format: responseFormat,
      }),
    };
  }

  private buildSpeechToSpeechRequestInit(
    request: SpeechToSpeechRequest,
    stream: boolean,
  ): RequestInit {
    if (request.audio_file) {
      const form = new FormData();
      form.append(
        "file",
        request.audio_file,
        request.audio_filename || "audio.wav",
      );
      form.append("stream", stream ? "true" : "false");
      if (request.model_id) form.append("model", request.model_id);
      if (request.language) form.append("language", request.language);
      if (request.system_prompt) form.append("system_prompt", request.system_prompt);
      if (typeof request.temperature === "number") {
        form.append("temperature", request.temperature.toString());
      }
      if (typeof request.top_k === "number") {
        form.append("top_k", request.top_k.toString());
      }

      return {
        method: "POST",
        body: form,
      };
    }

    if (!request.audio_base64) {
      throw new Error("Missing audio input");
    }

    return {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        audio_base64: request.audio_base64,
        model: request.model_id,
        language: request.language,
        system_prompt: request.system_prompt,
        temperature: request.temperature,
        top_k: request.top_k,
        stream,
      }),
    };
  }

  private wavBlobFromBase64(audioBase64: string): Blob {
    const binary = atob(audioBase64);
    const bytes = new Uint8Array(binary.length);
    for (let index = 0; index < binary.length; index += 1) {
      bytes[index] = binary.charCodeAt(index);
    }
    return new Blob([bytes], { type: "audio/wav" });
  }

  // ========================================================================
  // Thread-based Chat API
  // ========================================================================

  async listChatThreads(): Promise<ChatThread[]> {
    const payload = await this.request<{ threads: ChatThread[] }>("/chat/threads");
    return payload.threads;
  }

  async createChatThread(request?: ChatThreadCreateRequest): Promise<ChatThread> {
    return this.request("/chat/threads", {
      method: "POST",
      body: JSON.stringify({
        title: request?.title,
        model_id: request?.model_id,
      }),
    });
  }

  async updateChatThread(
    threadId: string,
    request: ChatThreadUpdateRequest,
  ): Promise<ChatThread> {
    return this.request(`/chat/threads/${encodeURIComponent(threadId)}`, {
      method: "PATCH",
      body: JSON.stringify({
        title: request.title,
      }),
    });
  }

  async getChatThread(threadId: string): Promise<ChatThreadDetail> {
    return this.request(`/chat/threads/${encodeURIComponent(threadId)}`);
  }

  async listChatThreadMessages(threadId: string): Promise<ChatThreadMessageRecord[]> {
    return this.request(`/chat/threads/${encodeURIComponent(threadId)}/messages`);
  }

  async deleteChatThread(
    threadId: string,
  ): Promise<{ id: string; deleted: boolean }> {
    return this.request(`/chat/threads/${encodeURIComponent(threadId)}`, {
      method: "DELETE",
    });
  }

  async sendChatThreadMessage(
    threadId: string,
    request: ChatThreadSendMessageRequest,
  ): Promise<ChatThreadSendMessageResponse> {
    return this.request(
      `/chat/threads/${encodeURIComponent(threadId)}/messages`,
      {
        method: "POST",
        body: JSON.stringify({
          model: request.model_id ?? "Qwen3.5-0.8B",
          content: request.content,
          content_parts: request.content_parts,
          max_tokens: request.max_tokens,
          stream: false,
          system_prompt: request.system_prompt,
          enable_thinking: request.enable_thinking,
        }),
      },
    );
  }

  sendChatThreadMessageStream(
    threadId: string,
    request: ChatThreadSendMessageRequest,
    callbacks: ChatThreadStreamCallbacks,
  ): AbortController {
    const abortController = new AbortController();

    const startStream = async () => {
      try {
        const response = await fetch(
          `${this.baseUrl}/chat/threads/${encodeURIComponent(threadId)}/messages`,
          {
            method: "POST",
            headers: {
              "Content-Type": "application/json",
            },
            body: JSON.stringify({
              model: request.model_id ?? "Qwen3.5-0.8B",
              content: request.content,
              content_parts: request.content_parts,
              max_tokens: request.max_tokens,
              stream: true,
              system_prompt: request.system_prompt,
              enable_thinking: request.enable_thinking,
            }),
            signal: abortController.signal,
          },
        );

        if (!response.ok) {
          const error = await response
            .json()
            .catch(() => ({ error: { message: "Thread chat streaming failed" } }));
          callbacks.onError?.(
            error.error?.message || "Thread chat streaming failed",
          );
          callbacks.onClose?.();
          return;
        }

        const reader = response.body?.getReader();
        if (!reader) {
          callbacks.onError?.("No response body");
          callbacks.onClose?.();
          return;
        }

        const decoder = new TextDecoder();
        let buffer = "";

        while (true) {
          const { done, value } = await reader.read();
          if (done) break;

          buffer += decoder.decode(value, { stream: true });
          const lines = buffer.split("\n");
          buffer = lines.pop() || "";

          for (const line of lines) {
            if (!line.startsWith("data:")) continue;
            const data = line.slice(5).trim();
            if (!data) continue;

            if (data === "[DONE]") {
              callbacks.onClose?.();
              return;
            }

            try {
              const event = JSON.parse(data) as ChatThreadStreamEvent;
              switch (event.event) {
                case "start":
                  callbacks.onStart?.({
                    threadId: event.thread_id,
                    modelId: event.model_id,
                    userMessage: event.user_message,
                  });
                  break;
                case "delta":
                  callbacks.onDelta?.(event.delta);
                  break;
                case "done":
                  callbacks.onDone?.({
                    threadId: event.thread_id,
                    modelId: event.model_id,
                    assistantMessage: event.assistant_message,
                    stats: event.stats,
                  });
                  break;
                case "error":
                  callbacks.onError?.(event.error);
                  break;
              }
            } catch {
              // Skip malformed SSE payloads.
            }
          }
        }

        callbacks.onClose?.();
      } catch (error) {
        if ((error as Error).name !== "AbortError") {
          callbacks.onError?.(
            error instanceof Error ? error.message : "Thread chat stream error",
          );
        }
        callbacks.onClose?.();
      }
    };

    startStream();
    return abortController;
  }

  // ========================================================================
  // Agent API (minimal MVP)
  // ========================================================================

  async createAgentSession(
    request: AgentSessionCreateRequest,
  ): Promise<AgentSession> {
    return this.request("/agent/sessions", {
      method: "POST",
      body: JSON.stringify({
        agent_id: request.agent_id,
        model_id: request.model_id,
        system_prompt: request.system_prompt,
        planning_mode: request.planning_mode,
        title: request.title,
      }),
    });
  }

  async getAgentSession(sessionId: string): Promise<AgentSession> {
    return this.request(`/agent/sessions/${encodeURIComponent(sessionId)}`);
  }

  async createAgentTurn(
    sessionId: string,
    request: AgentTurnRequest,
    signal?: AbortSignal,
  ): Promise<AgentTurnResponse> {
    return this.request(`/agent/sessions/${encodeURIComponent(sessionId)}/turns`, {
      method: "POST",
      body: JSON.stringify({
        input: request.input,
        model_id: request.model_id,
        max_output_tokens: request.max_output_tokens,
      }),
      signal,
    });
  }

  // ========================================================================
  // OpenAI-compatible Chat API
  // ========================================================================

  async chatCompletions(
    request: ChatCompletionRequest,
  ): Promise<ChatCompletionResponse> {
    const response = await this.request<OpenAiChatCompletion>(
      "/chat/completions",
      {
        method: "POST",
        body: JSON.stringify({
          model: request.model_id ?? "Qwen3.5-0.8B",
          messages: request.messages,
          max_tokens: request.max_tokens,
          stream: false,
        }),
      },
    );

    const firstChoice = response.choices[0];
    if (!firstChoice) {
      throw new Error("Missing assistant response");
    }

    return {
      model_id: response.model,
      message: {
        role: firstChoice.message.role,
        content: firstChoice.message.content,
      },
      stats: {
        tokens_generated: response.usage?.completion_tokens ?? 0,
        generation_time_ms: response.izwi_generation_time_ms ?? 0,
      },
    };
  }

  chatCompletionsStream(
    request: ChatCompletionRequest,
    callbacks: ChatStreamCallbacks,
  ): AbortController {
    const abortController = new AbortController();

    const startStream = async () => {
      try {
        const response = await fetch(`${this.baseUrl}/chat/completions`, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            model: request.model_id ?? "Qwen3.5-0.8B",
            messages: request.messages,
            max_tokens: request.max_tokens,
            stream: true,
          }),
          signal: abortController.signal,
        });

        if (!response.ok) {
          const error = await response
            .json()
            .catch(() => ({ error: { message: "Chat streaming failed" } }));
          callbacks.onError?.(error.error?.message || "Chat streaming failed");
          return;
        }

        callbacks.onStart?.(request.model_id ?? "Qwen3.5-0.8B");

        const reader = response.body?.getReader();
        if (!reader) {
          callbacks.onError?.("No response body");
          return;
        }

        const decoder = new TextDecoder();
        let buffer = "";
        let fullText = "";
        const streamStartedAt = performance.now();
        let completionTokens: number | null = null;
        let generationTimeMs: number | null = null;

        while (true) {
          const { done, value } = await reader.read();
          if (done) break;

          buffer += decoder.decode(value, { stream: true });

          const lines = buffer.split("\n");
          buffer = lines.pop() || "";

          for (const line of lines) {
            if (!line.startsWith("data:")) continue;
            const data = line.slice(5).trim();
            if (!data) continue;

            if (data === "[DONE]") {
              const elapsedMs = Math.max(
                1,
                Math.round(performance.now() - streamStartedAt),
              );
              callbacks.onDone?.(fullText, {
                tokens_generated:
                  completionTokens ?? Math.max(1, Math.floor(fullText.length / 4)),
                generation_time_ms: generationTimeMs ?? elapsedMs,
              });
              return;
            }

            try {
              const payload = JSON.parse(data) as OpenAiChatChunk;
              if (typeof payload.izwi_generation_time_ms === "number") {
                generationTimeMs = payload.izwi_generation_time_ms;
              }
              if (typeof payload.usage?.completion_tokens === "number") {
                completionTokens = payload.usage.completion_tokens;
              }
              const choice = payload.choices?.[0];
              const delta = choice?.delta?.content;
              if (delta) {
                fullText += delta;
                callbacks.onDelta?.(delta);
              }
            } catch {
              // Skip malformed SSE payloads.
            }
          }
        }
      } catch (error) {
        if ((error as Error).name !== "AbortError") {
          callbacks.onError?.(
            error instanceof Error ? error.message : "Chat stream error",
          );
        }
      }
    };

    startStream();
    return abortController;
  }

  // ========================================================================
  // OpenAI-compatible Responses API
  // ========================================================================

  async createResponse(request: ResponsesCreateRequest): Promise<ResponsesObject> {
    const payload = await this.request<OpenAiResponseObject>("/responses", {
      method: "POST",
      body: JSON.stringify({
        model: request.model_id ?? "Qwen3.5-0.8B",
        input: request.input,
        instructions: request.instructions,
        max_output_tokens: request.max_output_tokens,
        metadata: request.metadata,
        store: request.store,
      }),
    });

    return this.mapResponseObject(payload);
  }

  createResponseStream(
    request: ResponsesCreateRequest,
    callbacks: ResponsesStreamCallbacks,
  ): AbortController {
    const abortController = new AbortController();

    const startStream = async () => {
      try {
        const response = await fetch(`${this.baseUrl}/responses`, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            model: request.model_id ?? "Qwen3.5-0.8B",
            input: request.input,
            instructions: request.instructions,
            max_output_tokens: request.max_output_tokens,
            metadata: request.metadata,
            store: request.store,
            stream: true,
          }),
          signal: abortController.signal,
        });

        if (!response.ok) {
          const error = await response
            .json()
            .catch(() => ({ error: { message: "Responses streaming failed" } }));
          callbacks.onError?.(error.error?.message || "Responses streaming failed");
          callbacks.onDone?.();
          return;
        }

        const reader = response.body?.getReader();
        if (!reader) {
          callbacks.onError?.("No response body");
          callbacks.onDone?.();
          return;
        }

        const decoder = new TextDecoder();
        let buffer = "";

        while (true) {
          const { done, value } = await reader.read();
          if (done) break;

          buffer += decoder.decode(value, { stream: true });
          const lines = buffer.split("\n");
          buffer = lines.pop() || "";

          for (const line of lines) {
            if (!line.startsWith("data:")) continue;
            const data = line.slice(5).trim();
            if (!data) continue;

            if (data === "[DONE]") {
              callbacks.onDone?.();
              return;
            }

            try {
              const event = JSON.parse(data) as {
                type: string;
                response?: OpenAiResponseObject;
                delta?: string;
                error?: { message?: string };
              };

              if (event.type === "response.created" && event.response) {
                callbacks.onCreated?.(this.mapResponseObject(event.response));
              } else if (event.type === "response.output_text.delta") {
                callbacks.onDelta?.(event.delta ?? "");
              } else if (event.type === "response.completed" && event.response) {
                callbacks.onCompleted?.(this.mapResponseObject(event.response));
              } else if (event.type === "response.failed") {
                callbacks.onError?.(event.error?.message ?? "Responses request failed");
              }
            } catch {
              // Skip malformed SSE payloads.
            }
          }
        }

        callbacks.onDone?.();
      } catch (error) {
        if ((error as Error).name !== "AbortError") {
          callbacks.onError?.(
            error instanceof Error ? error.message : "Responses stream error",
          );
        }
        callbacks.onDone?.();
      }
    };

    startStream();
    return abortController;
  }

  async getResponse(id: string): Promise<ResponsesObject> {
    const payload = await this.request<OpenAiResponseObject>(`/responses/${id}`);
    return this.mapResponseObject(payload);
  }

  async cancelResponse(id: string): Promise<ResponsesObject> {
    const payload = await this.request<OpenAiResponseObject>(`/responses/${id}/cancel`, {
      method: "POST",
    });
    return this.mapResponseObject(payload);
  }

  async deleteResponse(id: string): Promise<{ id: string; object: string; deleted: boolean }> {
    return this.request(`/responses/${id}`, { method: "DELETE" });
  }

  // ========================================================================
  // Convenience aliases
  // ========================================================================

  async synthesize(request: TTSRequest): Promise<Blob> {
    return this.generateTTS(request);
  }

  async transcribe(request: STTRequest): Promise<STTResponse> {
    const result = await this.asrTranscribe({
      audio_base64: request.audio_base64,
      model_id: request.model_id,
      language: request.language,
    });

    return {
      transcription: result.transcription,
      language: result.language,
    };
  }
}

export const api = new ApiClient();
