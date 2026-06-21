import {
  ApiHttpClient,
  consumeDataStream,
  isAbortError,
} from "@/shared/api/http";
import {
  buildCursorQueryString,
  normalizeCursorPaginationMeta,
  type CursorPageResult,
  type CursorPaginationQuery,
} from "@/shared/api/pagination";

const FIRST_PARTY_AUDIO_UPLOAD_LIMIT_BYTES = 64 * 1024 * 1024;

export interface TTSRequest {
  text: string;
  model_id: string;
  language?: string;
  speaker?: string;
  voice_description?: string;
  reference_audio?: string;
  reference_text?: string;
  saved_voice_id?: string;
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
      event: "start" | "audio.started";
      request_id: string;
      sample_rate: number;
      audio_format: "wav" | "pcm_i16" | "pcm_f32";
    }
  | {
      event: "chunk" | "audio.chunk";
      request_id: string;
      sequence: number;
      audio_base64: string;
      sample_count: number;
      is_final: boolean;
    }
  | {
      event: "final" | "audio.done";
      request_id: string;
      tokens_generated: number;
      generation_time_ms: number;
      audio_duration_secs: number;
      rtf: number;
    }
  | { event: "error" | "audio.failed"; request_id?: string; error: string }
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

export type SpeechHistoryProcessingStatus =
  | "pending"
  | "processing"
  | "ready"
  | "failed";

export interface SpeechHistoryRecordSummary {
  id: string;
  created_at: number;
  route_kind: "text_to_speech" | "voice_design" | "voice_cloning";
  processing_status: SpeechHistoryProcessingStatus;
  processing_error?: string | null;
  model_id: string | null;
  speaker: string | null;
  language: string | null;
  saved_voice_id?: string | null;
  speed?: number | null;
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
  processing_status: SpeechHistoryProcessingStatus;
  processing_error?: string | null;
  model_id: string | null;
  speaker: string | null;
  language: string | null;
  saved_voice_id?: string | null;
  speed?: number | null;
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
  saved_voice_id?: string;
  temperature?: number;
  speed?: number;
  max_tokens?: number;
  max_output_tokens?: number;
  top_k?: number;
}

type SpeechHistoryRecordStreamEvent =
  | {
      event: "created";
      record: SpeechHistoryRecord;
    }
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
  onCreated?: (record: SpeechHistoryRecord) => void;
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

export type StudioProjectVoiceMode = "built_in" | "saved";
export type StudioProjectExportFormat = "wav" | "mp3" | "flac";
export type StudioProjectRenderJobStatus =
  | "queued"
  | "running"
  | "completed"
  | "failed"
  | "cancelled";

export interface StudioProjectSummary {
  id: string;
  created_at: number;
  updated_at: number;
  name: string;
  source_filename: string | null;
  model_id: string | null;
  voice_mode: StudioProjectVoiceMode;
  speaker: string | null;
  saved_voice_id: string | null;
  speed: number | null;
  segment_count: number;
  rendered_segment_count: number;
  total_chars: number;
}

export interface StudioProjectSegmentRecord {
  id: string;
  project_id: string;
  position: number;
  text: string;
  model_id: string | null;
  voice_mode: StudioProjectVoiceMode | null;
  speaker: string | null;
  saved_voice_id: string | null;
  input_chars: number;
  speech_record_id: string | null;
  updated_at: number;
  generation_time_ms: number | null;
  audio_duration_secs: number | null;
  audio_filename: string | null;
}

export interface StudioProjectRecord {
  id: string;
  created_at: number;
  updated_at: number;
  name: string;
  source_filename: string | null;
  source_text: string;
  model_id: string | null;
  voice_mode: StudioProjectVoiceMode;
  speaker: string | null;
  saved_voice_id: string | null;
  speed: number | null;
  segments: StudioProjectSegmentRecord[];
}

export interface StudioProjectCreateRequest {
  name?: string;
  source_filename?: string;
  source_text: string;
  model_id: string;
  voice_mode?: StudioProjectVoiceMode;
  speaker?: string;
  saved_voice_id?: string;
  speed?: number;
}

export interface StudioProjectUpdateRequest {
  name?: string;
  model_id?: string;
  voice_mode?: StudioProjectVoiceMode;
  speaker?: string;
  saved_voice_id?: string;
  speed?: number;
}

export interface StudioProjectSegmentUpdateRequest {
  text?: string;
  model_id?: string;
  voice_mode?: StudioProjectVoiceMode;
  speaker?: string;
  saved_voice_id?: string;
}

export interface StudioProjectSegmentCreateRequest {
  text: string;
  after_segment_id?: string;
}

export interface StudioProjectSegmentSplitRequest {
  before_text: string;
  after_text: string;
}

export interface StudioProjectSegmentReorderRequest {
  ordered_segment_ids: string[];
}

export interface StudioProjectSegmentBulkDeleteRequest {
  segment_ids: string[];
}

export interface StudioProjectFolderRecord {
  id: string;
  created_at: number;
  updated_at: number;
  name: string;
  parent_id: string | null;
  sort_order: number;
}

export interface StudioProjectFolderCreateRequest {
  name: string;
  parent_id?: string;
  sort_order?: number;
}

export interface StudioProjectMetaRecord {
  project_id: string;
  folder_id: string | null;
  tags: string[];
  default_export_format: StudioProjectExportFormat;
  last_render_job_id: string | null;
  last_rendered_at: number | null;
}

export interface StudioProjectMetaUpdateRequest {
  folder_id?: string;
  tags?: string[];
  default_export_format?: StudioProjectExportFormat;
  last_render_job_id?: string;
  last_rendered_at?: number;
}

export interface StudioProjectPronunciationRecord {
  id: string;
  project_id: string;
  source_text: string;
  replacement_text: string;
  locale: string | null;
  created_at: number;
  updated_at: number;
}

export interface StudioProjectPronunciationCreateRequest {
  source_text: string;
  replacement_text: string;
  locale?: string;
}

export interface StudioProjectSnapshotRecord {
  id: string;
  project_id: string;
  created_at: number;
  label: string | null;
  project_name: string;
}

export interface StudioProjectSnapshotCreateRequest {
  label?: string;
}

export interface StudioProjectRenderJobRecord {
  id: string;
  project_id: string;
  created_at: number;
  updated_at: number;
  status: StudioProjectRenderJobStatus;
  error_message: string | null;
  queued_segment_ids: string[];
}

export interface StudioProjectRenderJobCreateRequest {
  queued_segment_ids?: string[];
}

export interface StudioProjectRenderJobUpdateRequest {
  status?: StudioProjectRenderJobStatus;
  error_message?: string;
}

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
  prompt?: string;
  max_tokens?: number;
  timestamp_granularities?: Array<"word" | "words" | "segment" | "segments">;
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

export type SpeechTextJobKind = "transcription" | "diarization";

export type SpeechTextJobProcessingStatus =
  | "pending"
  | "processing"
  | "ready"
  | "failed";

export type SpeechTextJobSummaryStatus =
  | "not_requested"
  | "pending"
  | "ready"
  | "failed";

export interface SpeechTextJobSummaryBase {
  id: string;
  kind: SpeechTextJobKind;
  created_at: number;
  model_id: string | null;
  processing_status: SpeechTextJobProcessingStatus;
  processing_error?: string | null;
  duration_secs: number | null;
  processing_time_ms: number;
  rtf: number | null;
  audio_mime_type: string;
  audio_filename: string | null;
  summary_status?: SpeechTextJobSummaryStatus;
  summary_preview?: string | null;
  summary_chars?: number;
}

export interface SpeechTextTranscriptionSummary extends SpeechTextJobSummaryBase {
  kind: "transcription";
  language: string | null;
  transcription_preview: string;
  transcription_chars: number;
}

export interface SpeechTextDiarizationSummary extends SpeechTextJobSummaryBase {
  kind: "diarization";
  speaker_count: number;
  corrected_speaker_count?: number;
  speaker_name_override_count?: number;
  transcript_preview: string;
  transcript_chars: number;
}

export type SpeechTextJobSummary =
  | SpeechTextTranscriptionSummary
  | SpeechTextDiarizationSummary;

export interface SpeechTextJobBase {
  id: string;
  kind: SpeechTextJobKind;
  created_at: number;
  model_id: string | null;
  processing_status: SpeechTextJobProcessingStatus;
  processing_error?: string | null;
  duration_secs: number | null;
  processing_time_ms: number;
  rtf: number | null;
  audio_mime_type: string;
  audio_filename: string | null;
  summary_status?: SpeechTextJobSummaryStatus;
  summary_model_id?: string | null;
  summary_text?: string | null;
  summary_error?: string | null;
  summary_updated_at?: number | null;
}

export interface SpeechTextTranscriptionJob extends SpeechTextJobBase {
  kind: "transcription";
  aligner_model_id: string | null;
  language: string | null;
  transcription: string;
  segments: TranscriptionSegment[];
  words: TranscriptionWord[];
}

export interface SpeechTextDiarizationJob extends SpeechTextJobBase {
  kind: "diarization";
  asr_model_id: string | null;
  aligner_model_id: string | null;
  llm_model_id: string | null;
  min_speakers: number | null;
  max_speakers: number | null;
  min_speech_duration_ms: number | null;
  min_silence_duration_ms: number | null;
  enable_llm_refinement: boolean;
  speaker_count: number;
  corrected_speaker_count?: number;
  alignment_coverage: number | null;
  unattributed_words: number;
  llm_refined: boolean;
  asr_text: string;
  raw_transcript: string;
  transcript: string;
  segments: DiarizationSegment[];
  words: DiarizationWord[];
  utterances: DiarizationUtterance[];
  speaker_name_overrides?: Record<string, string>;
}

export type SpeechTextJob = SpeechTextTranscriptionJob | SpeechTextDiarizationJob;

export interface SpeechTextJobCreateRequestBase {
  kind: SpeechTextJobKind;
  audio_base64?: string;
  audio_file?: Blob;
  audio_filename?: string;
}

export interface SpeechTextTranscriptionJobCreateRequest
  extends SpeechTextJobCreateRequestBase {
  kind: "transcription";
  model_id?: string;
  aligner_model_id?: string;
  language?: string;
  include_timestamps?: boolean;
  generate_summary?: boolean;
  stream?: boolean;
}

export interface SpeechTextDiarizationJobCreateRequest
  extends SpeechTextJobCreateRequestBase {
  kind: "diarization";
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

export type SpeechTextJobCreateRequest =
  | SpeechTextTranscriptionJobCreateRequest
  | SpeechTextDiarizationJobCreateRequest;

export type SpeechTextJobQueryKind = SpeechTextJobKind | "all";

export interface SpeechTextJobPageQuery extends CursorPaginationQuery {
  job_kind?: SpeechTextJobQueryKind;
}

export interface TranscriptionRecordSummary {
  id: string;
  created_at: number;
  model_id: string | null;
  language: string | null;
  processing_status: TranscriptionProcessingStatus;
  processing_error?: string | null;
  processing_progress?: TranscriptionProcessingProgress | null;
  duration_secs: number | null;
  processing_time_ms: number;
  rtf: number | null;
  audio_mime_type: string;
  audio_filename: string | null;
  transcription_preview: string;
  transcription_chars: number;
  summary_status?: TranscriptionSummaryStatus;
  summary_preview?: string | null;
  summary_chars?: number;
}

export interface TranscriptionSegment {
  start: number;
  end: number;
  text: string;
  word_start: number;
  word_end: number;
}

export interface TranscriptionWord {
  word: string;
  start: number;
  end: number;
}

export type TranscriptionSummaryStatus =
  | "not_requested"
  | "pending"
  | "ready"
  | "failed";

export type TranscriptionProcessingStatus =
  | "pending"
  | "processing"
  | "ready"
  | "failed";

export type TranscriptionProcessingProgressPhase =
  | "processing"
  | "chunk_started"
  | "chunk_finished"
  | "aligning"
  | "complete";

export interface TranscriptionProcessingProgress {
  phase: TranscriptionProcessingProgressPhase;
  current_chunk?: number | null;
  total_chunks?: number | null;
  processed_audio_secs?: number | null;
  total_audio_secs?: number | null;
  percent?: number | null;
}

export interface TranscriptionRecord {
  id: string;
  created_at: number;
  model_id: string | null;
  aligner_model_id: string | null;
  language: string | null;
  processing_status: TranscriptionProcessingStatus;
  processing_error?: string | null;
  processing_progress?: TranscriptionProcessingProgress | null;
  duration_secs: number | null;
  processing_time_ms: number;
  rtf: number | null;
  audio_mime_type: string;
  audio_filename: string | null;
  transcription: string;
  segments: TranscriptionSegment[];
  words: TranscriptionWord[];
  summary_status?: TranscriptionSummaryStatus;
  summary_model_id?: string | null;
  summary_text?: string | null;
  summary_error?: string | null;
  summary_updated_at?: number | null;
}

export interface TranscriptionRecordCreateRequest {
  audio_base64?: string;
  audio_file?: Blob;
  audio_filename?: string;
  model_id?: string;
  aligner_model_id?: string;
  language?: string;
  include_timestamps?: boolean;
  generate_summary?: boolean;
}

type TranscriptionRecordStreamEvent =
  | { event: "created"; record: TranscriptionRecord }
  | { event: "start" }
  | { event: "delta"; delta: string }
  | { event: "progress"; progress: TranscriptionProcessingProgress }
  | { event: "final"; record: TranscriptionRecord }
  | { event: "error"; error: string }
  | { event: "done" };

export interface TranscriptionRecordStreamCallbacks {
  onCreated?: (record: TranscriptionRecord) => void;
  onStart?: () => void;
  onDelta?: (delta: string) => void;
  onProgress?: (progress: TranscriptionProcessingProgress) => void;
  onFinal?: (record: TranscriptionRecord) => void;
  onError?: (error: string) => void;
  onDone?: () => void;
  onUploadProgress?: (progress: UploadProgressInfo) => void;
}

export interface UploadProgressInfo {
  loadedBytes: number;
  totalBytes: number | null;
  percent: number | null;
  lengthComputable: boolean;
}

export interface SpeechTextJobUploadOptions {
  onUploadProgress?: (progress: UploadProgressInfo) => void;
  signal?: AbortSignal;
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

export interface DiarizationRecordSummary {
  id: string;
  created_at: number;
  model_id: string | null;
  processing_status: DiarizationProcessingStatus;
  processing_error?: string | null;
  speaker_count: number;
  corrected_speaker_count?: number;
  speaker_name_override_count?: number;
  duration_secs: number | null;
  processing_time_ms: number;
  rtf: number | null;
  audio_mime_type: string;
  audio_filename: string | null;
  transcript_preview: string;
  transcript_chars: number;
  summary_status?: DiarizationSummaryStatus;
  summary_preview?: string | null;
  summary_chars?: number;
}

export type DiarizationSummaryStatus =
  | "not_requested"
  | "pending"
  | "ready"
  | "failed";

export type DiarizationProcessingStatus =
  | "pending"
  | "processing"
  | "ready"
  | "failed";

export interface DiarizationRecord {
  id: string;
  created_at: number;
  model_id: string | null;
  asr_model_id: string | null;
  aligner_model_id: string | null;
  llm_model_id: string | null;
  processing_status: DiarizationProcessingStatus;
  processing_error?: string | null;
  min_speakers: number | null;
  max_speakers: number | null;
  min_speech_duration_ms: number | null;
  min_silence_duration_ms: number | null;
  enable_llm_refinement: boolean;
  processing_time_ms: number;
  duration_secs: number | null;
  rtf: number | null;
  speaker_count: number;
  corrected_speaker_count?: number;
  alignment_coverage: number | null;
  unattributed_words: number;
  llm_refined: boolean;
  asr_text: string;
  raw_transcript: string;
  transcript: string;
  summary_status?: DiarizationSummaryStatus;
  summary_model_id?: string | null;
  summary_text?: string | null;
  summary_error?: string | null;
  summary_updated_at?: number | null;
  segments: DiarizationSegment[];
  words: DiarizationWord[];
  utterances: DiarizationUtterance[];
  speaker_name_overrides?: Record<string, string>;
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

export interface DiarizationRecordUpdateRequest {
  speaker_name_overrides: Record<string, string>;
}

export interface DiarizationRecordRerunRequest {
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
  | { event: "done" }
  | { type: "transcript.text.delta"; delta: string }
  | {
      type: "transcript.text.done";
      text: string;
      language: string | null;
      audio_duration_secs: number | null;
    }
  | { type: "error"; error: string | { message?: string } };

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

type AsrResponseFormat = "json" | "verbose_json";

function normalizeAsrStreamEventName(
  event: ASRStreamEvent,
): "start" | "delta" | "partial" | "final" | "error" | "done" | null {
  if ("event" in event) {
    return event.event;
  }
  switch (event.type) {
    case "transcript.text.delta":
      return "delta";
    case "transcript.text.done":
      return "final";
    case "error":
      return "error";
    default:
      return null;
  }
}

function streamErrorMessage(
  error: string | { message?: string } | undefined,
  fallback: string,
): string {
  if (typeof error === "string") {
    return error;
  }
  if (error?.message) {
    return error.message;
  }
  return fallback;
}

function createAbortError(message = "Request aborted"): Error {
  if (typeof DOMException !== "undefined") {
    return new DOMException(message, "AbortError");
  }

  const error = new Error(message);
  error.name = "AbortError";
  return error;
}

function parseXhrErrorMessage(
  responseText: string | null | undefined,
  fallback: string,
): string {
  if (!responseText) {
    return fallback;
  }

  try {
    const payload = JSON.parse(responseText) as {
      error?: { message?: string };
      message?: string;
    };
    return payload.error?.message || payload.message || fallback;
  } catch {
    return fallback;
  }
}

function normalizeUploadProgress(event: ProgressEvent): UploadProgressInfo {
  const lengthComputable = event.lengthComputable && event.total > 0;
  return {
    loadedBytes: event.loaded,
    totalBytes: lengthComputable ? event.total : null,
    percent: lengthComputable
      ? Math.min(100, Math.max(0, (event.loaded / event.total) * 100))
      : null,
    lengthComputable,
  };
}

export class AudioApiClient {
  constructor(private readonly http: ApiHttpClient) {}

  private applyXhrHeaders(
    xhr: XMLHttpRequest,
    headers: HeadersInit | undefined,
  ) {
    if (!headers) {
      return;
    }

    new Headers(headers).forEach((value, key) => {
      xhr.setRequestHeader(key, value);
    });
  }

  private async requestJsonWithUploadProgress<T>(
    url: string,
    init: RequestInit,
    options: SpeechTextJobUploadOptions,
    fallbackMessage: string,
  ): Promise<T> {
    if (typeof XMLHttpRequest === "undefined") {
      const response = await fetch(url, {
        ...init,
        signal: options.signal,
      });

      if (!response.ok) {
        throw await this.http.createError(response, fallbackMessage);
      }

      return response.json();
    }

    return new Promise<T>((resolve, reject) => {
      const xhr = new XMLHttpRequest();
      let settled = false;
      const handleSignalAbort = () => {
        xhr.abort();
      };
      const settle = (callback: () => void) => {
        if (settled) {
          return;
        }
        settled = true;
        options.signal?.removeEventListener("abort", handleSignalAbort);
        callback();
      };

      if (options.signal?.aborted) {
        reject(createAbortError());
        return;
      }

      xhr.open(init.method || "GET", url, true);
      this.applyXhrHeaders(xhr, init.headers);
      xhr.responseType = "text";

      xhr.upload.onprogress = (event) => {
        options.onUploadProgress?.(normalizeUploadProgress(event));
      };

      xhr.onload = () => {
        settle(() => {
          if (xhr.status >= 200 && xhr.status < 300) {
            try {
              resolve(JSON.parse(xhr.responseText) as T);
            } catch {
              reject(new Error(fallbackMessage));
            }
            return;
          }

          reject(
            new Error(parseXhrErrorMessage(xhr.responseText, fallbackMessage)),
          );
        });
      };

      xhr.onerror = () => {
        settle(() => reject(new Error(fallbackMessage)));
      };

      xhr.onabort = () => {
        settle(() => reject(createAbortError()));
      };

      options.signal?.addEventListener("abort", handleSignalAbort, {
        once: true,
      });

      xhr.send((init.body ?? null) as XMLHttpRequestBodyInit | null);
    });
  }

  private dispatchTranscriptionRecordStreamData(
    data: string,
    callbacks: TranscriptionRecordStreamCallbacks,
  ): boolean {
    try {
      const event = JSON.parse(data) as TranscriptionRecordStreamEvent;
      switch (event.event) {
        case "created":
          callbacks.onCreated?.(event.record);
          break;
        case "start":
          callbacks.onStart?.();
          break;
        case "delta":
          callbacks.onDelta?.(event.delta);
          break;
        case "progress":
          callbacks.onProgress?.(event.progress);
          break;
        case "final":
          callbacks.onFinal?.(event.record);
          break;
        case "error":
          callbacks.onError?.(event.error);
          break;
        case "done":
          return true;
      }
    } catch {
      // Skip malformed SSE payloads.
    }

    return false;
  }

  private createTranscriptionRecordStreamWithUploadProgress(
    request: TranscriptionRecordCreateRequest,
    callbacks: TranscriptionRecordStreamCallbacks,
  ): AbortController {
    const abortController = new AbortController();
    const path = this.buildSpeechTextJobPath(
      this.speechTextJobsCollectionPath(),
      "transcription",
    );
    const init = this.buildTranscriptionRecordRequestInit(request, true);

    const xhr = new XMLHttpRequest();
    let responseOffset = 0;
    let responseBuffer = "";
    let completed = false;
    const handleAbort = () => {
      xhr.abort();
    };

    const finish = () => {
      if (completed) {
        return;
      }
      completed = true;
      abortController.signal.removeEventListener("abort", handleAbort);
      callbacks.onDone?.();
    };

    const processLine = (line: string) => {
      if (!line.startsWith("data:")) {
        return;
      }

      const data = line.slice(5).trim();
      if (!data) {
        return;
      }

      if (this.dispatchTranscriptionRecordStreamData(data, callbacks)) {
        finish();
      }
    };

    const consumeResponse = () => {
      if (completed || (xhr.status !== 0 && (xhr.status < 200 || xhr.status >= 300))) {
        return;
      }

      const chunk = xhr.responseText.slice(responseOffset);
      responseOffset = xhr.responseText.length;
      responseBuffer += chunk;
      const lines = responseBuffer.split("\n");
      responseBuffer = lines.pop() || "";
      lines.forEach((line) => processLine(line.trimEnd()));
    };

    xhr.open(init.method || "GET", this.http.url(path), true);
    this.applyXhrHeaders(xhr, init.headers);
    xhr.responseType = "text";

    xhr.upload.onprogress = (event) => {
      callbacks.onUploadProgress?.(normalizeUploadProgress(event));
    };

    xhr.onprogress = consumeResponse;
    xhr.onload = () => {
      if (xhr.status < 200 || xhr.status >= 300) {
        callbacks.onError?.(
          parseXhrErrorMessage(xhr.responseText, "Streaming transcription failed"),
        );
        finish();
        return;
      }

      consumeResponse();
      if (responseBuffer.trim().startsWith("data:")) {
        processLine(responseBuffer.trimEnd());
        responseBuffer = "";
      }
      finish();
    };
    xhr.onerror = () => {
      callbacks.onError?.("Streaming transcription failed");
      finish();
    };
    xhr.onabort = finish;

    abortController.signal.addEventListener("abort", handleAbort, {
      once: true,
    });
    xhr.send((init.body ?? null) as XMLHttpRequestBodyInit | null);

    return abortController;
  }

  async generateTTS(request: TTSRequest): Promise<Blob> {
    const result = await this.generateTTSWithStats(request);
    return result.audioBlob;
  }

  async generateTTSWithStats(request: TTSRequest): Promise<TTSGenerateResult> {
    const response = await fetch(this.http.url("/audio/speech"), {
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
        saved_voice_id: request.saved_voice_id,
        max_tokens: request.max_tokens,
        temperature: request.temperature,
        speed: request.speed,
        response_format: request.format ?? "wav",
      }),
    });

    if (!response.ok) {
      throw await this.http.createError(response, "TTS generation failed");
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
        const response = await fetch(this.http.url("/audio/speech"), {
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
            saved_voice_id: request.saved_voice_id,
            max_tokens: request.max_tokens,
            temperature: request.temperature,
            speed: request.speed,
            response_format: request.format ?? "pcm",
            stream: true,
          }),
          signal: abortController.signal,
        });

        if (!response.ok) {
          callbacks.onError?.(
            (await this.http.createError(response, "TTS streaming failed"))
              .message,
          );
          callbacks.onDone?.();
          return;
        }

        await consumeDataStream(response, (data) => {
          try {
            const event = JSON.parse(data) as TTSStreamEvent;
            switch (event.event) {
              case "start":
              case "audio.started":
                callbacks.onStart?.({
                  requestId: event.request_id,
                  sampleRate: event.sample_rate,
                  audioFormat: event.audio_format,
                });
                break;
              case "chunk":
              case "audio.chunk":
                callbacks.onChunk?.({
                  requestId: event.request_id,
                  sequence: event.sequence,
                  audioBase64: event.audio_base64,
                  sampleCount: event.sample_count,
                  isFinal: event.is_final,
                });
                break;
              case "final":
              case "audio.done":
                callbacks.onFinal?.({
                  generation_time_ms: event.generation_time_ms,
                  audio_duration_secs: event.audio_duration_secs,
                  rtf: event.rtf,
                  tokens_generated: event.tokens_generated,
                });
                break;
              case "error":
              case "audio.failed":
                callbacks.onError?.(
                  streamErrorMessage(event.error, "TTS stream error"),
                );
                break;
              case "done":
                return true;
            }
          } catch {
            // Skip malformed payloads.
          }

          return false;
        });

        callbacks.onDone?.();
      } catch (error) {
        if (!isAbortError(error)) {
          callbacks.onError?.(
            error instanceof Error ? error.message : "TTS stream error",
          );
        }
        callbacks.onDone?.();
      }
    };

    void startStream();
    return abortController;
  }

  private speechHistoryCollectionPath(route: SpeechHistoryRoute): string {
    switch (route) {
      case "text-to-speech":
        return "/text-to-speech";
      case "voice-design":
        return "/voice-designs";
      case "voice-cloning":
        return "/voice-clones";
    }
  }

  private speechHistoryRecordPath(
    route: SpeechHistoryRoute,
    recordId: string,
  ): string {
    return `${this.speechHistoryCollectionPath(route)}/${encodeURIComponent(recordId)}`;
  }

  private speechTextJobsCollectionPath(): string {
    return "/speech-to-text/jobs";
  }

  private speechTextJobPath(recordId: string): string {
    return `${this.speechTextJobsCollectionPath()}/${encodeURIComponent(recordId)}`;
  }

  private buildSpeechTextJobPath(
    path: string,
    jobKind?: SpeechTextJobQueryKind,
  ): string {
    if (!jobKind) {
      return path;
    }
    const suffix = path.includes("?") ? "&" : "?";
    return `${path}${suffix}job_kind=${encodeURIComponent(jobKind)}`;
  }

  private normalizeSpeechTextJobRecord(
    payload: unknown,
    preferredKind?: SpeechTextJobKind,
  ): SpeechTextJob {
    const raw =
      payload && typeof payload === "object"
        ? (payload as Record<string, unknown>)
        : {};
    const candidateKind = raw.kind;
    const explicitKind =
      candidateKind === "transcription" || candidateKind === "diarization"
        ? candidateKind
        : null;
    const inferredKind =
      preferredKind ??
      (typeof raw.transcription === "string" ? "transcription" : "diarization");
    return {
      ...raw,
      kind: explicitKind ?? inferredKind,
    } as SpeechTextJob;
  }

  private resolveSpeechTextPreferredKind(
    jobKind?: SpeechTextJobQueryKind,
  ): SpeechTextJobKind | undefined {
    return jobKind === "transcription" || jobKind === "diarization"
      ? jobKind
      : undefined;
  }

  private stripSpeechTextJobKind<T>(record: SpeechTextJob): T {
    const { kind: _kind, ...rest } = record as SpeechTextJob &
      Record<string, unknown>;
    return rest as T;
  }

  private studioProjectsCollectionPath(): string {
    return "/studio/projects";
  }

  private studioProjectFoldersPath(): string {
    return "/studio/folders";
  }

  private studioProjectPath(projectId: string): string {
    return `${this.studioProjectsCollectionPath()}/${encodeURIComponent(projectId)}`;
  }

  private studioProjectMetaPath(projectId: string): string {
    return `${this.studioProjectPath(projectId)}/meta`;
  }

  private studioProjectSegmentPath(projectId: string, segmentId: string): string {
    return `${this.studioProjectPath(projectId)}/segments/${encodeURIComponent(segmentId)}`;
  }

  private studioProjectSegmentsPath(projectId: string): string {
    return `${this.studioProjectPath(projectId)}/segments`;
  }

  private studioProjectSegmentsReorderPath(projectId: string): string {
    return `${this.studioProjectPath(projectId)}/segments/reorder`;
  }

  private studioProjectSegmentsBulkDeletePath(projectId: string): string {
    return `${this.studioProjectPath(projectId)}/segments/bulk-delete`;
  }

  private studioProjectPronunciationsPath(projectId: string): string {
    return `${this.studioProjectPath(projectId)}/pronunciations`;
  }

  private studioProjectPronunciationPath(
    projectId: string,
    pronunciationId: string,
  ): string {
    return `${this.studioProjectPronunciationsPath(projectId)}/${encodeURIComponent(pronunciationId)}`;
  }

  private studioProjectSnapshotsPath(projectId: string): string {
    return `${this.studioProjectPath(projectId)}/snapshots`;
  }

  private studioProjectSnapshotRestorePath(
    projectId: string,
    snapshotId: string,
  ): string {
    return `${this.studioProjectSnapshotsPath(projectId)}/${encodeURIComponent(snapshotId)}/restore`;
  }

  private studioProjectRenderJobsPath(projectId: string): string {
    return `${this.studioProjectPath(projectId)}/render-jobs`;
  }

  private studioProjectRenderJobPath(projectId: string, jobId: string): string {
    return `${this.studioProjectRenderJobsPath(projectId)}/${encodeURIComponent(jobId)}`;
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
      saved_voice_id: request.saved_voice_id,
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
    const page = await this.listSpeechHistoryRecordPage(route);
    return page.items;
  }

  async listSpeechHistoryRecordPage(
    route: SpeechHistoryRoute,
    query?: CursorPaginationQuery,
  ): Promise<CursorPageResult<SpeechHistoryRecordSummary>> {
    const path = `${this.speechHistoryCollectionPath(route)}${buildCursorQueryString(query)}`;
    const payload = await this.http.request<{
      records: SpeechHistoryRecordSummary[];
      pagination?: {
        next_cursor?: string | null;
        has_more?: boolean;
        limit?: number;
      };
    }>(path);
    const items = payload.records ?? [];
    const fallbackLimit = query?.limit ?? Math.max(items.length, 1);
    return {
      items,
      pagination: normalizeCursorPaginationMeta(payload.pagination, fallbackLimit),
    };
  }

  async getSpeechHistoryRecord(
    route: SpeechHistoryRoute,
    recordId: string,
  ): Promise<SpeechHistoryRecord> {
    return this.http.request(this.speechHistoryRecordPath(route, recordId));
  }

  async createSpeechHistoryRecord(
    route: SpeechHistoryRoute,
    request: SpeechHistoryRecordCreateRequest,
  ): Promise<SpeechHistoryRecord> {
    const response = await fetch(
      this.http.url(this.speechHistoryCollectionPath(route)),
      {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(
          this.buildSpeechHistoryRecordCreateBody(request, false),
        ),
      },
    );

    if (!response.ok) {
      throw await this.http.createError(response, "Speech generation failed");
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
          this.http.url(this.speechHistoryCollectionPath(route)),
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
          callbacks.onError?.(
            (
              await this.http.createError(response, "Speech streaming failed")
            ).message,
          );
          callbacks.onDone?.();
          return;
        }

        await consumeDataStream(response, (data) => {
          try {
            const event = JSON.parse(data) as SpeechHistoryRecordStreamEvent;
            switch (event.event) {
              case "created":
                callbacks.onCreated?.(event.record);
                break;
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
                return true;
            }
          } catch {
            // Skip malformed SSE payloads.
          }

          return false;
        });

        callbacks.onDone?.();
      } catch (error) {
        if (!isAbortError(error)) {
          callbacks.onError?.(
            error instanceof Error ? error.message : "Speech stream error",
          );
        }
        callbacks.onDone?.();
      }
    };

    void startStream();
    return abortController;
  }

  speechHistoryRecordAudioUrl(
    route: SpeechHistoryRoute,
    recordId: string,
    options?: { download?: boolean },
  ): string {
    const base = this.http.url(
      `${this.speechHistoryRecordPath(route, recordId)}/audio`,
    );
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

      if (typeof internals?.invoke === "function" && /^https?:\/\//i.test(url)) {
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
    return this.http.request(this.speechHistoryRecordPath(route, recordId), {
      method: "DELETE",
    });
  }

  async listTextToSpeechRecords(): Promise<SpeechHistoryRecordSummary[]> {
    return this.listSpeechHistoryRecords("text-to-speech");
  }

  async listTextToSpeechRecordPage(
    query?: CursorPaginationQuery,
  ): Promise<CursorPageResult<SpeechHistoryRecordSummary>> {
    return this.listSpeechHistoryRecordPage("text-to-speech", query);
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
    options?: { download?: boolean },
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

  async listVoiceDesignRecordPage(
    query?: CursorPaginationQuery,
  ): Promise<CursorPageResult<SpeechHistoryRecordSummary>> {
    return this.listSpeechHistoryRecordPage("voice-design", query);
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
    options?: { download?: boolean },
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

  async listVoiceCloningRecordPage(
    query?: CursorPaginationQuery,
  ): Promise<CursorPageResult<SpeechHistoryRecordSummary>> {
    return this.listSpeechHistoryRecordPage("voice-cloning", query);
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
    options?: { download?: boolean },
  ): string {
    return this.speechHistoryRecordAudioUrl("voice-cloning", recordId, options);
  }

  async deleteVoiceCloningRecord(
    recordId: string,
  ): Promise<{ id: string; deleted: boolean }> {
    return this.deleteSpeechHistoryRecord("voice-cloning", recordId);
  }

  async listSavedVoices(): Promise<SavedVoiceSummary[]> {
    const page = await this.listSavedVoicePage();
    return page.items;
  }

  async listSavedVoicePage(
    query?: CursorPaginationQuery,
  ): Promise<CursorPageResult<SavedVoiceSummary>> {
    const path = `/voices${buildCursorQueryString(query)}`;
    const payload = await this.http.request<{
      voices: SavedVoiceSummary[];
      pagination?: {
        next_cursor?: string | null;
        has_more?: boolean;
        limit?: number;
      };
    }>(path);
    const items = payload.voices ?? [];
    const fallbackLimit = query?.limit ?? Math.max(items.length, 1);
    return {
      items,
      pagination: normalizeCursorPaginationMeta(payload.pagination, fallbackLimit),
    };
  }

  async getSavedVoice(voiceId: string): Promise<SavedVoice> {
    return this.http.request(`/voices/${encodeURIComponent(voiceId)}`);
  }

  async createSavedVoice(request: SavedVoiceCreateRequest): Promise<SavedVoice> {
    const response = await fetch(this.http.url("/voices"), {
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
      throw await this.http.createError(response, "Failed to save voice");
    }

    return response.json();
  }

  savedVoiceAudioUrl(
    voiceId: string,
    options?: { download?: boolean },
  ): string {
    const base = this.http.url(`/voices/${encodeURIComponent(voiceId)}/audio`);
    if (options?.download) {
      return `${base}?download=true`;
    }
    return base;
  }

  async deleteSavedVoice(
    voiceId: string,
  ): Promise<{ id: string; deleted: boolean }> {
    return this.http.request(`/voices/${encodeURIComponent(voiceId)}`, {
      method: "DELETE",
    });
  }

  async listStudioProjects(): Promise<StudioProjectSummary[]> {
    const page = await this.listStudioProjectPage();
    return page.items;
  }

  async listStudioProjectPage(
    query?: CursorPaginationQuery,
  ): Promise<CursorPageResult<StudioProjectSummary>> {
    const path = `${this.studioProjectsCollectionPath()}${buildCursorQueryString(query)}`;
    const payload = await this.http.request<{
      projects: StudioProjectSummary[];
      pagination?: {
        next_cursor?: string | null;
        has_more?: boolean;
        limit?: number;
      };
    }>(path);
    const items = payload.projects ?? [];
    const fallbackLimit = query?.limit ?? Math.max(items.length, 1);
    return {
      items,
      pagination: normalizeCursorPaginationMeta(payload.pagination, fallbackLimit),
    };
  }

  async createStudioProject(
    request: StudioProjectCreateRequest,
  ): Promise<StudioProjectRecord> {
    return this.http.request(this.studioProjectsCollectionPath(), {
      method: "POST",
      body: JSON.stringify({
        name: request.name,
        source_filename: request.source_filename,
        source_text: request.source_text,
        model_id: request.model_id,
        voice_mode: request.voice_mode,
        speaker: request.speaker,
        saved_voice_id: request.saved_voice_id,
        speed: request.speed,
      }),
    });
  }

  async getStudioProject(projectId: string): Promise<StudioProjectRecord> {
    return this.http.request(this.studioProjectPath(projectId));
  }

  async updateStudioProject(
    projectId: string,
    request: StudioProjectUpdateRequest,
  ): Promise<StudioProjectRecord> {
    return this.http.request(this.studioProjectPath(projectId), {
      method: "PATCH",
      body: JSON.stringify({
        name: request.name,
        model_id: request.model_id,
        voice_mode: request.voice_mode,
        speaker: request.speaker,
        saved_voice_id: request.saved_voice_id,
        speed: request.speed,
      }),
    });
  }

  async updateStudioProjectSegment(
    projectId: string,
    segmentId: string,
    request: StudioProjectSegmentUpdateRequest,
  ): Promise<StudioProjectRecord> {
    return this.http.request(this.studioProjectSegmentPath(projectId, segmentId), {
      method: "PATCH",
      body: JSON.stringify({
        text: request.text,
        model_id: request.model_id,
        voice_mode: request.voice_mode,
        speaker: request.speaker,
        saved_voice_id: request.saved_voice_id,
      }),
    });
  }

  async createStudioProjectSegment(
    projectId: string,
    request: StudioProjectSegmentCreateRequest,
  ): Promise<StudioProjectRecord> {
    return this.http.request(this.studioProjectSegmentsPath(projectId), {
      method: "POST",
      body: JSON.stringify({
        text: request.text,
        after_segment_id: request.after_segment_id,
      }),
    });
  }

  async splitStudioProjectSegment(
    projectId: string,
    segmentId: string,
    request: StudioProjectSegmentSplitRequest,
  ): Promise<StudioProjectRecord> {
    return this.http.request(
      `${this.studioProjectSegmentPath(projectId, segmentId)}/split`,
      {
        method: "POST",
        body: JSON.stringify({
          before_text: request.before_text,
          after_text: request.after_text,
        }),
      },
    );
  }

  async mergeStudioProjectSegmentWithNext(
    projectId: string,
    segmentId: string,
  ): Promise<StudioProjectRecord> {
    return this.http.request(
      `${this.studioProjectSegmentPath(projectId, segmentId)}/merge-next`,
      {
        method: "POST",
      },
    );
  }

  async reorderStudioProjectSegments(
    projectId: string,
    request: StudioProjectSegmentReorderRequest,
  ): Promise<StudioProjectRecord> {
    return this.http.request(this.studioProjectSegmentsReorderPath(projectId), {
      method: "PATCH",
      body: JSON.stringify({
        ordered_segment_ids: request.ordered_segment_ids,
      }),
    });
  }

  async bulkDeleteStudioProjectSegments(
    projectId: string,
    request: StudioProjectSegmentBulkDeleteRequest,
  ): Promise<StudioProjectRecord> {
    return this.http.request(this.studioProjectSegmentsBulkDeletePath(projectId), {
      method: "POST",
      body: JSON.stringify({
        segment_ids: request.segment_ids,
      }),
    });
  }

  async deleteStudioProjectSegment(
    projectId: string,
    segmentId: string,
  ): Promise<StudioProjectRecord> {
    return this.http.request(this.studioProjectSegmentPath(projectId, segmentId), {
      method: "DELETE",
    });
  }

  async renderStudioProjectSegment(
    projectId: string,
    segmentId: string,
  ): Promise<StudioProjectRecord> {
    return this.http.request(
      `${this.studioProjectSegmentPath(projectId, segmentId)}/render`,
      {
        method: "POST",
      },
    );
  }

  studioProjectAudioUrl(
    projectId: string,
    options?: {
      download?: boolean;
      format?: "wav" | "raw_i16" | "raw_f32";
      segment_ids?: string[];
    },
  ): string {
    const base = this.http.url(`${this.studioProjectPath(projectId)}/audio`);
    const params = new URLSearchParams();
    if (options?.download) {
      params.set("download", "true");
    }
    if (options?.format) {
      params.set("format", options.format);
    }
    if (options?.segment_ids && options.segment_ids.length > 0) {
      params.set("segment_ids", options.segment_ids.join(","));
    }
    const query = params.toString();
    return query ? `${base}?${query}` : base;
  }

  async deleteStudioProject(
    projectId: string,
  ): Promise<{ id: string; deleted: boolean }> {
    return this.http.request(this.studioProjectPath(projectId), {
      method: "DELETE",
    });
  }

  async listStudioProjectFolders(): Promise<StudioProjectFolderRecord[]> {
    const payload = await this.http.request<{
      folders: StudioProjectFolderRecord[];
    }>(this.studioProjectFoldersPath());
    return payload.folders ?? [];
  }

  async createStudioProjectFolder(
    request: StudioProjectFolderCreateRequest,
  ): Promise<StudioProjectFolderRecord> {
    return this.http.request(this.studioProjectFoldersPath(), {
      method: "POST",
      body: JSON.stringify({
        name: request.name,
        parent_id: request.parent_id,
        sort_order: request.sort_order,
      }),
    });
  }

  async getStudioProjectMeta(projectId: string): Promise<StudioProjectMetaRecord> {
    return this.http.request(this.studioProjectMetaPath(projectId));
  }

  async updateStudioProjectMeta(
    projectId: string,
    request: StudioProjectMetaUpdateRequest,
  ): Promise<StudioProjectMetaRecord> {
    return this.http.request(this.studioProjectMetaPath(projectId), {
      method: "PATCH",
      body: JSON.stringify({
        folder_id: request.folder_id,
        tags: request.tags,
        default_export_format: request.default_export_format,
        last_render_job_id: request.last_render_job_id,
        last_rendered_at: request.last_rendered_at,
      }),
    });
  }

  async listStudioProjectPronunciations(
    projectId: string,
  ): Promise<StudioProjectPronunciationRecord[]> {
    const payload = await this.http.request<{
      entries: StudioProjectPronunciationRecord[];
    }>(this.studioProjectPronunciationsPath(projectId));
    return payload.entries ?? [];
  }

  async createStudioProjectPronunciation(
    projectId: string,
    request: StudioProjectPronunciationCreateRequest,
  ): Promise<StudioProjectPronunciationRecord> {
    return this.http.request(this.studioProjectPronunciationsPath(projectId), {
      method: "POST",
      body: JSON.stringify({
        source_text: request.source_text,
        replacement_text: request.replacement_text,
        locale: request.locale,
      }),
    });
  }

  async deleteStudioProjectPronunciation(
    projectId: string,
    pronunciationId: string,
  ): Promise<{ id: string; deleted: boolean }> {
    return this.http.request(
      this.studioProjectPronunciationPath(projectId, pronunciationId),
      {
        method: "DELETE",
      },
    );
  }

  async listStudioProjectSnapshots(
    projectId: string,
  ): Promise<StudioProjectSnapshotRecord[]> {
    const payload = await this.http.request<{
      snapshots: StudioProjectSnapshotRecord[];
    }>(this.studioProjectSnapshotsPath(projectId));
    return payload.snapshots ?? [];
  }

  async createStudioProjectSnapshot(
    projectId: string,
    request?: StudioProjectSnapshotCreateRequest,
  ): Promise<StudioProjectSnapshotRecord> {
    return this.http.request(this.studioProjectSnapshotsPath(projectId), {
      method: "POST",
      body: JSON.stringify({
        label: request?.label,
      }),
    });
  }

  async restoreStudioProjectSnapshot(
    projectId: string,
    snapshotId: string,
  ): Promise<StudioProjectRecord> {
    return this.http.request(
      this.studioProjectSnapshotRestorePath(projectId, snapshotId),
      {
        method: "POST",
      },
    );
  }

  async listStudioProjectRenderJobs(
    projectId: string,
  ): Promise<StudioProjectRenderJobRecord[]> {
    const payload = await this.http.request<{
      jobs: StudioProjectRenderJobRecord[];
    }>(this.studioProjectRenderJobsPath(projectId));
    return payload.jobs ?? [];
  }

  async createStudioProjectRenderJob(
    projectId: string,
    request?: StudioProjectRenderJobCreateRequest,
  ): Promise<StudioProjectRenderJobRecord> {
    return this.http.request(this.studioProjectRenderJobsPath(projectId), {
      method: "POST",
      body: JSON.stringify({
        queued_segment_ids: request?.queued_segment_ids ?? [],
      }),
    });
  }

  async updateStudioProjectRenderJob(
    projectId: string,
    jobId: string,
    request: StudioProjectRenderJobUpdateRequest,
  ): Promise<StudioProjectRenderJobRecord> {
    return this.http.request(this.studioProjectRenderJobPath(projectId, jobId), {
      method: "PATCH",
      body: JSON.stringify({
        status: request.status,
        error_message: request.error_message,
      }),
    });
  }

  async listSpeechTextJobs(
    query?: SpeechTextJobPageQuery,
  ): Promise<SpeechTextJobSummary[]> {
    const page = await this.listSpeechTextJobPage(query);
    return page.items;
  }

  async listSpeechTextJobPage(
    query?: SpeechTextJobPageQuery,
  ): Promise<CursorPageResult<SpeechTextJobSummary>> {
    const cursorQuery = buildCursorQueryString({
      limit: query?.limit,
      cursor: query?.cursor,
    });
    const path = this.buildSpeechTextJobPath(
      `${this.speechTextJobsCollectionPath()}${cursorQuery}`,
      query?.job_kind,
    );
    const payload = await this.http.request<{
      records: SpeechTextJobSummary[];
      pagination?: {
        next_cursor?: string | null;
        has_more?: boolean;
        limit?: number;
      };
    }>(path);
    const items = payload.records ?? [];
    const fallbackLimit = query?.limit ?? Math.max(items.length, 1);
    return {
      items,
      pagination: normalizeCursorPaginationMeta(payload.pagination, fallbackLimit),
    };
  }

  async getSpeechTextJob(
    recordId: string,
    options?: { job_kind?: SpeechTextJobQueryKind },
  ): Promise<SpeechTextJob> {
    const path = this.buildSpeechTextJobPath(
      this.speechTextJobPath(recordId),
      options?.job_kind,
    );
    const payload = await this.http.request(path);
    const preferredKind = this.resolveSpeechTextPreferredKind(options?.job_kind);
    return this.normalizeSpeechTextJobRecord(payload, preferredKind);
  }

  async createSpeechTextJob(
    request: SpeechTextJobCreateRequest,
    options: SpeechTextJobUploadOptions = {},
  ): Promise<SpeechTextJob> {
    if (request.kind === "transcription" && request.stream) {
      throw new Error(
        "Streaming transcription jobs must use createTranscriptionRecordStream.",
      );
    }

    const path = this.buildSpeechTextJobPath(
      this.speechTextJobsCollectionPath(),
      request.kind,
    );
    const init = this.buildSpeechTextJobCreateRequestInit(request);

    if (options.onUploadProgress || options.signal) {
      const payload = await this.requestJsonWithUploadProgress<unknown>(
        this.http.url(path),
        init,
        options,
        "Speech text job failed",
      );
      return this.normalizeSpeechTextJobRecord(payload, request.kind);
    }

    const response = await fetch(this.http.url(path), init);

    if (!response.ok) {
      throw await this.http.createError(response, "Speech text job failed");
    }

    const payload = await response.json();
    return this.normalizeSpeechTextJobRecord(payload, request.kind);
  }

  async updateSpeechTextJob(
    recordId: string,
    request: DiarizationRecordUpdateRequest,
    options?: { job_kind?: SpeechTextJobQueryKind },
  ): Promise<SpeechTextJob> {
    const path = this.buildSpeechTextJobPath(
      this.speechTextJobPath(recordId),
      options?.job_kind,
    );
    const body = JSON.stringify({
      speaker_name_overrides: request.speaker_name_overrides,
    });

    const sendUpdate = (method: "PATCH" | "PUT") =>
      fetch(this.http.url(path), {
        method,
        headers: {
          "Content-Type": "application/json",
        },
        body,
      });

    let response = await sendUpdate("PATCH");

    // Some embedded-webview stacks still reject PATCH for local API calls.
    if (response.status === 405) {
      response = await sendUpdate("PUT");
    }

    if (!response.ok) {
      throw await this.http.createError(
        response,
        "Failed to save speaker corrections",
      );
    }

    const payload = await response.json();
    const preferredKind = this.resolveSpeechTextPreferredKind(options?.job_kind);
    return this.normalizeSpeechTextJobRecord(payload, preferredKind);
  }

  async rerunSpeechTextJob(
    recordId: string,
    request: DiarizationRecordRerunRequest,
    options?: { job_kind?: SpeechTextJobQueryKind },
  ): Promise<SpeechTextJob> {
    const path = this.buildSpeechTextJobPath(
      `${this.speechTextJobPath(recordId)}/reruns`,
      options?.job_kind,
    );
    const payload = await this.http.request(path, {
      method: "POST",
      body: JSON.stringify({
        min_speakers: request.min_speakers,
        max_speakers: request.max_speakers,
        min_speech_duration_ms: request.min_speech_duration_ms,
        min_silence_duration_ms: request.min_silence_duration_ms,
        enable_llm_refinement: request.enable_llm_refinement,
      }),
    });
    const preferredKind = this.resolveSpeechTextPreferredKind(options?.job_kind);
    return this.normalizeSpeechTextJobRecord(payload, preferredKind);
  }

  async cancelSpeechTextJob(
    recordId: string,
    options?: { job_kind?: SpeechTextJobQueryKind },
  ): Promise<SpeechTextJob> {
    const path = this.buildSpeechTextJobPath(
      `${this.speechTextJobPath(recordId)}/cancel`,
      options?.job_kind,
    );
    const payload = await this.http.request(path, {
      method: "POST",
    });
    const preferredKind = this.resolveSpeechTextPreferredKind(options?.job_kind);
    return this.normalizeSpeechTextJobRecord(payload, preferredKind);
  }

  speechTextJobAudioUrl(
    recordId: string,
    options?: { job_kind?: SpeechTextJobQueryKind },
  ): string {
    const path = this.buildSpeechTextJobPath(
      `${this.speechTextJobPath(recordId)}/audio`,
      options?.job_kind,
    );
    return this.http.url(path);
  }

  async regenerateSpeechTextJobSummary(
    recordId: string,
    options?: { job_kind?: SpeechTextJobQueryKind },
  ): Promise<SpeechTextJob> {
    const path = this.buildSpeechTextJobPath(
      `${this.speechTextJobPath(recordId)}/summary/regenerate`,
      options?.job_kind,
    );
    const payload = await this.http.request(path, {
      method: "POST",
    });
    const preferredKind = this.resolveSpeechTextPreferredKind(options?.job_kind);
    return this.normalizeSpeechTextJobRecord(payload, preferredKind);
  }

  async deleteSpeechTextJob(
    recordId: string,
    options?: { job_kind?: SpeechTextJobQueryKind },
  ): Promise<{ id: string; deleted: boolean }> {
    const path = this.buildSpeechTextJobPath(
      this.speechTextJobPath(recordId),
      options?.job_kind,
    );
    return this.http.request(path, {
      method: "DELETE",
    });
  }

  async listTranscriptionRecords(): Promise<TranscriptionRecordSummary[]> {
    const page = await this.listTranscriptionRecordPage();
    return page.items;
  }

  async listTranscriptionRecordPage(
    query?: CursorPaginationQuery,
  ): Promise<CursorPageResult<TranscriptionRecordSummary>> {
    const page = await this.listSpeechTextJobPage({
      ...query,
      job_kind: "transcription",
    });
    return {
      items: page.items as TranscriptionRecordSummary[],
      pagination: page.pagination,
    };
  }

  async getTranscriptionRecord(recordId: string): Promise<TranscriptionRecord> {
    const record = await this.getSpeechTextJob(recordId, {
      job_kind: "transcription",
    });
    return this.stripSpeechTextJobKind<TranscriptionRecord>(record);
  }

  async createTranscriptionRecord(
    request: TranscriptionRecordCreateRequest,
    options: SpeechTextJobUploadOptions = {},
  ): Promise<TranscriptionRecord> {
    const record = await this.createSpeechTextJob(
      {
        ...request,
        kind: "transcription",
        stream: false,
      },
      options,
    );
    return this.stripSpeechTextJobKind<TranscriptionRecord>(record);
  }

  createTranscriptionRecordStream(
    request: TranscriptionRecordCreateRequest,
    callbacks: TranscriptionRecordStreamCallbacks,
  ): AbortController {
    if (callbacks.onUploadProgress) {
      return this.createTranscriptionRecordStreamWithUploadProgress(
        request,
        callbacks,
      );
    }

    const abortController = new AbortController();

    const startStream = async () => {
      try {
        const path = this.buildSpeechTextJobPath(
          this.speechTextJobsCollectionPath(),
          "transcription",
        );
        const response = await fetch(this.http.url(path), {
          ...this.buildTranscriptionRecordRequestInit(request, true),
          signal: abortController.signal,
        });

        if (!response.ok) {
          callbacks.onError?.(
            (
              await this.http.createError(
                response,
                "Streaming transcription failed",
              )
            ).message,
          );
          callbacks.onDone?.();
          return;
        }

        await consumeDataStream(response, (data) => {
          return this.dispatchTranscriptionRecordStreamData(data, callbacks);
        });

        callbacks.onDone?.();
      } catch (error) {
        if (!isAbortError(error)) {
          callbacks.onError?.(
            error instanceof Error
              ? error.message
              : "Transcription stream error",
          );
        }
        callbacks.onDone?.();
      }
    };

    void startStream();
    return abortController;
  }

  transcriptionRecordAudioUrl(recordId: string): string {
    return this.speechTextJobAudioUrl(recordId, {
      job_kind: "transcription",
    });
  }

  async regenerateTranscriptionSummary(
    recordId: string,
  ): Promise<TranscriptionRecord> {
    const record = await this.regenerateSpeechTextJobSummary(recordId, {
      job_kind: "transcription",
    });
    return this.stripSpeechTextJobKind<TranscriptionRecord>(record);
  }

  async deleteTranscriptionRecord(
    recordId: string,
  ): Promise<{ id: string; deleted: boolean }> {
    return this.deleteSpeechTextJob(recordId, {
      job_kind: "transcription",
    });
  }

  async asrStatus(): Promise<ASRStatusResponse> {
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
      this.http.url("/audio/transcriptions"),
      this.buildAsrRequestInit(request, "verbose_json", false),
    );

    if (!response.ok) {
      throw await this.http.createError(response, "Transcription failed");
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
    const page = await this.listDiarizationRecordPage();
    return page.items;
  }

  async listDiarizationRecordPage(
    query?: CursorPaginationQuery,
  ): Promise<CursorPageResult<DiarizationRecordSummary>> {
    const page = await this.listSpeechTextJobPage({
      ...query,
      job_kind: "diarization",
    });
    return {
      items: page.items as DiarizationRecordSummary[],
      pagination: page.pagination,
    };
  }

  async getDiarizationRecord(recordId: string): Promise<DiarizationRecord> {
    const record = await this.getSpeechTextJob(recordId, {
      job_kind: "diarization",
    });
    return this.stripSpeechTextJobKind<DiarizationRecord>(record);
  }

  async updateDiarizationRecord(
    recordId: string,
    request: DiarizationRecordUpdateRequest,
  ): Promise<DiarizationRecord> {
    const record = await this.updateSpeechTextJob(recordId, request, {
      job_kind: "diarization",
    });
    return this.stripSpeechTextJobKind<DiarizationRecord>(record);
  }

  async rerunDiarizationRecord(
    recordId: string,
    request: DiarizationRecordRerunRequest,
  ): Promise<DiarizationRecord> {
    const record = await this.rerunSpeechTextJob(recordId, request, {
      job_kind: "diarization",
    });
    return this.stripSpeechTextJobKind<DiarizationRecord>(record);
  }

  async cancelDiarizationRecord(recordId: string): Promise<DiarizationRecord> {
    const record = await this.cancelSpeechTextJob(recordId, {
      job_kind: "diarization",
    });
    return this.stripSpeechTextJobKind<DiarizationRecord>(record);
  }

  async createDiarizationRecord(
    request: DiarizationRecordCreateRequest,
    options: SpeechTextJobUploadOptions = {},
  ): Promise<DiarizationRecord> {
    const record = await this.createSpeechTextJob(
      {
        ...request,
        kind: "diarization",
      },
      options,
    );
    return this.stripSpeechTextJobKind<DiarizationRecord>(record);
  }

  diarizationRecordAudioUrl(recordId: string): string {
    return this.speechTextJobAudioUrl(recordId, {
      job_kind: "diarization",
    });
  }

  async regenerateDiarizationSummary(
    recordId: string,
  ): Promise<DiarizationRecord> {
    const record = await this.regenerateSpeechTextJobSummary(recordId, {
      job_kind: "diarization",
    });
    return this.stripSpeechTextJobKind<DiarizationRecord>(record);
  }

  async deleteDiarizationRecord(
    recordId: string,
  ): Promise<{ id: string; deleted: boolean }> {
    return this.deleteSpeechTextJob(recordId, {
      job_kind: "diarization",
    });
  }

  asrTranscribeStream(
    request: ASRTranscribeRequest,
    callbacks: ASRStreamCallbacks,
  ): AbortController {
    const abortController = new AbortController();

    const startStream = async () => {
      try {
        const response = await fetch(this.http.url("/audio/transcriptions"), {
          ...this.buildAsrRequestInit(request, "json", true),
          signal: abortController.signal,
        });

        if (!response.ok) {
          callbacks.onError?.(
            (
              await this.http.createError(
                response,
                "Streaming transcription failed",
              )
            ).message,
          );
          callbacks.onDone?.();
          return;
        }

        let assembledText = "";

        await consumeDataStream(response, (data) => {
          if (data === "[DONE]") {
            return true;
          }

          try {
            const event = JSON.parse(data) as ASRStreamEvent;
            switch (normalizeAsrStreamEventName(event)) {
              case "start":
                callbacks.onStart?.(
                  "audio_duration_secs" in event
                    ? event.audio_duration_secs
                    : null,
                );
                break;
              case "delta":
                {
                  const deltaEvent = event as { delta: string };
                  assembledText += deltaEvent.delta;
                  callbacks.onDelta?.(deltaEvent.delta);
                }
                callbacks.onPartial?.(assembledText);
                break;
              case "partial":
                if ("type" in event) {
                  break;
                }
                {
                  const partialEvent = event as { text: string };
                  if (partialEvent.text.startsWith(assembledText)) {
                    const delta = partialEvent.text.slice(assembledText.length);
                    if (delta) {
                      callbacks.onDelta?.(delta);
                    }
                  } else if (partialEvent.text !== assembledText) {
                    callbacks.onDelta?.(partialEvent.text);
                  }
                  assembledText = partialEvent.text;
                  callbacks.onPartial?.(partialEvent.text);
                }
                break;
              case "final":
                {
                  const finalEvent = event as {
                    text: string;
                    language: string | null;
                    audio_duration_secs: number | null;
                  };
                  assembledText = finalEvent.text;
                  callbacks.onFinal?.(
                    finalEvent.text,
                    finalEvent.language,
                    finalEvent.audio_duration_secs,
                  );
                }
                break;
              case "error":
                callbacks.onError?.(
                  streamErrorMessage(
                    (event as { error?: string | { message?: string } }).error,
                    "Transcription stream error",
                  ),
                );
                break;
              case "done":
                return true;
            }
          } catch {
            // Skip malformed SSE payloads.
          }

          return false;
        });

        callbacks.onDone?.();
      } catch (error) {
        if (!isAbortError(error)) {
          callbacks.onError?.(
            error instanceof Error ? error.message : "Stream error",
          );
        }
        callbacks.onDone?.();
      }
    };

    void startStream();
    return abortController;
  }

  private buildSpeechTextJobCreateRequestInit(
    request: SpeechTextJobCreateRequest,
  ): RequestInit {
    if (request.kind === "transcription") {
      return this.buildTranscriptionRecordRequestInit(request, Boolean(request.stream));
    }
    return this.buildDiarizationRecordRequestInit(request);
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
      if (request.model_id) {
        form.append("model", request.model_id);
      }
      if (request.language) {
        form.append("language", request.language);
      }
      if (request.prompt?.trim()) {
        form.append("prompt", request.prompt.trim());
      }
      if (typeof request.max_tokens === "number") {
        form.append("max_tokens", String(request.max_tokens));
      }
      if (responseFormat === "verbose_json") {
        for (const granularity of request.timestamp_granularities ?? []) {
          form.append("timestamp_granularities[]", granularity);
        }
      }
      form.append("response_format", responseFormat);
      if (stream) {
        form.append("stream", "true");
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
        prompt: request.prompt?.trim() || undefined,
        max_tokens: request.max_tokens,
        timestamp_granularities:
          responseFormat === "verbose_json"
            ? request.timestamp_granularities
            : undefined,
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
      assertFirstPartyAudioUploadWithinLimit(request.audio_file);
      const form = new FormData();
      form.append(
        "file",
        request.audio_file,
        request.audio_filename || "audio.wav",
      );
      if (request.model_id) {
        form.append("model", request.model_id);
      }
      if (request.aligner_model_id) {
        form.append("aligner_model", request.aligner_model_id);
      }
      if (request.language) {
        form.append("language", request.language);
      }
      if (request.include_timestamps) {
        form.append("include_timestamps", "true");
      }
      if (request.generate_summary) {
        form.append("generate_summary", "true");
      }
      if (stream) {
        form.append("stream", "true");
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
        aligner_model: request.aligner_model_id,
        language: request.language,
        include_timestamps: Boolean(request.include_timestamps),
        generate_summary: Boolean(request.generate_summary),
        stream,
      }),
    };
  }

  private buildDiarizationRecordRequestInit(
    request: DiarizationRecordCreateRequest,
  ): RequestInit {
    const enableLlmRefinement = request.enable_llm_refinement ?? true;

    if (request.audio_file) {
      assertFirstPartyAudioUploadWithinLimit(request.audio_file);
      const form = new FormData();
      form.append(
        "file",
        request.audio_file,
        request.audio_filename || "audio.wav",
      );
      if (request.model_id) {
        form.append("model", request.model_id);
      }
      if (request.asr_model_id) {
        form.append("asr_model", request.asr_model_id);
      }
      if (request.aligner_model_id) {
        form.append("aligner_model", request.aligner_model_id);
      }
      if (request.llm_model_id) {
        form.append("llm_model", request.llm_model_id);
      }
      form.append(
        "enable_llm_refinement",
        enableLlmRefinement ? "true" : "false",
      );
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

function assertFirstPartyAudioUploadWithinLimit(file: Blob): void {
  if (file.size > FIRST_PARTY_AUDIO_UPLOAD_LIMIT_BYTES) {
    throw new Error("Uploaded audio is too large for this server.");
  }
}
