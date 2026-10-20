import { ApiHttpClient } from "@/shared/api/http";

export interface SpeechModelCapabilities {
  supports_builtin_voices: boolean;
  built_in_voice_count: number | null;
  supports_reference_voice: boolean;
  supports_voice_description: boolean;
  supports_streaming: boolean;
  supports_speed_control: boolean;
  supports_auto_long_form: boolean;
}

export type ChatReasoningEffort = "xhigh" | "medium" | "low";

export interface ChatModelCapabilities {
  supports_thinking: boolean;
  /** False when the model always reasons (LFM2.5 Thinking): no switch. */
  thinking_toggleable?: boolean;
  default_thinking_enabled: boolean;
  reasoning_efforts: ChatReasoningEffort[];
  default_reasoning_effort: ChatReasoningEffort | null;
  supports_preserve_thinking: boolean;
}

export interface ModelRouteCapabilities {
  openai_chat_completions: boolean;
  openai_responses: boolean;
  openai_audio_speech: boolean;
  openai_audio_transcriptions: boolean;
  speech_to_text_jobs: boolean;
  speech_to_text_realtime: boolean;
  diarization_records: boolean;
  text_to_speech_records: boolean;
  voice_design_records: boolean;
  voice_clone_records: boolean;
  saved_voice_reuse: boolean;
  studio_projects: boolean;
  voice_realtime_text_model: boolean;
  voice_realtime_modular_asr: boolean;
  voice_realtime_modular_tts: boolean;
  voice_realtime_unified: boolean;
  forced_alignment: boolean;
  tokenizer: boolean;
}

export interface ModelInfo {
  variant: string;
  /** Catalog display name, e.g. "Qwen3-TTS 0.6B CustomVoice". */
  display_name?: string;
  /** Primary task: tts, asr, diarization, chat, forced_align, audio_chat, tokenizer. */
  task?: string;
  /** Expected download size, for models not yet on disk. */
  estimated_size_bytes?: number;
  /** Catalog estimate of memory needed to run the model, in GB. */
  memory_required_gb?: number;
  /** Memory the model needs to load, from the catalog. */
  estimated_memory_bytes?: number;
  /** Whether the model can load on this device at all. */
  device_fit?: "fits" | "too_large" | "unknown";
  /** Free disk a download needs (0 once downloaded). */
  disk_required_bytes?: number;
  enabled?: boolean;
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
  /** "Keep loaded": other loads cannot evict it until it is unloaded. */
  pinned?: boolean;
  modalities?: string[];
  route_capabilities?: ModelRouteCapabilities;
  chat_capabilities?: ChatModelCapabilities | null;
  speech_capabilities?: SpeechModelCapabilities | null;
}

export interface ModelResidencySummary {
  resident_count: number;
  max_loaded_models: number | null;
  model_keep_alive_secs: number;
}

/** A download of a variant whose support was removed, still on disk. */
export interface RemovedModelInfo {
  variant: string;
  display_name: string;
  status: "removed";
  local_path: string;
  size_bytes: number;
  /** The supported variant to use instead. */
  replacement: string;
  reason: string;
}

/** Where models are stored and the room left on that volume. */
export interface ModelStorage {
  models_dir: string;
  free_bytes: number | null;
  total_bytes: number | null;
  models_bytes: number;
}

export interface ModelActionResponse {
  status: string;
  message: string;
  /** Why the action may be slow (a load larger than free memory). */
  warning?: string;
}

export interface ModelsResponse {
  models: ModelInfo[];
  removed_models?: RemovedModelInfo[];
  residency?: ModelResidencySummary;
}

export class ModelApiClient {
  constructor(private readonly http: ApiHttpClient) {}

  async listModels(): Promise<ModelsResponse> {
    return this.http.request("/admin/models");
  }

  async getModelInfo(variant: string): Promise<ModelInfo> {
    return this.http.request(`/admin/models/${variant}`);
  }

  async downloadModel(
    variant: string,
  ): Promise<{ status: string; message: string }> {
    return this.http.request(`/admin/models/${variant}/download`, {
      method: "POST",
    });
  }

  /**
   * Loads a model. The app loads without a pin: the model stays loaded
   * while idle, but a load that needs its room evicts it ("Keep loaded"
   * pins it).
   */
  async loadModel(
    variant: string,
    options?: { signal?: AbortSignal },
  ): Promise<ModelActionResponse> {
    return this.http.request(`/admin/models/${variant}/load?pin=false`, {
      method: "POST",
      signal: options?.signal,
    });
  }

  async unloadModel(variant: string): Promise<ModelActionResponse> {
    return this.http.request(`/admin/models/${variant}/unload`, {
      method: "POST",
    });
  }

  /** Keep a loaded model loaded until it is unloaded. */
  async pinModel(variant: string): Promise<ModelActionResponse> {
    return this.http.request(`/admin/models/${variant}/pin`, {
      method: "POST",
    });
  }

  async unpinModel(variant: string): Promise<ModelActionResponse> {
    return this.http.request(`/admin/models/${variant}/unpin`, {
      method: "POST",
    });
  }

  async getModelStorage(): Promise<ModelStorage> {
    return this.http.request("/admin/storage");
  }

  async deleteModel(
    variant: string,
  ): Promise<{ status: string; message: string }> {
    return this.http.request(`/admin/models/${variant}`, {
      method: "DELETE",
    });
  }

  async cancelDownload(
    variant: string,
  ): Promise<{ status: string; message: string }> {
    return this.http.request(`/admin/models/${variant}/download/cancel`, {
      method: "POST",
    });
  }
}
