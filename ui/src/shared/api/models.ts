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
  speech_capabilities?: SpeechModelCapabilities | null;
}

export interface ModelsResponse {
  models: ModelInfo[];
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

  async loadModel(
    variant: string,
  ): Promise<{ status: string; message: string }> {
    return this.http.request(`/admin/models/${variant}/load`, {
      method: "POST",
    });
  }

  async unloadModel(
    variant: string,
  ): Promise<{ status: string; message: string }> {
    return this.http.request(`/admin/models/${variant}/unload`, {
      method: "POST",
    });
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
