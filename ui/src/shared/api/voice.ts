import { ApiHttpClient } from "@/shared/api/http";

export interface VoiceProfile {
  id: string;
  name: string;
  system_prompt: string;
  observational_memory_enabled: boolean;
  created_at: number;
  updated_at: number;
  default_system_prompt: string;
}

export interface VoiceProfileUpdateRequest {
  name?: string;
  system_prompt?: string;
  observational_memory_enabled?: boolean;
}

export class VoiceApiClient {
  constructor(private readonly http: ApiHttpClient) {}

  async getVoiceProfile(): Promise<VoiceProfile> {
    return this.http.request("/voice/profile");
  }

  async updateVoiceProfile(
    request: VoiceProfileUpdateRequest,
  ): Promise<VoiceProfile> {
    return this.http.request("/voice/profile", {
      method: "PATCH",
      body: JSON.stringify({
        name: request.name,
        system_prompt: request.system_prompt,
        observational_memory_enabled: request.observational_memory_enabled,
      }),
    });
  }
}
