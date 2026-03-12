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

export interface VoiceObservation {
  id: string;
  profile_id: string;
  category: string;
  summary: string;
  confidence: number;
  source_turn_id: string | null;
  source_user_text: string | null;
  source_assistant_text: string | null;
  times_seen: number;
  created_at: number;
  updated_at: number;
  forgotten_at: number | null;
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

  async listVoiceObservations(limit = 25): Promise<VoiceObservation[]> {
    return this.http.request(`/voice/observations?limit=${limit}`);
  }

  async deleteVoiceObservation(
    observationId: string,
  ): Promise<{ id: string; deleted: boolean }> {
    return this.http.request(
      `/voice/observations/${encodeURIComponent(observationId)}`,
      {
        method: "DELETE",
      },
    );
  }

  async clearVoiceObservations(): Promise<{ cleared: number }> {
    return this.http.request("/voice/observations", {
      method: "DELETE",
    });
  }
}
