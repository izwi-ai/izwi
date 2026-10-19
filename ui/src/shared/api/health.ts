import { ApiHttpClient } from "@/shared/api/http";

export interface RuntimeMemory {
  /** Null when the OS probe is unavailable. */
  system_total_bytes: number | null;
  system_available_bytes: number | null;
  /** Memory reserved for loaded models and running work. */
  reserved_bytes: number;
}

/** The subset of `/v1/health` the UI reads. */
export interface HealthResponse {
  status: string;
  version: string;
  runtime: {
    selected_backend: string;
    memory: RuntimeMemory;
  };
}

export class HealthApiClient {
  constructor(private readonly http: ApiHttpClient) {}

  async getHealth(): Promise<HealthResponse> {
    return this.http.request("/health");
  }
}
