import { ApiHttpClient } from "@/shared/api/http";

export interface OnboardingStateResponse {
  completed: boolean;
  completed_at: number | null;
  analytics_opt_in: boolean;
}

export class OnboardingApiClient {
  constructor(private readonly http: ApiHttpClient) {}

  async getOnboardingState(): Promise<OnboardingStateResponse> {
    return this.http.request("/onboarding");
  }

  async completeOnboarding(): Promise<OnboardingStateResponse> {
    return this.http.request("/onboarding/complete", {
      method: "POST",
    });
  }

  /** Marks onboarding not completed, so first run shows again. */
  async resetOnboarding(): Promise<OnboardingStateResponse> {
    return this.http.request("/onboarding/reset", {
      method: "POST",
    });
  }
}
