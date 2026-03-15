import { ApiHttpClient } from "@/shared/api/http";

export interface OnboardingStateResponse {
  completed: boolean;
  completed_at: number | null;
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
}
