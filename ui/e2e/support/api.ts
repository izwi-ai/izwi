import type { Page } from "@playwright/test";

export const READY_MODELS = [
  {
    variant: "Qwen3-1.7B-GGUF",
    status: "ready",
    local_path: "/tmp/qwen.gguf",
    size_bytes: 1_000_000,
    download_progress: null,
    error_message: null,
    chat_capabilities: {
      supports_thinking: true,
      supports_streaming: true,
    },
  },
  {
    variant: "Kokoro-82M",
    status: "ready",
    local_path: "/tmp/kokoro",
    size_bytes: 1_000_000,
    download_progress: null,
    error_message: null,
    speech_capabilities: {
      supports_builtin_voices: true,
      built_in_voice_count: 1,
      supports_reference_voice: false,
      supports_voice_description: false,
      supports_streaming: true,
      supports_speed_control: true,
      supports_auto_long_form: true,
    },
  },
  {
    variant: "Parakeet-TDT-0.6B-v3",
    status: "ready",
    local_path: "/tmp/parakeet",
    size_bytes: 1_000_000,
    download_progress: null,
    error_message: null,
  },
] as const;

interface ApiStubOptions {
  models?: readonly object[];
  onboardingCompleted?: boolean;
}

export async function stubBootstrapRequests(
  page: Page,
  options: ApiStubOptions = {},
) {
  await page.route("**/v1/admin/models", async (route) => {
    await route.fulfill({
      json: { models: options.models ?? [] },
    });
  });
  await page.route("**/v1/onboarding", async (route) => {
    await route.fulfill({
      json: {
        completed: options.onboardingCompleted ?? true,
        completed_at: options.onboardingCompleted === false ? null : 1,
        analytics_opt_in: false,
      },
    });
  });
  await page.route("**/v1/preferences", async (route) => {
    await route.fulfill({ json: { analytics_opt_in: false } });
  });
  await page.route("**/v1/voice/profile", async (route) => {
    await route.fulfill({
      json: {
        id: "default",
        name: "Default",
        system_prompt: "Be concise and helpful.",
        default_system_prompt: "Be concise and helpful.",
        observational_memory_enabled: true,
        created_at: 1,
        updated_at: 1,
      },
    });
  });
  await page.route("**/v1/voice/observations?**", async (route) => {
    await route.fulfill({ json: [] });
  });
}
