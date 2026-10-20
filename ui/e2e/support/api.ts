import type { Page } from "@playwright/test";

const GIB = 1024 ** 3;

export const READY_MODELS = [
  {
    variant: "Qwen3.5-4B",
    display_name: "Qwen3.5 4B GGUF",
    task: "chat",
    estimated_size_bytes: 3_413_361_504,
    estimated_memory_bytes: 9 * GIB,
    device_fit: "fits",
    disk_required_bytes: 0,
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
    display_name: "Kokoro 82M",
    task: "tts",
    estimated_size_bytes: 363_323_757,
    estimated_memory_bytes: 2 * GIB,
    device_fit: "fits",
    disk_required_bytes: 0,
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
    display_name: "Parakeet TDT 0.6B v3",
    task: "asr",
    estimated_size_bytes: 2_509_332_480,
    estimated_memory_bytes: 12 * GIB,
    device_fit: "fits",
    disk_required_bytes: 0,
    status: "ready",
    local_path: "/tmp/parakeet",
    size_bytes: 1_000_000,
    download_progress: null,
    error_message: null,
  },
] as const;

/** The same models before their download: what a first run sees. */
export const NOT_DOWNLOADED_MODELS = READY_MODELS.map((model) => ({
  ...model,
  status: "not_downloaded",
  local_path: null,
  disk_required_bytes: model.estimated_size_bytes,
}));

interface ApiStubOptions {
  models?: readonly object[];
  onboardingCompleted?: boolean;
}

export async function stubBootstrapRequests(
  page: Page,
  options: ApiStubOptions = {},
) {
  // Anything not stubbed below fails as it would with no engine running.
  // The dev and preview servers otherwise proxy /v1 to a real Izwi on :8080.
  // Routes registered later take precedence over this one.
  await page.route("**/v1/**", async (route) => {
    await route.fulfill({
      status: 503,
      json: { error: { message: "Not stubbed in this test" } },
    });
  });
  await page.routeWebSocket(/\/v1\//, (socket) => socket.close());
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
  await page.route("**/v1/admin/storage", async (route) => {
    await route.fulfill({
      json: {
        models_dir: "/tmp/izwi-models",
        free_bytes: 100_000_000_000,
        total_bytes: 500_000_000_000,
        models_bytes: 0,
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
