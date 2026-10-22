import { act, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { MemoryRouter, Route, Routes, useLocation } from "react-router";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import type { ModelInfo } from "@/api";
import { FirstRunGate, useFirstRun } from "@/app/onboarding/FirstRunGate";

const GB = 1e9;
const GIB = 1024 ** 3;

const mocks = vi.hoisted(() => ({
  getOnboardingState: vi.fn(),
  completeOnboarding: vi.fn(),
  resetOnboarding: vi.fn(),
  updateAnalyticsPreference: vi.fn(),
  getModelStorage: vi.fn(),
  getHealth: vi.fn(),
  notify: vi.fn(),
  setAnalyticsEnabled: vi.fn(),
  trackOnboardingViewed: vi.fn(),
  trackOnboardingCompleted: vi.fn(),
  trackAnalyticsConsentChanged: vi.fn(),
  canRevealPaths: vi.fn(() => false),
  revealLogs: vi.fn(),
}));

/** The model catalog, as a tiny store the mocked hook subscribes to. */
const catalog = vi.hoisted(() => {
  const listeners = new Set<() => void>();
  const store = {
    state: {} as Record<string, unknown>,
    get: () => store.state,
    set(next: Record<string, unknown>) {
      store.state = { ...store.state, ...next };
      listeners.forEach((listener) => listener());
    },
    subscribe(listener: () => void) {
      listeners.add(listener);
      return () => listeners.delete(listener);
    },
  };
  return store;
});

vi.mock("@/api", () => ({
  api: {
    baseUrl: "http://127.0.0.1:8080/v1",
    getOnboardingState: mocks.getOnboardingState,
    completeOnboarding: mocks.completeOnboarding,
    resetOnboarding: mocks.resetOnboarding,
    updateAnalyticsPreference: mocks.updateAnalyticsPreference,
    getModelStorage: mocks.getModelStorage,
    getHealth: mocks.getHealth,
  },
}));

vi.mock("@/app/providers/ModelCatalogProvider", async () => {
  const { useSyncExternalStore } = await import("react");
  return {
    useModelCatalog: () => useSyncExternalStore(catalog.subscribe, catalog.get),
  };
});

vi.mock("@/app/providers/NotificationProvider", () => ({
  useNotifications: () => ({ notify: mocks.notify, dismiss: vi.fn() }),
}));

vi.mock("@/app/analytics/client", () => ({
  setAnalyticsEnabled: mocks.setAnalyticsEnabled,
}));

vi.mock("@/app/analytics/events", () => ({
  trackOnboardingViewed: mocks.trackOnboardingViewed,
  trackOnboardingCompleted: mocks.trackOnboardingCompleted,
  trackAnalyticsConsentChanged: mocks.trackAnalyticsConsentChanged,
}));

vi.mock("@/shared/revealPath", () => ({
  canRevealPaths: mocks.canRevealPaths,
  revealLogs: mocks.revealLogs,
}));

function model(
  variant: string,
  task: string,
  overrides: Partial<ModelInfo> = {},
): ModelInfo {
  return {
    variant,
    task,
    status: "not_downloaded",
    device_fit: "fits",
    estimated_size_bytes: GB,
    disk_required_bytes: GB,
    estimated_memory_bytes: 2 * GIB,
    local_path: null,
    size_bytes: null,
    download_progress: null,
    error_message: null,
    ...overrides,
  };
}

const voices = (count: number) => ({
  supports_builtin_voices: true,
  built_in_voice_count: count,
  supports_reference_voice: false,
  supports_voice_description: false,
  supports_streaming: true,
  supports_speed_control: false,
  supports_auto_long_form: false,
});

function catalogModels(): ModelInfo[] {
  return [
    model("Parakeet-TDT-0.6B-v3", "asr", {
      estimated_size_bytes: 2.5 * GB,
      disk_required_bytes: 5 * GB,
    }),
    model("Qwen3-ASR-0.6B-GGUF", "asr"),
    model("Qwen3.5-4B", "chat", {
      estimated_size_bytes: 3.4 * GB,
      disk_required_bytes: 3.4 * GB,
    }),
    model("Qwen3.8-27B-FP8", "chat", {
      device_fit: "too_large",
      estimated_size_bytes: 30.9 * GB,
      estimated_memory_bytes: 80 * GIB,
    }),
    model("Kokoro-82M", "tts", {
      estimated_size_bytes: 0.4 * GB,
      disk_required_bytes: 0.4 * GB,
      speech_capabilities: voices(41),
    }),
    model("Qwen3-TTS-Tokenizer-12Hz", "tokenizer"),
  ];
}

const TALK = ["Parakeet-TDT-0.6B-v3", "Qwen3.5-4B", "Kokoro-82M"];

function setModels(update: (models: ModelInfo[]) => ModelInfo[]) {
  catalog.set({ models: update(catalog.get().models as ModelInfo[]) });
}

function withStatus(variants: string[], status: ModelInfo["status"], extra: Partial<ModelInfo> = {}) {
  setModels((models) =>
    models.map((entry) =>
      variants.includes(entry.variant) ? { ...entry, status, ...extra } : entry,
    ),
  );
}

function LocationProbe() {
  const location = useLocation();
  const { runSetupAgain } = useFirstRun();
  return (
    <div>
      <p>App shell at {location.pathname}</p>
      <button type="button" onClick={() => void runSetupAgain()}>
        Run setup again
      </button>
    </div>
  );
}

function renderGate(path = "/models") {
  return render(
    <MemoryRouter initialEntries={[path]}>
      <Routes>
        <Route
          path="*"
          element={
            <FirstRunGate>
              <LocationProbe />
            </FirstRunGate>
          }
        />
      </Routes>
    </MemoryRouter>,
  );
}

const notCompleted = { completed: false, completed_at: null, analytics_opt_in: false };
const completed = { completed: true, completed_at: 1, analytics_opt_in: false };

async function openSetup() {
  renderGate();
  fireEvent.click(await screen.findByRole("button", { name: "Get started" }));
  return screen.findByRole("heading", { name: "Set up Izwi for this Mac" });
}

function taskRow(name: string) {
  return screen.getByRole("checkbox", { name }).closest("li") as HTMLElement;
}

beforeEach(() => {
  vi.clearAllMocks();
  Object.defineProperty(navigator, "platform", {
    configurable: true,
    value: "MacIntel",
  });
  mocks.canRevealPaths.mockReturnValue(false);
  mocks.getOnboardingState.mockResolvedValue(notCompleted);
  mocks.completeOnboarding.mockResolvedValue(completed);
  mocks.resetOnboarding.mockResolvedValue(notCompleted);
  mocks.updateAnalyticsPreference.mockImplementation(async ({ opt_in }) => ({
    analytics_opt_in: opt_in,
  }));
  mocks.getModelStorage.mockResolvedValue({
    models_dir: "/models",
    free_bytes: 100 * GB,
    total_bytes: 500 * GB,
    models_bytes: 0,
  });
  mocks.getHealth.mockResolvedValue({
    status: "ok",
    version: "0.1.0",
    runtime: {
      selected_backend: "metal",
      memory: {
        system_total_bytes: 16 * GIB,
        system_available_bytes: 8 * GIB,
        reserved_bytes: 0,
      },
    },
  });
  catalog.state = {
    models: catalogModels(),
    loading: false,
    catalogError: null,
    downloadProgress: {},
    downloadModel: vi.fn(async (variant: string) =>
      withStatus([variant], "downloading"),
    ),
    cancelModelDownload: vi.fn(async (variant: string) =>
      withStatus([variant], "not_downloaded"),
    ),
    refreshModels: vi.fn(async () => true),
    selectModel: vi.fn(),
  };
});

afterEach(() => {
  vi.useRealTimers();
});

describe("FirstRunGate", () => {
  it("shows the app when first run is done", async () => {
    mocks.getOnboardingState.mockResolvedValue(completed);
    renderGate();

    expect(await screen.findByText("App shell at /models")).toBeVisible();
    expect(screen.queryByRole("heading", { name: "Welcome to Izwi" })).toBeNull();
  });

  it("shows first run in place of the app, focused on Get started", async () => {
    renderGate();

    expect(
      await screen.findByRole("heading", { name: "Welcome to Izwi" }),
    ).toBeVisible();
    expect(screen.queryByText(/App shell/)).toBeNull();
    expect(screen.getByRole("button", { name: "Get started" })).toHaveFocus();
    expect(
      screen.getByRole("switch", { name: "Share anonymous usage data" }),
    ).not.toBeChecked();
    expect(screen.getByRole("img", { name: "Step 1 of 3" })).toBeVisible();
  });

  it("says the engine isn't responding and retries until it answers", async () => {
    vi.useFakeTimers();
    mocks.getOnboardingState
      .mockRejectedValueOnce(new TypeError("Failed to fetch"))
      .mockRejectedValueOnce(new TypeError("Failed to fetch"))
      .mockResolvedValue(notCompleted);
    renderGate();
    await act(async () => {});

    expect(
      screen.getByRole("heading", { name: "Izwi's engine isn't responding" }),
    ).toBeVisible();
    expect(screen.getByText("Couldn't connect to 127.0.0.1:8080")).toBeVisible();
    expect(screen.getByText("Trying again in 2 seconds")).toBeVisible();
    // A browser can't show the log folder.
    expect(screen.queryByRole("button", { name: "Open the log folder" })).toBeNull();

    await act(async () => {
      await vi.advanceTimersByTimeAsync(2_000);
    });
    expect(mocks.getOnboardingState).toHaveBeenCalledTimes(2);
    expect(screen.getByText("Trying again in 4 seconds")).toBeVisible();

    await act(async () => {
      await vi.advanceTimersByTimeAsync(4_000);
    });
    expect(mocks.getOnboardingState).toHaveBeenCalledTimes(3);
    expect(screen.getByRole("heading", { name: "Welcome to Izwi" })).toBeVisible();
  });

  it("retries at once on request, and offers the log folder in the desktop app", async () => {
    mocks.canRevealPaths.mockReturnValue(true);
    mocks.revealLogs.mockResolvedValue(undefined);
    mocks.getOnboardingState
      .mockRejectedValueOnce(new Error("database is locked"))
      .mockResolvedValue(completed);
    renderGate();

    expect(await screen.findByText("database is locked")).toBeVisible();
    fireEvent.click(screen.getByRole("button", { name: "Open the log folder" }));
    expect(mocks.revealLogs).toHaveBeenCalled();

    fireEvent.click(screen.getByRole("button", { name: "Try again now" }));
    expect(await screen.findByText("App shell at /models")).toBeVisible();
  });

  it("runs setup again in place of the app", async () => {
    mocks.getOnboardingState.mockResolvedValue(completed);
    renderGate();

    fireEvent.click(await screen.findByRole("button", { name: "Run setup again" }));

    expect(
      await screen.findByRole("heading", { name: "Welcome to Izwi" }),
    ).toBeVisible();
    expect(mocks.resetOnboarding).toHaveBeenCalledTimes(1);
  });
});

describe("FirstRunOnboarding", () => {
  it("picks models for this Mac and counts shared ones once", async () => {
    await openSetup();

    expect(
      screen.getByText("Picked for 16 GB of memory · Apple Metal"),
    ).toBeVisible();
    expect(within(taskRow("Talk to Izwi")).getByText("6.3 GB")).toBeVisible();
    expect(
      within(taskRow("Talk to Izwi")).getByText("Parakeet v3 · Qwen3.5 4B · Kokoro 82M"),
    ).toBeVisible();
    for (const name of ["Chat", "Transcribe recordings", "Read text aloud"]) {
      expect(screen.getByRole("checkbox", { name })).toBeChecked();
      expect(within(taskRow(name)).getByText("Included")).toBeVisible();
    }
    expect(
      screen.getByRole("button", { name: "Download and set up · 6.3 GB" }),
    ).toBeEnabled();
    expect(
      await screen.findByText("Needs 8.8 GB of free disk space."),
    ).toBeVisible();
  });

  it("can't set up nothing", async () => {
    await openSetup();

    for (const name of ["Talk to Izwi", "Chat", "Transcribe recordings", "Read text aloud"]) {
      fireEvent.click(screen.getByRole("checkbox", { name }));
    }

    expect(screen.getByRole("button", { name: "Set up" })).toBeDisabled();
    expect(screen.queryByText(/free disk space/)).toBeNull();
  });

  it("hides the disk line until free space is known", async () => {
    mocks.getModelStorage.mockRejectedValue(new Error("unavailable"));
    await openSetup();

    expect(screen.getByRole("button", { name: /Download and set up/ })).toBeVisible();
    expect(screen.queryByText(/free disk space/)).toBeNull();
  });

  it("goes back to Welcome from setup", async () => {
    await openSetup();

    fireEvent.click(screen.getByRole("button", { name: "Back" }));

    expect(screen.getByRole("heading", { name: "Welcome to Izwi" })).toHaveFocus();
  });

  it("lists models by task, with ones too large for this Mac unselectable", async () => {
    await openSetup();
    fireEvent.click(screen.getByRole("button", { name: "Choose models myself" }));

    expect(screen.getByRole("heading", { name: "Choose models" })).toBeVisible();
    const chat = screen.getByRole("region", { name: "Chat" });
    const large = within(chat).getByRole("checkbox", { name: "Qwen3.8 27B" });
    expect(large).toBeDisabled();
    expect(within(chat).getByText("Needs 80 GB of memory")).toBeVisible();
    expect(within(chat).getByRole("checkbox", { name: "Qwen3.5 4B" })).toBeChecked();
    expect(screen.queryByText(/Tokenizer/)).toBeNull();

    fireEvent.click(screen.getByRole("checkbox", { name: "Qwen3-ASR 0.6B" }));
    expect(
      screen.getByRole("button", { name: "Download 4 models · 7.3 GB" }),
    ).toBeEnabled();

    for (const name of ["Parakeet v3", "Qwen3-ASR 0.6B", "Qwen3.5 4B", "Kokoro 82M"]) {
      fireEvent.click(screen.getByRole("checkbox", { name }));
    }
    expect(screen.getByRole("button", { name: "Set up" })).toBeDisabled();

    fireEvent.click(screen.getByRole("button", { name: "Recommended setup" }));
    expect(
      screen.getByRole("heading", { name: "Set up Izwi for this Mac" }),
    ).toBeVisible();
  });

  it("starts the downloads, records completion once and shows progress", async () => {
    await openSetup();
    const { downloadModel, selectModel } = catalog.get() as {
      downloadModel: ReturnType<typeof vi.fn>;
      selectModel: ReturnType<typeof vi.fn>;
    };

    fireEvent.click(screen.getByRole("button", { name: "Download and set up · 6.3 GB" }));

    expect(downloadModel.mock.calls.map(([variant]) => variant)).toEqual(TALK);
    expect(selectModel.mock.calls).toEqual([
      ["Parakeet-TDT-0.6B-v3", "transcription"],
      ["Qwen3.5-4B", "chat"],
      ["Kokoro-82M", "text-to-speech"],
    ]);
    expect(mocks.completeOnboarding).toHaveBeenCalledTimes(1);
    expect(mocks.trackOnboardingCompleted).toHaveBeenCalledWith("quick");
    expect(
      await screen.findByRole("heading", { name: "Setting up Izwi" }),
    ).toBeVisible();
    // No way back once setup is recorded.
    expect(screen.queryByRole("button", { name: "Skip setup" })).toBeNull();
    expect(screen.queryByRole("button", { name: "Back" })).toBeNull();
    expect(
      screen.getByRole("progressbar", { name: "Speech recognition download" }),
    ).toBeVisible();

    fireEvent.click(screen.getByRole("button", { name: "Start using Izwi" }));

    expect(await screen.findByText("App shell at /models")).toBeVisible();
    expect(mocks.completeOnboarding).toHaveBeenCalledTimes(1);
    expect(mocks.trackOnboardingCompleted).toHaveBeenCalledTimes(1);
  });

  it("shows why a download failed, retries it, and cancels the rest", async () => {
    await openSetup();
    const { downloadModel, cancelModelDownload } = catalog.get() as {
      downloadModel: ReturnType<typeof vi.fn>;
      cancelModelDownload: ReturnType<typeof vi.fn>;
    };
    fireEvent.click(screen.getByRole("button", { name: /Download and set up/ }));
    await screen.findByRole("heading", { name: "Setting up Izwi" });

    act(() =>
      withStatus(["Qwen3.5-4B"], "error", {
        error_message: "Not enough disk space. Free up 3.4 GB, then retry.",
      }),
    );
    expect(screen.getByText("Couldn't download")).toBeVisible();
    expect(
      screen.getByText("Not enough disk space. Free up 3.4 GB, then retry."),
    ).toBeVisible();

    fireEvent.click(screen.getByRole("button", { name: "Retry Qwen3.5 4B" }));
    expect(downloadModel).toHaveBeenLastCalledWith("Qwen3.5-4B");

    fireEvent.click(screen.getByRole("button", { name: "Cancel downloads" }));
    expect(cancelModelDownload.mock.calls.map(([variant]) => variant)).toEqual(TALK);
    await waitFor(() => expect(screen.getAllByText("Cancelled")).toHaveLength(3));
    expect(screen.queryByRole("button", { name: "Cancel downloads" })).toBeNull();
  });

  it("finishes when the downloads land, with links to what's ready", async () => {
    await openSetup();
    fireEvent.click(screen.getByRole("button", { name: /Download and set up/ }));
    await screen.findByRole("heading", { name: "Setting up Izwi" });

    act(() => withStatus(TALK, "downloaded"));

    expect(
      await screen.findByRole("heading", { name: "You're all set" }),
    ).toBeVisible();
    const links = within(screen.getByRole("navigation", { name: "Start with" }));
    expect(links.getAllByRole("link").map((link) => link.textContent)).toEqual([
      "Talk to Izwi",
      "Chat",
      "Transcribe a recording",
      "Read text aloud",
    ]);

    fireEvent.click(links.getByRole("link", { name: "Transcribe a recording" }));

    expect(await screen.findByText("App shell at /transcription")).toBeVisible();
  });

  it("goes straight to You're all set when everything is on disk", async () => {
    withStatus(TALK, "downloaded");
    await openSetup();

    expect(within(taskRow("Talk to Izwi")).getByText("Downloaded")).toBeVisible();
    fireEvent.click(screen.getByRole("button", { name: "Set up" }));

    expect(
      await screen.findByRole("heading", { name: "You're all set" }),
    ).toBeVisible();
    const { downloadModel } = catalog.get() as { downloadModel: ReturnType<typeof vi.fn> };
    expect(downloadModel).not.toHaveBeenCalled();
    expect(mocks.completeOnboarding).toHaveBeenCalledTimes(1);

    fireEvent.click(screen.getByRole("button", { name: "Open Izwi" }));
    expect(await screen.findByText("App shell at /models")).toBeVisible();
  });

  it("skips: saves the usage data choice, records completion and opens the app", async () => {
    renderGate();
    fireEvent.click(
      await screen.findByRole("switch", { name: "Share anonymous usage data" }),
    );
    fireEvent.click(screen.getByRole("button", { name: "Skip setup" }));

    expect(await screen.findByText("App shell at /models")).toBeVisible();
    expect(mocks.updateAnalyticsPreference).toHaveBeenCalledWith({ opt_in: true });
    expect(mocks.setAnalyticsEnabled).toHaveBeenLastCalledWith(true);
    // Once usage data is on, the view that just happened counts.
    expect(mocks.trackOnboardingViewed).toHaveBeenCalledTimes(1);
    expect(mocks.trackAnalyticsConsentChanged).toHaveBeenCalledWith(
      "opted_in",
      "onboarding",
    );
    expect(mocks.completeOnboarding).toHaveBeenCalledTimes(1);
    expect(mocks.trackOnboardingCompleted).toHaveBeenCalledWith("skip");
  });

  it("leaves usage data off unless turned on", async () => {
    await openSetup();

    expect(mocks.updateAnalyticsPreference).not.toHaveBeenCalled();
    expect(mocks.trackOnboardingViewed).not.toHaveBeenCalled();
  });

  it("keeps retrying a failed completion save", async () => {
    mocks.completeOnboarding
      .mockRejectedValueOnce(new Error("offline"))
      .mockResolvedValue(completed);
    vi.spyOn(console, "warn").mockImplementation(() => {});
    renderGate();
    await screen.findByRole("button", { name: "Skip setup" });
    vi.useFakeTimers();

    fireEvent.click(screen.getByRole("button", { name: "Skip setup" }));
    await act(async () => {});
    expect(screen.getByText("App shell at /models")).toBeVisible();
    expect(mocks.completeOnboarding).toHaveBeenCalledTimes(1);

    await act(async () => {
      await vi.advanceTimersByTimeAsync(2_000);
    });
    expect(mocks.completeOnboarding).toHaveBeenCalledTimes(2);
  });

  it("says the engine isn't responding when the models can't load", async () => {
    catalog.set({ models: [], catalogError: "Couldn't reach the model service" });
    renderGate();
    fireEvent.click(await screen.findByRole("button", { name: "Get started" }));

    expect(
      await screen.findByRole("heading", { name: "Izwi's engine isn't responding" }),
    ).toBeVisible();
    expect(screen.getByText("Couldn't reach the model service")).toBeVisible();
    fireEvent.click(screen.getByRole("button", { name: "Try again now" }));
    expect(catalog.get().refreshModels).toHaveBeenCalled();
  });
});
