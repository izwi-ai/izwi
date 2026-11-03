import type { ReactNode } from "react";
import {
  act,
  fireEvent,
  render,
  screen,
  waitFor,
  within,
} from "@testing-library/react";
import { MemoryRouter, Route, Routes } from "react-router";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { AppLayout } from "@/app/layouts/AppLayout";
import { NotificationProvider } from "@/app/providers/NotificationProvider";
import { TextToSpeechPage } from "./route";
import type { ModelInfo, SpeechHistoryRecordStreamCallbacks } from "@/api";

const apiMocks = vi.hoisted(() => ({
  listTextToSpeechRecords: vi.fn(),
  listTextToSpeechRecordPage: vi.fn(),
  getTextToSpeechRecord: vi.fn(),
  textToSpeechRecordAudioUrl: vi.fn(),
  deleteTextToSpeechRecord: vi.fn(),
  createTextToSpeechRecord: vi.fn(),
  createTextToSpeechRecordStream: vi.fn(),
  attachTextToSpeechRecordStream: vi.fn(),
  listSavedVoices: vi.fn(),
  downloadAudioFile: vi.fn(),
  cancelTextToSpeechRecord: vi.fn(),
}));

const hookMocks = vi.hoisted(() => ({
  useRouteModelSelection: vi.fn(),
}));

const componentMocks = vi.hoisted(() => ({
  routeModelModal: vi.fn(
    ({
      isOpen,
      title,
      zIndexClassName,
      onClose,
    }: {
      isOpen: boolean;
      title: string;
      zIndexClassName?: string;
      onClose: () => void;
    }) =>
      isOpen ? (
        <div
          data-testid="route-model-modal"
          data-z-index={zIndexClassName ?? ""}
        >
          {title}
          <button type="button" onClick={onClose}>
            Close model modal
          </button>
        </div>
      ) : null,
  ),
}));

vi.mock("@/app/onboarding/FirstRunGate", () => ({
  FirstRunGate: ({ children }: { children: ReactNode }) => children,
}));
vi.mock("@/app/providers/AppUpdateProvider", () => ({
  useAppUpdates: () => ({ availableUpdate: null, isPromptOpen: false }),
}));

vi.mock("@/api", () => ({
  api: {
    listTextToSpeechRecords: apiMocks.listTextToSpeechRecords,
    listTextToSpeechRecordPage: apiMocks.listTextToSpeechRecordPage,
    getTextToSpeechRecord: apiMocks.getTextToSpeechRecord,
    textToSpeechRecordAudioUrl: apiMocks.textToSpeechRecordAudioUrl,
    deleteTextToSpeechRecord: apiMocks.deleteTextToSpeechRecord,
    createTextToSpeechRecord: apiMocks.createTextToSpeechRecord,
    createTextToSpeechRecordStream: apiMocks.createTextToSpeechRecordStream,
    attachTextToSpeechRecordStream: apiMocks.attachTextToSpeechRecordStream,
    listSavedVoices: apiMocks.listSavedVoices,
    downloadAudioFile: apiMocks.downloadAudioFile,
    cancelTextToSpeechRecord: apiMocks.cancelTextToSpeechRecord,
    getHealth: () => new Promise(() => {}),
  },
}));

vi.mock("@/features/models/hooks/useRouteModelSelection", () => ({
  useRouteModelSelection: hookMocks.useRouteModelSelection,
}));

vi.mock("@/features/models/components/RouteModelModal", () => ({
  RouteModelModal: componentMocks.routeModelModal,
}));

const baseProps = {
  models: [] as ModelInfo[],
  selectedModel: null,
  loading: false,
  downloadProgress: {},
  onDownload: vi.fn(),
  onCancelDownload: vi.fn(),
  onLoad: vi.fn(),
  onUnload: vi.fn(),
  onDelete: vi.fn(),
  onSelect: vi.fn(),
  onError: vi.fn(),
};

function buildRecord(
  overrides: Partial<Record<string, unknown>> = {},
) {
  return {
    id: "tts-1",
    created_at: 1,
    route_kind: "text_to_speech",
    processing_status: "pending",
    processing_error: null,
    model_id: "Qwen3-TTS-12Hz-1.7B-Chat",
    speaker: "Vivian",
    language: null,
    saved_voice_id: null,
    speed: 1,
    input_text: "Hello world",
    voice_description: null,
    reference_text: null,
    generation_time_ms: 0,
    audio_duration_secs: null,
    rtf: null,
    tokens_generated: null,
    audio_mime_type: "audio/wav",
    audio_filename: "tts.wav",
    ...overrides,
  };
}

function buildSummary(
  overrides: Partial<Record<string, unknown>> = {},
) {
  return {
    id: "tts-1",
    created_at: 1,
    route_kind: "text_to_speech",
    processing_status: "pending",
    processing_error: null,
    model_id: "Qwen3-TTS-12Hz-1.7B-Chat",
    speaker: "Vivian",
    language: null,
    input_preview: "Hello world",
    input_chars: 11,
    generation_time_ms: 0,
    audio_duration_secs: null,
    rtf: null,
    tokens_generated: null,
    audio_mime_type: "audio/wav",
    audio_filename: "tts.wav",
    ...overrides,
  };
}

function buildSavedVoice(id: string, name: string) {
  return {
    id,
    created_at: 1,
    updated_at: 1,
    name,
    reference_text_preview: "preview",
    reference_text_chars: 7,
    audio_mime_type: "audio/wav",
    audio_filename: "voice.wav",
    source_route_kind: null,
    source_record_id: null,
  };
}

function buildSpeechCapabilities(overrides: Partial<Record<string, boolean>> = {}) {
  return {
    supports_builtin_voices: true,
    supports_reference_voice: false,
    supports_voice_description: true,
    supports_streaming: true,
    supports_speed_control: true,
    supports_auto_long_form: false,
    ...overrides,
  };
}

function buildFishModel(supportsInProgressReplay: boolean): ModelInfo {
  return {
    variant: "FishAudio-S2-Pro",
    status: "ready",
    local_path: null,
    size_bytes: null,
    download_progress: null,
    error_message: null,
    speech_capabilities: {
      supports_builtin_voices: false,
      built_in_voice_count: null,
      supports_reference_voice: true,
      supports_voice_description: false,
      supports_streaming: true,
      supports_speed_control: false,
      supports_auto_long_form: true,
      ...(supportsInProgressReplay ? { supports_in_progress_replay: true } : {}),
    },
  };
}

function renderRoute(
  initialEntry: string,
  propsOverrides: Partial<typeof baseProps> = {},
  withLayout = false,
) {
  const routeProps = { ...baseProps, ...propsOverrides };
  if (withLayout) {
    vi.stubGlobal("localStorage", { getItem: vi.fn().mockReturnValue(null), setItem: vi.fn() });
  }
  return render(
    <NotificationProvider>
      <MemoryRouter initialEntries={[initialEntry]}>
        <Routes>
          <Route
            element={withLayout ? (
              <AppLayout
                readyModelsCount={1}
                catalogError={null}
                onRetryModelCatalog={vi.fn().mockResolvedValue(undefined)}
              />
            ) : undefined}
          >
            <Route
              path="/text-to-speech"
              element={<TextToSpeechPage {...routeProps} />}
            />
            <Route
              path="/text-to-speech/:recordId"
              element={<TextToSpeechPage {...routeProps} />}
            />
          </Route>
        </Routes>
      </MemoryRouter>
    </NotificationProvider>,
  );
}

function deferredPromise<T>() {
  let resolve!: (value: T) => void;
  let reject!: (reason?: unknown) => void;
  const promise = new Promise<T>((res, rej) => {
    resolve = res;
    reject = rej;
  });
  return { promise, resolve, reject };
}

/** Opens the record page's header menu; its items. */
async function openRecordMenu() {
  const recordMenu = await screen.findByRole("button", {
    name: /^More actions for /,
  });
  fireEvent.pointerDown(recordMenu, { button: 0, ctrlKey: false });
  fireEvent.click(recordMenu);
  return screen.findByRole("menu");
}

/** Chooses Open in dialogue editor from the record page's header menu. */
async function openInDialogueEditor() {
  const menu = await openRecordMenu();
  fireEvent.click(
    within(menu).getByRole("menuitem", { name: "Open in dialogue editor" }),
  );
}

describe("TextToSpeechPage", () => {
  beforeEach(() => {
    vi.stubGlobal("AudioContext", class {
      resume = vi.fn().mockResolvedValue(undefined);
      close = vi.fn().mockResolvedValue(undefined);
      currentTime = 0;
      destination = {};
      createBuffer = vi.fn(() => ({ copyToChannel: vi.fn() }));
      createBufferSource = vi.fn(() => ({
        buffer: null, connect: vi.fn(), disconnect: vi.fn(), start: vi.fn(), stop: vi.fn(), onended: null,
      }));
    });
    apiMocks.cancelTextToSpeechRecord.mockReset().mockResolvedValue({ record: buildRecord() });
    apiMocks.attachTextToSpeechRecordStream.mockReset();
    apiMocks.listTextToSpeechRecords.mockReset();
    apiMocks.listTextToSpeechRecordPage.mockReset();
    apiMocks.getTextToSpeechRecord.mockReset();
    apiMocks.textToSpeechRecordAudioUrl.mockReset();
    apiMocks.deleteTextToSpeechRecord.mockReset();
    apiMocks.createTextToSpeechRecord.mockReset();
    apiMocks.createTextToSpeechRecordStream.mockReset();
    apiMocks.listSavedVoices.mockReset();
    apiMocks.downloadAudioFile.mockReset();
    hookMocks.useRouteModelSelection.mockReset();
    componentMocks.routeModelModal.mockClear();

    apiMocks.listTextToSpeechRecords.mockResolvedValue([]);
    apiMocks.listTextToSpeechRecordPage.mockImplementation(async () => ({
      items: await apiMocks.listTextToSpeechRecords(),
      pagination: {
        next_cursor: null,
        has_more: false,
        limit: 25,
      },
    }));
    apiMocks.getTextToSpeechRecord.mockResolvedValue(buildRecord());
    apiMocks.deleteTextToSpeechRecord.mockResolvedValue({
      id: "tts-1",
      deleted: true,
    });
    apiMocks.textToSpeechRecordAudioUrl.mockReturnValue("/audio/tts.wav");
    apiMocks.createTextToSpeechRecord.mockResolvedValue(buildRecord());
    apiMocks.listSavedVoices.mockResolvedValue([
      buildSavedVoice("voice-1", "Narrator"),
    ]);
    apiMocks.downloadAudioFile.mockResolvedValue(undefined);
    apiMocks.createTextToSpeechRecordStream.mockImplementation(
      (_request, callbacks) => {
        callbacks.onCreated?.(buildRecord());
        return new AbortController();
      },
    );

    hookMocks.useRouteModelSelection.mockReturnValue({
      routeModels: [],
      resolvedSelectedModel: "Qwen3-TTS-12Hz-1.7B-Chat",
      selectedModelInfo: {
        variant: "Qwen3-TTS-12Hz-1.7B-Chat",
        status: "ready",
        speech_capabilities: {
          supports_builtin_voices: true,
          supports_reference_voice: false,
          supports_voice_description: true,
          supports_streaming: true,
          supports_speed_control: true,
        },
      },
      selectedModelReady: true,
      isModelModalOpen: false,
      intentVariant: null,
      closeModelModal: vi.fn(),
      openModelManager: vi.fn(),
      requestModel: vi.fn(),
    });
  });

  it("renders the text-to-speech history table on /text-to-speech", async () => {
    renderRoute("/text-to-speech");

    await waitFor(() =>
      expect(apiMocks.listTextToSpeechRecords).toHaveBeenCalled(),
    );

    expect(
      await screen.findByRole("heading", { name: "Text to Speech" }),
    ).toBeInTheDocument();
    expect(
      screen.getByRole("button", { name: /New generation/i }),
    ).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /Models/i })).toBeInTheDocument();
    expect(
      screen.getByRole("heading", { name: "No speech yet" }),
    ).toBeInTheDocument();
  });

  it("loads more text-to-speech history rows", async () => {
    apiMocks.listTextToSpeechRecordPage.mockReset();
    apiMocks.listTextToSpeechRecordPage
      .mockResolvedValueOnce({
        items: [
          buildSummary({
            id: "tts-page-1",
            audio_filename: "tts-page-one.wav",
            input_preview: "Page one preview",
            processing_status: "ready",
          }),
        ],
        pagination: {
          next_cursor: "tts-cursor-2",
          has_more: true,
          limit: 25,
        },
      })
      .mockResolvedValueOnce({
        items: [
          buildSummary({
            id: "tts-page-2",
            audio_filename: "tts-page-two.wav",
            input_preview: "Page two preview",
            processing_status: "ready",
          }),
        ],
        pagination: {
          next_cursor: null,
          has_more: false,
          limit: 25,
        },
      });

    renderRoute("/text-to-speech");

    expect(await screen.findByText("Page one preview")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Load more" }));
    expect(await screen.findByText("Page two preview")).toBeInTheDocument();
    expect(await screen.findByText("Page one preview")).toBeInTheDocument();

    expect(apiMocks.listTextToSpeechRecordPage).toHaveBeenNthCalledWith(1, {
      limit: 25,
      cursor: null,
      filters: {},
    });
    expect(apiMocks.listTextToSpeechRecordPage).toHaveBeenNthCalledWith(2, {
      limit: 25,
      cursor: "tts-cursor-2",
      filters: {},
    });
  });

  it("keeps history rows visible while text-to-speech polling refreshes in the background", async () => {
    const intervalCallbacks: Array<() => void> = [];
    const setIntervalSpy = vi
      .spyOn(window, "setInterval")
      .mockImplementation((handler) => {
        if (typeof handler === "function") {
          intervalCallbacks.push(handler);
        }
        return 1 as unknown as ReturnType<typeof window.setInterval>;
      });
    const clearIntervalSpy = vi
      .spyOn(window, "clearInterval")
      .mockImplementation(() => {});
    const backgroundRefresh =
      deferredPromise<Awaited<ReturnType<typeof apiMocks.listTextToSpeechRecords>>>();
    apiMocks.listTextToSpeechRecords
      .mockResolvedValueOnce([
        buildSummary({
          id: "tts-history-polling-1",
          processing_status: "processing",
          audio_filename: "voice-note.wav",
          input_preview: "Rendering still in progress",
        }),
      ])
      .mockImplementationOnce(() => backgroundRefresh.promise);

    const view = renderRoute("/text-to-speech");

    try {
      expect(
        await screen.findByText("Rendering still in progress"),
      ).toBeInTheDocument();

      await waitFor(() => expect(setIntervalSpy).toHaveBeenCalled());
      if (intervalCallbacks.length === 0) {
        throw new Error("Expected text-to-speech history polling to register an interval.");
      }

      intervalCallbacks.forEach((callback) => callback());

      await waitFor(() =>
        expect(apiMocks.listTextToSpeechRecords).toHaveBeenCalledTimes(2),
      );

      expect(screen.getByText("Rendering still in progress")).toBeInTheDocument();
      expect(
        screen.queryByText("Loading text-to-speech history..."),
      ).not.toBeInTheDocument();

      backgroundRefresh.resolve([
        buildSummary({
          id: "tts-history-polling-1",
          processing_status: "ready",
          audio_filename: "voice-note.wav",
          input_preview: "Rendering complete",
          audio_duration_secs: 2.3,
          generation_time_ms: 80,
          rtf: 0.4,
          tokens_generated: 42,
        }),
      ]);

      expect(await screen.findByText("Rendering complete")).toBeInTheDocument();
    } finally {
      view.unmount();
      setIntervalSpy.mockRestore();
      clearIntervalSpy.mockRestore();
    }
  });

  it("hides status column and uses saved voice names in history rows", async () => {
    apiMocks.listTextToSpeechRecords.mockResolvedValue([
      buildSummary({
        saved_voice_id: "voice-1",
        speaker: null,
      }),
    ]);
    apiMocks.listSavedVoices.mockResolvedValue([
      buildSavedVoice("voice-1", "Narrator Prime"),
    ]);

    renderRoute("/text-to-speech");

    await waitFor(() =>
      expect(apiMocks.listTextToSpeechRecords).toHaveBeenCalled(),
    );

    expect(
      screen.queryByRole("columnheader", { name: /Status/i }),
    ).not.toBeInTheDocument();
    expect(screen.getByText("Narrator Prime")).toBeInTheDocument();
    expect(screen.queryByText("voice-1")).not.toBeInTheDocument();
  });

  it("uses built-in speaker display names in history rows", async () => {
    apiMocks.listTextToSpeechRecords.mockResolvedValue([
      buildSummary({
        saved_voice_id: null,
        speaker: "Ono_anna",
        model_id: "Qwen3-TTS-12Hz-1.7B-Chat",
      }),
    ]);

    renderRoute("/text-to-speech");

    await waitFor(() =>
      expect(apiMocks.listTextToSpeechRecords).toHaveBeenCalled(),
    );

    expect(screen.getByText("Anna")).toBeInTheDocument();
    expect(screen.queryByText("Ono_anna")).not.toBeInTheDocument();
  });

  it("shows standard row actions from the text-to-speech history menu", async () => {
    apiMocks.listTextToSpeechRecords.mockResolvedValue([
      buildSummary({
        id: "tts-history-1",
        processing_status: "ready",
        audio_filename: "voice-note.wav",
      }),
    ]);

    renderRoute("/text-to-speech");

    expect(await screen.findByText("Hello world")).toBeInTheDocument();

    fireEvent.pointerDown(
      screen.getByRole("button", { name: "More actions for Hello world" }),
      { button: 0, ctrlKey: false },
    );

    expect(await screen.findByRole("menuitem", { name: /Open record/i })).toBeVisible();
    expect(screen.getByRole("menuitem", { name: /Copy text/i })).toBeVisible();
    expect(screen.getByRole("menuitem", { name: /Download/i })).toBeVisible();
    expect(screen.getByRole("menuitem", { name: /^Delete$/i })).toBeVisible();
  });

  it("opens the new text-to-speech modal from the header action", async () => {
    renderRoute("/text-to-speech");

    await waitFor(() =>
      expect(apiMocks.listTextToSpeechRecords).toHaveBeenCalled(),
    );

    fireEvent.click(screen.getByRole("button", { name: /New generation/i }));

    expect(
      await screen.findByRole("heading", { name: "New text-to-speech job" }),
    ).toBeInTheDocument();
    expect(screen.getByRole("textbox", { name: "Script" })).toBeInTheDocument();
    expect(screen.getByText("Model")).toBeInTheDocument();
    expect(screen.getByText("Built-in voice")).toBeInTheDocument();
    expect(screen.getByText("Vivian")).toBeInTheDocument();
    expect(screen.queryByText("Saved voice")).not.toBeInTheDocument();
    // This fixture also supports voice descriptions, offered as optional style.
    expect(
      screen.getByLabelText("Style direction (optional)"),
    ).toBeInTheDocument();
  });

  it("resets modal state after close and reopen", async () => {
    renderRoute("/text-to-speech");

    await waitFor(() =>
      expect(apiMocks.listTextToSpeechRecords).toHaveBeenCalled(),
    );

    fireEvent.click(screen.getByRole("button", { name: /New generation/i }));

    expect(
      await screen.findByRole("heading", { name: "New text-to-speech job" }),
    ).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: /Create generation/i }));
    expect(screen.getByText("Enter text to generate speech.")).toBeInTheDocument();

    fireEvent.change(screen.getByRole("textbox", { name: "Script" }), {
      target: { value: "Temporary draft text" },
    });

    const streamToggle = screen.getByRole("switch", { name: /Stream results/ });
    expect(streamToggle).toBeChecked();
    fireEvent.click(streamToggle);
    expect(streamToggle).not.toBeChecked();

    fireEvent.click(screen.getByRole("button", { name: /^Close$/i }));

    await waitFor(() =>
      expect(
        screen.queryByRole("heading", { name: "New text-to-speech job" }),
      ).not.toBeInTheDocument(),
    );

    fireEvent.click(screen.getByRole("button", { name: /New generation/i }));
    expect(
      await screen.findByRole("heading", { name: "New text-to-speech job" }),
    ).toBeInTheDocument();

    expect(screen.getByRole("textbox", { name: "Script" })).toHaveValue("");
    expect(screen.getByRole("switch", { name: /Stream results/ })).toBeChecked();
    expect(
      screen.queryByText("Enter text to generate speech."),
    ).not.toBeInTheDocument();
  });

  it("offers optional style direction next to built-in voices", async () => {
    hookMocks.useRouteModelSelection.mockReturnValue({
      routeModels: [],
      resolvedSelectedModel: "Qwen3-TTS-12Hz-1.7B-CustomVoice",
      selectedModelInfo: {
        variant: "Qwen3-TTS-12Hz-1.7B-CustomVoice",
        status: "ready",
        speech_capabilities: {
          supports_builtin_voices: true,
          supports_reference_voice: false,
          supports_voice_description: true,
          supports_streaming: true,
          supports_speed_control: false,
        },
      },
      selectedModelReady: true,
      isModelModalOpen: false,
      intentVariant: null,
      closeModelModal: vi.fn(),
      openModelManager: vi.fn(),
      requestModel: vi.fn(),
    });

    renderRoute("/text-to-speech");
    await waitFor(() =>
      expect(apiMocks.listTextToSpeechRecords).toHaveBeenCalled(),
    );
    fireEvent.click(screen.getByRole("button", { name: /New generation/i }));

    expect(await screen.findByText("Built-in voice")).toBeInTheDocument();
    const style = screen.getByLabelText("Style direction (optional)");
    expect(style).not.toBeRequired();
  });

  it("requires a voice description for description-only models", async () => {
    hookMocks.useRouteModelSelection.mockReturnValue({
      routeModels: [],
      resolvedSelectedModel: "Qwen3-TTS-12Hz-1.7B-VoiceDesign",
      selectedModelInfo: {
        variant: "Qwen3-TTS-12Hz-1.7B-VoiceDesign",
        status: "ready",
        speech_capabilities: {
          supports_builtin_voices: false,
          supports_reference_voice: false,
          supports_voice_description: true,
          supports_streaming: true,
          supports_speed_control: false,
        },
      },
      selectedModelReady: true,
      isModelModalOpen: false,
      intentVariant: null,
      closeModelModal: vi.fn(),
      openModelManager: vi.fn(),
      requestModel: vi.fn(),
    });

    renderRoute("/text-to-speech");
    await waitFor(() =>
      expect(apiMocks.listTextToSpeechRecords).toHaveBeenCalled(),
    );
    fireEvent.click(screen.getByRole("button", { name: /New generation/i }));

    const direction = await screen.findByLabelText("Voice direction");
    expect(direction).toBeRequired();
    expect(direction).toHaveAttribute(
      "placeholder",
      expect.stringMatching(/^Describe the voice/),
    );
    expect(screen.queryByText("Built-in voice")).not.toBeInTheDocument();
  });

  it("shows only saved voice controls for clone-capable models", async () => {
    hookMocks.useRouteModelSelection.mockReturnValue({
      routeModels: [],
      resolvedSelectedModel: "Qwen3-TTS-12Hz-1.7B-Base",
      selectedModelInfo: {
        variant: "Qwen3-TTS-12Hz-1.7B-Base",
        status: "ready",
        speech_capabilities: {
          supports_builtin_voices: false,
          supports_reference_voice: true,
          supports_voice_description: false,
          supports_streaming: true,
          supports_speed_control: true,
        },
      },
      selectedModelReady: true,
      isModelModalOpen: false,
      intentVariant: null,
      closeModelModal: vi.fn(),
      openModelManager: vi.fn(),
      requestModel: vi.fn(),
    });

    renderRoute("/text-to-speech");

    await waitFor(() =>
      expect(apiMocks.listTextToSpeechRecords).toHaveBeenCalled(),
    );

    fireEvent.click(screen.getByRole("button", { name: /New generation/i }));

    expect(
      await screen.findByRole("heading", { name: "New text-to-speech job" }),
    ).toBeInTheDocument();
    expect(screen.getByText("Saved voice")).toBeInTheDocument();
    expect(
      screen.getByRole("combobox", { name: "Saved voice" }),
    ).toBeInTheDocument();
    expect(screen.getByText("Select saved voice")).toBeInTheDocument();
    expect(screen.queryByText("Built-in voice")).not.toBeInTheDocument();
    expect(screen.queryByLabelText(/direction/i)).not.toBeInTheDocument();
  });

  it("shows kokoro built-in voices with display names in the modal", async () => {
    hookMocks.useRouteModelSelection.mockReturnValue({
      routeModels: [],
      resolvedSelectedModel: "Kokoro-82M",
      selectedModelInfo: {
        variant: "Kokoro-82M",
        status: "ready",
        speech_capabilities: {
          supports_builtin_voices: true,
          supports_reference_voice: false,
          supports_voice_description: false,
          supports_streaming: true,
          supports_speed_control: true,
        },
      },
      selectedModelReady: true,
      isModelModalOpen: false,
      intentVariant: null,
      closeModelModal: vi.fn(),
      openModelManager: vi.fn(),
      requestModel: vi.fn(),
    });

    renderRoute("/text-to-speech?speaker=bf_alice");

    expect(
      await screen.findByRole("heading", { name: "New text-to-speech job" }),
    ).toBeInTheDocument();

    expect(
      screen.getByRole("combobox", { name: "Built-in voice" }),
    ).toBeInTheDocument();
    expect(screen.getByText("Alice")).toBeInTheDocument();
    expect(screen.queryByText("bf_alice")).not.toBeInTheDocument();
  });

  it("opens the creation modal from a create intent", async () => {
    renderRoute("/text-to-speech?create=speech");

    expect(
      await screen.findByRole("heading", { name: "New text-to-speech job" }),
    ).toBeInTheDocument();
  });

  it("uses voxtral tts built-in voices instead of qwen defaults", async () => {
    hookMocks.useRouteModelSelection.mockReturnValue({
      routeModels: [],
      resolvedSelectedModel: "Voxtral-4B-TTS-2603",
      selectedModelInfo: {
        variant: "Voxtral-4B-TTS-2603",
        status: "ready",
        speech_capabilities: buildSpeechCapabilities({
          supports_reference_voice: false,
          supports_voice_description: false,
        }),
      },
      selectedModelReady: true,
      isModelModalOpen: false,
      intentVariant: null,
      closeModelModal: vi.fn(),
      openModelManager: vi.fn(),
      requestModel: vi.fn(),
    });

    renderRoute("/text-to-speech?speaker=Vivian");

    expect(
      await screen.findByRole("heading", { name: "New text-to-speech job" }),
    ).toBeInTheDocument();

    expect(screen.getByText("Casual Female")).toBeInTheDocument();
    expect(screen.queryByText("Vivian")).not.toBeInTheDocument();

    fireEvent.change(screen.getByRole("textbox", { name: "Script" }), {
      target: { value: "Hello from Voxtral" },
    });
    fireEvent.click(screen.getByRole("button", { name: /Create generation/i }));

    await waitFor(() =>
      expect(apiMocks.createTextToSpeechRecordStream).toHaveBeenCalled(),
    );
    expect(apiMocks.createTextToSpeechRecordStream.mock.calls[0][0]).toMatchObject({
      model_id: "Voxtral-4B-TTS-2603",
      speaker: "casual_female",
      text: "Hello from Voxtral",
    });
  });

  it("loads the selected model from the modal readiness action", async () => {
    const onLoad = vi.fn();

    hookMocks.useRouteModelSelection.mockReturnValue({
      routeModels: [],
      resolvedSelectedModel: "Qwen3-TTS-12Hz-1.7B-Chat",
      selectedModelInfo: {
        variant: "Qwen3-TTS-12Hz-1.7B-Chat",
        status: "downloaded",
        speech_capabilities: {
          supports_builtin_voices: true,
          supports_reference_voice: false,
          supports_voice_description: true,
          supports_streaming: true,
          supports_speed_control: true,
        },
      },
      selectedModelReady: false,
      isModelModalOpen: false,
      intentVariant: null,
      closeModelModal: vi.fn(),
      openModelManager: vi.fn(),
      requestModel: vi.fn(),
    });

    renderRoute("/text-to-speech", { onLoad });

    await waitFor(() =>
      expect(apiMocks.listTextToSpeechRecords).toHaveBeenCalled(),
    );

    fireEvent.click(screen.getByRole("button", { name: /New generation/i }));
    await screen.findByRole("heading", { name: "New text-to-speech job" });

    fireEvent.click(screen.getByRole("button", { name: "Load" }));
    expect(onLoad).toHaveBeenCalledWith("Qwen3-TTS-12Hz-1.7B-Chat");
  });

  it("opens TTS models from the new generation modal", async () => {
    const openModelManager = vi.fn();

    hookMocks.useRouteModelSelection.mockReturnValue({
      routeModels: [],
      resolvedSelectedModel: "Qwen3-TTS-12Hz-1.7B-Chat",
      selectedModelInfo: {
        variant: "Qwen3-TTS-12Hz-1.7B-Chat",
        status: "downloaded",
        speech_capabilities: {
          supports_builtin_voices: true,
          supports_reference_voice: false,
          supports_voice_description: true,
          supports_streaming: true,
          supports_speed_control: true,
        },
      },
      selectedModelReady: false,
      isModelModalOpen: false,
      intentVariant: null,
      closeModelModal: vi.fn(),
      openModelManager,
      requestModel: vi.fn(),
    });

    renderRoute("/text-to-speech");

    await waitFor(() =>
      expect(apiMocks.listTextToSpeechRecords).toHaveBeenCalled(),
    );

    fireEvent.click(screen.getByRole("button", { name: /New generation/i }));
    await screen.findByRole("heading", { name: "New text-to-speech job" });

    fireEvent.click(screen.getByRole("button", { name: "Change model" }));
    expect(openModelManager).toHaveBeenCalledTimes(1);
  });

  it("raises the model modal above the new generation modal", async () => {
    const closeModelModal = vi.fn();

    hookMocks.useRouteModelSelection.mockReturnValue({
      routeModels: [],
      resolvedSelectedModel: "Qwen3-TTS-12Hz-1.7B-Chat",
      selectedModelInfo: {
        variant: "Qwen3-TTS-12Hz-1.7B-Chat",
        status: "downloaded",
        speech_capabilities: {
          supports_builtin_voices: true,
          supports_reference_voice: false,
          supports_voice_description: true,
          supports_streaming: true,
          supports_speed_control: true,
        },
      },
      selectedModelReady: false,
      isModelModalOpen: true,
      intentVariant: null,
      closeModelModal,
      openModelManager: vi.fn(),
      requestModel: vi.fn(),
    });

    renderRoute("/text-to-speech");

    await waitFor(() =>
      expect(apiMocks.listTextToSpeechRecords).toHaveBeenCalled(),
    );

    expect(screen.getByTestId("route-model-modal")).toHaveAttribute(
      "data-z-index",
      "z-50",
    );

    fireEvent.click(screen.getByRole("button", { name: /New generation/i }));
    await screen.findByRole("heading", { name: "New text-to-speech job" });

    expect(screen.getByTestId("route-model-modal")).toHaveAttribute(
      "data-z-index",
      "z-70",
    );

    fireEvent.click(screen.getByRole("button", { name: /Close model modal/i }));
    expect(closeModelModal).toHaveBeenCalledTimes(1);
  });

  it("unloads the selected model from the modal readiness action", async () => {
    const onUnload = vi.fn();

    renderRoute("/text-to-speech", { onUnload });

    await waitFor(() =>
      expect(apiMocks.listTextToSpeechRecords).toHaveBeenCalled(),
    );

    fireEvent.click(screen.getByRole("button", { name: /New generation/i }));
    await screen.findByRole("heading", { name: "New text-to-speech job" });

    fireEvent.click(screen.getByRole("button", { name: "Unload" }));
    expect(onUnload).toHaveBeenCalledWith("Qwen3-TTS-12Hz-1.7B-Chat");
  });

  it("selects a saved-voice-capable model when opening with a saved voice id", async () => {
    const onSelect = vi.fn();

    hookMocks.useRouteModelSelection.mockReturnValue({
      routeModels: [
        {
          variant: "Kokoro-82M",
          status: "ready",
          speech_capabilities: buildSpeechCapabilities({
            supports_reference_voice: false,
            supports_voice_description: false,
          }),
        },
        {
          variant: "Qwen3-TTS-12Hz-0.6B-CustomVoice",
          status: "downloaded",
          speech_capabilities: buildSpeechCapabilities({
            supports_reference_voice: true,
            supports_voice_description: false,
          }),
        },
      ],
      resolvedSelectedModel: "Kokoro-82M",
      selectedModelInfo: {
        variant: "Kokoro-82M",
        status: "ready",
        speech_capabilities: buildSpeechCapabilities({
          supports_reference_voice: false,
          supports_voice_description: false,
        }),
      },
      selectedModelReady: true,
      isModelModalOpen: false,
      intentVariant: null,
      closeModelModal: vi.fn(),
      openModelManager: vi.fn(),
      requestModel: vi.fn(),
    });

    renderRoute("/text-to-speech?voiceId=voice-1", { onSelect });

    await waitFor(() =>
      expect(onSelect).toHaveBeenCalledWith(
        "Qwen3-TTS-12Hz-0.6B-CustomVoice",
      ),
    );
    expect(onSelect).toHaveBeenCalledTimes(1);
  });

  it("retries redirect model selection after route model options hydrate", async () => {
    const onSelect = vi.fn();

    const emptySelection = {
      routeModels: [],
      resolvedSelectedModel: "Kokoro-82M",
      selectedModelInfo: {
        variant: "Kokoro-82M",
        status: "ready",
        speech_capabilities: buildSpeechCapabilities({
          supports_reference_voice: false,
          supports_voice_description: false,
        }),
      },
      selectedModelReady: true,
      isModelModalOpen: false,
      intentVariant: null,
      closeModelModal: vi.fn(),
      openModelManager: vi.fn(),
      requestModel: vi.fn(),
    };

    const hydratedSelection = {
      routeModels: [
        {
          variant: "Qwen3-TTS-12Hz-0.6B-CustomVoice",
          status: "downloaded",
          speech_capabilities: buildSpeechCapabilities({
            supports_reference_voice: true,
            supports_voice_description: false,
          }),
        },
      ],
      resolvedSelectedModel: "Kokoro-82M",
      selectedModelInfo: {
        variant: "Kokoro-82M",
        status: "ready",
        speech_capabilities: buildSpeechCapabilities({
          supports_reference_voice: false,
          supports_voice_description: false,
        }),
      },
      selectedModelReady: true,
      isModelModalOpen: false,
      intentVariant: null,
      closeModelModal: vi.fn(),
      openModelManager: vi.fn(),
      requestModel: vi.fn(),
    };

    hookMocks.useRouteModelSelection
      .mockReturnValueOnce(emptySelection)
      .mockReturnValue(hydratedSelection);

    renderRoute("/text-to-speech?voiceId=voice-1", { onSelect });

    await waitFor(() =>
      expect(onSelect).toHaveBeenCalledWith(
        "Qwen3-TTS-12Hz-0.6B-CustomVoice",
      ),
    );
  });

  it("lets a reloaded Fish record attach playback from the beginning without creating a job", async () => {
    const record = buildRecord({ id: "existing-fish", model_id: "FishAudio-S2-Pro", processing_status: "processing" });
    apiMocks.getTextToSpeechRecord.mockResolvedValue(record);
    const controller = new AbortController();
    let events!: SpeechHistoryRecordStreamCallbacks;
    apiMocks.attachTextToSpeechRecordStream.mockImplementation((_id, callbacks) => {
      events = callbacks;
      callbacks.onDurable?.();
      callbacks.onCreated?.(record);
      return controller;
    });
    renderRoute("/text-to-speech/existing-fish", { models: [buildFishModel(true)] }, true);
    const listen = await screen.findByRole("button", { name: "Listen from beginning" });
    expect(apiMocks.attachTextToSpeechRecordStream).not.toHaveBeenCalled();
    fireEvent.click(listen);
    expect(apiMocks.attachTextToSpeechRecordStream).toHaveBeenCalledWith("existing-fish", expect.any(Object));
    expect(apiMocks.createTextToSpeechRecordStream).not.toHaveBeenCalled();
    await act(async () => {
      events.onStart?.({ requestId: "existing-fish", sampleRate: 44100, audioFormat: "pcm_i16" });
      await events.onChunk?.({ requestId: "existing-fish", sequence: 0, sampleCount: 1, audioBase64: "AAA=" });
    });
    expect(screen.getByText("Playing generated speech")).toBeInTheDocument();
    fireEvent.click(within(screen.getByRole("navigation", { name: "Breadcrumb" })).getByRole("link", { name: "Text to Speech" }));
    expect(controller.signal.aborted).toBe(true);
    expect(apiMocks.cancelTextToSpeechRecord).not.toHaveBeenCalled();
  });

  it("offers no replay from the beginning for a model the server does not mark replayable", async () => {
    apiMocks.getTextToSpeechRecord.mockResolvedValue(
      buildRecord({ id: "existing-fish", model_id: "FishAudio-S2-Pro", processing_status: "processing" }),
    );
    renderRoute("/text-to-speech/existing-fish", { models: [buildFishModel(false)] });
    expect(await screen.findByTitle("Voice")).toHaveTextContent("Vivian");
    expect(screen.queryByRole("button", { name: "Listen from beginning" })).not.toBeInTheDocument();
    expect(apiMocks.attachTextToSpeechRecordStream).not.toHaveBeenCalled();
  });

  it.each([
    { withLayout: false, leave: false, durable: false },
    { withLayout: true, leave: false, durable: false },
    { withLayout: true, leave: true, durable: false },
    { withLayout: true, leave: true, durable: true },
    { withLayout: true, leave: false, durable: true },
  ])("keeps streaming through created navigation (layout: $withLayout, leave: $leave, durable: $durable)", async ({ withLayout, leave, durable }) => {
    let streamEvents!: SpeechHistoryRecordStreamCallbacks;
    const controller = new AbortController();
    if (durable) apiMocks.listTextToSpeechRecords.mockResolvedValue([buildSummary({ id: "tts-created-1", input_preview: "Durable narration" })]);
    apiMocks.getTextToSpeechRecord.mockResolvedValue(
      buildRecord({
        id: "tts-created-1",
        processing_status: "processing",
      }),
    );
    apiMocks.createTextToSpeechRecordStream.mockImplementation(
      (_request, callbacks) => {
        streamEvents = callbacks;
        if (durable) callbacks.onDurable?.();
        callbacks.onCreated?.(
          buildRecord({
            id: "tts-created-1",
            processing_status: "pending",
          }),
        );
        return controller;
      },
    );

    renderRoute("/text-to-speech", {}, withLayout);

    await waitFor(() =>
      expect(apiMocks.listTextToSpeechRecords).toHaveBeenCalled(),
    );
    fireEvent.click(screen.getByRole("button", { name: /New generation/i }));

    fireEvent.change(screen.getByRole("textbox", { name: "Script" }), {
      target: { value: "Hello from modal" },
    });
    fireEvent.click(screen.getByRole("button", { name: /Create generation/i }));

    await waitFor(() =>
      expect(apiMocks.createTextToSpeechRecordStream).toHaveBeenCalled(),
    );
    await waitFor(() =>
      expect(apiMocks.getTextToSpeechRecord).toHaveBeenCalledWith("tts-created-1"),
    );

    expect(
      await screen.findByRole("navigation", { name: "Breadcrumb" }),
    ).toBeInTheDocument();
    expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
    expect(controller.signal.aborted).toBe(false);
    expect(apiMocks.cancelTextToSpeechRecord).not.toHaveBeenCalled();
    await act(async () => {
      streamEvents.onStart?.({ requestId: "request", sampleRate: 44100, audioFormat: "pcm_i16" });
      await streamEvents.onChunk?.({ requestId: "request", sequence: 0, sampleCount: 1, audioBase64: "AAA=" });
    });
    expect(screen.getByText("Playing generated speech")).toBeInTheDocument();
    fireEvent.click(
      leave
        ? within(screen.getByRole("navigation", { name: "Breadcrumb" })).getByRole("link", { name: "Text to Speech" })
        : screen.getByRole("button", { name: "Stop generation and playback" }),
    );
    expect(controller.signal.aborted).toBe(true);
    // Leaving detaches: the server finishes the take into history.
    if (leave) {
      expect(apiMocks.cancelTextToSpeechRecord).not.toHaveBeenCalled();
    }
    if (durable && leave) {
      expect(apiMocks.cancelTextToSpeechRecord).not.toHaveBeenCalled();
      // Returning to the record must not revive stale transport callbacks or
      // send a cancellation while the durable worker continues independently.
      fireEvent.click(await screen.findByText("Durable narration"));
      expect(await screen.findByRole("navigation", { name: "Breadcrumb" })).toBeInTheDocument();
      expect(apiMocks.createTextToSpeechRecordStream).toHaveBeenCalledOnce();
      await act(async () => { streamEvents.onError?.("late disconnected callback"); });
      expect(apiMocks.cancelTextToSpeechRecord).not.toHaveBeenCalled();
      expect(screen.queryByText("late disconnected callback")).not.toBeInTheDocument();
    } else if (!leave) {
      // Stop cancels the take.
      expect(apiMocks.cancelTextToSpeechRecord).toHaveBeenCalledWith("tts-created-1");
    }
  });

  it("navigates to /text-to-speech/:id when stream emits final without created", async () => {
    apiMocks.getTextToSpeechRecord.mockResolvedValue(
      buildRecord({
        id: "tts-created-2",
        processing_status: "processing",
      }),
    );
    apiMocks.createTextToSpeechRecordStream.mockImplementation(
      (_request, callbacks) => {
        callbacks.onStart?.({
          requestId: "req-1",
          sampleRate: 24000,
          audioFormat: "pcm_i16",
        });
        callbacks.onFinal?.({
          record: buildRecord({
            id: "tts-created-2",
            processing_status: "processing",
          }),
          stats: {
            generation_time_ms: 0,
            audio_duration_secs: 0,
            rtf: 0,
            tokens_generated: 0,
          },
        });
        callbacks.onDone?.();
        return new AbortController();
      },
    );

    renderRoute("/text-to-speech");

    await waitFor(() =>
      expect(apiMocks.listTextToSpeechRecords).toHaveBeenCalled(),
    );
    fireEvent.click(screen.getByRole("button", { name: /New generation/i }));

    fireEvent.change(screen.getByRole("textbox", { name: "Script" }), {
      target: { value: "Hello from final event" },
    });
    fireEvent.click(screen.getByRole("button", { name: /Create generation/i }));

    await waitFor(() =>
      expect(apiMocks.createTextToSpeechRecordStream).toHaveBeenCalled(),
    );
    await waitFor(() =>
      expect(apiMocks.getTextToSpeechRecord).toHaveBeenCalledWith("tts-created-2"),
    );

    expect(
      await screen.findByRole("navigation", { name: "Breadcrumb" }),
    ).toBeInTheDocument();
  });

  it("shows a job created without a stream as rendering once the server starts it", async () => {
    hookMocks.useRouteModelSelection.mockReturnValue({
      ...hookMocks.useRouteModelSelection(),
      selectedModelInfo: {
        variant: "Qwen3-TTS-12Hz-1.7B-Chat",
        status: "ready",
        speech_capabilities: buildSpeechCapabilities({ supports_streaming: false }),
      },
    });
    apiMocks.createTextToSpeechRecord.mockResolvedValue(
      buildRecord({ id: "tts-batch", processing_status: "pending" }),
    );
    apiMocks.getTextToSpeechRecord.mockResolvedValue(
      buildRecord({ id: "tts-batch", processing_status: "processing" }),
    );

    renderRoute("/text-to-speech");

    await waitFor(() =>
      expect(apiMocks.listTextToSpeechRecords).toHaveBeenCalled(),
    );
    fireEvent.click(screen.getByRole("button", { name: /New generation/i }));
    fireEvent.change(screen.getByRole("textbox", { name: "Script" }), {
      target: { value: "A long script rendered as a job" },
    });
    fireEvent.click(screen.getByRole("button", { name: /Create generation/i }));

    await waitFor(() =>
      expect(apiMocks.createTextToSpeechRecord).toHaveBeenCalled(),
    );
    expect(apiMocks.createTextToSpeechRecordStream).not.toHaveBeenCalled();
    await waitFor(() =>
      expect(apiMocks.getTextToSpeechRecord).toHaveBeenCalledWith("tts-batch"),
    );
    expect(
      await screen.findByText(/actively rendering audio/),
    ).toBeInTheDocument();
    expect(screen.queryByText(/queued and will start shortly/)).toBeNull();
  });

  it("deletes from the record detail page and navigates back to history", async () => {
    apiMocks.listTextToSpeechRecords.mockResolvedValue([buildSummary()]);
    apiMocks.getTextToSpeechRecord.mockResolvedValue(
      buildRecord({
        processing_status: "ready",
        generation_time_ms: 120,
        audio_duration_secs: 2.5,
        rtf: 0.4,
        tokens_generated: 120,
      }),
    );

    renderRoute("/text-to-speech/tts-1");

    expect(
      await screen.findByRole("navigation", { name: "Breadcrumb" }),
    ).toBeInTheDocument();

    const recordMenu = await screen.findByRole("button", {
      name: /^More actions for /,
    });
    fireEvent.pointerDown(recordMenu, { button: 0, ctrlKey: false });
    fireEvent.click(recordMenu);
    fireEvent.click(await screen.findByRole("menuitem", { name: /^Delete$/i }));
    fireEvent.click(screen.getByRole("button", { name: /Delete generation/i }));

    await waitFor(() =>
      expect(apiMocks.deleteTextToSpeechRecord).toHaveBeenCalledWith("tts-1"),
    );

    expect(
      await screen.findByRole("heading", { name: "Text to Speech" }),
    ).toBeInTheDocument();
  });

  it("deletes from the history menu and refreshes the table", async () => {
    apiMocks.listTextToSpeechRecords
      .mockResolvedValueOnce([
        buildSummary({
          id: "tts-history-1",
          processing_status: "ready",
          audio_filename: "voice-note.wav",
        }),
      ])
      .mockResolvedValueOnce([]);
    apiMocks.deleteTextToSpeechRecord.mockResolvedValue({
      id: "tts-history-1",
      deleted: true,
    });

    renderRoute("/text-to-speech");

    expect(await screen.findByText("Hello world")).toBeInTheDocument();

    fireEvent.pointerDown(
      screen.getByRole("button", { name: "More actions for Hello world" }),
      { button: 0, ctrlKey: false },
    );
    fireEvent.click(await screen.findByRole("menuitem", { name: /^Delete$/i }));
    fireEvent.click(
      await screen.findByRole("button", { name: /Delete generation/i }),
    );

    await waitFor(() =>
      expect(apiMocks.deleteTextToSpeechRecord).toHaveBeenCalledWith(
        "tts-history-1",
      ),
    );
    await waitFor(() =>
      expect(apiMocks.listTextToSpeechRecords).toHaveBeenCalledTimes(2),
    );
    expect(
      await screen.findByRole("heading", { name: "No speech yet" }),
    ).toBeInTheDocument();
  });

  it("uses saved voice names on detail headers and removes header status badges", async () => {
    apiMocks.getTextToSpeechRecord.mockResolvedValue(
      buildRecord({
        saved_voice_id: "voice-1",
        speaker: null,
        processing_status: "ready",
        generation_time_ms: 120,
        audio_duration_secs: 2.5,
        rtf: 0.4,
        tokens_generated: 120,
      }),
    );
    apiMocks.listSavedVoices.mockResolvedValue([
      buildSavedVoice("voice-1", "Narrator Prime"),
    ]);

    renderRoute("/text-to-speech/tts-1");

    expect(
      await screen.findByRole("navigation", { name: "Breadcrumb" }),
    ).toBeInTheDocument();

    expect(await screen.findByTitle("Voice")).toHaveTextContent("Narrator Prime");
    expect(screen.queryByText(/^READY$/i)).not.toBeInTheDocument();
    expect(screen.queryByText(/Saved voice:\s*voice-1/i)).not.toBeInTheDocument();
  });

  it("uses built-in speaker display names on detail headers", async () => {
    apiMocks.getTextToSpeechRecord.mockResolvedValue(
      buildRecord({
        saved_voice_id: null,
        speaker: "Ono_anna",
        model_id: "Qwen3-TTS-12Hz-1.7B-Chat",
        processing_status: "ready",
      }),
    );

    renderRoute("/text-to-speech/tts-1");

    expect(
      await screen.findByRole("navigation", { name: "Breadcrumb" }),
    ).toBeInTheDocument();

    expect(await screen.findByTitle("Voice")).toHaveTextContent("Anna");
    expect(screen.queryByText("Voice: Ono_anna")).not.toBeInTheDocument();
  });

  describe("a Fish Audio Drama 3 dialogue take", () => {
    const drama3: ModelInfo = {
      variant: "FishAudio-Drama-3",
      status: "ready",
      local_path: null,
      size_bytes: null,
      download_progress: null,
      error_message: null,
      speech_capabilities: {
        supports_builtin_voices: false,
        built_in_voice_count: null,
        supports_reference_voice: true,
        reference_transcript_required: true,
        supports_voice_description: false,
        supports_streaming: true,
        supports_speed_control: false,
        supports_auto_long_form: true,
        supports_seeded_voice: true,
        supports_speaker_dialogue: true,
        max_speakers: 16,
        max_references: 8,
      },
    };
    const dialogueRecord = buildRecord({
      id: "scene-1",
      model_id: "FishAudio-Drama-3",
      speaker: null,
      processing_status: "ready",
      input_text:
        "<|speaker:0|> Ready when you are.\n<|speaker:2|> Always. [laughs] Let's go.",
    });

    function selectModel(variant: string, info: ModelInfo | null) {
      hookMocks.useRouteModelSelection.mockReturnValue({
        routeModels: [drama3],
        resolvedSelectedModel: variant,
        selectedModelInfo: info,
        selectedModelReady: true,
        isModelModalOpen: false,
        intentVariant: null,
        closeModelModal: vi.fn(),
        openModelManager: vi.fn(),
        requestModel: vi.fn(),
      });
    }

    it("shows in history like any record, without its speaker tags", async () => {
      apiMocks.listTextToSpeechRecords.mockResolvedValue([
        buildSummary({
          id: "scene-1",
          model_id: "FishAudio-Drama-3",
          speaker: null,
          processing_status: "ready",
          input_preview: "<|speaker:0|> Ready when you are.\n<|speaker:1|> Always.",
        }),
      ]);
      renderRoute("/text-to-speech", { models: [drama3] });
      expect(
        await screen.findByText("Ready when you are. Always."),
      ).toBeInTheDocument();
      expect(screen.queryByText(/<\|speaker:/)).not.toBeInTheDocument();
    });

    it("reopens in the dialogue editor, one row per line, in seeded voices", async () => {
      selectModel("FishAudio-Drama-3", drama3);
      apiMocks.getTextToSpeechRecord.mockResolvedValue(dialogueRecord);
      const onSelect = vi.fn();
      renderRoute("/text-to-speech/scene-1", { models: [drama3], onSelect });

      await openInDialogueEditor();

      const dialog = await screen.findByRole("dialog", {
        name: "New text-to-speech job",
      });
      expect(
        within(dialog).getByRole("radio", { name: "Dialogue" }),
      ).toHaveAttribute("aria-checked", "true");
      expect(
        within(dialog).getByRole("textbox", { name: "Line 1 text" }),
      ).toHaveValue("Ready when you are.");
      expect(
        within(dialog).getByRole("textbox", { name: "Line 2 text" }),
      ).toHaveValue("Always. [laughs] Let's go.");
      expect(
        within(dialog).getByRole("combobox", { name: "Line 2 speaker" }),
      ).toHaveTextContent("Speaker 2");
      // The record keeps no voices: both speakers start seeded, and say so.
      for (const speaker of ["Speaker 0 voice", "Speaker 2 voice"]) {
        expect(
          within(dialog).getByRole("combobox", { name: speaker }),
        ).toHaveTextContent("Seeded voice");
      }
      expect(within(dialog).getByRole("note")).toHaveTextContent(
        "pick their voices again",
      );
      expect(onSelect).not.toHaveBeenCalled();

      // Closed and opened again, it reads the take afresh.
      fireEvent.click(within(dialog).getByRole("button", { name: "Cancel" }));
      await openInDialogueEditor();
      expect(
        await screen.findByRole("textbox", { name: "Line 1 text" }),
      ).toHaveValue("Ready when you are.");
    });

    it("selects the take's model when another one is selected", async () => {
      apiMocks.getTextToSpeechRecord.mockResolvedValue(dialogueRecord);
      const onSelect = vi.fn();
      renderRoute("/text-to-speech/scene-1", { models: [drama3], onSelect });
      await openInDialogueEditor();
      expect(onSelect).toHaveBeenCalledWith("FishAudio-Drama-3");
    });

    it("is not offered for a single-voice take or a model without dialogue", async () => {
      apiMocks.getTextToSpeechRecord.mockResolvedValue(
        buildRecord({
          id: "plain",
          model_id: "FishAudio-Drama-3",
          input_text: "Just one voice.",
          processing_status: "ready",
        }),
      );
      const view = renderRoute("/text-to-speech/plain", { models: [drama3] });
      expect(
        await screen.findByRole("button", { name: /Copy text/ }),
      ).toBeInTheDocument();
      let menu = await openRecordMenu();
      expect(
        within(menu).getByRole("menuitem", { name: /^Delete$/ }),
      ).toBeInTheDocument();
      expect(
        within(menu).queryByRole("menuitem", {
          name: "Open in dialogue editor",
        }),
      ).toBeNull();
      view.unmount();

      apiMocks.getTextToSpeechRecord.mockResolvedValue({
        ...dialogueRecord,
        model_id: "FishAudio-S2-Pro",
      });
      renderRoute("/text-to-speech/scene-1", {
        models: [buildFishModel(false)],
      });
      expect(
        await screen.findByRole("button", { name: /Copy text/ }),
      ).toBeInTheDocument();
      menu = await openRecordMenu();
      expect(
        within(menu).getByRole("menuitem", { name: /^Delete$/ }),
      ).toBeInTheDocument();
      expect(
        within(menu).queryByRole("menuitem", {
          name: "Open in dialogue editor",
        }),
      ).toBeNull();
    });

    describe("take editing", () => {
      const editableScene = buildRecord({
        ...dialogueRecord,
        editable: true,
        references: [
          { speaker: 0, saved_voice_id: "voice-anna" },
          { speaker: 2 },
        ],
        seed: 77,
      });

      beforeEach(() => {
        apiMocks.listSavedVoices.mockResolvedValue([
          buildSavedVoice("voice-anna", "Anna"),
          buildSavedVoice("voice-ben", "Ben"),
        ]);
      });

      it("offers Edit this take only for a take the server marks editable", async () => {
        for (const take of [
          dialogueRecord,
          { ...dialogueRecord, editable: false },
          { ...dialogueRecord, processing_status: "pending" },
          buildRecord({ id: "kokoro", model_id: "Kokoro-82M", processing_status: "ready" }),
        ]) {
          apiMocks.getTextToSpeechRecord.mockResolvedValue(take);
          const view = renderRoute(`/text-to-speech/${take.id}`, {
            models: [drama3],
          });
          expect(await screen.findByTitle("Model")).toBeInTheDocument();
          expect(
            screen.queryByRole("button", { name: "Edit this take" }),
          ).toBeNull();
          view.unmount();
        }

        apiMocks.getTextToSpeechRecord.mockResolvedValue(editableScene);
        renderRoute("/text-to-speech/scene-1", { models: [drama3] });
        expect(
          await screen.findByRole("button", { name: "Edit this take" }),
        ).toBeInTheDocument();
      });

      it("edits a dialogue take: its lines, voices and seed, continuing from it", async () => {
        selectModel("FishAudio-Drama-3", drama3);
        apiMocks.getTextToSpeechRecord.mockResolvedValue(editableScene);
        renderRoute("/text-to-speech/scene-1", { models: [drama3] });

        fireEvent.click(
          await screen.findByRole("button", { name: "Edit this take" }),
        );
        const dialog = await screen.findByRole("dialog", { name: "Edit take" });
        expect(dialog).toHaveTextContent(
          "Editing a take: the new take continues from it.",
        );
        expect(
          within(dialog).getByRole("radio", { name: "Dialogue" }),
        ).toHaveAttribute("aria-checked", "true");
        const speaker0 = within(dialog).getByRole("combobox", {
          name: "Speaker 0 voice",
        });
        await waitFor(() => expect(speaker0).toHaveTextContent("Anna (Saved)"));
        // Speaker 2 spoke in a clip, which the take does not keep.
        expect(
          within(dialog).getByRole("combobox", { name: "Speaker 2 voice" }),
        ).toHaveTextContent("Seeded voice");
        expect(within(dialog).getByRole("note")).toHaveTextContent(
          "Speaker 2 spoke in a reference clip, which takes do not keep, so they start in a seeded voice.",
        );
        expect(within(dialog).getByLabelText("Seed")).toHaveValue("77");

        fireEvent.click(
          within(dialog).getByRole("button", { name: /Generate scene/ }),
        );
        await waitFor(() =>
          expect(apiMocks.createTextToSpeechRecordStream).toHaveBeenCalledTimes(1),
        );
        expect(apiMocks.createTextToSpeechRecordStream.mock.calls[0][0]).toEqual({
          model_id: "FishAudio-Drama-3",
          script: [
            { speaker: 0, text: "Ready when you are." },
            { speaker: 2, text: "Always. [laughs] Let's go." },
          ],
          references: [{ speaker: 0, saved_voice_id: "voice-anna" }],
          seed: 77,
          previous_take_ids: ["scene-1"],
        });
      });

      it("edits a single-voice take in the single-voice form", async () => {
        selectModel("FishAudio-Drama-3", drama3);
        apiMocks.getTextToSpeechRecord.mockResolvedValue(
          buildRecord({
            id: "plain",
            model_id: "FishAudio-Drama-3",
            speaker: null,
            processing_status: "ready",
            input_text: "Just one voice.",
            editable: true,
            references: [{ speaker: 0, saved_voice_id: "voice-ben" }],
            seed: 5,
          }),
        );
        renderRoute("/text-to-speech/plain", { models: [drama3] });

        fireEvent.click(
          await screen.findByRole("button", { name: "Edit this take" }),
        );
        const dialog = await screen.findByRole("dialog", { name: "Edit take" });
        expect(
          within(dialog).getByRole("radio", { name: "Single voice" }),
        ).toHaveAttribute("aria-checked", "true");
        expect(
          within(dialog).getByRole("textbox", { name: "Script" }),
        ).toHaveValue("Just one voice.");
        await waitFor(() =>
          expect(
            within(dialog).getByRole("combobox", { name: "Saved voice" }),
          ).toHaveTextContent("Ben (Saved)"),
        );
        expect(within(dialog).getByLabelText("Seed")).toHaveValue("5");
        expect(within(dialog).queryByRole("note")).toBeNull();

        fireEvent.click(
          within(dialog).getByRole("button", { name: /Create generation/ }),
        );
        await waitFor(() =>
          expect(apiMocks.createTextToSpeechRecordStream).toHaveBeenCalledTimes(1),
        );
        expect(apiMocks.createTextToSpeechRecordStream.mock.calls[0][0]).toEqual({
          model_id: "FishAudio-Drama-3",
          text: "Just one voice.",
          saved_voice_id: "voice-ben",
          seed: 5,
          previous_take_ids: ["plain"],
        });
      });

      it("restores a take's voices and seed in the dialogue editor, as a new take", async () => {
        selectModel("FishAudio-Drama-3", drama3);
        apiMocks.getTextToSpeechRecord.mockResolvedValue({
          ...editableScene,
          references: [
            { speaker: 0, saved_voice_id: "voice-anna" },
            { speaker: 2, saved_voice_id: "voice-ben" },
          ],
        });
        renderRoute("/text-to-speech/scene-1", { models: [drama3] });

        await openInDialogueEditor();
        const dialog = await screen.findByRole("dialog", {
          name: "New text-to-speech job",
        });
        const speaker0 = within(dialog).getByRole("combobox", {
          name: "Speaker 0 voice",
        });
        await waitFor(() => expect(speaker0).toHaveTextContent("Anna (Saved)"));
        expect(
          within(dialog).getByRole("combobox", { name: "Speaker 2 voice" }),
        ).toHaveTextContent("Ben (Saved)");
        expect(within(dialog).getByLabelText("Seed")).toHaveValue("77");
        // Everything came back: nothing to say.
        expect(within(dialog).queryByRole("note")).toBeNull();

        fireEvent.click(
          within(dialog).getByRole("button", { name: /Generate scene/ }),
        );
        await waitFor(() =>
          expect(apiMocks.createTextToSpeechRecordStream).toHaveBeenCalledTimes(1),
        );
        const request = apiMocks.createTextToSpeechRecordStream.mock.calls[0][0];
        expect(request).toMatchObject({
          references: [
            { speaker: 0, saved_voice_id: "voice-anna" },
            { speaker: 2, saved_voice_id: "voice-ben" },
          ],
          seed: 77,
        });
        expect(request.previous_take_ids).toBeUndefined();
      });

      it("links an edited take back to the take it was edited from", async () => {
        apiMocks.getTextToSpeechRecord.mockImplementation(async (id: string) =>
          id === "scene-0"
            ? buildRecord({
                id: "scene-0",
                model_id: "FishAudio-Drama-3",
                processing_status: "ready",
                input_text: "<|speaker:0|> First try.\n<|speaker:2|> Again.",
              })
            : { ...editableScene, parent_take_id: "scene-0" },
        );
        renderRoute("/text-to-speech/scene-1", { models: [drama3] });

        // Titled like the parent's page: its words, without speaker tags.
        const link = await screen.findByRole("link", {
          name: "First try. Again.",
        });
        expect(link).toHaveAttribute("href", "/text-to-speech/scene-0");
        expect(screen.getByText("Edited from")).toBeInTheDocument();
      });

      it("says when the take it was edited from is deleted", async () => {
        apiMocks.getTextToSpeechRecord.mockImplementation(async (id: string) => {
          if (id === "scene-0") {
            throw new Error("Speech history record not found");
          }
          return { ...editableScene, parent_take_id: "scene-0" };
        });
        renderRoute("/text-to-speech/scene-1", { models: [drama3] });

        expect(await screen.findByText("Deleted take")).toBeInTheDocument();
        expect(screen.queryByRole("link", { name: /Earlier take/ })).toBeNull();
      });
    });
  });
});
