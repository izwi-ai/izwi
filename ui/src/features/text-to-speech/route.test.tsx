import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { MemoryRouter, Route, Routes } from "react-router-dom";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { NotificationProvider } from "@/app/providers/NotificationProvider";
import { TextToSpeechPage } from "./route";

const apiMocks = vi.hoisted(() => ({
  listTextToSpeechRecords: vi.fn(),
  listTextToSpeechRecordPage: vi.fn(),
  getTextToSpeechRecord: vi.fn(),
  textToSpeechRecordAudioUrl: vi.fn(),
  deleteTextToSpeechRecord: vi.fn(),
  createTextToSpeechRecord: vi.fn(),
  createTextToSpeechRecordStream: vi.fn(),
  listSavedVoices: vi.fn(),
  downloadAudioFile: vi.fn(),
}));

const hookMocks = vi.hoisted(() => ({
  useRouteModelSelection: vi.fn(),
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
    listSavedVoices: apiMocks.listSavedVoices,
    downloadAudioFile: apiMocks.downloadAudioFile,
  },
}));

vi.mock("@/features/models/hooks/useRouteModelSelection", () => ({
  useRouteModelSelection: hookMocks.useRouteModelSelection,
}));

vi.mock("@/features/models/components/RouteModelModal", () => ({
  RouteModelModal: () => null,
}));

const baseProps = {
  models: [],
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

function renderRoute(
  initialEntry: string,
  propsOverrides: Partial<typeof baseProps> = {},
) {
  const routeProps = { ...baseProps, ...propsOverrides };
  return render(
    <NotificationProvider>
      <MemoryRouter initialEntries={[initialEntry]}>
        <Routes>
          <Route
            path="/text-to-speech"
            element={<TextToSpeechPage {...routeProps} />}
          />
          <Route
            path="/text-to-speech/:recordId"
            element={<TextToSpeechPage {...routeProps} />}
          />
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

describe("TextToSpeechPage", () => {
  beforeEach(() => {
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
      screen.getByRole("heading", { name: "No text-to-speech jobs yet" }),
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
    });
    expect(apiMocks.listTextToSpeechRecordPage).toHaveBeenNthCalledWith(2, {
      limit: 25,
      cursor: "tts-cursor-2",
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
      screen.getByRole("button", { name: /More actions for voice-note\.wav/i }),
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
    expect(screen.getByText("Enter text for generation")).toBeInTheDocument();
    expect(screen.getByText("Review settings")).toBeInTheDocument();
    expect(screen.getByText("Built-in voice")).toBeInTheDocument();
    expect(screen.getByText("Vivian")).toBeInTheDocument();
    expect(screen.queryByText("Saved voice")).not.toBeInTheDocument();
    expect(screen.queryByText("Voice direction")).not.toBeInTheDocument();
    expect(
      screen.queryByPlaceholderText("Optional style guidance"),
    ).not.toBeInTheDocument();
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

    fireEvent.change(screen.getByPlaceholderText("Write the text to speak..."), {
      target: { value: "Temporary draft text" },
    });

    const streamToggle = screen.getByRole("checkbox");
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

    expect(screen.getByPlaceholderText("Write the text to speak...")).toHaveValue("");
    expect(screen.getByRole("checkbox")).toBeChecked();
    expect(
      screen.queryByText("Enter text to generate speech."),
    ).not.toBeInTheDocument();
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
    expect(screen.getByText("Select saved voice")).toBeInTheDocument();
    expect(screen.queryByText("Built-in voice")).not.toBeInTheDocument();
    expect(screen.queryByText("Voice direction")).not.toBeInTheDocument();
    expect(
      screen.queryByPlaceholderText("Optional style guidance"),
    ).not.toBeInTheDocument();
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

    expect(screen.getByText("Alice")).toBeInTheDocument();
    expect(screen.queryByText("bf_alice")).not.toBeInTheDocument();
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

    fireEvent.change(screen.getByPlaceholderText("Write the text to speak..."), {
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

    fireEvent.click(screen.getByRole("button", { name: /Load model/i }));
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

    fireEvent.click(screen.getByRole("button", { name: /Open models/i }));
    expect(openModelManager).toHaveBeenCalledTimes(1);
  });

  it("unloads the selected model from the modal readiness action", async () => {
    const onUnload = vi.fn();

    renderRoute("/text-to-speech", { onUnload });

    await waitFor(() =>
      expect(apiMocks.listTextToSpeechRecords).toHaveBeenCalled(),
    );

    fireEvent.click(screen.getByRole("button", { name: /New generation/i }));
    await screen.findByRole("heading", { name: "New text-to-speech job" });

    fireEvent.click(screen.getByRole("button", { name: /Unload model/i }));
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

  it("navigates to /text-to-speech/:id after stream created event", async () => {
    apiMocks.getTextToSpeechRecord.mockResolvedValue(
      buildRecord({
        id: "tts-created-1",
        processing_status: "processing",
      }),
    );
    apiMocks.createTextToSpeechRecordStream.mockImplementation(
      (_request, callbacks) => {
        callbacks.onCreated?.(
          buildRecord({
            id: "tts-created-1",
            processing_status: "pending",
          }),
        );
        return new AbortController();
      },
    );

    renderRoute("/text-to-speech");

    await waitFor(() =>
      expect(apiMocks.listTextToSpeechRecords).toHaveBeenCalled(),
    );
    fireEvent.click(screen.getByRole("button", { name: /New generation/i }));

    fireEvent.change(screen.getByPlaceholderText("Write the text to speak..."), {
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
      await screen.findByRole("heading", { name: "Text-to-Speech Record" }),
    ).toBeInTheDocument();
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

    fireEvent.change(screen.getByPlaceholderText("Write the text to speak..."), {
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
      await screen.findByRole("heading", { name: "Text-to-Speech Record" }),
    ).toBeInTheDocument();
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
      await screen.findByRole("heading", { name: "Text-to-Speech Record" }),
    ).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: /^Delete$/i }));
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
      screen.getByRole("button", { name: /More actions for voice-note\.wav/i }),
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
      await screen.findByRole("heading", { name: "No text-to-speech jobs yet" }),
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
      await screen.findByRole("heading", { name: "Text-to-Speech Record" }),
    ).toBeInTheDocument();

    expect(await screen.findByText("Voice: Narrator Prime")).toBeInTheDocument();
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
      await screen.findByRole("heading", { name: "Text-to-Speech Record" }),
    ).toBeInTheDocument();

    expect(await screen.findByText("Voice: Anna")).toBeInTheDocument();
    expect(screen.queryByText("Voice: Ono_anna")).not.toBeInTheDocument();
  });
});
