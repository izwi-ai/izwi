import { fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { MemoryRouter, Route, Routes, useLocation } from "react-router";
import { beforeEach, describe, expect, it, vi } from "vitest";

import type { ModelInfo, SavedVoiceSummary } from "@/api";

import { VoicesPage } from "./route";

const apiMocks = vi.hoisted(() => ({
  listSavedVoices: vi.fn(),
  listSavedVoicePage: vi.fn(),
  deleteSavedVoice: vi.fn(),
  savedVoiceAudioUrl: vi.fn(),
  generateTTSWithStats: vi.fn(),
}));

const hookMocks = vi.hoisted(() => ({
  useRouteModelSelection: vi.fn(),
}));

const typeMocks = vi.hoisted(() => ({
  getSpeakerProfilesForVariant: vi.fn(),
  isLfm25AudioVariant: vi.fn(),
}));

vi.mock("@/api", () => ({
  api: {
    getHealth: () => Promise.reject(new Error("offline")),
    listSavedVoices: apiMocks.listSavedVoices,
    listSavedVoicePage: apiMocks.listSavedVoicePage,
    deleteSavedVoice: apiMocks.deleteSavedVoice,
    savedVoiceAudioUrl: apiMocks.savedVoiceAudioUrl,
    generateTTSWithStats: apiMocks.generateTTSWithStats,
  },
}));

vi.mock("@/features/models/hooks/useRouteModelSelection", () => ({
  useRouteModelSelection: hookMocks.useRouteModelSelection,
}));

vi.mock("@/types", async () => {
  const actual = await vi.importActual<typeof import("@/types")>("@/types");
  return {
    ...actual,
    getSpeakerProfilesForVariant: typeMocks.getSpeakerProfilesForVariant,
    isLfm25AudioVariant: typeMocks.isLfm25AudioVariant,
  };
});

function buildModel(overrides: Partial<ModelInfo> = {}): ModelInfo {
  return {
    variant: "MockVoiceModel",
    status: "ready",
    local_path: "/tmp/mock-voice-model",
    size_bytes: 1_000,
    download_progress: null,
    error_message: null,
    speech_capabilities: {
      supports_builtin_voices: true,
      built_in_voice_count: 1,
      supports_reference_voice: true,
      supports_voice_description: false,
      supports_streaming: true,
      supports_speed_control: true,
      supports_auto_long_form: false,
    },
    ...overrides,
  };
}

function LocationSearchEcho() {
  const location = useLocation();
  return <div data-testid="tts-location-search">{location.search}</div>;
}

describe("VoicesPage", () => {
  beforeEach(() => {
    apiMocks.listSavedVoices.mockReset();
    apiMocks.listSavedVoicePage.mockReset();
    apiMocks.deleteSavedVoice.mockReset();
    apiMocks.savedVoiceAudioUrl.mockReset();
    apiMocks.generateTTSWithStats.mockReset();
    hookMocks.useRouteModelSelection.mockReset();
    typeMocks.getSpeakerProfilesForVariant.mockReset();
    typeMocks.isLfm25AudioVariant.mockReset();

    apiMocks.savedVoiceAudioUrl.mockImplementation((voiceId: string) => {
      return `/voices/${voiceId}/audio`;
    });
    apiMocks.listSavedVoices.mockResolvedValue([
      {
        id: "voice-balanced",
        created_at: 1711000000000,
        updated_at: 1711100000000,
        name: "Balanced 21 yo",
        reference_text_preview:
          "Hello, this is Izwi. This short preview helps compare the voice.",
        reference_text_chars: 64,
        audio_mime_type: "audio/wav",
        audio_filename: "balanced.wav",
        source_route_kind: "voice_design",
        source_record_id: "design-1",
      } satisfies SavedVoiceSummary,
    ]);
    apiMocks.deleteSavedVoice.mockResolvedValue(undefined);
    // The server searches saved voices by name and reference text.
    apiMocks.listSavedVoicePage.mockImplementation(
      async (query?: { filters?: { q?: string } }) => ({
      items: ((await apiMocks.listSavedVoices()) as SavedVoiceSummary[]).filter(
        (voice) =>
          !query?.filters?.q ||
          voice.name.toLowerCase().includes(query.filters.q.toLowerCase()),
      ),
      pagination: {
        next_cursor: null,
        has_more: false,
        limit: 25,
      },
    }),
    );
    apiMocks.generateTTSWithStats.mockImplementation(
      () => new Promise(() => {}),
    );

    hookMocks.useRouteModelSelection.mockReturnValue({
      routeModels: [buildModel()],
      resolvedSelectedModel: "MockVoiceModel",
      selectedModelInfo: buildModel(),
      selectedModelReady: true,
      isModelModalOpen: false,
      intentVariant: null,
      closeModelModal: vi.fn(),
      openModelManager: vi.fn(),
      requestModel: vi.fn(),
      handleModelSelect: vi.fn(),
    });

    typeMocks.getSpeakerProfilesForVariant.mockReturnValue([
      {
        id: "alloy",
        name: "Alloy",
        language: "English",
        description: "Warm and balanced built-in speaker",
      },
    ]);
    typeMocks.isLfm25AudioVariant.mockReturnValue(false);

    Object.defineProperty(URL, "createObjectURL", {
      writable: true,
      value: vi.fn(() => "blob:voice-preview"),
    });
    Object.defineProperty(URL, "revokeObjectURL", {
      writable: true,
      value: vi.fn(),
    });
  });

  const baseProps = {
    models: [buildModel()],
    selectedModel: "MockVoiceModel",
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

  it("renders voice cards and keeps the built-in preview flow", async () => {
    render(
      <MemoryRouter>
        <VoicesPage {...baseProps} />
      </MemoryRouter>,
    );

    await waitFor(() =>
    expect(apiMocks.listSavedVoices).toHaveBeenCalled(),
    );

    expect(screen.getByRole("list", { name: "Voices" })).toBeInTheDocument();
    expect(screen.getByTestId("voice-card-voice-balanced")).toBeInTheDocument();

    expect(screen.getByText("Balanced 21 yo")).toBeInTheDocument();
    expect(screen.getByText(/^Designed · /)).toBeInTheDocument();
    expect(screen.getByText("Designed")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Use in TTS" })).toBeInTheDocument();
    const menu = screen.getByRole("button", {
      name: "More actions for Balanced 21 yo",
    });
    fireEvent.pointerDown(menu, { button: 0, ctrlKey: false });
    fireEvent.click(menu);
    expect(
      await screen.findByRole("menuitem", { name: "Delete" }),
    ).toBeInTheDocument();
    fireEvent.keyDown(document.activeElement ?? document.body, { key: "Escape" });

    const builtInFilter = screen.getByRole("radio", { name: /^Built-in/ });
    fireEvent.click(builtInFilter);
    await waitFor(() => expect(builtInFilter).toBeChecked());

    expect(screen.getByTestId("voice-card-alloy")).toBeInTheDocument();
    expect(await screen.findByText("Alloy")).toBeInTheDocument();
    expect(screen.getAllByText("Built-in").length).toBeGreaterThan(0);

    // Play on a built-in voice generates its preview first.
    fireEvent.click(
      screen.getByRole("button", { name: "Play preview for Alloy" }),
    );

    await waitFor(() =>
      expect(apiMocks.generateTTSWithStats).toHaveBeenCalledWith({
        model_id: "MockVoiceModel",
        text: "Hello. This is an Izwi built-in voice preview.",
        speaker: "alloy",
      }),
    );

    await waitFor(() =>
      expect(
        screen.getByRole("button", { name: "Play preview for Alloy" }),
      ).toBeDisabled(),
    );
  });

  it("includes a saved-voice-capable model when redirecting saved voices to text to speech", async () => {
    render(
      <MemoryRouter initialEntries={["/voices"]}>
        <Routes>
          <Route path="/voices" element={<VoicesPage {...baseProps} />} />
          <Route path="/text-to-speech" element={<LocationSearchEcho />} />
        </Routes>
      </MemoryRouter>,
    );

    await waitFor(() => expect(apiMocks.listSavedVoices).toHaveBeenCalled());

    fireEvent.click(screen.getByRole("button", { name: "Use in TTS" }));

    const searchEcho = await screen.findByTestId("tts-location-search");
    expect(searchEcho).toHaveTextContent("voiceId=voice-balanced");
    expect(searchEcho).toHaveTextContent("model=MockVoiceModel");
  });

  it("loads more saved voices", async () => {
    apiMocks.listSavedVoicePage.mockReset();
    apiMocks.listSavedVoicePage
      .mockResolvedValueOnce({
        items: [
          {
            id: "voice-page-1",
            created_at: 1711000000000,
            updated_at: 1711100000000,
            name: "Page One Voice",
            reference_text_preview: "Page one",
            reference_text_chars: 8,
            audio_mime_type: "audio/wav",
            audio_filename: "page-one.wav",
            source_route_kind: "voice_design",
            source_record_id: "design-1",
          } satisfies SavedVoiceSummary,
        ],
        pagination: {
          next_cursor: "voice-cursor-2",
          has_more: true,
          limit: 25,
        },
      })
      .mockResolvedValueOnce({
        items: [
          {
            id: "voice-page-2",
            created_at: 1712000000000,
            updated_at: 1712100000000,
            name: "Page Two Voice",
            reference_text_preview: "Page two",
            reference_text_chars: 8,
            audio_mime_type: "audio/wav",
            audio_filename: "page-two.wav",
            source_route_kind: "voice_cloning",
            source_record_id: "clone-2",
          } satisfies SavedVoiceSummary,
        ],
        pagination: {
          next_cursor: null,
          has_more: false,
          limit: 25,
        },
      });

    render(
      <MemoryRouter>
        <VoicesPage {...baseProps} />
      </MemoryRouter>,
    );

    expect(await screen.findByText("Page One Voice")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Load more" }));
    expect(await screen.findByText("Page Two Voice")).toBeInTheDocument();
    expect(await screen.findByText("Page One Voice")).toBeInTheDocument();

    expect(apiMocks.listSavedVoicePage).toHaveBeenNthCalledWith(1, {
      limit: 25,
      cursor: null,
    });
    expect(apiMocks.listSavedVoicePage).toHaveBeenNthCalledWith(2, {
      limit: 25,
      cursor: "voice-cursor-2",
    });
  });

  it("uses a confirmation modal before deleting saved voices", async () => {
    apiMocks.listSavedVoices
      .mockResolvedValueOnce([
        {
          id: "voice-balanced",
          created_at: 1711000000000,
          updated_at: 1711100000000,
          name: "Balanced 21 yo",
          reference_text_preview:
            "Hello, this is Izwi. This short preview helps compare the voice.",
          reference_text_chars: 64,
          audio_mime_type: "audio/wav",
          audio_filename: "balanced.wav",
          source_route_kind: "voice_design",
          source_record_id: "design-1",
        } satisfies SavedVoiceSummary,
      ])
      .mockResolvedValueOnce([]);

    render(
      <MemoryRouter>
        <VoicesPage {...baseProps} />
      </MemoryRouter>,
    );

    await waitFor(() => expect(apiMocks.listSavedVoices).toHaveBeenCalled());

    const menu = screen.getByRole("button", {
      name: "More actions for Balanced 21 yo",
    });
    fireEvent.pointerDown(menu, { button: 0, ctrlKey: false });
    fireEvent.click(menu);
    fireEvent.click(await screen.findByRole("menuitem", { name: "Delete" }));

    expect(
      await screen.findByRole("dialog", { name: "Delete voice?" }),
    ).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Delete voice" }));

    await waitFor(() =>
      expect(apiMocks.deleteSavedVoice).toHaveBeenCalledWith("voice-balanced"),
    );
    await waitFor(() => expect(apiMocks.listSavedVoices).toHaveBeenCalledTimes(2));
    await waitFor(() =>
      expect(screen.queryByTestId("voice-card-voice-balanced")).not.toBeInTheDocument(),
    );
  });

  it("searches saved voices on the server and built-in voices here", async () => {
    render(
      <MemoryRouter>
        <VoicesPage {...baseProps} />
      </MemoryRouter>,
    );

    await waitFor(() => expect(apiMocks.listSavedVoices).toHaveBeenCalled());
    fireEvent.click(screen.getByRole("radio", { name: /^All/ }));
    expect(await screen.findByTestId("voice-card-voice-balanced")).toBeInTheDocument();

    fireEvent.change(screen.getByRole("searchbox", { name: "Search voices" }), {
      target: { value: "alloy" },
    });

    await waitFor(() =>
      expect(screen.queryByTestId("voice-card-voice-balanced")).not.toBeInTheDocument(),
    );
    expect(screen.getByTestId("voice-card-alloy")).toBeInTheDocument();
    expect(apiMocks.listSavedVoicePage).toHaveBeenLastCalledWith(
      expect.objectContaining({ filters: { q: "alloy" } }),
    );

    fireEvent.change(screen.getByRole("searchbox", { name: "Search voices" }), {
      target: { value: "nothing like it" },
    });
    expect(await screen.findByText("No matches")).toBeInTheDocument();
  });

  it("keeps only top-level model action while removing secondary strips", async () => {
    render(
      <MemoryRouter>
        <VoicesPage {...baseProps} />
      </MemoryRouter>,
    );

    await waitFor(() => expect(apiMocks.listSavedVoices).toHaveBeenCalled());

    const allVoicesFilter = screen.getByRole("radio", { name: /^All/ });
    fireEvent.click(allVoicesFilter);
    await waitFor(() => expect(allVoicesFilter).toBeChecked());

    expect(await screen.findByText("Balanced 21 yo")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /Models/i })).toBeInTheDocument();
    expect(screen.queryByRole("radio", { name: "Cloned" })).not.toBeInTheDocument();
    expect(screen.queryByRole("radio", { name: "Designed" })).not.toBeInTheDocument();
    expect(
      screen.queryByPlaceholderText("Search voices by name or notes"),
    ).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /Refresh/i })).not.toBeInTheDocument();
    expect(screen.queryByText("Library Statistics")).not.toBeInTheDocument();
    expect(screen.queryByText("Filter By")).not.toBeInTheDocument();
    expect(
      screen.queryByRole("columnheader", { name: "Metadata" }),
    ).not.toBeInTheDocument();
    expect(
      screen.queryByRole("button", { name: /Add New Voice/i }),
    ).not.toBeInTheDocument();
    expect(
      screen.queryByText(
        "Select and load a supported voice model to generate built-in previews.",
      ),
    ).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Model" })).not.toBeInTheDocument();
    expect(screen.queryByText(/voices listed/i)).not.toBeInTheDocument();
    expect(
      screen.queryByText(
        /Scroll horizontally for preview and actions on narrow screens/i,
      ),
    ).not.toBeInTheDocument();
    expect(screen.queryByText(/RESULTS/)).not.toBeInTheDocument();
    expect(screen.queryByText(/SAVED\s+\d+/)).not.toBeInTheDocument();
    expect(screen.queryByText(/CLONED\s+\d+/)).not.toBeInTheDocument();
    expect(screen.queryByText(/DESIGNED\s+\d+/)).not.toBeInTheDocument();
    expect(screen.queryByText(/BUILT-IN\s+\d+/)).not.toBeInTheDocument();
  });

  it("switches built-in model selection and updates built-in voice previews", async () => {
    const onSelect = vi.fn();
    const kokoroVariant = "Kokoro-82M";
    const qwenVariant = "Qwen3-TTS-12Hz-0.6B-CustomVoice";
    let isModelModalOpen = false;
    const openModelManager = vi.fn(() => {
      isModelModalOpen = true;
    });
    const closeModelModal = vi.fn(() => {
      isModelModalOpen = false;
    });

    typeMocks.getSpeakerProfilesForVariant.mockImplementation((variant: string) => {
      if (variant === kokoroVariant) {
        return [
          {
            id: "bf_alice",
            name: "Alice",
            language: "English",
            description: "Kokoro built-in voice",
          },
        ];
      }
      if (variant === qwenVariant) {
        return [
          {
            id: "Chelsie",
            name: "Chelsie",
            language: "English",
            description: "Qwen built-in voice",
          },
        ];
      }
      return [];
    });

    hookMocks.useRouteModelSelection.mockImplementation(
      ({ selectedModel: currentSelectedModel }: { selectedModel: string | null }) => ({
        routeModels: [
          buildModel({ variant: kokoroVariant }),
          buildModel({ variant: qwenVariant }),
        ],
        resolvedSelectedModel: currentSelectedModel ?? kokoroVariant,
        selectedModelInfo: buildModel({
          variant: currentSelectedModel ?? kokoroVariant,
        }),
        selectedModelReady: true,
        isModelModalOpen,
        intentVariant: null,
        closeModelModal,
        openModelManager,
        requestModel: vi.fn(),
        handleModelSelect: onSelect,
        modelOptions: [
          {
            value: kokoroVariant,
            label: kokoroVariant,
            statusLabel: "Loaded",
            isReady: true,
          },
          {
            value: qwenVariant,
            label: qwenVariant,
            statusLabel: "Loaded",
            isReady: true,
          },
        ],
      }),
    );

    const { rerender } = render(
      <MemoryRouter>
        <VoicesPage {...baseProps} selectedModel={kokoroVariant} onSelect={onSelect} />
      </MemoryRouter>,
    );

    await waitFor(() => expect(apiMocks.listSavedVoices).toHaveBeenCalled());

    const builtInFilter = screen.getByRole("radio", { name: /^Built-in/ });
    fireEvent.click(builtInFilter);
    await waitFor(() => expect(builtInFilter).toBeChecked());

    expect(await screen.findByText("Alice")).toBeInTheDocument();
    expect(screen.queryByText("Chelsie")).not.toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: /Models/i }));
    expect(openModelManager).toHaveBeenCalled();

    rerender(
      <MemoryRouter>
        <VoicesPage {...baseProps} selectedModel={kokoroVariant} onSelect={onSelect} />
      </MemoryRouter>,
    );

    const qwenModelRow = await screen.findByTestId(
      `model-row-${qwenVariant}`,
    );
    fireEvent.click(
      within(qwenModelRow).getByRole("button", { name: /^Use / }),
    );
    expect(onSelect).toHaveBeenCalledWith(qwenVariant);

    rerender(
      <MemoryRouter>
        <VoicesPage {...baseProps} selectedModel={qwenVariant} onSelect={onSelect} />
      </MemoryRouter>,
    );

    expect(await screen.findByText("Chelsie")).toBeInTheDocument();
    expect(screen.queryByText("Alice")).not.toBeInTheDocument();

    fireEvent.click(
      screen.getByRole("button", { name: "Play preview for Chelsie" }),
    );

    await waitFor(() =>
      expect(apiMocks.generateTTSWithStats).toHaveBeenCalledWith({
        model_id: qwenVariant,
        text: "Hello. This is an Izwi built-in voice preview.",
        speaker: "Chelsie",
      }),
    );
  });
});
