import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { beforeEach, describe, expect, it, vi } from "vitest";

import type { ModelInfo, SavedVoiceSummary } from "@/api";

import { VoicesPage } from "./route";

const apiMocks = vi.hoisted(() => ({
  listSavedVoices: vi.fn(),
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
    listSavedVoices: apiMocks.listSavedVoices,
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

describe("VoicesPage", () => {
  beforeEach(() => {
    apiMocks.listSavedVoices.mockReset();
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

  it("renders saved voice actions and shows built-in preview loading state", async () => {
    render(
      <MemoryRouter>
        <VoicesPage {...baseProps} />
      </MemoryRouter>,
    );

    await waitFor(() =>
      expect(apiMocks.listSavedVoices).toHaveBeenCalled(),
    );

    expect(screen.getByText("Balanced 21 yo")).toBeInTheDocument();
    expect(screen.getByText("Designed voice")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Use in TTS" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Delete" })).toBeInTheDocument();

    const builtInTab = screen.getByRole("tab", { name: /Built-in Voices/i });
    fireEvent.mouseDown(builtInTab, { button: 0 });
    fireEvent.click(builtInTab);

    await waitFor(() =>
      expect(builtInTab).toHaveAttribute("data-state", "active"),
    );

    expect(await screen.findByText("Alloy")).toBeInTheDocument();
    expect(screen.getAllByText("Built-in voice").length).toBeGreaterThan(0);
    expect(screen.getAllByText("MockVoiceModel").length).toBeGreaterThan(0);

    fireEvent.click(screen.getByRole("button", { name: "Preview" }));

    await waitFor(() =>
      expect(apiMocks.generateTTSWithStats).toHaveBeenCalledWith({
        model_id: "MockVoiceModel",
        text: "Hello. This is an Izwi built-in voice preview.",
        speaker: "alloy",
      }),
    );

    expect(
      await screen.findByText("Generating a preview sample for this speaker."),
    ).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Preview" })).toBeDisabled();
  });

  it("supports the all-voices tab with sidebar filter and search controls", async () => {
    render(
      <MemoryRouter>
        <VoicesPage {...baseProps} />
      </MemoryRouter>,
    );

    await waitFor(() => expect(apiMocks.listSavedVoices).toHaveBeenCalled());

    const allVoicesTab = screen.getByRole("tab", { name: /All Voices/i });
    fireEvent.mouseDown(allVoicesTab, { button: 0 });
    fireEvent.click(allVoicesTab);

    await waitFor(() =>
      expect(allVoicesTab).toHaveAttribute("data-state", "active"),
    );

    const designedFilter = screen.getByRole("radio", { name: "Designed" });
    fireEvent.click(designedFilter);
    expect(designedFilter).toHaveAttribute("aria-checked", "true");

    fireEvent.change(screen.getByPlaceholderText("Search voices by name or notes"), {
      target: { value: "balanced" },
    });

    expect(await screen.findByText("Balanced 21 yo")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /Add New Voice/i })).toBeInTheDocument();
  });
});
