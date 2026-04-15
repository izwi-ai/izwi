import { render, screen, waitFor, within } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { NotificationProvider } from "@/app/providers/NotificationProvider";
import { StudioPage } from "./studio/route";
import { VoiceCloningPage } from "./voice-cloning/route";
import { TranscriptionPage } from "./transcription/route";

const apiMocks = vi.hoisted(() => ({
  listSpeechHistoryRecords: vi.fn(),
  listTextToSpeechRecordPage: vi.fn(),
  getSpeechHistoryRecord: vi.fn(),
  deleteSpeechHistoryRecord: vi.fn(),
  speechHistoryRecordAudioUrl: vi.fn(),
  listDiarizationRecords: vi.fn(),
  listDiarizationRecordPage: vi.fn(),
  getDiarizationRecord: vi.fn(),
  deleteDiarizationRecord: vi.fn(),
  diarizationRecordAudioUrl: vi.fn(),
  listTranscriptionRecords: vi.fn(),
  listSpeechTextJobPage: vi.fn(),
  getTranscriptionRecord: vi.fn(),
  deleteTranscriptionRecord: vi.fn(),
  transcriptionRecordAudioUrl: vi.fn(),
  listStudioProjects: vi.fn(),
  listStudioProjectPage: vi.fn(),
  listSavedVoices: vi.fn(),
}));

vi.mock("@/api", () => ({
  api: {
    listSpeechHistoryRecords: apiMocks.listSpeechHistoryRecords,
    listTextToSpeechRecordPage: apiMocks.listTextToSpeechRecordPage,
    getSpeechHistoryRecord: apiMocks.getSpeechHistoryRecord,
    deleteSpeechHistoryRecord: apiMocks.deleteSpeechHistoryRecord,
    speechHistoryRecordAudioUrl: apiMocks.speechHistoryRecordAudioUrl,
    listDiarizationRecords: apiMocks.listDiarizationRecords,
    listDiarizationRecordPage: apiMocks.listDiarizationRecordPage,
    getDiarizationRecord: apiMocks.getDiarizationRecord,
    deleteDiarizationRecord: apiMocks.deleteDiarizationRecord,
    diarizationRecordAudioUrl: apiMocks.diarizationRecordAudioUrl,
    listTranscriptionRecords: apiMocks.listTranscriptionRecords,
    listSpeechTextJobPage: apiMocks.listSpeechTextJobPage,
    getTranscriptionRecord: apiMocks.getTranscriptionRecord,
    deleteTranscriptionRecord: apiMocks.deleteTranscriptionRecord,
    transcriptionRecordAudioUrl: apiMocks.transcriptionRecordAudioUrl,
    listStudioProjects: apiMocks.listStudioProjects,
    listStudioProjectPage: apiMocks.listStudioProjectPage,
    listSavedVoices: apiMocks.listSavedVoices,
  },
}));
vi.mock("../api", () => ({
  api: {
    listSpeechHistoryRecords: apiMocks.listSpeechHistoryRecords,
    listTextToSpeechRecordPage: apiMocks.listTextToSpeechRecordPage,
    getSpeechHistoryRecord: apiMocks.getSpeechHistoryRecord,
    deleteSpeechHistoryRecord: apiMocks.deleteSpeechHistoryRecord,
    speechHistoryRecordAudioUrl: apiMocks.speechHistoryRecordAudioUrl,
    listDiarizationRecords: apiMocks.listDiarizationRecords,
    listDiarizationRecordPage: apiMocks.listDiarizationRecordPage,
    getDiarizationRecord: apiMocks.getDiarizationRecord,
    deleteDiarizationRecord: apiMocks.deleteDiarizationRecord,
    diarizationRecordAudioUrl: apiMocks.diarizationRecordAudioUrl,
    listTranscriptionRecords: apiMocks.listTranscriptionRecords,
    listSpeechTextJobPage: apiMocks.listSpeechTextJobPage,
    getTranscriptionRecord: apiMocks.getTranscriptionRecord,
    deleteTranscriptionRecord: apiMocks.deleteTranscriptionRecord,
    transcriptionRecordAudioUrl: apiMocks.transcriptionRecordAudioUrl,
    listStudioProjects: apiMocks.listStudioProjects,
    listStudioProjectPage: apiMocks.listStudioProjectPage,
    listSavedVoices: apiMocks.listSavedVoices,
  },
}));

describe("Page header history buttons", () => {
  beforeEach(() => {
    apiMocks.listSpeechHistoryRecords.mockReset();
    apiMocks.listTextToSpeechRecordPage.mockReset();
    apiMocks.getSpeechHistoryRecord.mockReset();
    apiMocks.deleteSpeechHistoryRecord.mockReset();
    apiMocks.speechHistoryRecordAudioUrl.mockReset();
    apiMocks.listDiarizationRecords.mockReset();
    apiMocks.listDiarizationRecordPage.mockReset();
    apiMocks.getDiarizationRecord.mockReset();
    apiMocks.deleteDiarizationRecord.mockReset();
    apiMocks.diarizationRecordAudioUrl.mockReset();
    apiMocks.listTranscriptionRecords.mockReset();
    apiMocks.listSpeechTextJobPage.mockReset();
    apiMocks.getTranscriptionRecord.mockReset();
    apiMocks.deleteTranscriptionRecord.mockReset();
    apiMocks.transcriptionRecordAudioUrl.mockReset();
    apiMocks.listStudioProjects.mockReset();
    apiMocks.listStudioProjectPage.mockReset();
    apiMocks.listSavedVoices.mockReset();

    apiMocks.listSpeechHistoryRecords.mockResolvedValue([]);
    apiMocks.listTextToSpeechRecordPage.mockImplementation(async () => ({
      items: await apiMocks.listSpeechHistoryRecords(),
      pagination: {
        next_cursor: null,
        has_more: false,
        limit: 25,
      },
    }));
    apiMocks.listDiarizationRecords.mockResolvedValue([]);
    apiMocks.listDiarizationRecordPage.mockImplementation(async () => ({
      items: await apiMocks.listDiarizationRecords(),
      pagination: {
        next_cursor: null,
        has_more: false,
        limit: 25,
      },
    }));
    apiMocks.listTranscriptionRecords.mockResolvedValue([]);
    apiMocks.listSpeechTextJobPage.mockImplementation(async () => ({
      items: await apiMocks.listTranscriptionRecords(),
      pagination: {
        next_cursor: null,
        has_more: false,
        limit: 25,
      },
    }));
    apiMocks.listStudioProjects.mockResolvedValue([]);
    apiMocks.listStudioProjectPage.mockImplementation(async () => ({
      items: await apiMocks.listStudioProjects(),
      pagination: {
        next_cursor: null,
        has_more: false,
        limit: 24,
      },
    }));
    apiMocks.listSavedVoices.mockResolvedValue([]);

    HTMLElement.prototype.scrollIntoView = vi.fn();
  });

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
    onRefresh: vi.fn().mockResolvedValue(undefined),
  };

  it("VoiceCloningPage renders the history button in the page header slot", async () => {
    render(
      <NotificationProvider>
        <MemoryRouter>
          <VoiceCloningPage {...baseProps} />
        </MemoryRouter>
      </NotificationProvider>,
    );

    const slot = screen.getByTestId("page-header-history-slot");

    await waitFor(() =>
      expect(within(slot).getByRole("button", { name: /History/i })).toBeInTheDocument(),
    );
    expect(within(slot).getByRole("button", { name: /History/i })).not.toHaveClass(
      "fixed",
    );
  });

  it("TranscriptionPage renders new transcript and model actions in the header", async () => {
    render(
      <NotificationProvider>
        <MemoryRouter>
          <TranscriptionPage {...baseProps} />
        </MemoryRouter>
      </NotificationProvider>,
    );

    expect(
      await screen.findByRole("heading", { name: "Transcription" }),
    ).toBeInTheDocument();
    expect(
      screen.getByRole("button", { name: /New transcript/i }),
    ).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /Models/i })).toBeInTheDocument();
  });

  it("StudioPage renders project actions in the page header slot", async () => {
    render(
      <NotificationProvider>
        <MemoryRouter>
          <StudioPage {...baseProps} />
        </MemoryRouter>
      </NotificationProvider>,
    );

    const slot = screen.getByTestId("page-header-history-slot");

    await waitFor(() =>
      expect(
        within(slot).getByRole("button", { name: /New project/i }),
      ).toBeInTheDocument(),
    );
    expect(
      within(slot).queryByRole("button", { name: /Project Library/i }),
    ).not.toBeInTheDocument();
  });

  it("StudioPage does not refetch projects on model-status rerenders", async () => {
    const studioModelLoading = {
      variant: "Orpheus-3B-0.1-ft-Q8_0-GGUF",
      status: "loading" as const,
      local_path: "/models/orpheus.gguf",
      size_bytes: 1024,
      download_progress: null,
      error_message: null,
      speech_capabilities: {
        supports_builtin_voices: true,
        built_in_voice_count: 1,
        supports_reference_voice: true,
        supports_voice_description: true,
        supports_streaming: false,
        supports_speed_control: true,
        supports_auto_long_form: false,
      },
    };
    const studioModelReady = {
      ...studioModelLoading,
      status: "ready" as const,
    };

    const onSelect = vi.fn();
    const props = {
      ...baseProps,
      onSelect,
      models: [studioModelLoading],
      selectedModel: studioModelLoading.variant,
    };

    const { rerender } = render(
      <NotificationProvider>
        <MemoryRouter>
          <StudioPage {...props} />
        </MemoryRouter>
      </NotificationProvider>,
    );

    await waitFor(() =>
      expect(apiMocks.listStudioProjects).toHaveBeenCalledTimes(1),
    );

    rerender(
      <NotificationProvider>
        <MemoryRouter>
          <StudioPage
            {...props}
            models={[studioModelReady]}
            selectedModel={studioModelReady.variant}
          />
        </MemoryRouter>
      </NotificationProvider>,
    );

    await new Promise((resolve) => window.setTimeout(resolve, 150));
    expect(apiMocks.listStudioProjects).toHaveBeenCalledTimes(1);
  });
});
