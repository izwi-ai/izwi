import { render, screen, waitFor, within } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { TextToSpeechPage } from "./text-to-speech/route";
import { TextToSpeechProjectsPage } from "./text-to-speech-projects/route";
import { VoiceCloningPage } from "./voice-cloning/route";
import { DiarizationPage } from "./diarization/route";
import { TranscriptionPage } from "./transcription/route";

const apiMocks = vi.hoisted(() => ({
  listSpeechHistoryRecords: vi.fn(),
  getSpeechHistoryRecord: vi.fn(),
  deleteSpeechHistoryRecord: vi.fn(),
  speechHistoryRecordAudioUrl: vi.fn(),
  listDiarizationRecords: vi.fn(),
  getDiarizationRecord: vi.fn(),
  deleteDiarizationRecord: vi.fn(),
  diarizationRecordAudioUrl: vi.fn(),
  listTranscriptionRecords: vi.fn(),
  getTranscriptionRecord: vi.fn(),
  deleteTranscriptionRecord: vi.fn(),
  transcriptionRecordAudioUrl: vi.fn(),
  listTtsProjects: vi.fn(),
  listSavedVoices: vi.fn(),
}));

vi.mock("@/api", () => ({
  api: {
    listSpeechHistoryRecords: apiMocks.listSpeechHistoryRecords,
    getSpeechHistoryRecord: apiMocks.getSpeechHistoryRecord,
    deleteSpeechHistoryRecord: apiMocks.deleteSpeechHistoryRecord,
    speechHistoryRecordAudioUrl: apiMocks.speechHistoryRecordAudioUrl,
    listDiarizationRecords: apiMocks.listDiarizationRecords,
    getDiarizationRecord: apiMocks.getDiarizationRecord,
    deleteDiarizationRecord: apiMocks.deleteDiarizationRecord,
    diarizationRecordAudioUrl: apiMocks.diarizationRecordAudioUrl,
    listTranscriptionRecords: apiMocks.listTranscriptionRecords,
    getTranscriptionRecord: apiMocks.getTranscriptionRecord,
    deleteTranscriptionRecord: apiMocks.deleteTranscriptionRecord,
    transcriptionRecordAudioUrl: apiMocks.transcriptionRecordAudioUrl,
    listTtsProjects: apiMocks.listTtsProjects,
    listSavedVoices: apiMocks.listSavedVoices,
  },
}));
vi.mock("../api", () => ({
  api: {
    listSpeechHistoryRecords: apiMocks.listSpeechHistoryRecords,
    getSpeechHistoryRecord: apiMocks.getSpeechHistoryRecord,
    deleteSpeechHistoryRecord: apiMocks.deleteSpeechHistoryRecord,
    speechHistoryRecordAudioUrl: apiMocks.speechHistoryRecordAudioUrl,
    listDiarizationRecords: apiMocks.listDiarizationRecords,
    getDiarizationRecord: apiMocks.getDiarizationRecord,
    deleteDiarizationRecord: apiMocks.deleteDiarizationRecord,
    diarizationRecordAudioUrl: apiMocks.diarizationRecordAudioUrl,
    listTranscriptionRecords: apiMocks.listTranscriptionRecords,
    getTranscriptionRecord: apiMocks.getTranscriptionRecord,
    deleteTranscriptionRecord: apiMocks.deleteTranscriptionRecord,
    transcriptionRecordAudioUrl: apiMocks.transcriptionRecordAudioUrl,
    listTtsProjects: apiMocks.listTtsProjects,
    listSavedVoices: apiMocks.listSavedVoices,
  },
}));

describe("Page header history buttons", () => {
  beforeEach(() => {
    apiMocks.listSpeechHistoryRecords.mockReset();
    apiMocks.getSpeechHistoryRecord.mockReset();
    apiMocks.deleteSpeechHistoryRecord.mockReset();
    apiMocks.speechHistoryRecordAudioUrl.mockReset();
    apiMocks.listDiarizationRecords.mockReset();
    apiMocks.getDiarizationRecord.mockReset();
    apiMocks.deleteDiarizationRecord.mockReset();
    apiMocks.diarizationRecordAudioUrl.mockReset();
    apiMocks.listTranscriptionRecords.mockReset();
    apiMocks.getTranscriptionRecord.mockReset();
    apiMocks.deleteTranscriptionRecord.mockReset();
    apiMocks.transcriptionRecordAudioUrl.mockReset();
    apiMocks.listTtsProjects.mockReset();
    apiMocks.listSavedVoices.mockReset();

    apiMocks.listSpeechHistoryRecords.mockResolvedValue([]);
    apiMocks.listDiarizationRecords.mockResolvedValue([]);
    apiMocks.listTranscriptionRecords.mockResolvedValue([]);
    apiMocks.listTtsProjects.mockResolvedValue([]);
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

  it.each([
    ["TextToSpeechPage", <TextToSpeechPage {...baseProps} />],
    ["VoiceCloningPage", <VoiceCloningPage {...baseProps} />],
    ["DiarizationPage", <DiarizationPage {...baseProps} />],
    ["TranscriptionPage", <TranscriptionPage {...baseProps} />],
  ])("%s renders the history button in the page header slot", async (_, ui) => {
    render(<MemoryRouter>{ui}</MemoryRouter>);

    const slot = screen.getByTestId("page-header-history-slot");

    await waitFor(() =>
      expect(within(slot).getByRole("button", { name: /History/i })).toBeInTheDocument(),
    );
    expect(within(slot).getByRole("button", { name: /History/i })).not.toHaveClass(
      "fixed",
    );
  });

  it("TextToSpeechProjectsPage renders project actions in the page header slot", async () => {
    render(
      <MemoryRouter>
        <TextToSpeechProjectsPage {...baseProps} />
      </MemoryRouter>,
    );

    const slot = screen.getByTestId("page-header-history-slot");

    await waitFor(() =>
      expect(
        within(slot).getByRole("button", { name: /New project/i }),
      ).toBeInTheDocument(),
    );
    expect(
      within(slot).getByRole("button", { name: /Project Library/i }),
    ).toBeInTheDocument();
  });
});
