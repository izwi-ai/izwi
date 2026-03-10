import { render, screen, waitFor, within } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { TextToSpeechPage } from "./text-to-speech/route";
import { VoiceCloningPage } from "./voice-cloning/route";
import { VoiceDesignPage } from "./voice-design/route";
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

    apiMocks.listSpeechHistoryRecords.mockResolvedValue([]);
    apiMocks.listDiarizationRecords.mockResolvedValue([]);
    apiMocks.listTranscriptionRecords.mockResolvedValue([]);

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
  };

  it.each([
    ["TextToSpeechPage", <TextToSpeechPage {...baseProps} />],
    ["VoiceCloningPage", <VoiceCloningPage {...baseProps} />],
    ["VoiceDesignPage", <VoiceDesignPage {...baseProps} />],
    ["DiarizationPage", <DiarizationPage {...baseProps} />],
    ["TranscriptionPage", <TranscriptionPage {...baseProps} />],
  ])("%s renders the history button in the page header slot", async (_, ui) => {
    render(ui);

    const slot = screen.getByTestId("page-header-history-slot");

    await waitFor(() =>
      expect(within(slot).getByRole("button", { name: /History/i })).toBeInTheDocument(),
    );
    expect(within(slot).getByRole("button", { name: /History/i })).not.toHaveClass(
      "fixed",
    );
  });
});
