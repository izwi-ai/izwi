import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { SpeechHistoryPanel } from "./SpeechHistoryPanel";
import { DiarizationHistoryPanel } from "./DiarizationHistoryPanel";

const apiMocks = vi.hoisted(() => ({
  listSpeechHistoryRecords: vi.fn(),
  getSpeechHistoryRecord: vi.fn(),
  deleteSpeechHistoryRecord: vi.fn(),
  speechHistoryRecordAudioUrl: vi.fn(),
  listDiarizationRecords: vi.fn(),
  getDiarizationRecord: vi.fn(),
  deleteDiarizationRecord: vi.fn(),
  diarizationRecordAudioUrl: vi.fn(),
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
  },
}));

describe("History panels", () => {
  beforeEach(() => {
    apiMocks.listSpeechHistoryRecords.mockReset();
    apiMocks.getSpeechHistoryRecord.mockReset();
    apiMocks.deleteSpeechHistoryRecord.mockReset();
    apiMocks.speechHistoryRecordAudioUrl.mockReset();
    apiMocks.listDiarizationRecords.mockReset();
    apiMocks.getDiarizationRecord.mockReset();
    apiMocks.deleteDiarizationRecord.mockReset();
    apiMocks.diarizationRecordAudioUrl.mockReset();

    apiMocks.speechHistoryRecordAudioUrl.mockReturnValue("/audio/speech.wav");
    apiMocks.diarizationRecordAudioUrl.mockReturnValue("/audio/diarization.wav");

    HTMLElement.prototype.scrollIntoView = vi.fn();
  });

  it("keeps the speech history drawer open while confirming a delete", async () => {
    apiMocks.listSpeechHistoryRecords.mockResolvedValue([
      {
        id: "speech-1",
        created_at: 1,
        route_kind: "text_to_speech",
        model_id: "Kokoro-82M",
        speaker: "Vivian",
        language: "en",
        input_preview: "Hello from saved speech history.",
        input_chars: 31,
        generation_time_ms: 80,
        audio_duration_secs: 1.2,
        rtf: 0.4,
        tokens_generated: 12,
        audio_mime_type: "audio/wav",
        audio_filename: "speech.wav",
      },
    ]);
    apiMocks.getSpeechHistoryRecord.mockResolvedValue({
      id: "speech-1",
      created_at: 1,
      model_id: "Kokoro-82M",
      speaker: "Vivian",
      language: "en",
      input_text: "Hello from saved speech history.",
      generation_time_ms: 80,
      audio_duration_secs: 1.2,
      rtf: 0.4,
      tokens_generated: 12,
      audio_mime_type: "audio/wav",
      audio_filename: "speech.wav",
    });
    apiMocks.deleteSpeechHistoryRecord.mockResolvedValue(undefined);

    render(
      <SpeechHistoryPanel
        route="text-to-speech"
        title="Speech History"
        emptyMessage="No speech history yet."
      />,
    );

    await waitFor(() =>
      expect(apiMocks.listSpeechHistoryRecords).toHaveBeenCalledWith(
        "text-to-speech",
      ),
    );

    const historyButton = screen.getByRole("button", { name: /History/i });
    expect(historyButton).not.toHaveClass("fixed");
    fireEvent.click(historyButton);

    expect(await screen.findByText("Speech History")).toBeInTheDocument();

    fireEvent.pointerDown(
      screen.getByRole("button", { name: "Delete speech.wav" }),
    );
    fireEvent.click(screen.getByRole("button", { name: "Delete speech.wav" }));

    expect(
      await screen.findByRole("button", { name: "Delete record" }),
    ).toBeInTheDocument();
    expect(screen.getByText("Speech History")).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: "Delete record" }));

    await waitFor(() =>
      expect(apiMocks.deleteSpeechHistoryRecord).toHaveBeenCalledWith(
        "text-to-speech",
        "speech-1",
      ),
    );
  });

  it("keeps the diarization history drawer open while confirming a delete", async () => {
    apiMocks.listDiarizationRecords.mockResolvedValue([
      {
        id: "diar-1",
        created_at: 1,
        model_id: "diar_streaming_sortformer_4spk-v2.1",
        speaker_count: 2,
        duration_secs: 6.4,
        processing_time_ms: 120,
        rtf: 0.5,
        audio_mime_type: "audio/wav",
        audio_filename: "meeting.wav",
        transcript_preview: "SPEAKER_00 [0.00s - 1.00s]: Hello there.",
        transcript_chars: 40,
      },
    ]);
    apiMocks.getDiarizationRecord.mockResolvedValue({
      id: "diar-1",
      created_at: 1,
      model_id: "diar_streaming_sortformer_4spk-v2.1",
      speaker_count: 2,
      duration_secs: 6.4,
      processing_time_ms: 120,
      rtf: 0.5,
      audio_mime_type: "audio/wav",
      audio_filename: "meeting.wav",
      transcript: "",
      raw_transcript: "",
      utterances: [
        {
          speaker: "SPEAKER_00",
          start: 0,
          end: 1,
          text: "Hello there.",
        },
      ],
    });
    apiMocks.deleteDiarizationRecord.mockResolvedValue(undefined);

    render(<DiarizationHistoryPanel />);

    await waitFor(() =>
      expect(apiMocks.listDiarizationRecords).toHaveBeenCalled(),
    );

    const historyButton = screen.getByRole("button", { name: /History/i });
    expect(historyButton).not.toHaveClass("fixed");
    fireEvent.click(historyButton);

    expect(await screen.findByText("Diarization History")).toBeInTheDocument();

    fireEvent.pointerDown(
      screen.getByRole("button", { name: "Delete meeting.wav" }),
    );
    fireEvent.click(screen.getByRole("button", { name: "Delete meeting.wav" }));

    expect(
      await screen.findByRole("button", { name: "Delete record" }),
    ).toBeInTheDocument();
    expect(screen.getByText("Diarization History")).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: "Delete record" }));

    await waitFor(() =>
      expect(apiMocks.deleteDiarizationRecord).toHaveBeenCalledWith("diar-1"),
    );
  });
});
