import { fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { SpeechHistoryPanel } from "./SpeechHistoryPanel";
import { DiarizationHistoryPanel } from "./DiarizationHistoryPanel";

function activateTab(scope: ReturnType<typeof within>, name: string): void {
  const tab = scope.getByRole("tab", { name });
  fireEvent.mouseDown(tab);
  fireEvent.click(tab);
}

const apiMocks = vi.hoisted(() => ({
  listSpeechHistoryRecords: vi.fn(),
  getSpeechHistoryRecord: vi.fn(),
  deleteSpeechHistoryRecord: vi.fn(),
  speechHistoryRecordAudioUrl: vi.fn(),
  listDiarizationRecords: vi.fn(),
  getDiarizationRecord: vi.fn(),
  updateDiarizationRecord: vi.fn(),
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
    updateDiarizationRecord: apiMocks.updateDiarizationRecord,
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
    apiMocks.updateDiarizationRecord.mockReset();
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
    expect(screen.queryByTitle("Refresh history")).not.toBeInTheDocument();

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
    expect(screen.queryByTitle("Refresh history")).not.toBeInTheDocument();

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

  it("saves speaker corrections from the diarization history modal", async () => {
    const initialRecord = {
      id: "diar-1",
      created_at: 1,
      model_id: "diar_streaming_sortformer_4spk-v2.1",
      asr_model_id: "Qwen3-ASR-0.6B",
      aligner_model_id: "Qwen3-ForcedAligner-0.6B",
      llm_model_id: null,
      min_speakers: 1,
      max_speakers: 4,
      min_speech_duration_ms: 240,
      min_silence_duration_ms: 200,
      enable_llm_refinement: false,
      speaker_count: 2,
      corrected_speaker_count: 2,
      duration_secs: 6.4,
      processing_time_ms: 120,
      rtf: 0.5,
      alignment_coverage: 1,
      unattributed_words: 0,
      llm_refined: false,
      asr_text: "Hello there. Hi back.",
      audio_mime_type: "audio/wav",
      audio_filename: "meeting.wav",
      transcript: "",
      raw_transcript: "",
      speaker_name_overrides: {},
      utterances: [
        {
          speaker: "SPEAKER_00",
          start: 0,
          end: 1,
          text: "Hello there.",
          word_start: 0,
          word_end: 1,
        },
        {
          speaker: "SPEAKER_01",
          start: 1,
          end: 2,
          text: "Hi back.",
          word_start: 2,
          word_end: 3,
        },
      ],
      words: [],
      segments: [],
    };

    apiMocks.listDiarizationRecords.mockResolvedValue([
      {
        id: "diar-1",
        created_at: 1,
        model_id: "diar_streaming_sortformer_4spk-v2.1",
        speaker_count: 2,
        corrected_speaker_count: 2,
        duration_secs: 6.4,
        processing_time_ms: 120,
        rtf: 0.5,
        audio_mime_type: "audio/wav",
        audio_filename: "meeting.wav",
        transcript_preview: "SPEAKER_00 [0.00s - 1.00s]: Hello there.",
        transcript_chars: 40,
      },
    ]);
    apiMocks.getDiarizationRecord.mockResolvedValue(initialRecord);
    apiMocks.updateDiarizationRecord.mockResolvedValue({
      ...initialRecord,
      speaker_name_overrides: {
        SPEAKER_00: "Alice",
      },
    });

    const { container } = render(
      <DiarizationHistoryPanel latestRecord={initialRecord} />,
    );
    const scope = within(container);

    await waitFor(() =>
      expect(apiMocks.listDiarizationRecords).toHaveBeenCalled(),
    );

    fireEvent.click(scope.getByRole("button", { name: /History/i }));
    const historyRecordEntries = await screen.findAllByText("meeting.wav");
    const historyRecordEntry = historyRecordEntries[historyRecordEntries.length - 1];
    expect(historyRecordEntry).toBeDefined();
    fireEvent.click(historyRecordEntry!);

    expect(await scope.findByText("Diarization Record")).toBeInTheDocument();

    activateTab(scope, "Speakers");
    expect(await scope.findByText("Speaker Corrections")).toBeInTheDocument();
    const displayNameInput = scope.getAllByLabelText("Display name")[0];
    fireEvent.change(displayNameInput, {
      target: { value: "Alice" },
    });
    const saveButton = scope.getByRole("button", {
      name: "Save corrections",
    });
    fireEvent.click(saveButton);

    await waitFor(() =>
      expect(apiMocks.updateDiarizationRecord).toHaveBeenCalledWith("diar-1", {
        speaker_name_overrides: {
          SPEAKER_00: "Alice",
        },
      }),
    );
    await apiMocks.updateDiarizationRecord.mock.results[0]?.value;

    activateTab(scope, "Transcript");
    expect(await scope.findByText("Alice")).toBeInTheDocument();
  });
});
