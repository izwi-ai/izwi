import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { TranscriptionPlayground } from "./TranscriptionPlayground";

const apiMocks = vi.hoisted(() => ({
  listTranscriptionRecords: vi.fn(),
  getTranscriptionRecord: vi.fn(),
  deleteTranscriptionRecord: vi.fn(),
  transcriptionRecordAudioUrl: vi.fn(),
  createTranscriptionRecord: vi.fn(),
  createTranscriptionRecordStream: vi.fn(),
}));

vi.mock("@/api", () => ({
  api: {
    listTranscriptionRecords: apiMocks.listTranscriptionRecords,
    getTranscriptionRecord: apiMocks.getTranscriptionRecord,
    deleteTranscriptionRecord: apiMocks.deleteTranscriptionRecord,
    transcriptionRecordAudioUrl: apiMocks.transcriptionRecordAudioUrl,
    createTranscriptionRecord: apiMocks.createTranscriptionRecord,
    createTranscriptionRecordStream: apiMocks.createTranscriptionRecordStream,
  },
}));

describe("TranscriptionPlayground history", () => {
  beforeEach(() => {
    apiMocks.listTranscriptionRecords.mockReset();
    apiMocks.getTranscriptionRecord.mockReset();
    apiMocks.deleteTranscriptionRecord.mockReset();
    apiMocks.transcriptionRecordAudioUrl.mockReset();
    apiMocks.createTranscriptionRecord.mockReset();
    apiMocks.createTranscriptionRecordStream.mockReset();

    apiMocks.transcriptionRecordAudioUrl.mockReturnValue(
      "/audio/transcription.wav",
    );
    apiMocks.createTranscriptionRecordStream.mockReturnValue(
      new AbortController(),
    );

    HTMLElement.prototype.scrollIntoView = vi.fn();
  });

  it("keeps the transcription history drawer open while confirming a delete", async () => {
    apiMocks.listTranscriptionRecords.mockResolvedValue([
      {
        id: "transcription-1",
        created_at: 1,
        model_id: "Qwen3-ASR-0.6B",
        language: "English",
        duration_secs: 3.2,
        processing_time_ms: 160,
        rtf: 0.4,
        audio_mime_type: "audio/wav",
        audio_filename: "clip.wav",
        transcription_preview: "Testing saved transcription history.",
        transcription_chars: 34,
      },
    ]);
    apiMocks.getTranscriptionRecord.mockResolvedValue({
      id: "transcription-1",
      created_at: 1,
      model_id: "Qwen3-ASR-0.6B",
      aligner_model_id: null,
      language: "English",
      duration_secs: 3.2,
      processing_time_ms: 160,
      rtf: 0.4,
      audio_mime_type: "audio/wav",
      audio_filename: "clip.wav",
      transcription: "Testing saved transcription history.",
      segments: [],
      words: [],
    });
    apiMocks.deleteTranscriptionRecord.mockResolvedValue(undefined);

    render(
      <TranscriptionPlayground
        selectedModel="Qwen3-ASR-0.6B"
        selectedModelReady={true}
        modelOptions={[
          {
            value: "Qwen3-ASR-0.6B",
            label: "Qwen3 ASR 0.6B",
            statusLabel: "Ready",
            isReady: true,
          },
        ]}
        onSelectModel={vi.fn()}
        onOpenModelManager={vi.fn()}
        onModelRequired={vi.fn()}
      />,
    );

    await waitFor(() =>
      expect(apiMocks.listTranscriptionRecords).toHaveBeenCalled(),
    );

    const historyButton = screen.getByRole("button", { name: /History/i });
    expect(historyButton).not.toHaveClass("fixed");
    fireEvent.click(historyButton);

    expect(await screen.findByText("Transcriptions")).toBeInTheDocument();
    expect(screen.queryByTitle("Refresh history")).not.toBeInTheDocument();

    fireEvent.pointerDown(
      screen.getByRole("button", { name: "Delete clip.wav" }),
    );
    fireEvent.click(screen.getByRole("button", { name: "Delete clip.wav" }));

    expect(
      await screen.findByRole("button", { name: "Delete record" }),
    ).toBeInTheDocument();
    expect(screen.getByText("Transcriptions")).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: "Delete record" }));

    await waitFor(() =>
      expect(apiMocks.deleteTranscriptionRecord).toHaveBeenCalledWith(
        "transcription-1",
      ),
    );
  });

  it("renders the cleaned sidebar and the streamlined history modal", async () => {
    apiMocks.listTranscriptionRecords.mockResolvedValue([
      {
        id: "transcription-1",
        created_at: 1,
        model_id: "Qwen3-ASR-0.6B",
        language: "English",
        duration_secs: 3.2,
        processing_time_ms: 160,
        rtf: 0.4,
        audio_mime_type: "audio/wav",
        audio_filename: "clip.wav",
        transcription_preview: "Testing saved transcription history.",
        transcription_chars: 34,
      },
    ]);
    apiMocks.getTranscriptionRecord.mockResolvedValue({
      id: "transcription-1",
      created_at: 1,
      model_id: "Qwen3-ASR-0.6B",
      aligner_model_id: "Qwen3-ForcedAligner-0.6B",
      language: "English",
      duration_secs: 3.2,
      processing_time_ms: 160,
      rtf: 0.4,
      audio_mime_type: "audio/wav",
      audio_filename: "clip.wav",
      transcription: "Testing saved transcription history.",
      segments: [
        {
          start: 0,
          end: 3.2,
          text: "Testing saved transcription history.",
          word_start: 0,
          word_end: 3,
        },
      ],
      words: [
        { word: "Testing", start: 0, end: 0.8 },
        { word: "saved", start: 0.85, end: 1.5 },
        { word: "transcription", start: 1.55, end: 2.5 },
        { word: "history.", start: 2.55, end: 3.2 },
      ],
    });

    render(
      <TranscriptionPlayground
        selectedModel="Qwen3-ASR-0.6B"
        selectedModelReady={true}
        modelOptions={[
          {
            value: "Qwen3-ASR-0.6B",
            label: "Qwen3 ASR 0.6B",
            statusLabel: "Ready",
            isReady: true,
          },
        ]}
        onSelectModel={vi.fn()}
        onOpenModelManager={vi.fn()}
        onModelRequired={vi.fn()}
        timestampAlignerModelId="Qwen3-ForcedAligner-0.6B"
        timestampAlignerReady={true}
      />,
    );

    await waitFor(() =>
      expect(apiMocks.listTranscriptionRecords).toHaveBeenCalled(),
    );
    await waitFor(() =>
      expect(apiMocks.getTranscriptionRecord).toHaveBeenCalledWith(
        "transcription-1",
      ),
    );

    expect(screen.queryByText("Latest input")).not.toBeInTheDocument();
    expect(
      screen.getByRole("button", { name: /Qwen3 ASR 0.6B/i }),
    ).toHaveClass("bg-[var(--bg-surface-0)]");

    fireEvent.click(screen.getByRole("button", { name: /History/i }));
    fireEvent.click(await screen.findByText("Testing saved transcription history."));

    expect(await screen.findByText("Timed transcript")).toBeInTheDocument();
    const copyButtons = screen.getAllByRole("button", { name: /Copy/i });
    const exportButtons = screen.getAllByRole("button", { name: /Export/i });

    expect(copyButtons[copyButtons.length - 1]).toBeInTheDocument();
    expect(exportButtons[exportButtons.length - 1]).toBeInTheDocument();
    expect(screen.getByTitle("Open older record")).toBeInTheDocument();
    expect(screen.queryByText("Performance")).not.toBeInTheDocument();
  });

  it("asks for the timestamp aligner before enabling timestamps", async () => {
    apiMocks.listTranscriptionRecords.mockResolvedValue([]);
    const onTimestampAlignerRequired = vi.fn();

    render(
      <TranscriptionPlayground
        selectedModel="Qwen3-ASR-0.6B"
        selectedModelReady={true}
        modelOptions={[
          {
            value: "Qwen3-ASR-0.6B",
            label: "Qwen3 ASR 0.6B",
            statusLabel: "Ready",
            isReady: true,
          },
        ]}
        onSelectModel={vi.fn()}
        onOpenModelManager={vi.fn()}
        onModelRequired={vi.fn()}
        onTimestampAlignerRequired={onTimestampAlignerRequired}
        timestampAlignerModelId={null}
        timestampAlignerReady={false}
      />,
    );

    await waitFor(() =>
      expect(apiMocks.listTranscriptionRecords).toHaveBeenCalled(),
    );

    fireEvent.click(screen.getByLabelText(/Timestamps/i));

    expect(onTimestampAlignerRequired).toHaveBeenCalledTimes(1);
    expect(screen.getByText(/Load the timestamp aligner model/i)).toBeInTheDocument();
  });

  it("keeps streaming and timestamps mutually exclusive", async () => {
    apiMocks.listTranscriptionRecords.mockResolvedValue([]);

    render(
      <TranscriptionPlayground
        selectedModel="Qwen3-ASR-0.6B"
        selectedModelReady={true}
        modelOptions={[
          {
            value: "Qwen3-ASR-0.6B",
            label: "Qwen3 ASR 0.6B",
            statusLabel: "Ready",
            isReady: true,
          },
        ]}
        onSelectModel={vi.fn()}
        onOpenModelManager={vi.fn()}
        onModelRequired={vi.fn()}
        onTimestampAlignerRequired={vi.fn()}
        timestampAlignerModelId="Qwen3-ForcedAligner-0.6B"
        timestampAlignerReady={true}
      />,
    );

    await waitFor(() =>
      expect(apiMocks.listTranscriptionRecords).toHaveBeenCalled(),
    );

    const streamToggle = screen.getByLabelText(/Stream/i) as HTMLInputElement;
    const timestampToggle = screen.getByLabelText(
      /Timestamps/i,
    ) as HTMLInputElement;

    expect(streamToggle.checked).toBe(true);
    expect(timestampToggle.checked).toBe(false);

    fireEvent.click(timestampToggle);

    expect(timestampToggle.checked).toBe(true);
    expect(streamToggle.checked).toBe(false);

    fireEvent.click(streamToggle);

    expect(streamToggle.checked).toBe(true);
    expect(timestampToggle.checked).toBe(false);
  });
});
