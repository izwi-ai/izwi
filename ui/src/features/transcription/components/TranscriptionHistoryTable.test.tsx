import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import type {
  SpeechTextDiarizationSummary,
  SpeechTextJobSummary,
  SpeechTextSpeakerAttributedAsrSummary,
  SpeechTextTranscriptionSummary,
} from "@/api";
import { TranscriptionHistoryTable } from "@/features/transcription/components/TranscriptionHistoryTable";

const mocks = vi.hoisted(() => ({
  notify: vi.fn(),
  writeText: vi.fn(),
}));

vi.mock("@/app/providers/NotificationProvider", () => ({
  useNotifications: () => ({ notify: mocks.notify }),
}));

vi.mock("@/features/transcription/components/TranscriptionExportDialog", () => ({
  TranscriptionExportDialog: () => null,
}));

vi.mock("@/components/DiarizationExportDialog", () => ({
  DiarizationExportDialog: () => null,
}));

const baseRecord = {
  created_at: Date.UTC(2026, 8, 2),
  model_id: "asr-model",
  duration_secs: 12,
  processing_time_ms: 200,
  rtf: 0.2,
  audio_mime_type: "audio/wav",
  summary_status: "not_requested" as const,
  summary_preview: null,
};

function transcription(
  overrides: Partial<SpeechTextTranscriptionSummary>,
): SpeechTextTranscriptionSummary {
  return {
    ...baseRecord,
    id: "transcription-1",
    kind: "transcription",
    processing_status: "ready",
    processing_error: null,
    audio_filename: "interview.wav",
    language: "en",
    transcription_preview: "A useful transcript preview.",
    transcription_chars: 28,
    ...overrides,
  };
}

function speakerAttributed(
  overrides: Partial<SpeechTextSpeakerAttributedAsrSummary>,
): SpeechTextSpeakerAttributedAsrSummary {
  return {
    ...baseRecord,
    id: "speaker-1",
    kind: "speaker_attributed_asr",
    processing_status: "ready",
    processing_error: null,
    audio_filename: "panel.wav",
    language: "en",
    transcription_preview: "A panel transcript.",
    transcription_chars: 19,
    speaker_attributed_text_preview: "Speaker 1: Welcome.",
    ...overrides,
  };
}

function diarization(
  overrides: Partial<SpeechTextDiarizationSummary>,
): SpeechTextDiarizationSummary {
  return {
    ...baseRecord,
    id: "diarization-1",
    kind: "diarization",
    processing_status: "ready",
    processing_error: null,
    audio_filename: "meeting.wav",
    speaker_count: 2,
    transcript_preview: "Speaker 1: Welcome.",
    transcript_chars: 19,
    ...overrides,
  };
}

describe("TranscriptionHistoryTable job status", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    Object.defineProperty(navigator, "clipboard", {
      configurable: true,
      value: { writeText: mocks.writeText },
    });
    mocks.writeText.mockResolvedValue(undefined);
  });

  it("shows status before output and preserves a successful transcript preview", () => {
    const records: SpeechTextJobSummary[] = [
      transcription({ id: "queued", processing_status: "pending" }),
      speakerAttributed({ id: "processing", processing_status: "processing" }),
      diarization({ id: "ready" }),
    ];

    render(
      <TranscriptionHistoryTable records={records} onOpenRecord={vi.fn()} />,
    );

    expect(screen.getByText("Queued")).toBeVisible();
    expect(screen.getByText("Processing")).toBeVisible();
    expect(screen.getByText("Ready")).toBeVisible();
    expect(screen.getByText("Transcription is queued and waiting to start.")).toBeVisible();
    expect(screen.getByText("Speaker transcription is in progress.")).toBeVisible();
    expect(screen.getAllByText("Speaker 1: Welcome.")).toHaveLength(2);
  });

  it("keeps raw engine failures behind details and offers an explicit recovery path", async () => {
    const technicalError =
      "CUDA_ERROR_OUT_OF_MEMORY: allocation 0x7f9 failed in engine::execute";
    const onOpenRecord = vi.fn();
    const record = diarization({
      processing_status: "failed",
      processing_error: technicalError,
      transcript_preview: technicalError,
    });

    render(
      <TranscriptionHistoryTable
        records={[record]}
        onOpenRecord={onOpenRecord}
      />,
    );

    expect(screen.getByText("Failed")).toBeVisible();
    expect(
      screen.getByText(
        "Diarization failed. Review the error, then retry when you’re ready.",
      ),
    ).toBeVisible();
    expect(screen.getByText(technicalError)).not.toBeVisible();

    const errorDetailsDisclosure = screen.getByText("Error details");
    errorDetailsDisclosure.focus();
    expect(errorDetailsDisclosure).toHaveFocus();
    fireEvent.click(errorDetailsDisclosure);
    expect(screen.getByText(technicalError)).toBeVisible();

    fireEvent.click(
      screen.getByRole("button", {
        name: "Copy error details for meeting.wav",
      }),
    );
    await waitFor(() => expect(mocks.writeText).toHaveBeenCalledWith(technicalError));

    fireEvent.click(screen.getByRole("button", { name: "Review and retry" }));
    expect(onOpenRecord).toHaveBeenCalledWith(record);
  });

  it("does not present an unknown backend state as a completed transcript", () => {
    const record = transcription({
      processing_status: "mystery" as SpeechTextTranscriptionSummary["processing_status"],
      transcription_preview: "Misleading output",
    });

    render(
      <TranscriptionHistoryTable records={[record]} onOpenRecord={vi.fn()} />,
    );

    expect(screen.getByText("Status Unknown")).toBeVisible();
    expect(
      screen.getByText("Job status is unavailable. Open the record for details."),
    ).toBeVisible();
    expect(screen.queryByText("Misleading output")).not.toBeInTheDocument();
  });
});
