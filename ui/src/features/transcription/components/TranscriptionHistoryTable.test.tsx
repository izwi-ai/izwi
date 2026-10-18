import { render, screen } from "@testing-library/react";
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
    speaker_attributed_text_chars: 19,
    speaker_turn_count: 1,
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
    expect(screen.getByText("Transcribing")).toBeVisible();
    expect(screen.getByText("2 speakers")).toBeVisible();
    // Partial output shows while running; the finished preview stays.
    expect(screen.getByText("A useful transcript preview.")).toBeVisible();
    expect(screen.getAllByText("Speaker 1: Welcome.")).toHaveLength(2);
  });

  it("shows the error message directly without inline error actions", () => {
    const technicalError =
      "CUDA_ERROR_OUT_OF_MEMORY: allocation 0x7f9 failed in engine::execute";
    const record = diarization({
      processing_status: "failed",
      processing_error: technicalError,
      transcript_preview: technicalError,
    });

    render(
      <TranscriptionHistoryTable
        records={[record]}
        onOpenRecord={vi.fn()}
      />,
    );

    expect(screen.getByText("Failed")).toBeVisible();
    expect(screen.getByText(technicalError)).toBeVisible();
    expect(screen.queryByText("Error details")).not.toBeInTheDocument();
    expect(
      screen.queryByRole("button", { name: "Review and retry" }),
    ).not.toBeInTheDocument();
    expect(
      screen.queryByRole("button", { name: /Copy error details/i }),
    ).not.toBeInTheDocument();
  });

  it("does not present an unknown backend state as a completed transcript", () => {
    const record = transcription({
      processing_status: "mystery" as SpeechTextTranscriptionSummary["processing_status"],
      transcription_preview: "Misleading output",
    });

    render(
      <TranscriptionHistoryTable records={[record]} onOpenRecord={vi.fn()} />,
    );

    expect(screen.getByText("Status unknown")).toBeVisible();
    expect(
      screen.getByText("Job status is unavailable. Open the record for details."),
    ).toBeVisible();
    expect(screen.queryByText("Misleading output")).not.toBeInTheDocument();
  });

  it("drops diarization timestamps and shows empty jobs as empty", () => {
    render(
      <TranscriptionHistoryTable
        records={[
          diarization({
            id: "diarized",
            transcript_preview: "SPEAKER_00 [0.00s - 4.49s]: So Erin, in your email",
            transcript_chars: 499,
          }),
          speakerAttributed({
            id: "empty",
            audio_filename: "silence.mp3",
            transcription_preview: "No transcript",
            transcription_chars: 0,
            speaker_attributed_text_preview: null,
            speaker_attributed_text_chars: 0,
            speaker_turn_count: 0,
          }),
        ]}
        onOpenRecord={vi.fn()}
      />,
    );

    expect(screen.getByText("SPEAKER_00: So Erin, in your email")).toBeVisible();
    expect(screen.getByText("No speech recognised.")).toBeVisible();
    expect(screen.queryByText("No transcript")).not.toBeInTheDocument();
    // Only the diarization row identified speakers.
    expect(screen.getAllByText(/speakers?$/i)).toHaveLength(1);
  });
});
