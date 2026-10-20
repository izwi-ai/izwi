import { fireEvent, render, screen } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { describe, expect, it, vi } from "vitest";

import type { TranscriptionRecord } from "@/api";
import { TranscriptionRecordDetail } from "./TranscriptionRecordDetail";

const record = (
  overrides: Partial<TranscriptionRecord>,
): TranscriptionRecord => ({
  id: "txr-1",
  created_at: 1,
  model_id: "Parakeet-TDT-0.6B-v3",
  aligner_model_id: null,
  language: null,
  processing_status: "ready",
  processing_error: null,
  duration_secs: 4,
  processing_time_ms: 0,
  rtf: null,
  audio_mime_type: "audio/wav",
  audio_filename: "meeting.wav",
  transcription: "",
  segments: [],
  words: [],
  summary_status: "not_requested",
  summary_model_id: null,
  summary_text: null,
  summary_error: null,
  summary_updated_at: null,
  ...overrides,
});

function renderDetail(
  overrides: Partial<TranscriptionRecord>,
  props: Partial<Parameters<typeof TranscriptionRecordDetail>[0]> = {},
) {
  render(
    <MemoryRouter>
      <TranscriptionRecordDetail
        record={record(overrides)}
        audioUrl={null}
        parentCrumb={{ label: "Transcription", to: "/transcription" }}
        {...props}
      />
    </MemoryRouter>,
  );
}

describe("TranscriptionRecordDetail", () => {
  it("offers Cancel while the job runs", () => {
    const onCancel = vi.fn();
    const onRetry = vi.fn();
    renderDetail({ processing_status: "processing" }, { onCancel, onRetry });

    fireEvent.click(screen.getByRole("button", { name: "Cancel" }));
    expect(onCancel).toHaveBeenCalledOnce();
    expect(screen.queryByRole("button", { name: /Retry/ })).not.toBeInTheDocument();
  });

  it("offers Retry next to the reason a job failed", () => {
    const onCancel = vi.fn();
    const onRetry = vi.fn();
    renderDetail(
      { processing_status: "failed", processing_error: "Cancelled by user." },
      { onCancel, onRetry },
    );

    expect(screen.getByText("Cancelled by user.")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: /Retry/ }));
    expect(onRetry).toHaveBeenCalledOnce();
    expect(screen.queryByRole("button", { name: "Cancel" })).not.toBeInTheDocument();
  });
});
