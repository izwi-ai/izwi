import { render, screen } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { describe, expect, it, vi } from "vitest";

import type { SpeechHistoryRecord } from "@/api";
import { TextToSpeechRecordDetail } from "./TextToSpeechRecordDetail";

const record = (overrides: Partial<SpeechHistoryRecord>): SpeechHistoryRecord => ({
  id: "tts-1",
  created_at: 1,
  route_kind: "text_to_speech",
  processing_status: "ready",
  processing_error: null,
  model_id: "Kokoro-82M",
  speaker: "af_heart",
  language: null,
  input_text: "Hello there.",
  voice_description: null,
  reference_text: null,
  generation_time_ms: 120,
  audio_duration_secs: 1.2,
  rtf: 0.1,
  tokens_generated: null,
  audio_mime_type: "audio/wav",
  audio_filename: "speech.wav",
  ...overrides,
});

function renderDetail(overrides: Partial<SpeechHistoryRecord>) {
  render(
    <MemoryRouter>
      <TextToSpeechRecordDetail
        record={record(overrides)}
        audioUrl={null}
        parentCrumb={{ label: "Text to Speech", to: "/text-to-speech" }}
        onCancel={vi.fn()}
      />
    </MemoryRouter>,
  );
}

describe("TextToSpeechRecordDetail", () => {
  it("shows a failed attempt that will run again as retrying, not failed", () => {
    renderDetail({
      processing_status: "failed",
      processing_error: "Metal out of memory",
      retry_pending: true,
    });

    expect(screen.getByRole("status")).toHaveTextContent(
      "Retrying after an error: Metal out of memory.",
    );
    expect(screen.queryByRole("alert")).not.toBeInTheDocument();
  });

  it("says when the model ran out of room before the end of the script", () => {
    renderDetail({ stop_reason: "length" });

    expect(screen.getByRole("status")).toHaveTextContent(/reached its length limit/);
  });

  it("shows a final failure as an error", () => {
    renderDetail({ processing_status: "failed", processing_error: "Model missing" });

    expect(screen.getByRole("alert")).toHaveTextContent("Model missing");
  });
});
