import { fireEvent, render, screen } from "@testing-library/react";
import { MemoryRouter } from "react-router";
import { describe, expect, it, vi } from "vitest";

import type { SpeechHistoryRecord } from "@/api";
import { TextToSpeechRecordDetail } from "./TextToSpeechRecordDetail";

const apiMocks = vi.hoisted(() => ({
  downloadAudioFile: vi.fn(),
  textToSpeechRecordAudioUrl: vi.fn(),
}));

vi.mock("@/api", async (importOriginal) => {
  const actual = await importOriginal<typeof import("@/api")>();
  return { ...actual, api: { ...actual.api, ...apiMocks } };
});

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

function renderDetail(overrides: Partial<SpeechHistoryRecord>, audioUrl: string | null = null) {
  render(
    <MemoryRouter>
      <TextToSpeechRecordDetail
        record={record(overrides)}
        audioUrl={audioUrl}
        parentCrumb={{ label: "Text to Speech", to: "/text-to-speech" }}
        onCancel={vi.fn()}
      />
    </MemoryRouter>,
  );
}

describe("TextToSpeechRecordDetail", () => {
  it("downloads the take in the format picked from the Download menu", async () => {
    apiMocks.downloadAudioFile.mockResolvedValue(true);
    apiMocks.textToSpeechRecordAudioUrl.mockImplementation(
      (id: string, options?: { format?: string }) => `/v1/text-to-speech/${id}/audio?format=${options?.format}`,
    );
    renderDetail({}, "/v1/text-to-speech/tts-1/audio");

    const trigger = screen.getByRole("button", { name: /Download/ });
    fireEvent.pointerDown(trigger);
    fireEvent.click(trigger);
    expect(
      (await screen.findAllByRole("menuitem")).map((item) => item.textContent),
    ).toEqual(["WAV", "MP3", "FLAC", "Opus"]);
    fireEvent.click(screen.getByRole("menuitem", { name: "MP3" }));

    expect(apiMocks.textToSpeechRecordAudioUrl).toHaveBeenCalledWith("tts-1", { format: "mp3" });
    expect(apiMocks.downloadAudioFile).toHaveBeenCalledWith(
      "/v1/text-to-speech/tts-1/audio?format=mp3",
      "Hello there.mp3",
    );
  });

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
