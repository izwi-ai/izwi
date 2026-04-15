import { render, screen } from "@testing-library/react";
import { MemoryRouter, Route, Routes } from "react-router-dom";
import { describe, expect, it, vi } from "vitest";

import { SpeechTextPage } from "@/features/speech-text/route";

vi.mock("@/features/transcription/route", () => ({
  TranscriptionPage: () => <div data-testid="transcription-page">transcription</div>,
}));

vi.mock("@/features/diarization/route", () => ({
  DiarizationPage: ({ routeBasePath }: { routeBasePath?: string }) => (
    <div data-testid="diarization-page">diarization:{routeBasePath ?? ""}</div>
  ),
}));

const sharedProps = {
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
  onRefresh: vi.fn(async () => undefined),
};

function renderRoute(entry: string) {
  return render(
    <MemoryRouter initialEntries={[entry]}>
      <Routes>
        <Route path="/transcription" element={<SpeechTextPage {...sharedProps} />} />
        <Route
          path="/transcription/:recordId"
          element={<SpeechTextPage {...sharedProps} />}
        />
      </Routes>
    </MemoryRouter>,
  );
}

describe("SpeechTextPage", () => {
  it("defaults to transcription mode on /transcription", async () => {
    renderRoute("/transcription");
    expect(await screen.findByTestId("transcription-page")).toBeInTheDocument();
  });

  it("keeps unified transcription mode on /transcription when mode=diarization", async () => {
    renderRoute("/transcription?mode=diarization");
    expect(await screen.findByTestId("transcription-page")).toBeInTheDocument();
  });

  it("keeps unified transcription mode on /transcription when job_kind=diarization", async () => {
    renderRoute("/transcription?job_kind=diarization");
    expect(await screen.findByTestId("transcription-page")).toBeInTheDocument();
  });

  it("uses diarization mode for detail routes when mode=diarization", async () => {
    renderRoute("/transcription/diar-1?mode=diarization");
    expect(await screen.findByTestId("diarization-page")).toHaveTextContent(
      "diarization:/transcription",
    );
  });

  it("uses diarization mode for detail routes when job_kind=diarization", async () => {
    renderRoute("/transcription/diar-1?job_kind=diarization");
    expect(await screen.findByTestId("diarization-page")).toHaveTextContent(
      "diarization:/transcription",
    );
  });
});
