import { render, screen } from "@testing-library/react";
import { MemoryRouter, Route, Routes, useLocation } from "react-router-dom";
import { describe, expect, it } from "vitest";

import { LegacyDiarizationRedirect } from "@/app/router/AppRoutes";

function LocationProbe() {
  const location = useLocation();
  return <div data-testid="location">{`${location.pathname}${location.search}`}</div>;
}

function renderRedirect(initialEntry: string) {
  return render(
    <MemoryRouter initialEntries={[initialEntry]}>
      <Routes>
        <Route path="/diarization" element={<LegacyDiarizationRedirect />} />
        <Route path="/diarization/:recordId" element={<LegacyDiarizationRedirect />} />
        <Route path="/transcription" element={<LocationProbe />} />
        <Route path="/transcription/:recordId" element={<LocationProbe />} />
      </Routes>
    </MemoryRouter>,
  );
}

describe("LegacyDiarizationRedirect", () => {
  it("redirects /diarization to /transcription?mode=diarization", async () => {
    renderRedirect("/diarization");
    expect(await screen.findByTestId("location")).toHaveTextContent(
      "/transcription?mode=diarization",
    );
  });

  it("redirects /diarization/:recordId to /transcription/:recordId?mode=diarization", async () => {
    renderRedirect("/diarization/diar-42");
    expect(await screen.findByTestId("location")).toHaveTextContent(
      "/transcription/diar-42?mode=diarization",
    );
  });
});
