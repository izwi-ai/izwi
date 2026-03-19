import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import { DiarizationQualityPanel } from "./DiarizationQualityPanel";

const record = {
  alignment_coverage: 0.82,
  corrected_speaker_count: 2,
  enable_llm_refinement: true,
  llm_refined: false,
  max_speakers: 4,
  min_silence_duration_ms: 200,
  min_speakers: 1,
  min_speech_duration_ms: 240,
  speaker_count: 3,
  unattributed_words: 5,
};

describe("DiarizationQualityPanel", () => {
  it("surfaces quality warnings and sends rerun settings", async () => {
    const onRerun = vi.fn().mockResolvedValue(undefined);

    const { container } = render(
      <DiarizationQualityPanel
        record={record}
        onRerun={onRerun}
      />,
    );

    expect(container.querySelectorAll('input[type="number"]')).toHaveLength(0);

    expect(
      screen.getByText("Alignment coverage needs review"),
    ).toBeInTheDocument();
    expect(screen.getByText("Unattributed words detected")).toBeInTheDocument();
    expect(screen.getByText("LLM refinement did not apply")).toBeInTheDocument();

    fireEvent.change(screen.getByLabelText("Min speakers"), {
      target: { value: "2" },
    });
    fireEvent.change(screen.getByLabelText("Max speakers"), {
      target: { value: "5" },
    });
    fireEvent.change(screen.getByLabelText("Min speech (ms)"), {
      target: { value: "180" },
    });
    fireEvent.change(screen.getByLabelText("Min silence (ms)"), {
      target: { value: "120" },
    });
    fireEvent.click(
      screen.getByRole("switch", { name: "LLM transcript refinement" }),
    );
    fireEvent.click(screen.getByRole("button", { name: "Rerun saved audio" }));

    await waitFor(() =>
      expect(onRerun).toHaveBeenCalledWith({
        min_speakers: 2,
        max_speakers: 5,
        min_speech_duration_ms: 180,
        min_silence_duration_ms: 120,
        enable_llm_refinement: false,
      }),
    );
  });
});
