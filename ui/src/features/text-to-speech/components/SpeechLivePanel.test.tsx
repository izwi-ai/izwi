import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import { SpeechLivePanel } from "./SpeechLivePanel";

describe("SpeechLivePanel", () => {
  it("waits for first audio and stops generation and playback", () => {
    const onStop = vi.fn();
    render(<SpeechLivePanel state="generating" onStop={onStop} />);
    expect(screen.getByText("Waiting for first audio")).toBeInTheDocument();
    expect(screen.queryByRole("progressbar")).not.toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Stop generation and playback" }));
    expect(onStop).toHaveBeenCalledTimes(1);
  });

  it("shows how many sections are generated while playing", () => {
    render(
      <SpeechLivePanel
        state="playing"
        progress={{ completedSegments: 2, totalSegments: 5 }}
        onStop={() => {}}
      />,
    );
    expect(screen.getByText("Playing generated speech")).toBeInTheDocument();
    const bar = screen.getByRole("progressbar", { name: "Sections generated" });
    expect(bar).toHaveAttribute("aria-valuenow", "2");
    expect(bar).toHaveAttribute("aria-valuemax", "5");
    expect(screen.getByText("2 of 5 sections generated")).toBeInTheDocument();
  });

  it("offers to listen from the beginning to a take rendering elsewhere", () => {
    const onListen = vi.fn();
    render(<SpeechLivePanel state="available" onListen={onListen} onStop={() => {}} />);
    expect(
      screen.queryByRole("button", { name: "Stop generation and playback" }),
    ).not.toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Listen from beginning" }));
    expect(onListen).toHaveBeenCalledTimes(1);
  });
});
