import { fireEvent, render, screen } from "@testing-library/react";
import { Mic2 } from "lucide-react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { VoiceLibraryGrid } from "./VoiceLibraryGrid";

describe("VoiceLibraryGrid", () => {
  let play: ReturnType<typeof vi.spyOn>;
  let pause: ReturnType<typeof vi.spyOn>;

  beforeEach(() => {
    play = vi
      .spyOn(HTMLMediaElement.prototype, "play")
      .mockImplementation(async function (this: HTMLMediaElement) {
        this.dispatchEvent(new Event("play"));
      });
    pause = vi
      .spyOn(HTMLMediaElement.prototype, "pause")
      .mockImplementation(function (this: HTMLMediaElement) {
        this.dispatchEvent(new Event("pause"));
      });
  });

  afterEach(() => {
    play.mockRestore();
    pause.mockRestore();
  });

  it("plays one preview at a time and runs the primary action", async () => {
    const onUse = vi.fn();
    render(
      <VoiceLibraryGrid
        items={[
          {
            id: "fox",
            name: "Fox",
            meta: "Cloned · 1 Apr 2026",
            kindLabel: "Cloned",
            previewUrl: "/voices/fox.wav",
            primaryAction: { label: "Use in TTS", icon: Mic2, onClick: onUse },
          },
          {
            id: "narrator",
            name: "Narrator – warm",
            kindLabel: "Designed",
            previewUrl: "/voices/narrator.wav",
          },
        ]}
        emptyTitle="No voices"
        emptyDescription="Create one."
      />,
    );

    expect(screen.getByText("NW")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Play preview for Fox" }));
    expect(
      await screen.findByRole("button", { name: "Pause preview for Fox" }),
    ).toBeVisible();

    fireEvent.click(
      screen.getByRole("button", { name: "Play preview for Narrator – warm" }),
    );
    expect(
      await screen.findByRole("button", { name: "Play preview for Fox" }),
    ).toBeVisible();

    fireEvent.click(screen.getByRole("button", { name: "Use in TTS" }));
    expect(onUse).toHaveBeenCalledOnce();
  });

  it("asks for a preview and plays it when it arrives", async () => {
    const onRequestPreview = vi.fn();
    const item = {
      id: "alloy",
      name: "Alloy",
      kindLabel: "Built-in",
      onRequestPreview,
    };
    const { rerender } = render(
      <VoiceLibraryGrid items={[item]} emptyTitle="" emptyDescription="" />,
    );

    fireEvent.click(screen.getByRole("button", { name: "Play preview for Alloy" }));
    expect(onRequestPreview).toHaveBeenCalledOnce();

    rerender(
      <VoiceLibraryGrid
        items={[{ ...item, previewUrl: "blob:alloy" }]}
        emptyTitle=""
        emptyDescription=""
      />,
    );
    expect(
      await screen.findByRole("button", { name: "Pause preview for Alloy" }),
    ).toBeVisible();
  });
});
