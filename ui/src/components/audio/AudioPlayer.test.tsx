import { fireEvent, render, screen } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { RecordAudioPlayer } from "@/components/audio/RecordAudioPlayer";

describe("RecordAudioPlayer", () => {
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

  it("plays, seeks within the known length and changes speed", async () => {
    const { container } = render(
      <RecordAudioPlayer src="/audio/take.wav" durationHint={27} />,
    );
    const audio = container.querySelector("audio") as HTMLAudioElement;

    expect(screen.getByText("/ 0:27")).toBeVisible();
    fireEvent.click(screen.getByRole("button", { name: "Play" }));
    expect(play).toHaveBeenCalled();
    expect(await screen.findByRole("button", { name: "Pause" })).toBeVisible();

    fireEvent.change(screen.getByRole("slider", { name: "Seek audio timeline" }), {
      target: { value: "12" },
    });
    expect(audio.currentTime).toBe(12);
    fireEvent.click(screen.getByRole("button", { name: "Forward 10 seconds" }));
    expect(audio.currentTime).toBe(22);
    fireEvent.click(screen.getByRole("button", { name: "Forward 10 seconds" }));
    expect(audio.currentTime).toBe(27);

    fireEvent.change(screen.getByRole("combobox", { name: "Playback speed" }), {
      target: { value: "1.5" },
    });
    expect(audio.playbackRate).toBe(1.5);
  });

  it("rounds the total length like the rest of the app", () => {
    render(<RecordAudioPlayer src="/audio/take.wav" durationHint={2.85} />);
    expect(screen.getByText("/ 0:03")).toBeVisible();
  });

  it("disables the controls without audio", () => {
    render(<RecordAudioPlayer src="" />);
    expect(screen.getByRole("button", { name: "Play" })).toBeDisabled();
  });
});
