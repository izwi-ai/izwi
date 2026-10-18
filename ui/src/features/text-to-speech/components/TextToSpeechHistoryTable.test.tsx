import { fireEvent, render, screen } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import type { SpeechHistoryRecordSummary } from "@/api";
import { TextToSpeechHistoryTable } from "@/features/text-to-speech/components/TextToSpeechHistoryTable";

vi.mock("@/app/providers/NotificationProvider", () => ({
  useNotifications: () => ({ notify: vi.fn() }),
}));

function record(
  overrides: Partial<SpeechHistoryRecordSummary> = {},
): SpeechHistoryRecordSummary {
  return {
    id: "tts-1",
    created_at: Date.UTC(2026, 8, 2),
    route_kind: "text_to_speech",
    processing_status: "ready",
    processing_error: null,
    model_id: "Kokoro-82M",
    speaker: "af_bella",
    language: "en",
    input_preview: "My name is Riley",
    input_chars: 16,
    generation_time_ms: 100,
    audio_duration_secs: 2.4,
    rtf: 0.1,
    tokens_generated: 10,
    audio_mime_type: "audio/wav",
    audio_filename: null,
    ...overrides,
  };
}

describe("TextToSpeechHistoryTable", () => {
  let play: ReturnType<typeof vi.spyOn>;
  let pause: ReturnType<typeof vi.spyOn>;

  beforeEach(() => {
    play = vi
      .spyOn(HTMLMediaElement.prototype, "play")
      .mockResolvedValue(undefined);
    pause = vi
      .spyOn(HTMLMediaElement.prototype, "pause")
      .mockImplementation(() => {});
  });

  afterEach(() => {
    play.mockRestore();
    pause.mockRestore();
  });

  it("plays a take from its row without opening the record", () => {
    const onOpenRecord = vi.fn();
    render(
      <TextToSpeechHistoryTable
        records={[
          record(),
          record({
            id: "tts-2",
            processing_status: "processing",
            input_preview: "Still rendering",
          }),
        ]}
        total={2}
        onOpenRecord={onOpenRecord}
      />,
    );

    const [playReady, playRunning] = screen.getAllByRole("button", {
      name: /^Play /,
    });
    expect(playRunning).toBeDisabled();
    expect(screen.getByText("Generating")).toBeVisible();
    expect(screen.getAllByText("0:02")).toHaveLength(2);

    fireEvent.click(playReady);
    expect(play).toHaveBeenCalledTimes(1);
    expect(onOpenRecord).not.toHaveBeenCalled();
    fireEvent.click(screen.getByRole("button", { name: /^Pause / }));
    expect(pause).toHaveBeenCalled();
    expect(screen.getAllByRole("button", { name: /^Play / })).toHaveLength(2);

    fireEvent.click(screen.getByText("My name is Riley", { selector: "div" }));
    expect(onOpenRecord).toHaveBeenCalledWith("tts-1");
  });
});
