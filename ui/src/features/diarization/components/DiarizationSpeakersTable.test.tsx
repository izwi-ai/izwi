import { act, fireEvent, render, screen, within } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { DiarizationSpeakersTable } from "./DiarizationSpeakersTable";

const record = {
  id: "diar-1",
  segments: [
    { speaker: "SPEAKER_00", start: 0, end: 4 },
    { speaker: "SPEAKER_01", start: 4, end: 5 },
    { speaker: "SPEAKER_02", start: 5, end: 5.5 },
  ],
  utterances: [
    { speaker: "SPEAKER_00", start: 0, end: 3, text: "Hello there", word_start: 0, word_end: 1 },
    { speaker: "SPEAKER_01", start: 3, end: 4, text: "Hi", word_start: 2, word_end: 2 },
  ],
  words: [
    { word: "Hello", speaker: "SPEAKER_00", start: 0, end: 1 },
    { word: "there", speaker: "SPEAKER_00", start: 1, end: 3 },
    { word: "Hi", speaker: "SPEAKER_01", start: 3, end: 4 },
  ],
  speaker_name_overrides: { SPEAKER_00: "Erin" },
} as unknown as Parameters<typeof DiarizationSpeakersTable>[0]["record"];

describe("DiarizationSpeakersTable", () => {
  beforeEach(() => {
    vi.useFakeTimers();
  });
  afterEach(() => {
    vi.useRealTimers();
  });

  it("lists every detected speaker with talk time, turns and words", () => {
    render(
      <DiarizationSpeakersTable record={record} audioUrl={null} onSave={vi.fn()} />,
    );

    expect(screen.getByLabelText("Name for SPEAKER_00")).toHaveValue("Erin");
    expect(screen.getByText("Detected as SPEAKER_00")).toBeInTheDocument();
    expect(screen.getByText("3.0 s · 75%")).toBeInTheDocument();
    expect(screen.getByText("Click the name to rename")).toBeInTheDocument();
    // Diarized, but the ASR gave it no words.
    expect(screen.getByText("Detected, but no words")).toBeInTheDocument();
  });

  it("saves names as you type, once typing pauses", async () => {
    const onSave = vi.fn().mockResolvedValue(undefined);
    render(
      <DiarizationSpeakersTable record={record} audioUrl={null} onSave={onSave} />,
    );

    const input = screen.getByLabelText("Name for SPEAKER_01");
    fireEvent.change(input, { target: { value: "Aar" } });
    fireEvent.change(input, { target: { value: "Aaron " } });
    expect(onSave).not.toHaveBeenCalled();

    await act(async () => {
      await vi.advanceTimersByTimeAsync(700);
    });
    expect(onSave).toHaveBeenCalledTimes(1);
    expect(onSave).toHaveBeenCalledWith({ SPEAKER_00: "Erin", SPEAKER_01: "Aaron" });
  });

  it("merges a speaker into another and resets a name from the row menu", async () => {
    vi.useRealTimers();
    const onSave = vi.fn().mockResolvedValue(undefined);
    render(
      <DiarizationSpeakersTable record={record} audioUrl={null} onSave={onSave} />,
    );

    fireEvent.pointerDown(
      screen.getByRole("button", { name: "More actions for SPEAKER_01" }),
      { button: 0, ctrlKey: false },
    );
    fireEvent.click(await screen.findByRole("menuitem", { name: /Merge into Erin/ }));
    expect(screen.getByLabelText("Name for SPEAKER_01")).toHaveValue("Erin");

    fireEvent.pointerDown(
      screen.getByRole("button", { name: "More actions for Erin (SPEAKER_00)" }),
      { button: 0, ctrlKey: false },
    );
    const menu = await screen.findByRole("menu");
    fireEvent.click(within(menu).getByRole("menuitem", { name: /Reset name/ }));
    expect(screen.getByLabelText("Name for SPEAKER_00")).toHaveValue("");
  });
});
