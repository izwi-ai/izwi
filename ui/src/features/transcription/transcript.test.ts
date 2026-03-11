import { describe, expect, it } from "vitest";

import {
  formatTranscriptionText,
  transcriptionEntriesFromRecord,
  transcriptionHasTimestamps,
} from "./transcript";

const timedRecord = {
  duration_secs: 4.5,
  transcription: "Hello there. General Kenobi.",
  segments: [
    {
      start: 0,
      end: 1.5,
      text: "Hello there.",
      word_start: 0,
      word_end: 1,
    },
    {
      start: 1.9,
      end: 4.5,
      text: "General Kenobi.",
      word_start: 2,
      word_end: 3,
    },
  ],
  words: [
    { word: "Hello", start: 0, end: 0.5 },
    { word: "there.", start: 0.6, end: 1.5 },
    { word: "General", start: 1.9, end: 2.8 },
    { word: "Kenobi.", start: 2.9, end: 4.5 },
  ],
};

describe("transcription transcript utilities", () => {
  it("prefers stored segments for workspace entries", () => {
    expect(transcriptionEntriesFromRecord(timedRecord)).toEqual([
      {
        start: 0,
        end: 1.5,
        text: "Hello there.",
        wordStart: 0,
        wordEnd: 1,
        timed: true,
      },
      {
        start: 1.9,
        end: 4.5,
        text: "General Kenobi.",
        wordStart: 2,
        wordEnd: 3,
        timed: true,
      },
    ]);
  });

  it("formats timestamped transcript exports when timing data exists", () => {
    expect(transcriptionHasTimestamps(timedRecord)).toBe(true);
    expect(formatTranscriptionText(timedRecord)).toBe(
      "[00:00.000 - 00:01.500] Hello there.\n[00:01.900 - 00:04.500] General Kenobi.",
    );
  });

  it("falls back to normalized plain text when timestamps are unavailable", () => {
    expect(
      formatTranscriptionText({
        duration_secs: 5,
        transcription: " Hello   world ",
        segments: [],
        words: [],
      }),
    ).toBe("Hello world");
  });
});
