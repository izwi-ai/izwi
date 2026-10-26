import { describe, expect, it } from "vitest";

import {
  formatTranscriptionText,
  speakerTurnBlocksFromRecord,
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

  it("preserves speaker-attributed ASR turns as separate transcript lines", () => {
    const record = {
      duration_secs: 8,
      transcription: "[Speaker 1]: hello [Speaker 2]: hi there",
      speaker_attributed_text: "[Speaker 1]: hello [Speaker 2]: hi there",
      speaker_turns: [
        { speaker: "Speaker 1", text: "hello" },
        { speaker: "Speaker 2", text: "hi there" },
      ],
      segments: [],
      words: [],
    };

    expect(formatTranscriptionText(record)).toBe(
      "[Speaker 1]: hello\n[Speaker 2]: hi there",
    );
    expect(transcriptionEntriesFromRecord(record).map((entry) => entry.text)).toEqual([
      "[Speaker 1]: hello",
      "[Speaker 2]: hi there",
    ]);
  });

  it("turns speaker turns into one block per speaker change", () => {
    const blocks = speakerTurnBlocksFromRecord({
      speaker_turns: [
        { speaker: "Speaker 1", text: "  Hi everybody,  " },
        { speaker: "Speaker 1", text: "how did it all start?" },
        { speaker: "[Speaker 2]", text: "Cement." },
        { speaker: "Speaker 3", text: "   " },
        { speaker: "UNKNOWN", text: "inaudible" },
        { speaker: "Speaker 1", text: "And then?" },
      ],
    });

    expect(blocks).toEqual([
      {
        speaker: "Speaker 1",
        speakerIndex: 0,
        text: "Hi everybody, how did it all start?",
        start: 0,
        end: 0,
        timed: false,
      },
      {
        speaker: "Speaker 2",
        speakerIndex: 1,
        text: "Cement.",
        start: 0,
        end: 0,
        timed: false,
      },
      {
        speaker: "Unknown speaker",
        speakerIndex: 2,
        text: "inaudible",
        start: 0,
        end: 0,
        timed: false,
      },
      {
        speaker: "Speaker 1",
        speakerIndex: 0,
        text: "And then?",
        start: 0,
        end: 0,
        timed: false,
      },
    ]);
  });

  it("orders timed speaker turns by the audio and keeps their span", () => {
    const blocks = speakerTurnBlocksFromRecord({
      speaker_turns: [
        { speaker: "Speaker 2", text: "Answer.", start: 4, end: 6 },
        { speaker: "Speaker 1", text: "Question,", start: 0, end: 1.5 },
        { speaker: "Speaker 1", text: "continued.", start: 1.6, end: 3.5 },
      ],
    });

    expect(blocks.map(({ speaker, text, start, end, timed }) => ({
      speaker,
      text,
      start,
      end,
      timed,
    }))).toEqual([
      { speaker: "Speaker 1", text: "Question, continued.", start: 0, end: 3.5, timed: true },
      { speaker: "Speaker 2", text: "Answer.", start: 4, end: 6, timed: true },
    ]);
  });
});
