import { describe, expect, it } from "vitest";

import {
  parseSpeechFilter,
  speechListFilters,
} from "@/features/text-to-speech/historyFilters";
import {
  parseTranscriptionFilter,
  transcriptionListFilters,
} from "@/features/transcription/historyFilters";

describe("history filter mappings", () => {
  it("maps speech filters to the records API", () => {
    const now = new Date(2026, 9, 18, 15, 30);
    expect(speechListFilters(" fox ", "all", now)).toEqual({
      q: "fox",
      voice_kind: undefined,
      created_after: undefined,
    });
    expect(speechListFilters("", "saved", now).voice_kind).toBe("saved");
    expect(speechListFilters("", "today", now).created_after).toBe(
      new Date(2026, 9, 18).getTime(),
    );
    expect(parseSpeechFilter("bogus")).toBe("all");
  });

  it("maps transcription filters to the jobs API", () => {
    expect(transcriptionListFilters("", "active")).toEqual({
      q: undefined,
      status: "pending,processing",
      with_speakers: undefined,
    });
    expect(transcriptionListFilters("", "speakers").with_speakers).toBe(true);
    expect(transcriptionListFilters("", "failed").status).toBe("failed");
    expect(parseTranscriptionFilter(null)).toBe("all");
  });
});
