import { describe, expect, it } from "vitest";

import {
  speechDownloadFilename,
  speechTextLabel,
} from "@/features/text-to-speech/support";

describe("speechTextLabel", () => {
  it("drops a dialogue's speaker tags and keeps performance tags", () => {
    expect(
      speechTextLabel(
        "<|speaker:0|> Ready when you are.\n<|speaker:12|> Always. [laughs] Let's go.",
      ),
    ).toBe("Ready when you are. Always. [laughs] Let's go.");
    expect(speechTextLabel("  Plain\ttext  ")).toBe("Plain text");
  });
});

describe("speechDownloadFilename", () => {
  it("names a take after what it says", () => {
    expect(
      speechDownloadFilename({
        id: "5c3b4f99-aaaa",
        input_text: "  The harbour woke: gulls/terns circled the boats.\n",
        audio_mime_type: "audio/wav",
      }),
    ).toBe("The harbour woke gulls terns circled the boats.wav");
    expect(
      speechDownloadFilename({
        id: "5c3b4f99-aaaa",
        input_text: "<|speaker:0|> Ready?\n<|speaker:1|> Always.",
        audio_mime_type: "audio/wav",
      }),
    ).toBe("Ready Always.wav");
  });

  it("cuts long text at a word and falls back to the id", () => {
    const long = speechDownloadFilename({
      id: "x",
      input_preview: "word ".repeat(40),
      audio_mime_type: "audio/wav",
    });
    expect(long.length).toBeLessThanOrEqual(64);
    expect(long).toMatch(/^word( word)*\.wav$/);
    expect(
      speechDownloadFilename({ id: "5c3b4f99-aaaa", input_text: "", audio_mime_type: "audio/wav" }),
    ).toBe("speech-5c3b4f99.wav");
  });
});
