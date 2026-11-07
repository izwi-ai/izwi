import { describe, expect, it } from "vitest";

import {
  furthestSpeechProcessingStatus,
  speechDownloadFilename,
  speechTextLabel,
} from "@/features/text-to-speech/support";

describe("furthestSpeechProcessingStatus", () => {
  it("never holds a polled record behind a local copy", () => {
    // A job created without a stream: its placeholder stays pending while
    // the server renders.
    expect(
      furthestSpeechProcessingStatus(
        { processing_status: "processing" },
        { processing_status: "pending" },
      ),
    ).toBe("processing");
    expect(
      furthestSpeechProcessingStatus(
        { processing_status: "failed", processing_error: "boom" },
        { processing_status: "pending" },
      ),
    ).toBe("failed");
  });

  it("lets a stream's events run ahead of the polled record", () => {
    expect(
      furthestSpeechProcessingStatus(
        { processing_status: "pending" },
        { processing_status: "processing" },
      ),
    ).toBe("processing");
    expect(
      furthestSpeechProcessingStatus(
        { processing_status: "processing" },
        { processing_status: "ready" },
      ),
    ).toBe("ready");
    // A tie between ends keeps the record's.
    expect(
      furthestSpeechProcessingStatus(
        { processing_status: "failed", processing_error: "boom" },
        { processing_status: "ready" },
      ),
    ).toBe("failed");
  });
});

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

  it("names a download in another format after that format", () => {
    expect(
      speechDownloadFilename(
        { id: "5c3b4f99-aaaa", input_text: "Hello there.", audio_mime_type: "audio/wav" },
        "opus",
      ),
    ).toBe("Hello there.opus");
  });
});
