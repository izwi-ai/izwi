import { describe, expect, it } from "vitest";

import { speechDownloadFilename } from "@/features/text-to-speech/support";

describe("speechDownloadFilename", () => {
  it("names a take after what it says", () => {
    expect(
      speechDownloadFilename({
        id: "5c3b4f99-aaaa",
        input_text: "  The harbour woke: gulls/terns circled the boats.\n",
        audio_mime_type: "audio/wav",
      }),
    ).toBe("The harbour woke gulls terns circled the boats.wav");
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
