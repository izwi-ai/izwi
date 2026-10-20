import { describe, expect, it } from "vitest";

import { AUDIO_FILE_ACCEPT } from "@/shared/audioFileAccept";

describe("AUDIO_FILE_ACCEPT", () => {
  it("advertises every desktop-supported audio extension explicitly", () => {
    expect(AUDIO_FILE_ACCEPT.split(",")).toEqual(
      expect.arrayContaining([".wav", ".mp3", ".m4a", ".aac", ".flac", ".ogg"]),
    );
  });

  it("retains MIME hints for browser file pickers", () => {
    expect(AUDIO_FILE_ACCEPT.split(",")).toEqual(
      expect.arrayContaining([
        "audio/wav",
        "audio/x-wav",
        "audio/mpeg",
        "audio/mp4",
        "audio/aac",
        "audio/flac",
        "audio/ogg",
      ]),
    );
  });

  it("does not offer Opus, which the server cannot decode", () => {
    const accepted = AUDIO_FILE_ACCEPT.split(",");
    expect(accepted).not.toContain(".opus");
    expect(accepted).not.toContain(".webm");
  });
});
