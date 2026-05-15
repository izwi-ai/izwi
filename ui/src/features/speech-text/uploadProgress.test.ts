import { describe, expect, it } from "vitest";

import {
  clampUploadPercent,
  createSpeechTextUploadState,
  formatUploadFileSize,
  resolveUploadFileKind,
} from "./uploadProgress";

describe("speech-text upload progress helpers", () => {
  it("formats byte sizes for compact upload metadata", () => {
    expect(formatUploadFileSize(0)).toBe("0 B");
    expect(formatUploadFileSize(950)).toBe("950 B");
    expect(formatUploadFileSize(1536)).toBe("1.5 KB");
    expect(formatUploadFileSize(5 * 1024 * 1024)).toBe("5.0 MB");
    expect(formatUploadFileSize(125 * 1024 * 1024)).toBe("125 MB");
  });

  it("prefers the uploaded filename extension for file type labels", () => {
    const blob = new Blob(["audio"], { type: "audio/mpeg" });

    expect(resolveUploadFileKind(blob, "Aliko Dangote.mp3")).toBe("MP3");
    expect(resolveUploadFileKind(blob, "board.review.m4a")).toBe("M4A");
  });

  it("falls back to MIME type labels when the filename has no extension", () => {
    expect(
      resolveUploadFileKind(new Blob(["audio"], { type: "audio/webm;codecs=opus" })),
    ).toBe("WEBM");
    expect(resolveUploadFileKind(new Blob(["audio"], { type: "" }))).toBe(
      "Audio",
    );
  });

  it("creates stable initial upload state from the selected file", () => {
    const file = new File(["audio"], "meeting.wav", { type: "audio/wav" });

    expect(createSpeechTextUploadState(file, file.name)).toEqual({
      phase: "preparing",
      fileName: "meeting.wav",
      fileSizeBytes: 5,
      fileKind: "WAV",
      loadedBytes: 0,
      totalBytes: 5,
      percent: 0,
    });
  });

  it("clamps progress percentages to the visible range", () => {
    expect(clampUploadPercent(-12)).toBe(0);
    expect(clampUploadPercent(45.5)).toBe(45.5);
    expect(clampUploadPercent(120)).toBe(100);
  });
});
