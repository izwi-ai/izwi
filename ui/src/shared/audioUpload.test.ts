import { describe, expect, it } from "vitest";

import {
  prepareSpeechTextUploadBlob,
  resolveSpeechTextUploadFilename,
  shouldPreserveSpeechTextUploadBlob,
} from "./audioUpload";

describe("speech-text audio upload preparation", () => {
  it("preserves uploaded compressed files instead of expanding them to wav", async () => {
    const sourceFile = new File(["compressed audio"], "meeting.mp3", {
      type: "audio/mpeg",
    });

    await expect(prepareSpeechTextUploadBlob(sourceFile)).resolves.toBe(
      sourceFile,
    );
  });

  it("preserves supported in-memory recording blobs", async () => {
    const sourceBlob = new Blob(["recording"], { type: "audio/webm" });

    expect(shouldPreserveSpeechTextUploadBlob(sourceBlob)).toBe(true);
    await expect(prepareSpeechTextUploadBlob(sourceBlob)).resolves.toBe(
      sourceBlob,
    );
  });

  it("keeps the original uploaded filename when the source blob is uploaded", () => {
    const sourceFile = new File(["compressed"], "board-review.m4a", {
      type: "audio/mp4",
    });

    expect(
      resolveSpeechTextUploadFilename({
        sourceFileName: sourceFile.name,
        sourceBlob: sourceFile,
        uploadedBlob: sourceFile,
      }),
    ).toBe("board-review.m4a");
  });

  it("renames fallback wav output with the source base filename", () => {
    const sourceFile = new File(["source"], "customer-call.mp3", {
      type: "audio/mpeg",
    });
    const uploadedBlob = new Blob(["wav"], { type: "audio/wav" });

    expect(
      resolveSpeechTextUploadFilename({
        sourceFileName: sourceFile.name,
        sourceBlob: sourceFile,
        uploadedBlob,
      }),
    ).toBe("customer-call.wav");
  });
});
