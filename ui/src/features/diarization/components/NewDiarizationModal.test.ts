import { describe, expect, it } from "vitest";

import {
  prepareDiarizationUploadBlob,
  resolveDiarizationUploadFilename,
} from "@/features/diarization/audioUpload";

describe("resolveDiarizationUploadFilename", () => {
  it("preserves the uploaded base filename when transcoding to wav", () => {
    const sourceBlob = new File(["source"], "customer-call.mp3", {
      type: "audio/mpeg",
    });
    const uploadedBlob = new Blob(["wav"], { type: "audio/wav" });

    expect(
      resolveDiarizationUploadFilename({
        sourceFileName: sourceBlob.name,
        sourceBlob,
        uploadedBlob,
      }),
    ).toBe("customer-call.wav");
  });

  it("keeps the original uploaded filename when no transcode is needed", () => {
    const sourceBlob = new File(["source"], "board-review.wav", {
      type: "audio/wav",
    });

    expect(
      resolveDiarizationUploadFilename({
        sourceFileName: sourceBlob.name,
        sourceBlob,
        uploadedBlob: sourceBlob,
      }),
    ).toBe("board-review.wav");
  });
});

describe("prepareDiarizationUploadBlob", () => {
  it("preserves uploaded files instead of expanding compressed audio to wav", async () => {
    const sourceFile = new File(["compressed audio"], "aliko-dangote.mp3", {
      type: "audio/mpeg",
    });

    await expect(prepareDiarizationUploadBlob(sourceFile)).resolves.toBe(
      sourceFile,
    );
  });
});
