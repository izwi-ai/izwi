import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { SpeechTextUploadProgress } from "@/features/speech-text/components/SpeechTextUploadProgress";

const baseProps = {
  fileName: "interview.wav",
  fileSizeBytes: 2048,
  fileKind: "Audio",
  loadedBytes: 1024,
  totalBytes: 2048,
  percent: 50,
} as const;

describe("SpeechTextUploadProgress accessibility", () => {
  it("announces phase changes politely without putting changing percentages in the live region", () => {
    render(<SpeechTextUploadProgress {...baseProps} phase="uploading" />);

    const status = screen.getByRole("status");
    expect(status).toHaveAttribute("aria-live", "polite");
    expect(status).toHaveTextContent("Uploading audio");
    expect(status).not.toHaveTextContent("50%");
    expect(screen.getByRole("progressbar")).toHaveAttribute(
      "aria-valuenow",
      "50",
    );
  });

  it("announces upload failures assertively", () => {
    render(
      <SpeechTextUploadProgress
        {...baseProps}
        phase="failed"
        errorMessage="The upload was interrupted."
      />,
    );

    const alert = screen.getByRole("alert");
    expect(alert).toHaveAttribute("aria-live", "assertive");
    expect(alert).toHaveTextContent("The upload was interrupted.");
  });
});
