import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import type { ModelInfo } from "@/api";
import { NewDiarizationModal } from "@/features/diarization/components/NewDiarizationModal";
import { NewTextToSpeechModal } from "@/features/text-to-speech/components/NewTextToSpeechModal";
import { NewTranscriptionModal } from "@/features/transcription/components/NewTranscriptionModal";

const readyTtsModel = {
  variant: "Kokoro-82M",
  status: "ready",
  speech_capabilities: {
    supports_builtin_voices: true,
    supports_reference_voice: false,
    supports_voice_description: false,
    supports_streaming: true,
    supports_speed_control: false,
    supports_auto_long_form: false,
  },
} as ModelInfo;

describe("viewport-safe creation dialogs", () => {
  it("keeps the full text-to-speech flow inside the shared scroll contract", () => {
    render(
      <NewTextToSpeechModal
        isOpen
        onClose={vi.fn()}
        selectedModel={readyTtsModel.variant}
        selectedModelInfo={readyTtsModel}
        selectedModelReady
        onOpenModelManager={vi.fn()}
        onModelRequired={vi.fn()}
        onCreated={vi.fn()}
        onCreateStream={vi.fn()}
      />,
    );

    const dialog = screen.getByRole("dialog", {
      name: "New text-to-speech job",
    });
    expect(dialog).toHaveClass("overflow-y-auto");
    expect(dialog).not.toHaveClass("overflow-hidden");
    expect(screen.getByRole("button", { name: "Close" })).toBeInTheDocument();
    expect(
      screen.getByRole("button", { name: "Create generation" }),
    ).toBeInTheDocument();
    expect(screen.getByText("Stream results")).toBeInTheDocument();
  });

  it("keeps the transcription title, upload action, and final setting reachable", () => {
    render(
      <NewTranscriptionModal
        isOpen
        onClose={vi.fn()}
        selectedModel="Parakeet-TDT-0.6B-v3"
        selectedModelReady
        timestampAlignerModelId={null}
        timestampAlignerReady={false}
        onOpenModelManager={vi.fn()}
        onModelRequired={vi.fn()}
        onTimestampAlignerRequired={vi.fn()}
        onCreated={vi.fn()}
      />,
    );

    const dialog = screen.getByRole("dialog", { name: "New transcript" });
    expect(dialog).toHaveClass("overflow-y-auto");
    expect(dialog).not.toHaveClass("overflow-hidden");
    expect(screen.getByRole("button", { name: "Close" })).toBeInTheDocument();
    expect(
      screen.getByRole("button", { name: "Upload audio file" }),
    ).toBeInTheDocument();
    expect(screen.getByText("Generate summary")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Cancel" })).toBeInTheDocument();
  });

  it("gates the conflicting transcription options behind an explanation", () => {
    render(
      <NewTranscriptionModal
        isOpen
        onClose={vi.fn()}
        selectedModel="Parakeet-TDT-0.6B-v3"
        selectedModelReady
        timestampAlignerModelId="Qwen3-ForcedAligner-0.6B"
        timestampAlignerReady
        onOpenModelManager={vi.fn()}
        onModelRequired={vi.fn()}
        onTimestampAlignerRequired={vi.fn()}
        onCreated={vi.fn()}
      />,
    );

    const streamToggle = screen.getByLabelText(
      /Stream results/i,
    ) as HTMLInputElement;
    const timestampToggle = screen.getByLabelText(
      /Include timestamps/i,
    ) as HTMLInputElement;

    // Streaming defaults on, so timestamps are disabled with an explanation
    // instead of being allowed and silently flipping streaming off.
    expect(streamToggle.checked).toBe(true);
    expect(timestampToggle.disabled).toBe(true);
    expect(
      screen.getByText(
        /Timestamps are unavailable while streaming is enabled/i,
      ),
    ).toBeInTheDocument();

    fireEvent.click(streamToggle);
    expect(streamToggle.checked).toBe(false);
    expect(timestampToggle.disabled).toBe(false);

    fireEvent.click(timestampToggle);
    expect(timestampToggle.checked).toBe(true);
    expect(streamToggle.disabled).toBe(true);
    expect(
      screen.getByText(/Streaming is unavailable while timestamps are enabled/i),
    ).toBeInTheDocument();
    expect(streamToggle.checked).toBe(false);
  });

  it("keeps the diarization title, capture actions, and final setting reachable", () => {
    render(
      <NewDiarizationModal
        isOpen
        onClose={vi.fn()}
        selectedModel="diar_streaming_sortformer_4spk-v2.1"
        selectedModelReady
        diarizationModels={[
          {
            variant: "diar_streaming_sortformer_4spk-v2.1",
            status: "ready",
            local_path: "/models/diar",
            size_bytes: null,
            download_progress: null,
            error_message: null,
          },
        ]}
        onSelectModel={vi.fn()}
        onModelRequired={vi.fn()}
        onPipelineModelsRequired={vi.fn()}
        onOpenModelManager={vi.fn()}
        onUnloadAllManagedModels={vi.fn()}
        onCreated={vi.fn()}
      />,
    );

    const dialog = screen.getByRole("dialog", { name: "New diarization" });
    expect(dialog).toHaveClass("overflow-y-auto");
    expect(dialog).not.toHaveClass("overflow-hidden");
    expect(screen.getByRole("button", { name: "Close" })).toBeInTheDocument();
    expect(
      screen.getByRole("button", { name: "Upload audio file" }),
    ).toBeInTheDocument();
    expect(screen.getByText("Min silence (ms)")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Cancel" })).toBeInTheDocument();
  });
});
