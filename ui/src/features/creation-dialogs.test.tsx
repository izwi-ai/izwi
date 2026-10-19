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

    // CreationDialog frame: the viewport-capped dialog keeps its header and
    // footer in place while the body between them scrolls.
    const dialog = screen.getByRole("dialog", {
      name: "New text-to-speech job",
    });
    expect(dialog).toHaveClass("flex", "flex-col", "overflow-hidden");
    const body = screen.getByText("Stream results").closest(".overflow-y-auto");
    expect(body).not.toBeNull();
    expect(dialog).toContainElement(body as HTMLElement);
    const submit = screen.getByRole("button", { name: "Create generation" });
    expect(body).not.toContainElement(submit);
    expect(screen.getByRole("button", { name: "Close" })).toBeInTheDocument();
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

    // CreationDialog frame: the last setting sits in the scrolling body,
    // Cancel stays in the fixed footer.
    const dialog = screen.getByRole("dialog", { name: "New transcript" });
    expect(dialog).toHaveClass("flex", "flex-col", "overflow-hidden");
    const body = screen.getByText("Generate summary").closest(".overflow-y-auto");
    expect(body).not.toBeNull();
    expect(body).not.toContainElement(
      screen.getByRole("button", { name: "Cancel" }),
    );
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

    const streamToggle = screen.getByRole("switch", { name: /Stream results/i });
    const timestampToggle = screen.getByRole("switch", {
      name: /Include timestamps/i,
    });

    // Streaming defaults on, so timestamps are disabled with an explanation
    // instead of being allowed and silently flipping streaming off.
    expect(streamToggle).toBeChecked();
    expect(timestampToggle).toBeDisabled();
    expect(
      screen.getByText(
        /Timestamps are unavailable while streaming is enabled/i,
      ),
    ).toBeInTheDocument();

    fireEvent.click(streamToggle);
    expect(streamToggle).not.toBeChecked();
    expect(timestampToggle).toBeEnabled();

    fireEvent.click(timestampToggle);
    expect(timestampToggle).toBeChecked();
    expect(streamToggle).toBeDisabled();
    expect(
      screen.getByText(/Streaming is unavailable while timestamps are enabled/i),
    ).toBeInTheDocument();
    expect(streamToggle).not.toBeChecked();
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

    // CreationDialog frame: the last setting sits in the scrolling body,
    // Cancel stays in the fixed footer.
    const dialog = screen.getByRole("dialog", { name: "New diarization" });
    expect(dialog).toHaveClass("flex", "flex-col", "overflow-hidden");
    const body = screen.getByText("Min silence (ms)").closest(".overflow-y-auto");
    expect(body).not.toBeNull();
    expect(body).not.toContainElement(
      screen.getByRole("button", { name: "Cancel" }),
    );
    expect(screen.getByRole("button", { name: "Close" })).toBeInTheDocument();
    expect(
      screen.getByRole("button", { name: "Upload audio file" }),
    ).toBeInTheDocument();
    expect(screen.getByText("Min silence (ms)")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Cancel" })).toBeInTheDocument();
  });
});
