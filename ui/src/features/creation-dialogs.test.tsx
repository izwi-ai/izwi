import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import { api, type ModelInfo } from "@/api";
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

  it("offers a final-only model's speech controls with streaming off", async () => {
    vi.spyOn(api, "listSavedVoices").mockResolvedValue([]);
    const chatterbox = {
      variant: "Chatterbox-500M",
      status: "ready",
      speech_capabilities: {
        supports_builtin_voices: true,
        supports_reference_voice: true,
        reference_transcript_required: false,
        supports_voice_description: false,
        supports_streaming: false,
        supports_speed_control: false,
        supports_auto_long_form: false,
        controls: [
          { control: "exaggeration", min: 0, max: 2, default: 0.5 },
          { control: "cfg_weight", min: 0, max: 1, default: 0.5 },
          { control: "min_p", min: 0, max: 1, default: 0.05 },
        ],
      },
    } as ModelInfo;
    render(
      <NewTextToSpeechModal
        isOpen
        onClose={vi.fn()}
        selectedModel={chatterbox.variant}
        selectedModelInfo={chatterbox}
        selectedModelReady
        onOpenModelManager={vi.fn()}
        onModelRequired={vi.fn()}
        onCreated={vi.fn()}
        onCreateStream={vi.fn()}
      />,
    );

    for (const [name, value] of [
      ["Exaggeration", "0.5"],
      ["Guidance", "0.5"],
      ["Min-p", "0.05"],
    ]) {
      expect(screen.getByRole("slider", { name })).toHaveAttribute(
        "aria-valuenow",
        value,
      );
    }
    const stream = screen.getByRole("switch", { name: /Stream results/i });
    expect(stream).not.toBeChecked();
    expect(stream).toBeDisabled();

    // The built-in voice speaks unless a saved one is chosen.
    expect(screen.getByText("Built-in voice")).toBeInTheDocument();
    expect(await screen.findByText("Saved voice (optional)")).toBeInTheDocument();
    expect(screen.getByRole("combobox", { name: "Saved voice" })).toHaveTextContent(
      "None (built-in voice)",
    );
  });

  it("asks a multilingual model which language to speak", async () => {
    vi.spyOn(api, "listSavedVoices").mockResolvedValue([]);
    HTMLElement.prototype.scrollIntoView = vi.fn();
    const multilingual = {
      variant: "Chatterbox-Multilingual-500M",
      status: "ready",
      speech_capabilities: {
        supports_builtin_voices: true,
        supports_reference_voice: true,
        reference_transcript_required: false,
        supports_voice_description: false,
        supports_streaming: true,
        supports_speed_control: false,
        supports_auto_long_form: false,
        languages: [
          { code: "en", name: "English" },
          { code: "fr", name: "French" },
          { code: "sw", name: "Swahili" },
        ],
      },
    } as ModelInfo;
    const onCreateStream = vi.fn().mockResolvedValue({ id: "record" });
    render(
      <NewTextToSpeechModal
        isOpen
        onClose={vi.fn()}
        selectedModel={multilingual.variant}
        selectedModelInfo={multilingual}
        selectedModelReady
        onOpenModelManager={vi.fn()}
        onModelRequired={vi.fn()}
        onCreated={vi.fn()}
        onCreateStream={onCreateStream}
      />,
    );

    const picker = screen.getByRole("combobox", { name: "Language" });
    expect(picker).toHaveTextContent("English");
    fireEvent.keyDown(picker, { key: "Enter" });
    fireEvent.click(await screen.findByRole("option", { name: "French" }));
    expect(picker).toHaveTextContent("French");

    fireEvent.change(screen.getByRole("textbox", { name: "Script" }), {
      target: { value: "Bonjour à tous." },
    });
    fireEvent.click(screen.getByRole("button", { name: "Create generation" }));
    await waitFor(() => expect(onCreateStream).toHaveBeenCalled());
    expect(onCreateStream.mock.calls[0][0]).toMatchObject({
      model_id: "Chatterbox-Multilingual-500M",
      language: "fr",
    });
  });

  it("lets a model that reads the script detect the language", async () => {
    vi.spyOn(api, "listSavedVoices").mockResolvedValue([]);
    HTMLElement.prototype.scrollIntoView = vi.fn();
    const maya2 = {
      variant: "Maya2-Native-2.5B",
      status: "ready",
      speech_capabilities: {
        supports_builtin_voices: true,
        supports_reference_voice: true,
        reference_transcript_required: true,
        supports_voice_description: false,
        supports_streaming: true,
        supports_speed_control: false,
        supports_auto_long_form: true,
        languages: [
          { code: "en", name: "English" },
          { code: "hi", name: "Hindi" },
        ],
        language_auto_detect: true,
        inline_tags: ["[laughs]", "[sighs]"],
      },
    } as ModelInfo;
    const onCreateStream = vi.fn().mockResolvedValue({ id: "record" });
    render(
      <NewTextToSpeechModal
        isOpen
        onClose={vi.fn()}
        selectedModel={maya2.variant}
        selectedModelInfo={maya2}
        selectedModelReady
        onOpenModelManager={vi.fn()}
        onModelRequired={vi.fn()}
        onCreated={vi.fn()}
        onCreateStream={onCreateStream}
      />,
    );

    const picker = screen.getByRole("combobox", { name: "Language" });
    expect(picker).toHaveTextContent("Auto-detect");
    expect(
      screen.getByRole("combobox", { name: "Insert an expression tag" }),
    ).toHaveTextContent("Insert a tag: [laughs], [sighs]…");

    fireEvent.change(screen.getByRole("textbox", { name: "Script" }), {
      target: { value: "नमस्ते, आप कैसे हैं?" },
    });
    fireEvent.click(screen.getByRole("button", { name: "Create generation" }));
    await waitFor(() => expect(onCreateStream).toHaveBeenCalled());
    expect(onCreateStream.mock.calls[0][0].language).toBeUndefined();

    fireEvent.keyDown(picker, { key: "Enter" });
    fireEvent.click(await screen.findByRole("option", { name: "Hindi" }));
    expect(picker).toHaveTextContent("Hindi");
  });

  it("offers no language to a model that speaks one", () => {
    render(
      <NewTextToSpeechModal
        isOpen
        onClose={vi.fn()}
        selectedModel={readyTtsModel.variant}
        selectedModelInfo={{
          ...readyTtsModel,
          speech_capabilities: {
            ...readyTtsModel.speech_capabilities!,
            languages: [{ code: "en", name: "English" }],
          },
        }}
        selectedModelReady
        onOpenModelManager={vi.fn()}
        onModelRequired={vi.fn()}
        onCreated={vi.fn()}
        onCreateStream={vi.fn()}
      />,
    );
    expect(screen.queryByRole("combobox", { name: "Language" })).toBeNull();
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

  it("offers translation and keywords only to a model that takes them", () => {
    const props = {
      isOpen: true,
      onClose: vi.fn(),
      selectedModelReady: true,
      timestampAlignerModelId: "Qwen3-ForcedAligner-0.6B",
      timestampAlignerReady: true,
      onOpenModelManager: vi.fn(),
      onModelRequired: vi.fn(),
      onTimestampAlignerRequired: vi.fn(),
      onCreated: vi.fn(),
    };
    const { rerender } = render(
      <NewTranscriptionModal {...props} selectedModel="Parakeet-TDT-0.6B-v3" />,
    );
    expect(screen.queryByLabelText("Translate to")).not.toBeInTheDocument();
    expect(screen.queryByLabelText("Keywords")).not.toBeInTheDocument();

    rerender(
      <NewTranscriptionModal
        {...props}
        selectedModel="Granite-Speech-5.0-2B"
        selectedModelSupportsTranslation
        selectedModelSupportsKeywords
      />,
    );
    const translate = screen.getByRole("combobox", { name: "Translate to" });
    expect(translate).toBeEnabled();
    expect(screen.getByRole("textbox", { name: "Keywords" })).toBeEnabled();

    // Timestamps time the audio's own words, so turning them on (streaming
    // off first) disables translation with an explanation.
    fireEvent.click(screen.getByRole("switch", { name: /Stream results/i }));
    fireEvent.click(screen.getByRole("switch", { name: /Include timestamps/i }));
    expect(translate).toBeDisabled();
    expect(
      screen.getByText(/Translation is unavailable while timestamps are enabled/i),
    ).toBeInTheDocument();
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
