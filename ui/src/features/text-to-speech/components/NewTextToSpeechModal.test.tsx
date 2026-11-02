import { fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { api, type ModelInfo, type SavedVoiceSummary } from "@/api";
import { NewTextToSpeechModal } from "./NewTextToSpeechModal";

function model(
  variant: string,
  extra: Partial<NonNullable<ModelInfo["speech_capabilities"]>> = {},
): ModelInfo {
  return {
    variant,
    status: "ready",
    local_path: null,
    size_bytes: null,
    download_progress: null,
    error_message: null,
    speech_capabilities: {
      supports_builtin_voices: false,
      built_in_voice_count: null,
      supports_reference_voice: true,
      reference_transcript_required: true,
      supports_voice_description: false,
      supports_streaming: true,
      supports_speed_control: false,
      supports_auto_long_form: true,
      ...extra,
    },
  };
}

const FISH_S2 = model("FishAudio-S2-Pro");
const DRAMA3 = model("FishAudio-Drama-3", {
  supports_seeded_voice: true,
  supports_speaker_dialogue: true,
  max_speakers: 16,
  max_references: 8,
  inline_tags: ["[sighs]", "[laughs]"],
});

function savedVoice(
  id: string,
  name: string,
  transcriptChars = 12,
): SavedVoiceSummary {
  return {
    id,
    created_at: 1,
    updated_at: 1,
    name,
    reference_text_preview: transcriptChars ? "Hello there." : "",
    reference_text_chars: transcriptChars,
    audio_mime_type: "audio/wav",
    audio_filename: "voice.wav",
    source_route_kind: "voice_cloning",
    source_record_id: null,
  };
}

function renderModal(selected: ModelInfo, onCreateStream = vi.fn()) {
  render(
    <NewTextToSpeechModal
      isOpen
      onClose={vi.fn()}
      selectedModel={selected.variant}
      selectedModelInfo={selected}
      selectedModelReady
      onOpenModelManager={vi.fn()}
      onModelRequired={vi.fn()}
      onCreated={vi.fn()}
      onCreateStream={onCreateStream}
    />,
  );
  return onCreateStream;
}

async function pick(trigger: HTMLElement, option: string | RegExp) {
  fireEvent.keyDown(trigger, { key: "Enter" });
  fireEvent.click(await screen.findByRole("option", { name: option }));
}

function openDialogue() {
  fireEvent.click(screen.getByRole("radio", { name: "Dialogue" }));
}

function lineText(number: number) {
  return screen.getByRole("textbox", { name: `Line ${number} text` });
}

describe("NewTextToSpeechModal with Fish Audio Drama 3", () => {
  beforeEach(() => {
    HTMLElement.prototype.scrollIntoView = vi.fn();
    vi.spyOn(api, "listSavedVoices").mockResolvedValue([
      savedVoice("voice-anna", "Anna"),
      savedVoice("voice-ben", "Ben"),
      savedVoice("voice-mute", "Mute", 0),
    ]);
    URL.createObjectURL = vi.fn(() => "blob:line-preview");
    URL.revokeObjectURL = vi.fn();
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("offers Dialogue only for a model that reports speaker dialogue", async () => {
    renderModal(FISH_S2);
    expect(await screen.findByText("Saved voice")).toBeInTheDocument();
    expect(screen.queryByRole("radiogroup", { name: "Speech mode" })).toBeNull();
    expect(screen.queryByLabelText("Seed")).toBeNull();
  });

  it("speaks a single voice without a reference, from the seed", async () => {
    const onCreateStream = renderModal(
      DRAMA3,
      vi.fn().mockResolvedValue({ id: "record" }),
    );
    expect(
      screen.getByRole("radiogroup", { name: "Speech mode" }),
    ).toBeInTheDocument();
    expect(await screen.findByText("Saved voice (optional)")).toBeInTheDocument();
    expect(screen.getByRole("combobox", { name: "Saved voice" })).toHaveTextContent(
      "None (seeded voice)",
    );

    fireEvent.change(screen.getByRole("textbox", { name: "Script" }), {
      target: { value: "[sighs] A long day." },
    });
    fireEvent.change(screen.getByLabelText("Seed"), { target: { value: "42" } });
    fireEvent.click(screen.getByRole("button", { name: /Create generation/ }));

    await waitFor(() => expect(onCreateStream).toHaveBeenCalledTimes(1));
    const request = onCreateStream.mock.calls[0][0];
    expect(request).toMatchObject({
      model_id: "FishAudio-Drama-3",
      text: "[sighs] A long day.",
      seed: 42,
    });
    expect(request.saved_voice_id).toBeUndefined();
  });

  it("keeps a saved voice's transcript requirement in the single-voice flow", async () => {
    renderModal(DRAMA3);
    const picker = screen.getByRole("combobox", { name: "Saved voice" });
    await waitFor(() => expect(picker).not.toBeDisabled());
    fireEvent.keyDown(picker, { key: "Enter" });
    expect(await screen.findByRole("option", { name: "Anna (Clone)" })).toBeInTheDocument();
    // A voice saved without a transcript cannot voice this model.
    expect(screen.queryByRole("option", { name: /Mute/ })).toBeNull();
  });

  it("refuses a seed that is not a whole number", async () => {
    const onCreateStream = renderModal(DRAMA3);
    fireEvent.change(screen.getByRole("textbox", { name: "Script" }), {
      target: { value: "Hello." },
    });
    fireEvent.change(screen.getByLabelText("Seed"), { target: { value: "-3" } });
    fireEvent.click(screen.getByRole("button", { name: /Create generation/ }));
    expect(await screen.findByRole("alert")).toHaveTextContent(
      "The seed must be a whole number",
    );
    expect(onCreateStream).not.toHaveBeenCalled();
  });

  it("holds speakers to the model's limits and voices to those with a transcript", async () => {
    renderModal(
      model("FishAudio-Drama-3", {
        supports_seeded_voice: true,
        supports_speaker_dialogue: true,
        max_speakers: 3,
        max_references: 1,
      }),
    );
    openDialogue();
    expect(screen.getByText("2 of 3 speakers · 0 of 1 saved voices")).toBeInTheDocument();

    const addSpeaker = screen.getByRole("button", { name: "Add speaker" });
    fireEvent.click(addSpeaker);
    expect(screen.getByRole("combobox", { name: "Speaker 2 voice" })).toBeInTheDocument();
    expect(addSpeaker).toBeDisabled();
    // An unused speaker can go; one with lines cannot.
    expect(screen.getByRole("button", { name: "Remove Speaker 2" })).toBeEnabled();
    expect(screen.getByRole("button", { name: "Remove Speaker 0" })).toBeDisabled();

    const speaker0 = screen.getByRole("combobox", { name: "Speaker 0 voice" });
    await waitFor(() => expect(speaker0).not.toBeDisabled());
    fireEvent.keyDown(speaker0, { key: "Enter" });
    expect(
      await screen.findByRole("option", { name: "Mute (Clone) · no transcript" }),
    ).toHaveAttribute("aria-disabled", "true");
    fireEvent.click(screen.getByRole("option", { name: "Anna (Clone)" }));
    expect(speaker0).toHaveTextContent("Anna (Clone)");
    expect(
      screen.getByText("3 of 3 speakers · 1 of 1 saved voices"),
    ).toBeInTheDocument();

    // The model's one reference is taken: other speakers stay seeded.
    fireEvent.keyDown(screen.getByRole("combobox", { name: "Speaker 1 voice" }), {
      key: "Enter",
    });
    expect(
      await screen.findByRole("option", { name: "Ben (Clone)" }),
    ).toHaveAttribute("aria-disabled", "true");
  });

  it("previews one line with only its speaker's voice, outside history", async () => {
    const generate = vi
      .spyOn(api, "generateTTSWithStats")
      .mockResolvedValue({ audioBlob: new Blob(["wav"]), stats: null });
    const createRecord = vi.spyOn(api, "createTextToSpeechRecord");
    const onCreateStream = renderModal(DRAMA3);
    openDialogue();

    const speaker0 = screen.getByRole("combobox", { name: "Speaker 0 voice" });
    await waitFor(() => expect(speaker0).not.toBeDisabled());
    await pick(speaker0, "Anna (Clone)");
    fireEvent.change(lineText(1), { target: { value: "Ready when you are." } });
    fireEvent.change(lineText(2), { target: { value: " Always. " } });
    fireEvent.change(screen.getByLabelText("Seed"), { target: { value: "7" } });

    fireEvent.click(screen.getByRole("button", { name: "Preview line 2" }));
    await waitFor(() => expect(generate).toHaveBeenCalledTimes(1));
    expect(generate.mock.calls[0][0]).toEqual({
      model_id: "FishAudio-Drama-3",
      script: [{ speaker: 1, text: "Always." }],
      references: undefined,
      language: undefined,
      speed: undefined,
      seed: 7,
    });
    expect(
      await screen.findByRole("button", { name: "Stop the preview of line 2" }),
    ).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: "Preview line 1" }));
    await waitFor(() => expect(generate).toHaveBeenCalledTimes(2));
    expect(generate.mock.calls[1][0]).toMatchObject({
      script: [{ speaker: 0, text: "Ready when you are." }],
      references: [{ speaker: 0, saved_voice_id: "voice-anna" }],
      seed: 7,
    });
    expect(createRecord).not.toHaveBeenCalled();
    expect(onCreateStream).not.toHaveBeenCalled();
  });

  it("generates the scene through the record path with its script and voices", async () => {
    const onCreateStream = renderModal(
      DRAMA3,
      vi.fn().mockResolvedValue({ id: "record" }),
    );
    openDialogue();
    const speaker1 = screen.getByRole("combobox", { name: "Speaker 1 voice" });
    await waitFor(() => expect(speaker1).not.toBeDisabled());
    await pick(speaker1, "Ben (Clone)");
    fireEvent.change(lineText(1), { target: { value: "Ready when you are." } });
    fireEvent.change(lineText(2), { target: { value: "Always." } });
    fireEvent.click(screen.getByRole("button", { name: "Add line" }));
    // A new line goes to the next speaker in turn.
    expect(screen.getByRole("combobox", { name: "Line 3 speaker" })).toHaveTextContent(
      "Speaker 0",
    );
    fireEvent.change(lineText(3), { target: { value: "Let's go." } });

    // A tag helper writes the model's tag into the line.
    await pick(
      screen.getByRole("combobox", { name: "Insert a tag into line 2" }),
      "[laughs]",
    );
    expect(lineText(2)).toHaveValue("Always. [laughs] ");
    await pick(
      screen.getByRole("combobox", { name: "Insert a tag into line 3" }),
      "<whisper>…</whisper>",
    );
    expect(lineText(3)).toHaveValue("Let's go.<whisper></whisper>");

    fireEvent.change(screen.getByLabelText("Seed"), { target: { value: "11" } });
    expect(screen.getByText("3 lines · 64 characters")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: /Generate scene/ }));

    await waitFor(() => expect(onCreateStream).toHaveBeenCalledTimes(1));
    const request = onCreateStream.mock.calls[0][0];
    expect(request).toEqual({
      model_id: "FishAudio-Drama-3",
      script: [
        { speaker: 0, text: "Ready when you are." },
        { speaker: 1, text: "Always. [laughs]" },
        { speaker: 0, text: "Let's go.<whisper></whisper>" },
      ],
      references: [{ speaker: 1, saved_voice_id: "voice-ben" }],
      language: undefined,
      speed: undefined,
      seed: 11,
    });
    expect(request).not.toHaveProperty("text");
  });

  it("checks lines before sending, then shows the server's 400 as it is", async () => {
    const serverMessage =
      "`references[0]` is a saved voice without a transcript, which FishAudio-Drama-3 needs. Use a voice saved with one.";
    const onCreateStream = renderModal(
      DRAMA3,
      vi.fn().mockRejectedValue(new Error(serverMessage)),
    );
    openDialogue();
    fireEvent.change(lineText(1), { target: { value: "Hello." } });
    fireEvent.click(screen.getByRole("button", { name: /Generate scene/ }));
    expect(await screen.findByRole("alert")).toHaveTextContent(
      "Line 2 is empty. Write it or remove it.",
    );
    expect(onCreateStream).not.toHaveBeenCalled();

    fireEvent.click(screen.getByRole("button", { name: "Remove line 2" }));
    fireEvent.click(screen.getByRole("button", { name: /Generate scene/ }));
    await waitFor(() => expect(onCreateStream).toHaveBeenCalledTimes(1));
    expect(await screen.findByRole("alert")).toHaveTextContent(serverMessage);
  });

  it("shows a line preview's 400 as it is", async () => {
    vi.spyOn(api, "generateTTSWithStats").mockRejectedValue(
      new Error("Control tokens are written by the prompt builder: <|im_start|>"),
    );
    renderModal(DRAMA3);
    openDialogue();
    fireEvent.change(lineText(1), { target: { value: "Hi <|im_start|>" } });
    fireEvent.click(screen.getByRole("button", { name: "Preview line 1" }));
    expect(await screen.findByRole("alert")).toHaveTextContent(
      "Control tokens are written by the prompt builder: <|im_start|>",
    );
    expect(
      within(screen.getByRole("listitem", { name: "Line 1" })).getByRole("button", {
        name: "Preview line 1",
      }),
    ).toBeEnabled();
  });
});
