import { act, fireEvent, render, screen } from "@testing-library/react";
import { useState } from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { DesignVoiceDialog, PREVIEW_LINE_MAX_CHARS } from "./DesignVoiceDialog";

const mocks = vi.hoisted(() => ({
  createVoiceDesignRecord: vi.fn(),
  deleteVoiceDesignRecord: vi.fn(),
  createSavedVoice: vi.fn(),
}));

vi.mock("@/api", () => ({
  api: {
    createVoiceDesignRecord: mocks.createVoiceDesignRecord,
    deleteVoiceDesignRecord: mocks.deleteVoiceDesignRecord,
    createSavedVoice: mocks.createSavedVoice,
    voiceDesignRecordAudioUrl: (id: string) => `http://izwi.test/voice-designs/${id}/audio`,
  },
}));

vi.mock("@/utils/audioBase64", () => ({
  blobToBase64Payload: vi.fn(async () => "UklGRg=="),
}));

function designRecord(id: string) {
  return {
    id,
    input_text: "Hello, this is Izwi.",
    audio_duration_secs: 6.2,
    audio_mime_type: "audio/wav",
    audio_filename: `${id}.wav`,
  };
}

type DialogProps = React.ComponentProps<typeof DesignVoiceDialog>;

function Harness(props: Partial<DialogProps>) {
  const [open, setOpen] = useState(true);
  return (
    <DesignVoiceDialog
      open={open}
      onClose={() => setOpen(false)}
      model="Qwen3-TTS-12Hz-1.7B-VoiceDesign"
      modelReady
      onOpenModelManager={vi.fn()}
      onModelRequired={vi.fn()}
      onSaved={vi.fn()}
      {...props}
    />
  );
}

async function generateThreeOptions() {
  fireEvent.change(screen.getByLabelText("Describe the voice"), {
    target: { value: "A warm, calm narrator." },
  });
  await act(async () => {
    fireEvent.click(screen.getByRole("button", { name: "Generate options" }));
  });
  await screen.findByRole("radio", { name: "Option 3" });
}

const deletedIds = () => mocks.deleteVoiceDesignRecord.mock.calls.map(([id]) => id);

describe("DesignVoiceDialog", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    let next = 0;
    mocks.createVoiceDesignRecord.mockImplementation(async () =>
      designRecord(`design-${++next}`),
    );
    mocks.deleteVoiceDesignRecord.mockResolvedValue({ deleted: true });
    mocks.createSavedVoice.mockResolvedValue({ id: "voice-1", name: "Calm narrator" });
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => new Response(new Blob(["wav"], { type: "audio/wav" }))),
    );
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("caps the preview line at 300 characters", () => {
    render(<Harness />);
    const line = screen.getByLabelText("Preview line") as HTMLTextAreaElement;

    fireEvent.change(line, { target: { value: "a".repeat(350) } });

    expect(line.value).toHaveLength(PREVIEW_LINE_MAX_CHARS);
    expect(line).toHaveAttribute("maxLength", "300");
    expect(screen.getByText("300 / 300")).toBeInTheDocument();
  });

  it("fills the description from an example", () => {
    render(<Harness />);

    fireEvent.click(screen.getByRole("button", { name: "Examples" }));
    fireEvent.click(screen.getByRole("button", { name: "Calm narrator" }));

    expect(
      (screen.getByLabelText("Describe the voice") as HTMLTextAreaElement).value,
    ).toMatch(/warm, calm narrator/);
  });

  it("asks for the model instead of generating without one", () => {
    const onModelRequired = vi.fn();
    render(<Harness modelReady={false} onModelRequired={onModelRequired} />);

    fireEvent.change(screen.getByLabelText("Describe the voice"), {
      target: { value: "A warm, calm narrator." },
    });
    fireEvent.click(screen.getByRole("button", { name: "Generate options" }));

    expect(onModelRequired).toHaveBeenCalled();
    expect(mocks.createVoiceDesignRecord).not.toHaveBeenCalled();
  });

  it("saves the picked option and deletes the others", async () => {
    const onSaved = vi.fn();
    render(<Harness onSaved={onSaved} />);
    await generateThreeOptions();

    expect(mocks.createVoiceDesignRecord).toHaveBeenCalledTimes(3);
    expect(mocks.createVoiceDesignRecord.mock.calls[0][0]).toMatchObject({
      text: "Hello, this is Izwi. Here's how I sound reading a short paragraph aloud.",
      voice_description: "A warm, calm narrator.",
      language: undefined,
    });
    const save = screen.getByRole("button", { name: /Save option/ });
    expect(save).toBeDisabled();

    fireEvent.click(screen.getByRole("radio", { name: "Option 2" }));
    expect(screen.getByRole("button", { name: /Save option 2/ })).toBeDisabled();
    fireEvent.change(screen.getByLabelText("Name"), {
      target: { value: "Calm narrator" },
    });
    await act(async () => {
      fireEvent.click(screen.getByRole("button", { name: /Save option 2/ }));
    });

    expect(await screen.findByRole("dialog", { name: "Voice saved" })).toBeInTheDocument();
    expect(mocks.createSavedVoice).toHaveBeenCalledWith(
      expect.objectContaining({
        name: "Calm narrator",
        source_route_kind: "voice_design",
        source_record_id: "design-2",
      }),
    );
    expect(onSaved).toHaveBeenCalledWith("voice-1");
    expect(deletedIds().sort()).toEqual(["design-1", "design-3"]);

    fireEvent.click(screen.getByRole("button", { name: "Cancel" }));
    expect(deletedIds()).not.toContain("design-2");
  });

  it("deletes every option when cancelled", async () => {
    render(<Harness />);
    await generateThreeOptions();

    fireEvent.click(screen.getByRole("button", { name: "Cancel" }));

    expect(deletedIds().sort()).toEqual(["design-1", "design-2", "design-3"]);
    expect(mocks.createSavedVoice).not.toHaveBeenCalled();
  });

  it("deletes the previous options when generating again", async () => {
    render(<Harness />);
    await generateThreeOptions();

    await act(async () => {
      fireEvent.click(screen.getByRole("button", { name: "Generate again" }));
    });

    expect(deletedIds().sort()).toEqual(["design-1", "design-2", "design-3"]);
    expect(await screen.findByRole("radio", { name: "Option 3" })).toBeInTheDocument();
    expect(mocks.createVoiceDesignRecord).toHaveBeenCalledTimes(6);
  });

  it("stops generating when the dialog closes and deletes what arrives late", async () => {
    let finishFirst!: () => void;
    mocks.createVoiceDesignRecord.mockImplementationOnce(
      () =>
        new Promise((resolve) => {
          finishFirst = () => resolve(designRecord("late-1"));
        }),
    );
    render(<Harness />);
    fireEvent.change(screen.getByLabelText("Describe the voice"), {
      target: { value: "A warm, calm narrator." },
    });
    fireEvent.click(screen.getByRole("button", { name: "Generate options" }));
    expect(await screen.findByText(/generating…/)).toBeInTheDocument();

    const signal: AbortSignal = mocks.createVoiceDesignRecord.mock.calls[0][1];
    expect(signal.aborted).toBe(false);
    fireEvent.click(screen.getByRole("button", { name: "Cancel" }));
    // The request is cancelled; one the server finished anyway is deleted.
    expect(signal.aborted).toBe(true);
    await act(async () => {
      finishFirst();
    });

    expect(mocks.createVoiceDesignRecord).toHaveBeenCalledTimes(1);
    expect(deletedIds()).toEqual(["late-1"]);
  });
});
