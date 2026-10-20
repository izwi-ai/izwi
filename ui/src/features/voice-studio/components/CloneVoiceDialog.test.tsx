import { act, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { CloneVoiceDialog } from "./CloneVoiceDialog";

const mocks = vi.hoisted(() => ({
  createSavedVoice: vi.fn(),
  prepareReferenceClip: vi.fn(),
}));

vi.mock("@/api", () => ({
  api: { createSavedVoice: mocks.createSavedVoice },
}));

vi.mock("@/utils/audioBase64", () => ({
  blobToBase64Payload: vi.fn(async () => "UklGRg=="),
}));

vi.mock("./referenceClip", async (importOriginal) => ({
  ...(await importOriginal<typeof import("./referenceClip")>()),
  prepareReferenceClip: mocks.prepareReferenceClip,
}));

class MockMediaRecorder {
  static instances: MockMediaRecorder[] = [];

  static isTypeSupported(mimeType: string): boolean {
    return mimeType.startsWith("audio/webm");
  }

  readonly mimeType: string;
  state: RecordingState = "inactive";
  ondataavailable: ((event: { data: Blob }) => void) | null = null;
  onstop: (() => void) | null = null;
  stop = vi.fn(() => {
    this.state = "inactive";
    this.ondataavailable?.({ data: new Blob(["recorded"], { type: this.mimeType }) });
    this.onstop?.();
  });

  constructor(_stream: MediaStream, options?: MediaRecorderOptions) {
    this.mimeType = options?.mimeType ?? "audio/webm";
    MockMediaRecorder.instances.push(this);
  }

  start(): void {
    this.state = "recording";
  }
}

function installMicrophone() {
  MockMediaRecorder.instances = [];
  const stopTrack = vi.fn();
  vi.stubGlobal("MediaRecorder", MockMediaRecorder);
  Object.defineProperty(navigator, "mediaDevices", {
    configurable: true,
    value: {
      getUserMedia: vi.fn(async () => ({
        getTracks: () => [{ stop: stopTrack }],
      })),
    },
  });
  return { stopTrack };
}

function clipOf(durationSecs: number) {
  return {
    blob: new Blob(["wav"], { type: "audio/wav" }),
    durationSecs,
    peaks: [0.2, 1, 0.5],
  };
}

function renderDialog(
  overrides: Partial<React.ComponentProps<typeof CloneVoiceDialog>> = {},
) {
  const props: React.ComponentProps<typeof CloneVoiceDialog> = {
    open: true,
    onClose: vi.fn(),
    model: "Qwen3-TTS-12Hz-0.6B-Base",
    modelReady: true,
    onOpenModelManager: vi.fn(),
    onSaved: vi.fn(),
    onUseInTextToSpeech: vi.fn(),
    ...overrides,
  };
  return { props, view: render(<CloneVoiceDialog {...props} />) };
}

async function uploadClip(fileName: string) {
  const input = document.querySelector('input[type="file"]') as HTMLInputElement;
  fireEvent.change(input, {
    target: { files: [new File(["audio"], fileName, { type: "audio/wav" })] },
  });
  await screen.findByText(fileName);
}

function fillForm({ consent = true }: { consent?: boolean } = {}) {
  fireEvent.change(screen.getByLabelText("Name"), { target: { value: "Sam, narrator" } });
  fireEvent.change(screen.getByLabelText("What's said in the recording"), {
    target: { value: "Hi, I'm Sam. I read the evening news." },
  });
  if (consent) {
    fireEvent.click(
      screen.getByRole("checkbox", { name: "I have permission to clone this voice" }),
    );
  }
}

const saveButton = () => screen.getByRole("button", { name: /Save voice/ });

describe("CloneVoiceDialog", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    URL.createObjectURL = vi.fn(() => "blob:clip");
    URL.revokeObjectURL = vi.fn();
    mocks.prepareReferenceClip.mockResolvedValue(clipOf(12.4));
    mocks.createSavedVoice.mockResolvedValue({ id: "voice-1", name: "Sam, narrator" });
  });

  afterEach(() => {
    vi.useRealTimers();
    vi.unstubAllGlobals();
  });

  it("needs consent before saving, then sends it with the clone", async () => {
    const { props } = renderDialog();
    await uploadClip("sam-reference.wav");
    fillForm({ consent: false });

    expect(screen.getAllByText("0:12")).toHaveLength(2);
    expect(screen.getByText("8 words")).toBeInTheDocument();
    expect(saveButton()).toBeDisabled();

    fireEvent.click(
      screen.getByRole("checkbox", { name: "I have permission to clone this voice" }),
    );
    expect(saveButton()).toBeEnabled();
    fireEvent.click(saveButton());

    expect(await screen.findByRole("dialog", { name: "Voice saved" })).toBeInTheDocument();
    expect(mocks.createSavedVoice).toHaveBeenCalledWith({
      name: "Sam, narrator",
      reference_text: "Hi, I'm Sam. I read the evening news.",
      audio_base64: "UklGRg==",
      audio_mime_type: "audio/wav",
      audio_filename: "sam-reference.wav",
      source_route_kind: "voice_cloning",
      consent: true,
    });
    expect(props.onSaved).toHaveBeenCalledWith("voice-1");

    fireEvent.click(screen.getByRole("button", { name: /Use in Text to Speech/ }));
    expect(props.onUseInTextToSpeech).toHaveBeenCalledWith("voice-1");
    expect(props.onClose).toHaveBeenCalled();
  });

  it.each([
    [2.4, "Too short", "0:02"],
    [31.2, "Too long", "0:32"],
  ])("refuses a %s s clip before saving", async (secs, problem, shown) => {
    mocks.prepareReferenceClip.mockResolvedValue(clipOf(secs));
    renderDialog();
    await uploadClip("clip.mp3");
    fillForm();

    expect(saveButton()).toBeDisabled();
    expect(screen.getByText(`${problem}: 3 to 30 seconds`)).toBeInTheDocument();
    expect(screen.getAllByText(shown)).not.toHaveLength(0);

    fireEvent.keyDown(screen.getByLabelText("Name"), { key: "Enter", metaKey: true });
    expect(mocks.createSavedVoice).not.toHaveBeenCalled();
  });

  it("stops recording at 30 seconds and keeps the clip within the limit", async () => {
    vi.useFakeTimers({ shouldAdvanceTime: true });
    const { stopTrack } = installMicrophone();
    renderDialog();

    fireEvent.click(screen.getByRole("radio", { name: "Record" }));
    await act(async () => {
      fireEvent.click(screen.getByRole("button", { name: "Start recording" }));
    });
    expect(MockMediaRecorder.instances).toHaveLength(1);
    expect(screen.getByRole("button", { name: "Stop recording" })).toBeInTheDocument();

    await act(async () => {
      vi.advanceTimersByTime(29_000);
    });
    expect(MockMediaRecorder.instances[0].stop).not.toHaveBeenCalled();
    expect(screen.getByText("0:29 / 0:30")).toBeInTheDocument();

    await act(async () => {
      vi.advanceTimersByTime(1_000);
    });
    expect(MockMediaRecorder.instances[0].stop).toHaveBeenCalledTimes(1);
    expect(stopTrack).toHaveBeenCalledTimes(1);
    expect(mocks.prepareReferenceClip).toHaveBeenCalledWith(expect.any(Blob), {
      maxSecs: 30,
    });
    expect(await screen.findByText("Microphone recording")).toBeInTheDocument();
  });

  it("releases the microphone when the dialog closes mid-recording", async () => {
    const { stopTrack } = installMicrophone();
    const { props, view } = renderDialog();

    fireEvent.click(screen.getByRole("radio", { name: "Record" }));
    await act(async () => {
      fireEvent.click(screen.getByRole("button", { name: "Start recording" }));
    });
    const recorder = MockMediaRecorder.instances[0];

    view.rerender(<CloneVoiceDialog {...props} open={false} />);

    expect(stopTrack).toHaveBeenCalledTimes(1);
    expect(recorder.stop).toHaveBeenCalledTimes(1);
    expect(mocks.prepareReferenceClip).not.toHaveBeenCalled();
  });

  it("releases the microphone on unmount mid-recording", async () => {
    const { stopTrack } = installMicrophone();
    const { view } = renderDialog();

    fireEvent.click(screen.getByRole("radio", { name: "Record" }));
    await act(async () => {
      fireEvent.click(screen.getByRole("button", { name: "Start recording" }));
    });
    view.unmount();

    expect(stopTrack).toHaveBeenCalledTimes(1);
    expect(MockMediaRecorder.instances[0].stop).toHaveBeenCalledTimes(1);
  });

  it("stops a microphone that is granted after the dialog went away", async () => {
    MockMediaRecorder.instances = [];
    vi.stubGlobal("MediaRecorder", MockMediaRecorder);
    const stopTrack = vi.fn();
    let grant!: (stream: MediaStream) => void;
    Object.defineProperty(navigator, "mediaDevices", {
      configurable: true,
      value: {
        getUserMedia: vi.fn(
          () => new Promise<MediaStream>((resolve) => {
            grant = resolve;
          }),
        ),
      },
    });
    const { view } = renderDialog();

    fireEvent.click(screen.getByRole("radio", { name: "Record" }));
    fireEvent.click(screen.getByRole("button", { name: "Start recording" }));
    view.unmount();
    await act(async () => {
      grant({ getTracks: () => [{ stop: stopTrack }] } as unknown as MediaStream);
    });

    expect(stopTrack).toHaveBeenCalledTimes(1);
    expect(MockMediaRecorder.instances).toHaveLength(0);
  });
});
