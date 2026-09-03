import { act, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { NewDiarizationModal } from "./NewDiarizationModal";

const apiMocks = vi.hoisted(() => ({
  createDiarizationRecord: vi.fn(),
}));

vi.mock("@/api", () => ({
  api: {
    createDiarizationRecord: apiMocks.createDiarizationRecord,
  },
}));

vi.mock("@/shared/audioUpload", () => ({
  prepareSpeechTextUploadBlob: vi.fn(async (blob: Blob) => blob),
  resolveSourceAudioFilename: vi.fn(() => "audio.webm"),
  resolveSpeechTextUploadFilename: vi.fn(() => "audio.webm"),
}));

interface RecorderHarness {
  instances: MockMediaRecorder[];
  stopTrack: ReturnType<typeof vi.fn>;
  getUserMedia: ReturnType<typeof vi.fn>;
}

class MockMediaRecorder {
  static instances: MockMediaRecorder[] = [];

  static isTypeSupported(mimeType: string): boolean {
    return mimeType.startsWith("audio/webm");
  }

  readonly mimeType: string;
  state: RecordingState = "inactive";
  ondataavailable: ((event: { data: Blob }) => void) | null = null;
  onstop: (() => void | Promise<void>) | null = null;
  stop = vi.fn(() => {
    this.state = "inactive";
    this.ondataavailable?.({
      data: new Blob(["recorded"], { type: this.mimeType }),
    });
    void this.onstop?.();
  });

  constructor(_stream: MediaStream, options?: MediaRecorderOptions) {
    this.mimeType = options?.mimeType ?? "audio/webm";
    MockMediaRecorder.instances.push(this);
  }

  start(): void {
    this.state = "recording";
  }
}

function installRecorderHarness(): RecorderHarness {
  MockMediaRecorder.instances = [];
  const stopTrack = vi.fn();
  const getUserMedia = vi.fn(async () => ({
    getTracks: () => [{ stop: stopTrack }],
  }) as unknown as MediaStream);

  vi.stubGlobal("MediaRecorder", MockMediaRecorder);
  Object.defineProperty(navigator, "mediaDevices", {
    configurable: true,
    value: { getUserMedia },
  });

  return {
    instances: MockMediaRecorder.instances,
    stopTrack,
    getUserMedia,
  };
}

function renderModal(
  overrides: Partial<React.ComponentProps<typeof NewDiarizationModal>> = {},
) {
  const props: React.ComponentProps<typeof NewDiarizationModal> = {
    isOpen: true,
    onClose: vi.fn(),
    selectedModel: "diar_streaming_sortformer_4spk-v2.1",
    selectedModelReady: true,
    pipelineAsrModelId: "Parakeet-TDT-0.6B-v3",
    pipelineAlignerModelId: "Qwen3-ForcedAligner-0.6B",
    pipelineModelsReady: true,
    onModelRequired: vi.fn(),
    onPipelineModelsRequired: vi.fn(),
    onOpenModelManager: vi.fn(),
    onLoadAllManagedModels: vi.fn(),
    onUnloadAllManagedModels: vi.fn(),
    onCreated: vi.fn(),
    ...overrides,
  };

  return {
    props,
    view: render(<NewDiarizationModal {...props} />),
  };
}

describe("NewDiarizationModal recording lifecycle", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    apiMocks.createDiarizationRecord.mockResolvedValue({ id: "diar-1" });
  });

  it("stops microphone tracks before submitting a completed recording", async () => {
    const harness = installRecorderHarness();
    renderModal();

    fireEvent.click(screen.getByRole("button", { name: /Record audio/i }));
    await waitFor(() => expect(harness.instances).toHaveLength(1));
    fireEvent.click(screen.getByRole("button", { name: /Stop recording/i }));

    await waitFor(() =>
      expect(apiMocks.createDiarizationRecord).toHaveBeenCalledTimes(1),
    );
    expect(harness.stopTrack).toHaveBeenCalledTimes(1);
  });

  it("releases tracks when submission fails", async () => {
    const harness = installRecorderHarness();
    apiMocks.createDiarizationRecord.mockRejectedValue(new Error("backend failed"));
    renderModal();

    fireEvent.click(screen.getByRole("button", { name: /Record audio/i }));
    await waitFor(() => expect(harness.instances).toHaveLength(1));
    fireEvent.click(screen.getByRole("button", { name: /Stop recording/i }));

    expect(await screen.findAllByText("backend failed")).not.toHaveLength(0);
    expect(harness.stopTrack).toHaveBeenCalledTimes(1);
  });

  it("cancels an active recording when the modal is externally closed", async () => {
    const harness = installRecorderHarness();
    const { props, view } = renderModal();

    fireEvent.click(screen.getByRole("button", { name: /Record audio/i }));
    await waitFor(() => expect(harness.instances).toHaveLength(1));
    const lateOnStop = harness.instances[0].onstop;

    view.rerender(<NewDiarizationModal {...props} isOpen={false} />);
    await waitFor(() => expect(harness.stopTrack).toHaveBeenCalledTimes(1));
    expect(harness.instances[0].stop).toHaveBeenCalledTimes(1);

    await act(async () => {
      await lateOnStop?.();
    });
    expect(apiMocks.createDiarizationRecord).not.toHaveBeenCalled();
  });

  it("releases tracks on unmount and ignores a late recorder callback", async () => {
    const harness = installRecorderHarness();
    const { view } = renderModal();

    fireEvent.click(screen.getByRole("button", { name: /Record audio/i }));
    await waitFor(() => expect(harness.instances).toHaveLength(1));
    const lateOnStop = harness.instances[0].onstop;
    view.unmount();

    expect(harness.stopTrack).toHaveBeenCalledTimes(1);
    expect(harness.instances[0].stop).toHaveBeenCalledTimes(1);
    await act(async () => {
      await lateOnStop?.();
    });
    expect(apiMocks.createDiarizationRecord).not.toHaveBeenCalled();
  });

  it("stops a permission stream that resolves after unmount", async () => {
    MockMediaRecorder.instances = [];
    const stopTrack = vi.fn();
    let resolvePermission!: (stream: MediaStream) => void;
    const permission = new Promise<MediaStream>((resolve) => {
      resolvePermission = resolve;
    });
    vi.stubGlobal("MediaRecorder", MockMediaRecorder);
    Object.defineProperty(navigator, "mediaDevices", {
      configurable: true,
      value: { getUserMedia: vi.fn(() => permission) },
    });

    const { view } = renderModal();
    fireEvent.click(screen.getByRole("button", { name: /Record audio/i }));
    view.unmount();

    await act(async () => {
      resolvePermission({
        getTracks: () => [{ stop: stopTrack }],
      } as unknown as MediaStream);
      await permission;
    });

    expect(stopTrack).toHaveBeenCalledTimes(1);
    expect(MockMediaRecorder.instances).toHaveLength(0);
    expect(apiMocks.createDiarizationRecord).not.toHaveBeenCalled();
  });
});
