import {
  act,
  fireEvent,
  render,
  screen,
  waitFor,
  within,
} from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import type { ModelInfo } from "@/api";
import { VoicePage } from "./route";

const apiMocks = vi.hoisted(() => ({
  getVoiceProfile: vi.fn(),
  listVoiceObservations: vi.fn(),
}));

vi.mock("@/api", () => ({
  api: {
    baseUrl: "http://localhost:3000",
    getVoiceProfile: apiMocks.getVoiceProfile,
    listVoiceObservations: apiMocks.listVoiceObservations,
  },
}));

describe("VoicePage model readiness", () => {
  beforeEach(() => {
    apiMocks.getVoiceProfile.mockReset();
    apiMocks.listVoiceObservations.mockReset();
    apiMocks.getVoiceProfile.mockResolvedValue({
      id: "default",
      name: "Default",
      system_prompt: "Be helpful.",
      default_system_prompt: "Be helpful.",
      observational_memory_enabled: true,
      created_at: 0,
      updated_at: 0,
    });
    apiMocks.listVoiceObservations.mockResolvedValue([]);

    Object.defineProperty(window, "localStorage", {
      configurable: true,
      value: {
        getItem: vi.fn(() => null),
        setItem: vi.fn(),
        removeItem: vi.fn(),
        clear: vi.fn(),
      },
    });
    Object.defineProperty(window, "requestAnimationFrame", {
      configurable: true,
      value: (callback: FrameRequestCallback) => {
        callback(0);
        return 1;
      },
    });
    Object.defineProperty(window, "cancelAnimationFrame", {
      configurable: true,
      value: vi.fn(),
    });
    Object.defineProperty(HTMLElement.prototype, "scrollIntoView", {
      configurable: true,
      value: vi.fn(),
    });
  });

  it("keeps Start visible and provides one actionable route to model setup", async () => {
    render(
      <VoicePage
        models={[]}
        loading={false}
        downloadProgress={{}}
        onDownload={vi.fn()}
        onLoad={vi.fn()}
        onUnload={vi.fn()}
        onDelete={vi.fn()}
      />,
    );

    const start = screen.getByRole("button", { name: "Start Conversation" });
    const setup = screen.getByRole("button", { name: "Set up voice models" });

    expect(start).toBeVisible();
    expect(start).toBeDisabled();
    expect(start).toHaveAccessibleDescription(
      "Add the required ASR, text, and TTS models before starting a conversation.",
    );
    expect(
      screen.getAllByRole("button", { name: "Set up voice models" }),
    ).toHaveLength(1);

    setup.focus();
    fireEvent.click(setup);

    const dialog = await screen.findByRole("dialog", {
      name: "Voice configuration",
    });
    expect(dialog).toHaveFocus();
    expect(screen.getByRole("tab", { name: /Models/i })).toHaveAttribute(
      "data-state",
      "active",
    );

    fireEvent.click(within(dialog).getByRole("button", { name: "Close" }));
    await waitFor(() => expect(setup).toHaveFocus());
  });
});

function model(variant: string, status: ModelInfo["status"]): ModelInfo {
  return {
    variant,
    status,
    local_path: null,
    size_bytes: null,
    download_progress: null,
    error_message: null,
  };
}

const STANDARD_DOWNLOADED = [
  model("Parakeet-TDT-0.6B-v3", "downloaded"),
  model("Qwen3.5-4B", "downloaded"),
  model("Kokoro-82M", "downloaded"),
];

function renderVoicePage(models: ModelInfo[]) {
  return render(
    <VoicePage
      models={models}
      loading={false}
      downloadProgress={{}}
      onDownload={vi.fn()}
      onLoad={vi.fn()}
      onUnload={vi.fn()}
      onDelete={vi.fn()}
    />,
  );
}

class FakeWebSocket {
  static readonly CONNECTING = 0;
  static readonly OPEN = 1;
  static readonly CLOSING = 2;
  static readonly CLOSED = 3;
  static instances: FakeWebSocket[] = [];

  readyState = FakeWebSocket.CONNECTING;
  binaryType = "blob";
  sent: unknown[] = [];
  onopen: (() => void) | null = null;
  onmessage: ((event: { data: unknown }) => void) | null = null;
  onerror: (() => void) | null = null;
  onclose: (() => void) | null = null;
  private sequence = 0;

  constructor(readonly url: string) {
    FakeWebSocket.instances.push(this);
  }

  send(data: unknown) {
    this.sent.push(typeof data === "string" ? JSON.parse(data) : data);
  }

  close() {
    this.readyState = FakeWebSocket.CLOSED;
  }

  open() {
    this.readyState = FakeWebSocket.OPEN;
    this.onopen?.();
  }

  /** Deliver the next typed v2 event. */
  emit(type: string, fields: Record<string, unknown> = {}) {
    const sequence = this.sequence++;
    this.onmessage?.({
      data: JSON.stringify({
        protocol: "voice_realtime",
        version: 2,
        event_id: sequence + 1,
        sequence,
        session_id: "voice-session-1",
        connection_epoch: 0,
        timestamp_ms: 1_725_000_000_000,
        type,
        ...fields,
      }),
    });
  }

  serverClose() {
    this.readyState = FakeWebSocket.CLOSED;
    this.onclose?.();
  }
}

class FakeAudioContext {
  sampleRate = 48_000;
  currentTime = 0;
  state = "running";
  destination = {};
  createMediaStreamSource() {
    return { connect: vi.fn() };
  }
  createAnalyser() {
    return {
      fftSize: 2048,
      smoothingTimeConstant: 0,
      connect: vi.fn(),
      getByteTimeDomainData: vi.fn(),
    };
  }
  createScriptProcessor() {
    return { connect: vi.fn(), disconnect: vi.fn(), onaudioprocess: null };
  }
  createGain() {
    return { gain: { value: 1 }, connect: vi.fn(), disconnect: vi.fn() };
  }
  close() {
    return Promise.resolve();
  }
  resume() {
    return Promise.resolve();
  }
}

describe("VoicePage conversation", () => {
  const micTrack = { stop: vi.fn() };

  beforeEach(() => {
    apiMocks.getVoiceProfile.mockResolvedValue({
      id: "default",
      name: "Default",
      system_prompt: "Be helpful.",
      default_system_prompt: "Be helpful.",
      observational_memory_enabled: true,
      created_at: 0,
      updated_at: 0,
    });
    apiMocks.listVoiceObservations.mockResolvedValue([]);
    FakeWebSocket.instances = [];
    micTrack.stop.mockReset();
    vi.stubGlobal("WebSocket", FakeWebSocket);
    vi.stubGlobal("AudioContext", FakeAudioContext);
    Object.defineProperty(navigator, "mediaDevices", {
      configurable: true,
      value: {
        getUserMedia: vi.fn().mockResolvedValue({ getTracks: () => [micTrack] }),
      },
    });
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("starts on downloaded models, which the server loads first", async () => {
    renderVoicePage(STANDARD_DOWNLOADED);

    const start = screen.getByRole("button", { name: "Start Conversation" });
    expect(start).toBeEnabled();
    fireEvent.click(start);

    await waitFor(() => expect(FakeWebSocket.instances).toHaveLength(1));
    const socket = FakeWebSocket.instances[0];
    await act(async () => socket.open());
    expect(screen.getByRole("heading", { name: "Getting ready" })).toBeVisible();

    await act(async () => {
      socket.emit("session_ready", {
        data: {
          accepted_version: 2,
          owner_instance_id: "process-1",
          resumable: false,
          resume_window_ms: 0,
        },
      });
    });
    await waitFor(() =>
      expect(socket.sent).toContainEqual(
        expect.objectContaining({
          type: "input_stream_start",
          mode: "modular",
          speaker: "af_heart",
        }),
      ),
    );
    await act(async () => {
      socket.emit("session_started");
      socket.emit("models_status", {
        data: {
          models: [
            { stage: "asr", model_id: "Parakeet-TDT-0.6B-v3", ready: true },
            { stage: "text", model_id: "Qwen3.5-4B", ready: true },
            { stage: "tts", model_id: "Kokoro-82M", ready: true },
          ],
        },
      });
    });
    expect(screen.getByRole("heading", { name: "Listening" })).toBeVisible();
  });

  it("shows your words and the reply, which typed events send without start events", async () => {
    renderVoicePage(STANDARD_DOWNLOADED);
    fireEvent.click(screen.getByRole("button", { name: "Start Conversation" }));
    await waitFor(() => expect(FakeWebSocket.instances).toHaveLength(1));
    const socket = FakeWebSocket.instances[0];
    const turn = { utterance_id: "utterance-1", turn_id: "turn-1" };

    await act(async () => {
      socket.open();
      socket.emit("session_started");
      socket.emit("speech_started", turn);
      socket.emit("transcript_partial", {
        ...turn,
        data: { text: "What's a good", revision: 1 },
      });
      socket.emit("transcript_final", {
        ...turn,
        data: { text: "What's a good pace?", revision: 2 },
      });
      socket.emit("assistant_text_started", turn);
      socket.emit("assistant_text_final", {
        ...turn,
        data: { text: "Easy enough to talk." },
      });
    });

    expect(screen.getAllByText("What's a good pace?").length).toBeGreaterThan(0);
    expect(screen.getAllByText("Easy enough to talk.").length).toBeGreaterThan(0);
  });

  it("stops with the model error when a model cannot load", async () => {
    renderVoicePage(STANDARD_DOWNLOADED);
    fireEvent.click(screen.getByRole("button", { name: "Start Conversation" }));
    await waitFor(() => expect(FakeWebSocket.instances).toHaveLength(1));
    const socket = FakeWebSocket.instances[0];

    await act(async () => {
      socket.open();
      socket.emit("session_started");
      socket.emit("models_status", {
        data: {
          models: [
            {
              stage: "text",
              model_id: "Qwen3.5-4B",
              ready: false,
              error: "needs 6.47 GB, 1.97 GB available",
            },
          ],
        },
      });
    });

    expect(
      screen.getByText("Couldn't load Qwen3.5 4B: needs 6.47 GB, 1.97 GB available"),
    ).toBeInTheDocument();
    expect(micTrack.stop).toHaveBeenCalled();
    expect(screen.getByRole("heading", { name: "Idle" })).toBeVisible();
  });

  it("turns the microphone off when the connection drops", async () => {
    renderVoicePage(STANDARD_DOWNLOADED);
    fireEvent.click(screen.getByRole("button", { name: "Start Conversation" }));
    await waitFor(() => expect(FakeWebSocket.instances).toHaveLength(1));
    const socket = FakeWebSocket.instances[0];
    await act(async () => {
      socket.open();
      socket.emit("session_started");
    });

    await act(async () => socket.serverClose());

    expect(micTrack.stop).toHaveBeenCalled();
    expect(screen.getByRole("heading", { name: "Idle" })).toBeVisible();
    expect(screen.getByText("Voice realtime connection closed")).toBeInTheDocument();
  });

  it("starts the Compact engine without the Standard models", async () => {
    renderVoicePage([model("LFM2.5-Audio-1.5B-GGUF", "downloaded")]);
    expect(
      screen.getByRole("button", { name: "Start Conversation" }),
    ).toBeDisabled();

    fireEvent.click(screen.getByRole("button", { name: /Settings/ }));
    const dialog = await screen.findByRole("dialog", { name: "Voice configuration" });
    fireEvent.click(within(dialog).getByRole("button", { name: /Unified LFM2.5 Audio/ }));
    fireEvent.click(within(dialog).getByRole("button", { name: "Close" }));

    const start = screen.getByRole("button", { name: "Start Conversation" });
    expect(start).toBeEnabled();
    fireEvent.click(start);
    await waitFor(() => expect(FakeWebSocket.instances).toHaveLength(1));
    const socket = FakeWebSocket.instances[0];
    await act(async () => {
      socket.open();
    });
    await waitFor(() =>
      expect(socket.sent).toContainEqual(
        expect.objectContaining({
          type: "input_stream_start",
          mode: "unified",
          s2s_model_id: "LFM2.5-Audio-1.5B-GGUF",
        }),
      ),
    );
  });
});
