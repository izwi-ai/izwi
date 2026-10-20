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
  updateVoiceProfile: vi.fn(),
  listVoiceObservations: vi.fn(),
  deleteVoiceObservation: vi.fn(),
  clearVoiceObservations: vi.fn(),
}));

vi.mock("@/api", () => ({
  api: {
    baseUrl: "http://localhost:3000",
    ...apiMocks,
  },
}));

function model(
  variant: string,
  status: ModelInfo["status"],
  size = 1_000_000_000,
): ModelInfo {
  return {
    variant,
    status,
    local_path: null,
    size_bytes: status === "not_downloaded" ? null : size,
    estimated_size_bytes: size,
    download_progress: null,
    error_message: null,
  };
}

const STANDARD = ["Parakeet-TDT-0.6B-v3", "Qwen3.5-4B", "Kokoro-82M"];
const standardModels = (status: ModelInfo["status"]) =>
  STANDARD.map((variant) => model(variant, status));

const handlers = {
  onDownload: vi.fn(),
  onCancelDownload: vi.fn(),
  onLoad: vi.fn(),
  onUnload: vi.fn(),
  onDelete: vi.fn(),
  onError: vi.fn(),
};

function renderVoicePage(models: ModelInfo[]) {
  return render(
    <VoicePage
      models={models}
      loading={false}
      downloadProgress={{}}
      {...handlers}
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

const micTrack: { stop: ReturnType<typeof vi.fn>; onended?: (() => void) | null } = {
  stop: vi.fn(),
};
const storage = new Map<string, string>();

beforeEach(() => {
  Object.values(handlers).forEach((handler) => handler.mockReset());
  apiMocks.getVoiceProfile.mockResolvedValue({
    id: "default",
    name: "Default",
    system_prompt: "Be helpful.",
    default_system_prompt: "Be helpful.",
    observational_memory_enabled: true,
    created_at: 0,
    updated_at: 0,
  });
  apiMocks.listVoiceObservations.mockResolvedValue([
    {
      id: "memory-1",
      profile_id: "default",
      category: "preference",
      summary: "Prefers short answers",
      confidence: 0.9,
      source_turn_id: null,
      source_user_text: null,
      source_assistant_text: null,
      times_seen: 2,
      created_at: 1_760_000_000_000,
      updated_at: 1_760_000_000_000,
      forgotten_at: null,
    },
  ]);
  apiMocks.deleteVoiceObservation.mockResolvedValue({ id: "memory-1", deleted: true });
  storage.clear();
  Object.defineProperty(window, "localStorage", {
    configurable: true,
    value: {
      getItem: (key: string) => storage.get(key) ?? null,
      setItem: (key: string, value: string) => storage.set(key, value),
      removeItem: (key: string) => storage.delete(key),
      clear: () => storage.clear(),
    },
  });
  Object.defineProperty(HTMLElement.prototype, "scrollIntoView", {
    configurable: true,
    value: vi.fn(),
  });
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

async function startConversation() {
  fireEvent.click(screen.getByRole("button", { name: "Start talking" }));
  await waitFor(() => expect(FakeWebSocket.instances).toHaveLength(1));
  const socket = FakeWebSocket.instances[0];
  await act(async () => socket.open());
  return socket;
}

describe("VoicePage setup", () => {
  it("downloads every missing model of the engine with one action", () => {
    renderVoicePage(standardModels("not_downloaded"));

    expect(screen.getByRole("heading", { name: "Set up voice" })).toBeVisible();
    expect(screen.queryByRole("button", { name: "Start talking" })).toBeNull();
    fireEvent.click(
      screen.getByRole("button", { name: "Download and set up · 2.79 GB" }),
    );

    expect(handlers.onDownload.mock.calls.map(([variant]) => variant)).toEqual(
      STANDARD,
    );
  });

  it("switches to the Compact engine from setup and remembers it", async () => {
    renderVoicePage([
      ...standardModels("not_downloaded"),
      model("LFM2.5-Audio-1.5B-GGUF", "downloaded"),
    ]);

    fireEvent.click(screen.getByRole("button", { name: "Use the compact engine" }));

    expect(storage.get("izwi.voice.engine")).toBe("compact");
    const socket = await startConversation();
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

describe("VoicePage conversation", () => {
  it("starts on downloaded models, which the server loads first", async () => {
    renderVoicePage(standardModels("downloaded"));
    expect(screen.getByRole("heading", { name: "Talk to Izwi" })).toBeVisible();
    expect(screen.getByRole("button", { name: "Heart" })).toBeVisible();

    const socket = await startConversation();
    expect(screen.getByRole("status")).toHaveTextContent("Getting ready");

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
          models: STANDARD.map((variant, index) => ({
            stage: ["asr", "text", "tts"][index],
            model_id: variant,
            ready: true,
          })),
        },
      });
    });
    expect(screen.getByRole("status")).toHaveTextContent("Listening");
    expect(screen.getByRole("button", { name: /End/ })).toBeVisible();
  });

  it("shows your words and the reply, in the caption and the transcript", async () => {
    renderVoicePage(standardModels("downloaded"));
    const socket = await startConversation();
    const turn = { utterance_id: "utterance-1", turn_id: "turn-1" };

    await act(async () => {
      socket.emit("session_started");
      socket.emit("speech_started", turn);
      socket.emit("transcript_final", {
        ...turn,
        data: { text: "What's a good pace?", revision: 2 },
      });
    });
    expect(screen.getByText("What's a good pace?")).toBeVisible();

    await act(async () => {
      socket.emit("assistant_text_started", turn);
      socket.emit("assistant_text_final", {
        ...turn,
        data: { text: "Easy enough to talk." },
      });
    });
    fireEvent.click(screen.getByRole("button", { name: "Show transcript" }));

    const transcript = screen.getByRole("complementary", { name: "Transcript" });
    expect(within(transcript).getByText("What's a good pace?")).toBeVisible();
    expect(within(transcript).getByText("Easy enough to talk.")).toBeVisible();
    expect(within(transcript).getByText("You")).toBeVisible();
  });

  it("mutes the microphone and ends with Space", async () => {
    renderVoicePage(standardModels("downloaded"));
    await startConversation();

    fireEvent.click(screen.getByRole("button", { name: "Mute microphone" }));
    expect(screen.getByRole("status")).toHaveTextContent("Muted");
    expect(screen.getByRole("button", { name: "Unmute microphone" })).toHaveAttribute(
      "aria-pressed",
      "true",
    );

    fireEvent.keyDown(window, { code: "Space", key: " " });
    expect(micTrack.stop).toHaveBeenCalled();
    expect(screen.getByRole("button", { name: "Start talking" })).toBeVisible();
  });
});

describe("VoicePage failures", () => {
  it("stops with the model error when a model cannot load", async () => {
    renderVoicePage(standardModels("downloaded"));
    const socket = await startConversation();

    await act(async () => {
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
      screen.getByRole("heading", { name: "Couldn't load the voice models" }),
    ).toBeVisible();
    expect(
      screen.getByText("Couldn't load Qwen3.5 4B: needs 6.47 GB, 1.97 GB available"),
    ).toBeVisible();
    expect(micTrack.stop).toHaveBeenCalled();
    fireEvent.click(screen.getByRole("button", { name: "Change engine" }));
    expect(await screen.findByRole("dialog", { name: "Engine" })).toBeVisible();
  });

  it("turns the microphone off when the connection drops", async () => {
    renderVoicePage(standardModels("downloaded"));
    const socket = await startConversation();
    await act(async () => socket.emit("session_started"));

    await act(async () => socket.serverClose());

    expect(micTrack.stop).toHaveBeenCalled();
    expect(screen.getByRole("heading", { name: "Conversation ended" })).toBeVisible();
    expect(screen.getByRole("button", { name: "Start again" })).toBeVisible();
  });

  it("ends the conversation when the microphone goes away", async () => {
    renderVoicePage(standardModels("downloaded"));
    const socket = await startConversation();
    await act(async () => socket.emit("session_started"));

    await act(async () => micTrack.onended?.());

    expect(screen.getByRole("heading", { name: "Conversation ended" })).toBeVisible();
    expect(screen.getByText(/The microphone stopped/)).toBeVisible();
    expect(micTrack.stop).toHaveBeenCalled();
  });

  it("explains a blocked microphone", async () => {
    vi.mocked(navigator.mediaDevices.getUserMedia).mockRejectedValueOnce(
      new DOMException("denied", "NotAllowedError"),
    );
    renderVoicePage(standardModels("downloaded"));

    fireEvent.click(screen.getByRole("button", { name: "Start talking" }));

    expect(
      await screen.findByRole("heading", { name: "Microphone is blocked" }),
    ).toBeVisible();
    expect(FakeWebSocket.instances).toHaveLength(0);
  });
});

describe("VoicePage settings", () => {
  it("remembers the chosen voice", async () => {
    renderVoicePage(standardModels("downloaded"));
    fireEvent.click(screen.getByRole("button", { name: "Voice settings" }));
    const sheet = await screen.findByRole("dialog", { name: "Voice settings" });

    fireEvent.click(within(sheet).getByRole("button", { name: /Show all \d+ voices/ }));
    fireEvent.click(within(sheet).getByRole("radio", { name: /Emma/ }));

    expect(storage.get("izwi.voice.speaker")).toBe("bf_emma");
    expect(within(sheet).getByRole("radio", { name: /Emma/ })).toHaveAttribute(
      "aria-checked",
      "true",
    );
  });

  it("saves the instructions when you leave the field", async () => {
    apiMocks.updateVoiceProfile.mockResolvedValue({
      id: "default",
      name: "Default",
      system_prompt: "Answer like a running coach.",
      default_system_prompt: "Be helpful.",
      observational_memory_enabled: true,
      created_at: 0,
      updated_at: 0,
    });
    renderVoicePage(standardModels("downloaded"));
    fireEvent.click(screen.getByRole("button", { name: "Voice settings" }));
    const sheet = await screen.findByRole("dialog", { name: "Voice settings" });
    const instructions = within(sheet).getByLabelText("Instructions");
    await waitFor(() => expect(instructions).toBeEnabled());

    fireEvent.change(instructions, {
      target: { value: "Answer like a running coach." },
    });
    fireEvent.blur(instructions);

    await waitFor(() =>
      expect(apiMocks.updateVoiceProfile).toHaveBeenCalledWith({
        system_prompt: "Answer like a running coach.",
      }),
    );
    expect(
      await within(sheet).findByRole("button", { name: "Reset to default" }),
    ).toBeVisible();
  });

  it("lists and forgets memories", async () => {
    renderVoicePage(standardModels("downloaded"));
    fireEvent.click(screen.getByRole("button", { name: "Voice settings" }));
    const sheet = await screen.findByRole("dialog", { name: "Voice settings" });

    fireEvent.click(within(sheet).getByRole("button", { name: /Manage memories/ }));
    const memories = await screen.findByRole("dialog", { name: "Memories" });
    fireEvent.click(
      await within(memories).findByRole("button", {
        name: "Forget: Prefers short answers",
      }),
    );

    await waitFor(() =>
      expect(apiMocks.deleteVoiceObservation).toHaveBeenCalledWith("memory-1"),
    );
    expect(await within(memories).findByText("Nothing remembered yet.")).toBeVisible();
  });
});
