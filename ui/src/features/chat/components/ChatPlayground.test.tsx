import {
  act,
  fireEvent,
  render,
  screen,
  waitFor,
  within,
} from "@testing-library/react";
import { useEffect } from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { MemoryRouter, useNavigate, type NavigateFunction } from "react-router";
import { ChatPlayground } from "@/features/chat/components/ChatPlayground";
import { AUDIO_FILE_ACCEPT } from "@/shared/audioFileAccept";

const apiMocks = vi.hoisted(() => ({
  listChatThreads: vi.fn(),
  createResponse: vi.fn(),
  updateChatThread: vi.fn(),
  generateChatThreadTitle: vi.fn(),
  getChatThread: vi.fn(),
  createChatThread: vi.fn(),
  deleteChatThread: vi.fn(),
  sendChatThreadMessageStream: vi.fn(),
  generateTTSStream: vi.fn(),
  regenerateChatThreadMessageStream: vi.fn(),
  editChatThreadMessageStream: vi.fn(),
  deleteChatThreadMessage: vi.fn(),
}));

const playerMocks = vi.hoisted(() => ({
  instances: [] as Array<{
    unlock: ReturnType<typeof vi.fn>;
    start: ReturnType<typeof vi.fn>;
    push: ReturnType<typeof vi.fn>;
    finish: ReturnType<typeof vi.fn>;
    stop: ReturnType<typeof vi.fn>;
    onDrained: () => void;
  }>,
}));

vi.mock("@/features/text-to-speech/pcmPlayer", () => ({
  SpeechPcmPlayer: vi.fn().mockImplementation(function (
    this: Record<string, unknown>,
    _context: unknown,
    onDrained: () => void,
  ) {
    const instance = {
      unlock: vi.fn().mockResolvedValue(undefined),
      start: vi.fn(),
      push: vi.fn().mockResolvedValue(undefined),
      finish: vi.fn(),
      stop: vi.fn(),
      onDrained,
    };
    playerMocks.instances.push(instance);
    return instance;
  }),
}));

// jsdom has no AudioContext to decode with.
const clipMocks = vi.hoisted(() => ({
  prepareReferenceClip: vi.fn(),
}));

vi.mock("@/features/voice-studio/components/referenceClip", () => ({
  prepareReferenceClip: clipMocks.prepareReferenceClip,
}));

const createObjectUrlMock = vi.fn<(file: File) => string>();
const revokeObjectUrlMock = vi.fn<(url: string) => void>();

vi.mock("@/api", () => ({
  api: {
    listChatThreads: apiMocks.listChatThreads,
    createResponse: apiMocks.createResponse,
    updateChatThread: apiMocks.updateChatThread,
    generateChatThreadTitle: apiMocks.generateChatThreadTitle,
    getChatThread: apiMocks.getChatThread,
    createChatThread: apiMocks.createChatThread,
    deleteChatThread: apiMocks.deleteChatThread,
    sendChatThreadMessageStream: apiMocks.sendChatThreadMessageStream,
    generateTTSStream: apiMocks.generateTTSStream,
    regenerateChatThreadMessageStream: apiMocks.regenerateChatThreadMessageStream,
    editChatThreadMessageStream: apiMocks.editChatThreadMessageStream,
    deleteChatThreadMessage: apiMocks.deleteChatThreadMessage,
  },
}));

describe("ChatPlayground", () => {
  beforeEach(() => {
    apiMocks.listChatThreads.mockReset();
    apiMocks.createResponse.mockReset();
    apiMocks.updateChatThread.mockReset();
    apiMocks.generateChatThreadTitle.mockReset();
    apiMocks.getChatThread.mockReset();
    apiMocks.createChatThread.mockReset();
    apiMocks.deleteChatThread.mockReset();
    apiMocks.sendChatThreadMessageStream.mockReset();
    apiMocks.generateTTSStream.mockReset();
    apiMocks.regenerateChatThreadMessageStream.mockReset();
    apiMocks.editChatThreadMessageStream.mockReset();
    apiMocks.deleteChatThreadMessage.mockReset();
    playerMocks.instances.length = 0;
    createObjectUrlMock.mockReset();
    revokeObjectUrlMock.mockReset();
    createObjectUrlMock.mockImplementation((file) => `blob:preview-${file.name}`);
    Object.defineProperty(URL, "createObjectURL", {
      configurable: true,
      value: createObjectUrlMock,
    });
    Object.defineProperty(URL, "revokeObjectURL", {
      configurable: true,
      value: revokeObjectUrlMock,
    });

    apiMocks.listChatThreads.mockResolvedValue([]);

    HTMLElement.prototype.scrollIntoView = vi.fn();
  });

  it("opens the composer model dropdown and keeps the send action icon-only", async () => {
    const onOpenModelManager = vi.fn();
    render(
      <MemoryRouter initialEntries={["/chat"]}>
        <ChatPlayground
          selectedModel="Qwen3.5-4B"
          selectedModelReady={true}
          supportsThinking={true}
          modelLabel="Qwen3.5 4B GGUF (Q4_K_M)"
          modelOptions={[
            {
              value: "Qwen3.5-4B",
              label: "Qwen3.5 4B GGUF (Q4_K_M)",
              statusLabel: "Ready",
              isReady: true,
            },
            {
              value: "Qwen3.5-2B",
              label: "Qwen3.5 2B",
              statusLabel: "Downloaded",
              isReady: false,
              status: "downloaded",
            },
            {
              value: "Qwen3.8-27B-FP8",
              label: "Qwen3.8 27B (FP8)",
              statusLabel: "Not downloaded",
              isReady: false,
              status: "not_downloaded",
            },
          ]}
          onSelectModel={vi.fn()}
          onOpenModelManager={onOpenModelManager}
          onModelRequired={vi.fn()}
        />
      </MemoryRouter>,
    );

    await waitFor(() => expect(apiMocks.listChatThreads).toHaveBeenCalled());

    const modelPicker = screen.getByRole("combobox", { name: "Chat model" });
    expect(modelPicker).toHaveTextContent("Qwen3.5 4B GGUF (Q4_K_M)");
    expect(
      within(screen.getByTestId("chat-composer-actions")).getByRole("combobox", {
        name: "Chat model",
      }),
    ).toBe(modelPicker);
    expect(screen.queryByRole("button", { name: "Models" })).not.toBeInTheDocument();
    fireEvent.click(modelPicker);

    // Models on this device, by name; the rest are in the model manager.
    const listbox = await screen.findByRole("listbox");
    expect(
      within(listbox).getAllByRole("option").map((option) => option.textContent),
    ).toEqual(["Qwen3.5 4B GGUF (Q4_K_M)", "Qwen3.5 2B", "Manage chat models…"]);
    expect(
      within(listbox).getByRole("option", { name: "Qwen3.5 4B GGUF (Q4_K_M)" }),
    ).toHaveAttribute("aria-selected", "true");
    fireEvent.click(within(listbox).getByRole("option", { name: "Manage chat models…" }));
    expect(onOpenModelManager).toHaveBeenCalledOnce();
    await waitFor(() =>
      expect(screen.queryByRole("listbox")).not.toBeInTheDocument(),
    );
    expect(modelPicker).toHaveTextContent("Qwen3.5 4B GGUF (Q4_K_M)");

    const sendButton = screen.getByRole("button", { name: "Send message" });
    expect(sendButton).toBeInTheDocument();
    expect(sendButton).not.toHaveTextContent(/\bSend\b/i);
    expect(
      screen.queryByRole("button", { name: /Attach image or video/i }),
    ).not.toBeInTheDocument();
    expect(screen.getByTestId("chat-composer-actions")).not.toHaveClass(
      "border-t",
    );

    await waitFor(() =>
      expect(screen.getByRole("textbox")).toHaveStyle({ height: "72px" }),
    );
  });

  it("shows a failed thread as that thread's error state and retries it", async () => {
    const thread = {
      id: "thread-1",
      title: "Royal families in Europe",
      model_id: "Qwen3.5-4B",
      created_at: 1,
      updated_at: 2,
      last_message_preview: "How many ruling royal families are there?",
      message_count: 2,
    };
    apiMocks.listChatThreads.mockResolvedValue([thread]);
    apiMocks.getChatThread
      .mockRejectedValueOnce(
        new Error("Chat storage error: no such column: model_id"),
      )
      .mockResolvedValueOnce({
        thread,
        messages: [
          {
            id: "message-1",
            thread_id: "thread-1",
            role: "user",
            content: "How many ruling royal families are there?",
            created_at: 1,
            tokens_generated: null,
            generation_time_ms: null,
          },
        ],
      });

    render(
      <MemoryRouter initialEntries={["/chat?threadId=thread-1"]}>
        <ChatPlayground
          selectedModel="Qwen3.5-4B"
          selectedModelReady={true}
          supportsThinking={true}
          modelLabel="Qwen3.5 4B"
          modelOptions={[
            {
              value: "Qwen3.5-4B",
              label: "Qwen3.5 4B",
              statusLabel: "Ready",
              isReady: true,
            },
          ]}
          onSelectModel={vi.fn()}
          onOpenModelManager={vi.fn()}
          onModelRequired={vi.fn()}
        />
      </MemoryRouter>,
    );

    expect(
      await screen.findByText("This chat could not be loaded"),
    ).toBeInTheDocument();
    expect(
      screen.getByText("Chat storage error: no such column: model_id"),
    ).toBeInTheDocument();
    expect(
      screen.getByRole("heading", { name: "Royal families in Europe" }),
    ).toBeInTheDocument();
    expect(screen.queryByText("What can I help with?")).not.toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: "Retry" }));

    expect(
      await screen.findByText("How many ruling royal families are there?", {
        selector: "p, div, span",
      }),
    ).toBeInTheDocument();
    expect(
      screen.queryByText("This chat could not be loaded"),
    ).not.toBeInTheDocument();
  });

  it("shows the active thread title below the selector without the old conversation header", async () => {
    const thread = {
      id: "thread-1",
      title: "Royal families in Europe",
      model_id: "Qwen3.5-4B",
      created_at: 1,
      updated_at: 2,
      last_message_preview: "How many ruling royal families are there in Europe?",
      message_count: 2,
    };

    apiMocks.listChatThreads.mockResolvedValue([thread]);
    apiMocks.getChatThread.mockResolvedValue({
      thread,
      messages: [
        {
          id: "message-1",
          thread_id: "thread-1",
          role: "user",
          content: "How many ruling royal families are there in Europe?",
          created_at: 1,
          tokens_generated: null,
          generation_time_ms: null,
        },
        {
          id: "message-2",
          thread_id: "thread-1",
          role: "assistant",
          content: "There are several current ruling royal families in Europe.",
          created_at: 2,
          tokens_generated: 12,
          generation_time_ms: 120,
        },
      ],
    });

    render(
      <MemoryRouter initialEntries={["/chat?threadId=thread-1"]}>
        <ChatPlayground
          selectedModel="Qwen3.5-4B"
          selectedModelReady={true}
          supportsThinking={true}
          modelLabel="Qwen3.5 4B GGUF (Q4_K_M)"
          modelOptions={[
            {
              value: "Qwen3.5-4B",
              label: "Qwen3.5 4B GGUF (Q4_K_M)",
              statusLabel: "Ready",
              isReady: true,
            },
          ]}
          onSelectModel={vi.fn()}
          onOpenModelManager={vi.fn()}
          onModelRequired={vi.fn()}
        />
      </MemoryRouter>,
    );

    await waitFor(() =>
      expect(apiMocks.getChatThread).toHaveBeenCalledWith("thread-1"),
    );

    expect(screen.getByText("Royal families in Europe")).toBeInTheDocument();
    expect(screen.queryByText("2 messages")).not.toBeInTheDocument();
    expect(
      screen.queryByText("Using Qwen3.5 4B GGUF (Q4_K_M)"),
    ).not.toBeInTheDocument();

    // Stats moved from under the composer into the reply's own footer.
    const reply = screen.getByRole("article", { name: "Reply" });
    expect(within(reply).getByText("12 tokens")).toBeInTheDocument();
    expect(within(reply).getByText("100 tok/s")).toBeInTheDocument();
  });

  it("stops following streamed output while the user reads earlier messages", async () => {
    const thread = {
      id: "thread-scroll",
      title: "Scrollable thread",
      model_id: "Qwen3.5-0.8B",
      created_at: 1,
      updated_at: 2,
      last_message_preview: "Earlier answer",
      message_count: 2,
    };
    let streamCallbacks: { onDelta: (delta: string) => void } | null = null;

    apiMocks.listChatThreads.mockResolvedValue([thread]);
    apiMocks.getChatThread.mockResolvedValue({
      thread,
      messages: [
        {
          id: "message-user",
          thread_id: thread.id,
          role: "user",
          content: "Earlier question",
          created_at: 1,
          tokens_generated: null,
          generation_time_ms: null,
        },
        {
          id: "message-assistant",
          thread_id: thread.id,
          role: "assistant",
          content: "Earlier answer",
          created_at: 2,
          tokens_generated: 2,
          generation_time_ms: 10,
        },
      ],
    });
    apiMocks.sendChatThreadMessageStream.mockImplementation(
      (_threadId, _request, callbacks) => {
        streamCallbacks = callbacks;
        return new AbortController();
      },
    );

    render(
      <MemoryRouter initialEntries={["/chat?threadId=thread-scroll"]}>
        <ChatPlayground
          selectedModel="Qwen3.5-0.8B"
          selectedModelReady={true}
          supportsThinking={false}
          modelLabel="Qwen3.5 0.8B GGUF (Q4_K_M)"
          modelOptions={[
            {
              value: "Qwen3.5-0.8B",
              label: "Qwen3.5 0.8B GGUF (Q4_K_M)",
              statusLabel: "Ready",
              isReady: true,
            },
          ]}
          onSelectModel={vi.fn()}
          onOpenModelManager={vi.fn()}
          onModelRequired={vi.fn()}
        />
      </MemoryRouter>,
    );

    await screen.findByText("Earlier answer");
    const viewport = screen.getByTestId("chat-message-viewport");
    let scrollHeight = 1_200;
    Object.defineProperties(viewport, {
      clientHeight: { configurable: true, get: () => 400 },
      scrollHeight: { configurable: true, get: () => scrollHeight },
      scrollTop: { configurable: true, writable: true, value: 800 },
    });

    fireEvent.change(screen.getByRole("textbox"), {
      target: { value: "A new question" },
    });
    fireEvent.click(screen.getByRole("button", { name: "Send message" }));

    await waitFor(() => expect(streamCallbacks).not.toBeNull());
    expect(viewport.scrollTop).toBe(scrollHeight);

    expect(
      screen.queryByRole("button", { name: "Jump to latest" }),
    ).not.toBeInTheDocument();
    viewport.scrollTop = 300;
    fireEvent.scroll(viewport);
    scrollHeight = 1_300;
    act(() => streamCallbacks?.onDelta("First token"));
    expect(viewport.scrollTop).toBe(300);
    expect(
      screen.getByRole("button", { name: "Jump to latest" }),
    ).toBeInTheDocument();

    scrollHeight = 1_400;
    act(() => streamCallbacks?.onDelta(" second token"));
    expect(viewport.scrollTop).toBe(300);

    fireEvent.click(screen.getByRole("button", { name: "Jump to latest" }));
    expect(viewport.scrollTop).toBe(1_400);
    expect(
      screen.queryByRole("button", { name: "Jump to latest" }),
    ).not.toBeInTheDocument();
    scrollHeight = 1_500;
    act(() => streamCallbacks?.onDelta(" third token"));
    expect(viewport.scrollTop).toBe(scrollHeight);
  });

  it("lets the delete confirmation buttons work while the history drawer stays open", async () => {
    const thread = {
      id: "thread-1",
      title: "Royal families in Europe",
      model_id: "Qwen3.5-4B",
      created_at: 1,
      updated_at: 2,
      last_message_preview: "How many ruling royal families are there in Europe?",
      message_count: 2,
    };

    apiMocks.listChatThreads.mockResolvedValue([thread]);
    apiMocks.deleteChatThread.mockResolvedValue(undefined);

    render(
      <MemoryRouter initialEntries={["/chat"]}>
        <ChatPlayground
          selectedModel="Qwen3.5-4B"
          selectedModelReady={true}
          supportsThinking={true}
          modelLabel="Qwen3.5 4B GGUF (Q4_K_M)"
          modelOptions={[
            {
              value: "Qwen3.5-4B",
              label: "Qwen3.5 4B GGUF (Q4_K_M)",
              statusLabel: "Ready",
              isReady: true,
            },
          ]}
          onSelectModel={vi.fn()}
          onOpenModelManager={vi.fn()}
          onModelRequired={vi.fn()}
        />
      </MemoryRouter>,
    );

    await waitFor(() => expect(apiMocks.listChatThreads).toHaveBeenCalled());

    fireEvent.click(screen.getByRole("button", { name: /History/ }));

    expect(await screen.findByText("Chat history")).toBeInTheDocument();

    fireEvent.pointerDown(
      screen.getByRole("button", { name: "Delete Royal families in Europe" }),
    );
    fireEvent.click(
      screen.getByRole("button", { name: "Delete Royal families in Europe" }),
    );

    expect(
      await screen.findByRole("button", { name: "Cancel" }),
    ).toBeInTheDocument();
    expect(screen.getByText("Chat history")).toBeInTheDocument();
    expect(
      screen.getAllByText("Royal families in Europe").length,
    ).toBeGreaterThan(0);

    fireEvent.click(screen.getByRole("button", { name: "Cancel" }));

    await waitFor(() =>
      expect(screen.queryByText("Delete chat thread?")).not.toBeInTheDocument(),
    );
    expect(screen.getByText("Chat history")).toBeInTheDocument();

    fireEvent.pointerDown(
      screen.getByRole("button", { name: "Delete Royal families in Europe" }),
    );
    fireEvent.click(
      screen.getByRole("button", { name: "Delete Royal families in Europe" }),
    );

    expect(
      await screen.findByRole("button", { name: "Delete thread" }),
    ).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: "Delete thread" }));

    await waitFor(() =>
      expect(apiMocks.deleteChatThread).toHaveBeenCalledWith("thread-1"),
    );
    await waitFor(() =>
      expect(screen.queryByText("Delete chat thread?")).not.toBeInTheDocument(),
    );
  });

  it("wraps long delete-dialog thread titles instead of truncating them", async () => {
    const longTitle =
      "It seems It seems It seems It seems It seems It seems It seems It seems";
    const thread = {
      id: "thread-1",
      title: longTitle,
      model_id: "Qwen3.5-4B",
      created_at: 1,
      updated_at: 2,
      last_message_preview: "Preview",
      message_count: 2,
    };

    apiMocks.listChatThreads.mockResolvedValue([thread]);

    render(
      <MemoryRouter initialEntries={["/chat"]}>
        <ChatPlayground
          selectedModel="Qwen3.5-4B"
          selectedModelReady={true}
          supportsThinking={true}
          modelLabel="Qwen3.5 4B GGUF (Q4_K_M)"
          modelOptions={[
            {
              value: "Qwen3.5-4B",
              label: "Qwen3.5 4B GGUF (Q4_K_M)",
              statusLabel: "Ready",
              isReady: true,
            },
          ]}
          onSelectModel={vi.fn()}
          onOpenModelManager={vi.fn()}
          onModelRequired={vi.fn()}
        />
      </MemoryRouter>,
    );

    await waitFor(() => expect(apiMocks.listChatThreads).toHaveBeenCalled());

    fireEvent.click(screen.getByRole("button", { name: /History/ }));
    fireEvent.pointerDown(screen.getByRole("button", { name: `Delete ${longTitle}` }));
    fireEvent.click(screen.getByRole("button", { name: `Delete ${longTitle}` }));

    const dialog = await screen.findByRole("dialog");
    const threadTitle = within(dialog).getByText(longTitle);

    expect(threadTitle).toHaveClass("whitespace-normal");
    expect(threadTitle).toHaveClass("wrap-break-word");
    expect(threadTitle).not.toHaveClass("truncate");
  });

  it("shows the Qwen3.5 image affordance and sends image parts through the thread API", async () => {
    const thread = {
      id: "thread-1",
      title: "Vision thread",
      model_id: "Qwen3.5-4B",
      created_at: 1,
      updated_at: 2,
      last_message_preview: null,
      message_count: 0,
    };

    apiMocks.listChatThreads.mockResolvedValue([thread]);
    apiMocks.getChatThread.mockResolvedValue({
      thread,
      messages: [],
    });
    apiMocks.sendChatThreadMessageStream.mockReturnValue(new AbortController());

    render(
      <MemoryRouter initialEntries={["/chat?threadId=thread-1"]}>
        <ChatPlayground
          selectedModel="Qwen3.5-4B"
          selectedModelReady={true}
          supportsThinking={true}
          modelLabel="Qwen3.5 4B GGUF (Q4_K_M)"
          modelOptions={[
            {
              value: "Qwen3.5-4B",
              label: "Qwen3.5 4B GGUF (Q4_K_M)",
              statusLabel: "Ready",
              isReady: true,
            },
          ]}
          onSelectModel={vi.fn()}
          onOpenModelManager={vi.fn()}
          onModelRequired={vi.fn()}
        />
      </MemoryRouter>,
    );

    await waitFor(() =>
      expect(apiMocks.getChatThread).toHaveBeenCalledWith("thread-1"),
    );

    const composerActions = screen.getByTestId("chat-composer-actions");
    const attachImageButton = screen.getByRole("button", { name: "Attach image" });
    const thinkingButton = screen.getByRole("button", {
      name: "Disable thinking mode",
    });

    expect(attachImageButton).not.toHaveTextContent(/Attach image/i);

    const actionButtons = within(composerActions).getAllByRole("button");
    expect(actionButtons[0]).toBe(attachImageButton);
    expect(actionButtons[1]).toBe(thinkingButton);

    const imageInput = screen.getByTestId("chat-image-input");
    const imageFile = new File(["image"], "sample.png", { type: "image/png" });

    fireEvent.change(imageInput, {
      target: { files: [imageFile] },
    });

    expect(
      await screen.findByRole("button", { name: "Remove sample.png" }),
    ).toBeInTheDocument();

    fireEvent.change(screen.getByRole("textbox"), {
      target: { value: "Describe this" },
    });
    fireEvent.click(screen.getByRole("button", { name: "Send message" }));

    await waitFor(() =>
      expect(apiMocks.sendChatThreadMessageStream).toHaveBeenCalled(),
    );

    expect(apiMocks.sendChatThreadMessageStream).toHaveBeenCalledWith(
      "thread-1",
      expect.objectContaining({
        model_id: "Qwen3.5-4B",
        content: "Describe this",
        truncation: "auto",
        content_parts: [
          { type: "text", text: "Describe this" },
          {
            type: "input_image",
            input_image: {
              url: "data:image/png;base64,aW1hZ2U=",
              name: "sample.png",
            },
          },
        ],
      }),
      expect.any(Object),
    );
    expect(revokeObjectUrlMock).toHaveBeenCalledWith("blob:preview-sample.png");
  });

  it("allows attachment-only Qwen3.5 turns and sends a preview summary", async () => {
    const thread = {
      id: "thread-2",
      title: "Vision thread",
      model_id: "Qwen3.5-2B",
      created_at: 1,
      updated_at: 2,
      last_message_preview: null,
      message_count: 0,
    };

    apiMocks.listChatThreads.mockResolvedValue([thread]);
    apiMocks.getChatThread.mockResolvedValue({
      thread,
      messages: [],
    });
    apiMocks.sendChatThreadMessageStream.mockReturnValue(new AbortController());

    render(
      <MemoryRouter initialEntries={["/chat?threadId=thread-2"]}>
        <ChatPlayground
          selectedModel="Qwen3.5-2B"
          selectedModelReady={true}
          supportsThinking={true}
          modelLabel="Qwen3.5 2B GGUF (Q4_K_M)"
          modelOptions={[
            {
              value: "Qwen3.5-2B",
              label: "Qwen3.5 2B GGUF (Q4_K_M)",
              statusLabel: "Ready",
              isReady: true,
            },
          ]}
          onSelectModel={vi.fn()}
          onOpenModelManager={vi.fn()}
          onModelRequired={vi.fn()}
        />
      </MemoryRouter>,
    );

    await waitFor(() =>
      expect(apiMocks.getChatThread).toHaveBeenCalledWith("thread-2"),
    );

    const imageInput = screen.getByTestId("chat-image-input");
    fireEvent.change(imageInput, {
      target: {
        files: [new File(["vision"], "cat.png", { type: "image/png" })],
      },
    });

    expect(
      await screen.findByRole("button", { name: "Remove cat.png" }),
    ).toBeInTheDocument();

    const sendButton = screen.getByRole("button", { name: "Send message" });
    expect(sendButton).toBeEnabled();
    fireEvent.click(sendButton);

    await waitFor(() =>
      expect(apiMocks.sendChatThreadMessageStream).toHaveBeenCalled(),
    );

    expect(apiMocks.sendChatThreadMessageStream).toHaveBeenCalledWith(
      "thread-2",
      expect.objectContaining({
        model_id: "Qwen3.5-2B",
        content: "Attached image: cat.png",
        content_parts: [
          {
            type: "input_image",
            input_image: {
              url: "data:image/png;base64,dmlzaW9u",
              name: "cat.png",
            },
          },
        ],
      }),
      expect.any(Object),
    );
  });

  it("keeps pending images on a text-only model and blocks sending them", async () => {
    const thread = {
      id: "thread-switch",
      title: "Switch models",
      model_id: "Qwen3.5-4B",
      created_at: 1,
      updated_at: 2,
      last_message_preview: null,
      message_count: 0,
    };
    apiMocks.listChatThreads.mockResolvedValue([thread]);
    apiMocks.getChatThread.mockResolvedValue({ thread, messages: [] });
    apiMocks.sendChatThreadMessageStream.mockReturnValue(new AbortController());

    const renderWith = (model: string, label: string) => (
      <MemoryRouter initialEntries={["/chat?threadId=thread-switch"]}>
        <ChatPlayground
          selectedModel={model}
          selectedModelReady={true}
          supportsThinking={false}
          modelLabel={label}
          modelOptions={[
            { value: model, label, statusLabel: "Ready", isReady: true },
          ]}
          onSelectModel={vi.fn()}
          onOpenModelManager={vi.fn()}
          onModelRequired={vi.fn()}
        />
      </MemoryRouter>
    );

    const { rerender } = render(renderWith("Qwen3.5-4B", "Qwen3.5 4B"));
    await waitFor(() =>
      expect(apiMocks.getChatThread).toHaveBeenCalledWith("thread-switch"),
    );
    fireEvent.change(screen.getByTestId("chat-image-input"), {
      target: { files: [new File(["image"], "chart.png", { type: "image/png" })] },
    });
    expect(
      await screen.findByRole("button", { name: "Remove chart.png" }),
    ).toBeInTheDocument();

    rerender(renderWith("LFM2.5-1.2B-Instruct-GGUF", "LFM2.5 1.2B Instruct"));

    expect(
      screen.getByRole("button", { name: "Remove chart.png" }),
    ).toBeInTheDocument();
    expect(
      screen.getByText(/LFM2\.5 1\.2B Instruct can't read images/),
    ).toBeInTheDocument();
    fireEvent.change(screen.getByRole("textbox"), {
      target: { value: "Describe this" },
    });
    const send = screen.getByRole("button", { name: "Send message" });
    expect(send).toBeDisabled();
    fireEvent.click(send);
    expect(apiMocks.sendChatThreadMessageStream).not.toHaveBeenCalled();

    // Switching back makes the same attachment sendable again.
    rerender(renderWith("Qwen3.5-4B", "Qwen3.5 4B"));
    expect(screen.queryByText(/can't read images/)).not.toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Send message" })).toBeEnabled();
  });

  it("starts a new chat locally without creating an empty thread", async () => {
    const thread = {
      id: "thread-existing",
      title: "Existing chat",
      model_id: "Qwen3.5-4B",
      created_at: 1,
      updated_at: 2,
      last_message_preview: "Hello",
      message_count: 2,
    };
    apiMocks.listChatThreads.mockResolvedValue([thread]);
    apiMocks.getChatThread.mockResolvedValue({
      thread,
      messages: [
        {
          id: "m1",
          thread_id: thread.id,
          role: "user",
          content: "Hello",
          created_at: 1,
          tokens_generated: null,
          generation_time_ms: null,
        },
      ],
    });

    render(
      <MemoryRouter initialEntries={["/chat?threadId=thread-existing"]}>
        <ChatPlayground
          selectedModel="Qwen3.5-4B"
          selectedModelReady={true}
          supportsThinking={false}
          modelLabel="Qwen3.5 4B"
          modelOptions={[
            { value: "Qwen3.5-4B", label: "Qwen3.5 4B", statusLabel: "Ready", isReady: true },
          ]}
          onSelectModel={vi.fn()}
          onOpenModelManager={vi.fn()}
          onModelRequired={vi.fn()}
        />
      </MemoryRouter>,
    );

    await waitFor(() =>
      expect(apiMocks.getChatThread).toHaveBeenCalledWith("thread-existing"),
    );
    const newChat = screen.getByRole("button", { name: "New chat" });
    expect(newChat).toBeEnabled();
    fireEvent.click(newChat);

    expect(
      await screen.findByText(/No active chat selected/i),
    ).toBeInTheDocument();
    expect(apiMocks.createChatThread).not.toHaveBeenCalled();
    // Still enabled on the empty state: clicking it just focuses the composer.
    expect(screen.getByRole("button", { name: "New chat" })).toBeEnabled();
  });

  it("explains that a text-only model sees an image thread as text", async () => {
    const thread = {
      id: "thread-images",
      title: "Chart questions",
      model_id: "Qwen3.5-4B",
      created_at: 1,
      updated_at: 2,
      last_message_preview: "It is a bar chart",
      message_count: 2,
    };
    apiMocks.listChatThreads.mockResolvedValue([thread]);
    apiMocks.getChatThread.mockResolvedValue({
      thread,
      messages: [
        {
          id: "m1",
          thread_id: thread.id,
          role: "user",
          content: "What is this?",
          content_parts: [
            { type: "text", text: "What is this?" },
            {
              type: "input_image",
              input_image: { image_url: "data:image/png;base64,AAAA", detail: "auto" },
            },
          ],
          created_at: 1,
          tokens_generated: null,
          generation_time_ms: null,
        },
        {
          id: "m2",
          thread_id: thread.id,
          role: "assistant",
          content: "It is a bar chart",
          created_at: 2,
          tokens_generated: 5,
          generation_time_ms: 10,
        },
      ],
    });
    const onSelectModel = vi.fn();

    render(
      <MemoryRouter initialEntries={["/chat?threadId=thread-images"]}>
        <ChatPlayground
          selectedModel="LFM2.5-1.2B-Instruct-GGUF"
          selectedModelReady={true}
          supportsThinking={false}
          modelLabel="LFM2.5 1.2B Instruct"
          modelOptions={[
            {
              value: "LFM2.5-1.2B-Instruct-GGUF",
              label: "LFM2.5 1.2B Instruct",
              statusLabel: "Ready",
              isReady: true,
            },
            { value: "Qwen3.5-4B", label: "Qwen3.5 4B", statusLabel: "Downloaded", isReady: false },
          ]}
          onSelectModel={onSelectModel}
          onOpenModelManager={vi.fn()}
          onModelRequired={vi.fn()}
        />
      </MemoryRouter>,
    );

    expect(
      await screen.findByText(/earlier images are left out/i),
    ).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Switch to Qwen3.5 4B" }));
    expect(onSelectModel).toHaveBeenCalledWith("Qwen3.5-4B");
  });

  it("follows the catalog image capability over the model name", async () => {
    apiMocks.listChatThreads.mockResolvedValue([]);

    render(
      <MemoryRouter initialEntries={["/chat"]}>
        <ChatPlayground
          selectedModel="Qwen3.5-4B"
          selectedModelReady={true}
          supportsThinking={false}
          modelLabel="Qwen3.5 4B"
          modelOptions={[
            { value: "Qwen3.5-4B", label: "Qwen3.5 4B", statusLabel: "Ready", isReady: true },
          ]}
          modelSupportsImages={() => false}
          onSelectModel={vi.fn()}
          onOpenModelManager={vi.fn()}
          onModelRequired={vi.fn()}
        />
      </MemoryRouter>,
    );

    await waitFor(() => expect(apiMocks.listChatThreads).toHaveBeenCalled());
    expect(screen.queryByRole("button", { name: "Attach image" })).not.toBeInTheDocument();
  });

  it("isolates and restores text and image drafts when switching threads", async () => {
    const threads = [
      {
        id: "thread-a",
        title: "Thread A",
        model_id: "Qwen3.5-4B",
        created_at: 1,
        updated_at: 2,
        last_message_preview: "Earlier A message",
        message_count: 1,
      },
      {
        id: "thread-b",
        title: "Thread B",
        model_id: "Qwen3.5-4B",
        created_at: 1,
        updated_at: 2,
        last_message_preview: "Earlier B message",
        message_count: 1,
      },
    ];
    let streamCallbacks: { onClose: () => void } | null = null;

    apiMocks.listChatThreads.mockResolvedValue(threads);
    apiMocks.getChatThread.mockImplementation(async (threadId: string) => ({
      thread: threads.find((thread) => thread.id === threadId),
      messages: [
        {
          id: `message-${threadId}`,
          thread_id: threadId,
          role: "user",
          content: `Earlier ${threadId} message`,
          created_at: 1,
          tokens_generated: null,
          generation_time_ms: null,
        },
      ],
    }));
    apiMocks.sendChatThreadMessageStream.mockImplementation(
      (_threadId, _request, callbacks) => {
        streamCallbacks = callbacks;
        return new AbortController();
      },
    );

    render(
      <MemoryRouter initialEntries={["/chat?threadId=thread-a"]}>
        <ChatPlayground
          selectedModel="Qwen3.5-4B"
          selectedModelReady={true}
          supportsThinking={true}
          modelLabel="Qwen3.5 4B"
          modelOptions={[
            {
              value: "Qwen3.5-4B",
              label: "Qwen3.5 4B",
              statusLabel: "Ready",
              isReady: true,
            },
          ]}
          onSelectModel={vi.fn()}
          onOpenModelManager={vi.fn()}
          onModelRequired={vi.fn()}
        />
      </MemoryRouter>,
    );

    await waitFor(() =>
      expect(apiMocks.getChatThread).toHaveBeenCalledWith("thread-a"),
    );
    fireEvent.change(screen.getByRole("textbox"), {
      target: { value: "Draft intended only for A" },
    });
    fireEvent.change(screen.getByTestId("chat-image-input"), {
      target: {
        files: [new File(["image-a"], "a.png", { type: "image/png" })],
      },
    });
    expect(
      await screen.findByRole("button", { name: "Remove a.png" }),
    ).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: /History/ }));
    let drawer = await screen.findByRole("dialog", { name: "Chat history" });
    fireEvent.click(within(drawer).getByRole("button", { name: "Thread B" }));

    await waitFor(() => expect(screen.getByRole("textbox")).toHaveValue(""));
    expect(
      screen.queryByRole("button", { name: "Remove a.png" }),
    ).not.toBeInTheDocument();

    fireEvent.change(screen.getByRole("textbox"), {
      target: { value: "Draft intended only for B" },
    });
    fireEvent.click(screen.getByRole("button", { name: "Send message" }));

    await waitFor(() =>
      expect(apiMocks.sendChatThreadMessageStream).toHaveBeenCalledWith(
        "thread-b",
        expect.objectContaining({
          content: "Draft intended only for B",
          content_parts: undefined,
        }),
        expect.any(Object),
      ),
    );
    expect(screen.getByRole("textbox")).toHaveValue("");

    act(() => streamCallbacks?.onClose());
    fireEvent.click(screen.getByRole("button", { name: /History/ }));
    drawer = await screen.findByRole("dialog", { name: "Chat history" });
    fireEvent.click(within(drawer).getByRole("button", { name: "Thread A" }));

    await waitFor(() =>
      expect(screen.getByRole("textbox")).toHaveValue(
        "Draft intended only for A",
      ),
    );
    expect(
      screen.getByRole("button", { name: "Remove a.png" }),
    ).toBeInTheDocument();
  });

  it("transfers a new-chat draft to its created thread and revokes its preview on unmount", async () => {
    const createdThread = {
      id: "created-thread",
      title: "New chat",
      model_id: "Qwen3.5-4B",
      created_at: 1,
      updated_at: 1,
      last_message_preview: null,
      message_count: 0,
    };

    apiMocks.createChatThread.mockResolvedValue(createdThread);
    apiMocks.getChatThread.mockResolvedValue({
      thread: createdThread,
      messages: [],
    });
    apiMocks.sendChatThreadMessageStream.mockImplementation(() => {
      throw new Error("Stream setup failed");
    });

    const view = render(
      <MemoryRouter initialEntries={["/chat"]}>
        <ChatPlayground
          selectedModel="Qwen3.5-4B"
          selectedModelReady={true}
          supportsThinking={true}
          modelLabel="Qwen3.5 4B"
          modelOptions={[
            {
              value: "Qwen3.5-4B",
              label: "Qwen3.5 4B",
              statusLabel: "Ready",
              isReady: true,
            },
          ]}
          onSelectModel={vi.fn()}
          onOpenModelManager={vi.fn()}
          onModelRequired={vi.fn()}
        />
      </MemoryRouter>,
    );

    await waitFor(() => expect(apiMocks.listChatThreads).toHaveBeenCalled());
    fireEvent.change(screen.getByRole("textbox"), {
      target: { value: "Keep this if stream setup fails" },
    });
    fireEvent.change(screen.getByTestId("chat-image-input"), {
      target: {
        files: [new File(["image"], "draft.png", { type: "image/png" })],
      },
    });
    expect(
      await screen.findByRole("button", { name: "Remove draft.png" }),
    ).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: "Send message" }));

    await waitFor(() =>
      expect(apiMocks.sendChatThreadMessageStream).toHaveBeenCalledWith(
        "created-thread",
        expect.objectContaining({ content: "Keep this if stream setup fails" }),
        expect.any(Object),
      ),
    );
    // Stream setup runs before MemoryRouter commits its navigation transition.
    // Wait for the created thread to load before checking its transferred draft.
    await waitFor(() => {
      expect(apiMocks.getChatThread).toHaveBeenCalledWith("created-thread");
      expect(screen.getByRole("alert")).toHaveTextContent("Stream setup failed");
      expect(screen.getByRole("textbox")).toHaveValue(
        "Keep this if stream setup fails",
      );
      expect(
        screen.getByRole("button", { name: "Remove draft.png" }),
      ).toBeInTheDocument();
    });
    expect(revokeObjectUrlMock).not.toHaveBeenCalled();

    view.unmount();
    expect(revokeObjectUrlMock).toHaveBeenCalledTimes(1);
    expect(revokeObjectUrlMock).toHaveBeenCalledWith("blob:preview-draft.png");
  });

  it("uses Qwen3.5 variant defaults and forwards enable_thinking from the toggle", async () => {
    const thread = {
      id: "thread-3",
      title: "Thinking thread",
      model_id: "Qwen3.5-2B",
      created_at: 1,
      updated_at: 2,
      last_message_preview: null,
      message_count: 0,
    };

    apiMocks.listChatThreads.mockResolvedValue([thread]);
    apiMocks.getChatThread.mockResolvedValue({
      thread,
      messages: [],
    });
    apiMocks.sendChatThreadMessageStream.mockReturnValue(new AbortController());

    render(
      <MemoryRouter initialEntries={["/chat?threadId=thread-3"]}>
        <ChatPlayground
          selectedModel="Qwen3.5-2B"
          selectedModelReady={true}
          supportsThinking={true}
          modelLabel="Qwen3.5 2B GGUF (Q4_K_M)"
          modelOptions={[
            {
              value: "Qwen3.5-2B",
              label: "Qwen3.5 2B GGUF (Q4_K_M)",
              statusLabel: "Ready",
              isReady: true,
            },
          ]}
          onSelectModel={vi.fn()}
          onOpenModelManager={vi.fn()}
          onModelRequired={vi.fn()}
        />
      </MemoryRouter>,
    );

    await waitFor(() =>
      expect(apiMocks.getChatThread).toHaveBeenCalledWith("thread-3"),
    );

    expect(
      screen.getByRole("button", { name: "Enable thinking mode" }),
    ).toHaveTextContent("Thinking Off");
    expect(
      screen.queryByRole("combobox", { name: "Reasoning effort" }),
    ).not.toBeInTheDocument();

    fireEvent.click(
      screen.getByRole("button", { name: "Enable thinking mode" }),
    );
    expect(
      screen.getByRole("button", { name: "Disable thinking mode" }),
    ).toHaveTextContent("Thinking On");

    fireEvent.change(screen.getByRole("textbox"), {
      target: { value: "Think through this briefly." },
    });
    fireEvent.click(screen.getByRole("button", { name: "Send message" }));

    await waitFor(() =>
      expect(apiMocks.sendChatThreadMessageStream).toHaveBeenCalled(),
    );

    expect(apiMocks.sendChatThreadMessageStream).toHaveBeenCalledWith(
      "thread-3",
      expect.objectContaining({
        model_id: "Qwen3.5-2B",
        content: "Think through this briefly.",
        enable_thinking: true,
        system_prompt: "You are a helpful assistant.",
      }),
      expect.any(Object),
    );
  });

  it("sends the selected reasoning effort for Qwen3.8", async () => {
    const thread = {
      id: "thread-qwen38-effort",
      title: "Reasoning effort",
      model_id: "Qwen3.8-27B-FP8",
      created_at: 1,
      updated_at: 2,
      last_message_preview: null,
      message_count: 0,
    };

    apiMocks.listChatThreads.mockResolvedValue([thread]);
    apiMocks.getChatThread.mockResolvedValue({ thread, messages: [] });
    apiMocks.sendChatThreadMessageStream.mockReturnValue(new AbortController());

    render(
      <MemoryRouter initialEntries={["/chat?threadId=thread-qwen38-effort"]}>
        <ChatPlayground
          selectedModel="Qwen3.8-27B-FP8"
          selectedModelReady={true}
          supportsThinking={true}
          chatCapabilities={{
            supports_thinking: true,
            default_thinking_enabled: true,
            reasoning_efforts: ["xhigh", "medium", "low"],
            default_reasoning_effort: "xhigh",
            supports_preserve_thinking: true,
          }}
          modelLabel="Qwen3.8 27B (FP8)"
          modelOptions={[
            {
              value: "Qwen3.8-27B-FP8",
              label: "Qwen3.8 27B (FP8)",
              statusLabel: "Ready",
              isReady: true,
            },
          ]}
          onSelectModel={vi.fn()}
          onOpenModelManager={vi.fn()}
          onModelRequired={vi.fn()}
        />
      </MemoryRouter>,
    );

    await waitFor(() =>
      expect(apiMocks.getChatThread).toHaveBeenCalledWith(
        "thread-qwen38-effort",
      ),
    );

    const effortSelect = screen.getByRole("combobox", {
      name: "Reasoning effort",
    });
    expect(effortSelect).toHaveTextContent("XHigh");
    fireEvent.click(effortSelect);
    fireEvent.click(await screen.findByRole("option", { name: "Low" }));

    fireEvent.change(screen.getByRole("textbox"), {
      target: { value: "Answer efficiently." },
    });
    fireEvent.click(screen.getByRole("button", { name: "Send message" }));

    await waitFor(() =>
      expect(apiMocks.sendChatThreadMessageStream).toHaveBeenCalled(),
    );
    expect(apiMocks.sendChatThreadMessageStream).toHaveBeenCalledWith(
      "thread-qwen38-effort",
      expect.objectContaining({
        enable_thinking: true,
        reasoning_effort: "low",
      }),
      expect.any(Object),
    );
  });

  it("disables Qwen3.8 thinking without sending a reasoning effort", async () => {
    const thread = {
      id: "thread-qwen38-direct",
      title: "Direct answer",
      model_id: "Qwen3.8-27B-FP8",
      created_at: 1,
      updated_at: 2,
      last_message_preview: null,
      message_count: 0,
    };

    apiMocks.listChatThreads.mockResolvedValue([thread]);
    apiMocks.getChatThread.mockResolvedValue({ thread, messages: [] });
    apiMocks.sendChatThreadMessageStream.mockReturnValue(new AbortController());

    render(
      <MemoryRouter initialEntries={["/chat?threadId=thread-qwen38-direct"]}>
        <ChatPlayground
          selectedModel="Qwen3.8-27B-FP8"
          selectedModelReady={true}
          supportsThinking={true}
          chatCapabilities={{
            supports_thinking: true,
            default_thinking_enabled: true,
            reasoning_efforts: ["xhigh", "medium", "low"],
            default_reasoning_effort: "xhigh",
            supports_preserve_thinking: true,
          }}
          modelLabel="Qwen3.8 27B (FP8)"
          modelOptions={[
            {
              value: "Qwen3.8-27B-FP8",
              label: "Qwen3.8 27B (FP8)",
              statusLabel: "Ready",
              isReady: true,
            },
          ]}
          onSelectModel={vi.fn()}
          onOpenModelManager={vi.fn()}
          onModelRequired={vi.fn()}
        />
      </MemoryRouter>,
    );

    await waitFor(() =>
      expect(apiMocks.getChatThread).toHaveBeenCalledWith(
        "thread-qwen38-direct",
      ),
    );
    fireEvent.click(
      screen.getByRole("button", { name: "Disable thinking mode" }),
    );
    expect(
      screen.queryByRole("combobox", { name: "Reasoning effort" }),
    ).not.toBeInTheDocument();

    fireEvent.change(screen.getByRole("textbox"), {
      target: { value: "Give me the direct answer." },
    });
    fireEvent.click(screen.getByRole("button", { name: "Send message" }));

    await waitFor(() =>
      expect(apiMocks.sendChatThreadMessageStream).toHaveBeenCalled(),
    );
    const request = apiMocks.sendChatThreadMessageStream.mock.calls[0]?.[1];
    expect(request).toEqual(
      expect.objectContaining({ enable_thinking: false }),
    );
    expect(request).not.toHaveProperty("reasoning_effort", expect.anything());
  });

  it("forwards enable_thinking for Qwen3.6, whose template reads it", async () => {
    const thread = {
      id: "thread-qwen36",
      title: "Qwen3.6 direct answer",
      model_id: "Qwen3.6-35B-A3B-FP8",
      created_at: 1,
      updated_at: 2,
      last_message_preview: null,
      message_count: 0,
    };

    apiMocks.listChatThreads.mockResolvedValue([thread]);
    apiMocks.getChatThread.mockResolvedValue({ thread, messages: [] });
    apiMocks.sendChatThreadMessageStream.mockReturnValue(new AbortController());

    render(
      <MemoryRouter initialEntries={["/chat?threadId=thread-qwen36"]}>
        <ChatPlayground
          selectedModel="Qwen3.6-35B-A3B-FP8"
          selectedModelReady={true}
          supportsThinking={true}
          chatCapabilities={{
            supports_thinking: true,
            default_thinking_enabled: true,
            reasoning_efforts: [],
            default_reasoning_effort: null,
            supports_preserve_thinking: false,
          }}
          modelLabel="Qwen3.6 35B A3B (FP8)"
          modelOptions={[
            {
              value: "Qwen3.6-35B-A3B-FP8",
              label: "Qwen3.6 35B A3B (FP8)",
              statusLabel: "Ready",
              isReady: true,
            },
          ]}
          onSelectModel={vi.fn()}
          onOpenModelManager={vi.fn()}
          onModelRequired={vi.fn()}
        />
      </MemoryRouter>,
    );

    await waitFor(() =>
      expect(apiMocks.getChatThread).toHaveBeenCalledWith("thread-qwen36"),
    );
    fireEvent.click(
      screen.getByRole("button", { name: "Disable thinking mode" }),
    );
    fireEvent.change(screen.getByRole("textbox"), {
      target: { value: "Answer directly." },
    });
    fireEvent.click(screen.getByRole("button", { name: "Send message" }));

    await waitFor(() =>
      expect(apiMocks.sendChatThreadMessageStream).toHaveBeenCalled(),
    );
    const request = apiMocks.sendChatThreadMessageStream.mock.calls[0]?.[1];
    expect(request).toEqual(
      expect.objectContaining({
        enable_thinking: false,
        system_prompt: "You are a helpful assistant.",
      }),
    );
  });

  it("renders Qwen3.5 close-only think output as reasoning plus final answer", async () => {
    const thread = {
      id: "thread-4",
      title: "Parsed thinking thread",
      model_id: "Qwen3.5-4B",
      created_at: 1,
      updated_at: 2,
      last_message_preview: null,
      message_count: 2,
    };

    apiMocks.listChatThreads.mockResolvedValue([thread]);
    apiMocks.getChatThread.mockResolvedValue({
      thread,
      messages: [
        {
          id: "message-1",
          thread_id: "thread-4",
          role: "user",
          content: "Solve this",
          created_at: 1,
          tokens_generated: null,
          generation_time_ms: null,
        },
        {
          id: "message-2",
          thread_id: "thread-4",
          role: "assistant",
          content: "reasoning first</think>\nFinal answer",
          created_at: 2,
          tokens_generated: 8,
          generation_time_ms: 120,
        },
      ],
    });

    render(
      <MemoryRouter initialEntries={["/chat?threadId=thread-4"]}>
        <ChatPlayground
          selectedModel="Qwen3.5-4B"
          selectedModelReady={true}
          supportsThinking={true}
          modelLabel="Qwen3.5 4B GGUF (Q4_K_M)"
          modelOptions={[
            {
              value: "Qwen3.5-4B",
              label: "Qwen3.5 4B GGUF (Q4_K_M)",
              statusLabel: "Ready",
              isReady: true,
            },
          ]}
          onSelectModel={vi.fn()}
          onOpenModelManager={vi.fn()}
          onModelRequired={vi.fn()}
        />
      </MemoryRouter>,
    );

    await waitFor(() =>
      expect(apiMocks.getChatThread).toHaveBeenCalledWith("thread-4"),
    );

    expect(screen.getByText("Final answer")).toBeInTheDocument();
    // History has no measured duration, so the row is not timed.
    const reasoningToggle = screen.getByRole("button", {
      name: "Thought process",
    });
    expect(reasoningToggle).toHaveAttribute("aria-expanded", "false");
    expect(screen.queryByText("reasoning first")).not.toBeInTheDocument();

    fireEvent.click(reasoningToggle);

    expect(reasoningToggle).toHaveAttribute("aria-expanded", "true");
    expect(screen.getByText("reasoning first")).toBeInTheDocument();

    const writeText = vi.fn().mockResolvedValue(undefined);
    Object.defineProperty(navigator, "clipboard", {
      configurable: true,
      value: { writeText },
    });
    fireEvent.click(screen.getByRole("button", { name: "Copy reply" }));
    const reply = screen.getByRole("article", { name: "Reply" });
    expect(await within(reply).findByRole("status")).toHaveTextContent(
      "Copied",
    );
    expect(writeText).toHaveBeenCalledWith("Final answer");
    fireEvent.click(screen.getByRole("button", { name: "Copy message" }));
    expect(writeText).toHaveBeenLastCalledWith("Solve this");
  });
  it("shows a model that always thinks as a label, not a switch", async () => {
    const model = "LFM2.5-1.2B-Thinking-GGUF";
    const thread = { id: "always-thread", title: "Always thinks", model_id: model,
      created_at: 1, updated_at: 2, last_message_preview: null, message_count: 0 };
    apiMocks.listChatThreads.mockResolvedValue([thread]);
    apiMocks.getChatThread.mockResolvedValue({ thread, messages: [] });
    apiMocks.sendChatThreadMessageStream.mockReturnValue(new AbortController());
    render(<MemoryRouter initialEntries={["/chat?threadId=always-thread"]}>
      <ChatPlayground selectedModel={model} selectedModelReady={true}
        supportsThinking={true} modelLabel={model}
        chatCapabilities={{ supports_thinking: true, thinking_toggleable: false,
          default_thinking_enabled: true, reasoning_efforts: [],
          default_reasoning_effort: null, supports_preserve_thinking: false }}
        modelOptions={[{ value: model, label: model, statusLabel: "Ready", isReady: true }]}
        modelSupportsThinking={() => true}
        onSelectModel={vi.fn()} onOpenModelManager={vi.fn()} onModelRequired={vi.fn()} />
    </MemoryRouter>);
    await waitFor(() => expect(apiMocks.getChatThread).toHaveBeenCalled());

    expect(screen.getByTitle("This model always thinks before it answers")).toHaveTextContent(
      "Thinking",
    );
    expect(screen.queryByRole("button", { name: /thinking mode/ })).not.toBeInTheDocument();
    fireEvent.change(screen.getByRole("textbox"), { target: { value: "Plan it" } });
    fireEvent.click(screen.getByRole("button", { name: "Send message" }));
    await waitFor(() =>
      expect(apiMocks.sendChatThreadMessageStream).toHaveBeenCalledWith(
        thread.id,
        expect.objectContaining({ enable_thinking: true }),
        expect.any(Object),
      ),
    );
  });

  it("shows live reasoning while it streams, then the time it took", async () => {
    const model = "Qwen3.5-4B";
    const thread = { id: "live-thread", title: "Live reasoning", model_id: model,
      created_at: 1, updated_at: 2, last_message_preview: null, message_count: 0 };
    apiMocks.listChatThreads.mockResolvedValue([thread]);
    apiMocks.getChatThread.mockResolvedValue({ thread, messages: [] });
    apiMocks.sendChatThreadMessageStream.mockReturnValue(new AbortController());
    let now = 1_000_000;
    const dateNow = vi.spyOn(Date, "now").mockImplementation(() => now);
    render(<MemoryRouter initialEntries={["/chat?threadId=live-thread"]}>
      <ChatPlayground selectedModel={model} selectedModelReady={true}
        supportsThinking={true} modelLabel={model}
        chatCapabilities={{ supports_thinking: true, default_thinking_enabled: true,
          reasoning_efforts: [], default_reasoning_effort: null,
          supports_preserve_thinking: false }}
        modelOptions={[{ value: model, label: model, statusLabel: "Ready", isReady: true }]}
        modelSupportsThinking={() => true}
        onSelectModel={vi.fn()} onOpenModelManager={vi.fn()} onModelRequired={vi.fn()} />
    </MemoryRouter>);
    await waitFor(() => expect(apiMocks.getChatThread).toHaveBeenCalled());
    fireEvent.change(screen.getByRole("textbox"), { target: { value: "Plan it" } });
    fireEvent.click(screen.getByRole("button", { name: "Send message" }));
    await waitFor(() => expect(apiMocks.sendChatThreadMessageStream).toHaveBeenCalledOnce());
    const callbacks = apiMocks.sendChatThreadMessageStream.mock.calls[0][2];

    act(() => callbacks.onDelta("<think>Weighing the options"));
    expect(screen.getByRole("button", { name: /^Thinking/ })).toBeInTheDocument();
    expect(screen.getByText("Weighing the options")).toBeInTheDocument();

    now += 6_000;
    act(() => callbacks.onDelta("</think>Here is the plan."));
    expect(await screen.findByRole("button", { name: "Thought for 6s" })).toBeInTheDocument();
    expect(screen.getByText("Here is the plan.")).toBeInTheDocument();
    expect(screen.queryByText("Weighing the options")).not.toBeInTheDocument();
    dateNow.mockRestore();
  });

  it.each([
    ["LFM2.5-1.2B-Instruct-GGUF", false, " \n", "No visible answer was generated. Please retry the message."],
    ["LFM2.5-1.2B-Instruct-GGUF", false, "<think>unfinished reasoning", "No visible answer was generated. Please retry the message."],
    ["LFM2.5-1.2B-Thinking-GGUF", true, "<think>unfinished reasoning", "No final answer was generated."],
    ["LFM2.5-1.2B-Thinking-GGUF", true, "<think>reasoning only</think>", "No final answer was generated."],
  ])("shows completed no-answer feedback for %s (%s, %s)", async (model, thinking, content, expected) => {
    const thread = {
      id: "empty-thread", title: "Empty answer", model_id: model,
      created_at: 1, updated_at: 2, last_message_preview: null, message_count: 2,
    };
    apiMocks.listChatThreads.mockResolvedValue([thread]);
    apiMocks.getChatThread.mockResolvedValue({ thread, messages: [
      { id: "user-1", thread_id: thread.id, role: "user", content: "Hello", created_at: 1,
        tokens_generated: null, generation_time_ms: null },
      { id: "assistant-1", thread_id: thread.id, role: "assistant", content, created_at: 2,
        tokens_generated: 48, generation_time_ms: 224 },
    ] });
    render(<MemoryRouter initialEntries={["/chat?threadId=empty-thread"]}>
      <ChatPlayground selectedModel={model} selectedModelReady={true}
        supportsThinking={thinking} modelLabel={model}
        modelOptions={[{ value: model, label: model, statusLabel: "Ready", isReady: true }]}
        onSelectModel={vi.fn()} onOpenModelManager={vi.fn()} onModelRequired={vi.fn()} />
    </MemoryRouter>);
    expect(await screen.findByText(expected)).toBeInTheDocument();
    expect(screen.queryByText("Thinking...")).not.toBeInTheDocument();
  });

  it("parses past replies by the model that wrote them", async () => {
    const thread = {
      id: "mixed-thread", title: "Mixed models", model_id: "LFM2.5-1.2B-Instruct-GGUF",
      created_at: 1, updated_at: 3, last_message_preview: null, message_count: 2,
    };
    apiMocks.listChatThreads.mockResolvedValue([thread]);
    apiMocks.getChatThread.mockResolvedValue({ thread, messages: [
      { id: "u1", thread_id: thread.id, role: "user", content: "Question", created_at: 1,
        tokens_generated: null, generation_time_ms: null },
      { id: "a1", thread_id: thread.id, role: "assistant",
        content: "private planning</think>Visible answer", created_at: 2,
        tokens_generated: 12, generation_time_ms: 50, model_id: "Qwen3.5-4B",
        finish_reason: "stop" },
    ] });
    render(<MemoryRouter initialEntries={["/chat?threadId=mixed-thread"]}>
      <ChatPlayground selectedModel="LFM2.5-1.2B-Instruct-GGUF" selectedModelReady={true}
        supportsThinking={false} modelLabel="LFM2.5 1.2B Instruct"
        modelOptions={[{ value: "LFM2.5-1.2B-Instruct-GGUF", label: "LFM2.5 1.2B Instruct",
          statusLabel: "Ready", isReady: true }]}
        modelSupportsThinking={(variant) => variant.startsWith("Qwen3.5")}
        onSelectModel={vi.fn()} onOpenModelManager={vi.fn()} onModelRequired={vi.fn()} />
    </MemoryRouter>);

    expect(await screen.findByText("Visible answer")).toBeInTheDocument();
    expect(screen.queryByText(/private planning<\/think>/)).not.toBeInTheDocument();
  });

  describe("thread titles", () => {
    const model = "Qwen3.5-4B";
    const thread = { id: "title-thread", title: "New chat", model_id: model,
      created_at: 1, updated_at: 2, last_message_preview: null, message_count: 0 };

    const sendFirstMessage = async () => {
      apiMocks.listChatThreads.mockResolvedValue([thread]);
      apiMocks.getChatThread.mockResolvedValue({ thread, messages: [] });
      apiMocks.sendChatThreadMessageStream.mockReturnValue(new AbortController());
      render(<MemoryRouter initialEntries={["/chat?threadId=title-thread"]}>
        <ChatPlayground selectedModel={model} selectedModelReady={true}
          supportsThinking={true} modelLabel={model}
          modelOptions={[{ value: model, label: model, statusLabel: "Ready", isReady: true }]}
          onSelectModel={vi.fn()} onOpenModelManager={vi.fn()} onModelRequired={vi.fn()} />
      </MemoryRouter>);
      await waitFor(() => expect(apiMocks.getChatThread).toHaveBeenCalled());
      fireEvent.change(screen.getByRole("textbox"), { target: { value: "Houseplant tips?" } });
      fireEvent.click(screen.getByRole("button", { name: "Send message" }));
      await waitFor(() => expect(apiMocks.sendChatThreadMessageStream).toHaveBeenCalledOnce());
      return apiMocks.sendChatThreadMessageStream.mock.calls[0][2];
    };

    it("asks the server for a title once the first turn finishes", async () => {
      apiMocks.generateChatThreadTitle.mockResolvedValue({
        ...thread, title: "Houseplant care tips",
      });
      const callbacks = await sendFirstMessage();
      expect(apiMocks.generateChatThreadTitle).not.toHaveBeenCalled();
      const assistantMessage = { id: "a1", thread_id: thread.id, role: "assistant",
        content: "Reasoning about light</think>Water less in winter.",
        created_at: 3, tokens_generated: 9, generation_time_ms: 90 };
      apiMocks.getChatThread.mockResolvedValue({ thread, messages: [
        { id: "u1", thread_id: thread.id, role: "user", content: "Houseplant tips?",
          created_at: 2, tokens_generated: null, generation_time_ms: null },
        assistantMessage,
      ] });
      // A slow list refresh still holds the old title: it must not win.
      apiMocks.listChatThreads.mockImplementation(
        () => new Promise((resolve) => setTimeout(() => resolve([thread]), 30)),
      );
      act(() => {
        callbacks.onDone({
          threadId: thread.id,
          modelId: model,
          assistantMessage,
          stats: { tokens_generated: 9, generation_time_ms: 90 },
        });
        callbacks.onClose();
      });

      await waitFor(() =>
        expect(apiMocks.generateChatThreadTitle).toHaveBeenCalledWith(
          thread.id, { model_id: model },
        ),
      );
      expect(apiMocks.createResponse).not.toHaveBeenCalled();
      expect(apiMocks.updateChatThread).not.toHaveBeenCalled();
      expect(
        await screen.findByRole("heading", { name: "Houseplant care tips" }),
      ).toBeInTheDocument();
      // ...and stays once the slow list refresh has landed.
      await act(() => new Promise((resolve) => setTimeout(resolve, 60)));
      expect(
        screen.getByRole("heading", { name: "Houseplant care tips" }),
      ).toBeInTheDocument();
    });

    it("titles a thread whose first turn was stopped, and not one that failed", async () => {
      apiMocks.generateChatThreadTitle.mockResolvedValue(thread);
      const callbacks = await sendFirstMessage();
      act(() => {
        callbacks.onError("The model failed");
        callbacks.onClose();
      });
      await waitFor(() => expect(screen.getByRole("alert")).toBeInTheDocument());
      expect(apiMocks.generateChatThreadTitle).not.toHaveBeenCalled();

      apiMocks.sendChatThreadMessageStream.mockClear();
      fireEvent.change(screen.getByRole("textbox"), { target: { value: "Try again" } });
      fireEvent.click(screen.getByRole("button", { name: "Send message" }));
      await waitFor(() => expect(apiMocks.sendChatThreadMessageStream).toHaveBeenCalledOnce());
      const stopped = apiMocks.sendChatThreadMessageStream.mock.calls[0][2];
      fireEvent.click(screen.getByRole("button", { name: "Cancel response" }));
      act(() => stopped.onClose());
      await waitFor(() =>
        expect(apiMocks.generateChatThreadTitle).toHaveBeenCalledWith(
          thread.id, { model_id: null },
        ),
      );
    });
  });

  it("shows decode speed and TTFT for a streamed turn after history refreshes", async () => {
    const model = "Qwen3.5-4B";
    const thread = { id: "timed-thread", title: "Timed answer", model_id: model,
      created_at: 1, updated_at: 2, last_message_preview: null, message_count: 0 };
    const assistantMessage = { id: "assistant-timed", thread_id: thread.id, role: "assistant",
      content: "Final answer", created_at: 3, tokens_generated: 41, generation_time_ms: 2000 };
    apiMocks.listChatThreads.mockResolvedValue([thread]);
    apiMocks.getChatThread.mockResolvedValue({ thread, messages: [] });
    apiMocks.sendChatThreadMessageStream.mockReturnValue(new AbortController());
    render(<MemoryRouter initialEntries={["/chat?threadId=timed-thread"]}>
      <ChatPlayground selectedModel={model} selectedModelReady={true}
        supportsThinking={false} modelLabel={model}
        modelOptions={[{ value: model, label: model, statusLabel: "Ready", isReady: true }]}
        onSelectModel={vi.fn()} onOpenModelManager={vi.fn()} onModelRequired={vi.fn()} />
    </MemoryRouter>);
    await waitFor(() => expect(apiMocks.getChatThread).toHaveBeenCalled());
    fireEvent.change(screen.getByRole("textbox"), { target: { value: "Hello" } });
    fireEvent.click(screen.getByRole("button", { name: "Send message" }));
    await waitFor(() => expect(apiMocks.sendChatThreadMessageStream).toHaveBeenCalledOnce());
    const callbacks = apiMocks.sendChatThreadMessageStream.mock.calls[0][2];
    apiMocks.getChatThread.mockResolvedValue({ thread, messages: [
      { id: "user-timed", thread_id: thread.id, role: "user", content: "Hello", created_at: 2,
        tokens_generated: null, generation_time_ms: null },
      assistantMessage,
    ] });
    const historyReads = apiMocks.getChatThread.mock.calls.length;
    act(() => {
      callbacks.onDone({
        threadId: thread.id,
        modelId: model,
        assistantMessage,
        stats: {
          tokens_generated: 41,
          generation_time_ms: 2000,
          timing: { queue_wait_ms: 300, prefill_ms: 1200, decode_ms: 380,
            decode_wall_ms: 400, decode_tokens: 40, ttft_ms: 1550, total_ms: 2000 },
        },
      });
      callbacks.onClose();
    });
    await waitFor(() => expect(apiMocks.getChatThread.mock.calls.length).toBeGreaterThan(historyReads));
    const speed = await screen.findByText("100 tok/s");
    expect(speed).toHaveAttribute(
      "title",
      expect.stringContaining("First token after 1550 ms"),
    );
    expect(speed.getAttribute("title")).toMatch(/^Decode speed/);
    expect(
      screen.getAllByRole("status").some(
        (region) => region.textContent === "Reply: Final answer",
      ),
    ).toBe(true);
    expect(screen.getByText("41 tokens")).toBeInTheDocument();
    expect(screen.getByText(model)).toBeInTheDocument();
  });

  it.each([
    ["cancelled", "Stopped"],
    ["length", "This reply reached the length limit, so it may be cut off."],
    ["error", "The model hit an error before finishing this reply."],
  ])("says when a reply ended early (%s)", async (finishReason, expected) => {
    const model = "LFM2.5-1.2B-Instruct-GGUF";
    const thread = { id: "ended-thread", title: "Ended", model_id: model,
      created_at: 1, updated_at: 2, last_message_preview: null, message_count: 2 };
    apiMocks.listChatThreads.mockResolvedValue([thread]);
    apiMocks.getChatThread.mockResolvedValue({ thread, messages: [
      { id: "u1", thread_id: thread.id, role: "user", content: "Write a story",
        created_at: 1, tokens_generated: null, generation_time_ms: null },
      { id: "a1", thread_id: thread.id, role: "assistant", content: "Once upon",
        created_at: 2, tokens_generated: 3, generation_time_ms: 30, model_id: model,
        finish_reason: finishReason },
    ] });
    render(<MemoryRouter initialEntries={["/chat?threadId=ended-thread"]}>
      <ChatPlayground selectedModel={model} selectedModelReady={true}
        supportsThinking={false} modelLabel={model}
        modelOptions={[{ value: model, label: "LFM2.5 1.2B", statusLabel: "Ready", isReady: true }]}
        onSelectModel={vi.fn()} onOpenModelManager={vi.fn()} onModelRequired={vi.fn()} />
    </MemoryRouter>);

    const reply = await screen.findByRole("article", { name: "Reply" });
    expect(within(reply).getByText(expected)).toBeInTheDocument();
    expect(within(reply).getByText("LFM2.5 1.2B")).toBeInTheDocument();
  });

  describe("read aloud", () => {
    const model = "LFM2.5-1.2B-Instruct-GGUF";
    const thread = { id: "spoken-thread", title: "Spoken", model_id: model,
      created_at: 1, updated_at: 2, last_message_preview: null, message_count: 2 };

    const renderThread = (props: Partial<Parameters<typeof ChatPlayground>[0]>) => {
      apiMocks.listChatThreads.mockResolvedValue([thread]);
      apiMocks.getChatThread.mockResolvedValue({ thread, messages: [
        { id: "u1", thread_id: thread.id, role: "user", content: "Say hi",
          created_at: 1, tokens_generated: null, generation_time_ms: null },
        { id: "a1", thread_id: thread.id, role: "assistant", content: "**Hi** there",
          created_at: 2, tokens_generated: 2, generation_time_ms: 20 },
      ] });
      render(<MemoryRouter initialEntries={["/chat?threadId=spoken-thread"]}>
        <ChatPlayground selectedModel={model} selectedModelReady={true}
          supportsThinking={false} modelLabel={model}
          modelOptions={[{ value: model, label: model, statusLabel: "Ready", isReady: true }]}
          onSelectModel={vi.fn()} onOpenModelManager={vi.fn()} onModelRequired={vi.fn()}
          {...props} />
      </MemoryRouter>);
    };

    it("streams a reply through the loaded voice and stops on request", async () => {
      vi.stubGlobal("AudioContext", class {});
      const controller = new AbortController();
      apiMocks.generateTTSStream.mockReturnValue(controller);
      renderThread({ readAloudVoice: { modelId: "Kokoro-82M", speaker: "af_heart" } });

      fireEvent.click(await screen.findByRole("button", { name: "Read aloud" }));

      expect(apiMocks.generateTTSStream).toHaveBeenCalledWith(
        { text: "Hi there", model_id: "Kokoro-82M", speaker: "af_heart", format: "pcm" },
        expect.any(Object),
      );
      expect(
        screen.getByRole("button", { name: "Preparing audio, select to cancel" }),
      ).toBeInTheDocument();
      const [player] = playerMocks.instances;
      expect(player.unlock).toHaveBeenCalledOnce();

      const callbacks = apiMocks.generateTTSStream.mock.calls[0][1];
      act(() => callbacks.onStart({ requestId: "r1", sampleRate: 24000, audioFormat: "pcm_i16" }));
      act(() => callbacks.onChunk({ requestId: "r1", sequence: 0, audioBase64: "AAAA",
        sampleCount: 3, isFinal: false }));
      expect(await screen.findByRole("button", { name: "Stop reading" })).toBeInTheDocument();
      expect(player.start).toHaveBeenCalledWith("r1", 24000, "pcm_i16");
      expect(player.push).toHaveBeenCalledOnce();

      fireEvent.click(screen.getByRole("button", { name: "Stop reading" }));
      expect(player.stop).toHaveBeenCalled();
      expect(controller.signal.aborted).toBe(true);
      expect(screen.getByRole("button", { name: "Read aloud" })).toBeInTheDocument();
      vi.unstubAllGlobals();
    });

    it("returns to idle when playback drains after the stream finishes", async () => {
      vi.stubGlobal("AudioContext", class {});
      apiMocks.generateTTSStream.mockReturnValue(new AbortController());
      renderThread({ readAloudVoice: { modelId: "Kokoro-82M", speaker: "af_heart" } });

      fireEvent.click(await screen.findByRole("button", { name: "Read aloud" }));
      const callbacks = apiMocks.generateTTSStream.mock.calls[0][1];
      const [player] = playerMocks.instances;
      act(() => callbacks.onStart({ requestId: "r1", sampleRate: 24000, audioFormat: "pcm_i16" }));
      act(() => callbacks.onChunk({ requestId: "r1", sequence: 0, audioBase64: "AAAA",
        sampleCount: 3, isFinal: true }));
      act(() => callbacks.onFinal({}));
      await waitFor(() => expect(player.finish).toHaveBeenCalledOnce());
      act(() => player.onDrained());

      expect(screen.getByRole("button", { name: "Read aloud" })).toBeInTheDocument();
      vi.unstubAllGlobals();
    });

    it("opens the speech model manager when no voice is loaded", async () => {
      const onOpenSpeechModels = vi.fn();
      renderThread({ readAloudVoice: null, onOpenSpeechModels });

      fireEvent.click(await screen.findByRole("button", { name: "Read aloud" }));

      expect(onOpenSpeechModels).toHaveBeenCalledOnce();
      expect(apiMocks.generateTTSStream).not.toHaveBeenCalled();
    });
  });

  it("keeps a stopped turn on screen, marked Stopped, when the server did not store it", async () => {
    const model = "LFM2.5-1.2B-Instruct-GGUF";
    const thread = { id: "stop-thread", title: "Stop", model_id: model,
      created_at: 1, updated_at: 2, last_message_preview: null, message_count: 0 };
    apiMocks.listChatThreads.mockResolvedValue([thread]);
    apiMocks.getChatThread.mockResolvedValue({ thread, messages: [] });
    const controller = new AbortController();
    apiMocks.sendChatThreadMessageStream.mockReturnValue(controller);
    render(<MemoryRouter initialEntries={["/chat?threadId=stop-thread"]}>
      <ChatPlayground selectedModel={model} selectedModelReady={true}
        supportsThinking={false} modelLabel={model}
        modelOptions={[{ value: model, label: model, statusLabel: "Ready", isReady: true }]}
        onSelectModel={vi.fn()} onOpenModelManager={vi.fn()} onModelRequired={vi.fn()} />
    </MemoryRouter>);
    await waitFor(() => expect(apiMocks.getChatThread).toHaveBeenCalled());
    fireEvent.change(screen.getByRole("textbox"), { target: { value: "Tell me a long story" } });
    fireEvent.click(screen.getByRole("button", { name: "Send message" }));
    await waitFor(() => expect(apiMocks.sendChatThreadMessageStream).toHaveBeenCalledOnce());
    const callbacks = apiMocks.sendChatThreadMessageStream.mock.calls[0][2];
    act(() =>
      callbacks.onStart({ threadId: thread.id, modelId: model, userMessage: {
        id: "server-user", thread_id: thread.id, role: "user", content: "Tell me a long story",
        created_at: Date.now() + 50, tokens_generated: null, generation_time_ms: null } }),
    );
    act(() => callbacks.onDelta("Once upon a time"));

    const historyReads = apiMocks.getChatThread.mock.calls.length;
    fireEvent.click(screen.getByRole("button", { name: "Cancel response" }));
    act(() => callbacks.onClose());

    expect(controller.signal.aborted).toBe(true);
    await waitFor(() =>
      expect(apiMocks.getChatThread.mock.calls.length).toBeGreaterThan(historyReads),
    );
    expect(await screen.findByText("Stopped")).toBeInTheDocument();
    expect(screen.getByText("Tell me a long story")).toBeInTheDocument();
    expect(screen.getByText("Once upon a time")).toBeInTheDocument();
    expect(screen.queryByText(/No visible answer/)).not.toBeInTheDocument();
    // The prompt stays above its reply even when the server dated it later.
    const prompt = screen.getByText("Tell me a long story");
    const chip = screen.getByText("Stopped");
    expect(
      prompt.compareDocumentPosition(chip) & Node.DOCUMENT_POSITION_FOLLOWING,
    ).not.toBe(0);
  });

  describe("retry, edit and delete", () => {
    const model = "Qwen3.5-4B";
    const otherModel = "LFM2.5-1.2B-Instruct-GGUF";
    const thread = { id: "turns-thread", title: "Turns", model_id: model,
      created_at: 1, updated_at: 5, last_message_preview: null, message_count: 4 };
    const history = [
      { id: "u1", thread_id: thread.id, role: "user", content: "First question",
        created_at: 1, tokens_generated: null, generation_time_ms: null },
      { id: "a1", thread_id: thread.id, role: "assistant", content: "First answer",
        created_at: 2, tokens_generated: 2, generation_time_ms: 20, model_id: model },
      { id: "u2", thread_id: thread.id, role: "user", content: "Second question",
        created_at: 3, tokens_generated: null, generation_time_ms: null },
      { id: "a2", thread_id: thread.id, role: "assistant", content: "Second answer",
        created_at: 4, tokens_generated: 2, generation_time_ms: 20, model_id: model },
    ];
    const onSelectModel = vi.fn();

    const renderTurns = async () => {
      onSelectModel.mockReset();
      apiMocks.listChatThreads.mockResolvedValue([thread]);
      apiMocks.getChatThread.mockResolvedValue({ thread, messages: history });
      render(<MemoryRouter initialEntries={["/chat?threadId=turns-thread"]}>
        <ChatPlayground selectedModel={model} selectedModelReady={true}
          supportsThinking={false} modelLabel={model}
          modelOptions={[
            { value: model, label: "Qwen3.5 4B", statusLabel: "Ready", isReady: true },
            { value: otherModel, label: "LFM2.5 1.2B", statusLabel: "Ready", isReady: true },
            { value: "Qwen3.5-2B", label: "Qwen3.5 2B", statusLabel: "Not loaded", isReady: false },
          ]}
          onSelectModel={onSelectModel} onOpenModelManager={vi.fn()} onModelRequired={vi.fn()} />
      </MemoryRouter>);
      await screen.findByText("Second answer");
    };

    const openMoreMenu = async (replyText: string) => {
      const reply = screen.getByText(replyText).closest("article") as HTMLElement;
      const trigger = within(reply).getByRole("button", { name: "More reply actions" });
      fireEvent.pointerDown(trigger);
      fireEvent.click(trigger);
      return screen.findByRole("menu");
    };

    it("retries only the latest reply, streaming a new one in its place", async () => {
      apiMocks.regenerateChatThreadMessageStream.mockReturnValue(new AbortController());
      await renderTurns();

      const firstReply = screen.getByText("First answer").closest("article") as HTMLElement;
      expect(within(firstReply).queryByRole("button", { name: "Retry" })).not.toBeInTheDocument();
      fireEvent.click(screen.getByRole("button", { name: "Retry" }));

      expect(apiMocks.regenerateChatThreadMessageStream).toHaveBeenCalledWith(
        thread.id, "a2", expect.objectContaining({ model_id: model, truncation: "auto" }),
        expect.any(Object),
      );
      expect(screen.queryByText("Second answer")).not.toBeInTheDocument();
      expect(screen.getByText("Second question")).toBeInTheDocument();
      const callbacks = apiMocks.regenerateChatThreadMessageStream.mock.calls[0][3];
      act(() => callbacks.onDelta("A better answer"));
      expect(screen.getByText("A better answer")).toBeInTheDocument();
    });

    it("retries with another loaded model and keeps it selected", async () => {
      apiMocks.regenerateChatThreadMessageStream.mockReturnValue(new AbortController());
      await renderTurns();

      await openMoreMenu("Second answer");
      const retryWith = screen.getByRole("menuitem", { name: "Retry with…" });
      fireEvent.keyDown(retryWith, { key: "ArrowRight" });
      const menus = await screen.findAllByRole("menu");
      const submenu = menus[menus.length - 1];
      expect(within(submenu).queryByText("Qwen3.5 2B")).not.toBeInTheDocument();
      fireEvent.click(within(submenu).getByRole("menuitem", { name: /LFM2.5 1.2B/ }));

      expect(onSelectModel).toHaveBeenCalledWith(otherModel);
      expect(apiMocks.regenerateChatThreadMessageStream).toHaveBeenCalledWith(
        thread.id, "a2", expect.objectContaining({ model_id: otherModel }), expect.any(Object),
      );
    });

    it("reloads history when a retry fails, since the server kept the old reply", async () => {
      apiMocks.regenerateChatThreadMessageStream.mockReturnValue(new AbortController());
      await renderTurns();
      fireEvent.click(screen.getByRole("button", { name: "Retry" }));
      const callbacks = apiMocks.regenerateChatThreadMessageStream.mock.calls[0][3];
      act(() => {
        callbacks.onError("Model crashed");
        callbacks.onClose();
      });

      expect(await screen.findByText("Second answer")).toBeInTheDocument();
      expect(screen.getByText("Model crashed")).toBeInTheDocument();
    });

    it("edits a message in place and replaces the replies after it", async () => {
      apiMocks.editChatThreadMessageStream.mockReturnValue(new AbortController());
      await renderTurns();

      const firstPrompt = screen.getByText("First question").closest("article") as HTMLElement;
      fireEvent.click(within(firstPrompt).getByRole("button", { name: "Edit message" }));
      const editor = screen.getByRole("textbox", { name: "Edit your message" });
      expect(editor).toHaveValue("First question");
      expect(screen.getByText("Sending replaces the replies below this message.")).toBeInTheDocument();
      fireEvent.change(editor, { target: { value: "First question, reworded" } });
      fireEvent.keyDown(editor, { key: "Enter" });

      expect(apiMocks.editChatThreadMessageStream).toHaveBeenCalledWith(
        thread.id, "u1",
        expect.objectContaining({ content: "First question, reworded", model_id: model }),
        expect.any(Object),
      );
      expect(screen.getByText("First question, reworded")).toBeInTheDocument();
      for (const gone of ["First answer", "Second question", "Second answer"]) {
        expect(screen.queryByText(gone)).not.toBeInTheDocument();
      }
    });

    it("opens your last message for editing with up arrow, and Escape cancels", async () => {
      await renderTurns();
      const composer = screen.getByRole("textbox");
      fireEvent.keyDown(composer, { key: "ArrowUp" });

      const editor = screen.getByRole("textbox", { name: "Edit your message" });
      expect(editor).toHaveValue("Second question");
      fireEvent.keyDown(editor, { key: "Escape" });
      expect(screen.queryByRole("textbox", { name: "Edit your message" })).not.toBeInTheDocument();
      expect(apiMocks.editChatThreadMessageStream).not.toHaveBeenCalled();
    });

    it("deletes a reply after you confirm", async () => {
      apiMocks.deleteChatThreadMessage.mockResolvedValue({ id: "a1", deleted: true });
      await renderTurns();

      await openMoreMenu("First answer");
      fireEvent.click(screen.getByRole("menuitem", { name: "Delete reply" }));
      const dialog = await screen.findByRole("dialog");
      expect(within(dialog).getByText("Delete reply?")).toBeInTheDocument();
      fireEvent.click(within(dialog).getByRole("button", { name: /Delete/ }));

      await waitFor(() =>
        expect(apiMocks.deleteChatThreadMessage).toHaveBeenCalledWith(thread.id, "a1"),
      );
      await waitFor(() => expect(screen.queryByText("First answer")).not.toBeInTheDocument());
      expect(screen.getByText("Second answer")).toBeInTheDocument();
    });

    it("shows the stored reasoning time of a reply from history", async () => {
      apiMocks.listChatThreads.mockResolvedValue([thread]);
      apiMocks.getChatThread.mockResolvedValue({ thread, messages: [
        history[0],
        { ...history[1], content: "Weighing it</think>First answer", reasoning_ms: 6_200 },
      ] });
      render(<MemoryRouter initialEntries={["/chat?threadId=turns-thread"]}>
        <ChatPlayground selectedModel={model} selectedModelReady={true}
          supportsThinking={true} modelLabel={model}
          modelOptions={[{ value: model, label: "Qwen3.5 4B", statusLabel: "Ready", isReady: true }]}
          modelSupportsThinking={() => true}
          onSelectModel={vi.fn()} onOpenModelManager={vi.fn()} onModelRequired={vi.fn()} />
      </MemoryRouter>);

      expect(await screen.findByRole("button", { name: "Thought for 6s" })).toBeInTheDocument();
    });
  });

  it("keeps failed streamed text visible when history refreshes", async () => {
    const model = "LFM2.5-1.2B-Instruct-GGUF";
    const thread = { id: "failed-thread", title: "Failed answer", model_id: model,
      created_at: 1, updated_at: 2, last_message_preview: null, message_count: 0 };
    apiMocks.listChatThreads.mockResolvedValue([thread]);
    apiMocks.getChatThread.mockResolvedValue({ thread, messages: [] });
    apiMocks.sendChatThreadMessageStream.mockReturnValue(new AbortController());
    render(<MemoryRouter initialEntries={["/chat?threadId=failed-thread"]}>
      <ChatPlayground selectedModel={model} selectedModelReady={true}
        supportsThinking={false} modelLabel={model}
        modelOptions={[{ value: model, label: model, statusLabel: "Ready", isReady: true }]}
        onSelectModel={vi.fn()} onOpenModelManager={vi.fn()} onModelRequired={vi.fn()} />
    </MemoryRouter>);
    await waitFor(() => expect(apiMocks.getChatThread).toHaveBeenCalled());
    fireEvent.change(screen.getByRole("textbox"), { target: { value: "Hello" } });
    fireEvent.click(screen.getByRole("button", { name: "Send message" }));
    await waitFor(() => expect(apiMocks.sendChatThreadMessageStream).toHaveBeenCalledOnce());
    const callbacks = apiMocks.sendChatThreadMessageStream.mock.calls[0][2];
    const historyReads = apiMocks.getChatThread.mock.calls.length;
    act(() => {
      callbacks.onDelta("Meaningful partial answer");
      callbacks.onError("Empty final response");
      callbacks.onClose();
    });
    expect(await screen.findByText("Meaningful partial answer")).toBeInTheDocument();
    expect(screen.getByText("Empty final response")).toBeInTheDocument();
    await waitFor(() => expect(apiMocks.listChatThreads.mock.calls.length).toBeGreaterThan(1));
    expect(apiMocks.getChatThread).toHaveBeenCalledTimes(historyReads);
    expect(screen.getByText("Meaningful partial answer")).toBeInTheDocument();
  });

  describe("streaming turns", () => {
    const model = "LFM2.5-1.2B-Instruct-GGUF";
    const thinkingModel = "Qwen3.5-4B";
    const thread = { id: "stream-thread", title: "Stories", model_id: model,
      created_at: 1, updated_at: 2, last_message_preview: null, message_count: 2 };
    const other = { ...thread, id: "other-thread", title: "Other" };
    const message = (id: string, role: "user" | "assistant", content: string,
      createdAt: number, extra: Record<string, unknown> = {}) => ({
      id, thread_id: thread.id, role, content, created_at: createdAt,
      tokens_generated: null, generation_time_ms: null, ...extra });
    const history = [
      message("u1", "user", "Hi", 1),
      message("a1", "assistant", "Hello there", 2, { model_id: model }),
    ];
    let server: Record<string, ReturnType<typeof message>[]>;
    const router: { navigate?: NavigateFunction } = {};
    const Navigator = () => {
      const navigate = useNavigate();
      useEffect(() => {
        router.navigate = navigate;
      }, [navigate]);
      return null;
    };

    const renderChat = (path: string, props: Record<string, unknown> = {}) => {
      apiMocks.listChatThreads.mockResolvedValue([thread, other]);
      apiMocks.getChatThread.mockImplementation(async (id: string) => ({
        thread: id === other.id ? other : thread,
        messages: server[id] ?? [],
      }));
      render(<MemoryRouter initialEntries={[path]}>
        <Navigator />
        <ChatPlayground selectedModel={model} selectedModelReady={true}
          supportsThinking={false} modelLabel="LFM2.5 1.2B"
          modelOptions={[
            { value: model, label: "LFM2.5 1.2B", statusLabel: "Ready", isReady: true },
            { value: thinkingModel, label: "Qwen3.5 4B", statusLabel: "Ready", isReady: true },
          ]}
          onSelectModel={vi.fn()} onOpenModelManager={vi.fn()} onModelRequired={vi.fn()}
          {...props} />
      </MemoryRouter>);
    };

    const send = async (text: string) => {
      fireEvent.change(screen.getByRole("textbox"), { target: { value: text } });
      fireEvent.click(screen.getByRole("button", { name: "Send message" }));
      await waitFor(() => expect(apiMocks.sendChatThreadMessageStream).toHaveBeenCalled());
      const calls = apiMocks.sendChatThreadMessageStream.mock.calls;
      return calls[calls.length - 1][2];
    };

    beforeEach(() => {
      server = { [thread.id]: history, [other.id]: [
        { ...message("o1", "user", "Elsewhere", 1), thread_id: other.id },
        { ...message("o2", "assistant", "Other answer", 2), thread_id: other.id },
      ] };
    });

    it("offers Retry on a stopped reply once the server has it, and a retry stays", async () => {
      const controller = new AbortController();
      apiMocks.sendChatThreadMessageStream.mockReturnValue(controller);
      apiMocks.regenerateChatThreadMessageStream.mockReturnValue(new AbortController());
      renderChat("/chat?threadId=stream-thread");
      await screen.findByText("Hello there");

      const callbacks = await send("Tell me a story");
      act(() => {
        callbacks.onStart({ threadId: thread.id, modelId: model,
          userMessage: message("u2", "user", "Tell me a story", 3) });
        callbacks.onDelta("Once upon");
      });
      fireEvent.click(screen.getByRole("button", { name: "Cancel response" }));
      act(() => callbacks.onClose());

      // Not stored yet: the local copy has nothing to retry on the server.
      expect(await screen.findByText("Stopped")).toBeInTheDocument();
      expect(controller.signal.aborted).toBe(true);
      expect(screen.queryByRole("button", { name: "Retry" })).not.toBeInTheDocument();

      server[thread.id] = [...history, message("u2", "user", "Tell me a story", 3),
        message("a2", "assistant", "Once upon", 4, { model_id: model, finish_reason: "cancelled" })];
      fireEvent.click(await screen.findByRole("button", { name: "Retry" }));
      expect(apiMocks.regenerateChatThreadMessageStream).toHaveBeenCalledWith(
        thread.id, "a2", expect.any(Object), expect.any(Object));

      const retried = message("a3", "assistant", "A whole story", 5, { model_id: model });
      server[thread.id] = [...history, message("u2", "user", "Tell me a story", 3), retried];
      const retry = apiMocks.regenerateChatThreadMessageStream.mock.calls[0][3];
      act(() => {
        retry.onDone({ threadId: thread.id, modelId: model, assistantMessage: retried,
          stats: { tokens_generated: 3, generation_time_ms: 30 } });
        retry.onClose();
      });
      await waitFor(() => expect(apiMocks.getChatThread.mock.calls.length).toBeGreaterThan(3));
      expect(await screen.findByText("A whole story")).toBeInTheDocument();
      await act(() => new Promise((resolve) => setTimeout(resolve, 30)));
      expect(screen.queryByText("Once upon")).not.toBeInTheDocument();
      expect(screen.queryByText("Stopped")).not.toBeInTheDocument();
    });

    it("keeps a reply stopped mid-thought as reasoning, not as the answer", async () => {
      apiMocks.sendChatThreadMessageStream.mockReturnValue(new AbortController());
      renderChat("/chat?threadId=stream-thread", {
        selectedModel: thinkingModel, supportsThinking: true,
        chatCapabilities: { supports_thinking: true, default_thinking_enabled: true,
          thinking_toggleable: true, reasoning_efforts: [], default_reasoning_effort: null,
          supports_preserve_thinking: false },
        modelSupportsThinking: (variant: string) => variant === thinkingModel,
      });
      await screen.findByText("Hello there");

      const callbacks = await send("Why is the sky blue?");
      act(() => callbacks.onDelta("Weighing Rayleigh scattering"));
      fireEvent.click(screen.getByRole("button", { name: "Cancel response" }));

      expect(await screen.findByText("Stopped")).toBeInTheDocument();
      expect(screen.queryByText("Weighing Rayleigh scattering")).not.toBeInTheDocument();
      expect(screen.getByRole("button", { name: /^Thought/ })).toBeInTheDocument();
    });

    it("stops a reply when you go back to another thread, and keeps it out of that thread", async () => {
      const controller = new AbortController();
      apiMocks.sendChatThreadMessageStream.mockReturnValue(controller);
      renderChat("/chat?threadId=stream-thread");
      await screen.findByText("Hello there");
      const callbacks = await send("Tell me a story");
      act(() => callbacks.onDelta("Once upon"));
      // The model a reply streams from does not change under it.
      expect(screen.getByRole("combobox", { name: "Chat model" })).toBeDisabled();

      act(() => void router.navigate?.("/chat?threadId=other-thread"));
      expect(await screen.findByText("Other answer")).toBeInTheDocument();
      expect(controller.signal.aborted).toBe(true);
      act(() => {
        callbacks.onDelta(" a time");
        callbacks.onClose();
      });
      expect(screen.queryByText(/Once upon/)).not.toBeInTheDocument();
      expect(screen.getByRole("button", { name: "Send message" })).toBeInTheDocument();
    });

    it("retries with another model using that model's thinking settings", async () => {
      apiMocks.regenerateChatThreadMessageStream.mockReturnValue(new AbortController());
      renderChat("/chat?threadId=stream-thread", {
        modelChatCapabilities: (variant: string) => variant === thinkingModel
          ? { supports_thinking: true, default_thinking_enabled: true, thinking_toggleable: true,
              reasoning_efforts: [], default_reasoning_effort: null, supports_preserve_thinking: false }
          : null,
      });
      const reply = (await screen.findByText("Hello there")).closest("article") as HTMLElement;
      const trigger = within(reply).getByRole("button", { name: "More reply actions" });
      fireEvent.pointerDown(trigger);
      fireEvent.click(trigger);
      const retryWith = await screen.findByRole("menuitem", { name: "Retry with…" });
      fireEvent.keyDown(retryWith, { key: "ArrowRight" });
      const menus = await screen.findAllByRole("menu");
      fireEvent.click(within(menus[menus.length - 1]).getByRole("menuitem", { name: /Qwen3.5 4B/ }));

      expect(apiMocks.regenerateChatThreadMessageStream).toHaveBeenCalledWith(
        thread.id, "a1",
        expect.objectContaining({ model_id: thinkingModel, enable_thinking: true }),
        expect.any(Object),
      );
    });

    it("does not send on the Enter that ends an IME composition", async () => {
      apiMocks.sendChatThreadMessageStream.mockReturnValue(new AbortController());
      renderChat("/chat?threadId=stream-thread");
      await screen.findByText("Hello there");
      const composer = screen.getByRole("textbox");
      fireEvent.change(composer, { target: { value: "こんにちは" } });

      fireEvent.keyDown(composer, { key: "Enter", isComposing: true });
      expect(apiMocks.sendChatThreadMessageStream).not.toHaveBeenCalled();
      fireEvent.keyDown(composer, { key: "Enter" });
      expect(apiMocks.sendChatThreadMessageStream).toHaveBeenCalledOnce();
    });

    it("gives a failed first message back and leaves no empty thread", async () => {
      const created = { ...thread, id: "created-thread", title: "New chat", message_count: 0 };
      apiMocks.createChatThread.mockResolvedValue(created);
      apiMocks.deleteChatThread.mockResolvedValue(undefined);
      apiMocks.sendChatThreadMessageStream.mockReturnValue(new AbortController());
      renderChat("/chat");
      await waitFor(() => expect(apiMocks.listChatThreads).toHaveBeenCalled());

      const callbacks = await send("Describe this");
      act(() => {
        callbacks.onError("Model crashed");
        callbacks.onClose();
      });

      await waitFor(() => expect(apiMocks.deleteChatThread).toHaveBeenCalledWith(created.id));
      expect(await screen.findByText("Model crashed")).toBeInTheDocument();
      await waitFor(() => expect(screen.getByRole("textbox")).toHaveValue("Describe this"));
      expect(screen.queryByRole("heading", { name: "New chat" })).not.toBeInTheDocument();
    });

    it("gives a failed message back to the composer", async () => {
      apiMocks.sendChatThreadMessageStream.mockReturnValue(new AbortController());
      renderChat("/chat?threadId=stream-thread");
      await screen.findByText("Hello there");
      const callbacks = await send("One more thing");
      expect(screen.getByRole("textbox")).toHaveValue("");
      act(() => {
        callbacks.onError("Model crashed");
        callbacks.onClose();
      });
      await waitFor(() => expect(screen.getByRole("textbox")).toHaveValue("One more thing"));
      expect(apiMocks.deleteChatThread).not.toHaveBeenCalled();
    });

    it("attaches only image formats the server reads", async () => {
      renderChat("/chat?threadId=stream-thread", {
        modelSupportsImages: () => true,
      });
      await screen.findByText("Hello there");
      const input = screen.getByTestId("chat-image-input");
      expect(input).toHaveAttribute("accept", "image/png,image/jpeg,image/gif,image/webp,image/bmp");

      fireEvent.change(input, { target: { files: [
        new File(["<svg/>"], "logo.svg", { type: "image/svg+xml" }),
      ] } });
      expect(await screen.findByText(/Attach PNG, JPEG, GIF, WebP or BMP images, not logo.svg/))
        .toBeInTheDocument();
    });
  });

  describe("audio attachments", () => {
    const audioModel = "LFM3-Audio-2B-GGUF";
    const thread = { id: "audio-thread", title: "Clips", model_id: audioModel,
      created_at: 1, updated_at: 2, last_message_preview: null, message_count: 0 };
    // What prepareReferenceClip hands back: "RIFF" is UklGRg== in base64.
    const wav = new File(["RIFF"], "clip.wav", { type: "audio/wav" });

    const chat = (model: string, label: string, props: Record<string, unknown> = {}) => (
      <MemoryRouter initialEntries={["/chat?threadId=audio-thread"]}>
        <ChatPlayground selectedModel={model} selectedModelReady={true}
          supportsThinking={false} chatCapabilities={null} modelLabel={label}
          modelOptions={[{ value: model, label, statusLabel: "Ready", isReady: true }]}
          onSelectModel={vi.fn()} onOpenModelManager={vi.fn()} onModelRequired={vi.fn()}
          {...props} />
      </MemoryRouter>
    );

    const renderChat = async (props: Record<string, unknown> = {}) => {
      const view = render(chat(audioModel, "LFM3 Audio 2B", props));
      await waitFor(() => expect(apiMocks.getChatThread).toHaveBeenCalledWith(thread.id));
      return view;
    };

    const attach = (...names: string[]) =>
      fireEvent.change(screen.getByTestId("chat-audio-input"), { target: {
        files: names.map((name) => new File(["audio"], name, { type: "audio/mpeg" })),
      } });

    beforeEach(() => {
      apiMocks.listChatThreads.mockResolvedValue([thread]);
      apiMocks.getChatThread.mockResolvedValue({ thread, messages: [] });
      apiMocks.sendChatThreadMessageStream.mockReturnValue(new AbortController());
      clipMocks.prepareReferenceClip.mockReset();
      clipMocks.prepareReferenceClip.mockResolvedValue({ blob: wav, durationSecs: 4.2, peaks: [] });
    });

    afterEach(() => {
      vi.unstubAllGlobals();
      Reflect.deleteProperty(navigator, "mediaDevices");
    });

    it("sends LFM3-Audio clips as 16 kHz WAV input_audio parts, without thinking controls", async () => {
      await renderChat();
      expect(screen.getByRole("textbox")).toHaveAttribute("placeholder", "Ask anything or attach audio...");
      expect(screen.queryByRole("button", { name: "Attach image" })).not.toBeInTheDocument();
      expect(screen.getByRole("button", { name: "Record audio" })).toBeInTheDocument();
      expect(screen.getByRole("button", { name: "Attach audio" })).toBeInTheDocument();
      expect(screen.getByTestId("chat-audio-input")).toHaveAttribute("accept", AUDIO_FILE_ACCEPT);

      attach("question.mp3");
      expect(await screen.findByRole("button", { name: "Remove question.mp3" })).toBeInTheDocument();
      expect(clipMocks.prepareReferenceClip).toHaveBeenCalledWith(
        expect.objectContaining({ name: "question.mp3" }),
        { sampleRate: 16_000, maxSecs: undefined },
      );
      expect(screen.getByText("0:04")).toBeInTheDocument();
      expect(screen.getByLabelText("Play question.mp3")).toHaveAttribute("src", "blob:preview-clip.wav");

      fireEvent.change(screen.getByRole("textbox"), { target: { value: "What is said?" } });
      fireEvent.click(screen.getByRole("button", { name: "Send message" }));
      await waitFor(() => expect(apiMocks.sendChatThreadMessageStream).toHaveBeenCalled());

      const request = apiMocks.sendChatThreadMessageStream.mock.calls[0][1];
      expect(request).toEqual(expect.objectContaining({
        model_id: audioModel,
        content: "What is said?",
        content_parts: [
          { type: "text", text: "What is said?" },
          { type: "input_audio", input_audio: { data: "UklGRg==", format: "wav", name: "question.mp3" } },
        ],
      }));
      // LFM3-Audio refuses both with a 400; undefined fields are not sent.
      const body = JSON.parse(JSON.stringify(request));
      expect(body).not.toHaveProperty("enable_thinking");
      expect(body).not.toHaveProperty("reasoning_effort");
      expect(revokeObjectUrlMock).toHaveBeenCalledWith("blob:preview-clip.wav");

      // The sent message plays its clip.
      const sent = screen.getByRole("article", { name: "Your message" });
      expect(within(sent).getByLabelText("Play question.mp3"))
        .toHaveAttribute("src", "data:audio/wav;base64,UklGRg==");
    });

    it("refuses a file over two minutes and more than eight clips", async () => {
      await renderChat();
      clipMocks.prepareReferenceClip.mockResolvedValueOnce({ blob: wav, durationSecs: 125.4, peaks: [] });
      attach("lecture.mp3");
      expect(await screen.findByText("lecture.mp3 is 2:06 long. Attach clips of up to 2:00."))
        .toBeInTheDocument();
      expect(screen.queryByRole("button", { name: "Remove lecture.mp3" })).not.toBeInTheDocument();

      clipMocks.prepareReferenceClip.mockClear();
      attach(...Array.from({ length: 9 }, (_, index) => `clip-${index}.mp3`));
      expect(await screen.findByText("Attach up to 8 audio clips per message.")).toBeInTheDocument();
      expect(clipMocks.prepareReferenceClip).not.toHaveBeenCalled();
    });

    it("keeps clips on a model that can't hear them and blocks sending them", async () => {
      const { rerender } = await renderChat();
      attach("memo.mp3");
      expect(await screen.findByRole("button", { name: "Remove memo.mp3" })).toBeInTheDocument();

      rerender(chat("LFM2.5-1.2B-Instruct-GGUF", "LFM2.5 1.2B Instruct"));
      expect(screen.queryByRole("button", { name: "Attach audio" })).not.toBeInTheDocument();
      expect(screen.queryByRole("button", { name: "Record audio" })).not.toBeInTheDocument();
      expect(screen.getByRole("textbox")).toHaveAttribute("placeholder", "Ask anything...");
      expect(screen.getByText(/LFM2\.5 1\.2B Instruct can't hear audio/)).toBeInTheDocument();
      fireEvent.change(screen.getByRole("textbox"), { target: { value: "Summarize it" } });
      expect(screen.getByRole("button", { name: "Send message" })).toBeDisabled();
      fireEvent.keyDown(screen.getByRole("textbox"), { key: "Enter" });
      expect(apiMocks.sendChatThreadMessageStream).not.toHaveBeenCalled();

      // Switching back makes the same clip sendable again.
      rerender(chat(audioModel, "LFM3 Audio 2B"));
      expect(screen.queryByText(/can't hear audio/)).not.toBeInTheDocument();
      expect(screen.getByRole("button", { name: "Send message" })).toBeEnabled();
    });

    it("follows the catalog audio capability over the model name", async () => {
      await renderChat({ modelSupportsAudio: () => false });
      expect(screen.queryByRole("button", { name: "Attach audio" })).not.toBeInTheDocument();
      expect(screen.queryByRole("button", { name: "Record audio" })).not.toBeInTheDocument();
    });

    it("records until the two-minute limit and attaches the recording as WAV", async () => {
      const track = { stop: vi.fn() };
      const getUserMedia = vi.fn().mockResolvedValue({ getTracks: () => [track] });
      Object.defineProperty(navigator, "mediaDevices", {
        configurable: true,
        value: { getUserMedia },
      });
      vi.stubGlobal("MediaRecorder", class {
        static isTypeSupported = (type: string) => type === "audio/webm;codecs=opus";
        state = "inactive";
        mimeType: string;
        ondataavailable: ((event: { data: Blob }) => void) | null = null;
        onstop: (() => void) | null = null;
        constructor(_stream: unknown, options?: { mimeType?: string }) {
          this.mimeType = options?.mimeType ?? "";
        }
        start() {
          this.state = "recording";
        }
        stop() {
          this.state = "inactive";
          this.ondataavailable?.({ data: new Blob(["opus"], { type: this.mimeType }) });
          this.onstop?.();
        }
      });
      const setTimeoutSpy = vi.spyOn(window, "setTimeout");
      await renderChat();

      fireEvent.click(screen.getByRole("button", { name: "Record audio" }));
      const stop = await screen.findByRole("button", { name: "Stop recording" });
      expect(getUserMedia).toHaveBeenCalledWith({ audio: true });
      expect(stop).toHaveTextContent("0:00 / 2:00");
      expect(screen.getByRole("button", { name: "Send message" })).toBeDisabled();

      const limit = setTimeoutSpy.mock.calls.find(([, delay]) => delay === 120_000);
      setTimeoutSpy.mockRestore();
      expect(limit).toBeDefined();
      act(() => (limit![0] as () => void)());

      expect(await screen.findByRole("button", { name: "Remove Microphone recording" }))
        .toBeInTheDocument();
      expect(track.stop).toHaveBeenCalled();
      const [recorded, options] = clipMocks.prepareReferenceClip.mock.calls[0];
      // Browsers record WebM/Opus, which the server does not decode.
      expect(recorded.type).toBe("audio/webm;codecs=opus");
      expect(options).toEqual({ sampleRate: 16_000, maxSecs: 120 });
      expect(screen.getByRole("button", { name: "Record audio" })).toBeInTheDocument();
    });
  });
});
