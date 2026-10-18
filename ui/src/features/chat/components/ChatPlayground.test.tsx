import {
  act,
  fireEvent,
  render,
  screen,
  waitFor,
  within,
} from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { MemoryRouter } from "react-router-dom";
import { ChatPlayground } from "@/features/chat/components/ChatPlayground";

const apiMocks = vi.hoisted(() => ({
  listChatThreads: vi.fn(),
  createResponse: vi.fn(),
  updateChatThread: vi.fn(),
  getChatThread: vi.fn(),
  createChatThread: vi.fn(),
  deleteChatThread: vi.fn(),
  sendChatThreadMessageStream: vi.fn(),
}));

const createObjectUrlMock = vi.fn<(file: File) => string>();
const revokeObjectUrlMock = vi.fn<(url: string) => void>();

vi.mock("@/api", () => ({
  api: {
    listChatThreads: apiMocks.listChatThreads,
    createResponse: apiMocks.createResponse,
    updateChatThread: apiMocks.updateChatThread,
    getChatThread: apiMocks.getChatThread,
    createChatThread: apiMocks.createChatThread,
    deleteChatThread: apiMocks.deleteChatThread,
    sendChatThreadMessageStream: apiMocks.sendChatThreadMessageStream,
  },
}));

describe("ChatPlayground", () => {
  beforeEach(() => {
    apiMocks.listChatThreads.mockReset();
    apiMocks.createResponse.mockReset();
    apiMocks.updateChatThread.mockReset();
    apiMocks.getChatThread.mockReset();
    apiMocks.createChatThread.mockReset();
    apiMocks.deleteChatThread.mockReset();
    apiMocks.sendChatThreadMessageStream.mockReset();
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
              value: "Qwen3.8-27B-FP8",
              label: "Qwen3.8 27B (FP8)",
              statusLabel: "Not loaded",
              isReady: false,
            },
          ]}
          onSelectModel={vi.fn()}
          onOpenModelManager={vi.fn()}
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
    fireEvent.click(modelPicker);

    const qwen38Option = await screen.findByRole("option", {
      name: /Qwen3.8 27B/,
    });
    expect(qwen38Option).toBeInTheDocument();
    fireEvent.keyDown(document.activeElement ?? qwen38Option, { key: "Escape" });
    await waitFor(() =>
      expect(screen.queryByRole("listbox")).not.toBeInTheDocument(),
    );

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

    const sendButton = screen.getByRole("button", { name: "Send message" });
    const tokensStat = screen.getByText("12 tokens");
    const position = sendButton.compareDocumentPosition(tokensStat);
    expect(position & Node.DOCUMENT_POSITION_FOLLOWING).not.toBe(0);
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

    viewport.scrollTop = 300;
    fireEvent.scroll(viewport);
    scrollHeight = 1_300;
    act(() => streamCallbacks?.onDelta("First token"));
    expect(viewport.scrollTop).toBe(300);

    scrollHeight = 1_400;
    act(() => streamCallbacks?.onDelta(" second token"));
    expect(viewport.scrollTop).toBe(300);

    viewport.scrollTop = 1_000;
    fireEvent.scroll(viewport);
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

    expect(await screen.findByText("Chat History")).toBeInTheDocument();

    fireEvent.pointerDown(
      screen.getByRole("button", { name: "Delete Royal families in Europe" }),
    );
    fireEvent.click(
      screen.getByRole("button", { name: "Delete Royal families in Europe" }),
    );

    expect(
      await screen.findByRole("button", { name: "Cancel" }),
    ).toBeInTheDocument();
    expect(screen.getByText("Chat History")).toBeInTheDocument();
    expect(
      screen.getAllByText("Royal families in Europe").length,
    ).toBeGreaterThan(0);

    fireEvent.click(screen.getByRole("button", { name: "Cancel" }));

    await waitFor(() =>
      expect(screen.queryByText("Delete chat thread?")).not.toBeInTheDocument(),
    );
    expect(screen.getByText("Chat History")).toBeInTheDocument();

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
    expect(threadTitle).toHaveClass("break-words");
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
    const modelsButton = screen.getByRole("button", { name: "Models" });
    const thinkingButton = screen.getByRole("button", {
      name: "Disable thinking mode",
    });

    expect(attachImageButton).not.toHaveTextContent(/Attach image/i);

    const actionButtons = within(composerActions).getAllByRole("button");
    expect(actionButtons[0]).toBe(attachImageButton);
    expect(actionButtons[1]).toBe(modelsButton);
    expect(actionButtons[2]).toBe(thinkingButton);

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
    let drawer = await screen.findByRole("dialog", { name: "Chat History" });
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
    drawer = await screen.findByRole("dialog", { name: "Chat History" });
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
    expect(
      screen.getByRole("button", { name: "Show thought process" }),
    ).toBeInTheDocument();
    expect(screen.queryByText("reasoning first")).not.toBeInTheDocument();

    fireEvent.click(
      screen.getByRole("button", { name: "Show thought process" }),
    );

    expect(screen.getByText("reasoning first")).toBeInTheDocument();

    const writeText = vi.fn().mockResolvedValue(undefined);
    Object.defineProperty(navigator, "clipboard", {
      configurable: true,
      value: { writeText },
    });
    fireEvent.click(screen.getByRole("button", { name: "Copy reply" }));
    expect(await screen.findByRole("status")).toHaveTextContent("Copied");
    expect(writeText).toHaveBeenCalledWith("Final answer");
    fireEvent.click(screen.getByRole("button", { name: "Copy message" }));
    expect(writeText).toHaveBeenLastCalledWith("Solve this");
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
    expect(await screen.findByText("100 t/s decode")).toBeInTheDocument();
    expect(screen.getByText("1550 ms TTFT")).toBeInTheDocument();
    expect(screen.getByText("41 tokens")).toBeInTheDocument();
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

});
