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

    apiMocks.listChatThreads.mockResolvedValue([]);

    HTMLElement.prototype.scrollIntoView = vi.fn();
  });

  it("opens the header model dropdown and keeps the send action icon-only", async () => {
    render(
      <MemoryRouter initialEntries={["/chat"]}>
        <ChatPlayground
          selectedModel="Qwen3-0.6B-GGUF"
          selectedModelReady={true}
          supportsThinking={true}
          modelLabel="Qwen3 0.6B GGUF (Q8_0)"
          modelOptions={[
            {
              value: "Qwen3-0.6B-GGUF",
              label: "Qwen3 0.6B GGUF (Q8_0)",
              statusLabel: "Ready",
              isReady: true,
            },
            {
              value: "Gemma-3-1b-it",
              label: "Gemma 3 1B",
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

    fireEvent.click(
      screen.getByRole("combobox", { name: "Qwen3 0.6B GGUF (Q8_0)" }),
    );

    const gemmaOption = await screen.findByRole("option", {
      name: /Gemma 3 1B/,
    });
    expect(gemmaOption).toBeInTheDocument();
    fireEvent.keyDown(document.activeElement ?? gemmaOption, { key: "Escape" });
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

  it("shows the active thread title below the selector without the old conversation header", async () => {
    const thread = {
      id: "thread-1",
      title: "Royal families in Europe",
      model_id: "Qwen3-0.6B-GGUF",
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
          selectedModel="Qwen3-0.6B-GGUF"
          selectedModelReady={true}
          supportsThinking={true}
          modelLabel="Qwen3 0.6B GGUF (Q8_0)"
          modelOptions={[
            {
              value: "Qwen3-0.6B-GGUF",
              label: "Qwen3 0.6B GGUF (Q8_0)",
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
      screen.queryByText("Using Qwen3 0.6B GGUF (Q8_0)"),
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
      model_id: "Gemma-3-1b-it",
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
          selectedModel="Gemma-3-1b-it"
          selectedModelReady={true}
          supportsThinking={false}
          modelLabel="Gemma 3 1B"
          modelOptions={[
            {
              value: "Gemma-3-1b-it",
              label: "Gemma 3 1B",
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
      model_id: "Qwen3-0.6B-GGUF",
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
          selectedModel="Qwen3-0.6B-GGUF"
          selectedModelReady={true}
          supportsThinking={true}
          modelLabel="Qwen3 0.6B GGUF (Q8_0)"
          modelOptions={[
            {
              value: "Qwen3-0.6B-GGUF",
              label: "Qwen3 0.6B GGUF (Q8_0)",
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
      model_id: "Qwen3-0.6B-GGUF",
      created_at: 1,
      updated_at: 2,
      last_message_preview: "Preview",
      message_count: 2,
    };

    apiMocks.listChatThreads.mockResolvedValue([thread]);

    render(
      <MemoryRouter initialEntries={["/chat"]}>
        <ChatPlayground
          selectedModel="Qwen3-0.6B-GGUF"
          selectedModelReady={true}
          supportsThinking={true}
          modelLabel="Qwen3 0.6B GGUF (Q8_0)"
          modelOptions={[
            {
              value: "Qwen3-0.6B-GGUF",
              label: "Qwen3 0.6B GGUF (Q8_0)",
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
  });
});
