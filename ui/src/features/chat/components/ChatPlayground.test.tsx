import { fireEvent, render, screen, waitFor, within } from "@testing-library/react";
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
          modelLabel="Qwen3 Chat 0.6B GGUF"
          modelOptions={[
            {
              value: "Qwen3-0.6B-GGUF",
              label: "Qwen3 Chat 0.6B GGUF",
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
      screen.getByRole("button", { name: "Qwen3 Chat 0.6B GGUF" }),
    );

    expect(await screen.findByText("Gemma 3 1B")).toBeInTheDocument();

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
          modelLabel="Qwen3 Chat 0.6B GGUF"
          modelOptions={[
            {
              value: "Qwen3-0.6B-GGUF",
              label: "Qwen3 Chat 0.6B GGUF",
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
      screen.queryByText("Using Qwen3 Chat 0.6B GGUF"),
    ).not.toBeInTheDocument();

    const sendButton = screen.getByRole("button", { name: "Send message" });
    const tokensStat = screen.getByText("12 tokens");
    const position = sendButton.compareDocumentPosition(tokensStat);
    expect(position & Node.DOCUMENT_POSITION_FOLLOWING).not.toBe(0);
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
          modelLabel="Qwen3 Chat 0.6B GGUF"
          modelOptions={[
            {
              value: "Qwen3-0.6B-GGUF",
              label: "Qwen3 Chat 0.6B GGUF",
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
          modelLabel="Qwen3 Chat 0.6B GGUF"
          modelOptions={[
            {
              value: "Qwen3-0.6B-GGUF",
              label: "Qwen3 Chat 0.6B GGUF",
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
});
