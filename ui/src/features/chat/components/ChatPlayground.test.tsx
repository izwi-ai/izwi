import { fireEvent, render, screen, waitFor } from "@testing-library/react";
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
  });
});
