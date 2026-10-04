import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { MemoryRouter } from "react-router-dom";

import type { ModelInfo } from "@/api";

import { ChatPage } from "./route";

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

function routeModel(
  variant: string,
  status: ModelInfo["status"],
): ModelInfo {
  return {
    variant,
    status,
    local_path: null,
    size_bytes: 1_000_000,
    download_progress: null,
    error_message: null,
    speech_capabilities: null,
  };
}

function renderChatPage(models: ModelInfo[]) {
  return render(
    <MemoryRouter initialEntries={["/chat"]}>
      <ChatPage
        models={models}
        selectedModel={null}
        loading={false}
        downloadProgress={{}}
        onDownload={vi.fn()}
        onLoad={vi.fn()}
        onUnload={vi.fn()}
        onDelete={vi.fn()}
        onSelect={vi.fn()}
        onError={vi.fn()}
      />
    </MemoryRouter>,
  );
}

describe("ChatPage route model list", () => {
  beforeEach(() => {
    apiMocks.listChatThreads.mockReset();
    apiMocks.listChatThreads.mockResolvedValue([]);
    HTMLElement.prototype.scrollIntoView = vi.fn();
  });

  it("lists the Qwen3.6-35B-A3B-FP8 MoE chat model in the Chat Models modal", async () => {
    renderChatPage([
      routeModel("LFM2.5-1.2B-Instruct-GGUF", "downloaded"),
      routeModel("Qwen3.6-35B-A3B-FP8", "not_downloaded"),
    ]);
    await waitFor(() => expect(apiMocks.listChatThreads).toHaveBeenCalled());

    fireEvent.click(screen.getByRole("button", { name: "Models" }));

    const modal = await screen.findByRole("dialog");
    expect(modal).toHaveTextContent("Chat Models");
    expect(
      screen.getByText("Qwen3.6 35B-A3B (FP8 MoE)"),
    ).toBeInTheDocument();
  });

  it("offers the Qwen3.6-35B-A3B-FP8 model in the composer model dropdown", async () => {
    renderChatPage([routeModel("Qwen3.6-35B-A3B-FP8", "not_downloaded")]);
    await waitFor(() => expect(apiMocks.listChatThreads).toHaveBeenCalled());

    fireEvent.click(screen.getByRole("combobox"));

    const option = await screen.findByRole("option", {
      name: /Qwen3\.6 35B-A3B \(FP8 MoE\)/,
    });
    expect(option).toBeInTheDocument();
  });
});
