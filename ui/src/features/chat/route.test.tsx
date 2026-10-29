import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { MemoryRouter } from "react-router";

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
    getHealth: () => Promise.reject(new Error("offline")),
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

    fireEvent.click(screen.getByRole("combobox", { name: "Chat model" }));
    fireEvent.click(
      await screen.findByRole("option", { name: "Manage chat models…" }),
    );

    const modal = await screen.findByRole("dialog");
    expect(modal).toHaveTextContent("Chat models");
    expect(
      screen.getByText("Qwen3.6 35B-A3B (FP8 MoE)"),
    ).toBeInTheDocument();
  });

  it("offers audio clips for LFM3 Audio, which the catalog says hears audio", async () => {
    renderChatPage([
      {
        ...routeModel("LFM3-Audio-2B-GGUF", "ready"),
        modalities: ["text_input", "audio_input", "text_output", "audio_output"],
      },
    ]);
    await waitFor(() => expect(apiMocks.listChatThreads).toHaveBeenCalled());

    expect(screen.getByRole("combobox", { name: "Chat model" })).toHaveTextContent(
      "LFM3 Audio 2B GGUF (Q4_K_M)",
    );
    expect(screen.getByRole("button", { name: "Attach audio" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Record audio" })).toBeInTheDocument();
  });

  it("offers no audio clips for a text chat model", async () => {
    renderChatPage([
      {
        ...routeModel("LFM2.5-1.2B-Instruct-GGUF", "ready"),
        modalities: ["text_input", "text_output"],
      },
    ]);
    await waitFor(() => expect(apiMocks.listChatThreads).toHaveBeenCalled());

    expect(screen.queryByRole("button", { name: "Attach audio" })).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Record audio" })).not.toBeInTheDocument();
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
