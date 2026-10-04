import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import type { ModelInfo } from "@/api";

import { MyModelsPage } from "./route";

function buildModel(overrides: Partial<ModelInfo>): ModelInfo {
  return {
    variant: "Qwen3.5-0.8B",
    status: "downloaded",
    local_path: "/tmp/model",
    size_bytes: null,
    download_progress: null,
    error_message: null,
    speech_capabilities: null,
    ...overrides,
  };
}

describe("MyModelsPage", () => {
  it("allows a loading model to be cancelled", () => {
    const onUnload = vi.fn();
    render(
      <MyModelsPage
        models={[buildModel({ status: "loading" })]}
        loading={false}
        downloadProgress={{}}
        onDownload={vi.fn()}
        onLoad={vi.fn()}
        onUnload={onUnload}
        onDelete={vi.fn()}
        onRefresh={vi.fn()}
      />,
    );

    fireEvent.click(screen.getByRole("button", { name: "Cancel load" }));
    expect(onUnload).toHaveBeenCalledWith("Qwen3.5-0.8B");
  });

  it("groups Qwen3.5 models under Qwen and uses Qwen3 model names without chat prefixes", () => {
    render(
      <MyModelsPage
        models={[
          buildModel({ variant: "Qwen3.5-0.8B", size_bytes: 715_600_000 }),
          buildModel({ variant: "Qwen3.8-27B-FP8", size_bytes: 30_889_968_808 }),
        ]}
        loading={false}
        downloadProgress={{}}
        onDownload={vi.fn()}
        onLoad={vi.fn()}
        onUnload={vi.fn()}
        onDelete={vi.fn()}
        onRefresh={vi.fn()}
      />,
    );

    expect(screen.getByText(/^Qwen$/)).toBeInTheDocument();
    expect(screen.queryByText(/^Other$/)).not.toBeInTheDocument();
    expect(screen.getByText("Qwen3.5 0.8B")).toBeInTheDocument();
    expect(screen.getByText("Qwen3.8 27B")).toBeInTheDocument();
    expect(screen.queryByText(/Qwen3 Chat 27B/i)).not.toBeInTheDocument();
  });

  it("lists Qwen3.6 under Qwen and orders Qwen chat models by version number", () => {
    render(
      <MyModelsPage
        models={[
          buildModel({ variant: "Qwen3.8-27B-FP8", size_bytes: 30_889_968_808 }),
          buildModel({
            variant: "Qwen3.6-35B-A3B-FP8",
            size_bytes: 37_470_000_000,
          }),
          buildModel({ variant: "Qwen3.5-9B", size_bytes: 6_598_688_544 }),
          buildModel({ variant: "Qwen3.5-0.8B", size_bytes: 715_600_000 }),
        ]}
        loading={false}
        downloadProgress={{}}
        onDownload={vi.fn()}
        onLoad={vi.fn()}
        onUnload={vi.fn()}
        onDelete={vi.fn()}
        onRefresh={vi.fn()}
      />,
    );

    expect(screen.getByText(/^Qwen$/)).toBeInTheDocument();
    expect(screen.queryByText(/^Other$/)).not.toBeInTheDocument();

    // Version order wins over size order: the 27B Qwen3.8 renders after the
    // 35B Qwen3.6, and the Qwen3.5 family stays before both.
    const orderedCards = [
      "Qwen3.5 0.8B",
      "Qwen3.5 9B",
      "Qwen3.6 35B-A3B",
      "Qwen3.8 27B",
    ].map((name) => screen.getByText(name));
    for (let index = 0; index + 1 < orderedCards.length; index += 1) {
      expect(
        orderedCards[index].compareDocumentPosition(orderedCards[index + 1]) &
          Node.DOCUMENT_POSITION_FOLLOWING,
      ).toBeTruthy();
    }
  });

  it("renders VibeVoice models from the backend catalog under Microsoft", () => {
    render(
      <MyModelsPage
        models={[
          buildModel({ variant: "VibeVoice-ASR", size_bytes: 17_348_198_410 }),
          buildModel({ variant: "VibeVoice-1.5B", size_bytes: 5_408_043_974 }),
        ]}
        loading={false}
        downloadProgress={{}}
        onDownload={vi.fn()}
        onLoad={vi.fn()}
        onUnload={vi.fn()}
        onDelete={vi.fn()}
        onRefresh={vi.fn()}
      />,
    );

    expect(screen.getByText(/^Microsoft$/)).toBeInTheDocument();
    expect(screen.getByText("VibeVoice ASR")).toBeInTheDocument();
    expect(screen.getByText("VibeVoice 1.5B TTS")).toBeInTheDocument();
    expect(screen.queryByText(/^Other$/)).not.toBeInTheDocument();
  });
});
