import { render, screen } from "@testing-library/react";
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
  it("groups Qwen3.5 models under Qwen and uses Qwen3 model names without chat prefixes", () => {
    render(
      <MyModelsPage
        models={[
          buildModel({ variant: "Qwen3.5-0.8B", size_bytes: 715_600_000 }),
          buildModel({ variant: "Qwen3-0.6B-GGUF", size_bytes: 1_073_741_824 }),
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
    expect(screen.getByText("Qwen3 0.6B")).toBeInTheDocument();
    expect(screen.queryByText(/Qwen3 Chat 0\.6B/i)).not.toBeInTheDocument();
  });
});
