import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import type { ModelInfo } from "@/api";

import { RouteModelModal } from "./RouteModelModal";

function buildModel(overrides: Partial<ModelInfo>): ModelInfo {
  return {
    variant: "diar_streaming_sortformer_4spk-v2.1",
    status: "ready",
    local_path: "/tmp/model",
    size_bytes: 899_100_000,
    download_progress: null,
    error_message: null,
    speech_capabilities: null,
    ...overrides,
  };
}

describe("RouteModelModal", () => {
  it("renders manage-mode rows without selected route affordances", () => {
    render(
      <RouteModelModal
        isOpen
        onClose={vi.fn()}
        title="Diarization Models"
        description="Manage pipeline models for /v1/diarizations."
        models={[
          buildModel({
            variant: "diar_streaming_sortformer_4spk-v2.1",
          }),
        ]}
        loading={false}
        selectedVariant="diar_streaming_sortformer_4spk-v2.1"
        selectionMode="manage"
        downloadProgress={{}}
        onDownload={vi.fn()}
        onLoad={vi.fn()}
        onUnload={vi.fn()}
        onDelete={vi.fn()}
        onUseModel={vi.fn()}
      />,
    );

    expect(
      screen.queryByRole("button", { name: /Selected/i }),
    ).not.toBeInTheDocument();
    expect(
      screen.queryByRole("button", { name: /Use model/i }),
    ).not.toBeInTheDocument();
    expect(
      screen.getByRole("button", { name: /Unload/i }),
    ).toBeInTheDocument();

    const row = screen.getByTestId(
      "route-model-row-diar_streaming_sortformer_4spk-v2.1",
    );
    expect(row.className).toContain("border-[var(--border-muted)]");
    expect(row.className).not.toContain("border-[var(--border-strong)]");
  });
});
