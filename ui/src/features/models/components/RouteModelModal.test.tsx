import { useState } from "react";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
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
  it("allows a loading model to be cancelled", () => {
    const onUnload = vi.fn();
    render(
      <RouteModelModal
        isOpen
        onClose={vi.fn()}
        title="ASR Models"
        description="Manage ASR models."
        models={[buildModel({ status: "loading" })]}
        loading={false}
        selectedVariant={null}
        downloadProgress={{}}
        onDownload={vi.fn()}
        onLoad={vi.fn()}
        onUnload={onUnload}
        onDelete={vi.fn()}
        onUseModel={vi.fn()}
      />,
    );

    fireEvent.click(screen.getByRole("button", { name: "Cancel load" }));
    expect(onUnload).toHaveBeenCalledWith(
      "diar_streaming_sortformer_4spk-v2.1",
    );
  });

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

  it("uses a named modal dialog, closes with Escape, and returns focus", async () => {
    function ModalHarness() {
      const [open, setOpen] = useState(false);
      return (
        <>
          <button type="button" onClick={() => setOpen(true)}>
            Open model manager
          </button>
          <RouteModelModal
            isOpen={open}
            onClose={() => setOpen(false)}
            title="Chat Models"
            description="Choose and manage a model for chat."
            models={[buildModel({ status: "downloaded" })]}
            loading={false}
            selectedVariant={null}
            downloadProgress={{}}
            onDownload={vi.fn()}
            onLoad={vi.fn()}
            onUnload={vi.fn()}
            onDelete={vi.fn()}
            onUseModel={vi.fn()}
          />
        </>
      );
    }

    render(<ModalHarness />);
    const opener = screen.getByRole("button", { name: "Open model manager" });
    opener.focus();
    fireEvent.click(opener);

    const dialog = await screen.findByRole("dialog", { name: "Chat Models" });
    expect(dialog).toHaveAccessibleDescription(
      "Choose and manage a model for chat.",
    );
    expect(dialog).toContainElement(document.activeElement as HTMLElement);

    fireEvent.keyDown(document.activeElement ?? dialog, { key: "Escape" });

    await waitFor(() => expect(dialog).not.toBeInTheDocument());
    await waitFor(() => expect(opener).toHaveFocus());
  });

  it("keeps delete confirmation modal and restores focus to its model action", async () => {
    render(
      <RouteModelModal
        isOpen
        onClose={vi.fn()}
        title="TTS Models"
        description="Manage TTS models."
        models={[buildModel({ variant: "ready-model", status: "ready" })]}
        loading={false}
        selectedVariant={null}
        downloadProgress={{}}
        onDownload={vi.fn()}
        onLoad={vi.fn()}
        onUnload={vi.fn()}
        onDelete={vi.fn()}
        onUseModel={vi.fn()}
      />,
    );

    const deleteButton = screen.getByRole("button", {
      name: "Delete ready-model",
    });
    fireEvent.click(deleteButton);

    const confirmDialog = await screen.findByRole("dialog", {
      name: "Delete model?",
    });
    expect(confirmDialog).toHaveAccessibleDescription(
      "This removes ready-model from local storage.",
    );
    expect(screen.getByText("TTS Models")).toBeInTheDocument();

    fireEvent.keyDown(document.activeElement ?? confirmDialog, { key: "Escape" });

    await waitFor(() => expect(confirmDialog).not.toBeInTheDocument());
    expect(screen.getByRole("dialog", { name: "TTS Models" })).toBeInTheDocument();
    expect(deleteButton).toHaveFocus();
  });

  it("offers an actionable manual-download guide", () => {
    render(
      <RouteModelModal
        isOpen
        onClose={vi.fn()}
        title="Chat Models"
        description="Manage chat models."
        models={[buildModel({ variant: "Gemma-3-1b-it", status: "not_downloaded" })]}
        loading={false}
        selectedVariant={null}
        downloadProgress={{}}
        onDownload={vi.fn()}
        onLoad={vi.fn()}
        onUnload={vi.fn()}
        onDelete={vi.fn()}
        onUseModel={vi.fn()}
      />,
    );

    expect(
      screen.getByRole("link", {
        name: "Open manual download guide for Gemma-3-1b-it",
      }),
    ).toHaveAttribute(
      "href",
      "https://github.com/izwi-ai/izwi/blob/main/docs/user/models/manual-gemma-3-1b-download.md",
    );
  });
});
