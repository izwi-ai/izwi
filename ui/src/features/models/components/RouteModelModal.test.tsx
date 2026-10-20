import { useState } from "react";
import { fireEvent, render, screen, waitFor, within } from "@testing-library/react";
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
  it("shows friendly model names instead of raw variant ids", () => {
    render(
      <RouteModelModal
        isOpen
        onClose={vi.fn()}
        title="Text-to-Speech Models"
        description="Manage TTS models."
        models={[
          buildModel({ variant: "Qwen3-TTS-12Hz-0.6B-CustomVoice", status: "downloaded" }),
          buildModel({ variant: "unlisted-model-x", status: "downloaded" }),
        ]}
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

    expect(screen.getByText("Qwen3 0.6B CustomVoice")).toBeInTheDocument();
    expect(
      screen.queryByText("Qwen3-TTS-12Hz-0.6B-CustomVoice"),
    ).not.toBeInTheDocument();
    // Models the catalog does not list keep their id rather than vanishing.
    expect(screen.getByText("unlisted-model-x")).toBeInTheDocument();
  });

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

    fireEvent.click(
      screen.getByRole("button", { name: /^Cancel loading / }),
    );
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

    expect(screen.queryByText(/^Selected/)).not.toBeInTheDocument();
    expect(
      screen.queryByRole("button", { name: /^Use / }),
    ).not.toBeInTheDocument();
    expect(
      screen.getByRole("button", { name: /^Unload / }),
    ).toBeInTheDocument();
  });

  it("marks kept-loaded models and shows the residency summary in manage mode", () => {
    render(
      <RouteModelModal
        isOpen
        onClose={vi.fn()}
        title="Diarization Models"
        description="Manage pipeline models for /v1/diarizations."
        models={[
          buildModel({ variant: "diar_streaming_sortformer_4spk-v2.1", pinned: true }),
          buildModel({ variant: "Nemotron-3-Diarization", status: "downloaded" }),
        ]}
        loading={false}
        selectedVariant={null}
        downloadProgress={{}}
        onDownload={vi.fn()}
        onLoad={vi.fn()}
        onUnload={vi.fn()}
        onDelete={vi.fn()}
        onUseModel={vi.fn()}
        selectionMode="manage"
        residencySummary={{
          resident_count: 1,
          max_loaded_models: 4,
          model_keep_alive_secs: 600,
        }}
      />,
    );

    expect(screen.getByText(/Kept loaded/)).toBeInTheDocument();
    const summary = screen.getByTestId("route-model-residency-summary");
    expect(summary).toHaveTextContent("1 model loaded of 4");
  });

  it("omits the residency summary when the server does not provide one", () => {
    render(
      <RouteModelModal
        isOpen
        onClose={vi.fn()}
        title="Diarization Models"
        description="Manage pipeline models for /v1/diarizations."
        models={[buildModel({})]}
        loading={false}
        selectedVariant={null}
        downloadProgress={{}}
        onDownload={vi.fn()}
        onLoad={vi.fn()}
        onUnload={vi.fn()}
        onDelete={vi.fn()}
        onUseModel={vi.fn()}
        selectionMode="manage"
      />,
    );

    expect(
      screen.getByTestId("route-model-residency-summary"),
    ).toBeEmptyDOMElement();
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

    const menuButton = screen.getByRole("button", {
      name: "More actions for ready-model",
    });
    fireEvent.pointerDown(menuButton);
    fireEvent.click(menuButton);
    fireEvent.click(
      await screen.findByRole("menuitem", { name: "Delete download" }),
    );

    const confirmDialog = await screen.findByRole("dialog", {
      name: /Delete download/,
    });
    expect(confirmDialog).toHaveTextContent("stops any job using the model");

    fireEvent.keyDown(document.activeElement ?? confirmDialog, { key: "Escape" });

    await waitFor(() => expect(confirmDialog).not.toBeInTheDocument());
    expect(screen.getByRole("dialog", { name: "TTS Models" })).toBeInTheDocument();
  });

  it("offers a download action for not-downloaded models", () => {
    render(
      <RouteModelModal
        isOpen
        onClose={vi.fn()}
        title="Chat Models"
        description="Manage chat models."
        models={[buildModel({ variant: "Qwen3.5-4B", status: "not_downloaded" })]}
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
      screen.getByRole("button", { name: /Download/i }),
    ).toBeInTheDocument();
  });

  it("groups Qwen3.6 under Qwen and orders Qwen chat models by version number", () => {
    render(
      <RouteModelModal
        isOpen
        onClose={vi.fn()}
        title="Chat Models"
        description="Manage chat models."
        models={[
          buildModel({ variant: "Qwen3.8-27B-FP8", size_bytes: 30_889_968_808 }),
          buildModel({
            variant: "Qwen3.6-35B-A3B-FP8",
            size_bytes: 37_470_000_000,
          }),
          buildModel({ variant: "Qwen3.5-9B", size_bytes: 6_598_688_544 }),
        ]}
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

    expect(screen.getAllByText(/^Qwen$/)).toHaveLength(3);
    expect(screen.queryByText(/^Other$/)).not.toBeInTheDocument();

    // Version order wins over size order: the 27B Qwen3.8 renders after the
    // 35B Qwen3.6, and the Qwen3.5 model stays before both.
    const orderedRows = [
      "model-row-Qwen3.5-9B",
      "model-row-Qwen3.6-35B-A3B-FP8",
      "model-row-Qwen3.8-27B-FP8",
    ].map((testId) => screen.getByTestId(testId));
    for (let index = 0; index + 1 < orderedRows.length; index += 1) {
      expect(
        orderedRows[index].compareDocumentPosition(orderedRows[index + 1]) &
          Node.DOCUMENT_POSITION_FOLLOWING,
      ).toBeTruthy();
    }
  });

  it("says why a model can't run here and why a download failed", () => {
    const onDownload = vi.fn();
    render(
      <RouteModelModal
        isOpen
        onClose={vi.fn()}
        title="Chat models"
        description="Manage chat models."
        models={[
          buildModel({
            variant: "Qwen3.8-27B-FP8",
            status: "not_downloaded",
            device_fit: "too_large",
            estimated_memory_bytes: 80 * 1024 ** 3,
          }),
          buildModel({
            variant: "Qwen3.5-9B",
            status: "error",
            error_message: "Not enough free disk space for downloading Qwen3.5-9B",
          }),
        ]}
        loading={false}
        selectedVariant={null}
        downloadProgress={{}}
        onDownload={onDownload}
        onLoad={vi.fn()}
        onUnload={vi.fn()}
        onDelete={vi.fn()}
        onUseModel={vi.fn()}
      />,
    );

    const tooLarge = screen.getByTestId("model-row-Qwen3.8-27B-FP8");
    expect(within(tooLarge).getByText("Needs 80 GB memory")).toBeInTheDocument();
    expect(within(tooLarge).queryByRole("button", { name: /^Download/ })).toBeNull();

    const failed = screen.getByTestId("model-row-Qwen3.5-9B");
    expect(within(failed).getByText(/Not enough free disk space/)).toBeInTheDocument();
    fireEvent.click(within(failed).getByRole("button", { name: /^Retry / }));
    expect(onDownload).toHaveBeenCalledWith("Qwen3.5-9B");
  });

  it("uses another loaded model on this page", () => {
    const onUseModel = vi.fn();
    const onClose = vi.fn();
    render(
      <RouteModelModal
        isOpen
        onClose={onClose}
        title="Chat models"
        description="Manage chat models."
        routeLabel="Chat"
        models={[
          buildModel({ variant: "Qwen3.5-4B", status: "ready" }),
          buildModel({ variant: "Qwen3.5-2B", status: "ready" }),
        ]}
        loading={false}
        selectedVariant="Qwen3.5-4B"
        downloadProgress={{}}
        onDownload={vi.fn()}
        onLoad={vi.fn()}
        onUnload={vi.fn()}
        onDelete={vi.fn()}
        onUseModel={onUseModel}
      />,
    );

    expect(
      within(screen.getByTestId("model-row-Qwen3.5-4B")).getByText(/Selected in Chat/),
    ).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: /^Use / }));
    expect(onUseModel).toHaveBeenCalledWith("Qwen3.5-2B");
    expect(onClose).toHaveBeenCalled();
  });
});
