import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import type { ModelInfo } from "@/api";
import {
  ModelListRow,
  revealModelActions,
} from "@/features/models/components/ModelListRow";

const reveal = vi.hoisted(() => ({
  canRevealPaths: vi.fn(() => true),
  revealPath: vi.fn(async () => undefined),
}));
vi.mock("@/shared/revealPath", () => reveal);

const model: ModelInfo = {
  variant: "Kokoro-82M",
  status: "ready",
  local_path: "/models/Kokoro-82M",
  size_bytes: 350_000_000,
  download_progress: null,
  error_message: null,
  pinned: false,
};

async function openMenu() {
  const trigger = screen.getByRole("button", { name: "More actions for Kokoro" });
  fireEvent.pointerDown(trigger);
  fireEvent.click(trigger);
}

describe("ModelListRow", () => {
  it("keeps a loaded model loaded and shows its folder", async () => {
    const onSetPinned = vi.fn();
    render(
      <ul>
        <ModelListRow
          model={model}
          label="Kokoro"
          subtitle="Hexgrad"
          progress={0}
          actions={{
            onDownload: vi.fn(),
            onLoad: vi.fn(),
            onUnload: vi.fn(),
            onSetPinned,
            ...revealModelActions(vi.fn()),
          }}
        />
      </ul>,
    );

    expect(screen.getByText("350 MB")).toBeInTheDocument();
    await openMenu();
    const keepLoaded = await screen.findByRole("menuitemcheckbox", {
      name: "Keep loaded",
    });
    expect(keepLoaded).toHaveAttribute("aria-checked", "false");
    fireEvent.click(keepLoaded);
    expect(onSetPinned).toHaveBeenCalledWith("Kokoro-82M", true);

    await openMenu();
    fireEvent.click(await screen.findByRole("menuitem", { name: /^Show in / }));
    expect(reveal.revealPath).toHaveBeenCalledWith("/models/Kokoro-82M");
  });

  it("offers no folder outside the desktop app", () => {
    reveal.canRevealPaths.mockReturnValueOnce(false);
    expect(revealModelActions(vi.fn())).toEqual({});
  });
});
