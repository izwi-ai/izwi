import { fireEvent, render, screen, within } from "@testing-library/react";
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

  it("shows the licence the API gives for both Fish models, and none without one", () => {
    const actions = { onDownload: vi.fn(), onLoad: vi.fn(), onUnload: vi.fn() };
    const fish = (variant: string, extra: Partial<ModelInfo>): ModelInfo => ({
      ...model,
      variant,
      status: "not_downloaded",
      local_path: null,
      size_bytes: null,
      ...extra,
    });
    render(
      <ul>
        <ModelListRow
          model={fish("FishAudio-S2-Pro", {
            license: "Fish Audio Research License",
          })}
          label="Fish S2 Pro"
          subtitle="Fish Audio"
          progress={0}
          actions={actions}
        />
        <ModelListRow
          model={fish("FishAudio-Drama-3", {
            license: "Fish Audio Community License v1.0",
            license_notice:
              "Free for research and non-commercial use; commercial use only below USD 500k annual revenue; offering it as a hosted service requires a licence from Fish Audio",
          })}
          label="Fish Drama 3"
          subtitle="Fish Audio"
          progress={0}
          actions={actions}
        />
        <ModelListRow
          model={fish("Kokoro-82M", {})}
          label="Kokoro"
          subtitle="hexgrad"
          progress={0}
          actions={actions}
        />
      </ul>,
    );

    expect(
      within(screen.getByTestId("model-row-FishAudio-S2-Pro")).getByText(
        "License: Fish Audio Research License",
      ),
    ).toBeInTheDocument();
    expect(
      within(screen.getByTestId("model-row-FishAudio-Drama-3")).getByText(
        "License: Fish Audio Community License v1.0. Free for research and non-commercial use; commercial use only below USD 500k annual revenue; offering it as a hosted service requires a licence from Fish Audio.",
      ),
    ).toBeInTheDocument();
    expect(
      within(screen.getByTestId("model-row-Kokoro-82M")).queryByText(/License/),
    ).toBeNull();
    // The licence sits beside the download, where the model is chosen.
    expect(
      within(screen.getByTestId("model-row-FishAudio-Drama-3")).getByRole(
        "button",
        { name: "Download Fish Drama 3" },
      ),
    ).toBeInTheDocument();
  });

  it("offers no folder outside the desktop app", () => {
    reveal.canRevealPaths.mockReturnValueOnce(false);
    expect(revealModelActions(vi.fn())).toEqual({});
  });
});
