import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import {
  Dialog,
  DialogContent,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";

describe("DialogContent", () => {
  it("provides a viewport-bounded internal scroll boundary", () => {
    render(
      <Dialog>
        <DialogTrigger>Open dialog</DialogTrigger>
        <DialogContent>
          <DialogTitle>Viewport-safe dialog</DialogTitle>
          <div>Long content</div>
        </DialogContent>
      </Dialog>,
    );

    fireEvent.click(screen.getByRole("button", { name: "Open dialog" }));

    const dialog = screen.getByRole("dialog", {
      name: "Viewport-safe dialog",
    });
    expect(dialog).toHaveClass("max-h-[calc(100dvh-2rem)]");
    expect(dialog).toHaveClass("w-[calc(100vw-2rem)]");
    expect(dialog).toHaveClass("overflow-y-auto");
    expect(dialog).toHaveClass("overscroll-contain");
    expect(screen.getByRole("button", { name: "Close" })).toBeInTheDocument();
  });

  it("preserves Escape dismissal and focus return", async () => {
    render(
      <Dialog>
        <DialogTrigger>Open dialog</DialogTrigger>
        <DialogContent>
          <DialogTitle>Keyboard dialog</DialogTitle>
          <button type="button">Focusable action</button>
        </DialogContent>
      </Dialog>,
    );

    const trigger = screen.getByRole("button", { name: "Open dialog" });
    fireEvent.click(trigger);

    const dialog = screen.getByRole("dialog", { name: "Keyboard dialog" });
    fireEvent.keyDown(dialog, { key: "Escape" });

    await waitFor(() => expect(dialog).not.toBeInTheDocument());
    await waitFor(() => expect(trigger).toHaveFocus());
  });
});
