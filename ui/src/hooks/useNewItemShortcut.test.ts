import { fireEvent, renderHook } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import { useNewItemShortcut } from "@/hooks/useNewItemShortcut";

describe("useNewItemShortcut", () => {
  it("runs on Cmd/Ctrl+N unless a dialog is open", () => {
    const onNew = vi.fn();
    renderHook(() => useNewItemShortcut(onNew));

    fireEvent.keyDown(window, { key: "n", metaKey: true });
    fireEvent.keyDown(window, { key: "n" });
    expect(onNew).toHaveBeenCalledTimes(1);

    const dialog = document.createElement("div");
    dialog.setAttribute("role", "dialog");
    dialog.setAttribute("data-state", "open");
    document.body.appendChild(dialog);
    fireEvent.keyDown(window, { key: "n", ctrlKey: true });
    expect(onNew).toHaveBeenCalledTimes(1);
    dialog.remove();
  });
});
