import { fireEvent, renderHook } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import { useWorkspaceShortcuts } from "@/hooks/useWorkspaceShortcuts";

describe("useWorkspaceShortcuts", () => {
  it("runs the latest handler without re-registering the listener", () => {
    const addListener = vi.spyOn(window, "addEventListener");
    const first = vi.fn();
    const second = vi.fn();

    const { rerender } = renderHook(
      ({ action }) => useWorkspaceShortcuts([{ key: "Escape", action }]),
      { initialProps: { action: first } },
    );
    rerender({ action: second });
    fireEvent.keyDown(window, { key: "Escape" });

    expect(first).not.toHaveBeenCalled();
    expect(second).toHaveBeenCalledTimes(1);
    expect(
      addListener.mock.calls.filter(([type]) => type === "keydown"),
    ).toHaveLength(1);
    addListener.mockRestore();
  });

  it("skips disabled shortcuts and editable targets", () => {
    const action = vi.fn();
    renderHook(() =>
      useWorkspaceShortcuts([
        { key: "a", action, enabled: false },
        { key: "b", action },
      ]),
    );
    const input = document.createElement("input");
    document.body.appendChild(input);

    fireEvent.keyDown(window, { key: "a" });
    fireEvent.keyDown(input, { key: "b" });
    expect(action).not.toHaveBeenCalled();

    fireEvent.keyDown(window, { key: "b" });
    expect(action).toHaveBeenCalledTimes(1);
    input.remove();
  });
});
