import { beforeEach, describe, expect, it, vi } from "vitest";

const tauri = vi.hoisted(() => ({
  isTauri: vi.fn(() => false),
  invoke: vi.fn(),
}));
vi.mock("@tauri-apps/api/core", () => tauri);

import { canRevealPaths, revealPath } from "@/shared/revealPath";

describe("revealPath", () => {
  beforeEach(() => {
    tauri.isTauri.mockReturnValue(false);
    tauri.invoke.mockReset();
  });

  it("is available only in the desktop app", () => {
    expect(canRevealPaths()).toBe(false);
    tauri.isTauri.mockReturnValue(true);
    expect(canRevealPaths()).toBe(true);
  });

  it("asks the desktop app to show the path", async () => {
    tauri.invoke.mockResolvedValue(null);
    await revealPath("/Users/me/Library/Application Support/izwi/models/Kokoro-82M");
    expect(tauri.invoke).toHaveBeenCalledWith("reveal_path", {
      path: "/Users/me/Library/Application Support/izwi/models/Kokoro-82M",
    });
  });

  it("passes on why a path can't be shown", async () => {
    tauri.invoke.mockRejectedValue("only files in the models folder can be shown");
    await expect(revealPath("/etc")).rejects.toBe(
      "only files in the models folder can be shown",
    );
  });
});
