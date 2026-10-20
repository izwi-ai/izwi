import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

const tauri = vi.hoisted(() => ({
  isTauri: vi.fn(() => false),
  invoke: vi.fn(),
}));
vi.mock("@tauri-apps/api/core", () => tauri);

import { saveFile, saveServerFile, savesWithDialog } from "@/shared/saveFile";

describe("saveFile", () => {
  beforeEach(() => {
    tauri.isTauri.mockReturnValue(false);
    tauri.invoke.mockReset();
  });

  afterEach(() => {
    vi.restoreAllMocks();
    vi.useRealTimers();
  });

  it("downloads through a link in a browser", async () => {
    vi.useFakeTimers();
    const createObjectUrl = vi
      .spyOn(URL, "createObjectURL")
      .mockReturnValue("blob:export");
    const revokeObjectUrl = vi
      .spyOn(URL, "revokeObjectURL")
      .mockImplementation(() => {});
    const click = vi
      .spyOn(HTMLAnchorElement.prototype, "click")
      .mockImplementation(() => {});

    await expect(saveFile("notes.txt", "Hello")).resolves.toBe(true);

    expect(savesWithDialog()).toBe(false);
    const blob = createObjectUrl.mock.calls[0]?.[0] as Blob;
    await expect(blob.text()).resolves.toBe("Hello");
    expect(blob.type).toBe("text/plain;charset=utf-8");
    const anchor = click.mock.instances[0] as unknown as HTMLAnchorElement;
    expect(anchor.download).toBe("notes.txt");
    expect(anchor.href).toBe("blob:export");
    expect(anchor.isConnected).toBe(false);
    // Revoked only once the browser has had time to read it.
    expect(revokeObjectUrl).not.toHaveBeenCalled();
    vi.advanceTimersByTime(1000);
    expect(revokeObjectUrl).toHaveBeenCalledWith("blob:export");
    expect(tauri.invoke).not.toHaveBeenCalled();
  });

  it("asks the desktop app to save the bytes under the suggested name", async () => {
    tauri.isTauri.mockReturnValue(true);
    tauri.invoke.mockResolvedValue("/Users/me/Réunion.srt");
    const createObjectUrl = vi.spyOn(URL, "createObjectURL");

    await expect(
      saveFile("Réunion.srt", "1\n00:00:00,000 --> 00:00:01,000\nBonjour", "application/x-subrip"),
    ).resolves.toBe(true);

    expect(savesWithDialog()).toBe(true);
    expect(createObjectUrl).not.toHaveBeenCalled();
    const [command, body, options] = tauri.invoke.mock.calls[0] ?? [];
    expect(command).toBe("save_file");
    expect(new TextDecoder().decode(body as Uint8Array)).toContain("Bonjour");
    expect(options).toEqual({
      headers: { "x-izwi-filename": "R%C3%A9union.srt" },
    });
  });

  it("reports a cancelled save dialog", async () => {
    tauri.isTauri.mockReturnValue(true);
    tauri.invoke.mockResolvedValue(null);

    await expect(saveFile("notes.txt", new Blob(["Hi"]))).resolves.toBe(false);
  });

  it("saves server files through the desktop app's download command", async () => {
    tauri.invoke.mockResolvedValueOnce("/Users/me/take.wav").mockResolvedValueOnce(null);

    await expect(
      saveServerFile("http://localhost:8080/v1/text-to-speech/1/audio", "take.wav"),
    ).resolves.toBe(true);
    await expect(
      saveServerFile("http://localhost:8080/v1/text-to-speech/1/audio", "take.wav"),
    ).resolves.toBe(false);

    expect(tauri.invoke).toHaveBeenCalledWith("download_audio_file", {
      url: "http://localhost:8080/v1/text-to-speech/1/audio",
      suggestedFilename: "take.wav",
    });
  });
});
