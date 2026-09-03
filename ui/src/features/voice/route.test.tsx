import {
  fireEvent,
  render,
  screen,
  waitFor,
  within,
} from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { VoicePage } from "./route";

const apiMocks = vi.hoisted(() => ({
  getVoiceProfile: vi.fn(),
  listVoiceObservations: vi.fn(),
}));

vi.mock("@/api", () => ({
  api: {
    baseUrl: "http://localhost:3000",
    getVoiceProfile: apiMocks.getVoiceProfile,
    listVoiceObservations: apiMocks.listVoiceObservations,
  },
}));

describe("VoicePage model readiness", () => {
  beforeEach(() => {
    apiMocks.getVoiceProfile.mockReset();
    apiMocks.listVoiceObservations.mockReset();
    apiMocks.getVoiceProfile.mockResolvedValue({
      id: "default",
      name: "Default",
      system_prompt: "Be helpful.",
      default_system_prompt: "Be helpful.",
      observational_memory_enabled: true,
      created_at: 0,
      updated_at: 0,
    });
    apiMocks.listVoiceObservations.mockResolvedValue([]);

    Object.defineProperty(window, "localStorage", {
      configurable: true,
      value: {
        getItem: vi.fn(() => null),
        setItem: vi.fn(),
        removeItem: vi.fn(),
        clear: vi.fn(),
      },
    });
    Object.defineProperty(window, "requestAnimationFrame", {
      configurable: true,
      value: (callback: FrameRequestCallback) => {
        callback(0);
        return 1;
      },
    });
    Object.defineProperty(window, "cancelAnimationFrame", {
      configurable: true,
      value: vi.fn(),
    });
    Object.defineProperty(HTMLElement.prototype, "scrollIntoView", {
      configurable: true,
      value: vi.fn(),
    });
  });

  it("keeps Start visible and provides one actionable route to model setup", async () => {
    render(
      <VoicePage
        models={[]}
        loading={false}
        downloadProgress={{}}
        onDownload={vi.fn()}
        onLoad={vi.fn()}
        onUnload={vi.fn()}
        onDelete={vi.fn()}
      />,
    );

    const start = screen.getByRole("button", { name: "Start Conversation" });
    const setup = screen.getByRole("button", { name: "Set up voice models" });

    expect(start).toBeVisible();
    expect(start).toBeDisabled();
    expect(start).toHaveAccessibleDescription(
      "Add the required ASR, text, and TTS models before starting a conversation.",
    );
    expect(
      screen.getAllByRole("button", { name: "Set up voice models" }),
    ).toHaveLength(1);

    setup.focus();
    fireEvent.click(setup);

    const dialog = await screen.findByRole("dialog", {
      name: "Voice configuration",
    });
    expect(dialog).toHaveFocus();
    expect(screen.getByRole("tab", { name: /Models/i })).toHaveAttribute(
      "data-state",
      "active",
    );

    fireEvent.click(within(dialog).getByRole("button", { name: "Close" }));
    await waitFor(() => expect(setup).toHaveFocus());
  });
});
