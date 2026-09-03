import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { RouteModelSelect } from "@/components/RouteModelSelect";

const options = [
  {
    value: "model-ready",
    label: "Ready Model",
    statusLabel: "Ready",
    isReady: true,
  },
  {
    value: "model-loading",
    label: "Loading Model",
    statusLabel: "Loading 42%",
    isReady: false,
  },
  {
    value: "model-disabled",
    label: "Unavailable Model",
    statusLabel: "Unavailable",
    isReady: false,
    disabled: true,
  },
];

describe("RouteModelSelect", () => {
  beforeEach(() => {
    HTMLElement.prototype.scrollIntoView = vi.fn();
  });

  it("provides named combobox/listbox semantics and complete keyboard selection", async () => {
    const onSelect = vi.fn();
    render(
      <RouteModelSelect
        aria-label="Render model"
        description="Choose the model used for narration."
        value="model-ready"
        options={options}
        onSelect={onSelect}
      />,
    );

    const trigger = screen.getByRole("combobox", { name: "Render model" });
    expect(trigger).toHaveAttribute("aria-expanded", "false");
    expect(trigger).toHaveAccessibleDescription(
      "Choose the model used for narration.",
    );

    trigger.focus();
    fireEvent.keyDown(trigger, { key: "ArrowDown" });

    expect(await screen.findByRole("listbox")).toBeInTheDocument();
    expect(trigger).toHaveAttribute("aria-expanded", "true");
    expect(
      screen.getByRole("option", { name: /Loading Model\. Status: Loading 42%/ }),
    ).toBeInTheDocument();
    expect(
      screen.getByRole("option", { name: /Unavailable Model/ }),
    ).toHaveAttribute("aria-disabled", "true");

    const readyOption = screen.getByRole("option", { name: /Ready Model/ });
    const loadingOption = screen.getByRole("option", { name: /Loading Model/ });
    await waitFor(() => expect(readyOption).toHaveAttribute("data-highlighted"));

    fireEvent.keyDown(document.activeElement ?? readyOption, { key: "End" });
    await waitFor(() => expect(loadingOption).toHaveAttribute("data-highlighted"));
    fireEvent.keyDown(document.activeElement ?? loadingOption, { key: "Home" });
    await waitFor(() => expect(readyOption).toHaveAttribute("data-highlighted"));
    fireEvent.keyDown(document.activeElement ?? readyOption, { key: "End" });
    await waitFor(() => expect(loadingOption).toHaveAttribute("data-highlighted"));
    fireEvent.keyDown(document.activeElement ?? loadingOption, { key: "Enter" });

    await waitFor(() => expect(onSelect).toHaveBeenCalledWith("model-loading"));
    expect(screen.queryByRole("listbox")).not.toBeInTheDocument();
    expect(trigger).toHaveFocus();
  });

  it("supports typeahead and Escape without changing the value", async () => {
    const onSelect = vi.fn();
    render(
      <RouteModelSelect
        aria-label="Chat model"
        value="model-ready"
        options={options}
        onSelect={onSelect}
      />,
    );

    const trigger = screen.getByRole("combobox", { name: "Chat model" });
    trigger.focus();
    fireEvent.keyDown(trigger, { key: "Enter" });
    await screen.findByRole("listbox");

    fireEvent.keyDown(document.activeElement ?? trigger, { key: "l" });
    await waitFor(() =>
      expect(
        screen.getByRole("option", { name: /Loading Model/ }),
      ).toHaveAttribute("data-highlighted"),
    );
    fireEvent.keyDown(document.activeElement ?? trigger, { key: "Escape" });

    await waitFor(() =>
      expect(screen.queryByRole("listbox")).not.toBeInTheDocument(),
    );
    expect(onSelect).not.toHaveBeenCalled();
    expect(trigger).toHaveFocus();
  });

  it("disables the named trigger when selection is unavailable", () => {
    render(
      <RouteModelSelect
        aria-label="Voice model"
        value={null}
        options={[]}
      />,
    );

    expect(screen.getByRole("combobox", { name: "Voice model" })).toBeDisabled();
  });
});
