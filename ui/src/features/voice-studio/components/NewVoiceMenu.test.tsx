import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { useState } from "react";
import { describe, expect, it, vi } from "vitest";

import { NewVoiceMenu, type NewVoiceKind } from "./NewVoiceMenu";

function Harness({ onSelect }: { onSelect: (kind: NewVoiceKind) => void }) {
  const [open, setOpen] = useState(false);
  return <NewVoiceMenu open={open} onOpenChange={setOpen} onSelect={onSelect} />;
}

function openMenu() {
  const trigger = screen.getByRole("button", { name: /New voice/ });
  fireEvent.pointerDown(trigger, { button: 0, ctrlKey: false });
  return trigger;
}

describe("NewVoiceMenu", () => {
  it("offers cloning and designing", async () => {
    render(<Harness onSelect={vi.fn()} />);
    openMenu();

    const items = await screen.findAllByRole("menuitem");
    expect(items.map((item) => item.textContent)).toEqual([
      "Clone a voiceFrom a short recording",
      "Design a voiceFrom a description",
    ]);
  });

  it.each([
    ["Clone a voice", "clone"],
    ["Design a voice", "design"],
  ] as const)("reports %s once focus is back on the button", async (label, kind) => {
    const onSelect = vi.fn();
    render(<Harness onSelect={onSelect} />);
    const trigger = openMenu();

    fireEvent.click(await screen.findByRole("menuitem", { name: new RegExp(label) }));

    await waitFor(() => expect(onSelect).toHaveBeenCalledWith(kind));
    expect(onSelect).toHaveBeenCalledTimes(1);
    expect(trigger).toHaveFocus();
    expect(screen.queryByRole("menu")).not.toBeInTheDocument();
  });

  it("reports nothing when dismissed", async () => {
    const onSelect = vi.fn();
    render(<Harness onSelect={onSelect} />);
    openMenu();

    fireEvent.keyDown(await screen.findByRole("menu"), { key: "Escape" });

    await waitFor(() => expect(screen.queryByRole("menu")).not.toBeInTheDocument());
    expect(onSelect).not.toHaveBeenCalled();
  });
});
