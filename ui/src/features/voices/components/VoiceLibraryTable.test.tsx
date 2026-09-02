import { fireEvent, render, screen, within } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import { Button } from "@/components/ui/button";

import { VoiceLibraryTable } from "./VoiceLibraryTable";

describe("VoiceLibraryTable", () => {
  it("keeps the voice identity and primary actions together in a responsive row", () => {
    const onUse = vi.fn();
    const onDelete = vi.fn();

    render(
      <VoiceLibraryTable
        items={[
          {
            id: "voice-1",
            name: "A deliberately long saved voice name",
            categoryLabel: "Cloned voice",
            description: "A reusable reference voice.",
            actions: (
              <>
                <Button onClick={onUse}>Use in TTS</Button>
                <Button onClick={onDelete}>Delete</Button>
              </>
            ),
          },
        ]}
        emptyTitle="No voices"
        emptyDescription="Create a voice first."
      />,
    );

    const row = screen.getByTestId("voice-row-voice-1");
    const actions = within(row).getByRole("group", {
      name: "Actions for A deliberately long saved voice name",
    });

    expect(row).toHaveClass("grid", "2xl:table-row");
    expect(screen.getByRole("table")).toHaveClass(
      "block",
      "2xl:min-w-[72rem]",
    );
    expect(screen.getByRole("table")).not.toHaveClass("min-w-[72rem]");

    fireEvent.click(within(actions).getByRole("button", { name: "Use in TTS" }));
    fireEvent.click(within(actions).getByRole("button", { name: "Delete" }));

    expect(onUse).toHaveBeenCalledOnce();
    expect(onDelete).toHaveBeenCalledOnce();
  });
});
