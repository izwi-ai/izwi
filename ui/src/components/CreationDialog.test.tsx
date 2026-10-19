import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import { CreationDialog } from "@/components/CreationDialog";

function renderDialog(
  props: Partial<Parameters<typeof CreationDialog>[0]> = {},
) {
  const onClose = vi.fn();
  const onSubmit = vi.fn();
  render(
    <CreationDialog
      open
      onClose={onClose}
      title="New thing"
      description="Make a thing."
      onSubmit={onSubmit}
      submitLabel="Create"
      footerStart="12 characters"
      {...props}
    >
      <textarea aria-label="Body" />
    </CreationDialog>,
  );
  return { onClose, onSubmit };
}

describe("CreationDialog", () => {
  it("submits from the footer and with Cmd/Ctrl+Enter", () => {
    const { onSubmit } = renderDialog();

    expect(screen.getByRole("dialog", { name: "New thing" })).toBeVisible();
    expect(screen.getByText("12 characters")).toBeVisible();

    fireEvent.click(screen.getByRole("button", { name: /Create/ }));
    fireEvent.keyDown(screen.getByRole("textbox", { name: "Body" }), {
      key: "Enter",
      ctrlKey: true,
    });
    fireEvent.keyDown(screen.getByRole("textbox", { name: "Body" }), {
      key: "Enter",
    });

    expect(onSubmit).toHaveBeenCalledTimes(2);
  });

  it("does not submit while submitting and closes from Cancel", () => {
    const { onSubmit, onClose } = renderDialog({ submitting: true });

    expect(screen.getByRole("button", { name: /Create/ })).toBeDisabled();
    fireEvent.keyDown(screen.getByRole("textbox", { name: "Body" }), {
      key: "Enter",
      metaKey: true,
    });
    expect(onSubmit).not.toHaveBeenCalled();

    fireEvent.click(screen.getByRole("button", { name: "Cancel" }));
    expect(onClose).toHaveBeenCalledTimes(1);
  });

  it("ignores Escape while dismissal is blocked", () => {
    const { onClose } = renderDialog({ blockDismiss: true });

    fireEvent.keyDown(screen.getByRole("dialog"), { key: "Escape" });

    expect(onClose).not.toHaveBeenCalled();
    expect(screen.getByRole("button", { name: "Cancel" })).toBeDisabled();
  });
});
