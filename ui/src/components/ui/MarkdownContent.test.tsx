import { fireEvent, render, screen } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { MarkdownContent } from "@/components/ui/MarkdownContent";

describe("MarkdownContent", () => {
  const writeText = vi.fn<(text: string) => Promise<void>>();

  beforeEach(() => {
    writeText.mockReset();
    writeText.mockResolvedValue(undefined);
    Object.defineProperty(navigator, "clipboard", {
      configurable: true,
      value: { writeText },
    });
  });

  it("labels fenced code with its language and copies only the code", async () => {
    render(
      <MarkdownContent
        content={"Use this:\n\n```rust\nfn main() {}\n```\n\nThen run it."}
      />,
    );

    expect(screen.getByText("rust")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Copy" }));

    expect(await screen.findByRole("button", { name: "Copied" })).toBeVisible();
    expect(writeText).toHaveBeenCalledWith("fn main() {}\n");
  });

  it("names unlabelled blocks as code", () => {
    render(<MarkdownContent content={"```\nplain\n```"} />);

    expect(screen.getByText("code")).toBeInTheDocument();
  });
});
