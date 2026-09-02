import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { StudioWorkspaceScaffold } from "@/features/studio/components/StudioWorkspaceScaffold";

describe("StudioWorkspaceScaffold", () => {
  it("keeps the editor first and groups project tools into one secondary rail", () => {
    render(
      <StudioWorkspaceScaffold
        overview={<div>Project overview</div>}
        statsRail={<div>Project status</div>}
        editor={<div>Segment editor</div>}
        actionRail={<div>Project configuration</div>}
        utilities={<div>Pronunciations and export history</div>}
      />,
    );

    const editor = screen.getByTestId("studio-editor-pane");
    const secondary = screen.getByRole("complementary", {
      name: "Project status and configuration",
    });
    const workspaceGrid = editor.parentElement;

    expect(workspaceGrid).toHaveClass(
      "xl:grid-cols-[minmax(0,1fr)_minmax(18rem,20rem)]",
      "min-[1800px]:grid-cols-[240px_minmax(0,1fr)_360px]",
    );
    expect(editor).toHaveClass("order-1", "min-w-0", "xl:col-start-1");
    expect(secondary).toHaveClass(
      "order-2",
      "xl:col-start-2",
      "xl:max-h-[calc(100dvh-2rem)]",
      "xl:overflow-y-auto",
      "min-[1800px]:contents",
      "min-[1800px]:space-y-0",
    );
    expect(screen.getByRole("region", { name: "Project status" })).toBeVisible();
    expect(
      screen.getByRole("region", { name: "Project configuration" }),
    ).toBeVisible();
    expect(editor.compareDocumentPosition(secondary)).toBe(
      Node.DOCUMENT_POSITION_FOLLOWING,
    );
  });

  it("does not create a secondary rail when no project tools are supplied", () => {
    render(
      <StudioWorkspaceScaffold
        overview={<div>Project overview</div>}
        editor={<div>Segment editor</div>}
      />,
    );

    expect(screen.queryByTestId("studio-secondary-pane")).not.toBeInTheDocument();
    expect(screen.getByTestId("studio-editor-pane")).toHaveClass("min-w-0");
  });
});
