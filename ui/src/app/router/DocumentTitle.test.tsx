import { render } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { describe, expect, it } from "vitest";

import { DocumentTitle, usePageTitle } from "@/app/router/DocumentTitle";

function Record({ title }: { title: string }) {
  usePageTitle(title);
  return null;
}

describe("DocumentTitle", () => {
  it("uses a page's own title while it is mounted, then the route's", () => {
    const { rerender } = render(
      <MemoryRouter initialEntries={["/text-to-speech/abc"]}>
        <DocumentTitle />
        <Record title="Izwi redesign test take." />
      </MemoryRouter>,
    );
    expect(document.title).toBe("Izwi redesign test take. · Izwi");

    rerender(
      <MemoryRouter initialEntries={["/text-to-speech/abc"]}>
        <DocumentTitle />
      </MemoryRouter>,
    );
    expect(document.title).toBe("Text-to-Speech Record · Izwi");
  });
});
