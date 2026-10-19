import { render, screen } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { RuntimeCard } from "@/app/layouts/RuntimeCard";

const apiMocks = vi.hoisted(() => ({ getHealth: vi.fn() }));

vi.mock("@/api", () => ({ api: apiMocks }));

const GIB = 1024 ** 3;

describe("RuntimeCard", () => {
  beforeEach(() => {
    apiMocks.getHealth.mockReset();
  });

  it("shows the backend, loaded models and reserved memory", async () => {
    apiMocks.getHealth.mockResolvedValue({
      status: "ok",
      version: "0.1.0",
      runtime: {
        selected_backend: "metal",
        memory: {
          system_total_bytes: 32 * GIB,
          system_available_bytes: 20 * GIB,
          reserved_bytes: 2.1 * GIB,
        },
      },
    });

    render(
      <RuntimeCard
        readyModelsCount={1}
        catalogError={null}
        collapsed={false}
      />,
    );

    expect(await screen.findByText("Runtime ready")).toBeVisible();
    expect(screen.getByText("Metal")).toBeVisible();
    expect(screen.getByText("1 model · 2.1 GB")).toBeVisible();
    expect(
      screen.getByRole("meter", { name: "Memory reserved by models" }),
    ).toHaveAttribute("aria-valuenow", "7");
  });

  it("reports the runtime offline when health or the catalog fails", async () => {
    apiMocks.getHealth.mockRejectedValue(new Error("offline"));

    render(
      <RuntimeCard
        readyModelsCount={0}
        catalogError={null}
        collapsed={false}
      />,
    );

    expect(await screen.findByText("Runtime offline")).toBeVisible();
    expect(screen.getByText("0 models")).toBeVisible();
    expect(screen.queryByRole("meter")).not.toBeInTheDocument();
  });
});
