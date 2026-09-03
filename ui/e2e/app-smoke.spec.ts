import { expect, test } from "@playwright/test";
import { stubBootstrapRequests } from "./support/api";

test.beforeEach(async ({ page }) => {
  await stubBootstrapRequests(page);
});

test("renders canonical management routes at supported viewports", async ({
  page,
}) => {
  await page.goto("/models");
  await expect(page.getByRole("heading", { name: "Models", level: 1 })).toBeVisible();
  await expect(page).toHaveURL(/\/models$/);

  await page.goto("/settings");
  await expect(
    page.getByRole("heading", { name: "Settings", level: 1 }),
  ).toBeVisible();
  await expect(page).toHaveURL(/\/settings$/);
});

test("keeps the application shell within the viewport", async ({ page }) => {
  await page.goto("/models");

  const viewportWidth = await page.evaluate(() => window.innerWidth);
  const documentWidth = await page.evaluate(
    () => document.documentElement.scrollWidth,
  );
  expect(documentWidth).toBeLessThanOrEqual(viewportWidth);

  if (viewportWidth < 1024) {
    await expect(page.getByRole("heading", { name: "Izwi" })).toBeVisible();
  } else {
    await expect(page.getByRole("link", { name: "Settings" })).toBeVisible();
  }
});
