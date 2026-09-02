import { expect, test, type Page } from "@playwright/test";

async function stubBootstrapRequests(page: Page) {
  await page.route("**/v1/admin/models", async (route) => {
    await route.fulfill({ json: { models: [] } });
  });
  await page.route("**/v1/onboarding", async (route) => {
    await route.fulfill({
      json: {
        completed: true,
        completed_at: 1,
        analytics_opt_in: false,
      },
    });
  });
  await page.route("**/v1/preferences", async (route) => {
    await route.fulfill({ json: { analytics_opt_in: false } });
  });
}

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
