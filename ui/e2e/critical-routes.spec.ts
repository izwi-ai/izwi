import { expect, test } from "@playwright/test";

import { READY_MODELS, stubBootstrapRequests } from "./support/api";

test.beforeEach(async ({ page }) => {
  await stubBootstrapRequests(page, { models: READY_MODELS });
});

test("keeps canonical creation entry points reachable", async ({ page }) => {
  const routes = [
    { path: "/text-to-speech", heading: "Text to Speech" },
    { path: "/transcription", heading: "Transcription" },
    { path: "/studio", heading: "Studio" },
    { path: "/voice", text: "Start Conversation" },
  ] as const;

  for (const route of routes) {
    await page.goto(route.path);
    if ("heading" in route) {
      await expect(
        page.getByRole("heading", { name: route.heading, level: 1 }),
      ).toBeVisible();
    } else {
      await expect(page.getByRole("button", { name: route.text })).toBeVisible();
    }
    await expect(page).toHaveURL(new RegExp(`${route.path.replaceAll("/", "\\/")}$`));
  }
});

test("no-model voice state opens the model setup surface and returns focus", async ({
  page,
}) => {
  await page.unroute("**/v1/admin/models");
  await stubBootstrapRequests(page, { models: [] });
  await page.goto("/voice");

  const start = page.getByRole("button", { name: "Start Conversation" });
  const setup = page.getByRole("button", { name: "Set up voice models" });
  await expect(start).toBeDisabled();
  await expect(setup).toBeVisible();

  await setup.click();
  const dialog = page.getByRole("dialog", { name: "Voice configuration" });
  await expect(dialog).toBeVisible();
  await expect(page.getByRole("tab", { name: /^Models/ })).toHaveAttribute(
    "data-state",
    "active",
  );

  await dialog.getByRole("button", { name: "Close" }).click();
  await expect(setup).toBeFocused();
});

test("completes quick onboarding without leaving the blocking dialog", async ({
  page,
}) => {
  await page.unroute("**/v1/onboarding");
  await stubBootstrapRequests(page, {
    models: READY_MODELS,
    onboardingCompleted: false,
  });
  await page.route("**/v1/onboarding/complete", async (route) => {
    await route.fulfill({
      json: { completed: true, completed_at: 2, analytics_opt_in: false },
    });
  });
  await page.route("**/v1/preferences/analytics", async (route) => {
    await route.fulfill({ json: { analytics_opt_in: false } });
  });

  await page.goto("/models");
  await expect(page.getByRole("dialog", { name: "Welcome to Izwi" })).toBeVisible();
  await page.getByRole("button", { name: "Next" }).click();
  await expect(page.getByRole("button", { name: "Quick setup" })).toBeVisible();
  await page.getByRole("button", { name: "Next" }).click();
  await expect(page.getByRole("heading", { name: "All setup" })).toBeVisible();
  await page.getByRole("button", { name: "Go to app" }).click();
  await expect(page.getByRole("dialog")).toHaveCount(0);
  await expect(page.getByRole("heading", { name: "Models", level: 1 })).toBeVisible();
});

test("supports custom onboarding and model-catalog error recovery", async ({
  page,
}) => {
  await page.unroute("**/v1/admin/models");
  await page.unroute("**/v1/onboarding");
  let modelRequestCount = 0;
  await page.route("**/v1/admin/models", async (route) => {
    modelRequestCount += 1;
    if (modelRequestCount === 1) {
      await route.fulfill({ status: 503, json: { error: "Model service unavailable" } });
      return;
    }
    await route.fulfill({ json: { models: READY_MODELS } });
  });
  await page.route("**/v1/onboarding", async (route) => {
    await route.fulfill({
      json: { completed: false, completed_at: null, analytics_opt_in: false },
    });
  });

  await page.goto("/models");
  const onboarding = page.getByRole("dialog", { name: "Welcome to Izwi" });
  await expect(
    onboarding.getByText("Model service unavailable", { exact: true }),
  ).toBeVisible();
  await onboarding.getByRole("button", { name: "Retry models" }).click();
  await expect(
    onboarding.getByText("Model service unavailable", { exact: true }),
  ).toHaveCount(0);

  await page.getByRole("button", { name: "Next" }).click();
  await page.getByRole("button", { name: "Custom setup" }).click();
  await expect(page.getByText("Choose your models")).toBeVisible();
  await expect(page.getByRole("checkbox").first()).toBeVisible();
});

test("recovers to an idle voice entry point when microphone permission fails", async ({
  page,
}) => {
  await page.addInitScript(() => {
    Object.defineProperty(navigator, "mediaDevices", {
      configurable: true,
      value: {
        getUserMedia: async () => {
          throw new Error("Microphone permission denied");
        },
      },
    });
  });
  await page.goto("/voice");

  const start = page.getByRole("button", { name: "Start Conversation" });
  await expect(start).toBeEnabled();
  await start.click();
  await expect(
    page.getByRole("alert").getByText("Microphone permission denied", {
      exact: true,
    }),
  ).toBeVisible();
  await expect(start).toBeEnabled();
});
