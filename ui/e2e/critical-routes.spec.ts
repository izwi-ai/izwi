import { expect, test, type Page } from "@playwright/test";

import {
  NOT_DOWNLOADED_MODELS,
  READY_MODELS,
  stubBootstrapRequests,
} from "./support/api";

test.beforeEach(async ({ page }) => {
  await stubBootstrapRequests(page, { models: READY_MODELS });
});

test("keeps canonical creation entry points reachable", async ({ page }) => {
  const routes = [
    { path: "/text-to-speech", heading: "Text to Speech" },
    { path: "/transcription", heading: "Transcription" },
    { path: "/studio", heading: "Studio" },
    { path: "/voice", text: "Start talking" },
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

test("voice without its models offers setup and keeps settings reachable", async ({
  page,
}) => {
  await page.unroute("**/v1/admin/models");
  await stubBootstrapRequests(page, { models: [] });
  await page.goto("/voice");

  await expect(page.getByRole("heading", { name: "Set up voice", level: 1 })).toBeVisible();
  await expect(page.getByRole("button", { name: /^Download and set up/ })).toBeVisible();
  await expect(page.getByRole("button", { name: "Start talking" })).toHaveCount(0);

  const settings = page.getByRole("button", { name: "Voice settings" });
  await settings.click();
  const sheet = page.getByRole("dialog", { name: "Voice settings" });
  await expect(sheet).toBeVisible();

  await sheet.getByRole("button", { name: "Close" }).click();
  await expect(sheet).toHaveCount(0);
  await expect(settings).toBeFocused();
});

async function stubFirstRun(page: Page, models: readonly object[]) {
  await page.unroute("**/v1/onboarding");
  await stubBootstrapRequests(page, { models, onboardingCompleted: false });
  const completions: string[] = [];
  await page.route("**/v1/onboarding/complete", async (route) => {
    completions.push(route.request().method());
    await route.fulfill({
      json: { completed: true, completed_at: 2, analytics_opt_in: false },
    });
  });
  await page.route("**/v1/preferences/analytics", async (route) => {
    await route.fulfill({ json: { analytics_opt_in: false } });
  });
  return completions;
}

test("first run fills the window, downloads the picks and opens the app", async ({
  page,
}) => {
  const downloads: string[] = [];
  let models: readonly object[] = NOT_DOWNLOADED_MODELS;
  const completions = await stubFirstRun(page, models);
  await page.unroute("**/v1/admin/models");
  await page.route("**/v1/admin/models", async (route) => {
    await route.fulfill({ json: { models } });
  });
  await page.route("**/v1/admin/models/*/download", async (route) => {
    const [variant] = new URL(route.request().url()).pathname.split("/").slice(-2);
    downloads.push(variant);
    models = models.map((model) =>
      (model as { variant: string }).variant === variant
        ? { ...model, status: "downloading", download_progress: 10 }
        : model,
    );
    await route.fulfill({ json: { status: "started", message: "Started" } });
  });

  await page.goto("/models");
  await expect(page.getByRole("heading", { name: "Welcome to Izwi", level: 1 })).toBeVisible();
  await expect(page.getByTestId("app-sidebar")).toHaveCount(0);
  await expect(page.getByRole("button", { name: "Get started" })).toBeFocused();

  await page.getByRole("button", { name: "Get started" }).click();
  await expect(page.getByRole("heading", { name: /^Set up Izwi for this/ })).toBeVisible();
  await expect(page.getByRole("checkbox", { name: "Talk to Izwi" })).toBeChecked();
  await expect(page.getByText("Needs 6.3 GB of free disk space.")).toBeVisible();
  await page.getByRole("button", { name: "Download and set up · 6.3 GB" }).click();

  await expect(page.getByRole("heading", { name: "Setting up Izwi" })).toBeVisible();
  await expect(page.getByRole("button", { name: "Skip setup" })).toHaveCount(0);
  expect(downloads.sort()).toEqual(
    ["Kokoro-82M", "Parakeet-TDT-0.6B-v3", "Qwen3.5-4B"],
  );
  await page.getByRole("button", { name: "Start using Izwi" }).click();

  await expect(page.getByRole("heading", { name: "Models", level: 1 })).toBeVisible();
  await expect(page.getByTestId("app-sidebar")).toHaveCount(1);
  expect(completions).toEqual(["POST"]);
});

test("first run says when the engine isn't responding and lets you choose models", async ({
  page,
}) => {
  await stubFirstRun(page, READY_MODELS);
  await page.unroute("**/v1/onboarding");
  let engineUp = false;
  await page.route("**/v1/onboarding", async (route) => {
    if (!engineUp) {
      await route.fulfill({ status: 503, json: { error: { message: "Starting up" } } });
      return;
    }
    await route.fulfill({
      json: { completed: false, completed_at: null, analytics_opt_in: false },
    });
  });

  await page.goto("/models");
  await expect(
    page.getByRole("heading", { name: "Izwi's engine isn't responding" }),
  ).toBeVisible();
  await expect(page.getByText("Starting up", { exact: true })).toBeVisible();
  // A browser has no log folder to open.
  await expect(page.getByRole("button", { name: "Open the log folder" })).toHaveCount(0);

  engineUp = true;
  await page.getByRole("button", { name: "Try again now" }).click();
  await page.getByRole("button", { name: "Get started" }).click();
  await page.getByRole("button", { name: "Choose models myself" }).click();

  await expect(page.getByRole("heading", { name: "Choose models", level: 1 })).toBeVisible();
  const speech = page.getByRole("region", { name: "Speech to text" });
  await expect(speech.getByRole("checkbox", { name: "Parakeet v3" })).toBeChecked();
  await expect(speech.getByText("Downloaded")).toBeVisible();
  await page.getByRole("button", { name: "Set up" }).click();

  await expect(page.getByRole("heading", { name: "You're all set" })).toBeVisible();
  await page.getByRole("link", { name: "Chat" }).click();
  await expect(page).toHaveURL(/\/chat$/);
});

test("recovers to an idle voice entry point when microphone permission fails", async ({
  page,
}) => {
  await page.addInitScript(() => {
    Object.defineProperty(navigator, "mediaDevices", {
      configurable: true,
      value: {
        getUserMedia: async () => {
          throw new DOMException("Permission denied", "NotAllowedError");
        },
      },
    });
  });
  await page.goto("/voice");

  const start = page.getByRole("button", { name: "Start talking" });
  await expect(start).toBeEnabled();
  await start.click();
  await expect(
    page.getByRole("alert").getByRole("heading", { name: "Microphone is blocked" }),
  ).toBeVisible();
  await expect(page.getByRole("button", { name: "Try again" })).toBeEnabled();
});
