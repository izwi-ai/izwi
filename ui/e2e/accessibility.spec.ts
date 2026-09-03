import AxeBuilder from "@axe-core/playwright";
import { expect, test, type ColorScheme, type Page } from "@playwright/test";

import { READY_MODELS, stubBootstrapRequests } from "./support/api";

const SERIOUS_IMPACTS = new Set(["serious", "critical"]);

async function expectNoSeriousViolations(page: Page) {
  const results = await new AxeBuilder({ page })
    .withTags(["wcag2a", "wcag2aa", "wcag21a", "wcag21aa"])
    .analyze();
  const violations = results.violations.filter((violation) =>
    SERIOUS_IMPACTS.has(violation.impact ?? ""),
  );

  expect(
    violations.map((violation) => ({
      id: violation.id,
      impact: violation.impact,
      nodes: violation.nodes.map((node) => ({
        target: node.target,
        html: node.html,
      })),
    })),
  ).toEqual([]);
}

const THEMES: ColorScheme[] = ["light", "dark"];

test.beforeEach(async ({ page }) => {
  await stubBootstrapRequests(page, { models: READY_MODELS });
});

test("shell and core route entry states have no serious axe violations", async ({
  page,
}) => {
  for (const colorScheme of THEMES) {
    await page.emulateMedia({ colorScheme, reducedMotion: "reduce" });
    for (const path of ["/models", "/chat", "/text-to-speech", "/voice"]) {
      await page.goto(path);
      await expect(page.locator("main")).toBeVisible();
      await expectNoSeriousViolations(page);
    }
  }
});

test("onboarding dialog has no serious axe violations", async ({ page }) => {
  await page.unroute("**/v1/onboarding");
  await stubBootstrapRequests(page, {
    models: READY_MODELS,
    onboardingCompleted: false,
  });
  for (const colorScheme of THEMES) {
    await page.emulateMedia({ colorScheme, reducedMotion: "reduce" });
    await page.goto("/models");
    await expect(page.getByRole("dialog", { name: "Welcome to Izwi" })).toBeVisible();
    await expectNoSeriousViolations(page);
  }
});

test("model controls and creation dialog have no serious axe violations", async ({
  page,
}) => {
  for (const colorScheme of THEMES) {
    await page.emulateMedia({ colorScheme, reducedMotion: "reduce" });
    await page.goto("/text-to-speech");
    await page.getByRole("button", { name: "New generation" }).click();
    await expect(page.getByRole("dialog")).toBeVisible();
    await expectNoSeriousViolations(page);
  }
});
