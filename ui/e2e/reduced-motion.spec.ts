import { expect, test } from "@playwright/test";

test.use({ reducedMotion: "reduce" });

test("disables nonessential CSS movement while preserving state changes", async ({
  page,
}) => {
  await page.emulateMedia({ reducedMotion: "reduce" });
  await page.goto("/");

  const reducedMotion = await page.evaluate(async () => {
    const durationInMilliseconds = (duration: string) =>
      duration.endsWith("ms")
        ? Number.parseFloat(duration)
        : Number.parseFloat(duration) * 1000;
    const element = document.createElement("div");
    element.className = "animate-pulse opacity-100 transition-opacity duration-300";
    document.body.append(element);

    const initialStyles = window.getComputedStyle(element);
    const animationDuration = durationInMilliseconds(
      initialStyles.animationDuration,
    );
    const transitionDuration = durationInMilliseconds(
      initialStyles.transitionDuration,
    );

    element.classList.replace("opacity-100", "opacity-0");
    await new Promise<void>((resolve) =>
      window.requestAnimationFrame(() => resolve()),
    );
    const opacity = window.getComputedStyle(element).opacity;
    element.remove();

    return {
      animationDuration,
      opacity,
      preferenceMatches: window.matchMedia(
        "(prefers-reduced-motion: reduce)",
      ).matches,
      transitionDuration,
    };
  });

  expect(reducedMotion.preferenceMatches).toBe(true);
  expect(reducedMotion.animationDuration).toBeLessThanOrEqual(0.01);
  expect(reducedMotion.transitionDuration).toBeLessThanOrEqual(0.01);
  expect(reducedMotion.opacity).toBe("0");
});
