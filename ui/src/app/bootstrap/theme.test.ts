import { readFileSync } from "node:fs";
import { resolve } from "node:path";

import { describe, expect, it } from "vitest";
import {
  THEME_STORAGE_KEY,
  applyDocumentTheme,
  getStoredThemePreference,
  getSystemTheme,
  resolveThemePreference,
} from "./theme";

describe("theme bootstrap", () => {
  it("falls back to system when storage is missing or invalid", () => {
    expect(getStoredThemePreference(null)).toBe("system");
    expect(
      getStoredThemePreference({
        getItem: () => "unexpected",
      }),
    ).toBe("system");
  });

  it("returns stored theme preference when it is valid", () => {
    expect(
      getStoredThemePreference({
        getItem: (key) => (key === THEME_STORAGE_KEY ? "dark" : null),
      }),
    ).toBe("dark");
  });

  it("resolves system and explicit theme preferences correctly", () => {
    expect(getSystemTheme(true)).toBe("dark");
    expect(getSystemTheme(false)).toBe("light");
    expect(resolveThemePreference("system", "dark")).toBe("dark");
    expect(resolveThemePreference("light", "dark")).toBe("light");
  });

  it("applies the resolved theme to the document root", () => {
    const root = document.documentElement;

    applyDocumentTheme("dark", root);
    expect(root.classList.contains("theme-dark")).toBe(true);
    expect(root.style.colorScheme).toBe("dark");

    applyDocumentTheme("light", root);
    expect(root.classList.contains("theme-light")).toBe(true);
    expect(root.style.colorScheme).toBe("light");
  });

  it("keeps the pre-paint theme script on the same storage key", () => {
    // Vitest runs from the ui package root.
    const script = readFileSync(resolve("public/theme-init.js"), "utf8");
    expect(script).toContain(`"${THEME_STORAGE_KEY}"`);
  });
});
