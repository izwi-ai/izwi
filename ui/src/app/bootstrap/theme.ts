export type ThemePreference = "system" | "light" | "dark";
export type ResolvedTheme = "light" | "dark";

export const THEME_STORAGE_KEY = "izwi.theme.preference";

export function getStoredThemePreference(
  storage: Pick<Storage, "getItem"> | null,
): ThemePreference {
  if (!storage) {
    return "system";
  }

  const stored = storage.getItem(THEME_STORAGE_KEY);
  if (stored === "light" || stored === "dark" || stored === "system") {
    return stored;
  }

  return "system";
}

export function getSystemTheme(prefersDark: boolean): ResolvedTheme {
  return prefersDark ? "dark" : "light";
}

export function resolveThemePreference(
  preference: ThemePreference,
  systemTheme: ResolvedTheme,
): ResolvedTheme {
  return preference === "system" ? systemTheme : preference;
}

export function getCurrentSystemTheme(): ResolvedTheme {
  if (typeof window === "undefined") {
    return "dark";
  }

  return getSystemTheme(
    window.matchMedia("(prefers-color-scheme: dark)").matches,
  );
}

export function applyDocumentTheme(
  resolvedTheme: ResolvedTheme,
  root: HTMLElement = document.documentElement,
) {
  root.classList.remove("theme-light", "theme-dark");
  root.classList.add(
    resolvedTheme === "dark" ? "theme-dark" : "theme-light",
  );
  root.style.colorScheme = resolvedTheme;
}

export function bootstrapDocumentTheme() {
  if (typeof window === "undefined") {
    return;
  }

  let storedPreference: ThemePreference = "system";
  try {
    storedPreference = getStoredThemePreference(window.localStorage);
  } catch {
    // Storage can be blocked (private mode, sandboxing); follow the system.
  }
  const resolvedTheme = resolveThemePreference(
    storedPreference,
    getCurrentSystemTheme(),
  );

  applyDocumentTheme(resolvedTheme);
}
