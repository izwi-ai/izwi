import {
  createContext,
  useContext,
  useEffect,
  useMemo,
  useState,
  type ReactNode,
} from "react";
import {
  THEME_STORAGE_KEY,
  applyDocumentTheme,
  getCurrentSystemTheme,
  getStoredThemePreference,
  resolveThemePreference,
  type ResolvedTheme,
  type ThemePreference,
} from "@/app/bootstrap/theme";

interface ThemeContextValue {
  themePreference: ThemePreference;
  resolvedTheme: ResolvedTheme;
  setThemePreference: (preference: ThemePreference) => void;
}

const ThemeContext = createContext<ThemeContextValue | null>(null);

interface ThemeProviderProps {
  children: ReactNode;
}

export function ThemeProvider({ children }: ThemeProviderProps) {
  const [themePreference, setThemePreferenceState] = useState<ThemePreference>(
    () =>
      getStoredThemePreference(
        typeof window === "undefined" ? null : window.localStorage,
      ),
  );
  const [resolvedTheme, setResolvedTheme] = useState<ResolvedTheme>(() =>
    resolveThemePreference(themePreference, getCurrentSystemTheme()),
  );

  useEffect(() => {
    if (typeof window === "undefined") {
      return;
    }

    const mediaQuery = window.matchMedia("(prefers-color-scheme: dark)");

    const syncResolvedTheme = () => {
      const nextResolvedTheme = resolveThemePreference(
        themePreference,
        getCurrentSystemTheme(),
      );

      setResolvedTheme(nextResolvedTheme);
      applyDocumentTheme(nextResolvedTheme);
    };

    syncResolvedTheme();

    const onSystemThemeChange = () => {
      if (themePreference === "system") {
        syncResolvedTheme();
      }
    };

    mediaQuery.addEventListener("change", onSystemThemeChange);

    return () => {
      mediaQuery.removeEventListener("change", onSystemThemeChange);
    };
  }, [themePreference]);

  useEffect(() => {
    if (typeof window === "undefined") {
      return;
    }

    window.localStorage.setItem(THEME_STORAGE_KEY, themePreference);
  }, [themePreference]);

  const value = useMemo<ThemeContextValue>(
    () => ({
      themePreference,
      resolvedTheme,
      setThemePreference: setThemePreferenceState,
    }),
    [resolvedTheme, themePreference],
  );

  return <ThemeContext.Provider value={value}>{children}</ThemeContext.Provider>;
}

export function useTheme() {
  const context = useContext(ThemeContext);

  if (!context) {
    throw new Error("useTheme must be used within ThemeProvider");
  }

  return context;
}
