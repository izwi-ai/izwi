// Applies the saved theme before the app bundle loads, so light-mode users
// do not see the dark default flash first. Mirrors bootstrapDocumentTheme in
// src/app/bootstrap/theme.ts; keep the storage key in sync with
// THEME_STORAGE_KEY there. Loaded as a same-origin file so the desktop
// Content Security Policy (script-src 'self') allows it without a hash.
(function () {
  var dark = true;
  try {
    var stored = window.localStorage.getItem("izwi.theme.preference");
    dark =
      stored === "dark" ||
      (stored !== "light" &&
        window.matchMedia("(prefers-color-scheme: dark)").matches);
  } catch (error) {
    // Storage or matchMedia unavailable: keep the dark default.
  }
  var root = document.documentElement;
  root.classList.add(dark ? "theme-dark" : "theme-light");
  root.style.colorScheme = dark ? "dark" : "light";
})();
