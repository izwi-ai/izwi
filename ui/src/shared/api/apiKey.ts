// A server bound to a network address requires its IZWI_API_KEY. The web UI
// asks for it once and keeps it in a same-site cookie, which the browser then
// sends on every same-origin request, including EventSource, WebSocket and
// media URLs that cannot carry an Authorization header.

export const API_KEY_COOKIE = "izwi_api_key";

let storedKeyVersion = 0;
let declined = false;

/** Increments whenever a new key is stored; lets concurrent 401s retry
 * without prompting again. */
export function apiKeyVersion(): number {
  return storedKeyVersion;
}

export function storeApiKey(key: string): void {
  const secure = window.location.protocol === "https:" ? "; Secure" : "";
  document.cookie =
    `${API_KEY_COOKIE}=${encodeURIComponent(key)}; Path=/; ` +
    `Max-Age=31536000; SameSite=Strict${secure}`;
  storedKeyVersion += 1;
}

/**
 * Handle a 401 for a request issued at `versionAtRequest`. Returns true when
 * the request should be retried: a key was stored meanwhile, or the user just
 * entered one. A declined prompt is not repeated for this page load.
 */
export function resolveUnauthorized(versionAtRequest: number): boolean {
  if (storedKeyVersion !== versionAtRequest) {
    return true;
  }
  if (declined || typeof window === "undefined" || !window.prompt) {
    return false;
  }
  const key = window.prompt(
    "This Izwi server requires an API key. Enter the value of IZWI_API_KEY:",
  );
  if (!key || !key.trim()) {
    declined = true;
    return false;
  }
  storeApiKey(key.trim());
  return true;
}

export function stopApiKeyPrompts(): void {
  declined = true;
}

export function resetApiKeyStateForTests(): void {
  storedKeyVersion = 0;
  declined = false;
}
