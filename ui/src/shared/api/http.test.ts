import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { API_KEY_COOKIE, resetApiKeyStateForTests } from "@/shared/api/apiKey";
import { ApiHttpClient, UnauthorizedError } from "@/shared/api/http";

function jsonResponse(status: number, body: unknown): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

describe("ApiHttpClient API key handling", () => {
  beforeEach(() => {
    resetApiKeyStateForTests();
    document.cookie = `${API_KEY_COOKIE}=; Path=/; Max-Age=0`;
  });

  afterEach(() => {
    vi.unstubAllGlobals();
    vi.restoreAllMocks();
  });

  it("asks for the key once on 401, stores it in a cookie, and retries", async () => {
    const fetchMock = vi
      .fn()
      .mockResolvedValueOnce(jsonResponse(401, { error: { message: "no key" } }))
      .mockResolvedValueOnce(jsonResponse(200, { ok: true }));
    vi.stubGlobal("fetch", fetchMock);
    const prompt = vi.spyOn(window, "prompt").mockReturnValue(" key;with=chars ");

    const result = await new ApiHttpClient("/v1").request<{ ok: boolean }>(
      "/admin/models",
    );

    expect(result).toEqual({ ok: true });
    expect(prompt).toHaveBeenCalledTimes(1);
    expect(fetchMock).toHaveBeenCalledTimes(2);
    expect(document.cookie).toContain(
      `${API_KEY_COOKIE}=${encodeURIComponent("key;with=chars")}`,
    );
  });

  it("stops prompting when the entered key is rejected too", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => jsonResponse(401, { error: { message: "no key" } })),
    );
    const prompt = vi.spyOn(window, "prompt").mockReturnValue("wrong-key");
    const client = new ApiHttpClient("/v1");

    await expect(client.request("/admin/models")).rejects.toBeInstanceOf(
      UnauthorizedError,
    );
    await expect(client.request("/admin/models")).rejects.toBeInstanceOf(
      UnauthorizedError,
    );
    expect(prompt).toHaveBeenCalledTimes(1);
  });

  it("does not prompt again after the user declines", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => jsonResponse(401, { error: { message: "no key" } })),
    );
    const prompt = vi.spyOn(window, "prompt").mockReturnValue(null);
    const client = new ApiHttpClient("/v1");

    await expect(client.request("/admin/models")).rejects.toBeInstanceOf(
      UnauthorizedError,
    );
    await expect(client.request("/admin/models")).rejects.toBeInstanceOf(
      UnauthorizedError,
    );
    expect(prompt).toHaveBeenCalledTimes(1);
  });
});
