import { API_BASE_URL } from "@/shared/config/runtime";

export const API_BASE = API_BASE_URL;

export class ApiHttpClient {
  readonly baseUrl: string;
  private readonly defaultTimeoutMs: number;

  constructor(baseUrl: string = API_BASE, defaultTimeoutMs: number = 15_000) {
    this.baseUrl = baseUrl;
    this.defaultTimeoutMs = defaultTimeoutMs;
  }

  url(path: string): string {
    return `${this.baseUrl}${path}`;
  }

  async request<T>(path: string, options?: RequestInit): Promise<T> {
    // Requests to the local model service must never hang the UI indefinitely:
    // a slow first /admin/models call otherwise leaves the catalog spinner up
    // forever. A caller-supplied signal wins; otherwise we bound the wait.
    const controller = new AbortController();
    const signal = options?.signal ?? controller.signal;
    const timeout =
      options?.signal === undefined
        ? setTimeout(() => controller.abort(), this.defaultTimeoutMs)
        : undefined;

    let response: Response;
    try {
      response = await fetch(this.url(path), {
        ...options,
        signal,
        headers: {
          "Content-Type": "application/json",
          ...options?.headers,
        },
      });
    } catch (err) {
      if (isAbortError(err)) {
        throw new Error("The local model service took too long to respond.");
      }
      throw err;
    } finally {
      if (timeout !== undefined) {
        clearTimeout(timeout);
      }
    }

    if (!response.ok) {
      throw await this.createError(response, "Request failed");
    }

    return response.json();
  }

  async createError(response: Response, fallbackMessage: string): Promise<Error> {
    const error = await response
      .json()
      .catch(() => ({ error: { message: fallbackMessage } }));

    return new Error(error.error?.message || fallbackMessage);
  }
}

export async function consumeDataStream(
  response: Response,
  onData: (data: string) => boolean | void | Promise<boolean | void>,
) {
  const reader = response.body?.getReader();
  if (!reader) {
    throw new Error("No response body");
  }

  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) {
      break;
    }

    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split("\n");
    buffer = lines.pop() || "";

    for (const line of lines) {
      if (!line.startsWith("data:")) {
        continue;
      }

      const data = line.slice(5).trim();
      if (!data) {
        continue;
      }

      const shouldStop = await onData(data);
      if (shouldStop) {
        return;
      }
    }
  }
}

export function isAbortError(error: unknown): boolean {
  return error instanceof Error && error.name === "AbortError";
}
