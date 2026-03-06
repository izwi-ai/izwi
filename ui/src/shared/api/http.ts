const tauriServerUrl =
  typeof window !== "undefined"
    ? ((window as { __IZWI_SERVER_URL__?: string }).__IZWI_SERVER_URL__ ?? null)
    : null;

export const API_BASE = tauriServerUrl
  ? `${tauriServerUrl.replace(/\/$/, "")}/v1`
  : "/v1";

export class ApiHttpClient {
  readonly baseUrl: string;

  constructor(baseUrl: string = API_BASE) {
    this.baseUrl = baseUrl;
  }

  url(path: string): string {
    return `${this.baseUrl}${path}`;
  }

  async request<T>(path: string, options?: RequestInit): Promise<T> {
    const response = await fetch(this.url(path), {
      ...options,
      headers: {
        "Content-Type": "application/json",
        ...options?.headers,
      },
    });

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
