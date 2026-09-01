import { afterEach, describe, expect, it, vi } from "vitest";

import { ChatApiClient } from "@/shared/api/chat";
import { ApiHttpClient } from "@/shared/api/http";

function sseResponse(events: Array<Record<string, unknown> | string>): Response {
  const encoder = new TextEncoder();
  return new Response(
    new ReadableStream({
      start(controller) {
        for (const event of events) {
          const payload =
            typeof event === "string" ? event : JSON.stringify(event);
          controller.enqueue(encoder.encode(`data: ${payload}\n\n`));
        }
        controller.close();
      },
    }),
    {
      status: 200,
      headers: {
        "Content-Type": "text/event-stream",
      },
    },
  );
}

function controlledSseResponse() {
  const encoder = new TextEncoder();
  let streamController!: ReadableStreamDefaultController<Uint8Array>;
  const response = new Response(
    new ReadableStream<Uint8Array>({
      start(controller) {
        streamController = controller;
      },
    }),
    {
      status: 200,
      headers: {
        "Content-Type": "text/event-stream",
      },
    },
  );

  return {
    response,
    send(event: Record<string, unknown> | string) {
      const payload = typeof event === "string" ? event : JSON.stringify(event);
      streamController.enqueue(encoder.encode(`data: ${payload}\n\n`));
    },
    close() {
      streamController.close();
    },
    fail(message = "late transport failure") {
      streamController.error(new Error(message));
    },
  };
}

function chatChunk(content: string): Record<string, unknown> {
  return {
    id: "chatcmpl-1",
    model: "LFM2.5-1.2B-Thinking-GGUF",
    choices: [
      {
        index: 0,
        delta: { content },
        finish_reason: null,
      },
    ],
  };
}

describe("ChatApiClient OpenAI streaming", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
    vi.restoreAllMocks();
  });

  it("completes only after the OpenAI done sentinel", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(
        sseResponse([
          chatChunk("Hello "),
          chatChunk("world"),
          {
            id: "chatcmpl-1",
            model: "LFM2.5-1.2B-Thinking-GGUF",
            choices: [
              {
                index: 0,
                delta: {},
                finish_reason: "stop",
              },
            ],
            usage: { completion_tokens: 2 },
            izwi_generation_time_ms: 42,
          },
          "[DONE]",
        ]),
      ),
    );

    const client = new ChatApiClient(
      new ApiHttpClient("http://localhost/v1"),
    );
    const onDelta = vi.fn();
    const onError = vi.fn();
    const result = await new Promise<{
      text: string;
      stats: { tokens_generated: number; generation_time_ms: number };
    }>((resolve) => {
      client.chatCompletionsStream(
        { messages: [{ role: "user", content: "Hello" }] },
        {
          onDelta,
          onError,
          onDone: (text, stats) => resolve({ text, stats }),
        },
      );
    });

    expect(onDelta.mock.calls).toEqual([["Hello "], ["world"]]);
    expect(onError).not.toHaveBeenCalled();
    expect(result).toEqual({
      text: "Hello world",
      stats: { tokens_generated: 2, generation_time_ms: 42 },
    });
  });

  it("delivers a delta while the response body remains open", async () => {
    const stream = controlledSseResponse();
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(stream.response));

    const client = new ChatApiClient(
      new ApiHttpClient("http://localhost/v1"),
    );
    const onDelta = vi.fn();
    const onDone = vi.fn();
    const onError = vi.fn();
    client.chatCompletionsStream(
      { messages: [{ role: "user", content: "Hello" }] },
      { onDelta, onDone, onError },
    );

    stream.send(chatChunk("visible now"));
    await vi.waitFor(() => {
      expect(onDelta).toHaveBeenCalledWith("visible now");
    });
    expect(onDone).not.toHaveBeenCalled();
    expect(onError).not.toHaveBeenCalled();

    stream.send({
      id: "chatcmpl-1",
      model: "LFM2.5-1.2B-Thinking-GGUF",
      choices: [{ index: 0, delta: {}, finish_reason: "stop" }],
    });
    stream.send("[DONE]");
    await vi.waitFor(() => {
      expect(onDone).toHaveBeenCalledOnce();
    });
    stream.close();
  });

  it("surfaces an OpenAI error frame and does not complete on done", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(
        sseResponse([
          { error: { message: "Inference failed", type: "server_error" } },
          chatChunk("late delta"),
          "[DONE]",
        ]),
      ),
    );

    const client = new ChatApiClient(
      new ApiHttpClient("http://localhost/v1"),
    );
    const onDone = vi.fn();
    const onDelta = vi.fn();
    const onError = vi.fn();
    client.chatCompletionsStream(
      { messages: [{ role: "user", content: "Hello" }] },
      { onDelta, onDone, onError },
    );

    await vi.waitFor(() => {
      expect(onError).toHaveBeenCalledWith("Inference failed");
    });
    expect(onError).toHaveBeenCalledTimes(1);
    expect(onDelta).not.toHaveBeenCalled();
    expect(onDone).not.toHaveBeenCalled();
  });

  it("does not duplicate an OpenAI terminal error after a read failure", async () => {
    const stream = controlledSseResponse();
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(stream.response));

    const client = new ChatApiClient(
      new ApiHttpClient("http://localhost/v1"),
    );
    const onDone = vi.fn();
    const onError = vi.fn();
    client.chatCompletionsStream(
      { messages: [{ role: "user", content: "Hello" }] },
      { onDone, onError },
    );

    stream.send({ error: { message: "Inference failed" } });
    await vi.waitFor(() => {
      expect(onError).toHaveBeenCalledWith("Inference failed");
    });
    stream.fail();
    await new Promise((resolve) => setTimeout(resolve, 0));

    expect(onError).toHaveBeenCalledTimes(1);
    expect(onDone).not.toHaveBeenCalled();
  });

  it("reports a truncated stream that closes without done", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(sseResponse([chatChunk("partial response")])),
    );

    const client = new ChatApiClient(
      new ApiHttpClient("http://localhost/v1"),
    );
    const onDelta = vi.fn();
    const onDone = vi.fn();
    const onError = vi.fn();
    client.chatCompletionsStream(
      { messages: [{ role: "user", content: "Hello" }] },
      { onDelta, onDone, onError },
    );

    await vi.waitFor(() => {
      expect(onError).toHaveBeenCalledWith(
        "Chat stream ended before a terminal event",
      );
    });
    expect(onDelta).toHaveBeenCalledWith("partial response");
    expect(onError).toHaveBeenCalledTimes(1);
    expect(onDone).not.toHaveBeenCalled();
  });

  it("rejects done without an OpenAI terminal chunk", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(
        sseResponse([chatChunk("partial response"), "[DONE]"]),
      ),
    );

    const client = new ChatApiClient(
      new ApiHttpClient("http://localhost/v1"),
    );
    const onDone = vi.fn();
    const onError = vi.fn();
    client.chatCompletionsStream(
      { messages: [{ role: "user", content: "Hello" }] },
      { onDone, onError },
    );

    await vi.waitFor(() => {
      expect(onError).toHaveBeenCalledWith(
        "Chat stream ended before a terminal event",
      );
    });
    expect(onDone).not.toHaveBeenCalled();
  });
});

describe("ChatApiClient thread streaming", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
    vi.restoreAllMocks();
  });

  it("forwards Qwen reasoning and sampling controls", async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      new Response(JSON.stringify({}), {
        status: 200,
        headers: { "Content-Type": "application/json" },
      }),
    );
    vi.stubGlobal("fetch", fetchMock);
    const client = new ChatApiClient(
      new ApiHttpClient("http://localhost/v1"),
    );

    await client.sendChatThreadMessage("thread-1", {
      model_id: "Qwen3.8-27B-FP8",
      content: "Hello",
      reasoning_effort: "low",
      preserve_thinking: false,
      top_k: 7,
      repetition_penalty: 1.2,
      presence_penalty: 0.4,
    });

    const init = fetchMock.mock.calls[0]?.[1] as RequestInit;
    expect(JSON.parse(String(init.body))).toMatchObject({
      model: "Qwen3.8-27B-FP8",
      reasoning_effort: "low",
      preserve_thinking: false,
      top_k: 7,
      repetition_penalty: 1.2,
      presence_penalty: 0.4,
    });
  });

  it("does not turn an explicit terminal error into success", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(
        sseResponse([
          { event: "delta", delta: "partial" },
          { event: "error", error: "Inference failed" },
          {
            event: "done",
            thread_id: "thread-1",
            model_id: "test-model",
            assistant_message: {},
            stats: { tokens_generated: 1, generation_time_ms: 1 },
          },
          "[DONE]",
        ]),
      ),
    );

    const client = new ChatApiClient(
      new ApiHttpClient("http://localhost/v1"),
    );
    const onDone = vi.fn();
    const onError = vi.fn();
    const onClose = vi.fn();
    client.sendChatThreadMessageStream(
      "thread-1",
      { content: "Hello" },
      { onDone, onError, onClose },
    );

    await vi.waitFor(() => {
      expect(onClose).toHaveBeenCalledOnce();
    });
    expect(onError).toHaveBeenCalledWith("Inference failed");
    expect(onError).toHaveBeenCalledTimes(1);
    expect(onDone).not.toHaveBeenCalled();
  });

  it("ignores a read failure after a completed thread terminal", async () => {
    const stream = controlledSseResponse();
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(stream.response));

    const client = new ChatApiClient(
      new ApiHttpClient("http://localhost/v1"),
    );
    const onDone = vi.fn();
    const onError = vi.fn();
    const onClose = vi.fn();
    client.sendChatThreadMessageStream(
      "thread-1",
      { content: "Hello" },
      { onDone, onError, onClose },
    );

    stream.send({
      event: "done",
      thread_id: "thread-1",
      model_id: "test-model",
      assistant_message: {},
      stats: { tokens_generated: 1, generation_time_ms: 1 },
    });
    await vi.waitFor(() => {
      expect(onDone).toHaveBeenCalledOnce();
    });
    stream.fail();
    await vi.waitFor(() => {
      expect(onClose).toHaveBeenCalledOnce();
    });

    expect(onError).not.toHaveBeenCalled();
  });

  it("reports closure before a terminal thread event", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(
        sseResponse([{ event: "delta", delta: "partial" }]),
      ),
    );

    const client = new ChatApiClient(
      new ApiHttpClient("http://localhost/v1"),
    );
    const onDone = vi.fn();
    const onError = vi.fn();
    client.sendChatThreadMessageStream(
      "thread-1",
      { content: "Hello" },
      { onDone, onError },
    );

    await vi.waitFor(() => {
      expect(onError).toHaveBeenCalledWith(
        "Chat stream ended before a terminal event",
      );
    });
    expect(onDone).not.toHaveBeenCalled();
  });
});

describe("ChatApiClient Responses streaming", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
    vi.restoreAllMocks();
  });

  it("keeps a failed response terminal after the done sentinel", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(
        sseResponse([
          {
            type: "response.failed",
            response_id: "resp-1",
            error: { message: "Response failed" },
          },
          { type: "response.completed", response: {} },
          "[DONE]",
        ]),
      ),
    );

    const client = new ChatApiClient(
      new ApiHttpClient("http://localhost/v1"),
    );
    const onCompleted = vi.fn();
    const onError = vi.fn();
    const onDone = vi.fn();
    client.createResponseStream(
      { input: "Hello" },
      { onCompleted, onError, onDone },
    );

    await vi.waitFor(() => {
      expect(onDone).toHaveBeenCalledOnce();
    });
    expect(onError).toHaveBeenCalledWith("Response failed");
    expect(onError).toHaveBeenCalledTimes(1);
    expect(onCompleted).not.toHaveBeenCalled();
  });

  it("ignores a read failure after a completed response terminal", async () => {
    const stream = controlledSseResponse();
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(stream.response));

    const client = new ChatApiClient(
      new ApiHttpClient("http://localhost/v1"),
    );
    const onCompleted = vi.fn();
    const onError = vi.fn();
    const onDone = vi.fn();
    client.createResponseStream(
      { input: "Hello" },
      { onCompleted, onError, onDone },
    );

    stream.send({ type: "response.completed", response: {} });
    await vi.waitFor(() => {
      expect(onCompleted).toHaveBeenCalledOnce();
    });
    stream.fail();
    await vi.waitFor(() => {
      expect(onDone).toHaveBeenCalledOnce();
    });

    expect(onError).not.toHaveBeenCalled();
  });

  it("reports closure before a terminal response event", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(
        sseResponse([{ type: "response.output_text.delta", delta: "partial" }]),
      ),
    );

    const client = new ChatApiClient(
      new ApiHttpClient("http://localhost/v1"),
    );
    const onCompleted = vi.fn();
    const onError = vi.fn();
    client.createResponseStream(
      { input: "Hello" },
      { onCompleted, onError },
    );

    await vi.waitFor(() => {
      expect(onError).toHaveBeenCalledWith(
        "Response stream ended before a terminal event",
      );
    });
    expect(onCompleted).not.toHaveBeenCalled();
  });
});
