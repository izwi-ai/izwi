import { afterEach, describe, expect, it, vi } from "vitest";
import {
  isSameCalendarDay,
  messageDayLabel,
  buildChatThreadMessagePayload,
  DEFAULT_THREAD_TITLE,
  defaultThinkingEnabledForModel,
  displayThreadTitle,
  isQwen35ThinkingModel,
  isQwen38ThinkingModel,
  parseAssistantContent,
  parseUserMessageDisplayFromContentParts,
  readChatImage,
  stoppedReplyContent,
  summarizeChatStats,
  supportsImageAttachmentsForModel,
  stripMarkdown,
  supportsImplicitOpenThinkTagParsing,
  systemPromptForModel,
  threadPreviewFromContent,
} from "./support";

describe("chat playground support", () => {
  it("splits assistant reasoning and answer across explicit think tags", () => {
    expect(
      parseAssistantContent("before <think>reasoning</think> after"),
    ).toEqual({
      thinking: "reasoning",
      answer: "before  after",
      hasThink: true,
      hasIncompleteThink: false,
    });
  });

  it("supports implicit open-think parsing for matching models", () => {
    expect(supportsImplicitOpenThinkTagParsing("LFM2.5-1.2B-Thinking-GGUF")).toBe(
      true,
    );
    expect(supportsImplicitOpenThinkTagParsing("Qwen3.5-4B")).toBe(true);
    expect(supportsImplicitOpenThinkTagParsing("Qwen3.8-27B-FP8")).toBe(true);
    expect(supportsImplicitOpenThinkTagParsing("Qwen3.6-35B-A3B-FP8")).toBe(true);
    expect(supportsImplicitOpenThinkTagParsing("LFM2.5-1.2B-Instruct-GGUF")).toBe(
      false,
    );
    expect(
      parseAssistantContent("reasoning first</think>final answer", {
        implicitOpenThinkTag: true,
      }),
    ).toEqual({
      thinking: "reasoning first",
      answer: "final answer",
      hasThink: true,
      hasIncompleteThink: false,
    });
  });

  it("uses variant-aware thinking defaults for supported chat models", () => {
    expect(defaultThinkingEnabledForModel("Qwen3-14B-GGUF")).toBe(true);
    expect(defaultThinkingEnabledForModel("Qwen3.5-0.8B")).toBe(false);
    expect(defaultThinkingEnabledForModel("Qwen3.5-2B")).toBe(false);
    expect(defaultThinkingEnabledForModel("Qwen3.5-4B")).toBe(true);
    expect(defaultThinkingEnabledForModel("Qwen3.5-9B")).toBe(true);
    expect(defaultThinkingEnabledForModel("Qwen3.8-27B-FP8")).toBe(true);
    expect(defaultThinkingEnabledForModel("LFM2.5-1.2B-thinking-gguf")).toBe(
      true,
    );
  });

  it("uses Qwen3.8 thinking behavior without enabling image input", () => {
    expect(isQwen38ThinkingModel("Qwen3.8-27B-FP8")).toBe(true);
    expect(systemPromptForModel(true)).toBe(
      "You are a helpful assistant.",
    );
    expect(supportsImageAttachmentsForModel("Qwen3.8-27B-FP8")).toBe(false);
  });

  it("uses the neutral system prompt for Qwen3.5 thinking control", () => {
    expect(isQwen35ThinkingModel("Qwen3.5-4B")).toBe(true);
    expect(systemPromptForModel(true)).toBe("You are a helpful assistant.");
    expect(systemPromptForModel(false)).toBe(
      "You are a helpful assistant. Provide only the final answer and do not output <think> tags or internal reasoning.",
    );
  });

  it("recognizes Qwen3.5 as the image-capable chat family", () => {
    expect(supportsImageAttachmentsForModel("Qwen3.5-4B")).toBe(true);
    expect(supportsImageAttachmentsForModel("Qwen3-14B-GGUF")).toBe(false);
  });

  it("builds multimodal request content with image parts", () => {
    expect(
      buildChatThreadMessagePayload({
        text: "Describe this",
        images: [
          {
            id: "image-1",
            source: "data:image/png;base64,AAAA",
            label: "example.png",
          },
        ],
      }),
    ).toEqual({
      content: "Describe this",
      contentParts: [
        { type: "text", text: "Describe this" },
        {
          type: "input_image",
          input_image: {
            url: "data:image/png;base64,AAAA",
            name: "example.png",
          },
        },
      ],
    });
  });

  it("summarizes attachment-only prompts for previews and titles", () => {
    expect(
      buildChatThreadMessagePayload({
        text: "   ",
        images: [
          {
            id: "image-1",
            source: "data:image/png;base64,AAAA",
            label: "cat.png",
          },
        ],
      }),
    ).toEqual({
      content: "Attached image: cat.png",
      contentParts: [
        {
          type: "input_image",
          input_image: {
            url: "data:image/png;base64,AAAA",
            name: "cat.png",
          },
        },
      ],
    });
  });

  it("treats no-tag output as final answer when implicit no-tag thinking is disabled", () => {
    expect(
      parseAssistantContent("Plain final answer", {
        implicitOpenThinkTag: true,
        treatNoTagAsThinking: false,
      }),
    ).toEqual({
      thinking: "",
      answer: "Plain final answer",
      hasThink: false,
      hasIncompleteThink: false,
    });
  });

  it("falls back to a default title when cleaned content is empty", () => {
    expect(displayThreadTitle("<think>hidden</think>")).toBe(
      DEFAULT_THREAD_TITLE,
    );
  });

  it("parses user content parts into display text and attachments", () => {
    expect(
      parseUserMessageDisplayFromContentParts({
        id: "message-1",
        thread_id: "thread-1",
        role: "user",
        content: "",
        created_at: Date.now(),
        tokens_generated: null,
        generation_time_ms: null,
        content_parts: [
          { type: "text", text: "Describe this" },
          {
            type: "input_image",
            input_image: {
              url: "https://example.com/example.png",
              name: "example.png",
            },
          },
        ],
      }),
    ).toEqual({
      text: "Describe this",
      attachments: [
        {
          kind: "image",
          source: "https://example.com/example.png",
          label: "example.png",
        },
      ],
    });
  });

  it("resolves relative media URLs to the API origin for desktop-safe rendering", () => {
    const expected = new URL(
      "/v1/media/images/example.png",
      window.location.origin,
    ).toString();

    expect(
      parseUserMessageDisplayFromContentParts({
        id: "message-2",
        thread_id: "thread-2",
        role: "user",
        content: "",
        created_at: Date.now(),
        tokens_generated: null,
        generation_time_ms: null,
        content_parts: [
          {
            type: "input_image",
            input_image: {
              url: "/v1/media/images/example.png",
              name: "example.png",
            },
          },
        ],
      }),
    ).toEqual({
      text: "",
      attachments: [
        {
          kind: "image",
          source: expected,
          label: "example.png",
        },
      ],
    });
  });

  it("reports decode speed and TTFT when the engine measured them", () => {
    expect(
      summarizeChatStats({
        tokens_generated: 41,
        generation_time_ms: 2_000,
        timing: {
          queue_wait_ms: 300,
          prefill_ms: 1_200,
          decode_ms: 380,
          decode_wall_ms: 400,
          decode_tokens: 40,
          ttft_ms: 1_550,
          total_ms: 2_000,
        },
      }),
    ).toEqual({
      tokens: 41,
      totalMs: 2_000,
      tokensPerSecond: 100,
      decodeOnly: true,
      ttftMs: 1_550,
    });
  });

  it("falls back to tokens over total time without decode timing", () => {
    expect(
      summarizeChatStats({ tokens_generated: 48, generation_time_ms: 240 }),
    ).toEqual({
      tokens: 48,
      totalMs: 240,
      tokensPerSecond: 200,
      decodeOnly: false,
      ttftMs: null,
    });
    expect(
      summarizeChatStats({
        tokens_generated: 1,
        generation_time_ms: 0,
        timing: {
          queue_wait_ms: 0,
          prefill_ms: 0,
          decode_ms: 0,
          ttft_ms: null,
          total_ms: 0,
        },
      }),
    ).toMatchObject({ tokensPerSecond: 0, decodeOnly: false, ttftMs: null });
  });

  it("strips markdown from thread previews", () => {
    expect(
      threadPreviewFromContent(
        "**MLOps** stands for **Machine Learning Operations**. See [the docs](https://x.test).",
      ),
    ).toBe("MLOps stands for Machine Learning Operations. See the docs.");
    expect(threadPreviewFromContent("## Steps\n- Use `npm ci`\n- Run *tests*")).toBe(
      "Steps Use npm ci Run tests",
    );
    expect(threadPreviewFromContent("<think>plan</think>> Quoted _answer_")).toBe(
      "Quoted answer",
    );
  });

  it("leaves underscores and asterisks inside words alone", () => {
    expect(stripMarkdown("snake_case_name and 2*3*4")).toBe(
      "snake_case_name and 2*3*4",
    );
  });

  it("strips emphasis without a lookbehind, which macOS 12 cannot parse", () => {
    expect(stripMarkdown("Use *this*, _that_ (or *a b*) and * spaced *.")).toBe(
      "Use this, that (or a b) and * spaced *.",
    );
    expect(stripMarkdown.toString()).not.toMatch(/\(\?<[=!]/);
  });

  it("keeps the opener of a reply stopped while it reasoned", () => {
    expect(stoppedReplyContent("Weighing it", true)).toBe("<think>Weighing it");
    expect(stoppedReplyContent("Weighing it</think>An answer", true)).toBe(
      "Weighing it</think>An answer",
    );
    expect(stoppedReplyContent("<think>Its own opener", true)).toBe(
      "<think>Its own opener",
    );
    expect(stoppedReplyContent("", true)).toBe("");
    expect(stoppedReplyContent("An answer", false)).toBe("An answer");
  });

  it("labels message days relative to now", () => {
    const now = new Date(2026, 9, 18, 9, 30).getTime();
    expect(messageDayLabel(new Date(2026, 9, 18, 0, 5).getTime(), now)).toBe(
      "Today",
    );
    expect(messageDayLabel(new Date(2026, 9, 17, 23, 59).getTime(), now)).toBe(
      "Yesterday",
    );
    const lastWeek = messageDayLabel(new Date(2026, 9, 12, 12).getTime(), now);
    expect(lastWeek).toMatch(/12/);
    expect(lastWeek).not.toMatch(/2026/);
    expect(messageDayLabel(new Date(2025, 11, 31, 12).getTime(), now)).toMatch(
      /2025/,
    );
  });

  it("compares calendar days, not 24-hour windows", () => {
    expect(
      isSameCalendarDay(
        new Date(2026, 9, 18, 0, 1).getTime(),
        new Date(2026, 9, 18, 23, 59).getTime(),
      ),
    ).toBe(true);
    expect(
      isSameCalendarDay(
        new Date(2026, 9, 17, 23, 59).getTime(),
        new Date(2026, 9, 18, 0, 1).getTime(),
      ),
    ).toBe(false);
  });
});

describe("readChatImage", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
    vi.restoreAllMocks();
  });

  const stubCanvas = () => {
    const context = { fillStyle: "", fillRect: vi.fn(), drawImage: vi.fn() };
    vi.spyOn(HTMLCanvasElement.prototype, "getContext").mockReturnValue(
      context as unknown as CanvasRenderingContext2D,
    );
    const toDataUrl = vi
      .spyOn(HTMLCanvasElement.prototype, "toDataURL")
      .mockImplementation(function (this: HTMLCanvasElement) {
        return `data:image/jpeg;base64,${this.width}x${this.height}`;
      });
    return { context, toDataUrl };
  };

  it("scales a large photo down to a 2048 px edge as JPEG", async () => {
    const close = vi.fn();
    vi.stubGlobal(
      "createImageBitmap",
      vi.fn().mockResolvedValue({ width: 4000, height: 3000, close }),
    );
    const { toDataUrl } = stubCanvas();
    const file = new File([new Uint8Array(10)], "photo.png", { type: "image/png" });

    await expect(readChatImage(file)).resolves.toBe(
      "data:image/jpeg;base64,2048x1536",
    );
    expect(toDataUrl).toHaveBeenCalledWith("image/jpeg", 0.9);
    expect(close).toHaveBeenCalled();
  });

  it("sends a small image as it is", async () => {
    vi.stubGlobal(
      "createImageBitmap",
      vi.fn().mockResolvedValue({ width: 800, height: 600, close: vi.fn() }),
    );
    const { toDataUrl } = stubCanvas();
    const file = new File([new Uint8Array([1, 2, 3])], "small.png", {
      type: "image/png",
    });

    await expect(readChatImage(file)).resolves.toBe("data:image/png;base64,AQID");
    expect(toDataUrl).not.toHaveBeenCalled();
  });
});
