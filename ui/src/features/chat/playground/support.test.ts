import { describe, expect, it } from "vitest";
import {
  buildChatThreadMessagePayload,
  DEFAULT_THREAD_TITLE,
  defaultThinkingEnabledForModel,
  displayThreadTitle,
  isQwen35ThinkingModel,
  isQwen38ThinkingModel,
  parseAssistantContent,
  parseUserMessageDisplayFromContentParts,
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
});
