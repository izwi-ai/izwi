import { describe, expect, it } from "vitest";
import {
  DEFAULT_THREAD_TITLE,
  buildThreadContentParts,
  displayThreadTitle,
  parseAssistantContent,
  parseUserMessageDisplayFromContentParts,
  supportsImplicitOpenThinkTagParsing,
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
    expect(supportsImplicitOpenThinkTagParsing("Qwen3.5-4B-Instruct")).toBe(
      true,
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

  it("builds thread content parts for text plus media", () => {
    expect(
      buildThreadContentParts("hello", [
        {
          id: "image-1",
          kind: "image",
          name: "cat.png",
          size: 10,
          mimeType: "image/png",
          dataUrl: "data:image/png;base64,abc",
          previewUrl: "blob:image",
        },
        {
          id: "video-1",
          kind: "video",
          name: "clip.mp4",
          size: 20,
          mimeType: "video/mp4",
          dataUrl: "data:video/mp4;base64,def",
          previewUrl: "blob:video",
        },
      ]),
    ).toEqual([
      { type: "text", text: "hello" },
      {
        type: "input_image",
        input_image: {
          url: "data:image/png;base64,abc",
          media_type: "image/png",
          name: "cat.png",
        },
      },
      {
        type: "input_video",
        input_video: {
          url: "data:video/mp4;base64,def",
          media_type: "video/mp4",
          name: "clip.mp4",
        },
      },
    ]);
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
});
