import {
  ApiHttpClient,
  consumeDataStream,
  isAbortError,
} from "@/shared/api/http";

const DEFAULT_CHAT_MODEL = "Qwen3-8B-GGUF";

export interface ChatMessage {
  role: "system" | "user" | "assistant";
  content: string;
}

export interface ChatCompletionRequest {
  model_id?: string;
  messages: ChatMessage[];
  max_tokens?: number;
}

export interface ChatCompletionResponse {
  model_id: string;
  message: ChatMessage;
  stats: {
    tokens_generated: number;
    generation_time_ms: number;
  };
}

export type ChatStreamEvent =
  | { event: "start"; model_id: string }
  | { event: "delta"; delta: string }
  | {
      event: "done";
      model_id: string;
      message: string;
      stats: {
        tokens_generated: number;
        generation_time_ms: number;
      };
    }
  | { event: "error"; error: string };

export interface ChatStreamCallbacks {
  onStart?: (modelId: string) => void;
  onDelta?: (delta: string) => void;
  onDone?: (
    message: string,
    stats: { tokens_generated: number; generation_time_ms: number },
  ) => void;
  onError?: (error: string) => void;
}

export interface ChatThread {
  id: string;
  title: string;
  model_id: string | null;
  created_at: number;
  updated_at: number;
  last_message_preview: string | null;
  message_count: number;
}

export interface ChatThreadMessageRecord {
  id: string;
  thread_id: string;
  role: "system" | "user" | "assistant";
  content: string;
  content_parts?: ChatThreadContentPart[] | null;
  created_at: number;
  tokens_generated: number | null;
  generation_time_ms: number | null;
}

export interface ChatThreadDetail {
  thread: ChatThread;
  messages: ChatThreadMessageRecord[];
}

export interface ChatThreadCreateRequest {
  title?: string;
  model_id?: string;
}

export interface ChatThreadUpdateRequest {
  title: string;
}

export interface ChatThreadContentPart {
  type:
    | "text"
    | "input_text"
    | "image_url"
    | "input_image"
    | "video"
    | "video_url"
    | "input_video";
  text?: string;
  input_text?: string;
  image_url?: unknown;
  input_image?: unknown;
  image?: unknown;
  video?: unknown;
  video_url?: unknown;
  input_video?: unknown;
}

export interface ChatThreadSendMessageRequest {
  model_id?: string;
  content: string;
  content_parts?: ChatThreadContentPart[];
  max_tokens?: number;
  system_prompt?: string;
  enable_thinking?: boolean;
}

export interface ChatThreadSendMessageResponse {
  thread_id: string;
  model_id: string;
  user_message: ChatThreadMessageRecord;
  assistant_message: ChatThreadMessageRecord;
  stats: {
    tokens_generated: number;
    generation_time_ms: number;
  };
}

type ChatThreadStreamEvent =
  | {
      event: "start";
      thread_id: string;
      model_id: string;
      user_message: ChatThreadMessageRecord;
    }
  | { event: "delta"; delta: string }
  | {
      event: "done";
      thread_id: string;
      model_id: string;
      assistant_message: ChatThreadMessageRecord;
      stats: {
        tokens_generated: number;
        generation_time_ms: number;
      };
    }
  | { event: "error"; error: string };

export interface ChatThreadStreamCallbacks {
  onStart?: (event: {
    threadId: string;
    modelId: string;
    userMessage: ChatThreadMessageRecord;
  }) => void;
  onDelta?: (delta: string) => void;
  onDone?: (event: {
    threadId: string;
    modelId: string;
    assistantMessage: ChatThreadMessageRecord;
    stats: {
      tokens_generated: number;
      generation_time_ms: number;
    };
  }) => void;
  onError?: (error: string) => void;
  onClose?: () => void;
}

export type AgentPlanningMode = "off" | "auto" | "on";

export interface AgentSessionCreateRequest {
  agent_id?: string;
  model_id?: string;
  system_prompt?: string;
  planning_mode?: AgentPlanningMode;
  title?: string;
}

export interface AgentSession {
  id: string;
  agent_id: string;
  thread_id: string;
  model_id: string;
  planning_mode: AgentPlanningMode;
  created_at: number;
  updated_at: number;
}

export type AgentEvent =
  | { event: "turn_started"; session_id: string; thread_id: string }
  | { event: "plan_created"; steps: string[] }
  | { event: "tool_call_started"; name: string }
  | { event: "tool_call_completed"; name: string; output: string }
  | { event: "assistant_message"; content: string }
  | { event: "turn_completed"; session_id: string; thread_id: string };

export interface AgentToolCallRecord {
  name: string;
  input_summary: string;
  output: string;
}

export interface AgentTurnRequest {
  input: string;
  model_id?: string;
  max_output_tokens?: number;
}

export interface AgentTurnResponse {
  session_id: string;
  thread_id: string;
  model_id: string;
  assistant_text: string;
  plan: { mode: AgentPlanningMode; steps: string[] } | null;
  tool_calls: AgentToolCallRecord[];
  events: AgentEvent[];
}

export interface ResponsesCreateRequest {
  model_id?: string;
  input: string;
  instructions?: string;
  max_output_tokens?: number;
  metadata?: Record<string, unknown>;
  store?: boolean;
}

export interface ResponsesObject {
  id: string;
  status: string;
  model: string;
  output_text: string;
  usage: {
    input_tokens: number;
    output_tokens: number;
    total_tokens: number;
  };
}

export interface ResponsesStreamCallbacks {
  onCreated?: (response: ResponsesObject) => void;
  onDelta?: (delta: string) => void;
  onCompleted?: (response: ResponsesObject) => void;
  onError?: (error: string) => void;
  onDone?: () => void;
}

interface OpenAiChatCompletion {
  id: string;
  model: string;
  choices: Array<{
    index: number;
    message: {
      role: "assistant" | "system" | "user";
      content: string;
    };
    finish_reason: string;
  }>;
  usage?: {
    completion_tokens?: number;
  };
  izwi_generation_time_ms?: number;
}

interface OpenAiChatChunk {
  id: string;
  model: string;
  choices: Array<{
    index: number;
    delta: {
      role?: "assistant" | "system" | "user";
      content?: string;
    };
    finish_reason: string | null;
  }>;
  usage?: {
    completion_tokens?: number;
  };
  izwi_generation_time_ms?: number;
}

interface OpenAiResponseObject {
  id: string;
  status: string;
  model: string;
  output?: Array<{
    type: string;
    role: string;
    content?: Array<{
      type: string;
      text?: string;
    }>;
  }>;
  usage?: {
    input_tokens?: number;
    output_tokens?: number;
    total_tokens?: number;
  };
}

export class ChatApiClient {
  constructor(private readonly http: ApiHttpClient) {}

  private mapResponseObject(payload: OpenAiResponseObject): ResponsesObject {
    const firstOutputText =
      payload.output?.[0]?.content
        ?.map((part) => part.text ?? "")
        .join("") ?? "";

    return {
      id: payload.id,
      status: payload.status,
      model: payload.model,
      output_text: firstOutputText,
      usage: {
        input_tokens: payload.usage?.input_tokens ?? 0,
        output_tokens: payload.usage?.output_tokens ?? 0,
        total_tokens: payload.usage?.total_tokens ?? 0,
      },
    };
  }

  async listChatThreads(): Promise<ChatThread[]> {
    const payload = await this.http.request<{ threads: ChatThread[] }>(
      "/chat/threads",
    );
    return payload.threads;
  }

  async createChatThread(
    request?: ChatThreadCreateRequest,
  ): Promise<ChatThread> {
    return this.http.request("/chat/threads", {
      method: "POST",
      body: JSON.stringify({
        title: request?.title,
        model_id: request?.model_id,
      }),
    });
  }

  async updateChatThread(
    threadId: string,
    request: ChatThreadUpdateRequest,
  ): Promise<ChatThread> {
    return this.http.request(`/chat/threads/${encodeURIComponent(threadId)}`, {
      method: "PATCH",
      body: JSON.stringify({
        title: request.title,
      }),
    });
  }

  async getChatThread(threadId: string): Promise<ChatThreadDetail> {
    return this.http.request(`/chat/threads/${encodeURIComponent(threadId)}`);
  }

  async listChatThreadMessages(
    threadId: string,
  ): Promise<ChatThreadMessageRecord[]> {
    return this.http.request(
      `/chat/threads/${encodeURIComponent(threadId)}/messages`,
    );
  }

  async deleteChatThread(
    threadId: string,
  ): Promise<{ id: string; deleted: boolean }> {
    return this.http.request(`/chat/threads/${encodeURIComponent(threadId)}`, {
      method: "DELETE",
    });
  }

  async sendChatThreadMessage(
    threadId: string,
    request: ChatThreadSendMessageRequest,
  ): Promise<ChatThreadSendMessageResponse> {
    return this.http.request(
        `/chat/threads/${encodeURIComponent(threadId)}/messages`,
      {
        method: "POST",
        body: JSON.stringify({
          model: request.model_id ?? DEFAULT_CHAT_MODEL,
          content: request.content,
          content_parts: request.content_parts,
          max_tokens: request.max_tokens,
          stream: false,
          system_prompt: request.system_prompt,
          enable_thinking: request.enable_thinking,
        }),
      },
    );
  }

  sendChatThreadMessageStream(
    threadId: string,
    request: ChatThreadSendMessageRequest,
    callbacks: ChatThreadStreamCallbacks,
  ): AbortController {
    const abortController = new AbortController();

    const startStream = async () => {
      try {
        const response = await fetch(
          this.http.url(
            `/chat/threads/${encodeURIComponent(threadId)}/messages`,
          ),
          {
            method: "POST",
            headers: {
              "Content-Type": "application/json",
            },
            body: JSON.stringify({
              model: request.model_id ?? DEFAULT_CHAT_MODEL,
              content: request.content,
              content_parts: request.content_parts,
              max_tokens: request.max_tokens,
              stream: true,
              system_prompt: request.system_prompt,
              enable_thinking: request.enable_thinking,
            }),
            signal: abortController.signal,
          },
        );

        if (!response.ok) {
          callbacks.onError?.(
            (
              await this.http.createError(
                response,
                "Thread chat streaming failed",
              )
            ).message,
          );
          callbacks.onClose?.();
          return;
        }

        await consumeDataStream(response, (data) => {
          if (data === "[DONE]") {
            return true;
          }

          try {
            const event = JSON.parse(data) as ChatThreadStreamEvent;
            switch (event.event) {
              case "start":
                callbacks.onStart?.({
                  threadId: event.thread_id,
                  modelId: event.model_id,
                  userMessage: event.user_message,
                });
                break;
              case "delta":
                callbacks.onDelta?.(event.delta);
                break;
              case "done":
                callbacks.onDone?.({
                  threadId: event.thread_id,
                  modelId: event.model_id,
                  assistantMessage: event.assistant_message,
                  stats: event.stats,
                });
                break;
              case "error":
                callbacks.onError?.(event.error);
                break;
            }
          } catch {
            // Skip malformed SSE payloads.
          }

          return false;
        });

        callbacks.onClose?.();
      } catch (error) {
        if (!isAbortError(error)) {
          callbacks.onError?.(
            error instanceof Error
              ? error.message
              : "Thread chat stream error",
          );
        }
        callbacks.onClose?.();
      }
    };

    void startStream();
    return abortController;
  }

  async createAgentSession(
    request: AgentSessionCreateRequest,
  ): Promise<AgentSession> {
    return this.http.request("/agent/sessions", {
      method: "POST",
      body: JSON.stringify({
        agent_id: request.agent_id,
        model_id: request.model_id,
        system_prompt: request.system_prompt,
        planning_mode: request.planning_mode,
        title: request.title,
      }),
    });
  }

  async getAgentSession(sessionId: string): Promise<AgentSession> {
    return this.http.request(`/agent/sessions/${encodeURIComponent(sessionId)}`);
  }

  async createAgentTurn(
    sessionId: string,
    request: AgentTurnRequest,
    signal?: AbortSignal,
  ): Promise<AgentTurnResponse> {
    return this.http.request(
      `/agent/sessions/${encodeURIComponent(sessionId)}/turns`,
      {
        method: "POST",
        body: JSON.stringify({
          input: request.input,
          model_id: request.model_id,
          max_output_tokens: request.max_output_tokens,
        }),
        signal,
      },
    );
  }

  async chatCompletions(
    request: ChatCompletionRequest,
  ): Promise<ChatCompletionResponse> {
    const response = await this.http.request<OpenAiChatCompletion>(
      "/chat/completions",
      {
        method: "POST",
        body: JSON.stringify({
          model: request.model_id ?? DEFAULT_CHAT_MODEL,
          messages: request.messages,
          max_tokens: request.max_tokens,
          stream: false,
        }),
      },
    );

    const firstChoice = response.choices[0];
    if (!firstChoice) {
      throw new Error("Missing assistant response");
    }

    return {
      model_id: response.model,
      message: {
        role: firstChoice.message.role,
        content: firstChoice.message.content,
      },
      stats: {
        tokens_generated: response.usage?.completion_tokens ?? 0,
        generation_time_ms: response.izwi_generation_time_ms ?? 0,
      },
    };
  }

  chatCompletionsStream(
    request: ChatCompletionRequest,
    callbacks: ChatStreamCallbacks,
  ): AbortController {
    const abortController = new AbortController();

    const startStream = async () => {
      try {
        const response = await fetch(this.http.url("/chat/completions"), {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            model: request.model_id ?? DEFAULT_CHAT_MODEL,
            messages: request.messages,
            max_tokens: request.max_tokens,
            stream: true,
          }),
          signal: abortController.signal,
        });

        if (!response.ok) {
          callbacks.onError?.(
            (await this.http.createError(response, "Chat streaming failed"))
              .message,
          );
          return;
        }

        callbacks.onStart?.(request.model_id ?? DEFAULT_CHAT_MODEL);

        let fullText = "";
        const streamStartedAt = performance.now();
        let completionTokens: number | null = null;
        let generationTimeMs: number | null = null;

        await consumeDataStream(response, (data) => {
          if (data === "[DONE]") {
            const elapsedMs = Math.max(
              1,
              Math.round(performance.now() - streamStartedAt),
            );
            callbacks.onDone?.(fullText, {
              tokens_generated:
                completionTokens ?? Math.max(1, Math.floor(fullText.length / 4)),
              generation_time_ms: generationTimeMs ?? elapsedMs,
            });
            return true;
          }

          try {
            const payload = JSON.parse(data) as OpenAiChatChunk;
            if (typeof payload.izwi_generation_time_ms === "number") {
              generationTimeMs = payload.izwi_generation_time_ms;
            }
            if (typeof payload.usage?.completion_tokens === "number") {
              completionTokens = payload.usage.completion_tokens;
            }
            const choice = payload.choices?.[0];
            const delta = choice?.delta?.content;
            if (delta) {
              fullText += delta;
              callbacks.onDelta?.(delta);
            }
          } catch {
            // Skip malformed SSE payloads.
          }

          return false;
        });
      } catch (error) {
        if (!isAbortError(error)) {
          callbacks.onError?.(
            error instanceof Error ? error.message : "Chat stream error",
          );
        }
      }
    };

    void startStream();
    return abortController;
  }

  async createResponse(
    request: ResponsesCreateRequest,
  ): Promise<ResponsesObject> {
    const payload = await this.http.request<OpenAiResponseObject>("/responses", {
      method: "POST",
      body: JSON.stringify({
        model: request.model_id ?? DEFAULT_CHAT_MODEL,
        input: request.input,
        instructions: request.instructions,
        max_output_tokens: request.max_output_tokens,
        metadata: request.metadata,
        store: request.store,
      }),
    });

    return this.mapResponseObject(payload);
  }

  createResponseStream(
    request: ResponsesCreateRequest,
    callbacks: ResponsesStreamCallbacks,
  ): AbortController {
    const abortController = new AbortController();

    const startStream = async () => {
      try {
        const response = await fetch(this.http.url("/responses"), {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            model: request.model_id ?? DEFAULT_CHAT_MODEL,
            input: request.input,
            instructions: request.instructions,
            max_output_tokens: request.max_output_tokens,
            metadata: request.metadata,
            store: request.store,
            stream: true,
          }),
          signal: abortController.signal,
        });

        if (!response.ok) {
          callbacks.onError?.(
            (
              await this.http.createError(
                response,
                "Responses streaming failed",
              )
            ).message,
          );
          callbacks.onDone?.();
          return;
        }

        await consumeDataStream(response, (data) => {
          if (data === "[DONE]") {
            callbacks.onDone?.();
            return true;
          }

          try {
            const event = JSON.parse(data) as {
              type: string;
              response?: OpenAiResponseObject;
              delta?: string;
              error?: { message?: string };
            };

            if (event.type === "response.created" && event.response) {
              callbacks.onCreated?.(this.mapResponseObject(event.response));
            } else if (event.type === "response.output_text.delta") {
              callbacks.onDelta?.(event.delta ?? "");
            } else if (event.type === "response.completed" && event.response) {
              callbacks.onCompleted?.(this.mapResponseObject(event.response));
            } else if (event.type === "response.failed") {
              callbacks.onError?.(
                event.error?.message ?? "Responses request failed",
              );
            }
          } catch {
            // Skip malformed SSE payloads.
          }

          return false;
        });
      } catch (error) {
        if (!isAbortError(error)) {
          callbacks.onError?.(
            error instanceof Error ? error.message : "Responses stream error",
          );
        }
        callbacks.onDone?.();
      }
    };

    void startStream();
    return abortController;
  }

  async getResponse(id: string): Promise<ResponsesObject> {
    const payload = await this.http.request<OpenAiResponseObject>(
      `/responses/${id}`,
    );
    return this.mapResponseObject(payload);
  }

  async cancelResponse(id: string): Promise<ResponsesObject> {
    const payload = await this.http.request<OpenAiResponseObject>(
      `/responses/${id}/cancel`,
      {
        method: "POST",
      },
    );
    return this.mapResponseObject(payload);
  }

  async deleteResponse(
    id: string,
  ): Promise<{ id: string; object: string; deleted: boolean }> {
    return this.http.request(`/responses/${id}`, {
      method: "DELETE",
    });
  }
}
