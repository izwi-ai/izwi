import {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
  type ChangeEvent,
} from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  Bot,
  Brain,
  Send,
  Square,
  User,
  Loader2,
  ChevronDown,
  ChevronRight,
  Settings2,
  Plus,
  Trash2,
  AlertTriangle,
  MessageSquare,
  History,
  Paperclip,
  ImageIcon,
  Film,
  X,
} from "lucide-react";
import { useSearchParams } from "react-router-dom";
import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";
import { Dialog, DialogContent, DialogTitle } from "@/components/ui/dialog";
import {
  api,
  ChatMessage,
  ChatThread,
  ChatThreadContentPart,
  ChatThreadMessageRecord,
} from "../api";
import { MarkdownContent } from "./ui/MarkdownContent";

interface ModelOption {
  value: string;
  label: string;
  statusLabel: string;
  isReady: boolean;
}

interface ChatPlaygroundProps {
  selectedModel: string | null;
  selectedModelReady: boolean;
  supportsThinking: boolean;
  modelLabel?: string | null;
  modelOptions: ModelOption[];
  onSelectModel: (variant: string) => void;
  onOpenModelManager: () => void;
  onModelRequired: () => void;
}

const THINKING_SYSTEM_PROMPT: ChatMessage = {
  role: "system",
  content:
    "You are a helpful assistant. Always reason inside <think>...</think> before giving the final answer. Keep thinking concise, always close </think>, then provide a clear final answer outside the tags.",
};

const DEFAULT_SYSTEM_PROMPT: ChatMessage = {
  role: "system",
  content:
    "You are a helpful assistant. Provide only the final answer and do not output <think> tags or internal reasoning.",
};

const DEFAULT_THREAD_TITLE = "New chat";
const MAX_THREAD_TITLE_CHARS = 80;
const MAX_MEDIA_ATTACHMENTS = 4;
const MAX_MEDIA_ATTACHMENT_MB = 32;
const MAX_MEDIA_ATTACHMENT_BYTES = MAX_MEDIA_ATTACHMENT_MB * 1024 * 1024;

type ComposerMediaKind = "image" | "video";

interface ComposerMediaItem {
  id: string;
  kind: ComposerMediaKind;
  name: string;
  size: number;
  mimeType: string;
  dataUrl: string;
  previewUrl: string;
}

interface ParsedAssistantContent {
  thinking: string;
  answer: string;
  hasThink: boolean;
  hasIncompleteThink: boolean;
}

interface ParseAssistantContentOptions {
  implicitOpenThinkTag?: boolean;
}

interface ParsedUserAttachment {
  kind: ComposerMediaKind;
  source: string | null;
  label: string;
}

interface ParsedUserMessageDisplay {
  text: string;
  attachments: ParsedUserAttachment[];
}

interface ImagePreviewState {
  source: string;
  label: string;
}

function parseAssistantContent(
  content: string,
  options?: ParseAssistantContentOptions,
): ParsedAssistantContent {
  const openTag = "<think>";
  const closeTag = "</think>";
  const raw = content ?? "";
  const implicitOpenThinkTag = !!options?.implicitOpenThinkTag;
  const lowered = raw.toLowerCase();
  const firstOpen = lowered.indexOf(openTag);
  const firstClose = lowered.indexOf(closeTag);

  if (
    implicitOpenThinkTag &&
    firstClose >= 0 &&
    (firstOpen === -1 || firstClose < firstOpen)
  ) {
    const thinking = raw.slice(0, firstClose).trim();
    const answer = raw.slice(firstClose + closeTag.length).trim();
    return {
      thinking,
      answer,
      hasThink: thinking.length > 0,
      hasIncompleteThink: false,
    };
  }

  if (implicitOpenThinkTag && firstOpen === -1 && firstClose === -1) {
    const thinking = raw.trim();
    if (thinking.length > 0) {
      return {
        thinking,
        answer: "",
        hasThink: true,
        hasIncompleteThink: true,
      };
    }
  }

  const thinkingParts: string[] = [];
  const answerParts: string[] = [];
  let cursor = 0;
  let hasIncompleteThink = false;

  while (true) {
    const openIdx = raw.indexOf(openTag, cursor);
    if (openIdx === -1) {
      answerParts.push(raw.slice(cursor));
      break;
    }

    answerParts.push(raw.slice(cursor, openIdx));
    const thinkStart = openIdx + openTag.length;
    const closeIdx = raw.indexOf(closeTag, thinkStart);

    if (closeIdx === -1) {
      thinkingParts.push(raw.slice(thinkStart));
      hasIncompleteThink = true;
      break;
    }

    thinkingParts.push(raw.slice(thinkStart, closeIdx));
    cursor = closeIdx + closeTag.length;
  }

  return {
    thinking: thinkingParts.join("\n\n").trim(),
    answer: answerParts.join("").trim(),
    hasThink: thinkingParts.length > 0,
    hasIncompleteThink,
  };
}

function getErrorMessage(error: unknown, fallback: string): string {
  if (error instanceof Error && error.message.trim()) {
    return error.message;
  }
  return fallback;
}

function formatThreadTimestamp(timestamp: number): string {
  const date = new Date(timestamp);
  const now = new Date();
  const isToday = date.toDateString() === now.toDateString();
  if (isToday) {
    return date.toLocaleTimeString([], { hour: "numeric", minute: "2-digit" });
  }
  return date.toLocaleDateString([], { month: "short", day: "numeric" });
}

function extractLatestStats(messages: ChatThreadMessageRecord[]): {
  tokens_generated: number;
  generation_time_ms: number;
} | null {
  for (let index = messages.length - 1; index >= 0; index -= 1) {
    const message = messages[index];
    if (
      message.role === "assistant" &&
      typeof message.tokens_generated === "number" &&
      typeof message.generation_time_ms === "number"
    ) {
      return {
        tokens_generated: message.tokens_generated,
        generation_time_ms: message.generation_time_ms,
      };
    }
  }
  return null;
}

function truncateText(text: string, maxChars: number): string {
  if (text.length <= maxChars) {
    return text;
  }
  return `${text.slice(0, Math.max(0, maxChars - 3)).trim()}...`;
}

function normalizeWhitespace(text: string): string {
  return text.replace(/\s+/g, " ").trim();
}

function stripThinkingArtifacts(text: string): string {
  let output = text;
  const openTag = "<think>";
  const closeTag = "</think>";
  const lowered = output.toLowerCase();
  const firstOpen = lowered.indexOf(openTag);
  const firstClose = lowered.indexOf(closeTag);
  if (firstClose >= 0 && (firstOpen === -1 || firstClose < firstOpen)) {
    output = output.slice(firstClose + closeTag.length);
  }

  return output
    .replace(/<think>[\s\S]*?<\/think>/gi, " ")
    .replace(/<think>[\s\S]*$/gi, " ")
    .replace(/<think>/gi, " ")
    .replace(/<\/think>/gi, " ");
}

function displayThreadTitle(rawTitle: string | null | undefined): string {
  const cleaned = normalizeWhitespace(stripThinkingArtifacts(rawTitle ?? ""));
  if (!cleaned) {
    return DEFAULT_THREAD_TITLE;
  }
  return truncateText(cleaned, MAX_THREAD_TITLE_CHARS);
}

function threadPreviewFromContent(content: string | null | undefined): string {
  const normalized = normalizeWhitespace(stripThinkingArtifacts(content ?? ""));
  if (!normalized) {
    return "No messages yet";
  }
  return truncateText(normalized, 120);
}

function normalizeGeneratedThreadTitle(raw: string): string | null {
  const compact = normalizeWhitespace(
    stripThinkingArtifacts(raw)
      .replace(/```[\s\S]*?```/g, " ")
      .replace(/<\/?[^>]+>/g, " "),
  );
  if (!compact) {
    return null;
  }

  let title = compact.replace(/^title\s*[:\-]\s*/i, "").trim();
  title = normalizeWhitespace(title.replace(/^['"`]+|['"`]+$/g, "").trim());

  if (!title || /^user\s*:/i.test(title) || /^assistant\s*:/i.test(title)) {
    return null;
  }

  return displayThreadTitle(title);
}

function fallbackThreadTitleFromUserMessage(content: string): string {
  const normalized = normalizeWhitespace(stripThinkingArtifacts(content));
  if (!normalized) {
    return DEFAULT_THREAD_TITLE;
  }
  return truncateText(normalized, MAX_THREAD_TITLE_CHARS);
}

function isQwen35ChatModel(variant: string | null): boolean {
  if (!variant) {
    return false;
  }
  return variant.trim().toLowerCase().startsWith("qwen3.5-");
}

function isLfm25ThinkingModel(variant: string | null): boolean {
  if (!variant) {
    return false;
  }
  return variant.trim().toLowerCase() === "lfm2.5-1.2b-thinking-gguf";
}

function supportsImplicitOpenThinkTagParsing(variant: string | null): boolean {
  return isQwen35ChatModel(variant) || isLfm25ThinkingModel(variant);
}

function formatBytes(size: number): string {
  if (!Number.isFinite(size) || size <= 0) {
    return "0 B";
  }
  const units = ["B", "KB", "MB", "GB"];
  let value = size;
  let unitIndex = 0;
  while (value >= 1024 && unitIndex < units.length - 1) {
    value /= 1024;
    unitIndex += 1;
  }
  const precision = unitIndex === 0 ? 0 : value >= 10 ? 1 : 2;
  return `${value.toFixed(precision)} ${units[unitIndex]}`;
}

function createMediaItemId(): string {
  if (
    typeof crypto !== "undefined" &&
    typeof crypto.randomUUID === "function"
  ) {
    return crypto.randomUUID();
  }
  return `media-${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

function fileToDataUrl(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => {
      if (typeof reader.result === "string") {
        resolve(reader.result);
        return;
      }
      reject(new Error(`Failed reading ${file.name}`));
    };
    reader.onerror = () => {
      reject(new Error(`Failed reading ${file.name}`));
    };
    reader.readAsDataURL(file);
  });
}

function buildUserDisplayContent(
  text: string,
  mediaItems: ComposerMediaItem[],
): string {
  const lines: string[] = [];
  if (text.trim()) {
    lines.push(text.trim());
  }
  for (const item of mediaItems) {
    const label = item.kind === "image" ? "[image]" : "[video]";
    lines.push(`${label} ${item.name}`);
  }
  return lines.join("\n").trim();
}

function buildThreadContentParts(
  text: string,
  mediaItems: ComposerMediaItem[],
): ChatThreadContentPart[] {
  const parts: ChatThreadContentPart[] = [];
  if (text.trim()) {
    parts.push({
      type: "text",
      text: text.trim(),
    });
  }
  for (const item of mediaItems) {
    const mediaPayload = {
      url: item.dataUrl,
      media_type: item.mimeType || undefined,
      name: item.name,
    };
    if (item.kind === "image") {
      parts.push({
        type: "input_image",
        input_image: mediaPayload,
      });
      continue;
    }
    parts.push({
      type: "input_video",
      input_video: mediaPayload,
    });
  }
  return parts;
}

function extractMarkdownImageSource(value: string): string | null {
  const match = value.match(/^!\[[^\]]*\]\((.+)\)$/);
  if (!match) {
    return null;
  }
  const source = match[1]?.trim();
  return source ? source : null;
}

function resolveAttachmentSource(
  value: string,
  kind: ComposerMediaKind,
): string | null {
  const trimmed = value.trim();
  if (!trimmed) {
    return null;
  }

  const source = extractMarkdownImageSource(trimmed) ?? trimmed;
  const lowered = source.toLowerCase();
  if (kind === "image") {
    if (lowered.startsWith("data:image/")) {
      return source;
    }
    if (
      lowered.startsWith("https://") ||
      lowered.startsWith("http://") ||
      lowered.startsWith("blob:") ||
      lowered.startsWith("/")
    ) {
      return source;
    }
    return null;
  }

  if (
    lowered.startsWith("data:video/") ||
    lowered.startsWith("https://") ||
    lowered.startsWith("http://") ||
    lowered.startsWith("blob:") ||
    lowered.startsWith("/")
  ) {
    return source;
  }
  return null;
}

function attachmentLabel(value: string, kind: ComposerMediaKind): string {
  const fallback = kind === "image" ? "Attached image" : "Attached video";
  const trimmed = value.trim();
  if (!trimmed) {
    return fallback;
  }

  const candidate = extractMarkdownImageSource(trimmed) ?? trimmed;
  if (candidate.toLowerCase().startsWith("data:")) {
    return fallback;
  }

  try {
    const url = new URL(candidate);
    const pathSegments = url.pathname.split("/").filter(Boolean);
    const lastSegment = decodeURIComponent(
      pathSegments[pathSegments.length - 1] || "",
    ).trim();
    if (lastSegment) {
      return lastSegment;
    }
  } catch {
    // Ignore invalid URLs and keep original string fallback below.
  }

  return trimmed;
}

function readObjectField(value: unknown, keys: string[]): string | null {
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    return null;
  }
  const record = value as Record<string, unknown>;
  for (const key of keys) {
    const entry = record[key];
    if (typeof entry === "string" && entry.trim()) {
      return entry.trim();
    }
  }
  return null;
}

function extractMediaSourceForDisplay(value: unknown): string | null {
  if (typeof value === "string" && value.trim()) {
    return value.trim();
  }

  return readObjectField(value, ["url", "src", "uri", "source"]);
}

function extractMediaLabel(value: unknown, kind: ComposerMediaKind): string {
  if (typeof value === "string" && value.trim()) {
    return attachmentLabel(value, kind);
  }

  const explicitName = readObjectField(value, [
    "name",
    "file_name",
    "filename",
  ]);
  if (explicitName) {
    return explicitName;
  }

  const source = extractMediaSourceForDisplay(value);
  if (source) {
    return attachmentLabel(source, kind);
  }

  return kind === "image" ? "Attached image" : "Attached video";
}

function parseUserMessageDisplayFromContentParts(
  message: ChatThreadMessageRecord,
): ParsedUserMessageDisplay | null {
  const parts = message.content_parts;
  if (!parts || parts.length === 0) {
    return null;
  }

  const textSegments: string[] = [];
  const attachments: ParsedUserAttachment[] = [];

  for (const part of parts) {
    const type = (part.type || "").toLowerCase();
    if (type === "text" || type === "input_text") {
      const textCandidate =
        (typeof part.text === "string" && part.text.trim()
          ? part.text
          : null) ||
        (typeof part.input_text === "string" && part.input_text.trim()
          ? part.input_text
          : null);
      if (textCandidate) {
        textSegments.push(textCandidate.trim());
      }
      continue;
    }

    if (
      type === "input_image" ||
      type === "image_url" ||
      type === "image" ||
      part.input_image
    ) {
      const payload = part.input_image ?? part.image_url ?? part.image ?? null;
      const source = extractMediaSourceForDisplay(payload);
      attachments.push({
        kind: "image",
        source: source ? resolveAttachmentSource(source, "image") : null,
        label: extractMediaLabel(payload, "image"),
      });
      continue;
    }

    if (
      type === "input_video" ||
      type === "video_url" ||
      type === "video" ||
      part.input_video
    ) {
      const payload = part.input_video ?? part.video_url ?? part.video ?? null;
      const source = extractMediaSourceForDisplay(payload);
      attachments.push({
        kind: "video",
        source: source ? resolveAttachmentSource(source, "video") : null,
        label: extractMediaLabel(payload, "video"),
      });
      continue;
    }
  }

  if (attachments.length === 0 && textSegments.length === 0) {
    return null;
  }

  return {
    text: textSegments.join("\n").trim(),
    attachments,
  };
}

interface GenerateTitleArgs {
  threadId: string;
  userContent: string;
  assistantContent: string;
  modelId: string | null;
}

export function ChatPlayground({
  selectedModel,
  selectedModelReady,
  supportsThinking,
  modelLabel,
  modelOptions,
  onSelectModel,
  onOpenModelManager,
  onModelRequired,
}: ChatPlaygroundProps) {
  const [searchParams, setSearchParams] = useSearchParams();
  const activeThreadId = searchParams.get("threadId");

  const [threads, setThreads] = useState<ChatThread[]>([]);
  const [messages, setMessages] = useState<ChatThreadMessageRecord[]>([]);
  const [expandedThoughts, setExpandedThoughts] = useState<
    Record<string, boolean>
  >({});
  const [input, setInput] = useState("");
  const [mediaItems, setMediaItems] = useState<ComposerMediaItem[]>([]);
  const [isThinkingEnabled, setIsThinkingEnabled] = useState(true);
  const [isStreaming, setIsStreaming] = useState(false);
  const [isPreparingThread, setIsPreparingThread] = useState(false);
  const [streamingThreadId, setStreamingThreadId] = useState<string | null>(
    null,
  );
  const [threadsLoading, setThreadsLoading] = useState(true);
  const [messagesLoading, setMessagesLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [stats, setStats] = useState<{
    tokens_generated: number;
    generation_time_ms: number;
  } | null>(null);
  const [isModelMenuOpen, setIsModelMenuOpen] = useState(false);
  const [imagePreview, setImagePreview] = useState<ImagePreviewState | null>(
    null,
  );
  const [deleteTargetThreadId, setDeleteTargetThreadId] = useState<
    string | null
  >(null);
  const [deleteThreadPending, setDeleteThreadPending] = useState(false);

  const initializedRef = useRef(false);
  const activeThreadIdRef = useRef<string | null>(null);
  const isStreamingRef = useRef(false);
  const streamingThreadIdRef = useRef<string | null>(null);
  const threadsRef = useRef<ChatThread[]>([]);
  const titleGenerationInFlightRef = useRef<Set<string>>(new Set());
  const streamAbortRef = useRef<AbortController | null>(null);
  const listEndRef = useRef<HTMLDivElement | null>(null);
  const modelMenuRef = useRef<HTMLDivElement | null>(null);
  const textareaRef = useRef<HTMLTextAreaElement | null>(null);
  const mediaInputRef = useRef<HTMLInputElement | null>(null);
  const mediaItemsRef = useRef<ComposerMediaItem[]>([]);

  const selectedOption = useMemo(() => {
    if (!selectedModel) {
      return null;
    }
    return (
      modelOptions.find((option) => option.value === selectedModel) || null
    );
  }, [selectedModel, modelOptions]);

  const activeThread = useMemo(
    () => threads.find((thread) => thread.id === activeThreadId) ?? null,
    [threads, activeThreadId],
  );
  const deleteTargetThread = useMemo(
    () =>
      deleteTargetThreadId
        ? (threads.find((thread) => thread.id === deleteTargetThreadId) ?? null)
        : null,
    [deleteTargetThreadId, threads],
  );

  const visibleMessages = useMemo(
    () => messages.filter((message) => message.role !== "system"),
    [messages],
  );

  const hasConversation =
    !!activeThreadId &&
    (visibleMessages.length > 0 || isStreaming || messagesLoading);
  const thinkingEnabledForModel = supportsThinking && isThinkingEnabled;
  const selectedModelSupportsMedia = isQwen35ChatModel(selectedModel);
  const renderModelId = activeThread?.model_id ?? selectedModel;
  const implicitOpenThinkTagModel =
    supportsImplicitOpenThinkTagParsing(renderModelId);

  const setActiveThreadInUrl = useCallback(
    (threadId: string | null, replace = false) => {
      const nextSearchParams = new URLSearchParams(searchParams);
      if (threadId) {
        nextSearchParams.set("threadId", threadId);
      } else {
        nextSearchParams.delete("threadId");
      }
      setSearchParams(nextSearchParams, { replace });
    },
    [searchParams, setSearchParams],
  );

  const refreshThreadList = useCallback(
    async (preferredThreadId?: string | null) => {
      try {
        const listedThreads = await api.listChatThreads();
        setThreads(listedThreads);

        const resolvedThreadId = preferredThreadId ?? activeThreadIdRef.current;
        if (
          resolvedThreadId &&
          !listedThreads.some((thread) => thread.id === resolvedThreadId)
        ) {
          setActiveThreadInUrl(null, true);
        }
      } catch {
        // Keep current thread state on refresh failures.
      }
    },
    [setActiveThreadInUrl],
  );

  const maybeGenerateThreadTitle = useCallback(
    async (args: GenerateTitleArgs) => {
      const { threadId, userContent, assistantContent, modelId } = args;

      const existingThread = threadsRef.current.find(
        (thread) => thread.id === threadId,
      );
      if (
        existingThread &&
        existingThread.title !== DEFAULT_THREAD_TITLE &&
        displayThreadTitle(existingThread.title) !== DEFAULT_THREAD_TITLE
      ) {
        return;
      }

      if (titleGenerationInFlightRef.current.has(threadId)) {
        return;
      }

      titleGenerationInFlightRef.current.add(threadId);

      let nextTitle: string | null = null;

      if (modelId) {
        try {
          const titleResponse = await api.createResponse({
            model_id: modelId,
            instructions:
              "Generate a concise chat title (max 8 words) that summarizes the conversation topic. Return only the title text with no quotes, punctuation suffix, or commentary.",
            input: `User: ${userContent}\nAssistant: ${assistantContent}`,
            max_output_tokens: 24,
            store: false,
          });

          nextTitle = normalizeGeneratedThreadTitle(titleResponse.output_text);
        } catch {
          // Fall back to deterministic title below.
        }
      }

      if (!nextTitle) {
        nextTitle = fallbackThreadTitleFromUserMessage(userContent);
      }

      if (!nextTitle || nextTitle === DEFAULT_THREAD_TITLE) {
        titleGenerationInFlightRef.current.delete(threadId);
        return;
      }

      try {
        const updatedThread = await api.updateChatThread(threadId, {
          title: nextTitle,
        });

        setThreads((previous) =>
          previous.map((thread) =>
            thread.id === updatedThread.id ? updatedThread : thread,
          ),
        );
      } catch {
        setThreads((previous) =>
          previous.map((thread) =>
            thread.id === threadId
              ? { ...thread, title: nextTitle ?? thread.title }
              : thread,
          ),
        );
      } finally {
        titleGenerationInFlightRef.current.delete(threadId);
      }
    },
    [],
  );

  useEffect(() => {
    activeThreadIdRef.current = activeThreadId;
  }, [activeThreadId]);

  useEffect(() => {
    isStreamingRef.current = isStreaming;
  }, [isStreaming]);

  useEffect(() => {
    streamingThreadIdRef.current = streamingThreadId;
  }, [streamingThreadId]);

  useEffect(() => {
    threadsRef.current = threads;
  }, [threads]);

  useEffect(() => {
    return () => {
      if (streamAbortRef.current) {
        streamAbortRef.current.abort();
      }
    };
  }, []);

  const revokeMediaPreviews = useCallback((items: ComposerMediaItem[]) => {
    for (const item of items) {
      if (item.previewUrl.startsWith("blob:")) {
        URL.revokeObjectURL(item.previewUrl);
      }
    }
  }, []);

  const clearMediaItems = useCallback(() => {
    setMediaItems((previous) => {
      revokeMediaPreviews(previous);
      return [];
    });
  }, [revokeMediaPreviews]);

  useEffect(() => {
    mediaItemsRef.current = mediaItems;
  }, [mediaItems]);

  useEffect(() => {
    return () => {
      revokeMediaPreviews(mediaItemsRef.current);
    };
  }, [revokeMediaPreviews]);

  useEffect(() => {
    listEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [visibleMessages, isStreaming, activeThreadId]);

  useEffect(() => {
    const onPointerDown = (event: MouseEvent) => {
      if (
        modelMenuRef.current &&
        event.target instanceof Node &&
        !modelMenuRef.current.contains(event.target)
      ) {
        setIsModelMenuOpen(false);
      }
    };

    window.addEventListener("mousedown", onPointerDown);
    return () => window.removeEventListener("mousedown", onPointerDown);
  }, []);

  useEffect(() => {
    if (initializedRef.current) {
      return;
    }
    initializedRef.current = true;

    const initializeThreads = async () => {
      setThreadsLoading(true);
      try {
        const listedThreads = await api.listChatThreads();
        setThreads(listedThreads);

        if (
          activeThreadIdRef.current &&
          !listedThreads.some(
            (thread) => thread.id === activeThreadIdRef.current,
          )
        ) {
          setActiveThreadInUrl(null, true);
        }
      } catch (loadError) {
        setError(getErrorMessage(loadError, "Failed to load chat threads."));
      } finally {
        setThreadsLoading(false);
      }
    };

    void initializeThreads();
  }, [setActiveThreadInUrl]);

  useEffect(() => {
    if (threadsLoading || !activeThreadId) {
      return;
    }

    if (!threads.some((thread) => thread.id === activeThreadId)) {
      setActiveThreadInUrl(null, true);
    }
  }, [activeThreadId, setActiveThreadInUrl, threads, threadsLoading]);

  useEffect(() => {
    if (!activeThreadId) {
      setMessages([]);
      setExpandedThoughts({});
      setStats(null);
      setMessagesLoading(false);
      return;
    }

    const requestedThreadId = activeThreadId;
    if (
      isStreamingRef.current &&
      streamingThreadIdRef.current === requestedThreadId
    ) {
      return;
    }

    let cancelled = false;

    const loadThread = async () => {
      setMessagesLoading(true);
      try {
        const detail = await api.getChatThread(requestedThreadId);
        if (cancelled || activeThreadIdRef.current !== requestedThreadId) {
          return;
        }
        if (
          isStreamingRef.current &&
          streamingThreadIdRef.current === requestedThreadId
        ) {
          return;
        }

        setMessages(detail.messages);
        setExpandedThoughts({});
        setStats(extractLatestStats(detail.messages));
        setThreads((previous) =>
          previous.map((thread) =>
            thread.id === detail.thread.id ? detail.thread : thread,
          ),
        );
      } catch (loadError) {
        if (!cancelled) {
          setError(
            getErrorMessage(loadError, "Failed to load this conversation."),
          );
        }
      } finally {
        if (!cancelled) {
          setMessagesLoading(false);
        }
      }
    };

    void loadThread();

    return () => {
      cancelled = true;
    };
  }, [activeThreadId]);

  const stopStreaming = useCallback(() => {
    if (streamAbortRef.current) {
      streamAbortRef.current.abort();
      streamAbortRef.current = null;
    }

    setIsStreaming(false);
    setStreamingThreadId(null);

    const activeId = activeThreadIdRef.current;
    if (activeId) {
      void api
        .getChatThread(activeId)
        .then((detail) => {
          if (detail.thread.id !== activeThreadIdRef.current) {
            return;
          }
          setMessages(detail.messages);
          setStats(extractLatestStats(detail.messages));
        })
        .catch(() => {
          // Ignore follow-up sync failures after cancel.
        });
      void refreshThreadList(activeId);
    }
  }, [refreshThreadList]);

  const handleCreateThread = useCallback(async () => {
    if (isStreaming) {
      return;
    }

    try {
      const thread = await api.createChatThread({
        model_id: selectedModel ?? undefined,
      });
      setThreads((previous) => [thread, ...previous]);
      setActiveThreadInUrl(thread.id);
      setMessages([]);
      setExpandedThoughts({});
      setStats(null);
      setError(null);
      setInput("");
      clearMediaItems();
    } catch (createError) {
      setError(getErrorMessage(createError, "Failed to create a new chat."));
    }
  }, [clearMediaItems, isStreaming, selectedModel, setActiveThreadInUrl]);

  const openDeleteThreadConfirm = useCallback((threadId: string) => {
    setDeleteTargetThreadId(threadId);
  }, []);

  const closeDeleteThreadConfirm = useCallback(() => {
    if (deleteThreadPending) {
      return;
    }
    setDeleteTargetThreadId(null);
  }, [deleteThreadPending]);

  const handleDeleteThread = useCallback(
    async (threadId: string) => {
      if (isStreaming || isPreparingThread || deleteThreadPending) {
        return;
      }

      setDeleteThreadPending(true);
      try {
        await api.deleteChatThread(threadId);
        setThreads((previous) =>
          previous.filter((thread) => thread.id !== threadId),
        );
        setDeleteTargetThreadId(null);

        if (activeThreadIdRef.current === threadId) {
          setActiveThreadInUrl(null, true);
          setMessages([]);
          setExpandedThoughts({});
          setStats(null);
        }
      } catch (deleteError) {
        setError(getErrorMessage(deleteError, "Failed to delete this chat."));
      } finally {
        setDeleteThreadPending(false);
      }
    },
    [deleteThreadPending, isPreparingThread, isStreaming, setActiveThreadInUrl],
  );

  const confirmDeleteThread = useCallback(() => {
    if (!deleteTargetThreadId) {
      return;
    }
    void handleDeleteThread(deleteTargetThreadId);
  }, [deleteTargetThreadId, handleDeleteThread]);

  const removeMediaItem = useCallback(
    (id: string) => {
      setMediaItems((previous) => {
        const item = previous.find((entry) => entry.id === id);
        if (item && item.previewUrl.startsWith("blob:")) {
          URL.revokeObjectURL(item.previewUrl);
        }
        return previous.filter((entry) => entry.id !== id);
      });
    },
    [setMediaItems],
  );

  const handleSelectMedia = useCallback(
    async (event: ChangeEvent<HTMLInputElement>) => {
      try {
        const pickedFiles = Array.from(event.target.files ?? []);
        event.target.value = "";
        if (pickedFiles.length === 0) {
          return;
        }
        if (!selectedModelSupportsMedia) {
          setError(
            "Image/video chat is currently supported only on Qwen3.5 models.",
          );
          return;
        }

        const availableSlots = Math.max(
          0,
          MAX_MEDIA_ATTACHMENTS - mediaItems.length,
        );
        if (availableSlots <= 0) {
          setError(
            `You can attach up to ${MAX_MEDIA_ATTACHMENTS} files per message.`,
          );
          return;
        }

        let unsupportedCount = 0;
        let oversizeCount = 0;
        const parsedItems: ComposerMediaItem[] = [];

        for (const file of pickedFiles) {
          const mime = file.type || "";
          const kind: ComposerMediaKind | null = mime.startsWith("image/")
            ? "image"
            : mime.startsWith("video/")
              ? "video"
              : null;

          if (!kind) {
            unsupportedCount += 1;
            continue;
          }
          if (file.size > MAX_MEDIA_ATTACHMENT_BYTES) {
            oversizeCount += 1;
            continue;
          }

          const dataUrl = await fileToDataUrl(file);
          parsedItems.push({
            id: createMediaItemId(),
            kind,
            name: file.name || `${kind}-attachment`,
            size: file.size,
            mimeType: mime,
            dataUrl,
            previewUrl: URL.createObjectURL(file),
          });
        }

        if (parsedItems.length === 0) {
          if (unsupportedCount > 0) {
            setError(
              "Only image and video files are supported for chat attachments.",
            );
            return;
          }
          if (oversizeCount > 0) {
            setError(
              `Attachments must be ${MAX_MEDIA_ATTACHMENT_MB} MB or smaller per file.`,
            );
            return;
          }
        }

        const acceptedItems = parsedItems.slice(0, availableSlots);
        const droppedItems = parsedItems.slice(availableSlots);
        revokeMediaPreviews(droppedItems);
        setMediaItems((previous) => [...previous, ...acceptedItems]);

        if (
          pickedFiles.length > availableSlots ||
          unsupportedCount > 0 ||
          oversizeCount > 0
        ) {
          const warnings: string[] = [];
          if (pickedFiles.length > availableSlots) {
            warnings.push(
              `Only ${availableSlots} attachment slot(s) were available.`,
            );
          }
          if (unsupportedCount > 0) {
            warnings.push(`${unsupportedCount} file(s) were not image/video.`);
          }
          if (oversizeCount > 0) {
            warnings.push(
              `${oversizeCount} file(s) exceeded ${MAX_MEDIA_ATTACHMENT_MB} MB.`,
            );
          }
          setError(warnings.join(" "));
        } else {
          setError(null);
        }
      } catch (error) {
        setError(
          getErrorMessage(error, "Failed to read selected media files."),
        );
      }
    },
    [mediaItems.length, revokeMediaPreviews, selectedModelSupportsMedia],
  );

  const sendMessage = async () => {
    const text = input.trim();
    if (
      (!text && mediaItems.length === 0) ||
      isStreaming ||
      isPreparingThread
    ) {
      return;
    }

    if (!selectedModel || !selectedModelReady) {
      onModelRequired();
      return;
    }

    if (mediaItems.length > 0 && !isQwen35ChatModel(selectedModel)) {
      setError(
        "Image/video chat is currently supported only on Qwen3.5 models.",
      );
      return;
    }

    let targetThreadId = activeThreadId;
    if (!targetThreadId) {
      setIsPreparingThread(true);
      try {
        const createdThread = await api.createChatThread({
          model_id: selectedModel ?? undefined,
        });
        setThreads((previous) => [createdThread, ...previous]);
        setActiveThreadInUrl(createdThread.id);
        setMessages([]);
        setExpandedThoughts({});
        setStats(null);
        targetThreadId = createdThread.id;
      } catch (threadError) {
        setError(getErrorMessage(threadError, "Failed to create a new chat."));
        setIsPreparingThread(false);
        return;
      }
      setIsPreparingThread(false);
    }

    if (!targetThreadId) {
      return;
    }

    setError(null);
    setStats(null);

    const isFirstTurn =
      targetThreadId === activeThreadId
        ? messages.filter((message) => message.role === "user").length === 0
        : true;

    const timestamp = Date.now();
    const userTempId = `tmp-user-${timestamp}`;
    const assistantTempId = `tmp-assistant-${timestamp}`;

    const userDisplayContent = buildUserDisplayContent(text, mediaItems);
    const contentParts = buildThreadContentParts(text, mediaItems);
    const optimisticUserMessage: ChatThreadMessageRecord = {
      id: userTempId,
      thread_id: targetThreadId,
      role: "user",
      content: userDisplayContent,
      created_at: timestamp,
      tokens_generated: null,
      generation_time_ms: null,
    };

    const optimisticAssistantMessage: ChatThreadMessageRecord = {
      id: assistantTempId,
      thread_id: targetThreadId,
      role: "assistant",
      content: "",
      created_at: timestamp + 1,
      tokens_generated: null,
      generation_time_ms: null,
    };

    const qwen35ThinkingControl = isQwen35ChatModel(selectedModel)
      ? thinkingEnabledForModel
      : undefined;
    const systemPrompt = isQwen35ChatModel(selectedModel)
      ? "You are a helpful assistant."
      : thinkingEnabledForModel
        ? isLfm25ThinkingModel(selectedModel)
          ? "You are a helpful assistant."
          : THINKING_SYSTEM_PROMPT.content
        : DEFAULT_SYSTEM_PROMPT.content;

    setMessages((previous) => [
      ...previous,
      optimisticUserMessage,
      optimisticAssistantMessage,
    ]);
    setInput("");
    clearMediaItems();
    setIsStreaming(true);
    setStreamingThreadId(targetThreadId);

    streamAbortRef.current = api.sendChatThreadMessageStream(
      targetThreadId,
      {
        model_id: selectedModel,
        content: text,
        content_parts: contentParts,
        system_prompt: systemPrompt,
        enable_thinking: qwen35ThinkingControl,
      },
      {
        onStart: ({ userMessage }) => {
          setMessages((previous) => {
            let replaced = false;
            const updated = previous.map((message) => {
              if (message.id === userTempId) {
                replaced = true;
                return userMessage;
              }
              return message;
            });

            if (
              replaced ||
              updated.some((message) => message.id === userMessage.id)
            ) {
              return updated;
            }

            return [...updated, userMessage];
          });
        },
        onDelta: (delta) => {
          setMessages((previous) => {
            let updatedAssistant = false;
            const updated = previous.map((message) => {
              if (message.id === assistantTempId) {
                updatedAssistant = true;
                return { ...message, content: `${message.content}${delta}` };
              }
              return message;
            });

            if (updatedAssistant) {
              return updated;
            }

            if (
              updated.some(
                (message) => message.id === optimisticAssistantMessage.id,
              )
            ) {
              return updated;
            }

            return [
              ...updated,
              { ...optimisticAssistantMessage, content: delta },
            ];
          });
        },
        onDone: ({ assistantMessage, stats: streamStats, modelId }) => {
          setMessages((previous) => {
            let replaced = false;
            const updated = previous.map((message) => {
              if (message.id === assistantTempId) {
                replaced = true;
                return assistantMessage;
              }
              return message;
            });

            if (
              replaced ||
              updated.some((message) => message.id === assistantMessage.id)
            ) {
              return updated;
            }

            return [...updated, assistantMessage];
          });
          setStats(streamStats);

          if (isFirstTurn) {
            void maybeGenerateThreadTitle({
              threadId: targetThreadId,
              userContent: userDisplayContent,
              assistantContent: assistantMessage.content,
              modelId,
            });
          }
        },
        onError: (message) => {
          setError(message);
          setMessages((previous) =>
            previous.filter(
              (entry) =>
                !(
                  entry.id === assistantTempId &&
                  entry.content.trim().length === 0
                ),
            ),
          );
        },
        onClose: () => {
          setIsStreaming(false);
          setStreamingThreadId(null);
          streamAbortRef.current = null;
          void refreshThreadList(targetThreadId);
          void api
            .getChatThread(targetThreadId)
            .then((detail) => {
              if (activeThreadIdRef.current !== targetThreadId) {
                return;
              }
              setMessages(detail.messages);
              setStats(extractLatestStats(detail.messages));
            })
            .catch(() => {
              // Ignore follow-up sync failures after stream close.
            });
        },
      },
    );
  };

  const handleOpenModels = () => {
    setIsModelMenuOpen(false);
    onOpenModelManager();
  };

  const renderModelSelector = () => (
    <div
      className={cn(
        "relative z-40 inline-block w-[240px] sm:w-[300px] max-w-[calc(100vw-9rem)]",
      )}
      ref={modelMenuRef}
    >
      <Button
        variant="outline"
        onClick={() => setIsModelMenuOpen((previous) => !previous)}
        className={cn(
          "w-full justify-between font-normal h-9",
          selectedOption?.isReady ? "border-primary/20 bg-primary/5" : "",
        )}
      >
        <span className="flex-1 min-w-0 truncate text-left">
          {selectedOption?.label || "Select model"}
        </span>
        <ChevronDown className="w-3.5 h-3.5 shrink-0 opacity-50" />
      </Button>

      <AnimatePresence>
        {isModelMenuOpen && (
          <motion.div
            initial={{ opacity: 0, y: 6, scale: 0.98 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: 6, scale: 0.98 }}
            transition={{ duration: 0.16 }}
            className="absolute left-0 right-0 bottom-11 rounded-md border bg-popover text-popover-foreground p-1 shadow-md z-[90]"
          >
            <div className="max-h-64 overflow-y-auto">
              {modelOptions.map((option) => (
                <button
                  key={option.value}
                  onClick={() => {
                    onSelectModel(option.value);
                    setIsModelMenuOpen(false);
                  }}
                  className={cn(
                    "relative flex w-full cursor-default select-none items-center rounded-sm py-1.5 px-2 text-sm outline-none transition-colors hover:bg-accent hover:text-accent-foreground focus:bg-accent focus:text-accent-foreground data-[disabled]:pointer-events-none data-[disabled]:opacity-50",
                    selectedOption?.value === option.value &&
                      "bg-accent text-accent-foreground",
                  )}
                >
                  <div className="flex flex-col items-start min-w-0">
                    <span className="truncate w-full text-left font-medium">
                      {option.label}
                    </span>
                    <span
                      className={cn(
                        "mt-1 text-[10px] uppercase tracking-wider font-semibold",
                        option.isReady
                          ? "text-green-500"
                          : "text-muted-foreground",
                      )}
                    >
                      {option.statusLabel}
                    </span>
                  </div>
                </button>
              ))}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );

  const renderComposer = (centered = false) => (
    <div
      className={cn(
        "relative rounded-xl border border-[var(--border-muted)] bg-background shadow-sm overflow-visible",
        centered && "max-w-3xl mx-auto shadow-md",
      )}
    >
      {mediaItems.length > 0 && (
        <div className="px-4 pt-3 pb-2 border-b border-[var(--border-muted)] bg-muted/20">
          <div className="flex items-center justify-between gap-2 mb-2">
            <p className="text-[11px] font-medium text-muted-foreground">
              Attachments ({mediaItems.length}/{MAX_MEDIA_ATTACHMENTS})
            </p>
            <Button
              variant="ghost"
              size="sm"
              onClick={clearMediaItems}
              disabled={isStreaming || isPreparingThread}
              className="h-7 px-2 text-[11px] text-muted-foreground"
            >
              Clear
            </Button>
          </div>
          <div className="flex flex-wrap gap-2">
            {mediaItems.map((item) => (
              <div
                key={item.id}
                className="relative w-[92px] h-[92px] rounded-md border border-[var(--border-muted)] overflow-hidden bg-muted/40"
              >
                {item.kind === "image" ? (
                  <img
                    src={item.previewUrl}
                    alt={item.name}
                    className="w-full h-full object-cover"
                  />
                ) : (
                  <>
                    <video
                      src={item.previewUrl}
                      className="w-full h-full object-cover"
                      preload="metadata"
                      muted
                    />
                    <div className="absolute inset-0 bg-black/25 flex items-center justify-center">
                      <Film className="w-4 h-4 text-white" />
                    </div>
                  </>
                )}
                <button
                  type="button"
                  onClick={() => removeMediaItem(item.id)}
                  disabled={isStreaming || isPreparingThread}
                  className="absolute top-1 right-1 inline-flex h-5 w-5 items-center justify-center rounded-full border border-black/40 bg-black/70 text-white transition-colors hover:bg-black/90 disabled:opacity-50"
                  title={`Remove ${item.name}`}
                >
                  <X className="w-3 h-3" />
                </button>
                <div className="absolute left-0 right-0 bottom-0 bg-black/65 px-1.5 py-1 text-[10px] leading-tight text-white">
                  <p className="truncate inline-flex items-center gap-1">
                    {item.kind === "image" ? (
                      <ImageIcon className="w-3 h-3 shrink-0" />
                    ) : (
                      <Film className="w-3 h-3 shrink-0" />
                    )}
                    <span className="truncate">{item.name}</span>
                  </p>
                  <p className="text-white/80">{formatBytes(item.size)}</p>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      <textarea
        ref={textareaRef}
        value={input}
        onChange={(event) => setInput(event.target.value)}
        onKeyDown={(event) => {
          if (event.key === "Enter" && !event.shiftKey) {
            event.preventDefault();
            void sendMessage();
          }
        }}
        placeholder={
          !activeThreadId
            ? "Ask anything..."
            : !selectedModel
              ? "Choose a model and ask anything..."
              : !selectedModelReady
                ? "Model selected but not loaded. Open Models to load it."
                : "Ask anything..."
        }
        className={cn(
          "w-full bg-transparent px-4 pt-4 pb-3 text-[0.9375rem] resize-none focus:outline-none placeholder:text-[var(--text-muted)]",
          centered ? "min-h-[140px]" : "min-h-[104px]",
        )}
        disabled={isStreaming || isPreparingThread}
      />

      <div className="flex flex-wrap items-center justify-between gap-2 px-3 pb-3 pt-1 border-t border-[var(--border-muted)]/30 mt-1">
        <div className="flex items-center gap-2 flex-wrap">
          <Button
            variant="ghost"
            size="sm"
            onClick={() => mediaInputRef.current?.click()}
            disabled={
              isStreaming ||
              isPreparingThread ||
              mediaItems.length >= MAX_MEDIA_ATTACHMENTS ||
              !selectedModelSupportsMedia
            }
            className="h-8 gap-1.5 text-xs text-muted-foreground hover:text-foreground"
            title={
              selectedModelSupportsMedia
                ? "Attach image or video"
                : "Image/video upload is available only for Qwen3.5 models"
            }
          >
            <Paperclip className="w-3.5 h-3.5" />
            Add Media
          </Button>
          <Button
            variant="ghost"
            size="sm"
            onClick={handleOpenModels}
            className="h-8 gap-1.5 text-xs text-muted-foreground hover:text-foreground"
          >
            <Settings2 className="w-3.5 h-3.5" />
            Models
          </Button>
          {supportsThinking && (
            <Button
              variant={thinkingEnabledForModel ? "secondary" : "ghost"}
              size="sm"
              onClick={() => setIsThinkingEnabled((previous) => !previous)}
              disabled={isStreaming || isPreparingThread}
              className={cn(
                "h-8 gap-1.5 text-xs",
                !thinkingEnabledForModel &&
                  "text-muted-foreground hover:text-foreground",
              )}
              title={
                thinkingEnabledForModel
                  ? "Thinking mode is enabled"
                  : "Thinking mode is disabled"
              }
            >
              <Brain className="w-3.5 h-3.5" />
              Thinking {thinkingEnabledForModel ? "On" : "Off"}
            </Button>
          )}
        </div>

        <div className="flex items-center gap-2 w-full sm:w-auto justify-end flex-wrap sm:flex-nowrap">
          {renderModelSelector()}

          <Button
            onClick={isStreaming ? stopStreaming : () => void sendMessage()}
            disabled={
              isPreparingThread ||
              (!isStreaming && !input.trim() && mediaItems.length === 0)
            }
            variant={isStreaming ? "destructive" : "default"}
            size="sm"
            className="h-9 gap-1.5 font-medium px-4"
          >
            {isStreaming ? (
              <Square className="w-3.5 h-3.5" />
            ) : isPreparingThread ? (
              <Loader2 className="w-3.5 h-3.5 animate-spin" />
            ) : (
              <Send className="w-3.5 h-3.5" />
            )}
            {isStreaming
              ? "Cancel"
              : isPreparingThread
                ? "Starting..."
                : "Send"}
          </Button>
        </div>
      </div>
    </div>
  );

  return (
    <div className="relative flex flex-col lg:flex-row gap-4 h-[calc(100dvh-12rem)] lg:h-[calc(100dvh-11.75rem)]">
      <input
        ref={mediaInputRef}
        type="file"
        accept="image/*,video/*"
        multiple
        disabled={!selectedModelSupportsMedia}
        onChange={handleSelectMedia}
        className="hidden"
      />
      <aside className="w-full lg:w-80 lg:min-w-[20rem] max-h-[38dvh] lg:max-h-none shrink-0 card app-sidebar-panel p-4 sm:p-5 flex flex-col overflow-hidden">
        <div className="flex items-start justify-between gap-3 mb-3">
          <div>
            <div className="inline-flex items-center gap-2 app-sidebar-header-eyebrow">
              <History className="w-3.5 h-3.5" />
              History
            </div>
            <h2 className="app-sidebar-header-title">Chat History</h2>
            <p className="app-sidebar-header-count">
              {threads.length} {threads.length === 1 ? "thread" : "threads"}
            </p>
          </div>
          <button
            onClick={handleCreateThread}
            disabled={isStreaming || isPreparingThread}
            className="btn btn-ghost app-sidebar-refresh-btn"
          >
            <Plus className="w-3.5 h-3.5" />
            New
          </button>
        </div>

        <div className="app-sidebar-list">
          {threadsLoading ? (
            <div className="app-sidebar-loading">
              <Loader2 className="w-3.5 h-3.5 animate-spin" />
              Loading chats...
            </div>
          ) : threads.length === 0 ? (
            <div className="app-sidebar-empty">
              No chats yet. Create one to begin.
            </div>
          ) : (
            <div className="flex flex-col gap-2.5">
              {threads.map((thread) => {
                const isActive = thread.id === activeThreadId;
                const preview = threadPreviewFromContent(
                  thread.last_message_preview,
                );

                return (
                  <div
                    key={thread.id}
                    role="button"
                    tabIndex={0}
                    onClick={() => {
                      if (isStreaming || isPreparingThread) {
                        return;
                      }
                      setActiveThreadInUrl(thread.id);
                      setError(null);
                    }}
                    onKeyDown={(event) => {
                      if (event.key === "Enter" || event.key === " ") {
                        event.preventDefault();
                        if (isStreaming || isPreparingThread) {
                          return;
                        }
                        setActiveThreadInUrl(thread.id);
                        setError(null);
                      }
                    }}
                    className={cn(
                      "group app-sidebar-row",
                      isActive
                        ? "app-sidebar-row-active"
                        : "app-sidebar-row-idle",
                      (isStreaming || isPreparingThread) && "opacity-70",
                    )}
                  >
                    <div className="flex items-center justify-between gap-2">
                      <p className="app-sidebar-row-label truncate">
                        {displayThreadTitle(thread.title)}
                      </p>
                      <div className="inline-flex items-center gap-1.5 shrink-0">
                        <span className="app-sidebar-row-meta">
                          {formatThreadTimestamp(thread.updated_at)}
                        </span>
                        <button
                          onClick={(event) => {
                            event.preventDefault();
                            event.stopPropagation();
                            if (isStreaming || isPreparingThread) {
                              return;
                            }
                            openDeleteThreadConfirm(thread.id);
                          }}
                          disabled={isStreaming || isPreparingThread}
                          className={cn(
                            "app-sidebar-delete-btn",
                            (isStreaming || isPreparingThread) && "opacity-50",
                          )}
                          title="Delete chat"
                        >
                          <Trash2 className="w-3.5 h-3.5" />
                        </button>
                      </div>
                    </div>
                    <p
                      className="app-sidebar-row-preview"
                      style={{
                        display: "-webkit-box",
                        WebkitLineClamp: 3,
                        WebkitBoxOrient: "vertical",
                        overflow: "hidden",
                      }}
                    >
                      {preview}
                    </p>
                  </div>
                );
              })}
            </div>
          )}
        </div>
      </aside>

      <div className="relative flex-1 min-h-0 flex flex-col">
        {!hasConversation ? (
          <div className="relative flex-1 flex items-center justify-center px-1 sm:px-4">
            <div className="w-full max-w-3xl">
              <motion.div
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.22 }}
              >
                {renderComposer(true)}
              </motion.div>

              <div className="mt-4 text-center text-xs text-muted-foreground min-h-[18px]">
                {!activeThreadId ? (
                  <span>
                    No active chat selected. Start typing and send to create a
                    new chat.
                  </span>
                ) : selectedModel ? (
                  selectedModelReady ? (
                    <span className="text-foreground/80 font-medium">
                      {modelLabel || selectedModel} is loaded and ready.
                    </span>
                  ) : (
                    <span>
                      {modelLabel || selectedModel} is selected but not loaded.
                    </span>
                  )
                ) : (
                  <span>No model selected.</span>
                )}
              </div>

              <AnimatePresence>
                {error && (
                  <motion.div
                    initial={{ opacity: 0, height: 0 }}
                    animate={{ opacity: 1, height: "auto" }}
                    exit={{ opacity: 0, height: 0 }}
                    className="mt-3 mx-auto max-w-3xl p-2 rounded bg-red-950/50 border border-red-900/50 text-red-300 text-xs"
                  >
                    {error}
                  </motion.div>
                )}
              </AnimatePresence>
            </div>
          </div>
        ) : (
          <div className="relative flex-1 min-h-0 flex flex-col overflow-hidden bg-card border border-[var(--border-muted)] rounded-xl shadow-sm">
            <div className="px-4 sm:px-6 py-4 border-b border-[var(--border-muted)] flex items-center justify-between gap-3 bg-muted/20">
              <div>
                <h2 className="text-sm font-semibold tracking-tight">
                  {activeThread
                    ? displayThreadTitle(activeThread.title)
                    : "Conversation"}
                </h2>
                <p className="text-xs text-muted-foreground mt-0.5">
                  {selectedModelReady
                    ? `Using ${modelLabel || selectedModel}`
                    : "Model not loaded"}
                </p>
              </div>
              <div className="text-xs text-muted-foreground inline-flex items-center gap-1.5 font-medium bg-muted px-2 py-1 rounded-md">
                <MessageSquare className="w-3.5 h-3.5" />
                {visibleMessages.length} messages
              </div>
            </div>

            <div className="relative flex-1 min-h-0 bg-background/50">
              <div className="h-full overflow-y-auto px-4 sm:px-6 pb-64 pt-6 scrollbar-thin">
                {messagesLoading ? (
                  <div className="max-w-4xl mx-auto p-4 text-center text-xs text-muted-foreground flex items-center justify-center gap-2">
                    <Loader2 className="w-4 h-4 animate-spin" />
                    Loading conversation...
                  </div>
                ) : (
                  <div className="max-w-4xl mx-auto space-y-6">
                    {visibleMessages.map((message, index) => {
                      const isUser = message.role === "user";
                      const parsedUserMessage = isUser
                        ? parseUserMessageDisplayFromContentParts(message)
                        : null;
                      const isLastAssistant =
                        !isUser &&
                        index === visibleMessages.length - 1 &&
                        isStreaming &&
                        streamingThreadId === activeThreadId;
                      const assistantDisplayContent = isUser
                        ? message.content
                        : thinkingEnabledForModel
                          ? message.content
                          : stripThinkingArtifacts(message.content || "");
                      const parsed = isUser
                        ? null
                        : thinkingEnabledForModel
                          ? parseAssistantContent(message.content || "", {
                              implicitOpenThinkTag:
                                implicitOpenThinkTagModel &&
                                thinkingEnabledForModel,
                            })
                          : null;
                      const messageKey = message.id;
                      const isThoughtExpanded = !!expandedThoughts[messageKey];
                      const showStreamingThinking =
                        !isUser &&
                        !!parsed &&
                        isLastAssistant &&
                        parsed.thinking.length > 0 &&
                        (parsed.hasIncompleteThink ||
                          parsed.answer.length === 0);
                      const showAnswerOnly =
                        !isUser &&
                        !!parsed &&
                        parsed.answer.length > 0 &&
                        parsed.hasThink &&
                        !showStreamingThinking;

                      return (
                        <motion.div
                          key={messageKey}
                          initial={{ opacity: 0, y: 8 }}
                          animate={{ opacity: 1, y: 0 }}
                          className={cn(
                            "flex gap-4 sm:gap-6",
                            isUser && "flex-row-reverse",
                          )}
                        >
                          <div
                            className={cn(
                              "w-8 h-8 rounded-full flex items-center justify-center flex-shrink-0 mt-1",
                              isUser
                                ? "bg-[var(--bg-surface-3)] text-[var(--text-primary)] border border-[var(--border-muted)]"
                                : "bg-[var(--accent-solid)] text-[var(--text-on-accent)] shadow-sm",
                            )}
                          >
                            {isUser ? (
                              <User className="w-4 h-4" />
                            ) : (
                              <Bot className="w-4 h-4" />
                            )}
                          </div>

                          <div
                            className={cn(
                              "max-w-[85%] text-[0.9375rem] leading-relaxed break-words flex flex-col",
                              isUser ? "items-end" : "items-start",
                            )}
                          >
                            <div
                              className={cn(
                                "rounded-2xl px-5 py-3 shadow-sm",
                                isUser
                                  ? "bg-[var(--bg-surface-3)] text-[var(--text-primary)] rounded-tr-sm border border-[var(--border-muted)]"
                                  : "bg-transparent text-[var(--text-primary)]",
                              )}
                            >
                              {isUser ? (
                                <>
                                  {(!parsedUserMessage ||
                                    parsedUserMessage.text ||
                                    parsedUserMessage.attachments.length ===
                                      0) && (
                                    <MarkdownContent
                                      content={
                                        parsedUserMessage &&
                                        parsedUserMessage.attachments.length > 0
                                          ? parsedUserMessage.text
                                          : message.content
                                      }
                                      className="prose-p:leading-relaxed prose-pre:bg-black/10 dark:prose-pre:bg-white/10 prose-pre:border-none prose-a:text-primary-foreground underline"
                                    />
                                  )}
                                  {parsedUserMessage &&
                                    parsedUserMessage.attachments.length >
                                      0 && (
                                      <div
                                        className={cn(
                                          "flex flex-wrap gap-2",
                                          parsedUserMessage.text && "mt-2",
                                        )}
                                      >
                                        {parsedUserMessage.attachments.map(
                                          (attachment, attachmentIndex) => (
                                            <div
                                              key={`${messageKey}-attachment-${attachmentIndex}`}
                                            >
                                              {attachment.kind === "image" &&
                                              attachment.source ? (
                                                <button
                                                  type="button"
                                                  onClick={() =>
                                                    setImagePreview({
                                                      source:
                                                        attachment.source!,
                                                      label: attachment.label,
                                                    })
                                                  }
                                                  title={`Open ${attachment.label}`}
                                                  className="block h-16 w-16 overflow-hidden rounded-md border border-primary-foreground/20 bg-primary-foreground/10 hover:bg-primary-foreground/15 transition-colors focus:outline-none focus:ring-2 focus:ring-primary-foreground/70"
                                                >
                                                  <img
                                                    src={attachment.source}
                                                    alt={attachment.label}
                                                    loading="lazy"
                                                    className="h-full w-full object-cover"
                                                  />
                                                </button>
                                              ) : (
                                                <div className="inline-flex max-w-[220px] items-center gap-1.5 rounded-md border border-primary-foreground/20 bg-primary-foreground/10 px-2 py-1 text-xs text-primary-foreground/90">
                                                  {attachment.kind ===
                                                  "image" ? (
                                                    <ImageIcon className="h-3.5 w-3.5 shrink-0" />
                                                  ) : (
                                                    <Film className="h-3.5 w-3.5 shrink-0" />
                                                  )}
                                                  <span className="truncate">
                                                    {attachment.label}
                                                  </span>
                                                </div>
                                              )}
                                            </div>
                                          ),
                                        )}
                                      </div>
                                    )}
                                </>
                              ) : (
                                <>
                                  {showStreamingThinking && parsed && (
                                    <div className="mb-3 rounded-lg bg-muted/50 border border-[var(--border-muted)] px-3 py-2 text-xs text-muted-foreground">
                                      <div className="mb-2 flex items-center gap-1.5 uppercase tracking-wider text-[10px] font-semibold">
                                        <Loader2 className="w-3 h-3 animate-spin text-primary" />
                                        Thinking
                                      </div>
                                      <div className="whitespace-pre-wrap font-mono text-[11px] leading-relaxed">
                                        {parsed.thinking}
                                      </div>
                                    </div>
                                  )}

                                  {parsed && parsed.answer.length > 0 ? (
                                    <MarkdownContent content={parsed.answer} />
                                  ) : parsed && parsed.hasThink ? (
                                    <div className="italic text-muted-foreground opacity-70">
                                      {isLastAssistant
                                        ? "Thinking..."
                                        : "No final answer was generated."}
                                    </div>
                                  ) : (
                                    <MarkdownContent
                                      content={assistantDisplayContent}
                                    />
                                  )}

                                  {parsed &&
                                    parsed.hasThink &&
                                    !showStreamingThinking && (
                                      <div className="mt-3 border-t pt-2">
                                        <button
                                          onClick={() =>
                                            setExpandedThoughts((previous) => ({
                                              ...previous,
                                              [messageKey]:
                                                !previous[messageKey],
                                            }))
                                          }
                                          className="inline-flex items-center gap-1.5 text-xs text-muted-foreground hover:text-foreground font-medium transition-colors"
                                        >
                                          {isThoughtExpanded ? (
                                            <ChevronDown className="w-3 h-3" />
                                          ) : (
                                            <ChevronRight className="w-3 h-3" />
                                          )}
                                          {isThoughtExpanded
                                            ? "Hide thought process"
                                            : "Show thought process"}
                                        </button>
                                      </div>
                                    )}

                                  {parsed &&
                                    parsed.hasThink &&
                                    !showStreamingThinking &&
                                    isThoughtExpanded && (
                                      <div className="mt-2 rounded-lg bg-muted/30 border border-[var(--border-muted)] px-3 py-2 text-xs whitespace-pre-wrap font-mono text-[11px] leading-relaxed text-muted-foreground">
                                        {parsed.thinking}
                                      </div>
                                    )}

                                  {isLastAssistant &&
                                    ((parsed && parsed.answer.length > 0) ||
                                      !showAnswerOnly) && (
                                      <span className="inline-flex items-center ml-2 align-middle">
                                        <span className="w-1.5 h-4 bg-primary animate-pulse inline-block" />
                                      </span>
                                    )}
                                </>
                              )}
                            </div>
                          </div>
                        </motion.div>
                      );
                    })}
                    <div ref={listEndRef} />
                  </div>
                )}
              </div>

              <div className="absolute z-30 bottom-0 left-0 right-0 px-4 sm:px-6 pb-6 pt-4 bg-gradient-to-t from-background via-background/95 to-transparent pointer-events-none">
                <div className="max-w-4xl mx-auto pointer-events-auto">
                  <AnimatePresence>
                    {error && (
                      <motion.div
                        initial={{ opacity: 0, height: 0, y: 10 }}
                        animate={{ opacity: 1, height: "auto", y: 0 }}
                        exit={{ opacity: 0, height: 0, y: 10 }}
                        className="mb-4 p-3 rounded-lg bg-destructive text-destructive-foreground shadow-lg text-sm font-medium flex items-center gap-2"
                      >
                        {error}
                      </motion.div>
                    )}
                  </AnimatePresence>

                  {stats && !isStreaming && (
                    <div className="mb-3 text-[11px] font-medium text-muted-foreground flex items-center justify-center gap-3">
                      <span className="bg-muted px-2 py-0.5 rounded-md border border-[var(--border-muted)] shadow-sm">
                        {stats.tokens_generated} tokens
                      </span>
                      <span className="bg-muted px-2 py-0.5 rounded-md border border-[var(--border-muted)] shadow-sm">
                        {Math.round(stats.generation_time_ms)} ms
                      </span>
                      <span className="bg-muted px-2 py-0.5 rounded-md border border-[var(--border-muted)] shadow-sm">
                        {Math.round(
                          stats.tokens_generated /
                            (stats.generation_time_ms / 1000),
                        )}{" "}
                        t/s
                      </span>
                    </div>
                  )}

                  {renderComposer(false)}
                </div>
              </div>
            </div>
          </div>
        )}
      </div>

      <AnimatePresence>
        {deleteTargetThread && (
          <motion.div
            className="fixed inset-0 z-[60] bg-black/75 p-4 backdrop-blur-sm"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            onClick={closeDeleteThreadConfirm}
          >
            <motion.div
              initial={{ y: 10, opacity: 0, scale: 0.98 }}
              animate={{ y: 0, opacity: 1, scale: 1 }}
              exit={{ y: 10, opacity: 0, scale: 0.98 }}
              transition={{ duration: 0.16 }}
              className="mx-auto mt-[18vh] max-w-md rounded-xl border border-[var(--border-strong)] bg-[var(--bg-surface-1)] p-5"
              onClick={(event) => event.stopPropagation()}
            >
              <div className="flex items-start gap-3">
                <div className="mt-0.5 rounded-full border border-[var(--danger-border)] bg-[var(--danger-bg)] p-2 text-[var(--danger-text)]">
                  <AlertTriangle className="h-4 w-4" />
                </div>
                <div className="min-w-0 flex-1">
                  <h3 className="text-sm font-semibold text-[var(--text-primary)]">
                    Delete chat thread?
                  </h3>
                  <p className="mt-1 text-sm text-[var(--text-muted)]">
                    This permanently removes the selected conversation and all
                    of its messages.
                  </p>
                  <p className="mt-2 truncate text-xs text-[var(--text-subtle)]">
                    {displayThreadTitle(deleteTargetThread.title)}
                  </p>
                </div>
              </div>

              <div className="mt-5 flex items-center justify-end gap-2">
                <button
                  onClick={closeDeleteThreadConfirm}
                  className="rounded-md border border-[var(--border-muted)] bg-[var(--bg-surface-2)] px-3 py-1.5 text-xs font-medium text-[var(--text-secondary)] transition-colors hover:bg-[var(--bg-surface-3)] disabled:opacity-50"
                  disabled={deleteThreadPending}
                >
                  Cancel
                </button>
                <button
                  onClick={confirmDeleteThread}
                  className="flex items-center gap-1.5 rounded-md border border-[var(--danger-border)] bg-[var(--danger-bg)] px-3 py-1.5 text-xs font-medium text-[var(--danger-text)] transition-colors hover:bg-[var(--danger-bg-hover)] disabled:opacity-50"
                  disabled={deleteThreadPending}
                >
                  {deleteThreadPending ? (
                    <Loader2 className="h-3.5 w-3.5 animate-spin" />
                  ) : (
                    <Trash2 className="h-3.5 w-3.5" />
                  )}
                  Delete thread
                </button>
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>

      <Dialog
        open={!!imagePreview}
        onOpenChange={(open) => {
          if (!open) {
            setImagePreview(null);
          }
        }}
      >
        <DialogContent className="max-w-[min(92vw,1100px)] p-3 sm:p-4 bg-background/98 border-[var(--border-muted)]">
          <DialogTitle className="sr-only">
            {imagePreview?.label || "Image preview"}
          </DialogTitle>
          {imagePreview && (
            <div className="flex max-h-[80vh] items-center justify-center overflow-auto rounded-lg border border-[var(--border-muted)] bg-muted/20 p-2">
              <img
                src={imagePreview.source}
                alt={imagePreview.label}
                className="max-h-[76vh] w-auto max-w-full rounded object-contain"
              />
            </div>
          )}
        </DialogContent>
      </Dialog>
    </div>
  );
}
