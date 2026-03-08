import type {
  ChatMessage,
  ChatThreadContentPart,
  ChatThreadMessageRecord,
} from "@/api";
import { API_BASE_URL } from "@/shared/config/runtime";

export interface ModelOption {
  value: string;
  label: string;
  statusLabel: string;
  isReady: boolean;
}

export interface ChatPlaygroundProps {
  selectedModel: string | null;
  selectedModelReady: boolean;
  supportsThinking: boolean;
  modelLabel?: string | null;
  modelOptions: ModelOption[];
  onSelectModel: (variant: string) => void;
  onOpenModelManager: () => void;
  onModelRequired: () => void;
}

export const THINKING_SYSTEM_PROMPT: ChatMessage = {
  role: "system",
  content:
    "You are a helpful assistant. Always reason inside <think>...</think> before giving the final answer. Keep thinking concise, always close </think>, then provide a clear final answer outside the tags.",
};

export const DEFAULT_SYSTEM_PROMPT: ChatMessage = {
  role: "system",
  content:
    "You are a helpful assistant. Provide only the final answer and do not output <think> tags or internal reasoning.",
};

export const DEFAULT_THREAD_TITLE = "New chat";
const MAX_THREAD_TITLE_CHARS = 80;
export const MAX_MEDIA_ATTACHMENTS = 4;
export const MAX_MEDIA_ATTACHMENT_MB = 32;
export const MAX_MEDIA_ATTACHMENT_BYTES = MAX_MEDIA_ATTACHMENT_MB * 1024 * 1024;

export type ComposerMediaKind = "image" | "video";

export interface ComposerMediaItem {
  id: string;
  kind: ComposerMediaKind;
  name: string;
  size: number;
  mimeType: string;
  dataUrl: string;
  previewUrl: string;
}

export interface ParsedAssistantContent {
  thinking: string;
  answer: string;
  hasThink: boolean;
  hasIncompleteThink: boolean;
}

interface ParseAssistantContentOptions {
  implicitOpenThinkTag?: boolean;
  treatNoTagAsThinking?: boolean;
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

export interface ImagePreviewState {
  source: string;
  label: string;
}

export interface GenerateTitleArgs {
  threadId: string;
  userContent: string;
  assistantContent: string;
  modelId: string | null;
}

export function parseAssistantContent(
  content: string,
  options?: ParseAssistantContentOptions,
): ParsedAssistantContent {
  const openTag = "<think>";
  const closeTag = "</think>";
  const raw = content ?? "";
  const implicitOpenThinkTag = !!options?.implicitOpenThinkTag;
  const treatNoTagAsThinking = options?.treatNoTagAsThinking ?? true;
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

  if (
    implicitOpenThinkTag &&
    treatNoTagAsThinking &&
    firstOpen === -1 &&
    firstClose === -1
  ) {
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

export function getErrorMessage(error: unknown, fallback: string): string {
  if (error instanceof Error && error.message.trim()) {
    return error.message;
  }
  return fallback;
}

export function formatThreadTimestamp(timestamp: number): string {
  const date = new Date(timestamp);
  const now = new Date();
  const isToday = date.toDateString() === now.toDateString();
  if (isToday) {
    return date.toLocaleTimeString([], { hour: "numeric", minute: "2-digit" });
  }
  return date.toLocaleDateString([], { month: "short", day: "numeric" });
}

export function extractLatestStats(messages: ChatThreadMessageRecord[]): {
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

export function stripThinkingArtifacts(text: string): string {
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

export function displayThreadTitle(
  rawTitle: string | null | undefined,
): string {
  const cleaned = normalizeWhitespace(stripThinkingArtifacts(rawTitle ?? ""));
  if (!cleaned) {
    return DEFAULT_THREAD_TITLE;
  }
  return truncateText(cleaned, MAX_THREAD_TITLE_CHARS);
}

export function threadPreviewFromContent(
  content: string | null | undefined,
): string {
  const normalized = normalizeWhitespace(stripThinkingArtifacts(content ?? ""));
  if (!normalized) {
    return "No messages yet";
  }
  return truncateText(normalized, 120);
}

export function normalizeGeneratedThreadTitle(raw: string): string | null {
  const compact = normalizeWhitespace(
    stripThinkingArtifacts(raw)
      .replace(/```[\s\S]*?```/g, " ")
      .replace(/<\/?[^>]+>/g, " "),
  );
  if (!compact) {
    return null;
  }

  let title = compact.replace(/^title\s*[:-]\s*/i, "").trim();
  title = normalizeWhitespace(title.replace(/^['"`]+|['"`]+$/g, "").trim());

  if (!title || /^user\s*:/i.test(title) || /^assistant\s*:/i.test(title)) {
    return null;
  }

  return displayThreadTitle(title);
}

export function fallbackThreadTitleFromUserMessage(content: string): string {
  const normalized = normalizeWhitespace(stripThinkingArtifacts(content));
  if (!normalized) {
    return DEFAULT_THREAD_TITLE;
  }
  return truncateText(normalized, MAX_THREAD_TITLE_CHARS);
}

export function isQwen35ChatModel(variant: string | null): boolean {
  if (!variant) {
    return false;
  }
  return variant.trim().toLowerCase().startsWith("qwen3.5-");
}

export function isLfm25ThinkingModel(variant: string | null): boolean {
  if (!variant) {
    return false;
  }
  return variant.trim().toLowerCase() === "lfm2.5-1.2b-thinking-gguf";
}

export function supportsImplicitOpenThinkTagParsing(
  variant: string | null,
): boolean {
  return isQwen35ChatModel(variant) || isLfm25ThinkingModel(variant);
}

export function formatBytes(size: number): string {
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

export function createMediaItemId(): string {
  if (
    typeof crypto !== "undefined" &&
    typeof crypto.randomUUID === "function"
  ) {
    return crypto.randomUUID();
  }
  return `media-${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

export function fileToDataUrl(file: File): Promise<string> {
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

export function buildUserDisplayContent(
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

export function buildThreadContentParts(
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
  const resolvedRelativeSource = resolveRelativeSourceToApiOrigin(source);
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
      return lowered.startsWith("/") ? resolvedRelativeSource : source;
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
    return lowered.startsWith("/") ? resolvedRelativeSource : source;
  }
  return null;
}

function resolveRelativeSourceToApiOrigin(source: string): string {
  if (!source.startsWith("/") || typeof window === "undefined") {
    return source;
  }

  try {
    const apiBase = new URL(API_BASE_URL, window.location.origin);
    return new URL(source, apiBase.origin).toString();
  } catch {
    return source;
  }
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

export function parseUserMessageDisplayFromContentParts(
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
