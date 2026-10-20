import {
  type ChangeEvent,
  Fragment,
  useCallback,
  useEffect,
  useLayoutEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  ArrowDown,
  Brain,
  History,
  Send,
  Square,
  Loader2,
  Settings2,
  Plus,
  Trash2,
  AlertTriangle,
} from "lucide-react";
import { useSearchParams } from "react-router-dom";
import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { RouteHistoryDrawer } from "@/components/RouteHistoryDrawer";
import { StatePanel } from "@/components/ui/state-panel";
import { ChatThreadList } from "@/features/chat/components/ChatThreadList";
import { useMediaQuery } from "@/hooks/useMediaQuery";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogTitle,
} from "@/components/ui/dialog";
import {
  buildChatThreadMessagePayload,
  CHAT_IMAGE_ACCEPT,
  DEFAULT_THREAD_TITLE,
  type ChatPlaygroundProps,
  type GenerateTitleArgs,
  type ImagePreviewState,
  type PendingImageAttachment,
  defaultThinkingEnabledForModel,
  displayThreadTitle,
  getErrorMessage,
  isChatImageFile,
  parseAssistantContent,
  stripMarkdown,
  isSameCalendarDay,
  messageDayLabel,
  readChatImage,
  stoppedReplyContent,
  systemPromptForModel,
  supportsImageAttachmentsForModel,
  threadHasImageAttachments,
  supportsImplicitOpenThinkTagParsing,
} from "@/features/chat/playground/support";
import {
  api,
  type ChatGenerationStats,
  type ChatReasoningEffort,
  type ChatThread,
  type ChatThreadMessageRecord,
  type ChatThreadStreamCallbacks,
} from "@/api";
import { AssistantMessage } from "@/features/chat/components/messages/AssistantMessage";
import { UserMessage } from "@/features/chat/components/messages/UserMessage";
import { DaySeparator } from "@/features/chat/components/messages/MessageParts";
import { useReadAloud } from "@/features/chat/hooks/useReadAloud";
import { ConfirmDeleteDialog } from "@/components/history/ConfirmDeleteDialog";
import { RouteModelSelect } from "@/components/RouteModelSelect";

const NEW_CHAT_DRAFT_KEY = "__new_chat__";

/** A turn the user stopped, kept on screen until the server's copy loads. */
interface StoppedTurn {
  /** The prompt and its partial reply, marked as stopped. */
  messages: ChatThreadMessageRecord[];
  /** Messages the thread had before the turn started. */
  priorIds: Set<string>;
  /** Messages the turn replaces (a retried reply, or replies after an edit). */
  replacedIds: string[];
}

/** The turn streaming now, kept apart from the view so a stop keeps its
 * prompt and reply whatever is on screen. */
interface StreamingTurn {
  threadId: string;
  /** The prompt as shown; the server's copy once the stream starts. */
  prompt: ChatThreadMessageRecord;
  assistantTempId: string;
  modelId: string;
  /** The reply so far. */
  content: string;
  /** The prompt opened `<think>`, so the reply streams reasoning first. */
  reasoningOpenedInPrompt: boolean;
  /** Messages the thread had before the turn started. */
  priorIds: Set<string>;
  /** Messages the turn replaces (a retried reply, or replies after an edit). */
  replacedIds: string[];
  controller: AbortController | null;
}

/** Thinking settings a turn is sent with. */
interface TurnThinking {
  supported: boolean;
  enabled: boolean;
  effort: ChatReasoningEffort | undefined;
}

/** Waits between history reloads after a stop, until the server has stored
 * the stopped turn. */
const STOPPED_TURN_SYNC_DELAYS_MS = [0, 250, 750, 2000];

interface ChatComposerDraft {
  input: string;
  pendingImages: PendingImageAttachment[];
}

function draftKeyForThread(threadId: string | null): string {
  return threadId ?? NEW_CHAT_DRAFT_KEY;
}

/** Text messages can be edited in place; ones with attachments cannot. */
function isEditableUserMessage(message: ChatThreadMessageRecord): boolean {
  return (
    message.role === "user" &&
    !message.id.startsWith("tmp-") &&
    !(message.content_parts ?? []).some(
      (part) => part.type !== "text" && part.type !== "input_text",
    )
  );
}

function isEmptyDraft(draft: ChatComposerDraft): boolean {
  return draft.input.length === 0 && draft.pendingImages.length === 0;
}

/** Whether the server has stored a stopped turn: a reply it did not have
 * before follows the turn's prompt, found by id once the server named it. */
function stoppedTurnStored(
  turn: StoppedTurn,
  history: ChatThreadMessageRecord[],
): boolean {
  const [prompt] = turn.messages;
  return history.some((message, index) => {
    const previous = history[index - 1];
    return (
      message.role === "assistant" &&
      !turn.priorIds.has(message.id) &&
      previous?.role === "user" &&
      (prompt.id.startsWith("tmp-")
        ? previous.content === prompt.content
        : previous.id === prompt.id)
    );
  });
}

export function ChatPlayground({
  selectedModel,
  selectedModelReady,
  supportsThinking,
  chatCapabilities,
  modelLabel,
  modelOptions,
  modelSupportsThinking,
  modelChatCapabilities,
  modelSupportsImages,
  onSelectModel,
  onOpenModelManager,
  onModelRequired,
  readAloudVoice = null,
  onOpenSpeechModels,
}: ChatPlaygroundProps) {
  const [searchParams, setSearchParams] = useSearchParams();
  const activeThreadId = searchParams.get("threadId");

  const [threads, setThreads] = useState<ChatThread[]>([]);
  const [messages, setMessages] = useState<ChatThreadMessageRecord[]>([]);
  const [expandedThoughts, setExpandedThoughts] = useState<
    Record<string, boolean>
  >({});
  // Reasoning time measured client-side while a reply streams, keyed by
  // message id. Replies loaded from history have none.
  const [reasoningDurations, setReasoningDurations] = useState<
    Record<string, number>
  >({});
  const [streamStartedAt, setStreamStartedAt] = useState<number | null>(null);
  // Thinking as the streaming turn was sent, whatever the toggle says now.
  const [streamThinkingEnabled, setStreamThinkingEnabled] = useState(false);
  const [showJumpToLatest, setShowJumpToLatest] = useState(false);
  const [editingMessageId, setEditingMessageId] = useState<string | null>(
    null,
  );
  const [deleteReplyId, setDeleteReplyId] = useState<string | null>(null);
  const [deleteReplyPending, setDeleteReplyPending] = useState(false);
  const [deleteReplyError, setDeleteReplyError] = useState<string | null>(
    null,
  );
  // Screen readers hear a finished reply once, not each streamed token.
  const [replyAnnouncement, setReplyAnnouncement] = useState("");
  // Drafts intentionally live only for this mounted session. Durable persistence
  // is out of scope; the thread key prevents in-session cross-conversation leaks.
  const [composerDrafts, setComposerDrafts] = useState<
    Record<string, ChatComposerDraft>
  >({});
  const [isThinkingEnabled, setIsThinkingEnabled] = useState(() =>
    chatCapabilities?.default_thinking_enabled ??
      defaultThinkingEnabledForModel(selectedModel),
  );
  const [reasoningEffort, setReasoningEffort] =
    useState<ChatReasoningEffort>(
      chatCapabilities?.default_reasoning_effort ??
        chatCapabilities?.reasoning_efforts[0] ??
        "xhigh",
    );
  const [isStreaming, setIsStreaming] = useState(false);
  const [isPreparingThread, setIsPreparingThread] = useState(false);
  const [streamingThreadId, setStreamingThreadId] = useState<string | null>(
    null,
  );
  const [threadsLoading, setThreadsLoading] = useState(true);
  const [messagesLoading, setMessagesLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  // A thread whose messages failed to load: shown as that thread's error
  // state (with Retry), not as an empty new chat.
  const [threadLoadError, setThreadLoadError] = useState<{
    threadId: string;
    message: string;
  } | null>(null);
  const [threadLoadAttempt, setThreadLoadAttempt] = useState(0);
  // Engine stats of replies streamed in this session, keyed by message id.
  // Stored replies keep only their token count and total time.
  const [streamedStats, setStreamedStats] = useState<
    Record<string, ChatGenerationStats>
  >({});
  const [imagePreview, setImagePreview] = useState<ImagePreviewState | null>(
    null,
  );
  const [isReadingImages, setIsReadingImages] = useState(false);
  const [isHistoryDrawerOpen, setIsHistoryDrawerOpen] = useState(false);
  // Desktop widths keep the thread list beside the conversation; narrower
  // windows open it as a drawer.
  const showThreadColumn = useMediaQuery("(min-width: 1024px)");
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
  const streamingTurnRef = useRef<StreamingTurn | null>(null);
  const messageViewportRef = useRef<HTMLDivElement | null>(null);
  const followingOutputRef = useRef(true);
  const followingOutputThreadRef = useRef<string | null>(null);
  const textareaRef = useRef<HTMLTextAreaElement | null>(null);
  const imageInputRef = useRef<HTMLInputElement | null>(null);
  const pendingImageSeqRef = useRef(0);
  const composerDraftsRef = useRef<Record<string, ChatComposerDraft>>({});
  const revokedPreviewUrlsRef = useRef<Set<string>>(new Set());
  const isMountedRef = useRef(true);
  const messagesRef = useRef<ChatThreadMessageRecord[]>([]);
  // The server stores a stopped turn once its stream has wound down, which
  // can be after the history reload that follows Stop. Until then the turn
  // lives here, per thread.
  const stoppedTurnsRef = useRef<Map<string, StoppedTurn[]>>(new Map());
  const deletedDraftKeysRef = useRef<Set<string>>(new Set());

  const activeDraftKey = draftKeyForThread(activeThreadId);
  const activeDraft = composerDrafts[activeDraftKey];
  const input = activeDraft?.input ?? "";
  const pendingImages = activeDraft?.pendingImages ?? [];

  composerDraftsRef.current = composerDrafts;

  useEffect(() => {
    messagesRef.current = messages;
  }, [messages]);

  /** History from the server plus this session's stopped turns that the
   * server has not stored yet, each kept whole and placed by when it was
   * sent. Once the server has a turn, its copy is the only one: a later
   * retry, edit or delete of it must not bring the local copy back. */
  const withStoppedTurns = useCallback(
    (threadId: string, history: ChatThreadMessageRecord[]) => {
      const pending = (stoppedTurnsRef.current.get(threadId) ?? []).filter(
        (turn) => !stoppedTurnStored(turn, history),
      );
      if (pending.length > 0) {
        stoppedTurnsRef.current.set(threadId, pending);
      } else {
        stoppedTurnsRef.current.delete(threadId);
      }
      let merged = history;
      for (const turn of pending) {
        const [prompt] = turn.messages;
        const hidden = new Set([prompt.id, ...turn.replacedIds]);
        merged = merged.filter((message) => !hidden.has(message.id));
        const insertAt = merged.findIndex(
          (message) => message.created_at > prompt.created_at,
        );
        merged =
          insertAt === -1
            ? [...merged, ...turn.messages]
            : [
                ...merged.slice(0, insertAt),
                ...turn.messages,
                ...merged.slice(insertAt),
              ];
      }
      return merged;
    },
    [],
  );

  const revokeImagePreviews = useCallback(
    (images: PendingImageAttachment[]) => {
      for (const image of images) {
        if (
          !image.previewUrl ||
          revokedPreviewUrlsRef.current.has(image.previewUrl) ||
          typeof URL.revokeObjectURL !== "function"
        ) {
          continue;
        }
        URL.revokeObjectURL(image.previewUrl);
        revokedPreviewUrlsRef.current.add(image.previewUrl);
      }
    },
    [],
  );

  const updateComposerDraft = useCallback(
    (
      draftKey: string,
      update: (draft: ChatComposerDraft) => ChatComposerDraft,
    ) => {
      setComposerDrafts((previous) => {
        const nextDraft = update(
          previous[draftKey] ?? { input: "", pendingImages: [] },
        );
        if (isEmptyDraft(nextDraft)) {
          if (!(draftKey in previous)) {
            return previous;
          }
          const next = { ...previous };
          delete next[draftKey];
          return next;
        }
        return { ...previous, [draftKey]: nextDraft };
      });
    },
    [],
  );

  const clearComposerDraft = useCallback(
    (draftKey: string) => {
      setComposerDrafts((previous) => {
        const removedDraft = previous[draftKey];
        if (!removedDraft) {
          return previous;
        }
        revokeImagePreviews(removedDraft.pendingImages);
        const next = { ...previous };
        delete next[draftKey];
        return next;
      });
    },
    [revokeImagePreviews],
  );

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
  const busy = isStreaming || isPreparingThread;
  // Retry with… offers the loaded chat models.
  const retryModelOptions = modelOptions.filter((option) => option.isReady);
  const lastAssistantIndex = useMemo(() => {
    for (let index = visibleMessages.length - 1; index >= 0; index -= 1) {
      if (visibleMessages[index].role === "assistant") {
        return index;
      }
    }
    return -1;
  }, [visibleMessages]);
  const historyBadgeLabel = useMemo(() => {
    if (threads.length > 99) {
      return "99+";
    }
    return String(Math.max(0, threads.length));
  }, [threads.length]);

  const activeThreadLoadError =
    threadLoadError && threadLoadError.threadId === activeThreadId
      ? threadLoadError.message
      : null;
  const hasConversation =
    !!activeThreadId &&
    (visibleMessages.length > 0 ||
      isStreaming ||
      messagesLoading ||
      activeThreadLoadError !== null);
  const thinkingEnabledForModel = supportsThinking && isThinkingEnabled;
  // A model that always reasons shows a label, not a switch that does nothing.
  const thinkingToggleable = chatCapabilities?.thinking_toggleable ?? true;
  const supportedReasoningEfforts =
    chatCapabilities?.reasoning_efforts ?? [];
  const reasoningEffortOptionsKey = supportedReasoningEfforts.join(",");
  const firstSupportedReasoningEffort = supportedReasoningEfforts[0];
  const acceptsImages = useCallback(
    (variant: string | null) =>
      variant
        ? (modelSupportsImages?.(variant) ??
          supportsImageAttachmentsForModel(variant))
        : false,
    [modelSupportsImages],
  );
  const supportsImageAttachments = acceptsImages(selectedModel);
  const supportsImageAttachmentsRef = useRef(supportsImageAttachments);
  supportsImageAttachmentsRef.current = supportsImageAttachments;
  // Attachments survive a switch to a text-only model so the user can switch
  // back; sending is blocked until they are removed or the model changes.
  const hasUnsupportedImages =
    pendingImages.length > 0 && !supportsImageAttachments;
  const unsupportedImagesMessage = `${modelLabel ?? "This model"} can't read images. Remove them or switch to a model that accepts images.`;
  // Earlier image turns reach a text-only model as text (server side), so say
  // so and offer a model that can see them.
  const threadHasImages = useMemo(
    () => threadHasImageAttachments(messages),
    [messages],
  );
  const showHistoryImageNotice =
    Boolean(activeThreadId) && threadHasImages && !supportsImageAttachments;
  const imageModelOption = modelOptions.find((option) =>
    acceptsImages(option.value),
  );
  /** The model emits its reasoning before `</think>`. */
  const writesImplicitThink = useCallback(
    (variant: string) =>
      modelSupportsThinking?.(variant) ??
      supportsImplicitOpenThinkTagParsing(variant),
    [modelSupportsThinking],
  );
  const renderModelId = activeThread?.model_id ?? selectedModel;
  // Parse by the capabilities of the model that wrote the thread, not the
  // model selected now.
  const implicitOpenThinkTagModel = renderModelId
    ? writesImplicitThink(renderModelId)
    : false;
  const chatSecondaryButtonClass =
    "rounded-lg border-[var(--border-muted)] bg-[var(--bg-surface-2)] text-[var(--text-secondary)] shadow-none hover:border-[var(--border-strong)] hover:bg-[var(--bg-surface-3)] hover:text-[var(--text-primary)]";
  const chatSecondaryButtonActiveClass =
    "rounded-lg border-[var(--border-strong)] bg-[var(--bg-surface-3)] text-[var(--text-primary)] shadow-none";
  const chatAccentButtonClass =
    "rounded-lg bg-[var(--accent-solid)] text-[var(--text-on-accent)] shadow-none hover:opacity-90";
  const chatComposerSendButtonClass =
    "chat-composer-send-button rounded-full border-transparent text-[var(--text-primary)] shadow-none";

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

  useEffect(() => {
    if (!supportsThinking) {
      setIsThinkingEnabled(false);
      return;
    }
    setIsThinkingEnabled(
      !thinkingToggleable ||
        (chatCapabilities?.default_thinking_enabled ??
          defaultThinkingEnabledForModel(selectedModel)),
    );
    setReasoningEffort(
      chatCapabilities?.default_reasoning_effort ??
        firstSupportedReasoningEffort ??
        "xhigh",
    );
  }, [
    chatCapabilities?.default_reasoning_effort,
    chatCapabilities?.default_thinking_enabled,
    firstSupportedReasoningEffort,
    reasoningEffortOptionsKey,
    selectedModel,
    supportsThinking,
    thinkingToggleable,
  ]);

  useEffect(() => {
    setImagePreview(null);
  }, [activeDraftKey]);

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

  /**
   * Asks the server to title a thread still called "New chat" once a turn
   * has finished. The server writes the title from the first exchange with
   * thinking off, so a reasoning model's thoughts never become the title. A
   * failure leaves "New chat" for the next turn to try again.
   */
  const maybeGenerateThreadTitle = useCallback(
    async ({ threadId, modelId }: GenerateTitleArgs) => {
      const existingThread = threadsRef.current.find(
        (thread) => thread.id === threadId,
      );
      if (
        existingThread &&
        displayThreadTitle(existingThread.title) !== DEFAULT_THREAD_TITLE
      ) {
        return;
      }
      if (titleGenerationInFlightRef.current.has(threadId)) {
        return;
      }
      titleGenerationInFlightRef.current.add(threadId);
      try {
        const updatedThread = await api.generateChatThreadTitle(threadId, {
          model_id: modelId,
        });
        setThreads((previous) =>
          previous.map((thread) =>
            thread.id === updatedThread.id ? updatedThread : thread,
          ),
        );
      } catch {
        // Keep "New chat"; the next finished turn asks again.
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
    isMountedRef.current = true;
    return () => {
      isMountedRef.current = false;
      streamingTurnRef.current?.controller?.abort();
      for (const draft of Object.values(composerDraftsRef.current)) {
        revokeImagePreviews(draft.pendingImages);
      }
    };
  }, [revokeImagePreviews]);

  useLayoutEffect(() => {
    if (followingOutputThreadRef.current !== activeThreadId) {
      followingOutputThreadRef.current = activeThreadId;
      followingOutputRef.current = true;
    }
    if (!followingOutputRef.current || !messageViewportRef.current) {
      return;
    }
    messageViewportRef.current.scrollTop =
      messageViewportRef.current.scrollHeight;
  }, [visibleMessages, isStreaming, activeThreadId]);

  const handleMessageViewportScroll = useCallback(() => {
    const viewport = messageViewportRef.current;
    if (!viewport) {
      return;
    }
    followingOutputRef.current =
      viewport.scrollHeight - viewport.scrollTop - viewport.clientHeight <= 48;
    setShowJumpToLatest(!followingOutputRef.current);
  }, []);

  const jumpToLatest = useCallback(() => {
    const viewport = messageViewportRef.current;
    if (!viewport) {
      return;
    }
    followingOutputRef.current = true;
    setShowJumpToLatest(false);
    const reduceMotion =
      typeof window.matchMedia === "function" &&
      window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    if (typeof viewport.scrollTo === "function") {
      viewport.scrollTo({
        top: viewport.scrollHeight,
        behavior: reduceMotion ? "auto" : "smooth",
      });
    } else {
      viewport.scrollTop = viewport.scrollHeight;
    }
  }, []);

  useEffect(() => {
    if (!textareaRef.current) {
      return;
    }

    const minHeight = hasConversation ? 56 : 72;
    const maxHeight = hasConversation ? 104 : 140;

    textareaRef.current.style.height = "0px";
    const nextHeight = Math.min(
      maxHeight,
      Math.max(textareaRef.current.scrollHeight, minHeight),
    );
    textareaRef.current.style.height = `${nextHeight}px`;
    textareaRef.current.style.overflowY =
      textareaRef.current.scrollHeight > maxHeight ? "auto" : "hidden";
  }, [hasConversation, input]);

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
      setThreadLoadError(null);
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

        setMessages(withStoppedTurns(requestedThreadId, detail.messages));
        setExpandedThoughts({});
        setThreads((previous) =>
          previous.map((thread) =>
            thread.id === detail.thread.id ? detail.thread : thread,
          ),
        );
      } catch (loadError) {
        if (!cancelled) {
          setMessages([]);
          setThreadLoadError({
            threadId: requestedThreadId,
            message: getErrorMessage(
              loadError,
              "Failed to load this conversation.",
            ),
          });
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
  }, [activeThreadId, threadLoadAttempt, withStoppedTurns]);

  /** Reloads a thread until the server has stored its stopped turns, which
   * it does once the stream has wound down, after the stop. */
  const syncStoppedTurns = useCallback(
    async (threadId: string) => {
      for (const delayMs of STOPPED_TURN_SYNC_DELAYS_MS) {
        if (delayMs > 0) {
          await new Promise((resolve) => window.setTimeout(resolve, delayMs));
        }
        // A new turn reloads history itself when it ends.
        if (
          !isMountedRef.current ||
          streamingTurnRef.current ||
          activeThreadIdRef.current !== threadId
        ) {
          return;
        }
        try {
          const detail = await api.getChatThread(threadId);
          if (
            streamingTurnRef.current ||
            activeThreadIdRef.current !== threadId
          ) {
            return;
          }
          setMessages(withStoppedTurns(threadId, detail.messages));
        } catch {
          return;
        }
        if (!stoppedTurnsRef.current.has(threadId)) {
          return;
        }
      }
    },
    [withStoppedTurns],
  );

  /** Stops the streaming turn. Its prompt and partial reply stay, marked as
   * stopped, until the server's copy loads. */
  const stopStreaming = useCallback(() => {
    const turn = streamingTurnRef.current;
    if (!turn) {
      return;
    }
    streamingTurnRef.current = null;
    turn.controller?.abort();
    setIsStreaming(false);
    setStreamingThreadId(null);

    const stoppedMessages: ChatThreadMessageRecord[] = [
      turn.prompt,
      {
        id: turn.assistantTempId,
        thread_id: turn.threadId,
        role: "assistant",
        content: stoppedReplyContent(
          turn.content,
          turn.reasoningOpenedInPrompt,
        ),
        created_at: turn.prompt.created_at + 1,
        tokens_generated: null,
        generation_time_ms: null,
        model_id: turn.modelId,
        finish_reason: "cancelled",
      },
    ];
    stoppedTurnsRef.current.set(turn.threadId, [
      ...(stoppedTurnsRef.current.get(turn.threadId) ?? []),
      {
        messages: stoppedMessages,
        priorIds: turn.priorIds,
        replacedIds: turn.replacedIds,
      },
    ]);
    if (activeThreadIdRef.current === turn.threadId) {
      // The turn is the last thing in its thread.
      const hidden = new Set([
        turn.prompt.id,
        turn.assistantTempId,
        ...turn.replacedIds,
      ]);
      setMessages((previous) => [
        ...previous.filter((message) => !hidden.has(message.id)),
        ...stoppedMessages,
      ]);
    }
    void syncStoppedTurns(turn.threadId);
    void refreshThreadList(turn.threadId);
  }, [refreshThreadList, syncStoppedTurns]);

  // Leaving a thread mid-reply (Back) stops the reply, as leaving Chat does,
  // so its stream never writes into another thread.
  useEffect(() => {
    const turn = streamingTurnRef.current;
    if (turn && turn.threadId !== activeThreadId) {
      stopStreaming();
    }
  }, [activeThreadId, stopStreaming]);

  const {
    state: readAloud,
    start: startReadAloud,
    stop: stopReadAloud,
  } = useReadAloud(setError);

  const toggleReadAloud = useCallback(
    (messageId: string, text: string) => {
      if (readAloud?.messageId === messageId) {
        stopReadAloud();
        return;
      }
      if (!readAloudVoice) {
        onOpenSpeechModels?.();
        return;
      }
      startReadAloud(messageId, stripMarkdown(text), readAloudVoice);
    },
    [
      onOpenSpeechModels,
      readAloud,
      readAloudVoice,
      startReadAloud,
      stopReadAloud,
    ],
  );

  // Audio belongs to the conversation on screen.
  useEffect(() => stopReadAloud, [activeThreadId, stopReadAloud]);

  // Starting a chat is local: the thread is created on the server by the
  // first send, so abandoned "New chat" clicks no longer leave empty threads.
  const handleStartNewChat = useCallback(() => {
    if (isStreaming || isPreparingThread) {
      return;
    }
    if (activeThreadId) {
      setActiveThreadInUrl(null);
      setMessages([]);
      setExpandedThoughts({});
    }
    setError(null);
    textareaRef.current?.focus();
  }, [activeThreadId, isPreparingThread, isStreaming, setActiveThreadInUrl]);

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
        const deletedDraftKey = draftKeyForThread(threadId);
        deletedDraftKeysRef.current.add(deletedDraftKey);
        clearComposerDraft(deletedDraftKey);
        setDeleteTargetThreadId(null);

        if (activeThreadIdRef.current === threadId) {
          setActiveThreadInUrl(null, true);
          setMessages([]);
          setExpandedThoughts({});
        }
      } catch (deleteError) {
        setError(getErrorMessage(deleteError, "Failed to delete this chat."));
      } finally {
        setDeleteThreadPending(false);
      }
    },
    [
      clearComposerDraft,
      deleteThreadPending,
      isPreparingThread,
      isStreaming,
      setActiveThreadInUrl,
    ],
  );

  const confirmDeleteThread = useCallback(() => {
    if (!deleteTargetThreadId) {
      return;
    }
    void handleDeleteThread(deleteTargetThreadId);
  }, [deleteTargetThreadId, handleDeleteThread]);

  const handleHistoryDrawerOpenChange = useCallback(
    (nextOpen: boolean) => {
      if (!nextOpen && deleteTargetThreadId) {
        return;
      }
      setIsHistoryDrawerOpen(nextOpen);
    },
    [deleteTargetThreadId],
  );

  const removePendingImage = useCallback((imageId: string) => {
    updateComposerDraft(activeDraftKey, (draft) => {
      const removedImages = draft.pendingImages.filter(
        (image) => image.id === imageId,
      );
      revokeImagePreviews(removedImages);
      return {
        ...draft,
        pendingImages: draft.pendingImages.filter((image) => image.id !== imageId),
      };
    });
  }, [activeDraftKey, revokeImagePreviews, updateComposerDraft]);

  const handleComposerImageChange = useCallback(
    async (event: ChangeEvent<HTMLInputElement>) => {
      const files = Array.from(event.target.files ?? []);
      if (files.length === 0) {
        return;
      }

      const invalidFile = files.find((file) => !isChatImageFile(file));
      if (invalidFile) {
        setError(
          `Attach PNG, JPEG, GIF, WebP or BMP images, not ${invalidFile.name}.`,
        );
        event.target.value = "";
        return;
      }

      setError(null);
      setIsReadingImages(true);
      const targetDraftKey = activeDraftKey;
      try {
        const nextImages = await Promise.all(
          files.map(async (file) => {
            pendingImageSeqRef.current += 1;
            const sequence = pendingImageSeqRef.current;
            const source = await readChatImage(file);
            return {
              id: `image-${sequence}`,
              source,
              label: file.name || `image-${sequence}.png`,
              previewUrl:
                typeof URL.createObjectURL === "function"
                  ? URL.createObjectURL(file)
                  : undefined,
            } satisfies PendingImageAttachment;
          }),
        );

        if (
          !isMountedRef.current ||
          !supportsImageAttachmentsRef.current ||
          deletedDraftKeysRef.current.has(targetDraftKey)
        ) {
          revokeImagePreviews(nextImages);
          return;
        }

        updateComposerDraft(targetDraftKey, (draft) => ({
          ...draft,
          pendingImages: [...draft.pendingImages, ...nextImages],
        }));
      } catch (imageError) {
        setError(
          getErrorMessage(imageError, "Failed to read the selected image."),
        );
      } finally {
        setIsReadingImages(false);
        event.target.value = "";
      }
    },
    [activeDraftKey, revokeImagePreviews, updateComposerDraft],
  );

  const openImagePicker = useCallback(() => {
    imageInputRef.current?.click();
  }, []);

  const sendMessage = async () => {
    const submittedDraftKey = activeDraftKey;
    const submittedDraft = composerDrafts[submittedDraftKey] ?? {
      input: "",
      pendingImages: [],
    };
    const text = submittedDraft.input.trim();
    if (
      (text.length === 0 && submittedDraft.pendingImages.length === 0) ||
      (submittedDraft.pendingImages.length > 0 && !supportsImageAttachments) ||
      isStreaming ||
      isPreparingThread
    ) {
      return;
    }

    if (!selectedModel || !selectedModelReady) {
      onModelRequired();
      return;
    }

    let targetThreadId = activeThreadId;
    let targetDraftKey = submittedDraftKey;
    const createsThread = !targetThreadId;
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
        targetThreadId = createdThread.id;
        targetDraftKey = draftKeyForThread(createdThread.id);
        setComposerDrafts((previous) => {
          const next = { ...previous, [targetDraftKey]: submittedDraft };
          delete next[submittedDraftKey];
          return next;
        });
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

    const requestPayload = buildChatThreadMessagePayload({
      text,
      images: submittedDraft.pendingImages,
    });

    const timestamp = Date.now();
    const userTempId = `tmp-user-${timestamp}`;
    const optimisticUserMessage: ChatThreadMessageRecord = {
      id: userTempId,
      thread_id: targetThreadId,
      role: "user",
      content: requestPayload.content,
      content_parts: requestPayload.contentParts,
      created_at: timestamp,
      tokens_generated: null,
      generation_time_ms: null,
    };
    const threadIdForTurn = targetThreadId;
    const draftKeyForTurn = targetDraftKey;

    const thinking = thinkingFor(selectedModel);
    startTurnStream({
      threadId: threadIdForTurn,
      startedAt: timestamp,
      modelId: selectedModel,
      thinking,
      shown: (previous) => [...previous, optimisticUserMessage],
      prompt: optimisticUserMessage,
      replacedIds: [],
      restoreOnFailure: false,
      failureMessage: "Failed to send this message.",
      start: (callbacks) =>
        api.sendChatThreadMessageStream(
          threadIdForTurn,
          {
            ...generationOptions(selectedModel, thinking),
            content: requestPayload.content,
            content_parts: requestPayload.contentParts,
          },
          callbacks,
        ),
      onStarted: () => {
        clearComposerDraft(draftKeyForTurn);
        setImagePreview(null);
      },
      onFailed: () => {
        // The server keeps no failed turn: give the message back to send
        // again. Its previews were released when it was sent.
        updateComposerDraft(draftKeyForTurn, (draft) =>
          isEmptyDraft(draft)
            ? {
                input: submittedDraft.input,
                pendingImages: submittedDraft.pendingImages.map((image) => ({
                  ...image,
                  previewUrl: undefined,
                })),
              }
            : draft,
        );
        if (createsThread) {
          void discardEmptyThread(threadIdForTurn);
        }
      },
    });
  };

  /** A first message that failed leaves its new thread empty. The thread
   * goes, and the message moves back to a new chat's composer. */
  const discardEmptyThread = async (threadId: string) => {
    try {
      const detail = await api.getChatThread(threadId);
      if (detail.messages.length > 0) {
        return;
      }
      await api.deleteChatThread(threadId);
    } catch {
      return;
    }
    setThreads((previous) => previous.filter((thread) => thread.id !== threadId));
    const threadDraftKey = draftKeyForThread(threadId);
    setComposerDrafts((previous) => {
      const draft = previous[threadDraftKey];
      const next = { ...previous };
      delete next[threadDraftKey];
      if (draft && !next[NEW_CHAT_DRAFT_KEY]) {
        next[NEW_CHAT_DRAFT_KEY] = draft;
      }
      return next;
    });
    if (activeThreadIdRef.current === threadId) {
      setActiveThreadInUrl(null, true);
      setMessages([]);
      setExpandedThoughts({});
    }
  };

  /** Thinking for a turn on `modelId`: the composer's settings for the
   * selected model, and another model's own defaults (Retry with…), which
   * the composer shows once it switches to that model. */
  const thinkingFor = (modelId: string): TurnThinking => {
    if (modelId === selectedModel) {
      return {
        supported: supportsThinking,
        enabled: thinkingEnabledForModel,
        effort:
          thinkingEnabledForModel &&
          supportedReasoningEfforts.includes(reasoningEffort)
            ? reasoningEffort
            : undefined,
      };
    }
    const capabilities = modelChatCapabilities?.(modelId) ?? null;
    const supported = capabilities?.supports_thinking ?? false;
    const enabled =
      supported &&
      (capabilities?.thinking_toggleable === false ||
        (capabilities?.default_thinking_enabled ??
          defaultThinkingEnabledForModel(modelId)));
    return {
      supported,
      enabled,
      effort: enabled
        ? (capabilities?.default_reasoning_effort ??
          capabilities?.reasoning_efforts[0])
        : undefined,
    };
  };

  /** Settings every turn sends: the model, thinking and truncation. */
  const generationOptions = (modelId: string, thinking: TurnThinking) => ({
    model_id: modelId,
    system_prompt: systemPromptForModel(thinking.supported),
    // Every thinking-capable template reads enable_thinking; sending it
    // only for some families left Qwen3.6 thinking regardless of the toggle.
    enable_thinking: thinking.supported ? thinking.enabled : undefined,
    reasoning_effort: thinking.effort,
    // Long threads keep working: the server drops the oldest turns once
    // the history outgrows the model's context.
    truncation: "auto" as const,
  });

  /**
   * Streams one turn into the view: a new message, a retried reply or an
   * edited prompt. `shown` lays out the messages the turn streams after.
   * A retry or edit that fails reloads history, since the server left the
   * thread as it was. The stream writes only to its own thread, and only
   * while it is the current turn.
   */
  const startTurnStream = ({
    threadId,
    startedAt,
    modelId,
    thinking,
    shown,
    prompt,
    replacedIds,
    restoreOnFailure,
    failureMessage,
    start,
    onStarted,
    onFailed,
  }: {
    threadId: string;
    startedAt: number;
    modelId: string;
    thinking: TurnThinking;
    shown: (previous: ChatThreadMessageRecord[]) => ChatThreadMessageRecord[];
    prompt: ChatThreadMessageRecord;
    replacedIds: string[];
    restoreOnFailure: boolean;
    failureMessage: string;
    start: (callbacks: ChatThreadStreamCallbacks) => AbortController;
    onStarted?: () => void;
    /** After a stream that failed has closed. */
    onFailed?: () => void;
  }) => {
    const assistantTempId = `tmp-assistant-${startedAt}`;
    const optimisticAssistantMessage: ChatThreadMessageRecord = {
      id: assistantTempId,
      thread_id: threadId,
      role: "assistant",
      content: "",
      created_at: startedAt + 1,
      tokens_generated: null,
      generation_time_ms: null,
      model_id: modelId,
    };
    const turn: StreamingTurn = {
      threadId,
      prompt,
      assistantTempId,
      modelId,
      content: "",
      reasoningOpenedInPrompt: thinking.enabled && writesImplicitThink(modelId),
      priorIds: new Set(
        messagesRef.current
          .filter((message) => !message.id.startsWith("tmp-"))
          .map((message) => message.id),
      ),
      replacedIds,
      controller: null,
    };
    // The server stores a stopped turn before it starts the next one, so
    // the history this turn reloads has it.
    stoppedTurnsRef.current.delete(threadId);
    streamingTurnRef.current = turn;
    const isCurrent = () => streamingTurnRef.current === turn;
    const onScreen = () =>
      isCurrent() && activeThreadIdRef.current === threadId;
    const reloadHistory = () =>
      void api
        .getChatThread(threadId)
        .then((detail) => {
          if (activeThreadIdRef.current !== threadId) {
            return;
          }
          setMessages(withStoppedTurns(threadId, detail.messages));
        })
        .catch(() => {
          // Ignore follow-up sync failures after stream close.
        });

    setError(null);
    setMessages((previous) => [...shown(previous), optimisticAssistantMessage]);
    followingOutputRef.current = true;
    setIsStreaming(true);
    setStreamStartedAt(startedAt);
    setStreamingThreadId(threadId);
    setStreamThinkingEnabled(thinking.enabled);

    let streamFailed = false;
    let turnModelId: string | null = null;
    try {
      turn.controller = start({
        onStart: ({ userMessage }) => {
          if (!isCurrent()) {
            return;
          }
          const promptId = turn.prompt.id;
          turn.prompt = userMessage;
          if (!onScreen()) {
            return;
          }
          setMessages((previous) => {
            let replaced = false;
            const updated = previous.map((message) => {
              if (message.id === promptId) {
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
          if (!isCurrent()) {
            return;
          }
          turn.content += delta;
          const content = turn.content;
          if (!onScreen()) {
            return;
          }
          setMessages((previous) => {
            let updatedAssistant = false;
            const updated = previous.map((message) => {
              if (message.id === assistantTempId) {
                updatedAssistant = true;
                return { ...message, content };
              }
              return message;
            });

            if (updatedAssistant) {
              return updated;
            }

            return [...updated, { ...optimisticAssistantMessage, content }];
          });
        },
        onDone: ({ assistantMessage, stats: streamStats, modelId: doneModelId }) => {
          turnModelId = doneModelId;
          if (!onScreen()) {
            return;
          }
          const carryOver = <T,>(previous: Record<string, T>) =>
            assistantTempId in previous
              ? { ...previous, [assistantMessage.id]: previous[assistantTempId] }
              : previous;
          setReasoningDurations(carryOver);
          setExpandedThoughts(carryOver);
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
          setStreamedStats((previous) => ({
            ...previous,
            [assistantMessage.id]: streamStats,
          }));
          const spokenAnswer = stripMarkdown(
            parseAssistantContent(assistantMessage.content, {
              implicitOpenThinkTag: true,
              treatNoTagAsThinking: false,
            }).answer,
          ).trim();
          setReplyAnnouncement(
            spokenAnswer ? `Reply: ${spokenAnswer}` : "Reply finished.",
          );
        },
        onError: (message) => {
          if (!isCurrent()) {
            return;
          }
          streamFailed = true;
          if (!onScreen()) {
            return;
          }
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
          // A stopped turn has already left the view to the stop.
          const current = isCurrent();
          if (current) {
            streamingTurnRef.current = null;
            setIsStreaming(false);
            setStreamingThreadId(null);
          }
          const listRefreshed = refreshThreadList(threadId);
          // Any turn that ended without an error (finished, stopped,
          // retried or edited) can title a thread still called "New chat".
          // Ask after the list refresh, so its older copy of the thread
          // cannot replace the new title.
          if (!streamFailed) {
            void listRefreshed.then(() =>
              maybeGenerateThreadTitle({ threadId, modelId: turnModelId }),
            );
          }
          if (!current) return;
          if (streamFailed) onFailed?.();
          // Failed partial text of a new message is useful feedback but is
          // deliberately absent from persisted history. Keep it visible until
          // the user navigates. A failed retry or edit changed nothing.
          if (streamFailed && !restoreOnFailure) return;
          reloadHistory();
        },
      });
      onStarted?.();
    } catch (sendError) {
      streamingTurnRef.current = null;
      setIsStreaming(false);
      setStreamingThreadId(null);
      setError(getErrorMessage(sendError, failureMessage));
      if (restoreOnFailure) {
        reloadHistory();
      } else {
        setMessages((previous) =>
          previous.filter(
            (message) =>
              message.id !== prompt.id && message.id !== assistantTempId,
          ),
        );
      }
    }
  };

  const canStartTurn = (modelId: string | null): modelId is string => {
    if (!activeThreadId || isStreaming || isPreparingThread) {
      return false;
    }
    if (!modelId || (modelId === selectedModel && !selectedModelReady)) {
      onModelRequired();
      return false;
    }
    return true;
  };

  /** Replaces the latest reply with a new one, optionally from another
   * loaded model (which then stays selected for the chat). */
  const retryReply = (replyId: string, modelId: string | null = selectedModel) => {
    if (!canStartTurn(modelId) || !activeThreadId) {
      return;
    }
    const current = messagesRef.current;
    const replyIndex = current.findIndex((message) => message.id === replyId);
    let promptIndex = -1;
    for (let index = replyIndex - 1; index >= 0; index -= 1) {
      if (current[index].role === "user") {
        promptIndex = index;
        break;
      }
    }
    if (replyIndex === -1 || promptIndex === -1) {
      return;
    }
    if (modelId !== selectedModel) {
      onSelectModel(modelId);
    }
    const threadId = activeThreadId;
    const prompt = current[promptIndex];
    const thinking = thinkingFor(modelId);
    startTurnStream({
      threadId,
      startedAt: Date.now(),
      modelId,
      thinking,
      shown: (previous) => {
        const index = previous.findIndex((message) => message.id === prompt.id);
        return index === -1 ? previous : previous.slice(0, index + 1);
      },
      prompt,
      replacedIds: current.slice(promptIndex + 1).map((message) => message.id),
      restoreOnFailure: true,
      failureMessage: "Failed to retry this reply.",
      start: (callbacks) =>
        api.regenerateChatThreadMessageStream(
          threadId,
          replyId,
          generationOptions(modelId, thinking),
          callbacks,
        ),
    });
  };

  /** Replaces one of your messages and the replies after it. */
  const submitEdit = (messageId: string, text: string) => {
    const content = text.trim();
    if (!content || !canStartTurn(selectedModel) || !activeThreadId) {
      return;
    }
    const current = messagesRef.current;
    const index = current.findIndex((message) => message.id === messageId);
    if (index === -1) {
      return;
    }
    const threadId = activeThreadId;
    const edited: ChatThreadMessageRecord = {
      ...current[index],
      content,
      content_parts: null,
    };
    setEditingMessageId(null);
    const thinking = thinkingFor(selectedModel);
    startTurnStream({
      threadId,
      startedAt: Date.now(),
      modelId: selectedModel,
      thinking,
      shown: (previous) => {
        const at = previous.findIndex((message) => message.id === messageId);
        return at === -1 ? previous : [...previous.slice(0, at), edited];
      },
      prompt: edited,
      replacedIds: current.slice(index + 1).map((message) => message.id),
      restoreOnFailure: true,
      failureMessage: "Failed to send your edit.",
      start: (callbacks) =>
        api.editChatThreadMessageStream(
          threadId,
          messageId,
          { ...generationOptions(selectedModel, thinking), content },
          callbacks,
        ),
    });
  };

  const confirmDeleteReply = async () => {
    const replyId = deleteReplyId;
    const threadId = activeThreadId;
    if (!replyId || !threadId) {
      return;
    }
    setDeleteReplyPending(true);
    setDeleteReplyError(null);
    try {
      await api.deleteChatThreadMessage(threadId, replyId);
      setMessages((previous) =>
        previous.filter((message) => message.id !== replyId),
      );
      setDeleteReplyId(null);
      void refreshThreadList(threadId);
    } catch (deleteError) {
      setDeleteReplyError(
        getErrorMessage(deleteError, "Failed to delete this reply."),
      );
    } finally {
      setDeleteReplyPending(false);
    }
  };

  const handleOpenModels = () => {
    onOpenModelManager();
  };

  const renderModelSelector = (menuPlacement: "top" | "bottom") => (
    <RouteModelSelect
      value={selectedModel}
      options={modelOptions}
      onSelect={onSelectModel}
      disabled={busy}
      menuPlacement={menuPlacement}
      aria-label="Chat model"
      className="w-[180px] max-w-[calc(100vw-12rem)] sm:w-[220px]"
      triggerClassName="h-8 text-xs"
    />
  );

  const renderComposer = (centered = false) => (
    <div
      className={cn(
        "chat-composer-body relative overflow-visible border shadow-none",
        centered
          ? "chat-composer-shell-centered mx-auto max-w-3xl"
          : "chat-composer-shell-docked",
      )}
    >
      {pendingImages.length > 0 && (
        <div className="border-b border-[var(--border-muted)] px-4 pb-3 pt-4">
          {hasUnsupportedImages && (
            <p
              role="status"
              className="mb-3 text-xs text-[var(--status-warning-text)]"
            >
              {unsupportedImagesMessage}
            </p>
          )}
          <div className="flex flex-wrap gap-3">
            {pendingImages.map((image) => (
              <div
                key={image.id}
                className={cn(
                  "group relative h-20 w-20 overflow-hidden rounded-xl border border-[var(--border-muted)] bg-[var(--bg-surface-2)]",
                  hasUnsupportedImages && "opacity-50",
                )}
              >
                <button
                  type="button"
                  onClick={() =>
                    setImagePreview({
                      source: image.previewUrl ?? image.source,
                      label: image.label,
                    })
                  }
                  className="h-full w-full"
                  title={`Preview ${image.label}`}
                >
                  <img
                    src={image.source}
                    alt={image.label}
                    className="h-full w-full object-cover"
                  />
                </button>
                <button
                  type="button"
                  onClick={() => removePendingImage(image.id)}
                  className="absolute right-1 top-1 inline-flex h-6 w-6 items-center justify-center rounded-full border border-black/10 bg-black/70 text-white opacity-0 transition-opacity group-hover:opacity-100 focus:opacity-100"
                  title={`Remove ${image.label}`}
                  aria-label={`Remove ${image.label}`}
                >
                  <Trash2 className="h-3 w-3" />
                </button>
                <div className="pointer-events-none absolute inset-x-0 bottom-0 bg-gradient-to-t from-black/70 to-transparent px-2 pb-1.5 pt-4 text-[10px] font-medium text-white">
                  <span className="block truncate">{image.label}</span>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      <textarea
        ref={textareaRef}
        value={input}
        onChange={(event) => {
          const value = event.target.value;
          updateComposerDraft(activeDraftKey, (draft) => ({
            ...draft,
            input: value,
          }));
        }}
        onKeyDown={(event) => {
          // Enter that ends an IME composition picks the text; it does not send.
          if (
            event.key === "Enter" &&
            !event.shiftKey &&
            !event.nativeEvent.isComposing
          ) {
            event.preventDefault();
            void sendMessage();
            return;
          }
          // ↑ in an empty composer edits your last message.
          if (event.key === "ArrowUp" && input.length === 0 && !isStreaming) {
            const lastUser = [...visibleMessages]
              .reverse()
              .find((message) => message.role === "user");
            if (lastUser && isEditableUserMessage(lastUser)) {
              event.preventDefault();
              setEditingMessageId(lastUser.id);
            }
          }
        }}
        placeholder={
          !activeThreadId
            ? supportsImageAttachments
              ? "Ask anything or attach an image..."
              : "Ask anything..."
            : !selectedModel
              ? "Choose a model and ask anything..."
              : !selectedModelReady
                ? "Model selected but not loaded. Open Models to load it."
                : supportsImageAttachments
                  ? "Ask anything or attach an image..."
                  : "Ask anything..."
        }
        className="chat-composer-input w-full resize-none bg-transparent px-5 pt-5 pb-2 text-[0.9375rem] focus:outline-none placeholder:text-[var(--text-muted)]"
        disabled={isStreaming || isPreparingThread || isReadingImages}
      />

      <input
        ref={imageInputRef}
        data-testid="chat-image-input"
        type="file"
        accept={CHAT_IMAGE_ACCEPT}
        multiple
        className="hidden"
        onChange={(event) => {
          void handleComposerImageChange(event);
        }}
      />

      <div
        data-testid="chat-composer-actions"
        className="chat-composer-actions flex flex-wrap items-center justify-between gap-x-3 gap-y-2 px-4 pb-4 pt-2"
      >
        <div className="flex min-w-0 flex-wrap items-center gap-2">
          {supportsImageAttachments && (
            <Button
              variant="outline"
              size="icon"
              onClick={openImagePicker}
              disabled={isStreaming || isPreparingThread || isReadingImages}
              className={cn("h-8 w-8 border p-0", chatSecondaryButtonClass)}
              title="Attach image"
              aria-label="Attach image"
            >
              {isReadingImages ? (
                <Loader2 className="h-3.5 w-3.5 animate-spin" />
              ) : (
                <Plus className="h-3.5 w-3.5" />
              )}
            </Button>
          )}
          {renderModelSelector(centered ? "bottom" : "top")}
          <Button
            variant="outline"
            size="sm"
            onClick={handleOpenModels}
            className={cn("h-8 gap-1.5 border text-xs", chatSecondaryButtonClass)}
          >
            <Settings2 className="w-3.5 h-3.5" />
            Models
          </Button>
          {supportsThinking && !thinkingToggleable && (
            <span
              className={cn(
                "inline-flex h-8 items-center gap-1.5 rounded-lg border px-3 text-xs font-medium",
                chatSecondaryButtonActiveClass,
              )}
              title="This model always thinks before it answers"
            >
              <Brain className="w-3.5 h-3.5" aria-hidden="true" />
              Thinking
            </span>
          )}
          {supportsThinking && thinkingToggleable && (
            <Button
              variant="outline"
              size="sm"
              onClick={() => setIsThinkingEnabled((previous) => !previous)}
              disabled={isStreaming || isPreparingThread}
              className={cn(
                "h-8 gap-1.5 border text-xs shadow-none",
                thinkingEnabledForModel
                  ? chatSecondaryButtonActiveClass
                  : chatSecondaryButtonClass,
              )}
              title={
                thinkingEnabledForModel
                  ? "Thinking mode is enabled"
                  : "Thinking mode is disabled"
              }
              aria-label={
                thinkingEnabledForModel
                  ? "Disable thinking mode"
                  : "Enable thinking mode"
              }
            >
              <Brain className="w-3.5 h-3.5" />
              Thinking {thinkingEnabledForModel ? "On" : "Off"}
            </Button>
          )}
          {thinkingEnabledForModel && supportedReasoningEfforts.length > 0 && (
            <Select
              value={reasoningEffort}
              onValueChange={(value) =>
                setReasoningEffort(value as ChatReasoningEffort)
              }
              disabled={isStreaming || isPreparingThread}
            >
              <SelectTrigger
                aria-label="Reasoning effort"
                title="Reasoning effort"
                className="h-8 w-[7.25rem] border bg-[var(--bg-surface-2)] px-2.5 text-xs shadow-none"
              >
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {supportedReasoningEfforts.map((effort) => (
                  <SelectItem key={effort} value={effort}>
                    {effort === "xhigh"
                      ? "XHigh"
                      : `${effort[0].toUpperCase()}${effort.slice(1)}`}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          )}
        </div>

        <div className="ml-auto flex items-center justify-end">
          <Button
            onClick={isStreaming ? stopStreaming : () => void sendMessage()}
            disabled={
              isPreparingThread ||
              isReadingImages ||
              (!isStreaming &&
                (hasUnsupportedImages ||
                  (input.trim().length === 0 && pendingImages.length === 0)))
            }
            variant="default"
            size="icon"
            title={
              isStreaming
                ? "Cancel response"
                : isPreparingThread
                  ? "Starting chat"
                  : hasUnsupportedImages
                    ? unsupportedImagesMessage
                    : "Send message"
            }
            aria-label={
              isStreaming
                ? "Cancel response"
                : isPreparingThread
                  ? "Starting chat"
                : "Send message"
            }
            className={cn("h-10 w-10 shrink-0", chatComposerSendButtonClass)}
          >
            {isStreaming ? (
              <Square className="w-3.5 h-3.5" />
            ) : isPreparingThread ? (
              <Loader2 className="w-3.5 h-3.5 animate-spin" />
            ) : (
              <Send className="w-3.5 h-3.5" />
            )}
          </Button>
        </div>
      </div>
    </div>
  );

  const selectThread = (threadId: string) => {
    if (isStreaming || isPreparingThread) {
      return;
    }
    setActiveThreadInUrl(threadId);
    setError(null);
  };
  const renderThreadList = (onAfterSelect?: () => void) => (
    <ChatThreadList
      threads={threads}
      activeThreadId={activeThreadId}
      loading={threadsLoading}
      busy={isStreaming || isPreparingThread}
      onSelect={(threadId) => {
        selectThread(threadId);
        onAfterSelect?.();
      }}
      onDelete={openDeleteThreadConfirm}
      onNewChat={() => {
        handleStartNewChat();
        onAfterSelect?.();
      }}
      className="h-full w-full"
    />
  );

  return (
    <div className="flex min-h-0 flex-1 gap-6">
      {showThreadColumn ? (
        <aside
          aria-label="Chat history"
          className="flex w-[15rem] shrink-0 border-r border-[var(--border-subtle)] pr-4"
        >
          {renderThreadList()}
        </aside>
      ) : null}
    <div className="flex min-h-0 min-w-0 flex-1 flex-col gap-4">
      <div
        className={cn(
          "mb-0 flex flex-wrap items-center justify-between gap-3",
          showThreadColumn && !hasConversation && "hidden",
        )}
      >
        <div className="min-w-0 flex-1">
          {hasConversation && (
            <h2 className="truncate text-[15px] font-semibold tracking-tight text-[var(--text-primary)]">
              {activeThread
                ? displayThreadTitle(activeThread.title)
                : "Conversation"}
            </h2>
          )}
        </div>

        {showThreadColumn ? null : (
        <div className="flex shrink-0 items-center gap-2 whitespace-nowrap">
          <RouteHistoryDrawer
            title="Chat History"
            countLabel={`${threads.length} ${threads.length === 1 ? "thread" : "threads"}`}
            open={isHistoryDrawerOpen}
            onOpenChange={handleHistoryDrawerOpenChange}
            trigger={
              <Button
                type="button"
                variant="outline"
                size="sm"
                className={cn("h-9 gap-2", chatSecondaryButtonClass)}
              >
                <History className="h-4 w-4" />
                <span>History</span>
                <span className="inline-flex min-w-5 items-center justify-center rounded-full border border-[var(--border-muted)] bg-[var(--bg-surface-3)] px-1.5 py-0.5 text-[10px] font-semibold text-[var(--text-primary)]">
                  {historyBadgeLabel}
                </span>
              </Button>
            }
          >
            {({ close }) => renderThreadList(close)}
          </RouteHistoryDrawer>

          <Button
            type="button"
            size="sm"
            className={cn("h-9 gap-2", chatAccentButtonClass)}
            onClick={handleStartNewChat}
            disabled={isStreaming || isPreparingThread}
          >
            <Plus className="h-4 w-4" />
            New chat
          </Button>
        </div>
        )}
      </div>

      <div className="relative flex min-h-0 flex-1 flex-col">
        {!hasConversation ? (
          <div className="relative flex-1 flex items-center justify-center px-1 sm:px-4">
            <div className="w-full max-w-3xl">
              <motion.div
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.22 }}
              >
                <h2 className="mb-6 text-center text-[1.75rem] font-semibold tracking-tight text-[var(--text-primary)]">
                  What can I help with?
                </h2>
                {renderComposer(true)}
              </motion.div>

              <div
                className="mt-4 text-center text-xs text-muted-foreground min-h-[18px]"
                role="status"
                aria-live="polite"
                aria-atomic="true"
              >
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
                    role="alert"
                    aria-live="assertive"
                    aria-atomic="true"
                    initial={{ opacity: 0, height: 0 }}
                    animate={{ opacity: 1, height: "auto" }}
                    exit={{ opacity: 0, height: 0 }}
                    className="mt-3 mx-auto max-w-3xl rounded-lg border border-[var(--danger-border)] bg-[var(--danger-bg)] p-2 text-xs text-[var(--danger-text)]"
                  >
                    {error}
                  </motion.div>
                )}
              </AnimatePresence>
            </div>
          </div>
        ) : (
          <div className="relative flex min-h-0 flex-1 flex-col overflow-hidden">
            <div className="relative flex-1 min-h-0">
              <div
                ref={messageViewportRef}
                data-testid="chat-message-viewport"
                onScroll={handleMessageViewportScroll}
                className="h-full overflow-y-auto px-4 sm:px-6 pb-48 pt-3 scrollbar-thin"
              >
                {activeThreadLoadError && !messagesLoading ? (
                  <div className="mx-auto mt-10 max-w-xl">
                    <StatePanel
                      tone="danger"
                      icon={AlertTriangle}
                      title="This chat could not be loaded"
                      description={activeThreadLoadError}
                      actions={
                        <Button
                          type="button"
                          size="sm"
                          variant="outline"
                          onClick={() =>
                            setThreadLoadAttempt((attempt) => attempt + 1)
                          }
                        >
                          Retry
                        </Button>
                      }
                    />
                  </div>
                ) : messagesLoading ? (
                  <div
                    className="max-w-3xl mx-auto p-4 text-center text-xs text-muted-foreground flex items-center justify-center gap-2"
                    role="status"
                    aria-live="polite"
                  >
                    <Loader2 className="w-4 h-4 animate-spin" />
                    Loading conversation...
                  </div>
                ) : (
                  <div className="mx-auto flex max-w-3xl flex-col gap-7">
                    {visibleMessages.map((message, index) => {
                      const isUser = message.role === "user";
                      // A stopped reply the server has not stored yet has
                      // nothing to retry or delete there.
                      const isStored = !message.id.startsWith("tmp-");
                      const isStreamingReply =
                        !isUser &&
                        index === visibleMessages.length - 1 &&
                        isStreaming &&
                        streamingThreadId === activeThreadId;
                      // Parse each reply by the model that wrote it (stored
                      // per message; the thread model for older replies), so
                      // past reasoning does not depend on today's toggle.
                      const writerImplicitThink = message.model_id
                        ? writesImplicitThink(message.model_id)
                        : implicitOpenThinkTagModel;
                      const messageKey = message.id;
                      const previous = visibleMessages[index - 1];
                      const startsDay =
                        !previous ||
                        !isSameCalendarDay(previous.created_at, message.created_at);

                      return (
                        <Fragment key={messageKey}>
                          {startsDay && (
                            <DaySeparator
                              label={messageDayLabel(message.created_at)}
                            />
                          )}
                          <motion.div
                            initial={{ opacity: 0, y: 8 }}
                            animate={{ opacity: 1, y: 0 }}
                            className="flex flex-col"
                          >
                            {isUser ? (
                              <UserMessage
                                message={message}
                                onOpenImage={setImagePreview}
                                editable={
                                  !busy && isEditableUserMessage(message)
                                }
                                editing={editingMessageId === messageKey}
                                onEdit={() => setEditingMessageId(messageKey)}
                                onCancelEdit={() => setEditingMessageId(null)}
                                onSubmitEdit={(text) =>
                                  submitEdit(messageKey, text)
                                }
                              />
                            ) : (
                              <AssistantMessage
                                message={message}
                                isStreamingReply={isStreamingReply}
                                isLatestReply={index === lastAssistantIndex}
                                implicitThink={writerImplicitThink}
                                thinkingEnabled={streamThinkingEnabled}
                                modelLabel={
                                  message.model_id
                                    ? (modelOptions.find(
                                        (option) =>
                                          option.value === message.model_id,
                                      )?.label ?? message.model_id)
                                    : null
                                }
                                stats={streamedStats[messageKey] ?? null}
                                readAloudState={
                                  readAloud?.messageId === messageKey
                                    ? readAloud.status
                                    : "idle"
                                }
                                onReadAloud={(text) =>
                                  toggleReadAloud(messageKey, text)
                                }
                                isThoughtExpanded={!!expandedThoughts[messageKey]}
                                onToggleThought={() =>
                                  setExpandedThoughts((previous) => ({
                                    ...previous,
                                    [messageKey]: !previous[messageKey],
                                  }))
                                }
                                turnStartedAt={
                                  isStreamingReply ? streamStartedAt : null
                                }
                                reasoningMs={
                                  reasoningDurations[messageKey] ??
                                  message.reasoning_ms ??
                                  null
                                }
                                canRetry={
                                  !busy &&
                                  isStored &&
                                  index === lastAssistantIndex &&
                                  index === visibleMessages.length - 1
                                }
                                retryModels={retryModelOptions}
                                currentModelId={selectedModel}
                                onRetry={(modelId) =>
                                  retryReply(messageKey, modelId)
                                }
                                onDelete={
                                  busy || !isStored
                                    ? undefined
                                    : () => {
                                        setDeleteReplyError(null);
                                        setDeleteReplyId(messageKey);
                                      }
                                }
                                onReasoningFinished={(durationMs) =>
                                  setReasoningDurations((previous) => ({
                                    ...previous,
                                    [messageKey]: durationMs,
                                  }))
                                }
                              />
                            )}
                          </motion.div>
                        </Fragment>
                      );
                    })}
                  </div>
                )}
              </div>

              <div className="pointer-events-none absolute bottom-0 left-0 right-0 z-30 bg-[var(--bg-app)] px-4 pb-4 pt-4 sm:px-6">
                <div className="relative max-w-3xl mx-auto pointer-events-auto">
                  {showJumpToLatest && (
                    <button
                      type="button"
                      onClick={jumpToLatest}
                      className="absolute -top-12 left-1/2 inline-flex h-8 -translate-x-1/2 items-center gap-1.5 rounded-full border border-[var(--border-default)] bg-[var(--surface-panel)] px-3 text-[12.5px] text-[var(--text-secondary)] shadow-[var(--shadow-soft)] transition-colors hover:text-[var(--text-primary)] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[var(--border-strong)]"
                    >
                      <ArrowDown className="h-3.5 w-3.5" aria-hidden="true" />
                      Jump to latest
                    </button>
                  )}
                  <AnimatePresence>
                    {error && (
                      <motion.div
                        role="alert"
                        aria-live="assertive"
                        aria-atomic="true"
                        initial={{ opacity: 0, height: 0, y: 10 }}
                        animate={{ opacity: 1, height: "auto", y: 0 }}
                        exit={{ opacity: 0, height: 0, y: 10 }}
                        className="mb-4 p-3 rounded-lg bg-destructive text-destructive-foreground shadow-lg text-sm font-medium flex items-center gap-2"
                      >
                        {error}
                      </motion.div>
                    )}
                  </AnimatePresence>

                  {showHistoryImageNotice && (
                    <div
                      role="note"
                      className="mb-2 flex flex-wrap items-center gap-x-3 gap-y-1 rounded-lg border border-[var(--status-info-border)] bg-[var(--status-info-bg)] px-3 py-2 text-xs text-[var(--status-info-text)]"
                    >
                      <span>
                        This chat has images. {modelLabel ?? "This model"} reads
                        text only, so earlier images are left out.
                      </span>
                      {imageModelOption && (
                        <button
                          type="button"
                          className="ml-auto font-medium underline underline-offset-2"
                          onClick={() => onSelectModel(imageModelOption.value)}
                        >
                          Switch to {imageModelOption.label}
                        </button>
                      )}
                    </div>
                  )}
                  {renderComposer(false)}
                  <p className="mt-2 text-center text-[11.5px] text-[var(--text-subtle)]">
                    Runs on this device. Enter to send, Shift+Enter for a new
                    line.
                  </p>
                </div>
              </div>
            </div>
          </div>
        )}
      </div>

      <ConfirmDeleteDialog
        noun={deleteReplyId ? "reply" : null}
        description="The reply is removed from this chat, and the model no longer sees it in later turns."
        pending={deleteReplyPending}
        error={deleteReplyError}
        onConfirm={() => void confirmDeleteReply()}
        onCancel={() => setDeleteReplyId(null)}
      />

      <div className="sr-only" role="status" aria-live="polite" aria-atomic="true">
        {replyAnnouncement}
      </div>

      <Dialog
        open={!!deleteTargetThread}
        onOpenChange={(open) => {
          if (!open) {
            closeDeleteThreadConfirm();
          }
        }}
      >
        {deleteTargetThread ? (
          <DialogContent className="max-w-md border-[var(--border-strong)] bg-[var(--bg-surface-1)] p-5">
            <DialogTitle className="sr-only">Delete chat thread?</DialogTitle>
            <div className="flex items-start gap-3">
              <div className="mt-0.5 rounded-full border border-[var(--danger-border)] bg-[var(--danger-bg)] p-2 text-[var(--danger-text)]">
                <AlertTriangle className="h-4 w-4" />
              </div>
              <div className="min-w-0 flex-1">
                <h3 className="text-sm font-semibold text-[var(--text-primary)]">
                  Delete chat thread?
                </h3>
                <DialogDescription className="mt-1 text-sm text-[var(--text-muted)]">
                  This permanently removes the selected conversation and all of
                  its messages.
                </DialogDescription>
                <p className="mt-2 max-w-full whitespace-normal break-words text-xs text-[var(--text-subtle)] [overflow-wrap:anywhere]">
                  {displayThreadTitle(deleteTargetThread.title)}
                </p>
              </div>
            </div>

            <div className="mt-5 flex items-center justify-end gap-2">
              <Button
                onClick={closeDeleteThreadConfirm}
                variant="outline"
                size="sm"
                className={cn("h-8", chatSecondaryButtonClass)}
                disabled={deleteThreadPending}
              >
                Cancel
              </Button>
              <Button
                onClick={confirmDeleteThread}
                variant="destructive"
                size="sm"
                className="h-8 gap-1.5"
                disabled={deleteThreadPending}
              >
                {deleteThreadPending ? (
                  <Loader2 className="h-3.5 w-3.5 animate-spin" />
                ) : (
                  <Trash2 className="h-3.5 w-3.5" />
                )}
                Delete thread
              </Button>
            </div>
          </DialogContent>
        ) : null}
      </Dialog>

      <Dialog
        open={!!imagePreview}
        onOpenChange={(open) => {
          if (!open) {
            setImagePreview(null);
          }
        }}
      >
        <DialogContent className="max-w-[min(92vw,1100px)] p-3 sm:p-4 bg-background/95 border-[var(--border-muted)]">
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
    </div>
  );
}
