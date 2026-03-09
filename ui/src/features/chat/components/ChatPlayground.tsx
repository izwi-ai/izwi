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
  History,
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
  ImageIcon,
  Film,
  X,
} from "lucide-react";
import { useSearchParams } from "react-router-dom";
import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";
import { RouteHistoryDrawer } from "@/components/RouteHistoryDrawer";
import { Dialog, DialogContent, DialogTitle } from "@/components/ui/dialog";
import {
  DEFAULT_SYSTEM_PROMPT,
  DEFAULT_THREAD_TITLE,
  MAX_MEDIA_ATTACHMENTS,
  MAX_MEDIA_ATTACHMENT_BYTES,
  MAX_MEDIA_ATTACHMENT_MB,
  THINKING_SYSTEM_PROMPT,
  type ChatPlaygroundProps,
  type ComposerMediaItem,
  type ComposerMediaKind,
  type GenerateTitleArgs,
  type ImagePreviewState,
  type ModelOption,
  buildThreadContentParts,
  buildUserDisplayContent,
  createMediaItemId,
  defaultThinkingEnabledForModel,
  displayThreadTitle,
  extractLatestStats,
  fallbackThreadTitleFromUserMessage,
  fileToDataUrl,
  formatBytes,
  formatThreadTimestamp,
  getErrorMessage,
  isLfm25ThinkingModel,
  isQwen35ChatModel,
  normalizeGeneratedThreadTitle,
  parseAssistantContent,
  parseUserMessageDisplayFromContentParts,
  stripThinkingArtifacts,
  supportsImplicitOpenThinkTagParsing,
  threadPreviewFromContent,
} from "@/features/chat/playground/support";
import { api, type ChatThread, type ChatThreadMessageRecord } from "@/api";
import { MarkdownContent } from "@/components/ui/MarkdownContent";

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
  const [isThinkingEnabled, setIsThinkingEnabled] = useState(() =>
    defaultThinkingEnabledForModel(selectedModel),
  );
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
  const streamingThinkingRef = useRef<HTMLDivElement | null>(null);
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
  const historyBadgeLabel = useMemo(() => {
    if (threads.length > 99) {
      return "99+";
    }
    return String(Math.max(0, threads.length));
  }, [threads.length]);

  const hasConversation =
    !!activeThreadId &&
    (visibleMessages.length > 0 || isStreaming || messagesLoading);
  const isEmptyChatWorkspace = !hasConversation;
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

  useEffect(() => {
    if (!supportsThinking) {
      setIsThinkingEnabled(false);
      return;
    }
    setIsThinkingEnabled(defaultThinkingEnabledForModel(selectedModel));
  }, [selectedModel, supportsThinking]);

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
    if (!isStreaming || !streamingThinkingRef.current) {
      return;
    }
    streamingThinkingRef.current.scrollTop =
      streamingThinkingRef.current.scrollHeight;
  }, [isStreaming, messages]);

  useEffect(() => {
    if (!textareaRef.current) {
      return;
    }

    const minHeight = hasConversation ? 72 : 88;
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
    if (isEmptyChatWorkspace || isStreaming || isPreparingThread) {
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
  }, [
    clearMediaItems,
    isEmptyChatWorkspace,
    isPreparingThread,
    isStreaming,
    selectedModel,
    setActiveThreadInUrl,
  ]);

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
      ? thinkingEnabledForModel === defaultThinkingEnabledForModel(selectedModel)
        ? undefined
        : thinkingEnabledForModel
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

  const getStatusToneClassName = (option: ModelOption): string => {
    if (option.isReady) {
      return "chat-model-status-ready";
    }

    const normalizedStatus = option.statusLabel.toLowerCase();

    if (
      normalizedStatus.includes("downloading") ||
      normalizedStatus.includes("loading")
    ) {
      return "chat-model-status-loading";
    }

    if (normalizedStatus.includes("error")) {
      return "chat-model-status-error";
    }

    return "chat-model-status-idle";
  };

  const renderModelSelector = (
    placement: "composer" | "header" = "composer",
  ) => (
    <div
      className={cn(
        "relative z-40 inline-block",
        placement === "header"
          ? "w-full max-w-[320px]"
          : "w-[240px] max-w-[calc(100vw-9rem)] sm:w-[300px]",
      )}
      ref={modelMenuRef}
    >
      <Button
        variant="outline"
        onClick={() => setIsModelMenuOpen((previous) => !previous)}
        className={cn(
          "h-9 w-full justify-between rounded-lg border px-3 text-xs font-medium shadow-none",
          selectedOption?.isReady
            ? "chat-model-selector-btn-ready"
            : "chat-model-selector-btn-idle",
        )}
      >
        <span className="flex-1 min-w-0 truncate text-left">
          {selectedOption?.label || "Select model"}
        </span>
        <ChevronDown
          className={cn(
            "w-3.5 h-3.5 shrink-0 opacity-50 transition-transform",
            isModelMenuOpen && "rotate-180",
          )}
        />
      </Button>

      <AnimatePresence>
        {isModelMenuOpen && (
          <motion.div
            initial={{ opacity: 0, y: 6, scale: 0.98 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: 6, scale: 0.98 }}
            transition={{ duration: 0.16 }}
            className={cn(
              "chat-model-menu absolute left-0 right-0 z-[90] rounded-xl border p-1.5 shadow-2xl",
              placement === "header" ? "top-full mt-2" : "bottom-11",
            )}
          >
            <div className="max-h-64 space-y-0.5 overflow-y-auto pr-1">
              {modelOptions.map((option) => (
                <button
                  key={option.value}
                  onClick={() => {
                    onSelectModel(option.value);
                    setIsModelMenuOpen(false);
                  }}
                  className={cn(
                    "relative flex w-full cursor-default select-none items-center rounded-lg border px-3 py-2 text-sm outline-none transition-colors focus-visible:ring-2 focus-visible:ring-ring/45 focus-visible:ring-offset-2 focus-visible:ring-offset-background",
                    selectedOption?.value === option.value &&
                      "chat-model-option-active",
                    selectedOption?.value !== option.value &&
                      "chat-model-option-idle",
                  )}
                >
                  <div className="flex min-w-0 w-full flex-col items-start">
                    <span className="chat-model-option-label w-full truncate text-left font-medium">
                      {option.label}
                    </span>
                    <span
                      className={cn(
                        "mt-1 inline-flex items-center rounded-sm border px-1.5 py-0.5 text-[10px] font-semibold uppercase tracking-wider",
                        getStatusToneClassName(option),
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
        "chat-composer-body relative overflow-visible rounded-xl border shadow-sm",
        centered
          ? "chat-composer-shell-centered mx-auto max-w-3xl shadow-md"
          : "chat-composer-shell-docked",
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
        className="chat-composer-input w-full resize-none bg-transparent px-4 pt-4 pb-3 text-[0.9375rem] focus:outline-none placeholder:text-[var(--text-muted)]"
        disabled={isStreaming || isPreparingThread}
      />

      <div className="flex flex-wrap items-center justify-between gap-2 px-3 pb-3 pt-1 border-t border-[var(--border-muted)]/30 mt-1">
        <div className="flex items-center gap-2 flex-wrap">
          <Button
            variant="outline"
            size="icon"
            onClick={() => mediaInputRef.current?.click()}
            disabled={
              isStreaming ||
              isPreparingThread ||
              mediaItems.length >= MAX_MEDIA_ATTACHMENTS ||
              !selectedModelSupportsMedia
            }
            title={
              selectedModelSupportsMedia
                ? "Attach image or video"
                : "Image/video upload is available only for Qwen3.5 models"
            }
            aria-label={
              selectedModelSupportsMedia
                ? "Attach image or video"
                : "Image/video upload is available only for Qwen3.5 models"
            }
            className="h-8 w-8 rounded-full border-[var(--border-muted)] bg-[var(--bg-surface-2)] text-[var(--text-muted)] shadow-none hover:border-[var(--border-strong)] hover:bg-[var(--bg-surface-3)] hover:text-[var(--text-primary)]"
          >
            <Plus className="w-4 h-4" />
          </Button>
          <Button
            variant="outline"
            size="sm"
            onClick={handleOpenModels}
            className="chat-models-button h-8 gap-1.5 border text-xs shadow-none"
          >
            <Settings2 className="w-3.5 h-3.5" />
            Models
          </Button>
          {supportsThinking && (
            <Button
              variant="outline"
              size="sm"
              onClick={() => setIsThinkingEnabled((previous) => !previous)}
              disabled={isStreaming || isPreparingThread}
              className={cn(
                "h-8 gap-1.5 border text-xs shadow-none",
                thinkingEnabledForModel
                  ? "chat-thinking-mode-btn-on"
                  : "chat-thinking-mode-btn-off",
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
        </div>

        <div className="flex items-center gap-2 w-full sm:w-auto justify-end flex-wrap sm:flex-nowrap">
          <Button
            onClick={isStreaming ? stopStreaming : () => void sendMessage()}
            disabled={
              isPreparingThread ||
              (!isStreaming && !input.trim() && mediaItems.length === 0)
            }
            variant={isStreaming ? "destructive" : "default"}
            size="icon"
            title={
              isStreaming
                ? "Cancel response"
                : isPreparingThread
                  ? "Starting chat"
                  : "Send message"
            }
            aria-label={
              isStreaming
                ? "Cancel response"
                : isPreparingThread
                  ? "Starting chat"
                  : "Send message"
            }
            className={cn(
              "h-9 w-9 shrink-0",
              !isStreaming && "chat-send-button shadow-none",
            )}
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

  return (
    <div className="flex min-h-0 flex-1 flex-col gap-4">
      <input
        ref={mediaInputRef}
        type="file"
        accept="image/*,video/*"
        multiple
        disabled={!selectedModelSupportsMedia}
        onChange={handleSelectMedia}
        className="hidden"
      />
      <div className="mb-0 flex flex-wrap items-center justify-between gap-3">
        <div className="min-w-[220px] max-w-full flex-1 sm:max-w-[320px]">
          {renderModelSelector("header")}
        </div>

        <div className="flex shrink-0 items-center gap-2 whitespace-nowrap">
          <RouteHistoryDrawer
            title="Chat History"
            countLabel={`${threads.length} ${threads.length === 1 ? "thread" : "threads"}`}
            trigger={
              <Button
                type="button"
                variant="outline"
                size="sm"
                className="h-9 gap-2 rounded-lg border-[var(--border-muted)] bg-[var(--bg-surface-2)] text-[var(--text-secondary)] shadow-none hover:border-[var(--border-strong)] hover:bg-[var(--bg-surface-3)] hover:text-[var(--text-primary)]"
              >
                <History className="h-4 w-4" />
                <span>History</span>
                <span className="inline-flex min-w-5 items-center justify-center rounded-full border border-[var(--border-muted)] bg-[var(--bg-surface-3)] px-1.5 py-0.5 text-[10px] font-semibold text-[var(--text-primary)]">
                  {historyBadgeLabel}
                </span>
              </Button>
            }
          >
            {({ close }) => (
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
                            close();
                          }}
                          onKeyDown={(event) => {
                            if (event.key === "Enter" || event.key === " ") {
                              event.preventDefault();
                              if (isStreaming || isPreparingThread) {
                                return;
                              }
                              setActiveThreadInUrl(thread.id);
                              setError(null);
                              close();
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
                                  (isStreaming || isPreparingThread) &&
                                    "opacity-50",
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
            )}
          </RouteHistoryDrawer>

          <Button
            type="button"
            size="sm"
            className="h-9 gap-2 rounded-lg bg-[var(--accent-solid)] text-[var(--text-on-accent)] shadow-none hover:opacity-90"
            onClick={() => void handleCreateThread()}
            disabled={isEmptyChatWorkspace || isStreaming || isPreparingThread}
          >
            <Plus className="h-4 w-4" />
            New chat
          </Button>
        </div>
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
                              treatNoTagAsThinking: isLastAssistant && isStreaming,
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
                                      className="leading-relaxed [&_p]:m-0 [&_p]:bg-transparent [&_mark]:bg-transparent [&_code]:bg-transparent [&_code]:border-0 [&_code]:px-0 [&_code]:py-0 [&_code]:font-inherit"
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
                                      <div
                                        ref={streamingThinkingRef}
                                        className="h-40 overflow-y-auto whitespace-pre-wrap rounded-md border border-[var(--border-muted)]/70 bg-background/60 px-2.5 py-2 font-mono text-[11px] leading-relaxed scrollbar-thin sm:h-48"
                                      >
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
                                      <div className="mt-2 h-40 overflow-y-auto whitespace-pre-wrap rounded-lg border border-[var(--border-muted)] bg-muted/30 px-3 py-2 text-xs font-mono text-[11px] leading-relaxed text-muted-foreground scrollbar-thin sm:h-48">
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
