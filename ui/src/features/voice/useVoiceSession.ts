import { useCallback, useEffect, useRef, useState } from "react";

import { api } from "@/api";
import {
  buildVoiceRealtimeV2SessionStart,
  buildVoiceRealtimeWebSocketUrl,
  decodePcmI16Bytes,
  encodeLiveMicPcm16,
  encodeVoiceRealtimeClientPcm16Frame,
  formatModelVariantLabel,
  isValidVoiceRealtimeV2Successor,
  isVoiceRealtimeServerEvent,
  isVoiceRealtimeV2EnvelopeHeader,
  isVoiceRealtimeV2ServerEnvelope,
  makeTranscriptEntryId,
  normalizeVoiceRealtimeV2Event,
  parseFinalAnswer,
  parseVoiceRealtimeAssistantAudioBinaryChunk,
  VOICE_REALTIME_DUCK_GAIN,
  voiceRealtimeEndsUserSpeech,
  voiceRealtimePlaybackAction,
  type RuntimeStatus,
  type TranscriptEntry,
  type VoiceRealtimeAssistantAudioBinaryChunk,
  type VoiceRealtimeClientMessage,
  type VoiceRealtimeServerEvent,
  type VoiceRealtimeV2EnvelopeHeader,
} from "@/features/voice/realtime/support";

/** The models one conversation runs on. */
export type VoiceSessionEngine =
  | { mode: "modular"; asr: string; text: string; tts: string }
  | { mode: "unified"; s2s: string };

/** Why a conversation ended without the user ending it. */
export interface VoiceSessionFailure {
  kind: "connection" | "microphone" | "model" | "other";
  message: string;
}

interface UseVoiceSessionOptions {
  engine: VoiceSessionEngine | null;
  speaker: string;
  systemPrompt: string;
}

const MICROPHONE_ERRORS = new Set([
  "NotAllowedError",
  "NotFoundError",
  "NotReadableError",
  "SecurityError",
  "OverconstrainedError",
]);

/**
 * One realtime voice conversation: the microphone stream, the
 * `/v1/voice/realtime/ws` socket and the assistant's streamed audio.
 * Settings (engine, voice, prompt) apply when a conversation starts.
 */
export function useVoiceSession({
  engine,
  speaker,
  systemPrompt,
}: UseVoiceSessionOptions) {
  const [runtimeStatus, setRuntimeStatus] = useState<RuntimeStatus>("idle");
  const [failure, setFailure] = useState<VoiceSessionFailure | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const [transcript, setTranscript] = useState<TranscriptEntry[]>([]);
  const [audioLevel, setAudioLevel] = useState(0);
  const [isMicMuted, setIsMicMuted] = useState(false);
  const [startedAt, setStartedAt] = useState<number | null>(null);
  const [isAwaitingMicrophone, setIsAwaitingMicrophone] = useState(false);

  const streamRef = useRef<MediaStream | null>(null);
  const audioContextRef = useRef<AudioContext | null>(null);
  const analyserRef = useRef<AnalyserNode | null>(null);
  const streamingProcessorRef = useRef<ScriptProcessorNode | null>(null);
  const streamingProcessorSinkRef = useRef<GainNode | null>(null);
  const vadTimerRef = useRef<number | null>(null);
  const processingRef = useRef(false);
  const runtimeStatusRef = useRef<RuntimeStatus>("idle");
  const isSessionActiveRef = useRef(false);
  const isMicMutedRef = useRef(false);
  const turnIdRef = useRef(0);

  const ttsPlaybackContextRef = useRef<AudioContext | null>(null);
  const ttsPlaybackGainRef = useRef<GainNode | null>(null);
  const ttsPlaybackSourcesRef = useRef<Set<AudioBufferSourceNode>>(new Set());
  const ttsNextPlaybackTimeRef = useRef(0);
  const ttsSampleRateRef = useRef(24000);
  const ttsStreamSessionRef = useRef(0);
  /** Assistant playback is ducked while the user may be speaking over it. */
  const playbackDuckedRef = useRef(false);
  const playbackGainValue = useCallback(
    () => (playbackDuckedRef.current ? VOICE_REALTIME_DUCK_GAIN : 1),
    [],
  );
  const voiceWsRef = useRef<WebSocket | null>(null);
  const voiceWsConnectingRef = useRef<Promise<WebSocket> | null>(null);
  const voiceWsSessionReadyRef = useRef(false);
  const voiceWsInputStreamStartedRef = useRef(false);
  const voiceWsInputStreamStartingRef = useRef<Promise<void> | null>(null);
  const voiceWsInputStreamReadyPromiseRef = useRef<Promise<void> | null>(null);
  const voiceWsInputStreamReadySettleRef = useRef<
    ((error?: Error) => void) | null
  >(null);
  const voiceWsInputFrameSeqRef = useRef(0);
  const voiceWsV2EventRef = useRef<VoiceRealtimeV2EnvelopeHeader | null>(null);
  const voiceMinAcceptedAssistantSeqRef = useRef(0);
  const voiceLatestSpeechSeqRef = useRef(0);
  const setPlaybackDucked = useCallback(
    (ducked: boolean) => {
      playbackDuckedRef.current = ducked;
      if (ttsPlaybackGainRef.current) {
        ttsPlaybackGainRef.current.gain.value = playbackGainValue();
      }
    },
    [playbackGainValue],
  );
  const voiceUserEntryIdsRef = useRef<Map<string, string>>(new Map());
  const voiceAssistantEntryIdsRef = useRef<Map<string, string>>(new Map());
  const voiceWsPlaybackRef = useRef<{
    utteranceId: string;
    utteranceSeq: number;
    streamSession: number;
    streamDone: boolean;
    playbackStarted: boolean;
  } | null>(null);

  useEffect(() => {
    runtimeStatusRef.current = runtimeStatus;
  }, [runtimeStatus]);

  const stopTtsStreamingPlayback = useCallback(() => {
    ttsStreamSessionRef.current += 1;

    for (const source of ttsPlaybackSourcesRef.current) {
      try {
        source.stop();
      } catch {
        // Ignore already-stopped sources.
      }
    }
    ttsPlaybackSourcesRef.current.clear();

    if (ttsPlaybackContextRef.current) {
      ttsPlaybackContextRef.current.close().catch(() => {});
      ttsPlaybackContextRef.current = null;
    }
    ttsPlaybackGainRef.current = null;

    ttsNextPlaybackTimeRef.current = 0;
    ttsSampleRateRef.current = 24000;
  }, []);

  const clearAudioPlayback = stopTtsStreamingPlayback;

  const settleVoiceRealtimeInputStreamReady = useCallback((error?: Error) => {
    const settle = voiceWsInputStreamReadySettleRef.current;
    if (!settle) return;

    voiceWsInputStreamReadySettleRef.current = null;
    voiceWsInputStreamReadyPromiseRef.current = null;
    settle(error);
  }, []);

  const waitForVoiceRealtimeInputStreamReady = useCallback(
    (timeoutMs = 8000): Promise<void> => {
      if (voiceWsInputStreamStartedRef.current) {
        return Promise.resolve();
      }
      if (voiceWsInputStreamReadyPromiseRef.current) {
        return voiceWsInputStreamReadyPromiseRef.current;
      }

      const promise = new Promise<void>((resolve, reject) => {
        const settle = (error?: Error) => {
          window.clearTimeout(timeoutId);
          if (error) {
            reject(error);
            return;
          }
          resolve();
        };

        const timeoutId = window.setTimeout(() => {
          if (voiceWsInputStreamReadySettleRef.current === settle) {
            voiceWsInputStreamReadySettleRef.current = null;
            voiceWsInputStreamReadyPromiseRef.current = null;
          }
          reject(
            new Error(
              "Voice realtime input stream did not become ready in time",
            ),
          );
        }, timeoutMs);

        voiceWsInputStreamReadySettleRef.current = settle;
      });

      voiceWsInputStreamReadyPromiseRef.current = promise;
      return promise;
    },
    [],
  );

  const closeVoiceRealtimeSocket = useCallback(
    (reason?: string) => {
      voiceWsSessionReadyRef.current = false;
      voiceWsInputStreamStartedRef.current = false;
      voiceWsInputFrameSeqRef.current = 0;
      voiceWsV2EventRef.current = null;
      voiceWsInputStreamStartingRef.current = null;
      voiceWsConnectingRef.current = null;
      settleVoiceRealtimeInputStreamReady(
        new Error("Voice realtime input stream stopped"),
      );

      const socket = voiceWsRef.current;
      voiceWsRef.current = null;
      if (socket) {
        try {
          if (
            socket.readyState === WebSocket.OPEN ||
            socket.readyState === WebSocket.CONNECTING
          ) {
            socket.close(1000, reason || "session_closed");
          }
        } catch {
          // Ignore close failures.
        }
      }
    },
    [settleVoiceRealtimeInputStreamReady],
  );

  const stopSession = useCallback(() => {
    isSessionActiveRef.current = false;
    turnIdRef.current += 1;
    processingRef.current = false;
    setRuntimeStatus("idle");
    setAudioLevel(0);

    if (vadTimerRef.current != null) {
      window.clearInterval(vadTimerRef.current);
      vadTimerRef.current = null;
    }

    if (
      voiceWsRef.current &&
      voiceWsRef.current.readyState === WebSocket.OPEN
    ) {
      try {
        voiceWsRef.current.send(
          JSON.stringify({
            type: "input_stream_stop",
          } satisfies VoiceRealtimeClientMessage),
        );
      } catch {
        // Best-effort during shutdown.
      }
    }

    if (streamingProcessorRef.current) {
      try {
        streamingProcessorRef.current.disconnect();
      } catch {
        // Ignore.
      }
      streamingProcessorRef.current.onaudioprocess = null;
      streamingProcessorRef.current = null;
    }
    if (streamingProcessorSinkRef.current) {
      try {
        streamingProcessorSinkRef.current.disconnect();
      } catch {
        // Ignore.
      }
      streamingProcessorSinkRef.current = null;
    }

    if (streamRef.current) {
      streamRef.current.getTracks().forEach((t) => t.stop());
      streamRef.current = null;
    }

    if (audioContextRef.current) {
      audioContextRef.current.close().catch(() => {});
      audioContextRef.current = null;
    }

    closeVoiceRealtimeSocket("session_stopped");
    voiceWsPlaybackRef.current = null;
    voiceMinAcceptedAssistantSeqRef.current = 0;
    voiceLatestSpeechSeqRef.current = 0;
    playbackDuckedRef.current = false;
    voiceUserEntryIdsRef.current.clear();
    voiceAssistantEntryIdsRef.current.clear();

    analyserRef.current = null;
    clearAudioPlayback();
  }, [clearAudioPlayback, closeVoiceRealtimeSocket]);

  useEffect(() => {
    return () => stopSession();
  }, [stopSession]);

  /** End the conversation because of `failure`, which the page then shows. */
  const failSession = useCallback(
    (nextFailure: VoiceSessionFailure) => {
      stopSession();
      setFailure(nextFailure);
    },
    [stopSession],
  );

  /** A problem with one turn; the conversation carries on. */
  const showNotice = useCallback((message: string) => {
    setNotice(message);
  }, []);

  const appendTranscriptEntry = useCallback((entry: TranscriptEntry) => {
    setTranscript((prev) => [...prev, entry]);
  }, []);

  const setTranscriptEntryText = useCallback(
    (entryId: string, text: string) => {
      setTranscript((prev) => {
        const index = prev.findIndex((entry) => entry.id === entryId);
        if (index === -1) {
          return prev;
        }
        const next = [...prev];
        next[index] = {
          ...next[index],
          text,
        };
        return next;
      });
    },
    [],
  );

  const removeTranscriptEntry = useCallback((entryId: string) => {
    setTranscript((prev) => prev.filter((entry) => entry.id !== entryId));
  }, []);

  const sendVoiceRealtimeJson = useCallback(
    (message: VoiceRealtimeClientMessage) => {
      const socket = voiceWsRef.current;
      if (!socket || socket.readyState !== WebSocket.OPEN) {
        throw new Error("Voice realtime websocket is not connected");
      }
      socket.send(JSON.stringify(message));
    },
    [],
  );

  const sendVoiceRealtimeBinary = useCallback((data: Uint8Array) => {
    const socket = voiceWsRef.current;
    if (!socket || socket.readyState !== WebSocket.OPEN) {
      throw new Error("Voice realtime websocket is not connected");
    }
    socket.send(data);
  }, []);

  const finalizeVoiceWsPlaybackIfComplete = useCallback(
    (utteranceSeq: number, streamSession: number) => {
      const active = voiceWsPlaybackRef.current;
      if (!active) return;
      if (active.utteranceSeq !== utteranceSeq) return;
      if (active.streamSession !== streamSession) return;
      if (!active.streamDone || ttsPlaybackSourcesRef.current.size > 0) {
        return;
      }

      if (ttsStreamSessionRef.current === streamSession) {
        if (ttsPlaybackContextRef.current) {
          ttsPlaybackContextRef.current.close().catch(() => {});
          ttsPlaybackContextRef.current = null;
        }

        ttsPlaybackSourcesRef.current.clear();
        ttsNextPlaybackTimeRef.current = 0;
      }

      if (
        voiceWsPlaybackRef.current &&
        voiceWsPlaybackRef.current.streamSession === streamSession
      ) {
        voiceWsPlaybackRef.current = null;
      }

      processingRef.current = false;
      if (
        isSessionActiveRef.current &&
        runtimeStatusRef.current !== "user_speaking"
      ) {
        setRuntimeStatus("listening");
      }
    },
    [],
  );

  const handleVoiceRealtimeAssistantAudioBinaryChunk = useCallback(
    (chunk: VoiceRealtimeAssistantAudioBinaryChunk) => {
      const playback = voiceWsPlaybackRef.current;
      if (!playback) return;
      if (playback.utteranceSeq !== chunk.utteranceSeq) {
        return;
      }

      const context = ttsPlaybackContextRef.current;
      if (!context) return;

      if (
        chunk.sampleRate > 0 &&
        ttsSampleRateRef.current !== chunk.sampleRate
      ) {
        ttsSampleRateRef.current = chunk.sampleRate;
      }

      const samples = decodePcmI16Bytes(chunk.pcm16Bytes);
      if (samples.length === 0) return;

      if (!playback.playbackStarted) {
        playback.playbackStarted = true;
        processingRef.current = false;
        if (isSessionActiveRef.current) {
          setRuntimeStatus("assistant_speaking");
        }
      }


      const buffer = context.createBuffer(
        1,
        samples.length,
        ttsSampleRateRef.current,
      );
      const samplesForPlayback = new Float32Array(samples.length);
      samplesForPlayback.set(samples);
      buffer.copyToChannel(samplesForPlayback, 0);

      const source = context.createBufferSource();
      source.buffer = buffer;
      source.connect(ttsPlaybackGainRef.current ?? context.destination);

      const scheduledAt = Math.max(
        context.currentTime + 0.02,
        ttsNextPlaybackTimeRef.current,
      );
      source.start(scheduledAt);
      ttsNextPlaybackTimeRef.current =
        scheduledAt + buffer.duration;

      const streamSession = playback.streamSession;
      const utteranceSeq = playback.utteranceSeq;
      ttsPlaybackSourcesRef.current.add(source);
      source.onended = () => {
        ttsPlaybackSourcesRef.current.delete(source);
        finalizeVoiceWsPlaybackIfComplete(utteranceSeq, streamSession);
      };

      if (chunk.isFinal) {
        playback.streamDone = true;
      }

      if (context.state === "suspended") {
        context.resume().catch(() => {});
      }
    },
    [finalizeVoiceWsPlaybackIfComplete],
  );

  /**
   * The transcript line of one utterance, added on its first event. The
   * typed protocol sends no start events, so any text event may be first.
   */
  const transcriptEntryId = useCallback(
    (role: TranscriptEntry["role"], utteranceId: string) => {
      const ids =
        role === "user"
          ? voiceUserEntryIdsRef.current
          : voiceAssistantEntryIdsRef.current;
      const existing = ids.get(utteranceId);
      if (existing) return existing;
      const entryId = makeTranscriptEntryId(role);
      ids.set(utteranceId, entryId);
      appendTranscriptEntry({
        id: entryId,
        role,
        text: "",
        timestamp: Date.now(),
      });
      return entryId;
    },
    [appendTranscriptEntry],
  );

  const appendTranscriptText = useCallback((entryId: string, delta: string) => {
    setTranscript((prev) =>
      prev.map((entry) =>
        entry.id === entryId ? { ...entry, text: `${entry.text}${delta}` } : entry,
      ),
    );
  }, []);

  /** Settle an utterance's line on its final text; an empty one is removed. */
  const finishTranscriptEntry = useCallback(
    (role: TranscriptEntry["role"], utteranceId: string, finalText: string) => {
      if (finalText) {
        setTranscriptEntryText(transcriptEntryId(role, utteranceId), finalText);
        return;
      }
      const ids =
        role === "user"
          ? voiceUserEntryIdsRef.current
          : voiceAssistantEntryIdsRef.current;
      const entryId = ids.get(utteranceId);
      if (entryId) {
        removeTranscriptEntry(entryId);
        ids.delete(utteranceId);
      }
    },
    [removeTranscriptEntry, setTranscriptEntryText, transcriptEntryId],
  );

  const handleVoiceRealtimeServerEvent = useCallback(
    (event: VoiceRealtimeServerEvent) => {
      const eventSeq =
        "utterance_seq" in event && typeof event.utterance_seq === "number"
          ? event.utterance_seq
          : null;
      const ignoreAssistantEvent =
        eventSeq != null &&
        eventSeq < voiceMinAcceptedAssistantSeqRef.current &&
        (event.type.startsWith("assistant_") ||
          (event.type === "turn_done" && event.status === "interrupted"));

      if (ignoreAssistantEvent) {
        return;
      }

      const playbackAction = voiceRealtimePlaybackAction(
        event,
        voiceWsPlaybackRef.current?.utteranceSeq ?? null,
      );
      if (playbackAction === "stop") {
        clearAudioPlayback();
        playbackDuckedRef.current = false;
        voiceWsPlaybackRef.current = null;
        processingRef.current = false;
      } else if (playbackAction === "duck" || playbackAction === "restore") {
        setPlaybackDucked(playbackAction === "duck");
      }

      if (
        voiceRealtimeEndsUserSpeech(event, voiceLatestSpeechSeqRef.current) &&
        isSessionActiveRef.current &&
        runtimeStatusRef.current === "user_speaking"
      ) {
        setRuntimeStatus(
          voiceWsPlaybackRef.current?.playbackStarted
            ? "assistant_speaking"
            : "listening",
        );
      }

      switch (event.type) {
        case "connected":
          return;
        case "session_ready":
          voiceWsSessionReadyRef.current = true;
          if (
            isSessionActiveRef.current &&
            runtimeStatusRef.current === "idle"
          ) {
            setRuntimeStatus("listening");
          }
          return;
        case "input_stream_ready":
          voiceWsInputStreamStartedRef.current = true;
          settleVoiceRealtimeInputStreamReady();
          return;
        case "models_ready": {
          const failed = event.models.filter((model) => !model.ready);
          if (failed.length > 0) {
            const message = `Couldn't load ${failed
              .map(
                (model) =>
                  `${formatModelVariantLabel(model.model_id)}: ${model.error ?? "unknown error"}`,
              )
              .join("; ")}`;
            failSession({ kind: "model", message });
            return;
          }
          if (
            isSessionActiveRef.current &&
            runtimeStatusRef.current === "preparing"
          ) {
            setRuntimeStatus("listening");
          }
          return;
        }
        case "input_stream_stopped":
          voiceWsInputStreamStartedRef.current = false;
          settleVoiceRealtimeInputStreamReady(
            new Error("Voice realtime input stream stopped"),
          );
          return;
        case "user_speech_start":
          // The reply keeps playing (ducked) until the server confirms the
          // speech: noise and echo also start speech and get rejected.
          setNotice(null);
          voiceLatestSpeechSeqRef.current = Math.max(
            voiceLatestSpeechSeqRef.current,
            event.utterance_seq,
          );
          if (isSessionActiveRef.current) {
            setRuntimeStatus("user_speaking");
          }
          return;
        case "user_speech_rejected":
          // Playback volume and status were restored above; a rejected
          // speech start never touched the turn in progress.
          return;
        case "user_speech_end":
          // A finished utterance becomes a turn that preempts earlier ones,
          // so their remaining assistant events are stale; its own reply
          // plays at full volume.
          setPlaybackDucked(false);
          voiceMinAcceptedAssistantSeqRef.current = Math.max(
            voiceMinAcceptedAssistantSeqRef.current,
            event.utterance_seq,
          );
          processingRef.current = true;
          if (isSessionActiveRef.current) {
            setRuntimeStatus("processing");
          }
          return;
        case "turn_processing":
          processingRef.current = true;
          if (
            isSessionActiveRef.current &&
            runtimeStatusRef.current !== "user_speaking"
          ) {
            setRuntimeStatus("processing");
          }
          return;
        case "user_transcript_start":
          transcriptEntryId("user", event.utterance_id);
          return;
        case "user_transcript_delta":
          if (event.delta) {
            appendTranscriptText(
              transcriptEntryId("user", event.utterance_id),
              event.delta,
            );
          }
          return;
        case "user_transcript_snapshot":
          setTranscriptEntryText(
            transcriptEntryId("user", event.utterance_id),
            event.text,
          );
          return;
        case "user_transcript_final":
          finishTranscriptEntry(
            "user",
            event.utterance_id,
            (event.text ?? "").trim(),
          );
          return;
        case "assistant_text_start":
          transcriptEntryId("assistant", event.utterance_id);
          return;
        case "assistant_text_delta":
          if (event.delta) {
            appendTranscriptText(
              transcriptEntryId("assistant", event.utterance_id),
              event.delta,
            );
          }
          return;
        case "assistant_text_snapshot":
          setTranscriptEntryText(
            transcriptEntryId("assistant", event.utterance_id),
            event.text,
          );
          return;
        case "assistant_text_final":
          finishTranscriptEntry(
            "assistant",
            event.utterance_id,
            parseFinalAnswer((event.text ?? "").trim()),
          );
          return;
        case "assistant_audio_start": {
          if (event.audio_format !== "pcm_i16") {
            showNotice(
              `Unsupported streamed audio format '${event.audio_format}'. Expected pcm_i16.`,
            );
            return;
          }

          clearAudioPlayback();

          const playbackContext = new AudioContext();
          const playbackGain = playbackContext.createGain();
          playbackGain.gain.value = playbackGainValue();
          playbackGain.connect(playbackContext.destination);
          ttsPlaybackContextRef.current = playbackContext;
          ttsPlaybackGainRef.current = playbackGain;
          ttsNextPlaybackTimeRef.current = playbackContext.currentTime + 0.05;
          ttsSampleRateRef.current = event.sample_rate || 24000;

          const streamSession = ++ttsStreamSessionRef.current;
          voiceWsPlaybackRef.current = {
            utteranceId: event.utterance_id,
            utteranceSeq: event.utterance_seq,
            streamSession,
            streamDone: false,
            playbackStarted: false,
          };
          return;
        }
        case "assistant_audio_done": {
          const playback = voiceWsPlaybackRef.current;
          if (!playback) {
            if (
              isSessionActiveRef.current &&
              runtimeStatusRef.current !== "user_speaking"
            ) {
              setRuntimeStatus("listening");
            }
            processingRef.current = false;
            return;
          }
          if (
            playback.utteranceId !== event.utterance_id ||
            playback.utteranceSeq !== event.utterance_seq
          ) {
            return;
          }
          playback.streamDone = true;
          finalizeVoiceWsPlaybackIfComplete(
            playback.utteranceSeq,
            playback.streamSession,
          );
          return;
        }
        case "turn_interrupted":
          processingRef.current = false;
          if (
            isSessionActiveRef.current &&
            runtimeStatusRef.current !== "user_speaking"
          ) {
            setRuntimeStatus("listening");
          }
          return;
        case "turn_done": {
          if (event.status !== "ok" && event.status !== "interrupted") {
            processingRef.current = false;
          }

          if (event.status === "interrupted") {
            return;
          }

          if (!voiceWsPlaybackRef.current && isSessionActiveRef.current) {
            if (runtimeStatusRef.current !== "user_speaking") {
              setRuntimeStatus("listening");
            }
            processingRef.current = false;
          }
          return;
        }
        case "error": {
          const message = event.message || "Voice realtime error";
          if (!voiceWsInputStreamStartedRef.current) {
            settleVoiceRealtimeInputStreamReady(new Error(message));
          }
          showNotice(message);
          processingRef.current = false;
          if (
            isSessionActiveRef.current &&
            runtimeStatusRef.current !== "user_speaking" &&
            runtimeStatusRef.current !== "assistant_speaking"
          ) {
            setRuntimeStatus("listening");
          }
          return;
        }
        case "pong":
          return;
      }
    },
    [
      appendTranscriptText,
      clearAudioPlayback,
      finalizeVoiceWsPlaybackIfComplete,
      failSession,
      finishTranscriptEntry,
      playbackGainValue,
      setPlaybackDucked,
      settleVoiceRealtimeInputStreamReady,
      setTranscriptEntryText,
      showNotice,
      transcriptEntryId,
    ],
  );

  const ensureVoiceRealtimeSocket =
    useCallback(async (): Promise<WebSocket> => {
      const existing = voiceWsRef.current;
      if (existing && existing.readyState === WebSocket.OPEN) {
        return existing;
      }
      if (voiceWsConnectingRef.current) {
        return voiceWsConnectingRef.current;
      }

      const url = buildVoiceRealtimeWebSocketUrl(api.baseUrl);
      const promise = new Promise<WebSocket>((resolve, reject) => {
        let settled = false;
        const ws = new WebSocket(url);
        ws.binaryType = "arraybuffer";

        const settle = (fn: () => void) => {
          if (settled) return;
          settled = true;
          fn();
        };

        ws.onopen = () => {
          voiceWsRef.current = ws;
          voiceWsSessionReadyRef.current = false;
          voiceWsInputStreamStartedRef.current = false;
          voiceWsInputFrameSeqRef.current = 0;
          voiceWsV2EventRef.current = null;
          try {
            ws.send(
              JSON.stringify(
                buildVoiceRealtimeV2SessionStart(systemPrompt),
              ),
            );
          } catch (error) {
            settle(() => {
              voiceWsConnectingRef.current = null;
              reject(
                error instanceof Error
                  ? error
                  : new Error("Failed to initialize voice realtime websocket"),
              );
            });
            return;
          }
          settle(() => resolve(ws));
        };

        ws.onmessage = (messageEvent) => {
          if (messageEvent.data instanceof ArrayBuffer) {
            const chunk = parseVoiceRealtimeAssistantAudioBinaryChunk(
              messageEvent.data,
            );
            if (chunk) {
              handleVoiceRealtimeAssistantAudioBinaryChunk(chunk);
            }
            return;
          }
          if (messageEvent.data instanceof Blob) {
            void messageEvent.data.arrayBuffer().then((buffer) => {
              const chunk = parseVoiceRealtimeAssistantAudioBinaryChunk(buffer);
              if (chunk) {
                handleVoiceRealtimeAssistantAudioBinaryChunk(chunk);
              }
            });
            return;
          }
          if (typeof messageEvent.data !== "string") return;
          try {
            const parsed: unknown = JSON.parse(messageEvent.data);
            if (isVoiceRealtimeV2EnvelopeHeader(parsed)) {
              if (
                !isValidVoiceRealtimeV2Successor(
                  voiceWsV2EventRef.current,
                  parsed,
                )
              ) {
                failSession({
                  kind: "connection",
                  message: "The voice connection got out of step.",
                });
                try {
                  ws.close(1002, "invalid_event_sequence");
                } catch {
                  // Best effort protocol close.
                }
                return;
              }
              voiceWsV2EventRef.current = parsed;
              // An event type this client does not know still counts
              // towards the sequence above.
              if (!isVoiceRealtimeV2ServerEnvelope(parsed)) {
                return;
              }
              const normalized = normalizeVoiceRealtimeV2Event(parsed);
              if (normalized) {
                handleVoiceRealtimeServerEvent(normalized);
              }
              return;
            }
            if (!isVoiceRealtimeServerEvent(parsed)) {
              return;
            }
            handleVoiceRealtimeServerEvent(parsed);
          } catch {
            // Ignore malformed events.
          }
        };

        ws.onerror = () => {
          const message = "Voice realtime websocket error";
          if (!settled) {
            settle(() => {
              voiceWsConnectingRef.current = null;
              reject(new Error(message));
            });
          }
        };

        ws.onclose = () => {
          if (!settled) {
            settle(() => {
              voiceWsConnectingRef.current = null;
              reject(
                new Error("Voice realtime connection closed during setup"),
              );
            });
          }
          const wasActive = isSessionActiveRef.current;
          const wasCurrent = voiceWsRef.current === ws;
          if (wasCurrent) {
            voiceWsRef.current = null;
          }
          voiceWsSessionReadyRef.current = false;
          voiceWsInputStreamStartedRef.current = false;
          voiceWsInputFrameSeqRef.current = 0;
          voiceWsV2EventRef.current = null;
          voiceWsInputStreamStartingRef.current = null;
          voiceWsConnectingRef.current = null;
          settleVoiceRealtimeInputStreamReady(
            new Error("Voice realtime connection closed"),
          );
          if (wasActive && wasCurrent) {
            // The microphone goes off with the conversation.
            failSession({
              kind: "connection",
              message: "The connection to the voice server dropped.",
            });
          }
        };
      });

      voiceWsConnectingRef.current = promise;
      try {
        return await promise;
      } finally {
        if (voiceWsConnectingRef.current === promise) {
          voiceWsConnectingRef.current = null;
        }
      }
    }, [
      failSession,
      handleVoiceRealtimeAssistantAudioBinaryChunk,
      handleVoiceRealtimeServerEvent,
      settleVoiceRealtimeInputStreamReady,
      systemPrompt,
    ]);

  const ensureVoiceRealtimeInputStreamStarted = useCallback(
    async (inputSampleRate: number) => {
      if (voiceWsInputStreamStartedRef.current) {
        return;
      }
      if (voiceWsInputStreamStartingRef.current) {
        return voiceWsInputStreamStartingRef.current;
      }

      const startPromise = (async () => {
        const socket = await ensureVoiceRealtimeSocket();
        if (socket.readyState !== WebSocket.OPEN) {
          throw new Error("Voice realtime websocket is not connected");
        }
        if (!engine) {
          throw new Error("No voice engine is set up.");
        }
        const readyPromise = waitForVoiceRealtimeInputStreamReady();
        const stream = {
          speaker,
          max_output_tokens: 1536,
          max_utterance_ms: 20_000,
          pre_roll_ms: 160,
          input_sample_rate: Math.round(inputSampleRate),
        };
        sendVoiceRealtimeJson(
          engine.mode === "unified"
            ? {
                type: "input_stream_start",
                mode: "unified",
                s2s_model_id: engine.s2s,
                ...stream,
              }
            : {
                type: "input_stream_start",
                mode: "modular",
                asr_model_id: engine.asr,
                text_model_id: engine.text,
                tts_model_id: engine.tts,
                asr_language: "Auto",
                ...stream,
              },
        );
        await readyPromise;
      })();

      voiceWsInputStreamStartingRef.current = startPromise.finally(() => {
        if (voiceWsInputStreamStartingRef.current === startPromise) {
          voiceWsInputStreamStartingRef.current = null;
        }
      });

      return voiceWsInputStreamStartingRef.current;
    },
    [
      engine,
      ensureVoiceRealtimeSocket,
      sendVoiceRealtimeJson,
      speaker,
      waitForVoiceRealtimeInputStreamReady,
    ],
  );


  const start = useCallback(async () => {
    if (!engine || isSessionActiveRef.current) {
      return;
    }

    setFailure(null);
    setNotice(null);
    setTranscript([]);
    voiceUserEntryIdsRef.current.clear();
    voiceAssistantEntryIdsRef.current.clear();
    isMicMutedRef.current = false;
    setIsMicMuted(false);
    isSessionActiveRef.current = true;
    setRuntimeStatus("preparing");

    let stream: MediaStream;
    try {
      setIsAwaitingMicrophone(true);
      stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
        },
      });
    } catch (err) {
      const name = err instanceof DOMException ? err.name : "";
      failSession(
        MICROPHONE_ERRORS.has(name)
          ? {
              kind: "microphone",
              message:
                name === "NotFoundError"
                  ? "No microphone was found. Connect one, then try again."
                  : "Allow Izwi to use the microphone in System Settings › Privacy & Security › Microphone, then try again.",
            }
          : {
              kind: "other",
              message:
                err instanceof Error
                  ? err.message
                  : "The microphone could not start.",
            },
      );
      return;
    } finally {
      setIsAwaitingMicrophone(false);
    }
    if (!isSessionActiveRef.current) {
      // Ended while the permission prompt was open.
      stream.getTracks().forEach((track) => track.stop());
      return;
    }

    try {
      streamRef.current = stream;

      const audioContext = new AudioContext();
      audioContextRef.current = audioContext;
      const source = audioContext.createMediaStreamSource(stream);
      const analyser = audioContext.createAnalyser();
      analyser.fftSize = 2048;
      analyser.smoothingTimeConstant = 0.2;
      source.connect(analyser);
      analyserRef.current = analyser;
      const processor = audioContext.createScriptProcessor(2048, 1, 1);
      const sink = audioContext.createGain();
      sink.gain.value = 0;

      processor.onaudioprocess = (event) => {
        if (!isSessionActiveRef.current) return;
        if (!voiceWsInputStreamStartedRef.current) return;

        const inputBuffer = event.inputBuffer;
        const channelCount = inputBuffer.numberOfChannels;
        const frameCount = inputBuffer.length;
        if (frameCount <= 0 || channelCount <= 0) return;

        // A muted microphone sends silence, so the server ends the speech
        // in progress instead of waiting for more.
        const mono = new Float32Array(frameCount);
        if (!isMicMutedRef.current) {
          for (let ch = 0; ch < channelCount; ch += 1) {
            const channel = inputBuffer.getChannelData(ch);
            for (let i = 0; i < frameCount; i += 1) {
              mono[i] += (channel[i] ?? 0) / channelCount;
            }
          }
        }

        const pcm16 = encodeLiveMicPcm16(mono);
        const nextSeq = (voiceWsInputFrameSeqRef.current + 1) >>> 0;
        voiceWsInputFrameSeqRef.current = nextSeq;

        try {
          sendVoiceRealtimeBinary(
            encodeVoiceRealtimeClientPcm16Frame(
              pcm16,
              Math.round(inputBuffer.sampleRate),
              nextSeq,
            ),
          );
        } catch {
          // Best-effort; the socket lifecycle reports a lost connection.
        }
      };

      source.connect(processor);
      processor.connect(sink);
      sink.connect(audioContext.destination);
      streamingProcessorRef.current = processor;
      streamingProcessorSinkRef.current = sink;

      processingRef.current = false;
      setStartedAt(Date.now());

      void ensureVoiceRealtimeInputStreamStarted(audioContext.sampleRate).catch(
        (err) => {
          if (!isSessionActiveRef.current) {
            return;
          }
          failSession({
            kind: "connection",
            message:
              err instanceof Error
                ? err.message
                : "Couldn't connect to the voice server.",
          });
        },
      );

      const LEVEL_INTERVAL_MS = 80;
      vadTimerRef.current = window.setInterval(() => {
        const analyserNode = analyserRef.current;
        if (!analyserNode || !isSessionActiveRef.current) return;
        if (isMicMutedRef.current) {
          setAudioLevel(0);
          return;
        }

        const data = new Uint8Array(analyserNode.fftSize);
        analyserNode.getByteTimeDomainData(data);

        let sumSquares = 0;
        for (let i = 0; i < data.length; i += 1) {
          const centered = (data[i] - 128) / 128;
          sumSquares += centered * centered;
        }
        setAudioLevel(Math.sqrt(sumSquares / data.length));
      }, LEVEL_INTERVAL_MS);
    } catch (err) {
      failSession({
        kind: "other",
        message:
          err instanceof Error ? err.message : "The conversation could not start.",
      });
    }
  }, [
    engine,
    ensureVoiceRealtimeInputStreamStarted,
    failSession,
    sendVoiceRealtimeBinary,
  ]);

  const stop = useCallback(() => {
    stopSession();
    setStartedAt(null);
  }, [stopSession]);

  const toggleMicMuted = useCallback(() => {
    isMicMutedRef.current = !isMicMutedRef.current;
    setIsMicMuted(isMicMutedRef.current);
  }, []);

  const dismissFailure = useCallback(() => setFailure(null), []);

  return {
    status: runtimeStatus,
    isActive: runtimeStatus !== "idle",
    isAwaitingMicrophone,
    transcript,
    audioLevel,
    failure,
    notice,
    isMicMuted,
    startedAt,
    start,
    stop,
    toggleMicMuted,
    dismissFailure,
  };
}
