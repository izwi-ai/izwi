import { act, renderHook, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import type { SpeechHistoryRecord } from "@/api";
import { useTextToSpeechRecord } from "./useTextToSpeechRecord";

const apiMocks = vi.hoisted(() => ({
  getTextToSpeechRecord: vi.fn(),
}));

vi.mock("@/api", () => ({
  api: {
    getTextToSpeechRecord: apiMocks.getTextToSpeechRecord,
  },
}));

function buildRecord(
  id: string,
  generationTimeMs: number,
  processingStatus: SpeechHistoryRecord["processing_status"] = "ready",
): SpeechHistoryRecord {
  return {
    id,
    created_at: 1,
    route_kind: "text_to_speech",
    processing_status: processingStatus,
    processing_error: null,
    model_id: "test-model",
    speaker: "Vivian",
    language: null,
    saved_voice_id: null,
    speed: 1,
    input_text: `Text for ${id}`,
    voice_description: null,
    reference_text: null,
    generation_time_ms: generationTimeMs,
    audio_duration_secs: null,
    rtf: null,
    tokens_generated: null,
    audio_mime_type: "audio/wav",
    audio_filename: `${id}.wav`,
  };
}

function deferredPromise<T>() {
  let resolve!: (value: T) => void;
  let reject!: (reason?: unknown) => void;
  const promise = new Promise<T>((nextResolve, nextReject) => {
    resolve = nextResolve;
    reject = nextReject;
  });
  return { promise, resolve, reject };
}

describe("useTextToSpeechRecord", () => {
  beforeEach(() => {
    apiMocks.getTextToSpeechRecord.mockReset();
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  it("ignores a late response for the previous record ID", async () => {
    const recordA = deferredPromise<SpeechHistoryRecord>();
    const recordB = deferredPromise<SpeechHistoryRecord>();
    apiMocks.getTextToSpeechRecord.mockImplementation((recordId: string) =>
      recordId === "tts-a" ? recordA.promise : recordB.promise,
    );

    const { result, rerender } = renderHook(
      ({ recordId }) => useTextToSpeechRecord(recordId),
      { initialProps: { recordId: "tts-a" } },
    );

    await waitFor(() =>
      expect(apiMocks.getTextToSpeechRecord).toHaveBeenCalledWith("tts-a"),
    );
    rerender({ recordId: "tts-b" });
    await waitFor(() =>
      expect(apiMocks.getTextToSpeechRecord).toHaveBeenCalledWith("tts-b"),
    );

    await act(async () => {
      recordB.resolve(buildRecord("tts-b", 20));
      await recordB.promise;
    });
    expect(result.current.record?.id).toBe("tts-b");

    await act(async () => {
      recordA.resolve(buildRecord("tts-a", 10));
      await recordA.promise;
    });
    expect(result.current.record?.id).toBe("tts-b");
    expect(result.current.loading).toBe(false);
    expect(result.current.error).toBeNull();
  });

  it("ignores a late failure for the previous record ID", async () => {
    const recordA = deferredPromise<SpeechHistoryRecord>();
    apiMocks.getTextToSpeechRecord.mockImplementation((recordId: string) =>
      recordId === "tts-a"
        ? recordA.promise
        : Promise.resolve(buildRecord("tts-b", 20)),
    );

    const { result, rerender } = renderHook(
      ({ recordId }) => useTextToSpeechRecord(recordId),
      { initialProps: { recordId: "tts-a" } },
    );

    await waitFor(() =>
      expect(apiMocks.getTextToSpeechRecord).toHaveBeenCalledWith("tts-a"),
    );
    rerender({ recordId: "tts-b" });
    await waitFor(() => expect(result.current.record?.id).toBe("tts-b"));

    await act(async () => {
      recordA.reject(new Error("TTS A failed late"));
      await expect(recordA.promise).rejects.toThrow("TTS A failed late");
    });

    expect(result.current.record?.id).toBe("tts-b");
    expect(result.current.error).toBeNull();
  });

  it("keeps the newest overlapping polling result", async () => {
    vi.useFakeTimers();
    const olderPoll = deferredPromise<SpeechHistoryRecord>();
    const newerPoll = deferredPromise<SpeechHistoryRecord>();
    apiMocks.getTextToSpeechRecord
      .mockResolvedValueOnce(buildRecord("tts-a", 10, "processing"))
      .mockImplementationOnce(() => olderPoll.promise)
      .mockImplementationOnce(() => newerPoll.promise);

    const { result } = renderHook(() => useTextToSpeechRecord("tts-a"));

    await act(async () => {
      await Promise.resolve();
    });
    expect(result.current.record?.generation_time_ms).toBe(10);

    await act(async () => {
      vi.advanceTimersByTime(2_500);
      await Promise.resolve();
    });
    expect(apiMocks.getTextToSpeechRecord).toHaveBeenCalledTimes(2);

    await act(async () => {
      vi.advanceTimersByTime(2_500);
      await Promise.resolve();
    });
    expect(apiMocks.getTextToSpeechRecord).toHaveBeenCalledTimes(3);

    await act(async () => {
      newerPoll.resolve(buildRecord("tts-a", 30));
      await newerPoll.promise;
    });
    expect(result.current.record?.generation_time_ms).toBe(30);

    await act(async () => {
      olderPoll.resolve(buildRecord("tts-a", 20, "processing"));
      await olderPoll.promise;
    });
    expect(result.current.record?.generation_time_ms).toBe(30);
  });

  it("ignores late success and failure after unmount", async () => {
    const lateSuccess = deferredPromise<SpeechHistoryRecord>();
    apiMocks.getTextToSpeechRecord.mockReturnValueOnce(lateSuccess.promise);
    const successHook = renderHook(() => useTextToSpeechRecord("tts-a"));
    const successResult = successHook.result;
    successHook.unmount();

    await act(async () => {
      lateSuccess.resolve(buildRecord("tts-a", 10));
      await lateSuccess.promise;
    });
    expect(successResult.current.record).toBeNull();

    const lateFailure = deferredPromise<SpeechHistoryRecord>();
    apiMocks.getTextToSpeechRecord.mockReturnValueOnce(lateFailure.promise);
    const failureHook = renderHook(() => useTextToSpeechRecord("tts-b"));
    const failureResult = failureHook.result;
    failureHook.unmount();

    await act(async () => {
      lateFailure.reject(new Error("late failure"));
      await expect(lateFailure.promise).rejects.toThrow("late failure");
    });
    expect(failureResult.current.error).toBeNull();
  });
});
