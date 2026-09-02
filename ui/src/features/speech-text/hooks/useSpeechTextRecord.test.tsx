import { act, renderHook, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { useSpeechTextRecord } from "./useSpeechTextRecord";

interface TestRecord {
  id: string;
  processing_status: "pending" | "processing" | "ready" | "failed";
  revision: number;
}

function buildRecord(
  id: string,
  revision: number,
  processingStatus: TestRecord["processing_status"] = "ready",
): TestRecord {
  return {
    id,
    processing_status: processingStatus,
    revision,
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

afterEach(() => {
  vi.useRealTimers();
});

describe("useSpeechTextRecord", () => {
  it("ignores a late response for the previous record ID", async () => {
    const recordA = deferredPromise<TestRecord>();
    const recordB = deferredPromise<TestRecord>();
    const getRecord = vi.fn((recordId: string) =>
      recordId === "record-a" ? recordA.promise : recordB.promise,
    );

    const { result, rerender } = renderHook(
      ({ recordId }) =>
        useSpeechTextRecord({
          recordId,
          getRecord,
          loadErrorMessage: "Could not load record.",
          enablePolling: false,
        }),
      { initialProps: { recordId: "record-a" } },
    );

    await waitFor(() => expect(getRecord).toHaveBeenCalledWith("record-a"));
    rerender({ recordId: "record-b" });
    await waitFor(() => expect(getRecord).toHaveBeenCalledWith("record-b"));

    await act(async () => {
      recordB.resolve(buildRecord("record-b", 1));
      await recordB.promise;
    });
    expect(result.current.record).toEqual(buildRecord("record-b", 1));

    await act(async () => {
      recordA.resolve(buildRecord("record-a", 1));
      await recordA.promise;
    });
    expect(result.current.record).toEqual(buildRecord("record-b", 1));
    expect(result.current.loading).toBe(false);
    expect(result.current.error).toBeNull();
  });

  it("ignores a late failure for the previous record ID", async () => {
    const recordA = deferredPromise<TestRecord>();
    const getRecord = vi.fn((recordId: string) =>
      recordId === "record-a"
        ? recordA.promise
        : Promise.resolve(buildRecord("record-b", 1)),
    );

    const { result, rerender } = renderHook(
      ({ recordId }) =>
        useSpeechTextRecord({
          recordId,
          getRecord,
          loadErrorMessage: "Could not load record.",
          enablePolling: false,
        }),
      { initialProps: { recordId: "record-a" } },
    );

    await waitFor(() => expect(getRecord).toHaveBeenCalledWith("record-a"));
    rerender({ recordId: "record-b" });
    await waitFor(() => expect(result.current.record?.id).toBe("record-b"));

    await act(async () => {
      recordA.reject(new Error("record A failed late"));
      await expect(recordA.promise).rejects.toThrow("record A failed late");
    });

    expect(result.current.record?.id).toBe("record-b");
    expect(result.current.error).toBeNull();
  });

  it("keeps the newest overlapping polling result", async () => {
    vi.useFakeTimers();
    const olderPoll = deferredPromise<TestRecord>();
    const newerPoll = deferredPromise<TestRecord>();
    const getRecord = vi
      .fn<(recordId: string) => Promise<TestRecord>>()
      .mockResolvedValueOnce(buildRecord("record-a", 1, "processing"))
      .mockImplementationOnce(() => olderPoll.promise)
      .mockImplementationOnce(() => newerPoll.promise);

    const { result } = renderHook(() =>
      useSpeechTextRecord({
        recordId: "record-a",
        getRecord,
        loadErrorMessage: "Could not load record.",
      }),
    );

    await act(async () => {
      await Promise.resolve();
    });
    expect(result.current.record?.revision).toBe(1);

    await act(async () => {
      vi.advanceTimersByTime(2_500);
      await Promise.resolve();
    });
    expect(getRecord).toHaveBeenCalledTimes(2);

    await act(async () => {
      vi.advanceTimersByTime(2_500);
      await Promise.resolve();
    });
    expect(getRecord).toHaveBeenCalledTimes(3);

    await act(async () => {
      newerPoll.resolve(buildRecord("record-a", 3, "ready"));
      await newerPoll.promise;
    });
    expect(result.current.record?.revision).toBe(3);

    await act(async () => {
      olderPoll.resolve(buildRecord("record-a", 2, "processing"));
      await olderPoll.promise;
    });
    expect(result.current.record?.revision).toBe(3);
  });

  it("ignores late success and failure after unmount", async () => {
    const lateSuccess = deferredPromise<TestRecord>();
    const successHook = renderHook(() =>
      useSpeechTextRecord({
        recordId: "record-a",
        getRecord: () => lateSuccess.promise,
        loadErrorMessage: "Could not load record.",
        enablePolling: false,
      }),
    );
    const successResult = successHook.result;
    successHook.unmount();

    await act(async () => {
      lateSuccess.resolve(buildRecord("record-a", 1));
      await lateSuccess.promise;
    });
    expect(successResult.current.record).toBeNull();

    const lateFailure = deferredPromise<TestRecord>();
    const failureHook = renderHook(() =>
      useSpeechTextRecord({
        recordId: "record-b",
        getRecord: () => lateFailure.promise,
        loadErrorMessage: "Could not load record.",
        enablePolling: false,
      }),
    );
    const failureResult = failureHook.result;
    failureHook.unmount();

    await act(async () => {
      lateFailure.reject(new Error("late failure"));
      await expect(lateFailure.promise).rejects.toThrow("late failure");
    });
    expect(failureResult.current.error).toBeNull();
  });
});
