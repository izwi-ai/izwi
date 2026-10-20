import { act, renderHook } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import type { SpeechHistoryRecord, SpeechHistoryRecordStreamCallbacks } from "@/api";
import { useSpeechStreamPlayback } from "./useSpeechStreamPlayback";

const mocks = vi.hoisted(() => ({ create: vi.fn(), attach: vi.fn(), cancel: vi.fn(), unlock: vi.fn(),
  start: vi.fn(), push: vi.fn(), finish: vi.fn(), stop: vi.fn() }));
vi.mock("@/api", () => ({ api: {
  createTextToSpeechRecordStream: mocks.create, attachTextToSpeechRecordStream: mocks.attach, cancelTextToSpeechRecord: mocks.cancel,
} }));
vi.mock("../pcmPlayer", () => ({ SpeechPcmPlayer: class {
  unlock = mocks.unlock; start = mocks.start; push = mocks.push;
  finish = mocks.finish; stop = mocks.stop;
} }));

const record = { id: "record", processing_status: "processing" } as SpeechHistoryRecord;
let events: SpeechHistoryRecordStreamCallbacks;
let controller: AbortController;

beforeEach(() => {
  vi.clearAllMocks();
  vi.stubGlobal("AudioContext", class {});
  mocks.unlock.mockResolvedValue(undefined);
  mocks.push.mockResolvedValue(undefined);
  mocks.cancel.mockResolvedValue({ record });
  mocks.create.mockImplementation((_request, callbacks) => {
    events = callbacks;
    controller = new AbortController();
    return controller;
  });
});

async function begin(result: { current: ReturnType<typeof useSpeechStreamPlayback> }) {
  let created!: Promise<SpeechHistoryRecord>;
  act(() => { created = result.current.create({ text: "Hello", model_id: "FishAudio-S2-Pro" }); });
  await act(async () => {
    events.onCreated?.(record);
    await created;
  });
}

describe("route-owned speech stream", () => {
  it("unlocks on generate and plays chunks after the dialog's created promise resolves", async () => {
    const { result, rerender } = renderHook(({ id }: { id?: string }) => useSpeechStreamPlayback(id, {}), {
      initialProps: { id: undefined as string | undefined },
    });
    await begin(result);
    expect(mocks.unlock.mock.invocationCallOrder[0]).toBeLessThan(mocks.create.mock.invocationCallOrder[0]);
    rerender({ id: "record" });
    expect(controller.signal.aborted).toBe(false);
    await act(async () => {
      events.onStart?.({ requestId: "request", sampleRate: 44100, audioFormat: "pcm_i16" });
      await events.onChunk?.({ requestId: "request", sequence: 0, sampleCount: 1, audioBase64: "AAA=" });
    });
    expect(mocks.push).toHaveBeenCalledOnce();
    expect(result.current.status).toBe("playing");
    expect(mocks.finish).not.toHaveBeenCalled();
  });

  it("unlocks replay on the listen gesture before attaching to an existing job", async () => {
    mocks.attach.mockImplementation((_id, callbacks) => {
      events = callbacks;
      controller = new AbortController();
      callbacks.onDurable?.();
      callbacks.onCreated?.(record);
      return controller;
    });
    const { result } = renderHook(() => useSpeechStreamPlayback("record", {}));
    await act(async () => { await result.current.listen(record); });
    expect(mocks.create).not.toHaveBeenCalled();
    expect(mocks.attach).toHaveBeenCalledWith("record", expect.any(Object));
    expect(mocks.unlock.mock.invocationCallOrder[0]).toBeLessThan(mocks.attach.mock.invocationCallOrder[0]);
    act(() => result.current.stop());
    expect(mocks.cancel).toHaveBeenCalledWith("record");
  });

  it("keeps final queued audio playing and does not cancel completed history on unmount", async () => {
    const onFinal = vi.fn();
    const { result, unmount } = renderHook(() => useSpeechStreamPlayback(undefined, { onFinal }));
    await begin(result);
    act(() => {
      events.onFinal?.({ record, stats: { generation_time_ms: 1, audio_duration_secs: 1, rtf: 0.1, tokens_generated: 1 } });
      events.onDone?.();
    });
    expect(mocks.finish).toHaveBeenCalledOnce();
    expect(mocks.stop).not.toHaveBeenCalled();
    expect(onFinal).toHaveBeenCalledOnce();
    unmount();
    expect(mocks.stop).toHaveBeenCalledOnce();
    expect(mocks.cancel).not.toHaveBeenCalled();
  });

  it("detaches on navigation: playback stops, the take goes on, late chunks are ignored", async () => {
    const { result, rerender } = renderHook(({ id }: { id?: string }) => useSpeechStreamPlayback(id, {}), {
      initialProps: { id: undefined as string | undefined },
    });
    await begin(result);
    rerender({ id: "record" });
    rerender({ id: "another-record" });
    expect(controller.signal.aborted).toBe(true);
    // The server finishes the take into history; only Stop cancels it.
    expect(mocks.cancel).not.toHaveBeenCalled();
    await act(async () => { await events.onChunk?.({ requestId: "request", sequence: 0, sampleCount: 1, audioBase64: "AAA=" }); });
    expect(mocks.push).not.toHaveBeenCalled();
  });

  it("detaches durable playback on navigation without cancelling the job", async () => {
    const { result, rerender } = renderHook(({ id }: { id?: string }) => useSpeechStreamPlayback(id, {}), {
      initialProps: { id: undefined as string | undefined },
    });
    await begin(result);
    act(() => events.onDurable?.());
    rerender({ id: "record" });
    rerender({ id: "another" });
    expect(controller.signal.aborted).toBe(true);
    expect(mocks.cancel).not.toHaveBeenCalled();
  });

  it("explicit stop cancels durable generation and exposes segment progress", async () => {
    const { result } = renderHook(() => useSpeechStreamPlayback(undefined, {}));
    await begin(result);
    act(() => {
      events.onDurable?.();
      events.onProgress?.({ completedSegments: 2, totalSegments: 10, processedTextBytes: 400 });
    });
    expect(result.current.progress?.completedSegments).toBe(2);
    act(() => result.current.stop());
    expect(mocks.cancel).toHaveBeenCalledWith("record");
  });

  it("a durable playback failure leaves generation running", async () => {
    const { result } = renderHook(() => useSpeechStreamPlayback(undefined, {}));
    await begin(result);
    act(() => { events.onDurable?.(); events.onError?.("Network disconnected"); });
    expect(result.current.error).toBe("Network disconnected");
    expect(controller.signal.aborted).toBe(true);
    expect(mocks.cancel).not.toHaveBeenCalled();
  });

  it("cancels navigation while waiting for the created event and ignores its late response", async () => {
    const { result, rerender } = renderHook(({ id }: { id?: string }) => useSpeechStreamPlayback(id, {}), {
      initialProps: { id: undefined as string | undefined },
    });
    let creation!: Promise<SpeechHistoryRecord>;
    act(() => { creation = result.current.create({ text: "Hello", model_id: "FishAudio-S2-Pro" }); });
    const rejected = expect(creation).rejects.toThrow("stopped");
    rerender({ id: "other-record" });
    await rejected;
    expect(controller.signal.aborted).toBe(true);
    act(() => { events.onCreated?.(record); });
    expect(result.current.status).toBe("idle");
  });

  it("reports truncated streams after creation and stops audio instead of claiming completion", async () => {
    const onError = vi.fn();
    const { result } = renderHook(() => useSpeechStreamPlayback(undefined, { onError }));
    await begin(result);
    act(() => { events.onDone?.(); });
    expect(result.current.error).toMatch(/before generation completed/);
    expect(result.current.status).toBe("idle");
    expect(onError).toHaveBeenCalledOnce();
    expect(mocks.stop).toHaveBeenCalledOnce();
    expect(controller.signal.aborted).toBe(true);
  });
});
