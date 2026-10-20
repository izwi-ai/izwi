import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { prepareReferenceClip } from "./referenceClip";

const SAMPLE_RATE = 1000;

// A 40 s stereo clip at 1 kHz, louder towards the end.
function fakeBuffer() {
  const length = 40 * SAMPLE_RATE;
  const left = Float32Array.from({ length }, (_, i) => i / length);
  const right = Float32Array.from({ length }, (_, i) => -i / length);
  return {
    length,
    sampleRate: SAMPLE_RATE,
    numberOfChannels: 2,
    getChannelData: (channel: number) => (channel === 0 ? left : right),
  };
}

const source = { arrayBuffer: async () => new ArrayBuffer(8) } as unknown as Blob;

describe("prepareReferenceClip", () => {
  const close = vi.fn(async () => {});

  beforeEach(() => {
    close.mockClear();
    vi.stubGlobal(
      "AudioContext",
      class {
        decodeAudioData = vi.fn(async () => fakeBuffer());
        close = close;
      },
    );
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("measures the whole clip as mono WAV", async () => {
    const clip = await prepareReferenceClip(source);

    expect(clip.durationSecs).toBe(40);
    expect(clip.blob.type).toBe("audio/wav");
    expect(clip.blob.size).toBe(44 + 40 * SAMPLE_RATE * 2);
    expect(close).toHaveBeenCalled();
  });

  it("cuts the clip to maxSecs", async () => {
    const clip = await prepareReferenceClip(source, { maxSecs: 30 });

    expect(clip.durationSecs).toBe(30);
    expect(clip.blob.size).toBe(44 + 30 * SAMPLE_RATE * 2);
  });

  it("keeps a small, normalised waveform", async () => {
    vi.stubGlobal(
      "AudioContext",
      class {
        decodeAudioData = vi.fn(async () => ({
          ...fakeBuffer(),
          numberOfChannels: 1,
        }));
        close = close;
      },
    );

    const { peaks } = await prepareReferenceClip(source);

    expect(peaks.length).toBeLessThanOrEqual(48);
    expect(Math.max(...peaks)).toBe(1);
    expect(peaks[0]).toBeLessThan(peaks[peaks.length - 1]);
  });
});
