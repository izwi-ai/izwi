import { encodeWavPcm16 } from "@/features/voice/realtime/support";

/** Shorter clips give the model too little of the voice to copy. */
export const MIN_REFERENCE_SECS = 3;
/** The server's limit, SpeechModelCapabilities::MAX_REFERENCE_AUDIO_SECS. */
export const MAX_REFERENCE_SECS = 30;

const PEAK_COUNT = 48;

export interface ReferenceClip {
  /** Mono 16-bit WAV, which every reference-capable model reads. */
  blob: Blob;
  durationSecs: number;
  /** Loudest sample in each of PEAK_COUNT slices, 0–1, for a small waveform. */
  peaks: number[];
}

function measurePeaks(samples: Float32Array): number[] {
  if (samples.length === 0) {
    return [];
  }
  const sliceLength = Math.max(1, Math.floor(samples.length / PEAK_COUNT));
  const peaks: number[] = [];
  for (let start = 0; start < samples.length && peaks.length < PEAK_COUNT; start += sliceLength) {
    let peak = 0;
    const end = Math.min(samples.length, start + sliceLength);
    for (let i = start; i < end; i += 1) {
      peak = Math.max(peak, Math.abs(samples[i]));
    }
    peaks.push(peak);
  }
  const loudest = Math.max(...peaks);
  return loudest > 0 ? peaks.map((peak) => peak / loudest) : peaks;
}

/**
 * Decodes a file or microphone recording to mono WAV and measures it.
 * `maxSecs` cuts the audio to that length (a recording stopped at the limit
 * can run a few milliseconds over it). `sampleRate` resamples the audio to
 * that rate; without it the clip keeps the device's rate.
 */
export async function prepareReferenceClip(
  source: Blob,
  { maxSecs, sampleRate }: { maxSecs?: number; sampleRate?: number } = {},
): Promise<ReferenceClip> {
  // Decoding resamples to the context's rate.
  const context =
    sampleRate === undefined
      ? new AudioContext()
      : new AudioContext({ sampleRate });
  try {
    const decoded = await context.decodeAudioData(await source.arrayBuffer());
    const frameCount =
      maxSecs === undefined
        ? decoded.length
        : Math.min(decoded.length, Math.floor(maxSecs * decoded.sampleRate));
    const mono = new Float32Array(frameCount);
    for (let channel = 0; channel < decoded.numberOfChannels; channel += 1) {
      const data = decoded.getChannelData(channel);
      for (let i = 0; i < frameCount; i += 1) {
        mono[i] += data[i] / decoded.numberOfChannels;
      }
    }
    return {
      blob: encodeWavPcm16(mono, decoded.sampleRate),
      durationSecs: frameCount / decoded.sampleRate,
      peaks: measurePeaks(mono),
    };
  } finally {
    void context.close().catch(() => {});
  }
}
