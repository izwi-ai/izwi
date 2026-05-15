function encodeWavPcm16(samples: Float32Array, sampleRate: number): Blob {
  const bytesPerSample = 2;
  const blockAlign = bytesPerSample;
  const byteRate = sampleRate * blockAlign;
  const dataSize = samples.length * bytesPerSample;
  const buffer = new ArrayBuffer(44 + dataSize);
  const view = new DataView(buffer);

  const writeString = (offset: number, value: string) => {
    for (let i = 0; i < value.length; i += 1) {
      view.setUint8(offset + i, value.charCodeAt(i));
    }
  };

  writeString(0, "RIFF");
  view.setUint32(4, 36 + dataSize, true);
  writeString(8, "WAVE");
  writeString(12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, byteRate, true);
  view.setUint16(32, blockAlign, true);
  view.setUint16(34, 16, true);
  writeString(36, "data");
  view.setUint32(40, dataSize, true);

  let offset = 44;
  for (let i = 0; i < samples.length; i += 1) {
    const clamped = Math.max(-1, Math.min(1, samples[i]));
    const int16 = clamped < 0 ? clamped * 0x8000 : clamped * 0x7fff;
    view.setInt16(offset, int16, true);
    offset += 2;
  }

  return new Blob([buffer], { type: "audio/wav" });
}

function normalizedMimeType(mimeType: string | null | undefined): string {
  return mimeType?.trim().toLowerCase() ?? "";
}

export function isWavMimeType(mimeType: string | null | undefined): boolean {
  const normalized = normalizedMimeType(mimeType);
  return (
    normalized === "audio/wav" ||
    normalized === "audio/x-wav" ||
    normalized === "audio/wave" ||
    normalized === "audio/vnd.wave"
  );
}

function hasSpecificMimeType(mimeType: string | null | undefined): boolean {
  const normalized = normalizedMimeType(mimeType);
  return normalized.length > 0 && normalized !== "application/octet-stream";
}

function isBackendSupportedSourceMimeType(
  mimeType: string | null | undefined,
): boolean {
  const normalized = normalizedMimeType(mimeType);
  if (!normalized) {
    return false;
  }

  return (
    isWavMimeType(normalized) ||
    normalized.includes("mpeg") ||
    normalized.includes("mp3") ||
    normalized.includes("mp4") ||
    normalized.includes("m4a") ||
    normalized.includes("webm") ||
    normalized.includes("ogg") ||
    normalized.includes("opus") ||
    normalized.includes("flac") ||
    normalized.includes("aac")
  );
}

function extensionForAudioMimeType(
  mimeType: string | null | undefined,
): string | null {
  const normalized = normalizedMimeType(mimeType);
  if (!normalized) {
    return null;
  }
  if (isWavMimeType(normalized)) {
    return "wav";
  }
  if (normalized.includes("webm")) {
    return "webm";
  }
  if (normalized.includes("ogg") || normalized.includes("opus")) {
    return "ogg";
  }
  if (normalized.includes("flac")) {
    return "flac";
  }
  if (normalized.includes("aac")) {
    return "aac";
  }
  if (normalized.includes("mp4") || normalized.includes("m4a")) {
    return "m4a";
  }
  if (normalized.includes("mpeg") || normalized.includes("mp3")) {
    return "mp3";
  }

  return null;
}

export function resolveSourceAudioFilename(inputBlob: Blob): string | undefined {
  if (typeof File === "undefined" || !(inputBlob instanceof File)) {
    return undefined;
  }

  const trimmed = inputBlob.name.trim();
  return trimmed ? trimmed : undefined;
}

function fallbackAudioFilename(inputBlob: Blob): string {
  const extension = extensionForAudioMimeType(inputBlob.type);
  return extension ? `audio.${extension}` : "audio.bin";
}

function replaceAudioFilenameExtension(
  filename: string,
  nextExtension: string,
): string {
  const trimmed = filename.trim();
  if (!trimmed) {
    return `audio.${nextExtension}`;
  }

  const normalized = trimmed.split(/[/\\]/).pop() ?? trimmed;
  const extensionIndex = normalized.lastIndexOf(".");
  if (extensionIndex <= 0) {
    return `${normalized}.${nextExtension}`;
  }
  return `${normalized.slice(0, extensionIndex)}.${nextExtension}`;
}

export function resolveSpeechTextUploadFilename(options: {
  sourceFileName?: string;
  sourceBlob: Blob;
  uploadedBlob: Blob;
}): string {
  const { sourceFileName, sourceBlob, uploadedBlob } = options;
  const trimmedSourceFileName = sourceFileName?.trim();
  if (!trimmedSourceFileName) {
    return fallbackAudioFilename(uploadedBlob);
  }

  if (uploadedBlob === sourceBlob) {
    return trimmedSourceFileName;
  }

  const uploadedExtension = extensionForAudioMimeType(uploadedBlob.type) ?? "wav";
  return replaceAudioFilenameExtension(trimmedSourceFileName, uploadedExtension);
}

function isFileBlob(inputBlob: Blob): inputBlob is File {
  return typeof File !== "undefined" && inputBlob instanceof File;
}

export function shouldPreserveSpeechTextUploadBlob(inputBlob: Blob): boolean {
  return isFileBlob(inputBlob) || isBackendSupportedSourceMimeType(inputBlob.type);
}

export async function prepareSpeechTextUploadBlob(
  inputBlob: Blob,
  targetSampleRate = 16000,
): Promise<Blob> {
  if (shouldPreserveSpeechTextUploadBlob(inputBlob)) {
    return inputBlob;
  }

  const sourceFileName = resolveSourceAudioFilename(inputBlob);
  return transcodeToWav(inputBlob, targetSampleRate, sourceFileName).catch(
    () => inputBlob,
  );
}

export async function transcodeToWav(
  inputBlob: Blob,
  targetSampleRate = 16000,
  sourceFileName?: string,
): Promise<Blob> {
  const filenameLooksWav = sourceFileName
    ? sourceFileName.toLowerCase().endsWith(".wav")
    : false;
  if (
    isWavMimeType(inputBlob.type) ||
    (!hasSpecificMimeType(inputBlob.type) && filenameLooksWav)
  ) {
    return inputBlob;
  }

  const decodeContext = new AudioContext();
  try {
    const sourceBytes = await inputBlob.arrayBuffer();
    const decoded = await decodeContext.decodeAudioData(sourceBytes.slice(0));

    const monoBuffer = decodeContext.createBuffer(
      1,
      decoded.length,
      decoded.sampleRate,
    );
    const mono = monoBuffer.getChannelData(0);

    for (let i = 0; i < decoded.length; i += 1) {
      let sum = 0;
      for (let ch = 0; ch < decoded.numberOfChannels; ch += 1) {
        sum += decoded.getChannelData(ch)[i] ?? 0;
      }
      mono[i] = sum / decoded.numberOfChannels;
    }

    const rendered = await (() => {
      if (decoded.sampleRate === targetSampleRate) {
        return Promise.resolve(monoBuffer);
      }

      const targetLength = Math.ceil(
        (monoBuffer.length * targetSampleRate) / monoBuffer.sampleRate,
      );
      const offline = new OfflineAudioContext(1, targetLength, targetSampleRate);
      const source = offline.createBufferSource();
      source.buffer = monoBuffer;
      source.connect(offline.destination);
      source.start(0);
      return offline.startRendering();
    })();

    return encodeWavPcm16(rendered.getChannelData(0), targetSampleRate);
  } finally {
    decodeContext.close().catch(() => {});
  }
}
