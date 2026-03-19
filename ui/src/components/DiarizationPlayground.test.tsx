import {
  fireEvent,
  render,
  screen,
  waitFor,
  within,
} from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { DiarizationPlayground } from "./DiarizationPlayground";

function activateTab(scope: ReturnType<typeof within>, name: string): void {
  const tab = scope.getByRole("tab", { name });
  fireEvent.mouseDown(tab);
  fireEvent.click(tab);
}

const apiMocks = vi.hoisted(() => ({
  createDiarizationRecord: vi.fn(),
  updateDiarizationRecord: vi.fn(),
  rerunDiarizationRecord: vi.fn(),
  listDiarizationRecords: vi.fn(),
  getDiarizationRecord: vi.fn(),
  deleteDiarizationRecord: vi.fn(),
  diarizationRecordAudioUrl: vi.fn(),
}));

vi.mock("../api", () => ({
  api: {
    createDiarizationRecord: apiMocks.createDiarizationRecord,
    updateDiarizationRecord: apiMocks.updateDiarizationRecord,
    rerunDiarizationRecord: apiMocks.rerunDiarizationRecord,
    listDiarizationRecords: apiMocks.listDiarizationRecords,
    getDiarizationRecord: apiMocks.getDiarizationRecord,
    deleteDiarizationRecord: apiMocks.deleteDiarizationRecord,
    diarizationRecordAudioUrl: apiMocks.diarizationRecordAudioUrl,
  },
}));

describe("DiarizationPlayground speaker corrections", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  beforeEach(() => {
    apiMocks.createDiarizationRecord.mockReset();
    apiMocks.updateDiarizationRecord.mockReset();
    apiMocks.rerunDiarizationRecord.mockReset();
    apiMocks.listDiarizationRecords.mockReset();
    apiMocks.getDiarizationRecord.mockReset();
    apiMocks.deleteDiarizationRecord.mockReset();
    apiMocks.diarizationRecordAudioUrl.mockReset();

    apiMocks.listDiarizationRecords.mockResolvedValue([]);
    apiMocks.diarizationRecordAudioUrl.mockReturnValue("/audio/meeting.wav");
  });

  it("applies saved speaker corrections to the active transcript", async () => {
    apiMocks.createDiarizationRecord.mockResolvedValue({
      id: "diar-1",
      created_at: 1,
      model_id: "diar_streaming_sortformer_4spk-v2.1",
      speaker_count: 2,
      corrected_speaker_count: 2,
      duration_secs: 6.4,
      processing_time_ms: 120,
      rtf: 0.5,
      audio_mime_type: "audio/wav",
      audio_filename: "meeting.wav",
      transcript: "",
      raw_transcript: "",
      speaker_name_overrides: {},
      utterances: [
        {
          speaker: "SPEAKER_00",
          start: 0,
          end: 1,
          text: "Hello there.",
        },
        {
          speaker: "SPEAKER_01",
          start: 1,
          end: 2,
          text: "Hi back.",
        },
      ],
      words: [],
      segments: [],
    });
    apiMocks.updateDiarizationRecord.mockResolvedValue({
      id: "diar-1",
      created_at: 1,
      model_id: "diar_streaming_sortformer_4spk-v2.1",
      speaker_count: 2,
      corrected_speaker_count: 2,
      duration_secs: 6.4,
      processing_time_ms: 120,
      rtf: 0.5,
      audio_mime_type: "audio/wav",
      audio_filename: "meeting.wav",
      transcript: "",
      raw_transcript: "",
      speaker_name_overrides: {
        SPEAKER_00: "Alice",
      },
      utterances: [
        {
          speaker: "SPEAKER_00",
          start: 0,
          end: 1,
          text: "Hello there.",
        },
        {
          speaker: "SPEAKER_01",
          start: 1,
          end: 2,
          text: "Hi back.",
        },
      ],
      words: [],
      segments: [],
    });

    const { container } = render(
      <DiarizationPlayground
        selectedModel="diar_streaming_sortformer_4spk-v2.1"
        selectedModelReady
        onModelRequired={vi.fn()}
        pipelineAsrModelId="Qwen3-ASR-0.6B"
        pipelineAlignerModelId="Qwen3-ForcedAligner-0.6B"
        pipelineModelsReady
      />,
    );
    const scope = within(container);

    expect(container.querySelectorAll('input[type="number"]')).toHaveLength(0);
    expect(screen.getByTestId("diarization-session-panel")).not.toHaveClass(
      "border",
    );
    expect(screen.getByTestId("diarization-settings-surface")).not.toHaveClass(
      "border",
    );
    expect(
      screen.queryByTestId("diarization-reset-rail"),
    ).not.toBeInTheDocument();

    expect(
      screen.queryByRole("heading", { name: "Transcript" }),
    ).not.toBeInTheDocument();

    const fileInput = container.querySelector<HTMLInputElement>(
      'input[type="file"]',
    );
    expect(fileInput).not.toBeNull();

    fireEvent.change(fileInput!, {
      target: {
        files: [new File(["sample"], "meeting.wav", { type: "audio/wav" })],
      },
    });

    expect((await scope.findAllByText("SPEAKER_00")).length).toBeGreaterThan(0);
    expect(
      screen.getAllByRole("heading", { name: "Transcript" }).length,
    ).toBeGreaterThan(0);
    expect(screen.getByTestId("diarization-session-panel")).toHaveClass(
      "border",
    );
    expect(screen.getByTestId("diarization-settings-surface")).toHaveClass(
      "border",
    );
    expect(screen.getByTestId("diarization-reset-rail")).toBeInTheDocument();
    expect(screen.getByTestId("diarization-stats-footer")).toBeInTheDocument();

    activateTab(scope, "Speakers");
    expect(await scope.findByText("Speaker Corrections")).toBeInTheDocument();
    const displayNameInput = scope.getAllByLabelText("Display name")[0];
    fireEvent.change(displayNameInput, {
      target: { value: "Alice" },
    });
    const saveButton = scope.getByRole("button", {
      name: "Save corrections",
    });
    fireEvent.click(saveButton);

    await waitFor(() =>
      expect(apiMocks.updateDiarizationRecord).toHaveBeenCalledWith("diar-1", {
        speaker_name_overrides: {
          SPEAKER_00: "Alice",
        },
      }),
    );
    await apiMocks.updateDiarizationRecord.mock.results[0]?.value;

    activateTab(scope, "Transcript");
    expect((await scope.findAllByText("Alice")).length).toBeGreaterThan(0);
  });

  it("reruns diarization from the quality tab", async () => {
    apiMocks.createDiarizationRecord.mockResolvedValue({
      id: "diar-1",
      created_at: 1,
      model_id: "diar_streaming_sortformer_4spk-v2.1",
      asr_model_id: "Qwen3-ASR-0.6B",
      aligner_model_id: "Qwen3-ForcedAligner-0.6B",
      llm_model_id: "Qwen3-1.7B-GGUF",
      min_speakers: 1,
      max_speakers: 4,
      min_speech_duration_ms: 240,
      min_silence_duration_ms: 200,
      enable_llm_refinement: true,
      speaker_count: 2,
      corrected_speaker_count: 2,
      alignment_coverage: 0.82,
      unattributed_words: 4,
      llm_refined: false,
      duration_secs: 6.4,
      processing_time_ms: 120,
      rtf: 0.5,
      audio_mime_type: "audio/wav",
      audio_filename: "meeting.wav",
      asr_text: "Hello there. Hi back.",
      transcript: "",
      raw_transcript: "",
      speaker_name_overrides: {},
      utterances: [
        {
          speaker: "SPEAKER_00",
          start: 0,
          end: 1,
          text: "Hello there.",
          word_start: 0,
          word_end: 1,
        },
      ],
      words: [],
      segments: [],
    });
    apiMocks.rerunDiarizationRecord.mockResolvedValue({
      id: "diar-2",
      created_at: 2,
      model_id: "diar_streaming_sortformer_4spk-v2.1",
      asr_model_id: "Qwen3-ASR-0.6B",
      aligner_model_id: "Qwen3-ForcedAligner-0.6B",
      llm_model_id: "Qwen3-1.7B-GGUF",
      min_speakers: 2,
      max_speakers: 5,
      min_speech_duration_ms: 180,
      min_silence_duration_ms: 120,
      enable_llm_refinement: false,
      speaker_count: 2,
      corrected_speaker_count: 2,
      alignment_coverage: 0.95,
      unattributed_words: 0,
      llm_refined: false,
      duration_secs: 6.4,
      processing_time_ms: 110,
      rtf: 0.45,
      audio_mime_type: "audio/wav",
      audio_filename: "meeting.wav",
      asr_text: "Hello there. Hi back.",
      transcript: "",
      raw_transcript: "",
      speaker_name_overrides: {},
      utterances: [
        {
          speaker: "SPEAKER_01",
          start: 0,
          end: 1,
          text: "Updated turn.",
          word_start: 0,
          word_end: 1,
        },
      ],
      words: [],
      segments: [],
    });

    const { container } = render(
      <DiarizationPlayground
        selectedModel="diar_streaming_sortformer_4spk-v2.1"
        selectedModelReady
        onModelRequired={vi.fn()}
        pipelineAsrModelId="Qwen3-ASR-0.6B"
        pipelineAlignerModelId="Qwen3-ForcedAligner-0.6B"
        pipelineModelsReady
      />,
    );
    const scope = within(container);

    expect(container.querySelectorAll('input[type="number"]')).toHaveLength(0);

    const fileInput = container.querySelector<HTMLInputElement>(
      'input[type="file"]',
    );
    fireEvent.change(fileInput!, {
      target: {
        files: [new File(["sample"], "meeting.wav", { type: "audio/wav" })],
      },
    });

    await scope.findByText("Talk Time");

    activateTab(scope, "Quality");
    const qualityPanel = within(scope.getByRole("tabpanel"));
    fireEvent.change(qualityPanel.getByLabelText("Min speakers"), {
      target: { value: "2" },
    });
    fireEvent.change(qualityPanel.getByLabelText("Max speakers"), {
      target: { value: "5" },
    });
    fireEvent.change(qualityPanel.getByLabelText("Min speech (ms)"), {
      target: { value: "180" },
    });
    fireEvent.change(qualityPanel.getByLabelText("Min silence (ms)"), {
      target: { value: "120" },
    });
    fireEvent.click(
      qualityPanel.getByRole("switch", { name: "LLM transcript refinement" }),
    );
    fireEvent.click(qualityPanel.getByRole("button", { name: "Rerun saved audio" }));

    await waitFor(() =>
      expect(apiMocks.rerunDiarizationRecord).toHaveBeenCalledWith("diar-1", {
        min_speakers: 2,
        max_speakers: 5,
        min_speech_duration_ms: 180,
        min_silence_duration_ms: 120,
        enable_llm_refinement: false,
      }),
    );

    await waitFor(() =>
      expect(apiMocks.diarizationRecordAudioUrl).toHaveBeenCalledWith("diar-2"),
    );

    activateTab(scope, "Transcript");
    expect(await scope.findByText("Updated turn.")).toBeInTheDocument();
  });

  it("restores the richer empty state when diarization returns no transcript entries", async () => {
    apiMocks.createDiarizationRecord.mockResolvedValue({
      id: "diar-empty",
      created_at: 3,
      model_id: "diar_streaming_sortformer_4spk-v2.1",
      asr_model_id: "Qwen3-ASR-0.6B",
      aligner_model_id: "Qwen3-ForcedAligner-0.6B",
      llm_model_id: "Qwen3-1.7B-GGUF",
      min_speakers: 1,
      max_speakers: 4,
      min_speech_duration_ms: 240,
      min_silence_duration_ms: 200,
      enable_llm_refinement: true,
      speaker_count: 0,
      corrected_speaker_count: 0,
      alignment_coverage: 0,
      unattributed_words: 0,
      llm_refined: false,
      duration_secs: 6.4,
      processing_time_ms: 140,
      rtf: 0.9,
      audio_mime_type: "audio/wav",
      audio_filename: "empty.wav",
      asr_text: "",
      transcript: "",
      raw_transcript: "",
      speaker_name_overrides: {},
      utterances: [],
      words: [],
      segments: [],
    });

    const { container } = render(
      <DiarizationPlayground
        selectedModel="diar_streaming_sortformer_4spk-v2.1"
        selectedModelReady
        onModelRequired={vi.fn()}
        pipelineAsrModelId="Qwen3-ASR-0.6B"
        pipelineAlignerModelId="Qwen3-ForcedAligner-0.6B"
        pipelineModelsReady
      />,
    );

    const fileInput = container.querySelector<HTMLInputElement>(
      'input[type="file"]',
    );
    expect(fileInput).not.toBeNull();

    fireEvent.change(fileInput!, {
      target: {
        files: [new File(["sample"], "empty.wav", { type: "audio/wav" })],
      },
    });

    await waitFor(() =>
      expect(apiMocks.createDiarizationRecord).toHaveBeenCalled(),
    );

    expect(container.querySelectorAll('input[type="number"]')).toHaveLength(0);

    expect(
      screen.getAllByRole("heading", { name: "Transcript" }).length,
    ).toBeGreaterThan(0);
    expect(screen.getByText("Ready to diarize")).toBeInTheDocument();
    expect(
      screen.getByText(
        /Record audio from your microphone or upload an audio file to start diarization\. Your speaker-segmented transcript will appear here\./i,
      ),
    ).toBeInTheDocument();
    expect(screen.getByTestId("diarization-stats-footer")).toBeInTheDocument();
  });

  it("transcodes recorded webm microphone audio before creating a diarization record", async () => {
    apiMocks.createDiarizationRecord.mockResolvedValue({
      id: "diar-live",
      created_at: 4,
      model_id: "diar_streaming_sortformer_4spk-v2.1",
      asr_model_id: "Qwen3-ASR-0.6B",
      aligner_model_id: "Qwen3-ForcedAligner-0.6B",
      llm_model_id: "Qwen3-1.7B-GGUF",
      min_speakers: 1,
      max_speakers: 4,
      min_speech_duration_ms: 240,
      min_silence_duration_ms: 200,
      enable_llm_refinement: true,
      speaker_count: 1,
      corrected_speaker_count: 1,
      alignment_coverage: 1,
      unattributed_words: 0,
      llm_refined: false,
      duration_secs: 1.2,
      processing_time_ms: 80,
      rtf: 0.3,
      audio_mime_type: "audio/wav",
      audio_filename: "audio.wav",
      asr_text: "Hello there.",
      transcript: "",
      raw_transcript: "",
      speaker_name_overrides: {},
      utterances: [
        {
          speaker: "SPEAKER_00",
          start: 0,
          end: 1,
          text: "Hello there.",
          word_start: 0,
          word_end: 1,
        },
      ],
      words: [],
      segments: [],
    });

    class MockAudioBuffer {
      readonly length: number;
      readonly sampleRate: number;
      readonly numberOfChannels: number;
      private readonly channels: Float32Array[];

      constructor(
        numberOfChannels: number,
        length: number,
        sampleRate: number,
      ) {
        this.length = length;
        this.sampleRate = sampleRate;
        this.numberOfChannels = numberOfChannels;
        this.channels = Array.from({ length: numberOfChannels }, () => {
          return new Float32Array(length);
        });
      }

      getChannelData(channel: number): Float32Array {
        return this.channels[channel]!;
      }
    }

    class MockAudioContext {
      decodeAudioData = vi.fn(async (_audioData: ArrayBuffer) => {
        const buffer = new MockAudioBuffer(1, 8, 16000);
        buffer.getChannelData(0).set([
          0.2, -0.2, 0.1, -0.1, 0.05, -0.05, 0.15, -0.15,
        ]);
        return buffer;
      });

      createBuffer(
        numberOfChannels: number,
        length: number,
        sampleRate: number,
      ): MockAudioBuffer {
        return new MockAudioBuffer(numberOfChannels, length, sampleRate);
      }

      close(): Promise<void> {
        return Promise.resolve();
      }
    }

    class MockMediaRecorder {
      static isTypeSupported(mimeType: string): boolean {
        return mimeType.startsWith("audio/webm");
      }

      readonly mimeType: string;
      ondataavailable: ((event: { data: Blob }) => void) | null = null;
      onstop: (() => void | Promise<void>) | null = null;

      constructor(
        _stream: MediaStream,
        options?: MediaRecorderOptions,
      ) {
        this.mimeType = options?.mimeType ?? "audio/webm";
      }

      start(): void {}

      stop(): void {
        this.ondataavailable?.({
          data: new Blob(["recorded"], { type: this.mimeType }),
        });
        void this.onstop?.();
      }
    }

    const stopTrack = vi.fn();
    const getUserMediaMock = vi.fn(async () => {
      return {
        getTracks: () => [{ stop: stopTrack }],
      } as unknown as MediaStream;
    });
    const originalMediaDevices = navigator.mediaDevices;

    vi.stubGlobal("AudioContext", MockAudioContext);
    vi.stubGlobal("MediaRecorder", MockMediaRecorder);
    Object.defineProperty(navigator, "mediaDevices", {
      configurable: true,
      value: {
        getUserMedia: getUserMediaMock,
      },
    });

    try {
      render(
        <DiarizationPlayground
          selectedModel="diar_streaming_sortformer_4spk-v2.1"
          selectedModelReady
          onModelRequired={vi.fn()}
          pipelineAsrModelId="Qwen3-ASR-0.6B"
          pipelineAlignerModelId="Qwen3-ForcedAligner-0.6B"
          pipelineModelsReady
        />,
      );

      fireEvent.click(screen.getByRole("button", { name: /record audio/i }));
      await waitFor(() => expect(getUserMediaMock).toHaveBeenCalled());

      fireEvent.click(screen.getByRole("button", { name: /recording/i }));

      await waitFor(() =>
        expect(apiMocks.createDiarizationRecord).toHaveBeenCalled(),
      );

      const request = apiMocks.createDiarizationRecord.mock.calls[0]?.[0];
      expect(request.audio_filename).toBe("audio.wav");
      expect(request.audio_file).toBeInstanceOf(Blob);
      expect(request.audio_file.type).toBe("audio/wav");
      expect(stopTrack).toHaveBeenCalled();
    } finally {
      Object.defineProperty(navigator, "mediaDevices", {
        configurable: true,
        value: originalMediaDevices,
      });
    }
  });
});
