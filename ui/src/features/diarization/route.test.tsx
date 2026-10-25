import {
  act,
  fireEvent,
  render,
  screen,
  waitFor,
  within,
} from "@testing-library/react";
import { MemoryRouter, Route, Routes, useLocation } from "react-router";
import { beforeEach, describe, expect, it, vi } from "vitest";
import type { ModelInfo } from "@/api";
import type { ModelRouteCapabilities } from "@/shared/api/models";

import { NotificationProvider } from "@/app/providers/NotificationProvider";
import { DiarizationPage } from "./route";

const apiMocks = vi.hoisted(() => ({
  getDiarizationRecord: vi.fn(),
  updateDiarizationRecord: vi.fn(),
  rerunDiarizationRecord: vi.fn(),
  cancelDiarizationRecord: vi.fn(),
  regenerateDiarizationSummary: vi.fn(),
  deleteDiarizationRecord: vi.fn(),
  diarizationRecordAudioUrl: vi.fn(),
}));

vi.mock("@/api", () => ({
  api: {
    getHealth: () => Promise.reject(new Error("offline")),
    getDiarizationRecord: apiMocks.getDiarizationRecord,
    updateDiarizationRecord: apiMocks.updateDiarizationRecord,
    rerunDiarizationRecord: apiMocks.rerunDiarizationRecord,
    cancelDiarizationRecord: apiMocks.cancelDiarizationRecord,
    regenerateDiarizationSummary: apiMocks.regenerateDiarizationSummary,
    deleteDiarizationRecord: apiMocks.deleteDiarizationRecord,
    diarizationRecordAudioUrl: apiMocks.diarizationRecordAudioUrl,
  },
}));

const componentMocks = vi.hoisted(() => ({
  routeModelModal: vi.fn(
    ({
      isOpen,
      title,
      models = [],
      sections = [],
    }: {
      isOpen: boolean;
      title: string;
      models?: ModelInfo[];
      sections?: Array<{ title?: string; models: ModelInfo[] }>;
    }) =>
      isOpen ? (
        <div
          data-testid="route-model-modal"
          data-models={models.map((model) => model.variant).join(",")}
          data-sections={sections
            .map((section) => section.title ?? "")
            .join(",")}
        >
          {title}
        </div>
      ) : null,
  ),
}));

vi.mock("@/features/models/components/RouteModelModal", () => ({
  RouteModelModal: componentMocks.routeModelModal,
}));

const baseModels: ModelInfo[] = [
  {
    variant: "diar_streaming_sortformer_4spk-v2.1",
    route_capabilities: caps({ diarization_records: true }),
    status: "ready" as const,
    local_path: "/models/diar",
    size_bytes: null,
    download_progress: null,
    error_message: null,
  },
  {
    variant: "Whisper-Large-v3-Turbo",
    route_capabilities: caps({ openai_audio_transcriptions: true, speech_to_text_jobs: true }),
    status: "ready" as const,
    local_path: "/models/asr",
    size_bytes: null,
    download_progress: null,
    error_message: null,
  },
  {
    variant: "Qwen3-ForcedAligner-0.6B",
    route_capabilities: caps({ forced_alignment: true }),
    status: "ready" as const,
    local_path: "/models/aligner",
    size_bytes: null,
    download_progress: null,
    error_message: null,
  },
  {
    variant: "Qwen3.5-4B",
    route_capabilities: caps({ openai_chat_completions: true }),
    status: "ready" as const,
    local_path: "/models/llm",
    size_bytes: null,
    download_progress: null,
    error_message: null,
  },
];

const graniteModel: ModelInfo = {
  variant: "Granite-Speech-4.1-2B-Plus",
  route_capabilities: caps({
    openai_audio_transcriptions: true,
    speech_to_text_jobs: true,
    speaker_attributed_asr: true,
  }),
  status: "ready" as const,
  local_path: "/models/granite",
  size_bytes: null,
  download_progress: null,
  error_message: null,
};

const baseProps = {
  models: baseModels,
  loading: false,
  downloadProgress: {},
  onDownload: vi.fn(),
  onCancelDownload: vi.fn(),
  onLoad: vi.fn(),
  onUnload: vi.fn(),
  onDelete: vi.fn(),
  onSelect: vi.fn(),
  onError: vi.fn(),
};

function createRouteProps(
  overrides: Partial<typeof baseProps> = {},
): typeof baseProps {
  return {
    ...baseProps,
    onDownload: vi.fn(),
    onCancelDownload: vi.fn(),
    onLoad: vi.fn(),
    onUnload: vi.fn(),
    onDelete: vi.fn(),
    onSelect: vi.fn(),
    onError: vi.fn(),
    ...overrides,
  };
}

function LocationProbe() {
  const location = useLocation();
  return (
    <div data-testid="location">{`${location.pathname}${location.search}`}</div>
  );
}

/** The page is a record's detail, reached through the transcription route. */
const DETAIL_ENTRY = "/transcription/diar-1?mode=diarization";

function renderRoute(props: typeof baseProps = createRouteProps()) {
  return render(
    <NotificationProvider>
      <MemoryRouter initialEntries={[DETAIL_ENTRY]}>
        <LocationProbe />
        <Routes>
          <Route path="/transcription" element={<h1>Transcription list</h1>} />
          <Route
            path="/transcription/:recordId"
            element={<DiarizationPage {...props} />}
          />
        </Routes>
      </MemoryRouter>
    </NotificationProvider>,
  );
}

async function openRecordMenu() {
  const recordMenu = await screen.findByRole("button", {
    name: /^More actions for /,
  });
  fireEvent.pointerDown(recordMenu, { button: 0, ctrlKey: false });
  fireEvent.click(recordMenu);
}

const fullRecord = {
  id: "diar-1",
  created_at: 1,
  model_id: "diar_streaming_sortformer_4spk-v2.1",
  asr_model_id: "Whisper-Large-v3-Turbo",
  aligner_model_id: "Qwen3-ForcedAligner-0.6B",
  llm_model_id: "Qwen3.5-4B",
  processing_status: "ready" as const,
  processing_error: null,
  min_speakers: 1,
  max_speakers: 4,
  min_speech_duration_ms: 240,
  min_silence_duration_ms: 200,
  processing_time_ms: 120,
  duration_secs: 42,
  rtf: 0.5,
  speaker_count: 2,
  corrected_speaker_count: 2,
  alignment_coverage: 0.82,
  unattributed_words: 0,
  asr_text: "Hello there.",
  raw_transcript: "Speaker 1: Hello there.",
  transcript: "Speaker 1: Hello there.",
  summary_status: "pending",
  summary_model_id: "Qwen3.5-4B",
  summary_text: null,
  summary_error: null,
  summary_updated_at: null,
  segments: [],
  words: [],
  utterances: [
    {
      speaker: "SPEAKER_00",
      start: 0,
      end: 1,
      text: "Hello there.",
    },
  ],
  speaker_name_overrides: {},
  audio_mime_type: "audio/wav",
  audio_filename: "meeting.wav",
};


/**
 * Full route-capability envelope for fixture models: every flag defaults to
 * false so tests opt in per role.
 */
function caps(
  overrides: Partial<ModelRouteCapabilities> = {},
): ModelRouteCapabilities {
  return {
    openai_chat_completions: false,
    openai_responses: false,
    openai_audio_speech: false,
    openai_audio_transcriptions: false,
    speech_to_text_jobs: false,
    speech_to_text_realtime: false,
    diarization_records: false,
    text_to_speech_records: false,
    voice_design_records: false,
    voice_clone_records: false,
    saved_voice_reuse: false,
    studio_projects: false,
    voice_realtime_text_model: false,
    voice_realtime_modular_asr: false,
    voice_realtime_modular_tts: false,
    voice_realtime_unified: false,
    forced_alignment: false,
    tokenizer: false,
    ...overrides,
  };
}

describe("DiarizationPage record detail", () => {
  beforeEach(() => {
    vi.useRealTimers();
    apiMocks.getDiarizationRecord.mockReset();
    apiMocks.updateDiarizationRecord.mockReset();
    apiMocks.rerunDiarizationRecord.mockReset();
    apiMocks.cancelDiarizationRecord.mockReset();
    apiMocks.regenerateDiarizationSummary.mockReset();
    apiMocks.deleteDiarizationRecord.mockReset();
    apiMocks.diarizationRecordAudioUrl.mockReset();
    componentMocks.routeModelModal.mockClear();

    apiMocks.getDiarizationRecord.mockResolvedValue(fullRecord);
    apiMocks.cancelDiarizationRecord.mockResolvedValue({
      ...fullRecord,
      processing_status: "failed",
      processing_error: "Cancelled by user.",
    });
    apiMocks.diarizationRecordAudioUrl.mockReturnValue("/audio/meeting.wav");
  });

  it("polls a newly created pending diarization record until processing completes", async () => {
    vi.useFakeTimers();
    const pendingCreatedRecord = {
      ...fullRecord,
      processing_status: "pending" as const,
      processing_time_ms: 0,
      duration_secs: null,
      rtf: null,
      speaker_count: 0,
      corrected_speaker_count: 0,
      alignment_coverage: null,
      asr_text: "",
      raw_transcript: "",
      transcript: "",
      summary_status: "not_requested" as const,
      summary_model_id: null,
      segments: [],
      words: [],
      utterances: [],
    };
    apiMocks.getDiarizationRecord
      .mockResolvedValueOnce(pendingCreatedRecord)
      .mockResolvedValueOnce(fullRecord);

    renderRoute();

    await act(async () => {
      await Promise.resolve();
    });
    expect(
      screen.getByText(
        "This diarization run is queued and will begin processing shortly.",
      ),
    ).toBeInTheDocument();

    await act(async () => {
      vi.advanceTimersByTime(2600);
    });

    expect(apiMocks.getDiarizationRecord).toHaveBeenCalledTimes(2);
  });

  it("shows a failed run as failed, without a loading state or speaker count", async () => {
    apiMocks.getDiarizationRecord.mockResolvedValue({
      ...fullRecord,
      processing_status: "failed",
      processing_error: "invocation tensor input has incompatible device",
      speaker_count: 0,
      corrected_speaker_count: 0,
      transcript: "",
      raw_transcript: "",
      asr_text: "",
      utterances: [],
      summary_status: "not_requested",
    });

    renderRoute();

    expect(
      await screen.findByText(
        "This run failed before producing a transcript. Re-run it from the Quality tab, or delete it.",
      ),
    ).toBeInTheDocument();
    expect(screen.queryByText(/Diarization record loading/)).not.toBeInTheDocument();
    expect(screen.queryByText(/0 speakers/)).not.toBeInTheDocument();
  });

  it("loads the selected diarization record on /transcription/:recordId", async () => {
    renderRoute();

    await waitFor(() =>
      expect(apiMocks.getDiarizationRecord).toHaveBeenCalledWith("diar-1"),
    );

    expect(
      await screen.findByRole("heading", { name: "meeting.wav" }),
    ).toBeInTheDocument();
    expect(screen.getByTestId("diarization-review-player")).toHaveClass("fixed");
  });

  it("navigates back to the transcription list from a record page", async () => {
    renderRoute();

    await waitFor(() =>
      expect(apiMocks.getDiarizationRecord).toHaveBeenCalledWith("diar-1"),
    );

    fireEvent.click(
      within(screen.getByRole("navigation", { name: "Breadcrumb" })).getByRole("link", { name: "Transcription" }),
    );

    expect(
      await screen.findByRole("heading", { name: "Transcription list" }),
    ).toBeInTheDocument();
    expect(screen.getByTestId("location")).toHaveTextContent(/^\/transcription$/);
  });

  it("deletes a diarization record from the detail page and returns to the transcription list", async () => {
    apiMocks.deleteDiarizationRecord.mockResolvedValue({
      id: "diar-1",
      deleted: true,
    });

    renderRoute();

    await waitFor(() =>
      expect(apiMocks.getDiarizationRecord).toHaveBeenCalledWith("diar-1"),
    );

    await openRecordMenu();
    fireEvent.click(await screen.findByRole("menuitem", { name: /^Delete$/i }));
    fireEvent.click(
      await screen.findByRole("button", { name: /Delete record/i }),
    );

    await waitFor(() =>
      expect(apiMocks.deleteDiarizationRecord).toHaveBeenCalledWith("diar-1"),
    );
    expect(
      await screen.findByRole("heading", { name: "Transcription list" }),
    ).toBeInTheDocument();
    expect(screen.getByTestId("location")).toHaveTextContent(/^\/transcription$/);
  });

  it("cancels a pending diarization record from the detail page", async () => {
    apiMocks.getDiarizationRecord.mockResolvedValue({
      ...fullRecord,
      processing_status: "processing" as const,
      processing_error: null,
      transcript: "",
      raw_transcript: "",
      utterances: [],
      words: [],
      segments: [],
      summary_status: "not_requested" as const,
      summary_text: null,
    });

    renderRoute();

    await waitFor(() =>
      expect(apiMocks.getDiarizationRecord).toHaveBeenCalledWith("diar-1"),
    );

    await openRecordMenu();
    fireEvent.click(await screen.findByRole("menuitem", { name: /Cancel run/i }));

    await waitFor(() =>
      expect(apiMocks.cancelDiarizationRecord).toHaveBeenCalledWith("diar-1"),
    );
  });

  it("opens the diarization models, without Granite, from the record menu", async () => {
    const props = createRouteProps({
      models: [...baseModels, graniteModel],
    });

    renderRoute(props);

    await waitFor(() =>
      expect(apiMocks.getDiarizationRecord).toHaveBeenCalledWith("diar-1"),
    );

    await openRecordMenu();
    fireEvent.click(await screen.findByRole("menuitem", { name: "Models" }));

    const modelModal = await screen.findByTestId("route-model-modal");
    expect(modelModal).toHaveTextContent("Diarization models");
    expect(modelModal.getAttribute("data-models")).toBe(
      "diar_streaming_sortformer_4spk-v2.1,Whisper-Large-v3-Turbo,Qwen3-ForcedAligner-0.6B,Qwen3.5-4B",
    );
    expect(modelModal.getAttribute("data-sections")).toBe(
      "Diarization,ASR,Forced Aligner,Summary",
    );
    expect(props.onError).not.toHaveBeenCalled();
  });

  it("polls the selected record while its summary is pending", async () => {
    vi.useFakeTimers();
    apiMocks.getDiarizationRecord.mockResolvedValue(fullRecord);

    renderRoute();

    await act(async () => {
      await Promise.resolve();
    });

    expect(apiMocks.getDiarizationRecord).toHaveBeenCalledTimes(1);

    await act(async () => {
      vi.advanceTimersByTime(2600);
    });

    expect(apiMocks.getDiarizationRecord).toHaveBeenCalledTimes(2);
  });
});
