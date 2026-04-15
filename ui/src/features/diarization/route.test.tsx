import { act, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { MemoryRouter, Route, Routes } from "react-router-dom";
import { beforeEach, describe, expect, it, vi } from "vitest";
import type { ModelInfo } from "@/api";

import { NotificationProvider } from "@/app/providers/NotificationProvider";
import { DiarizationPage } from "./route";

const apiMocks = vi.hoisted(() => ({
  listDiarizationRecords: vi.fn(),
  listDiarizationRecordPage: vi.fn(),
  getDiarizationRecord: vi.fn(),
  updateDiarizationRecord: vi.fn(),
  rerunDiarizationRecord: vi.fn(),
  cancelDiarizationRecord: vi.fn(),
  regenerateDiarizationSummary: vi.fn(),
  deleteDiarizationRecord: vi.fn(),
  createDiarizationRecord: vi.fn(),
  diarizationRecordAudioUrl: vi.fn(),
}));

vi.mock("@/api", () => ({
  api: {
    listDiarizationRecords: apiMocks.listDiarizationRecords,
    listDiarizationRecordPage: apiMocks.listDiarizationRecordPage,
    getDiarizationRecord: apiMocks.getDiarizationRecord,
    updateDiarizationRecord: apiMocks.updateDiarizationRecord,
    rerunDiarizationRecord: apiMocks.rerunDiarizationRecord,
    cancelDiarizationRecord: apiMocks.cancelDiarizationRecord,
    regenerateDiarizationSummary: apiMocks.regenerateDiarizationSummary,
    deleteDiarizationRecord: apiMocks.deleteDiarizationRecord,
    createDiarizationRecord: apiMocks.createDiarizationRecord,
    diarizationRecordAudioUrl: apiMocks.diarizationRecordAudioUrl,
  },
}));

vi.mock("@/features/models/components/RouteModelModal", () => ({
  RouteModelModal: () => null,
}));

const baseModels: ModelInfo[] = [
  {
    variant: "diar_streaming_sortformer_4spk-v2.1",
    status: "ready" as const,
    local_path: "/models/diar",
    size_bytes: null,
    download_progress: null,
    error_message: null,
  },
  {
    variant: "Whisper-Large-v3-Turbo",
    status: "ready" as const,
    local_path: "/models/asr",
    size_bytes: null,
    download_progress: null,
    error_message: null,
  },
  {
    variant: "Qwen3-ForcedAligner-0.6B",
    status: "ready" as const,
    local_path: "/models/aligner",
    size_bytes: null,
    download_progress: null,
    error_message: null,
  },
  {
    variant: "Qwen3.5-4B",
    status: "ready" as const,
    local_path: "/models/llm",
    size_bytes: null,
    download_progress: null,
    error_message: null,
  },
];

const baseProps = {
  models: baseModels,
  selectedModel: "diar_streaming_sortformer_4spk-v2.1",
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

function renderRoute(
  initialEntry: string,
  props: typeof baseProps = createRouteProps(),
) {
  return render(
    <NotificationProvider>
      <MemoryRouter initialEntries={[initialEntry]}>
        <Routes>
          <Route path="/diarization" element={<DiarizationPage {...props} />} />
          <Route
            path="/diarization/:recordId"
            element={<DiarizationPage {...props} />}
          />
        </Routes>
      </MemoryRouter>
    </NotificationProvider>,
  );
}

const pendingSummaryRecord = {
  id: "diar-1",
  created_at: 1,
  model_id: "diar_streaming_sortformer_4spk-v2.1",
  processing_status: "ready" as const,
  processing_error: null,
  speaker_count: 2,
  corrected_speaker_count: 2,
  duration_secs: 42,
  processing_time_ms: 120,
  rtf: 0.5,
  audio_mime_type: "audio/wav",
  audio_filename: "meeting.wav",
  transcript_preview: "Hello there.",
  transcript_chars: 12,
  summary_status: "pending",
  summary_preview: null,
  summary_chars: 0,
};

const readySummaryRecord = {
  ...pendingSummaryRecord,
  id: "diar-2",
  audio_filename: "board-call.wav",
  summary_status: "ready",
  summary_preview: "Board sync covered runway, launch timing, and next hiring steps.",
  summary_chars: 63,
};

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
  enable_llm_refinement: true,
  processing_time_ms: 120,
  duration_secs: 42,
  rtf: 0.5,
  speaker_count: 2,
  corrected_speaker_count: 2,
  alignment_coverage: 0.82,
  unattributed_words: 0,
  llm_refined: true,
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

describe("DiarizationPage routes", () => {
  beforeEach(() => {
    vi.useRealTimers();
    apiMocks.listDiarizationRecords.mockReset();
    apiMocks.listDiarizationRecordPage.mockReset();
    apiMocks.getDiarizationRecord.mockReset();
    apiMocks.updateDiarizationRecord.mockReset();
    apiMocks.rerunDiarizationRecord.mockReset();
    apiMocks.cancelDiarizationRecord.mockReset();
    apiMocks.regenerateDiarizationSummary.mockReset();
    apiMocks.deleteDiarizationRecord.mockReset();
    apiMocks.createDiarizationRecord.mockReset();
    apiMocks.diarizationRecordAudioUrl.mockReset();

    apiMocks.listDiarizationRecords.mockResolvedValue([]);
    apiMocks.listDiarizationRecordPage.mockImplementation(async () => ({
      items: await apiMocks.listDiarizationRecords(),
      pagination: {
        next_cursor: null,
        has_more: false,
        limit: 25,
      },
    }));
    apiMocks.getDiarizationRecord.mockResolvedValue(fullRecord);
    apiMocks.cancelDiarizationRecord.mockResolvedValue({
      ...fullRecord,
      processing_status: "failed",
      processing_error: "Cancelled by user.",
    });
    apiMocks.createDiarizationRecord.mockResolvedValue(fullRecord);
    apiMocks.diarizationRecordAudioUrl.mockReturnValue("/audio/meeting.wav");
  });

  it("renders the diarization route and loads history from the route hook", async () => {
    renderRoute("/diarization");

    await waitFor(() =>
      expect(apiMocks.listDiarizationRecords).toHaveBeenCalledTimes(1),
    );

    expect(
      await screen.findByRole("heading", { name: "Diarization" }),
    ).toBeInTheDocument();
    expect(
      screen.getByRole("button", { name: /New diarization/i }),
    ).toBeInTheDocument();
    expect(
      screen.queryByRole("heading", { name: "History" }),
    ).not.toBeInTheDocument();
    expect(screen.getByText("No diarization records yet")).toBeInTheDocument();
  });

  it("loads more diarization history rows", async () => {
    apiMocks.listDiarizationRecordPage.mockReset();
    apiMocks.listDiarizationRecordPage
      .mockResolvedValueOnce({
        items: [{ ...readySummaryRecord, id: "diar-page-1", audio_filename: "page-one.wav" }],
        pagination: {
          next_cursor: "diar-cursor-2",
          has_more: true,
          limit: 25,
        },
      })
      .mockResolvedValueOnce({
        items: [{ ...readySummaryRecord, id: "diar-page-2", audio_filename: "page-two.wav" }],
        pagination: {
          next_cursor: null,
          has_more: false,
          limit: 25,
        },
      });

    renderRoute("/diarization");

    expect(await screen.findByText("page-one.wav")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Load more" }));
    expect(await screen.findByText("page-two.wav")).toBeInTheDocument();
    expect(await screen.findByText("page-one.wav")).toBeInTheDocument();

    expect(apiMocks.listDiarizationRecordPage).toHaveBeenNthCalledWith(1, {
      limit: 25,
      cursor: null,
    });
    expect(apiMocks.listDiarizationRecordPage).toHaveBeenNthCalledWith(2, {
      limit: 25,
      cursor: "diar-cursor-2",
    });
  });

  it("shows summaries in the diarization history table", async () => {
    apiMocks.listDiarizationRecords.mockResolvedValue([readySummaryRecord]);

    renderRoute("/diarization");

    await waitFor(() =>
      expect(apiMocks.listDiarizationRecords).toHaveBeenCalledTimes(1),
    );

    expect(screen.getByRole("columnheader", { name: "Summary" })).toBeInTheDocument();
    expect(
      screen.getByText(
        "Board sync covered runway, launch timing, and next hiring steps.",
      ),
    ).toBeInTheDocument();
    expect(
      screen.queryByText("diar_streaming_sortformer_4spk-v2.1"),
    ).not.toBeInTheDocument();
    expect(screen.queryByText("Hello there.")).not.toBeInTheDocument();
  });

  it("shows standard row actions from the diarization history menu", async () => {
    apiMocks.listDiarizationRecords.mockResolvedValue([readySummaryRecord]);

    renderRoute("/diarization");

    expect(await screen.findByText("board-call.wav")).toBeInTheDocument();

    fireEvent.pointerDown(
      screen.getByRole("button", { name: /More actions for board-call\.wav/i }),
      { button: 0, ctrlKey: false },
    );

    expect(await screen.findByRole("menuitem", { name: /Open record/i })).toBeVisible();
    expect(screen.getByRole("menuitem", { name: /Copy transcript/i })).toBeVisible();
    expect(screen.getByRole("menuitem", { name: /^Export$/i })).toBeVisible();
    expect(screen.getByRole("menuitem", { name: /^Delete$/i })).toBeVisible();
  });

  it("opens the creation modal and routes new diarization runs to their detail page", async () => {
    let resolveRefreshHistory: ((records: unknown[]) => void) | null = null;
    const pendingCreatedRecord = {
      ...fullRecord,
      processing_status: "pending" as const,
      processing_time_ms: 0,
      duration_secs: null,
      rtf: null,
      speaker_count: 0,
      corrected_speaker_count: 0,
      alignment_coverage: null,
      llm_refined: false,
      asr_text: "",
      raw_transcript: "",
      transcript: "",
      summary_status: "not_requested" as const,
      summary_model_id: null,
      segments: [],
      words: [],
      utterances: [],
    };
    const refreshedCreatedSummaryRecord = {
      ...readySummaryRecord,
      id: "diar-1",
      audio_filename: "meeting.wav",
    };
    apiMocks.listDiarizationRecords
      .mockResolvedValueOnce([])
      .mockImplementationOnce(
        () =>
          new Promise((resolve) => {
            resolveRefreshHistory = resolve;
          }),
      );
    apiMocks.createDiarizationRecord.mockResolvedValueOnce(pendingCreatedRecord);
    apiMocks.getDiarizationRecord
      .mockResolvedValueOnce(pendingCreatedRecord)
      .mockResolvedValueOnce(fullRecord);

    renderRoute("/diarization");

    await waitFor(() =>
      expect(apiMocks.listDiarizationRecords).toHaveBeenCalledTimes(1),
    );

    fireEvent.click(screen.getByRole("button", { name: /New diarization/i }));

    expect(
      await screen.findByRole("heading", { name: "New diarization" }),
    ).toBeInTheDocument();
    expect(screen.getByText("Choose how to start")).toBeInTheDocument();
    expect(screen.getByText("Model readiness")).toBeInTheDocument();

    const fileInput = document.querySelector(
      'input[type="file"]',
    ) as HTMLInputElement | null;
    expect(fileInput).not.toBeNull();

    fireEvent.change(fileInput!, {
      target: {
        files: [new File(["audio"], "meeting.wav", { type: "audio/wav" })],
      },
    });

    await waitFor(() =>
      expect(apiMocks.createDiarizationRecord).toHaveBeenCalledTimes(1),
    );
    expect(apiMocks.createDiarizationRecord).toHaveBeenCalledWith(
      expect.objectContaining({
        audio_filename: "meeting.wav",
      }),
    );
    await waitFor(() =>
      expect(apiMocks.getDiarizationRecord).toHaveBeenCalledWith("diar-1"),
    );

    expect(
      await screen.findByRole("heading", { name: "Diarization Record" }),
    ).toBeInTheDocument();
    expect(
      screen.queryByRole("heading", { name: "New diarization" }),
    ).not.toBeInTheDocument();

    await act(async () => {
      if (resolveRefreshHistory) {
        resolveRefreshHistory([refreshedCreatedSummaryRecord]);
      }
    });

    fireEvent.click(screen.getByRole("button", { name: /Back to diarization/i }));

    expect(
      await screen.findByText(
        "Board sync covered runway, launch timing, and next hiring steps.",
      ),
    ).toBeInTheDocument();
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
      llm_refined: false,
      asr_text: "",
      raw_transcript: "",
      transcript: "",
      summary_status: "not_requested" as const,
      summary_model_id: null,
      segments: [],
      words: [],
      utterances: [],
    };
    apiMocks.listDiarizationRecords.mockResolvedValue([]);
    apiMocks.getDiarizationRecord
      .mockResolvedValueOnce(pendingCreatedRecord)
      .mockResolvedValueOnce(fullRecord);

    renderRoute("/diarization/diar-1");

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

  it("shows a single load action until the full diarization stack is ready", async () => {
    const props = createRouteProps({
      models: [
        {
          variant: "diar_streaming_sortformer_4spk-v2.1",
          status: "downloaded" as const,
          local_path: "/models/diar",
          size_bytes: null,
          download_progress: null,
          error_message: null,
        },
        {
          variant: "Whisper-Large-v3-Turbo",
          status: "not_downloaded" as const,
          local_path: "/models/asr",
          size_bytes: null,
          download_progress: null,
          error_message: null,
        },
        {
          variant: "Qwen3-ForcedAligner-0.6B",
          status: "ready" as const,
          local_path: "/models/aligner",
          size_bytes: null,
          download_progress: null,
          error_message: null,
        },
        {
          variant: "Qwen3.5-4B",
          status: "downloaded" as const,
          local_path: "/models/llm",
          size_bytes: null,
          download_progress: null,
          error_message: null,
        },
      ],
      selectedModel: "diar_streaming_sortformer_4spk-v2.1",
    });

    renderRoute("/diarization", props);

    await waitFor(() =>
      expect(apiMocks.listDiarizationRecords).toHaveBeenCalledTimes(1),
    );

    fireEvent.click(screen.getByRole("button", { name: /New diarization/i }));
    expect(await screen.findByText("NOT LOADED")).toBeInTheDocument();
    expect(
      screen.queryByRole("button", { name: "Unload Models" }),
    ).not.toBeInTheDocument();
    fireEvent.click(await screen.findByRole("button", { name: "Load Models" }));

    expect(props.onLoad).toHaveBeenCalledWith("diar_streaming_sortformer_4spk-v2.1");
    expect(props.onLoad).toHaveBeenCalledWith("Qwen3.5-4B");
    expect(props.onDownload).toHaveBeenCalledWith("Whisper-Large-v3-Turbo");
    expect(props.onUnload).not.toHaveBeenCalled();
  });

  it("shows unload only after the full diarization stack is ready", async () => {
    const props = createRouteProps();

    renderRoute("/diarization", props);

    await waitFor(() =>
      expect(apiMocks.listDiarizationRecords).toHaveBeenCalledTimes(1),
    );

    fireEvent.click(screen.getByRole("button", { name: /New diarization/i }));
    expect(await screen.findByText("READY")).toBeInTheDocument();
    expect(
      screen.queryByRole("button", { name: "Load Models" }),
    ).not.toBeInTheDocument();
    fireEvent.click(
      await screen.findByRole("button", { name: "Unload Models" }),
    );

    expect(props.onUnload).toHaveBeenCalledWith("diar_streaming_sortformer_4spk-v2.1");
    expect(props.onUnload).toHaveBeenCalledWith("Whisper-Large-v3-Turbo");
    expect(props.onUnload).toHaveBeenCalledWith("Qwen3-ForcedAligner-0.6B");
    expect(props.onUnload).toHaveBeenCalledWith("Qwen3.5-4B");
  });

  it("shows a loading readiness state while diarization models are loading", async () => {
    const props = createRouteProps({
      models: [
        {
          variant: "diar_streaming_sortformer_4spk-v2.1",
          status: "loading" as const,
          local_path: "/models/diar",
          size_bytes: null,
          download_progress: null,
          error_message: null,
        },
        {
          variant: "Whisper-Large-v3-Turbo",
          status: "ready" as const,
          local_path: "/models/asr",
          size_bytes: null,
          download_progress: null,
          error_message: null,
        },
        {
          variant: "Qwen3-ForcedAligner-0.6B",
          status: "ready" as const,
          local_path: "/models/aligner",
          size_bytes: null,
          download_progress: null,
          error_message: null,
        },
        {
          variant: "Qwen3.5-4B",
          status: "ready" as const,
          local_path: "/models/llm",
          size_bytes: null,
          download_progress: null,
          error_message: null,
        },
      ],
      selectedModel: "diar_streaming_sortformer_4spk-v2.1",
    });

    renderRoute("/diarization", props);

    await waitFor(() =>
      expect(apiMocks.listDiarizationRecords).toHaveBeenCalledTimes(1),
    );

    fireEvent.click(screen.getByRole("button", { name: /New diarization/i }));
    expect(await screen.findByText("LOADING")).toBeInTheDocument();
    expect(
      screen.getByRole("button", { name: "Loading models..." }),
    ).toBeDisabled();
  });

  it("loads the selected diarization record on /diarization/:recordId", async () => {
    apiMocks.listDiarizationRecords.mockResolvedValue([pendingSummaryRecord]);
    apiMocks.getDiarizationRecord.mockResolvedValue(fullRecord);

    renderRoute("/diarization/diar-1");

    await waitFor(() =>
      expect(apiMocks.getDiarizationRecord).toHaveBeenCalledWith("diar-1"),
    );

    expect(await screen.findByText("meeting.wav")).toBeInTheDocument();
    expect(screen.getByText("Diarization Record")).toBeInTheDocument();
    expect(screen.getByTestId("diarization-review-player")).toHaveClass("fixed");
  });

  it("opens saved diarization records from the history table", async () => {
    apiMocks.listDiarizationRecords.mockResolvedValue([pendingSummaryRecord]);

    renderRoute("/diarization");

    await waitFor(() =>
      expect(apiMocks.listDiarizationRecords).toHaveBeenCalledTimes(1),
    );

    fireEvent.click(screen.getByText("meeting.wav"));

    await waitFor(() =>
      expect(apiMocks.getDiarizationRecord).toHaveBeenCalledWith("diar-1"),
    );
    expect(
      await screen.findByRole("heading", { name: "Diarization Record" }),
    ).toBeInTheDocument();
  });

  it("navigates back to the diarization index from a record page", async () => {
    apiMocks.listDiarizationRecords.mockResolvedValue([pendingSummaryRecord]);

    renderRoute("/diarization/diar-1");

    await waitFor(() =>
      expect(apiMocks.getDiarizationRecord).toHaveBeenCalledWith("diar-1"),
    );

    fireEvent.click(
      await screen.findByRole("button", { name: /Back to diarization/i }),
    );

    expect(
      await screen.findByRole("heading", { name: "Diarization" }),
    ).toBeInTheDocument();
  });

  it("deletes a diarization record from the detail page and returns to history", async () => {
    apiMocks.listDiarizationRecords.mockResolvedValue([pendingSummaryRecord]);
    apiMocks.deleteDiarizationRecord.mockResolvedValue({
      id: "diar-1",
      deleted: true,
    });

    renderRoute("/diarization/diar-1");

    await waitFor(() =>
      expect(apiMocks.getDiarizationRecord).toHaveBeenCalledWith("diar-1"),
    );

    fireEvent.click(await screen.findByRole("button", { name: /^Delete$/i }));
    fireEvent.click(
      await screen.findByRole("button", { name: /Delete record/i }),
    );

    await waitFor(() =>
      expect(apiMocks.deleteDiarizationRecord).toHaveBeenCalledWith("diar-1"),
    );
    expect(
      await screen.findByRole("heading", { name: "Diarization" }),
    ).toBeInTheDocument();
  });

  it("cancels a pending diarization record from the detail page", async () => {
    apiMocks.listDiarizationRecords.mockResolvedValue([pendingSummaryRecord]);
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

    renderRoute("/diarization/diar-1");

    await waitFor(() =>
      expect(apiMocks.getDiarizationRecord).toHaveBeenCalledWith("diar-1"),
    );

    fireEvent.click(await screen.findByRole("button", { name: /Cancel run/i }));

    await waitFor(() =>
      expect(apiMocks.cancelDiarizationRecord).toHaveBeenCalledWith("diar-1"),
    );
  });

  it("deletes from the diarization history menu and refreshes the table", async () => {
    apiMocks.listDiarizationRecords
      .mockResolvedValueOnce([readySummaryRecord])
      .mockResolvedValueOnce([]);
    apiMocks.deleteDiarizationRecord.mockResolvedValue({
      id: readySummaryRecord.id,
      deleted: true,
    });

    renderRoute("/diarization");

    expect(await screen.findByText("board-call.wav")).toBeInTheDocument();

    fireEvent.pointerDown(
      screen.getByRole("button", { name: /More actions for board-call\.wav/i }),
      { button: 0, ctrlKey: false },
    );
    fireEvent.click(await screen.findByRole("menuitem", { name: /^Delete$/i }));
    fireEvent.click(
      await screen.findByRole("button", { name: /Delete diarization/i }),
    );

    await waitFor(() =>
      expect(apiMocks.deleteDiarizationRecord).toHaveBeenCalledWith(
        readySummaryRecord.id,
      ),
    );
    await waitFor(() =>
      expect(apiMocks.listDiarizationRecords).toHaveBeenCalledTimes(2),
    );
    expect(
      await screen.findByRole("heading", { name: "No diarization records yet" }),
    ).toBeInTheDocument();
  });

  it("does not keep polling diarization history while viewing the route", async () => {
    vi.useFakeTimers();
    apiMocks.listDiarizationRecords.mockResolvedValue([pendingSummaryRecord]);

    renderRoute("/diarization");

    await act(async () => {
      await Promise.resolve();
    });

    expect(apiMocks.listDiarizationRecords).toHaveBeenCalledTimes(1);

    await act(async () => {
      vi.advanceTimersByTime(2600);
    });

    expect(apiMocks.listDiarizationRecords).toHaveBeenCalledTimes(1);
  });

  it("polls the selected record while its summary is pending", async () => {
    vi.useFakeTimers();
    apiMocks.listDiarizationRecords.mockResolvedValue([pendingSummaryRecord]);
    apiMocks.getDiarizationRecord.mockResolvedValue(fullRecord);

    renderRoute("/diarization/diar-1");

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
