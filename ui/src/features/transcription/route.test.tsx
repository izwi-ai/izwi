import { act, fireEvent, render, screen, waitFor } from "@testing-library/react";
import type { ComponentProps } from "react";
import { MemoryRouter, Route, Routes } from "react-router-dom";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { NotificationProvider } from "@/app/providers/NotificationProvider";
import { TranscriptionPage } from "./route";

const apiMocks = vi.hoisted(() => ({
  listTranscriptionRecords: vi.fn(),
  listSpeechTextJobPage: vi.fn(),
  getTranscriptionRecord: vi.fn(),
  getDiarizationRecord: vi.fn(),
  transcriptionRecordAudioUrl: vi.fn(),
  deleteTranscriptionRecord: vi.fn(),
  deleteDiarizationRecord: vi.fn(),
  regenerateTranscriptionSummary: vi.fn(),
  createTranscriptionRecord: vi.fn(),
  createTranscriptionRecordStream: vi.fn(),
  createDiarizationRecord: vi.fn(),
}));

const hookMocks = vi.hoisted(() => ({
  useRouteModelSelection: vi.fn(),
}));

const componentMocks = vi.hoisted(() => ({
  routeModelModal: vi.fn(
    ({
      isOpen,
      title,
      zIndexClassName,
      onUseModel,
    }: {
      isOpen: boolean;
      title: string;
      zIndexClassName?: string;
      onUseModel?: (variant: string) => void;
    }) =>
      isOpen ? (
        <div
          data-testid="route-model-modal"
          data-z-index={zIndexClassName ?? ""}
        >
          {title}
          <button
            type="button"
            onClick={() => onUseModel?.("Parakeet-TDT-0.6B-v3")}
          >
            Use mocked model
          </button>
        </div>
      ) : null,
  ),
}));

vi.mock("@/api", () => ({
  api: {
    listTranscriptionRecords: apiMocks.listTranscriptionRecords,
    listSpeechTextJobPage: apiMocks.listSpeechTextJobPage,
    getTranscriptionRecord: apiMocks.getTranscriptionRecord,
    getDiarizationRecord: apiMocks.getDiarizationRecord,
    transcriptionRecordAudioUrl: apiMocks.transcriptionRecordAudioUrl,
    deleteTranscriptionRecord: apiMocks.deleteTranscriptionRecord,
    deleteDiarizationRecord: apiMocks.deleteDiarizationRecord,
    regenerateTranscriptionSummary: apiMocks.regenerateTranscriptionSummary,
    createTranscriptionRecord: apiMocks.createTranscriptionRecord,
    createTranscriptionRecordStream: apiMocks.createTranscriptionRecordStream,
    createDiarizationRecord: apiMocks.createDiarizationRecord,
  },
}));

vi.mock("@/features/models/hooks/useRouteModelSelection", () => ({
  useRouteModelSelection: hookMocks.useRouteModelSelection,
}));

vi.mock("@/features/models/components/RouteModelModal", () => ({
  RouteModelModal: componentMocks.routeModelModal,
}));

type TranscriptionPageTestProps = ComponentProps<typeof TranscriptionPage>;

const baseProps: TranscriptionPageTestProps = {
  models: [],
  selectedModel: null,
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

function renderRoute(initialEntry: string) {
  return render(
    <NotificationProvider>
      <MemoryRouter initialEntries={[initialEntry]}>
        <Routes>
          <Route
            path="/transcription"
            element={<TranscriptionPage {...baseProps} />}
          />
          <Route
            path="/transcription/:recordId"
            element={<TranscriptionPage {...baseProps} />}
          />
        </Routes>
      </MemoryRouter>
    </NotificationProvider>,
  );
}

function deferredPromise<T>() {
  let resolve!: (value: T) => void;
  let reject!: (reason?: unknown) => void;
  const promise = new Promise<T>((res, rej) => {
    resolve = res;
    reject = rej;
  });
  return { promise, resolve, reject };
}

describe("TranscriptionPage detail route", () => {
  beforeEach(() => {
    apiMocks.getTranscriptionRecord.mockReset();
    apiMocks.listTranscriptionRecords.mockReset();
    apiMocks.listSpeechTextJobPage.mockReset();
    apiMocks.getDiarizationRecord.mockReset();
    apiMocks.transcriptionRecordAudioUrl.mockReset();
    apiMocks.deleteTranscriptionRecord.mockReset();
    apiMocks.deleteDiarizationRecord.mockReset();
    apiMocks.regenerateTranscriptionSummary.mockReset();
    apiMocks.createTranscriptionRecord.mockReset();
    apiMocks.createTranscriptionRecordStream.mockReset();
    apiMocks.createDiarizationRecord.mockReset();
    hookMocks.useRouteModelSelection.mockReset();
    componentMocks.routeModelModal.mockClear();
    baseProps.models = [
      {
        variant: "Parakeet-TDT-0.6B-v3",
        status: "ready",
        local_path: "/models/parakeet",
        size_bytes: null,
        download_progress: null,
        error_message: null,
      },
      {
        variant: "diar_streaming_sortformer_4spk-v2.1",
        status: "ready",
        local_path: "/models/diar",
        size_bytes: null,
        download_progress: null,
        error_message: null,
      },
      {
        variant: "Whisper-Large-v3-Turbo",
        status: "ready",
        local_path: "/models/whisper",
        size_bytes: null,
        download_progress: null,
        error_message: null,
      },
      {
        variant: "Qwen3-ForcedAligner-0.6B",
        status: "ready",
        local_path: "/models/aligner",
        size_bytes: null,
        download_progress: null,
        error_message: null,
      },
      {
        variant: "Qwen3.5-4B",
        status: "ready",
        local_path: "/models/qwen",
        size_bytes: null,
        download_progress: null,
        error_message: null,
      },
    ];
    baseProps.selectedModel = "Parakeet-TDT-0.6B-v3";
    baseProps.onDownload = vi.fn();
    baseProps.onCancelDownload = vi.fn();
    baseProps.onLoad = vi.fn();
    baseProps.onUnload = vi.fn();
    baseProps.onDelete = vi.fn();
    baseProps.onSelect = vi.fn();
    baseProps.onError = vi.fn();

    apiMocks.transcriptionRecordAudioUrl.mockReturnValue("/audio/transcription.wav");
    apiMocks.listTranscriptionRecords.mockResolvedValue([]);
    apiMocks.listSpeechTextJobPage.mockImplementation(async () => ({
      items: await apiMocks.listTranscriptionRecords(),
      pagination: {
        next_cursor: null,
        has_more: false,
        limit: 25,
      },
    }));
    apiMocks.deleteTranscriptionRecord.mockResolvedValue(undefined);
    apiMocks.deleteDiarizationRecord.mockResolvedValue(undefined);
    apiMocks.createTranscriptionRecord.mockResolvedValue({
      id: "txr-created-1",
      created_at: 1,
      model_id: "Parakeet-TDT-0.6B-v3",
      aligner_model_id: null,
      language: "English",
      processing_status: "pending",
      processing_error: null,
      duration_secs: null,
      processing_time_ms: 0,
      rtf: null,
      audio_mime_type: "audio/wav",
      audio_filename: "clip.wav",
      transcription: "",
      segments: [],
      words: [],
      summary_status: "not_requested",
      summary_model_id: null,
      summary_text: null,
      summary_error: null,
      summary_updated_at: null,
    });
    apiMocks.createTranscriptionRecordStream.mockImplementation(
      (_request, callbacks) => {
        callbacks.onCreated?.({
          id: "txr-created-1",
          created_at: 1,
          model_id: "Parakeet-TDT-0.6B-v3",
          aligner_model_id: null,
          language: "English",
          processing_status: "pending",
          processing_error: null,
          duration_secs: null,
          processing_time_ms: 0,
          rtf: null,
          audio_mime_type: "audio/wav",
          audio_filename: "clip.wav",
          transcription: "",
          segments: [],
          words: [],
          summary_status: "not_requested",
          summary_model_id: null,
          summary_text: null,
          summary_error: null,
          summary_updated_at: null,
        });
        return new AbortController();
      },
    );
    apiMocks.createDiarizationRecord.mockResolvedValue({
      id: "diar-created-1",
    });
    hookMocks.useRouteModelSelection.mockReturnValue({
      routeModels: [],
      resolvedSelectedModel: "Parakeet-TDT-0.6B-v3",
      selectedModelReady: true,
      isModelModalOpen: false,
      intentVariant: null,
      closeModelModal: vi.fn(),
      openModelManager: vi.fn(),
      requestModel: vi.fn(),
      handleModelSelect: vi.fn(),
      modelOptions: [],
    });

    HTMLElement.prototype.scrollIntoView = vi.fn();
    vi.spyOn(window.HTMLMediaElement.prototype, "pause").mockImplementation(
      () => {},
    );
  });

  it("renders the transcription history table on /transcription", async () => {
    renderRoute("/transcription");

    await waitFor(() =>
      expect(apiMocks.listTranscriptionRecords).toHaveBeenCalled(),
    );

    expect(
      await screen.findByRole("heading", { name: "Transcription" }),
    ).toBeInTheDocument();
    expect(
      screen.getByRole("button", { name: /New transcript/i }),
    ).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /Models/i })).toBeInTheDocument();
    expect(
      screen.getByRole("heading", { name: "No speech-text jobs yet" }),
    ).toBeInTheDocument();
  });

  it("loads more transcription history rows", async () => {
    apiMocks.listSpeechTextJobPage.mockReset();
    apiMocks.listSpeechTextJobPage
      .mockResolvedValueOnce({
        items: [
          {
            id: "txr-page-1",
            created_at: 1,
            model_id: "Parakeet-TDT-0.6B-v3",
            aligner_model_id: null,
            language: "English",
            processing_status: "ready",
            processing_error: null,
            duration_secs: 4,
            processing_time_ms: 120,
            rtf: 0.5,
            audio_mime_type: "audio/wav",
            audio_filename: "page-one.wav",
            transcription_preview: "Page one preview.",
            transcription_chars: 17,
            summary_status: "not_requested",
            summary_preview: null,
            summary_chars: 0,
          },
        ],
        pagination: {
          next_cursor: "txr-cursor-2",
          has_more: true,
          limit: 25,
        },
      })
      .mockResolvedValueOnce({
        items: [
          {
            id: "txr-page-2",
            created_at: 2,
            model_id: "Parakeet-TDT-0.6B-v3",
            aligner_model_id: null,
            language: "English",
            processing_status: "ready",
            processing_error: null,
            duration_secs: 5,
            processing_time_ms: 100,
            rtf: 0.4,
            audio_mime_type: "audio/wav",
            audio_filename: "page-two.wav",
            transcription_preview: "Page two preview.",
            transcription_chars: 17,
            summary_status: "not_requested",
            summary_preview: null,
            summary_chars: 0,
          },
        ],
        pagination: {
          next_cursor: null,
          has_more: false,
          limit: 25,
        },
      });

    renderRoute("/transcription");

    expect(await screen.findByText("page-one.wav")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Load more" }));
    expect(await screen.findByText("page-two.wav")).toBeInTheDocument();
    expect(await screen.findByText("page-one.wav")).toBeInTheDocument();

    expect(apiMocks.listSpeechTextJobPage).toHaveBeenNthCalledWith(1, {
      limit: 25,
      cursor: null,
    });
    expect(apiMocks.listSpeechTextJobPage).toHaveBeenNthCalledWith(2, {
      limit: 25,
      cursor: "txr-cursor-2",
    });
  });

  it("keeps history rows visible while transcription polling refreshes in the background", async () => {
    const intervalCallbacks: Array<() => void> = [];
    const setIntervalSpy = vi
      .spyOn(window, "setInterval")
      .mockImplementation((handler) => {
        if (typeof handler === "function") {
          intervalCallbacks.push(handler);
        }
        return 1 as unknown as ReturnType<typeof window.setInterval>;
      });
    const clearIntervalSpy = vi
      .spyOn(window, "clearInterval")
      .mockImplementation(() => {});
    const backgroundRefresh =
      deferredPromise<Awaited<ReturnType<typeof apiMocks.listTranscriptionRecords>>>();
    apiMocks.listTranscriptionRecords
      .mockResolvedValueOnce([
        {
          id: "txr-history-polling-1",
          created_at: 1,
          model_id: "Parakeet-TDT-0.6B-v3",
          aligner_model_id: null,
          language: "English",
          processing_status: "processing",
          processing_error: null,
          duration_secs: null,
          processing_time_ms: null,
          rtf: null,
          audio_mime_type: "audio/wav",
          audio_filename: "meeting.wav",
          transcription_preview: "Still transcribing...",
          transcription_chars: 20,
          summary_status: "pending",
          summary_preview: null,
          summary_chars: 0,
          summary_model_id: "Qwen3.5-4B",
        },
      ])
      .mockImplementationOnce(() => backgroundRefresh.promise);

    const view = renderRoute("/transcription");

    try {
      expect(await screen.findByText("Still transcribing...")).toBeInTheDocument();

      await waitFor(() => expect(setIntervalSpy).toHaveBeenCalled());
      if (intervalCallbacks.length === 0) {
        throw new Error("Expected transcription history polling to register an interval.");
      }

      intervalCallbacks.forEach((callback) => callback());

      await waitFor(() =>
        expect(apiMocks.listTranscriptionRecords).toHaveBeenCalledTimes(2),
      );

      expect(screen.getByText("Still transcribing...")).toBeInTheDocument();
      expect(
        screen.queryByText("Loading speech-text history..."),
      ).not.toBeInTheDocument();

      backgroundRefresh.resolve([
        {
          id: "txr-history-polling-1",
          created_at: 1,
          model_id: "Parakeet-TDT-0.6B-v3",
          aligner_model_id: null,
          language: "English",
          processing_status: "ready",
          processing_error: null,
          duration_secs: 5,
          processing_time_ms: 120,
          rtf: 0.8,
          audio_mime_type: "audio/wav",
          audio_filename: "meeting.wav",
          transcription_preview: "Transcription complete.",
          transcription_chars: 23,
          summary_status: "ready",
          summary_preview: "Summary complete.",
          summary_chars: 17,
          summary_model_id: "Qwen3.5-4B",
        },
      ]);

      expect(await screen.findByText("Transcription complete.")).toBeInTheDocument();
    } finally {
      view.unmount();
      setIntervalSpy.mockRestore();
      clearIntervalSpy.mockRestore();
    }
  });

  it("shows standard row actions from the history menu", async () => {
    apiMocks.listTranscriptionRecords.mockResolvedValue([
      {
        id: "txr-history-1",
        created_at: 1,
        model_id: "Parakeet-TDT-0.6B-v3",
        language: "English",
        duration_secs: 4,
        processing_status: "ready",
        processing_error: null,
        processing_time_ms: 120,
        rtf: 0.5,
        audio_mime_type: "audio/wav",
        audio_filename: "meeting.wav",
        transcription_preview: "Hello there.",
        transcription_chars: 12,
        summary_status: "ready",
        summary_preview: "Short summary",
        summary_chars: 13,
      },
    ]);

    renderRoute("/transcription");

    expect(await screen.findByText("meeting.wav")).toBeInTheDocument();

    fireEvent.pointerDown(
      screen.getByRole("button", { name: /More actions for meeting\.wav/i }),
      { button: 0, ctrlKey: false },
    );

    expect(await screen.findByRole("menuitem", { name: /Open record/i })).toBeVisible();
    expect(screen.getByRole("menuitem", { name: /Copy transcript/i })).toBeVisible();
    expect(screen.getByRole("menuitem", { name: /^Export$/i })).toBeVisible();
    expect(screen.getByRole("menuitem", { name: /^Delete$/i })).toBeVisible();
  });

  it("opens the new transcript modal from the header action", async () => {
    renderRoute("/transcription");

    await waitFor(() =>
      expect(apiMocks.listTranscriptionRecords).toHaveBeenCalled(),
    );

    fireEvent.click(screen.getByRole("button", { name: /New transcript/i }));

    expect(
      await screen.findByRole("heading", { name: "New transcript" }),
    ).toBeInTheDocument();
    expect(screen.getByRole("radio", { name: "Transcription" })).toBeChecked();
    expect(screen.getByRole("radio", { name: "Diarization" })).not.toBeChecked();
    expect(screen.getByText("Bring in a recording")).toBeInTheDocument();
    expect(screen.getByText("Model readiness")).toBeInTheDocument();
    expect(screen.getByText("Upload audio")).toBeInTheDocument();
    expect(screen.getByText("Stream results")).toBeInTheDocument();
  });

  it("switches modal content when selecting diarization from the mode radios", async () => {
    renderRoute("/transcription");

    await waitFor(() =>
      expect(apiMocks.listTranscriptionRecords).toHaveBeenCalled(),
    );

    fireEvent.click(screen.getByRole("button", { name: /New transcript/i }));
    expect(
      await screen.findByRole("heading", { name: "New transcript" }),
    ).toBeInTheDocument();

    fireEvent.click(screen.getByRole("radio", { name: "Diarization" }));

    expect(
      await screen.findByRole("heading", { name: "New diarization" }),
    ).toBeInTheDocument();
    expect(screen.getByRole("radio", { name: "Diarization" })).toBeChecked();
    expect(screen.getByText("Choose how to start")).toBeInTheDocument();
    expect(screen.getByText("Model readiness")).toBeInTheDocument();
  });

  it("opens model manager from the modal readiness button without raising an error toast", async () => {
    const openModelManager = vi.fn();
    const requestModel = vi.fn();
    baseProps.selectedModel = null;
    hookMocks.useRouteModelSelection.mockReturnValue({
      routeModels: [],
      resolvedSelectedModel: null,
      selectedModelReady: false,
      isModelModalOpen: false,
      intentVariant: null,
      closeModelModal: vi.fn(),
      openModelManager,
      requestModel,
      handleModelSelect: vi.fn(),
      modelOptions: [],
    });

    renderRoute("/transcription");

    await waitFor(() =>
      expect(apiMocks.listTranscriptionRecords).toHaveBeenCalled(),
    );

    fireEvent.click(screen.getByRole("button", { name: /New transcript/i }));
    expect(
      await screen.findByRole("heading", { name: "New transcript" }),
    ).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: "Open ASR models" }));

    expect(openModelManager).toHaveBeenCalledTimes(1);
    expect(requestModel).not.toHaveBeenCalled();
    expect(baseProps.onError).not.toHaveBeenCalled();
  });

  it("raises the model modal above the new transcript modal", async () => {
    hookMocks.useRouteModelSelection.mockReturnValue({
      routeModels: [],
      resolvedSelectedModel: "Parakeet-TDT-0.6B-v3",
      selectedModelReady: true,
      isModelModalOpen: true,
      intentVariant: null,
      closeModelModal: vi.fn(),
      openModelManager: vi.fn(),
      requestModel: vi.fn(),
      handleModelSelect: vi.fn(),
      modelOptions: [],
    });

    renderRoute("/transcription");

    await waitFor(() =>
      expect(apiMocks.listTranscriptionRecords).toHaveBeenCalled(),
    );

    expect(screen.getByTestId("route-model-modal")).toHaveAttribute(
      "data-z-index",
      "z-50",
    );

    fireEvent.click(screen.getByRole("button", { name: /New transcript/i }));

    expect(
      await screen.findByRole("heading", { name: "New transcript" }),
    ).toBeInTheDocument();
    expect(screen.getByTestId("route-model-modal")).toHaveAttribute(
      "data-z-index",
      "z-[70]",
    );
  });

  it("keeps the new transcript modal open while interacting with the stacked model modal", async () => {
    hookMocks.useRouteModelSelection.mockReturnValue({
      routeModels: [],
      resolvedSelectedModel: "Parakeet-TDT-0.6B-v3",
      selectedModelReady: false,
      isModelModalOpen: true,
      intentVariant: null,
      closeModelModal: vi.fn(),
      openModelManager: vi.fn(),
      requestModel: vi.fn(),
      handleModelSelect: vi.fn(),
      modelOptions: [],
    });

    renderRoute("/transcription");

    await waitFor(() =>
      expect(apiMocks.listTranscriptionRecords).toHaveBeenCalled(),
    );

    fireEvent.click(screen.getByRole("button", { name: /New transcript/i }));

    expect(
      await screen.findByRole("heading", { name: "New transcript" }),
    ).toBeInTheDocument();

    const topModalButton =
      screen.getByTestId("route-model-modal").querySelector("button");
    expect(topModalButton).not.toBeNull();

    fireEvent.click(topModalButton!);

    expect(baseProps.onSelect).toHaveBeenCalledWith("Parakeet-TDT-0.6B-v3");
    expect(
      screen.getByRole("heading", { name: "New transcript" }),
    ).toBeInTheDocument();
  });

  it("redirects to /transcription/:id after an upload creates a record", async () => {
    apiMocks.getTranscriptionRecord.mockResolvedValue({
      id: "txr-created-1",
      created_at: 1,
      model_id: "Parakeet-TDT-0.6B-v3",
      aligner_model_id: null,
      language: "English",
      processing_status: "processing",
      processing_error: null,
      duration_secs: null,
      processing_time_ms: 0,
      rtf: null,
      audio_mime_type: "audio/wav",
      audio_filename: "clip.wav",
      transcription: "",
      segments: [],
      words: [],
      summary_status: "not_requested",
      summary_model_id: null,
      summary_text: null,
      summary_error: null,
      summary_updated_at: null,
    });

    renderRoute("/transcription");

    await waitFor(() =>
      expect(apiMocks.listTranscriptionRecords).toHaveBeenCalled(),
    );

    fireEvent.click(screen.getByRole("button", { name: /New transcript/i }));

    const fileInput = document.querySelector(
      'input[type="file"]',
    ) as HTMLInputElement | null;
    expect(fileInput).not.toBeNull();

    fireEvent.change(fileInput!, {
      target: {
        files: [new File(["audio"], "clip.wav", { type: "audio/wav" })],
      },
    });

    await waitFor(() =>
      expect(apiMocks.createTranscriptionRecordStream).toHaveBeenCalled(),
    );
    await waitFor(() =>
      expect(apiMocks.getTranscriptionRecord).toHaveBeenCalledWith(
        "txr-created-1",
      ),
    );

    expect(
      await screen.findByRole("heading", { name: "Transcription Record" }),
    ).toBeInTheDocument();
  });

  it("creates a diarization record when diarization mode is selected in the modal", async () => {
    apiMocks.getTranscriptionRecord.mockResolvedValue({
      id: "diar-created-1",
      created_at: 1,
      model_id: "Parakeet-TDT-0.6B-v3",
      aligner_model_id: null,
      language: "English",
      processing_status: "processing",
      processing_error: null,
      duration_secs: null,
      processing_time_ms: 0,
      rtf: null,
      audio_mime_type: "audio/wav",
      audio_filename: "meeting.wav",
      transcription: "",
      segments: [],
      words: [],
      summary_status: "not_requested",
      summary_model_id: null,
      summary_text: null,
      summary_error: null,
      summary_updated_at: null,
    });

    renderRoute("/transcription");

    await waitFor(() =>
      expect(apiMocks.listTranscriptionRecords).toHaveBeenCalled(),
    );

    fireEvent.click(screen.getByRole("button", { name: /New transcript/i }));
    fireEvent.click(await screen.findByRole("radio", { name: "Diarization" }));

    await screen.findByRole("heading", { name: "New diarization" });
    const uploadButtons = screen.getAllByRole("button", {
      name: "Upload audio file",
    });
    const activeUploadButton = uploadButtons[uploadButtons.length - 1];
    const fileInput = activeUploadButton?.querySelector(
      'input[type="file"]',
    ) as HTMLInputElement | null;
    expect(fileInput).not.toBeNull();

    fireEvent.change(fileInput!, {
      target: {
        files: [new File(["audio"], "meeting.wav", { type: "audio/wav" })],
      },
    });

    await waitFor(() =>
      expect(apiMocks.createDiarizationRecord).toHaveBeenCalledWith(
        expect.objectContaining({
          audio_filename: "meeting.wav",
          model_id: "diar_streaming_sortformer_4spk-v2.1",
          asr_model_id: "Whisper-Large-v3-Turbo",
          aligner_model_id: "Qwen3-ForcedAligner-0.6B",
          llm_model_id: "Qwen3.5-4B",
        }),
      ),
    );
    expect(apiMocks.createTranscriptionRecordStream).not.toHaveBeenCalled();
  });

  it("shows streamed transcript deltas on the detail page while processing", async () => {
    let streamCallbacks:
      | {
          onStart?: () => void;
          onDelta?: (delta: string) => void;
          onFinal?: (record: unknown) => void;
        }
      | undefined;

    apiMocks.createTranscriptionRecordStream.mockImplementationOnce(
      (_request, callbacks) => {
        streamCallbacks = callbacks;
        callbacks.onCreated?.({
          id: "txr-stream-1",
          created_at: 1,
          model_id: "Parakeet-TDT-0.6B-v3",
          aligner_model_id: null,
          language: "English",
          processing_status: "pending",
          processing_error: null,
          duration_secs: null,
          processing_time_ms: 0,
          rtf: null,
          audio_mime_type: "audio/wav",
          audio_filename: "streamed.wav",
          transcription: "",
          segments: [],
          words: [],
          summary_status: "not_requested",
          summary_model_id: null,
          summary_text: null,
          summary_error: null,
          summary_updated_at: null,
        });
        return new AbortController();
      },
    );
    apiMocks.getTranscriptionRecord.mockResolvedValue({
      id: "txr-stream-1",
      created_at: 1,
      model_id: "Parakeet-TDT-0.6B-v3",
      aligner_model_id: null,
      language: "English",
      processing_status: "processing",
      processing_error: null,
      duration_secs: null,
      processing_time_ms: 0,
      rtf: null,
      audio_mime_type: "audio/wav",
      audio_filename: "streamed.wav",
      transcription: "",
      segments: [],
      words: [],
      summary_status: "not_requested",
      summary_model_id: null,
      summary_text: null,
      summary_error: null,
      summary_updated_at: null,
    });

    renderRoute("/transcription");

    await waitFor(() =>
      expect(apiMocks.listTranscriptionRecords).toHaveBeenCalled(),
    );

    fireEvent.click(screen.getByRole("button", { name: /New transcript/i }));

    const fileInput = document.querySelector(
      'input[type="file"]',
    ) as HTMLInputElement | null;
    expect(fileInput).not.toBeNull();

    fireEvent.change(fileInput!, {
      target: {
        files: [new File(["audio"], "streamed.wav", { type: "audio/wav" })],
      },
    });

    expect(
      await screen.findByRole("heading", { name: "Transcription Record" }),
    ).toBeInTheDocument();

    await act(async () => {
      streamCallbacks?.onStart?.();
      streamCallbacks?.onDelta?.("Hello ");
      streamCallbacks?.onDelta?.("world");
    });

    expect(screen.getByText("Hello world")).toBeInTheDocument();
  });

  it("refreshes transcription history after creating a record", async () => {
    apiMocks.listTranscriptionRecords
      .mockResolvedValueOnce([])
      .mockResolvedValueOnce([
        {
          id: "txr-created-1",
          created_at: 1,
          audio_filename: "clip.wav",
          duration_secs: null,
          processing_status: "pending",
          processing_error: null,
          transcription_preview: "",
          summary_status: "not_requested",
          summary_preview: null,
        },
      ]);
    apiMocks.getTranscriptionRecord.mockResolvedValue({
      id: "txr-created-1",
      created_at: 1,
      model_id: "Parakeet-TDT-0.6B-v3",
      aligner_model_id: null,
      language: "English",
      processing_status: "processing",
      processing_error: null,
      duration_secs: null,
      processing_time_ms: 0,
      rtf: null,
      audio_mime_type: "audio/wav",
      audio_filename: "clip.wav",
      transcription: "",
      segments: [],
      words: [],
      summary_status: "not_requested",
      summary_model_id: null,
      summary_text: null,
      summary_error: null,
      summary_updated_at: null,
    });

    renderRoute("/transcription");

    await waitFor(() =>
      expect(apiMocks.listTranscriptionRecords).toHaveBeenCalledTimes(1),
    );

    fireEvent.click(screen.getByRole("button", { name: /New transcript/i }));

    const fileInput = document.querySelector(
      'input[type="file"]',
    ) as HTMLInputElement | null;
    expect(fileInput).not.toBeNull();

    fireEvent.change(fileInput!, {
      target: {
        files: [new File(["audio"], "clip.wav", { type: "audio/wav" })],
      },
    });

    await waitFor(() =>
      expect(apiMocks.listTranscriptionRecords).toHaveBeenCalledTimes(2),
    );
    expect(
      await screen.findByRole("heading", { name: "Transcription Record" }),
    ).toBeInTheDocument();

    fireEvent.click(
      screen.getByRole("button", { name: /Back to transcriptions/i }),
    );

    expect(await screen.findByText("clip.wav")).toBeInTheDocument();
    expect(
      screen.queryByRole("heading", { name: "No speech-text jobs yet" }),
    ).not.toBeInTheDocument();
  });

  it("accepts drag and drop on the upload area", async () => {
    renderRoute("/transcription");

    await waitFor(() =>
      expect(apiMocks.listTranscriptionRecords).toHaveBeenCalled(),
    );

    fireEvent.click(screen.getByRole("button", { name: /New transcript/i }));

    const uploadArea = await screen.findByRole("button", {
      name: "Upload audio file",
    });
    const file = new File(["audio"], "dragged-clip.wav", {
      type: "audio/wav",
    });

    fireEvent.dragOver(uploadArea, {
      dataTransfer: { files: [file] },
    });
    fireEvent.drop(uploadArea, {
      dataTransfer: { files: [file] },
    });

    await waitFor(() =>
      expect(apiMocks.createTranscriptionRecordStream).toHaveBeenCalledWith(
        expect.objectContaining({
          audio_file: file,
          audio_filename: "dragged-clip.wav",
        }),
        expect.any(Object),
      ),
    );
  });

  it("loads an existing record directly from /transcription/:id", async () => {
    apiMocks.getTranscriptionRecord.mockResolvedValue({
      id: "txr-route-1",
      created_at: 1,
      model_id: "Parakeet-TDT-0.6B-v3",
      aligner_model_id: null,
      language: "English",
      processing_status: "ready",
      processing_error: null,
      duration_secs: 4,
      processing_time_ms: 120,
      rtf: 0.5,
      audio_mime_type: "audio/wav",
      audio_filename: "meeting.wav",
      transcription: "Hello there.",
      segments: [],
      words: [],
      summary_status: "not_requested",
      summary_model_id: null,
      summary_text: null,
      summary_error: null,
      summary_updated_at: null,
    });

    renderRoute("/transcription/txr-route-1");

    await waitFor(() =>
      expect(apiMocks.getTranscriptionRecord).toHaveBeenCalledWith(
        "txr-route-1",
      ),
    );

    expect(
      await screen.findByRole("heading", { name: "Transcription Record" }),
    ).toBeInTheDocument();
    expect(
      screen.getByRole("heading", { name: "meeting.wav" }),
    ).toBeInTheDocument();
    expect(screen.getByText("Hello there.")).toBeInTheDocument();
    expect(
      screen.getByRole("button", { name: /Back to transcriptions/i }),
    ).toBeInTheDocument();
    expect(screen.queryByText(/^Ready$/)).not.toBeInTheDocument();
    expect(screen.getByTestId("transcription-review-player")).toHaveClass(
      "fixed",
    );
  });

  it("confirms deletion before removing a transcription record", async () => {
    apiMocks.listTranscriptionRecords
      .mockResolvedValueOnce([
        {
          id: "txr-delete-1",
          created_at: 1,
          audio_filename: "meeting.wav",
          duration_secs: 4,
          processing_status: "ready",
          processing_error: null,
          transcription_preview: "Hello there.",
          summary_status: "not_requested",
          summary_preview: null,
        },
      ])
      .mockResolvedValueOnce([]);
    apiMocks.getTranscriptionRecord.mockResolvedValue({
      id: "txr-delete-1",
      created_at: 1,
      model_id: "Parakeet-TDT-0.6B-v3",
      aligner_model_id: null,
      language: "English",
      processing_status: "ready",
      processing_error: null,
      duration_secs: 4,
      processing_time_ms: 120,
      rtf: 0.5,
      audio_mime_type: "audio/wav",
      audio_filename: "meeting.wav",
      transcription: "Hello there.",
      segments: [],
      words: [],
      summary_status: "not_requested",
      summary_model_id: null,
      summary_text: null,
      summary_error: null,
      summary_updated_at: null,
    });

    renderRoute("/transcription/txr-delete-1");

    expect(
      await screen.findByRole("heading", { name: "meeting.wav" }),
    ).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: /^Delete$/i }));

    expect(
      await screen.findByText(
        /This permanently removes the saved audio and transcript from history\./i,
      ),
    ).toBeInTheDocument();
    expect(screen.getAllByText("meeting.wav").length).toBeGreaterThan(0);
    expect(apiMocks.deleteTranscriptionRecord).not.toHaveBeenCalled();

    fireEvent.click(
      screen.getByRole("button", { name: "Delete transcription" }),
    );

    await waitFor(() =>
      expect(apiMocks.deleteTranscriptionRecord).toHaveBeenCalledWith(
        "txr-delete-1",
      ),
    );
    await waitFor(() =>
      expect(apiMocks.listTranscriptionRecords).toHaveBeenCalledTimes(2),
    );

    expect(
      await screen.findByRole("heading", { name: "Transcription" }),
    ).toBeInTheDocument();
    expect(
      screen.queryByLabelText("Open transcription meeting.wav"),
    ).not.toBeInTheDocument();
  });

  it("confirms deletion from the history menu and refreshes the table", async () => {
    apiMocks.listTranscriptionRecords
      .mockResolvedValueOnce([
        {
          id: "txr-history-delete-1",
          created_at: 1,
          model_id: "Parakeet-TDT-0.6B-v3",
          language: "English",
          duration_secs: 4,
          processing_status: "ready",
          processing_error: null,
          processing_time_ms: 120,
          rtf: 0.5,
          audio_mime_type: "audio/wav",
          audio_filename: "meeting.wav",
          transcription_preview: "Hello there.",
          transcription_chars: 12,
          summary_status: "not_requested",
          summary_preview: null,
          summary_chars: 0,
        },
      ])
      .mockResolvedValueOnce([]);

    renderRoute("/transcription");

    expect(await screen.findByText("meeting.wav")).toBeInTheDocument();

    fireEvent.pointerDown(
      screen.getByRole("button", { name: /More actions for meeting\.wav/i }),
      { button: 0, ctrlKey: false },
    );
    fireEvent.click(await screen.findByRole("menuitem", { name: /^Delete$/i }));

    expect(
      await screen.findByText(
        /This permanently removes the saved audio and transcript from history\./i,
      ),
    ).toBeInTheDocument();
    expect(apiMocks.deleteTranscriptionRecord).not.toHaveBeenCalled();

    fireEvent.click(
      screen.getByRole("button", { name: "Delete transcription" }),
    );

    await waitFor(() =>
      expect(apiMocks.deleteTranscriptionRecord).toHaveBeenCalledWith(
        "txr-history-delete-1",
      ),
    );
    await waitFor(() =>
      expect(apiMocks.listTranscriptionRecords).toHaveBeenCalledTimes(2),
    );

    expect(
      screen.queryByLabelText("Open transcription meeting.wav"),
    ).not.toBeInTheDocument();
    expect(
      screen.getByRole("heading", { name: "No speech-text jobs yet" }),
    ).toBeInTheDocument();
  });

  it("opens diarization rows through the merged transcription route mode", async () => {
    apiMocks.listSpeechTextJobPage.mockResolvedValueOnce({
      items: [
        {
          id: "diar-route-1",
          kind: "diarization",
          created_at: 1,
          model_id: "nvidia-sortformer",
          processing_status: "ready",
          processing_error: null,
          duration_secs: 4,
          processing_time_ms: 120,
          rtf: 0.5,
          audio_mime_type: "audio/wav",
          audio_filename: "diar.wav",
          speaker_count: 2,
          corrected_speaker_count: 2,
          transcript_preview: "SPEAKER_00: Hello there.",
          transcript_chars: 23,
          summary_status: "not_requested",
          summary_preview: null,
          summary_chars: 0,
        },
      ],
      pagination: {
        next_cursor: null,
        has_more: false,
        limit: 25,
      },
    });

    renderRoute("/transcription");

    expect(await screen.findByText("diar.wav")).toBeInTheDocument();
    fireEvent.click(screen.getByLabelText("Open diarization diar.wav"));

    await waitFor(() =>
      expect(apiMocks.getTranscriptionRecord).toHaveBeenCalledWith("diar-route-1"),
    );
  });

  it("keeps the current detail view visible while polling in the background", async () => {
    const backgroundRefresh =
      deferredPromise<Awaited<ReturnType<typeof apiMocks.getTranscriptionRecord>>>();
    const intervalCallbacks: Array<() => void> = [];
    const setIntervalSpy = vi
      .spyOn(window, "setInterval")
      .mockImplementation((handler) => {
        if (typeof handler === "function") {
          intervalCallbacks.push(handler);
        }
        return 1 as unknown as ReturnType<typeof window.setInterval>;
      });
    const clearIntervalSpy = vi
      .spyOn(window, "clearInterval")
      .mockImplementation(() => {});

    try {
      apiMocks.getTranscriptionRecord
        .mockResolvedValueOnce({
          id: "txr-polling-1",
          created_at: 1,
          model_id: "Parakeet-TDT-0.6B-v3",
          aligner_model_id: null,
          language: "English",
          processing_status: "processing",
          processing_error: null,
          duration_secs: 4,
          processing_time_ms: 120,
          rtf: 0.5,
          audio_mime_type: "audio/wav",
          audio_filename: "meeting.wav",
          transcription: "",
          segments: [],
          words: [],
          summary_status: "not_requested",
          summary_model_id: null,
          summary_text: null,
          summary_error: null,
          summary_updated_at: null,
        })
        .mockImplementationOnce(() => backgroundRefresh.promise);

      renderRoute("/transcription/txr-polling-1");

      expect(
        await screen.findByRole("heading", { name: "meeting.wav" }),
      ).toBeInTheDocument();
      expect(screen.getByText("Transcription in progress")).toBeInTheDocument();
      expect(screen.queryByText("Loading transcript...")).not.toBeInTheDocument();

      await waitFor(() => expect(setIntervalSpy).toHaveBeenCalled());
      if (intervalCallbacks.length === 0) {
        throw new Error("Expected transcription polling to register an interval.");
      }

      intervalCallbacks.forEach((callback) => callback());

      await waitFor(() =>
        expect(apiMocks.getTranscriptionRecord).toHaveBeenCalledTimes(2),
      );

      expect(screen.getByText("Transcription in progress")).toBeInTheDocument();
      expect(screen.queryByText("Loading transcript...")).not.toBeInTheDocument();

      backgroundRefresh.resolve({
        id: "txr-polling-1",
        created_at: 1,
        model_id: "Parakeet-TDT-0.6B-v3",
        aligner_model_id: null,
        language: "English",
        processing_status: "ready",
        processing_error: null,
        duration_secs: 4,
        processing_time_ms: 120,
        rtf: 0.5,
        audio_mime_type: "audio/wav",
        audio_filename: "meeting.wav",
        transcription: "Hello there.",
        segments: [],
        words: [],
        summary_status: "not_requested",
        summary_model_id: null,
        summary_text: null,
        summary_error: null,
        summary_updated_at: null,
      });

      expect(await screen.findByText("Hello there.")).toBeInTheDocument();
    } finally {
      setIntervalSpy.mockRestore();
      clearIntervalSpy.mockRestore();
    }
  });

  it("shows route-level load errors for missing records", async () => {
    apiMocks.getTranscriptionRecord.mockRejectedValue(
      new Error("Transcription record not found"),
    );

    renderRoute("/transcription/missing");

    await waitFor(() =>
      expect(apiMocks.getTranscriptionRecord).toHaveBeenCalledWith("missing"),
    );

    expect(
      await screen.findByText("Transcription record not found"),
    ).toBeInTheDocument();
  });
});
