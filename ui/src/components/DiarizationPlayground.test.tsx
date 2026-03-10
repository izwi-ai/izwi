import { fireEvent, render, waitFor, within } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { DiarizationPlayground } from "./DiarizationPlayground";

function activateTab(scope: ReturnType<typeof within>, name: string): void {
  const tab = scope.getByRole("tab", { name });
  fireEvent.mouseDown(tab);
  fireEvent.click(tab);
}

const apiMocks = vi.hoisted(() => ({
  createDiarizationRecord: vi.fn(),
  updateDiarizationRecord: vi.fn(),
  listDiarizationRecords: vi.fn(),
  getDiarizationRecord: vi.fn(),
  deleteDiarizationRecord: vi.fn(),
  diarizationRecordAudioUrl: vi.fn(),
}));

vi.mock("../api", () => ({
  api: {
    createDiarizationRecord: apiMocks.createDiarizationRecord,
    updateDiarizationRecord: apiMocks.updateDiarizationRecord,
    listDiarizationRecords: apiMocks.listDiarizationRecords,
    getDiarizationRecord: apiMocks.getDiarizationRecord,
    deleteDiarizationRecord: apiMocks.deleteDiarizationRecord,
    diarizationRecordAudioUrl: apiMocks.diarizationRecordAudioUrl,
  },
}));

describe("DiarizationPlayground speaker corrections", () => {
  beforeEach(() => {
    apiMocks.createDiarizationRecord.mockReset();
    apiMocks.updateDiarizationRecord.mockReset();
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

    const fileInput = container.querySelector<HTMLInputElement>(
      'input[type="file"]',
    );
    expect(fileInput).not.toBeNull();

    fireEvent.change(fileInput!, {
      target: {
        files: [new File(["sample"], "meeting.wav", { type: "audio/wav" })],
      },
    });

    expect(await scope.findByText("SPEAKER_00")).toBeInTheDocument();

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
    expect(await scope.findByText("Alice")).toBeInTheDocument();
  });
});
