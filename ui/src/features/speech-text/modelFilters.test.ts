import { describe, expect, it } from "vitest";

import type { ModelInfo } from "@/api";
import {
  installedGraniteSpeech5Model,
  isDiarizationPipelineAsrModel,
  isDiarizationPipelineLlmModel,
  isOneModelDiarizationModel,
  isSpeakerAttributedAsrModel,
  isTranscriptionSummaryModel,
  resolveDiarizationStackRole,
} from "@/features/speech-text/modelFilters";

function model(variant: string, status: ModelInfo["status"] = "ready"): ModelInfo {
  return {
    variant,
    status,
    local_path: `/models/${variant}`,
    size_bytes: null,
    download_progress: null,
    error_message: null,
  };
}

describe("resolveDiarizationStackRole", () => {
  const preferredAsr = ["Whisper-Large-v3-Turbo"] as const;

  function resolveAsr(models: ModelInfo[], pick: string | null) {
    return resolveDiarizationStackRole({
      models,
      eligibleModels: models.filter((candidate) => candidate.variant !== "Qwen3.5-4B"),
      pick,
      preferredVariants: preferredAsr,
    });
  }

  it("uses a loaded model over a downloaded preferred default", () => {
    // The reported bug: Parakeet (or Granite 5.0) loaded, Whisper only on
    // disk. The stack must count the loaded model, not demand Whisper.
    expect(
      resolveAsr(
        [
          model("Whisper-Large-v3-Turbo", "downloaded"),
          model("Granite-Speech-5.0-2B-GGUF"),
        ],
        null,
      ),
    ).toEqual({ variant: "Granite-Speech-5.0-2B-GGUF", ready: true, missing: false });
  });

  it("prefers the loaded default over other loaded models", () => {
    expect(
      resolveAsr(
        [model("Parakeet-TDT-0.6B-v3"), model("Whisper-Large-v3-Turbo")],
        null,
      ).variant,
    ).toBe("Whisper-Large-v3-Turbo");
  });

  it("falls back to the preferred default when nothing is loaded", () => {
    expect(
      resolveAsr(
        [
          model("Parakeet-TDT-0.6B-v3", "downloaded"),
          model("Whisper-Large-v3-Turbo", "not_downloaded"),
        ],
        null,
      ),
    ).toEqual({ variant: "Whisper-Large-v3-Turbo", ready: false, missing: false });
  });

  it("honours the pick among several loaded models", () => {
    expect(
      resolveAsr(
        [
          model("Parakeet-TDT-0.6B-v3"),
          model("Whisper-Large-v3-Turbo"),
          model("Granite-Speech-5.0-2B-GGUF"),
        ],
        "Parakeet-TDT-0.6B-v3",
      ),
    ).toEqual({ variant: "Parakeet-TDT-0.6B-v3", ready: true, missing: false });
  });

  it("keeps a pick that is not loaded rather than swapping in a loaded model", () => {
    expect(
      resolveAsr(
        [model("Parakeet-TDT-0.6B-v3", "downloaded"), model("Whisper-Large-v3-Turbo")],
        "Parakeet-TDT-0.6B-v3",
      ),
    ).toEqual({ variant: "Parakeet-TDT-0.6B-v3", ready: false, missing: false });
  });

  it("surfaces a pick that left the catalog as missing", () => {
    expect(
      resolveAsr([model("Whisper-Large-v3-Turbo")], "Removed-ASR"),
    ).toEqual({ variant: "Removed-ASR", ready: false, missing: true });
  });

  it("ignores a pick that is no longer eligible for the role", () => {
    expect(
      resolveAsr([model("Whisper-Large-v3-Turbo"), model("Qwen3.5-4B")], "Qwen3.5-4B")
        .variant,
    ).toBe("Whisper-Large-v3-Turbo");
  });

  it("reports no model when none is eligible", () => {
    expect(resolveAsr([], null)).toEqual({ variant: null, ready: false, missing: false });
  });

  it("uses a pick-only model only when it is the user's pick", () => {
    const sortformer = model("diar_streaming_sortformer_4spk-v2.1", "downloaded");
    const transcribe = model("Nemotron-3.5-Transcribe-30B-A4B-FP8");
    const resolveDiarizer = (pick: string | null) =>
      resolveDiarizationStackRole({
        models: [sortformer, transcribe],
        eligibleModels: [sortformer],
        pickOnlyModels: [transcribe],
        pick,
        preferredVariants: ["diar_streaming_sortformer_4spk-v2.1"],
      });

    // Loaded, it still never becomes the default diarizer.
    expect(resolveDiarizer(null)).toEqual({
      variant: "diar_streaming_sortformer_4spk-v2.1",
      ready: false,
      missing: false,
    });
    expect(resolveDiarizer("Nemotron-3.5-Transcribe-30B-A4B-FP8")).toEqual({
      variant: "Nemotron-3.5-Transcribe-30B-A4B-FP8",
      ready: true,
      missing: false,
    });
  });
});

describe("isDiarizationPipelineAsrModel", () => {
  function withCapabilities(
    variant: string,
    capabilities: Partial<NonNullable<ModelInfo["route_capabilities"]>>,
  ): ModelInfo {
    return {
      ...model(variant),
      route_capabilities: {
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
        ...capabilities,
      },
    };
  }

  it("offers ASR models the server marks as pipeline ASR", () => {
    expect(
      isDiarizationPipelineAsrModel(
        withCapabilities("Parakeet-TDT-0.6B-v3", {
          openai_audio_transcriptions: true,
          speech_to_text_jobs: true,
          diarization_pipeline_asr: true,
        }),
      ),
    ).toBe(true);
  });

  it("selects by the server's capability, not by transcription routes", () => {
    // Serving transcription jobs is not enough: only the server says
    // whether the pipeline may use a model as its ASR stage.
    expect(
      isDiarizationPipelineAsrModel(
        withCapabilities("Parakeet-TDT-0.6B-v3", {
          openai_audio_transcriptions: true,
          speech_to_text_jobs: true,
        }),
      ),
    ).toBe(false);
  });

  it("hides audio-chat models the pipeline cannot run", () => {
    expect(
      isDiarizationPipelineAsrModel(
        withCapabilities("LFM2.5-Audio-1.5B-GGUF", {
          openai_audio_transcriptions: true,
          voice_realtime_unified: true,
        }),
      ),
    ).toBe(false);
  });

  it("offers LFM3 Audio, which runs speech-to-text jobs, as the pipeline ASR", () => {
    for (const variant of ["LFM3-Audio-2B-GGUF", "LFM3-Audio-2B-Q8_0-GGUF"]) {
      expect(
        isDiarizationPipelineAsrModel(
          withCapabilities(variant, {
            openai_audio_transcriptions: true,
            openai_audio_speech: true,
            speech_to_text_jobs: true,
            diarization_pipeline_asr: true,
            text_to_speech_records: true,
            voice_realtime_unified: true,
          }),
        ),
      ).toBe(true);
    }
  });

  it("offers Granite Speech 5.0 and its GGUF as the pipeline ASR", () => {
    for (const variant of ["Granite-Speech-5.0-2B", "Granite-Speech-5.0-2B-GGUF"]) {
      expect(
        isDiarizationPipelineAsrModel(
          withCapabilities(variant, {
            openai_audio_transcriptions: true,
            speech_to_text_jobs: true,
            diarization_pipeline_asr: true,
          }),
        ),
      ).toBe(true);
    }
  });

  it("offers speaker-attributed ASR models the server allows as the pipeline ASR", () => {
    const plus = withCapabilities("Granite-Speech-4.1-2B-Plus", {
      openai_audio_transcriptions: true,
      speech_to_text_jobs: true,
      diarization_pipeline_asr: true,
      speaker_attributed_asr: true,
    });
    expect(isSpeakerAttributedAsrModel(plus)).toBe(true);
    expect(isDiarizationPipelineAsrModel(plus)).toBe(true);
  });

  it("keeps Nemotron 3.5 Transcribe out of the pipeline but in speaker-attributed ASR", () => {
    // It tags speakers itself and the server refuses it as the pipeline ASR.
    const transcribe = withCapabilities("Nemotron-3.5-Transcribe-30B-A4B-FP8", {
      openai_audio_transcriptions: true,
      speech_to_text_jobs: true,
      diarization_pipeline_asr: false,
      speaker_attributed_asr: true,
      speech_translation: true,
      keyword_biasing: true,
      native_word_timestamps: true,
    });
    expect(isDiarizationPipelineAsrModel(transcribe)).toBe(false);
    expect(isSpeakerAttributedAsrModel(transcribe)).toBe(true);
  });

  it("offers one-model diarization only to models the server marks", () => {
    const transcribe = withCapabilities("Nemotron-3.5-Transcribe-30B-A4B-FP8", {
      openai_audio_transcriptions: true,
      speech_to_text_jobs: true,
      speaker_attributed_asr: true,
      native_word_timestamps: true,
      one_model_diarization: true,
    });
    const plus = withCapabilities("Granite-Speech-4.1-2B-Plus", {
      openai_audio_transcriptions: true,
      speech_to_text_jobs: true,
      diarization_pipeline_asr: true,
      speaker_attributed_asr: true,
    });
    const sortformer = withCapabilities("diar_streaming_sortformer_4spk-v2.1", {
      diarization_records: true,
    });
    expect(isOneModelDiarizationModel(transcribe)).toBe(true);
    expect(isOneModelDiarizationModel(plus)).toBe(false);
    expect(isOneModelDiarizationModel(sortformer)).toBe(false);
  });

  it("runs summaries on chat models, which LFM3 Audio is not", () => {
    const textChat = withCapabilities("Qwen3.5-4B", {
      openai_chat_completions: true,
    });
    // LFM3-Audio transcribes, speaks and talks, but serves no chat completions.
    const lfm3Audio = withCapabilities("LFM3-Audio-2B-GGUF", {
      openai_audio_transcriptions: true,
      openai_audio_speech: true,
      speech_to_text_jobs: true,
      text_to_speech_records: true,
      voice_realtime_unified: true,
    });
    expect(isTranscriptionSummaryModel(textChat)).toBe(true);
    expect(isDiarizationPipelineLlmModel(textChat)).toBe(true);
    expect(isTranscriptionSummaryModel(lfm3Audio)).toBe(false);
    expect(isDiarizationPipelineLlmModel(lfm3Audio)).toBe(false);
  });

  it("finds an installed Granite Speech 5.0 for the speaker-attribution hint", () => {
    const asr = { openai_audio_transcriptions: true, speech_to_text_jobs: true };
    expect(
      installedGraniteSpeech5Model([
        { ...withCapabilities("Granite-Speech-5.0-2B", asr), status: "not_downloaded" },
        withCapabilities("Granite-Speech-5.0-2B-GGUF", asr),
      ])?.variant,
    ).toBe("Granite-Speech-5.0-2B-GGUF");
    expect(
      installedGraniteSpeech5Model([
        { ...withCapabilities("Granite-Speech-5.0-2B", asr), status: "not_downloaded" },
      ]),
    ).toBeNull();
  });
});

