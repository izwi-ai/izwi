//! Runtime pipeline graph contracts for multi-stage inference flows.
//!
//! Phase 7 defines reusable graph shapes for voice turns and diarization while
//! leaving existing route/runtime orchestration in place.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum PipelineKind {
    ModularVoiceTurn,
    UnifiedVoiceTurn,
    DiarizationTranscript,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum PipelineStageKind {
    DecodeAudio,
    Vad,
    Endpointing,
    Asr,
    Chat,
    Tts,
    AudioChat,
    Diarization,
    ForcedAlignment,
    SpeakerAttribution,
    LlmRefinement,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct PipelineStage {
    pub(crate) kind: PipelineStageKind,
    pub(crate) name: &'static str,
    pub(crate) required: bool,
}

impl PipelineStage {
    const fn required(kind: PipelineStageKind, name: &'static str) -> Self {
        Self {
            kind,
            name,
            required: true,
        }
    }

    const fn optional(kind: PipelineStageKind, name: &'static str) -> Self {
        Self {
            kind,
            name,
            required: false,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct PipelineGraph {
    pub(crate) kind: PipelineKind,
    stages: Vec<PipelineStage>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum PipelineStageStatus {
    Recorded,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct PipelineStageExecution {
    pub(crate) kind: PipelineStageKind,
    pub(crate) name: &'static str,
    pub(crate) required: bool,
    pub(crate) status: PipelineStageStatus,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct PipelineExecutionSummary {
    kind: PipelineKind,
    stages: Vec<PipelineStageExecution>,
}

impl PipelineExecutionSummary {
    pub(crate) fn kind(&self) -> PipelineKind {
        self.kind
    }

    pub(crate) fn stages(&self) -> &[PipelineStageExecution] {
        &self.stages
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct PipelineExecutor;

impl PipelineExecutor {
    pub(crate) fn execute_contract(&self, graph: &PipelineGraph) -> PipelineExecutionSummary {
        let stages = graph
            .stages()
            .iter()
            .map(|stage| PipelineStageExecution {
                kind: stage.kind,
                name: stage.name,
                required: stage.required,
                status: PipelineStageStatus::Recorded,
            })
            .collect();

        PipelineExecutionSummary {
            kind: graph.kind,
            stages,
        }
    }
}

impl PipelineGraph {
    pub(crate) fn modular_voice_turn() -> Self {
        Self {
            kind: PipelineKind::ModularVoiceTurn,
            stages: vec![
                PipelineStage::required(PipelineStageKind::Vad, "voice.vad"),
                PipelineStage::required(PipelineStageKind::Endpointing, "voice.endpointing"),
                PipelineStage::required(PipelineStageKind::Asr, "voice.asr"),
                PipelineStage::required(PipelineStageKind::Chat, "voice.chat"),
                PipelineStage::required(PipelineStageKind::Tts, "voice.tts"),
            ],
        }
    }

    pub(crate) fn unified_voice_turn() -> Self {
        Self {
            kind: PipelineKind::UnifiedVoiceTurn,
            stages: vec![
                PipelineStage::required(PipelineStageKind::Vad, "voice.vad"),
                PipelineStage::required(PipelineStageKind::Endpointing, "voice.endpointing"),
                PipelineStage::required(PipelineStageKind::AudioChat, "voice.audio_chat"),
            ],
        }
    }

    pub(crate) fn diarization_transcript(enable_llm_refinement: bool) -> Self {
        let mut stages = vec![
            PipelineStage::required(PipelineStageKind::DecodeAudio, "diarization.decode_audio"),
            PipelineStage::required(PipelineStageKind::Diarization, "diarization.segment"),
            PipelineStage::required(PipelineStageKind::Asr, "diarization.asr"),
            PipelineStage::optional(PipelineStageKind::ForcedAlignment, "diarization.alignment"),
            PipelineStage::required(
                PipelineStageKind::SpeakerAttribution,
                "diarization.speaker_attribution",
            ),
        ];

        if enable_llm_refinement {
            stages.push(PipelineStage::optional(
                PipelineStageKind::LlmRefinement,
                "diarization.llm_refinement",
            ));
        }

        Self {
            kind: PipelineKind::DiarizationTranscript,
            stages,
        }
    }

    pub(crate) fn stages(&self) -> &[PipelineStage] {
        &self.stages
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn stage_kinds(graph: &PipelineGraph) -> Vec<PipelineStageKind> {
        graph.stages().iter().map(|stage| stage.kind).collect()
    }

    #[test]
    fn modular_voice_turn_graph_matches_current_route_order() {
        let graph = PipelineGraph::modular_voice_turn();

        assert_eq!(graph.kind, PipelineKind::ModularVoiceTurn);
        assert_eq!(
            stage_kinds(&graph),
            vec![
                PipelineStageKind::Vad,
                PipelineStageKind::Endpointing,
                PipelineStageKind::Asr,
                PipelineStageKind::Chat,
                PipelineStageKind::Tts,
            ]
        );
        assert!(graph.stages().iter().all(|stage| stage.required));
    }

    #[test]
    fn unified_voice_turn_graph_collapses_to_audio_chat_stage() {
        let graph = PipelineGraph::unified_voice_turn();

        assert_eq!(
            stage_kinds(&graph),
            vec![
                PipelineStageKind::Vad,
                PipelineStageKind::Endpointing,
                PipelineStageKind::AudioChat,
            ]
        );
    }

    #[test]
    fn diarization_graph_makes_refinement_explicit_and_optional() {
        let graph = PipelineGraph::diarization_transcript(true);

        assert_eq!(graph.kind, PipelineKind::DiarizationTranscript);
        assert_eq!(
            stage_kinds(&graph),
            vec![
                PipelineStageKind::DecodeAudio,
                PipelineStageKind::Diarization,
                PipelineStageKind::Asr,
                PipelineStageKind::ForcedAlignment,
                PipelineStageKind::SpeakerAttribution,
                PipelineStageKind::LlmRefinement,
            ]
        );
        assert!(
            !graph
                .stages()
                .iter()
                .find(|stage| stage.kind == PipelineStageKind::LlmRefinement)
                .expect("llm refinement stage")
                .required
        );
    }

    #[test]
    fn pipeline_executor_records_stage_contracts_in_order() {
        let graph = PipelineGraph::modular_voice_turn();
        let summary = PipelineExecutor.execute_contract(&graph);

        assert_eq!(summary.kind(), PipelineKind::ModularVoiceTurn);
        assert_eq!(
            summary
                .stages()
                .iter()
                .map(|stage| stage.kind)
                .collect::<Vec<_>>(),
            vec![
                PipelineStageKind::Vad,
                PipelineStageKind::Endpointing,
                PipelineStageKind::Asr,
                PipelineStageKind::Chat,
                PipelineStageKind::Tts,
            ]
        );
        assert!(summary
            .stages()
            .iter()
            .all(|stage| stage.status == PipelineStageStatus::Recorded));
    }
}
