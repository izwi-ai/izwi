//! Capability conformance contracts for runtime architecture migrations.
//!
//! These contracts are intentionally lightweight. They give integration tests
//! and future fake adapters a stable checklist before execution behavior is
//! moved behind new broker, adapter, or pipeline boundaries.

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ConformanceCapability {
    Tts,
    StreamingTts,
    Asr,
    RealtimeAsr,
    Chat,
    AudioChat,
    SpeechToSpeech,
    Diarization,
    ForcedAlignment,
    Vad,
    Endpointing,
    Tokenizer,
}

impl ConformanceCapability {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Tts => "tts",
            Self::StreamingTts => "streaming_tts",
            Self::Asr => "asr",
            Self::RealtimeAsr => "realtime_asr",
            Self::Chat => "chat",
            Self::AudioChat => "audio_chat",
            Self::SpeechToSpeech => "speech_to_speech",
            Self::Diarization => "diarization",
            Self::ForcedAlignment => "forced_alignment",
            Self::Vad => "vad",
            Self::Endpointing => "endpointing",
            Self::Tokenizer => "tokenizer",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConformanceExecutionClass {
    Scheduled,
    Streaming,
    Realtime,
    Batch,
    Pipeline,
    Artifact,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CapabilityConformanceCase {
    pub id: &'static str,
    pub capability: ConformanceCapability,
    pub execution_class: ConformanceExecutionClass,
    pub fixture: &'static str,
}

impl CapabilityConformanceCase {
    pub const fn new(
        id: &'static str,
        capability: ConformanceCapability,
        execution_class: ConformanceExecutionClass,
        fixture: &'static str,
    ) -> Self {
        Self {
            id,
            capability,
            execution_class,
            fixture,
        }
    }
}

pub const REQUIRED_CONFORMANCE_CAPABILITIES: &[ConformanceCapability] = &[
    ConformanceCapability::Tts,
    ConformanceCapability::StreamingTts,
    ConformanceCapability::Asr,
    ConformanceCapability::RealtimeAsr,
    ConformanceCapability::Chat,
    ConformanceCapability::AudioChat,
    ConformanceCapability::SpeechToSpeech,
    ConformanceCapability::Diarization,
    ConformanceCapability::ForcedAlignment,
    ConformanceCapability::Vad,
    ConformanceCapability::Endpointing,
    ConformanceCapability::Tokenizer,
];

pub const CAPABILITY_CONFORMANCE_CASES: &[CapabilityConformanceCase] = &[
    CapabilityConformanceCase::new(
        "tts.short_text.binary",
        ConformanceCapability::Tts,
        ConformanceExecutionClass::Scheduled,
        "short text with explicit model, voice option, and binary response format",
    ),
    CapabilityConformanceCase::new(
        "streaming_tts.short_text",
        ConformanceCapability::StreamingTts,
        ConformanceExecutionClass::Streaming,
        "short text with first audio chunk, multiple chunks, and terminal event",
    ),
    CapabilityConformanceCase::new(
        "asr.short_wav.transcript",
        ConformanceCapability::Asr,
        ConformanceExecutionClass::Batch,
        "short wav input with transcript text, language hint, and format mapping",
    ),
    CapabilityConformanceCase::new(
        "realtime_asr.partial_final",
        ConformanceCapability::RealtimeAsr,
        ConformanceExecutionClass::Realtime,
        "pcm frames with partial update, final update, cancellation, and close",
    ),
    CapabilityConformanceCase::new(
        "chat.single_prompt.streaming",
        ConformanceCapability::Chat,
        ConformanceExecutionClass::Scheduled,
        "single user prompt with non-streaming and streaming delta responses",
    ),
    CapabilityConformanceCase::new(
        "audio_chat.audio_prompt.response",
        ConformanceCapability::AudioChat,
        ConformanceExecutionClass::Scheduled,
        "audio bytes plus optional text prompt returning text and/or audio output",
    ),
    CapabilityConformanceCase::new(
        "speech_to_speech.audio_stream",
        ConformanceCapability::SpeechToSpeech,
        ConformanceExecutionClass::Streaming,
        "audio request to streaming audio response with cancellation",
    ),
    CapabilityConformanceCase::new(
        "diarization.short_multispeaker",
        ConformanceCapability::Diarization,
        ConformanceExecutionClass::Pipeline,
        "short multi-speaker fixture with speaker labels and ASR attribution",
    ),
    CapabilityConformanceCase::new(
        "forced_alignment.words",
        ConformanceCapability::ForcedAlignment,
        ConformanceExecutionClass::Batch,
        "transcript plus audio fixture producing ordered word timestamps",
    ),
    CapabilityConformanceCase::new(
        "voice.vad.speech_events",
        ConformanceCapability::Vad,
        ConformanceExecutionClass::Realtime,
        "pcm frames producing speech start and speech end events",
    ),
    CapabilityConformanceCase::new(
        "voice.endpointing.turn_boundary",
        ConformanceCapability::Endpointing,
        ConformanceExecutionClass::Realtime,
        "speech activity stream producing stable turn boundary decisions",
    ),
    CapabilityConformanceCase::new(
        "tokenizer.model_artifact.round_trip",
        ConformanceCapability::Tokenizer,
        ConformanceExecutionClass::Artifact,
        "tokenizer artifact load with deterministic encode/decode smoke check",
    ),
];

pub fn capability_conformance_cases() -> &'static [CapabilityConformanceCase] {
    CAPABILITY_CONFORMANCE_CASES
}

pub fn required_conformance_capabilities() -> &'static [ConformanceCapability] {
    REQUIRED_CONFORMANCE_CAPABILITIES
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;

    #[test]
    fn conformance_cases_cover_every_required_capability() {
        let covered = capability_conformance_cases()
            .iter()
            .map(|case| case.capability)
            .collect::<BTreeSet<_>>();

        for capability in required_conformance_capabilities() {
            assert!(
                covered.contains(capability),
                "missing conformance case for {}",
                capability.as_str()
            );
        }
    }

    #[test]
    fn conformance_case_ids_are_unique() {
        let mut ids = BTreeSet::new();
        for case in capability_conformance_cases() {
            assert!(ids.insert(case.id), "duplicate conformance case {}", case.id);
        }
    }
}
