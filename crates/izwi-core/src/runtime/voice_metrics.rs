//! Stable metric names for production voice runtime observability.

use serde::Serialize;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct VoiceMetricDescriptor {
    pub name: &'static str,
    pub description: &'static str,
}

pub const VOICE_SESSION_STARTED_TOTAL: &str = "voice.session.started_total";
pub const VOICE_SESSION_CLOSED_TOTAL: &str = "voice.session.closed_total";
pub const VOICE_SESSION_DURATION_SECONDS: &str = "voice.session.duration_seconds";
pub const VOICE_AUDIO_INGRESS_DROPPED_FRAMES_TOTAL: &str =
    "voice.audio.ingress.dropped_frames_total";
pub const VOICE_AUDIO_EGRESS_UNDERRUNS_TOTAL: &str = "voice.audio.egress.underruns_total";
pub const VOICE_ENDPOINTING_LATENCY_SECONDS: &str = "voice.endpointing.latency_seconds";
pub const VOICE_ASR_FIRST_PARTIAL_SECONDS: &str = "voice.asr.first_partial_seconds";
pub const VOICE_ASR_FINAL_SECONDS: &str = "voice.asr.final_seconds";
pub const VOICE_LLM_FIRST_TOKEN_SECONDS: &str = "voice.llm.first_token_seconds";
pub const VOICE_TTS_FIRST_AUDIO_SECONDS: &str = "voice.tts.first_audio_seconds";
pub const VOICE_TURN_FIRST_AUDIO_SECONDS: &str = "voice.turn.first_audio_seconds";
pub const VOICE_BARGE_IN_LATENCY_SECONDS: &str = "voice.barge_in.latency_seconds";
pub const VOICE_BARGE_IN_TOTAL: &str = "voice.barge_in.events_total";
pub const VOICE_SESSION_INTERRUPTED_TOTAL: &str = "voice.session.interruptions_total";
pub const VOICE_STREAM_BACKPRESSURE_TOTAL: &str = "voice.stream.backpressure_total";

/// The latencies a voice session records, each a histogram in seconds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VoiceLatency {
    /// Session start to close.
    SessionDuration,
    /// Trailing silence the endpoint detector waited for before ending the
    /// utterance (audio time; plus the stall for a client pause).
    Endpointing,
    /// Endpoint decision to the first transcript partial.
    AsrFirstPartial,
    /// Endpoint decision to the final transcript.
    AsrFinal,
    /// Final transcript to the first assistant text the voice pipeline can
    /// speak (the whole reply when the text model runs non-streaming).
    LlmFirstToken,
    /// Assistant text to its first synthesized audio sent.
    TtsFirstAudio,
    /// Endpoint decision to the first assistant audio sent: the turn's
    /// response latency.
    TurnFirstAudio,
    /// Start of the user's speech to the interrupted turn having stopped.
    BargeIn,
}

impl VoiceLatency {
    pub const ALL: [Self; 8] = [
        Self::SessionDuration,
        Self::Endpointing,
        Self::AsrFirstPartial,
        Self::AsrFinal,
        Self::LlmFirstToken,
        Self::TtsFirstAudio,
        Self::TurnFirstAudio,
        Self::BargeIn,
    ];

    pub const fn metric_name(self) -> &'static str {
        match self {
            Self::SessionDuration => VOICE_SESSION_DURATION_SECONDS,
            Self::Endpointing => VOICE_ENDPOINTING_LATENCY_SECONDS,
            Self::AsrFirstPartial => VOICE_ASR_FIRST_PARTIAL_SECONDS,
            Self::AsrFinal => VOICE_ASR_FINAL_SECONDS,
            Self::LlmFirstToken => VOICE_LLM_FIRST_TOKEN_SECONDS,
            Self::TtsFirstAudio => VOICE_TTS_FIRST_AUDIO_SECONDS,
            Self::TurnFirstAudio => VOICE_TURN_FIRST_AUDIO_SECONDS,
            Self::BargeIn => VOICE_BARGE_IN_LATENCY_SECONDS,
        }
    }
}

pub const VOICE_METRIC_CATALOG: &[VoiceMetricDescriptor] = &[
    VoiceMetricDescriptor {
        name: VOICE_SESSION_STARTED_TOTAL,
        description: "Voice sessions started.",
    },
    VoiceMetricDescriptor {
        name: VOICE_SESSION_CLOSED_TOTAL,
        description: "Voice sessions closed.",
    },
    VoiceMetricDescriptor {
        name: VOICE_SESSION_DURATION_SECONDS,
        description: "Voice session duration from start to close.",
    },
    VoiceMetricDescriptor {
        name: VOICE_AUDIO_INGRESS_DROPPED_FRAMES_TOTAL,
        description: "Input audio frames missing from the client's frame sequence.",
    },
    VoiceMetricDescriptor {
        name: VOICE_AUDIO_EGRESS_UNDERRUNS_TOTAL,
        description: "Assistant audio chunks sent after the audio already sent had finished playing.",
    },
    VoiceMetricDescriptor {
        name: VOICE_ENDPOINTING_LATENCY_SECONDS,
        description: "Trailing silence waited for before the endpoint decision.",
    },
    VoiceMetricDescriptor {
        name: VOICE_ASR_FIRST_PARTIAL_SECONDS,
        description: "Latency from the endpoint decision to the first transcript partial.",
    },
    VoiceMetricDescriptor {
        name: VOICE_ASR_FINAL_SECONDS,
        description: "Latency from the endpoint decision to the final transcript.",
    },
    VoiceMetricDescriptor {
        name: VOICE_LLM_FIRST_TOKEN_SECONDS,
        description: "Latency from the final transcript to the first assistant text the voice pipeline can speak.",
    },
    VoiceMetricDescriptor {
        name: VOICE_TTS_FIRST_AUDIO_SECONDS,
        description: "Latency from assistant text to its first synthesized audio sent.",
    },
    VoiceMetricDescriptor {
        name: VOICE_TURN_FIRST_AUDIO_SECONDS,
        description: "Latency from the endpoint decision to the turn's first assistant audio sent.",
    },
    VoiceMetricDescriptor {
        name: VOICE_BARGE_IN_LATENCY_SECONDS,
        description: "Latency from the user's speech start to the interrupted turn having stopped.",
    },
    VoiceMetricDescriptor {
        name: VOICE_BARGE_IN_TOTAL,
        description: "Voice barge-in interruptions.",
    },
    VoiceMetricDescriptor {
        name: VOICE_SESSION_INTERRUPTED_TOTAL,
        description: "Voice turns interrupted before completion.",
    },
    VoiceMetricDescriptor {
        name: VOICE_STREAM_BACKPRESSURE_TOTAL,
        description: "Runtime stream backpressure events.",
    },
];

pub fn voice_metric_catalog() -> &'static [VoiceMetricDescriptor] {
    VOICE_METRIC_CATALOG
}

pub fn prometheus_voice_metric_name(name: &str) -> String {
    format!("izwi_{}", name.replace('.', "_"))
}

pub fn prometheus_voice_metric_type(name: &str) -> &'static str {
    if name.ends_with("_total") {
        "counter"
    } else {
        "histogram"
    }
}

pub fn voice_metric_prometheus_contract() -> String {
    let mut payload = String::new();
    payload.push_str(
        "# HELP izwi_voice_metric_contract_info Voice metric catalog entry exposed by the runtime.\n\
# TYPE izwi_voice_metric_contract_info gauge\n",
    );
    for metric in voice_metric_catalog() {
        let prometheus_name = prometheus_voice_metric_name(metric.name);
        let metric_type = prometheus_voice_metric_type(metric.name);
        payload.push_str(&format!(
            "izwi_voice_metric_contract_info{{name=\"{}\",prometheus_name=\"{}\",metric_type=\"{}\",description=\"{}\"}} 1\n",
            prometheus_label_value(metric.name),
            prometheus_label_value(&prometheus_name),
            metric_type,
            prometheus_label_value(&metric.description.replace('\n', " "))
        ));
    }
    payload
}

fn prometheus_label_value(value: &str) -> String {
    value
        .replace('\\', "\\\\")
        .replace('"', "\\\"")
        .replace('\n', "\\n")
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use super::*;

    #[test]
    fn voice_metric_names_are_unique_and_prefixed() {
        let mut names = HashSet::new();

        for metric in voice_metric_catalog() {
            assert!(
                metric.name.starts_with("voice."),
                "{} should use voice prefix",
                metric.name
            );
            assert!(
                names.insert(metric.name),
                "duplicate metric {}",
                metric.name
            );
            assert!(!metric.description.trim().is_empty());
        }
    }

    #[test]
    fn voice_metric_catalog_covers_required_turn_latencies() {
        let names = voice_metric_catalog()
            .iter()
            .map(|metric| metric.name)
            .collect::<HashSet<_>>();

        for latency in VoiceLatency::ALL {
            assert!(
                names.contains(latency.metric_name()),
                "missing {}",
                latency.metric_name()
            );
            assert!(latency.metric_name().ends_with("_seconds"));
        }
    }

    #[test]
    fn voice_metric_prometheus_contract_exports_sanitized_metric_names() {
        let payload = voice_metric_prometheus_contract();

        assert!(payload.contains("# HELP izwi_voice_metric_contract_info"));
        assert!(payload.contains("prometheus_name=\"izwi_voice_session_started_total\""));
        assert!(payload.contains("metric_type=\"counter\""));
        assert!(payload.contains("name=\"voice.tts.first_audio_seconds\""));
        assert!(payload.contains("metric_type=\"histogram\""));
    }
}
