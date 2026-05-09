//! Stable metric names for production voice runtime observability.

use serde::Serialize;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct VoiceMetricDescriptor {
    pub name: &'static str,
    pub description: &'static str,
}

pub const VOICE_SESSION_STARTED_TOTAL: &str = "voice.session.started_total";
pub const VOICE_SESSION_CLOSED_TOTAL: &str = "voice.session.closed_total";
pub const VOICE_SESSION_DURATION_MS: &str = "voice.session.duration_ms";
pub const VOICE_AUDIO_INGRESS_JITTER_MS: &str = "voice.audio.ingress.jitter_ms";
pub const VOICE_AUDIO_INGRESS_DROPPED_FRAMES_TOTAL: &str =
    "voice.audio.ingress.dropped_frames_total";
pub const VOICE_AUDIO_EGRESS_UNDERRUNS_TOTAL: &str = "voice.audio.egress.underruns_total";
pub const VOICE_VAD_SPEECH_START_MS: &str = "voice.vad.speech_start_ms";
pub const VOICE_ENDPOINTING_LATENCY_MS: &str = "voice.endpointing.latency_ms";
pub const VOICE_ASR_FIRST_PARTIAL_MS: &str = "voice.asr.first_partial_ms";
pub const VOICE_ASR_FINAL_MS: &str = "voice.asr.final_ms";
pub const VOICE_LLM_FIRST_TOKEN_MS: &str = "voice.llm.first_token_ms";
pub const VOICE_TTS_FIRST_AUDIO_MS: &str = "voice.tts.first_audio_ms";
pub const VOICE_BARGE_IN_LATENCY_MS: &str = "voice.barge_in.latency_ms";
pub const VOICE_BARGE_IN_TOTAL: &str = "voice.barge_in.events_total";
pub const VOICE_SESSION_INTERRUPTED_TOTAL: &str = "voice.session.interruptions_total";
pub const VOICE_STREAM_BACKPRESSURE_TOTAL: &str = "voice.stream.backpressure_total";
pub const VOICE_MODEL_READY_TOTAL: &str = "voice.model.ready_total";

pub const VOICE_METRIC_CATALOG: &[VoiceMetricDescriptor] = &[
    VoiceMetricDescriptor {
        name: VOICE_SESSION_STARTED_TOTAL,
        description: "Voice sessions started.",
    },
    VoiceMetricDescriptor {
        name: VOICE_SESSION_CLOSED_TOTAL,
        description: "Voice sessions closed, labeled by close reason when emitted.",
    },
    VoiceMetricDescriptor {
        name: VOICE_SESSION_DURATION_MS,
        description: "End-to-end voice session duration in milliseconds.",
    },
    VoiceMetricDescriptor {
        name: VOICE_AUDIO_INGRESS_JITTER_MS,
        description: "Observed audio ingress jitter in milliseconds.",
    },
    VoiceMetricDescriptor {
        name: VOICE_AUDIO_INGRESS_DROPPED_FRAMES_TOTAL,
        description: "Audio ingress frames dropped before runtime processing.",
    },
    VoiceMetricDescriptor {
        name: VOICE_AUDIO_EGRESS_UNDERRUNS_TOTAL,
        description: "Audio egress underruns while streaming assistant audio.",
    },
    VoiceMetricDescriptor {
        name: VOICE_VAD_SPEECH_START_MS,
        description: "Latency to detect speech start from audio ingress.",
    },
    VoiceMetricDescriptor {
        name: VOICE_ENDPOINTING_LATENCY_MS,
        description: "Latency from final user audio to endpoint decision.",
    },
    VoiceMetricDescriptor {
        name: VOICE_ASR_FIRST_PARTIAL_MS,
        description: "Latency from endpointed audio to first transcript partial.",
    },
    VoiceMetricDescriptor {
        name: VOICE_ASR_FINAL_MS,
        description: "Latency from endpointed audio to final transcript.",
    },
    VoiceMetricDescriptor {
        name: VOICE_LLM_FIRST_TOKEN_MS,
        description: "Latency from final transcript to first assistant text token.",
    },
    VoiceMetricDescriptor {
        name: VOICE_TTS_FIRST_AUDIO_MS,
        description: "Latency from assistant text to first synthesized audio chunk.",
    },
    VoiceMetricDescriptor {
        name: VOICE_BARGE_IN_LATENCY_MS,
        description: "Latency from user speech during playback to active turn cancellation.",
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
    VoiceMetricDescriptor {
        name: VOICE_MODEL_READY_TOTAL,
        description: "Voice-capable models that became resident and ready.",
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
        "gauge"
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

        for required in [
            VOICE_ASR_FIRST_PARTIAL_MS,
            VOICE_ASR_FINAL_MS,
            VOICE_LLM_FIRST_TOKEN_MS,
            VOICE_TTS_FIRST_AUDIO_MS,
            VOICE_BARGE_IN_LATENCY_MS,
        ] {
            assert!(names.contains(required), "missing {required}");
        }
    }

    #[test]
    fn voice_metric_prometheus_contract_exports_sanitized_metric_names() {
        let payload = voice_metric_prometheus_contract();

        assert!(payload.contains("# HELP izwi_voice_metric_contract_info"));
        assert!(payload.contains("prometheus_name=\"izwi_voice_session_started_total\""));
        assert!(payload.contains("metric_type=\"counter\""));
        assert!(payload.contains("name=\"voice.tts.first_audio_ms\""));
    }
}
