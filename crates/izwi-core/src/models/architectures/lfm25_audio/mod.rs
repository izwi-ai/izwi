//! Native LFM2.5 Audio GGUF architecture support.

mod audio_output;
mod backbone;
mod bundle;
mod config;
mod conformer;
mod detokenizer;
mod model;
mod preprocessor;
mod sampling;
mod tokenizer;

pub const LFM25_AUDIO_BUILT_IN_SPEAKERS: [&str; 4] =
    ["US Female", "US Male", "UK Female", "UK Male"];
pub const LFM25_AUDIO_DEFAULT_INTERLEAVED_SYSTEM_PROMPT: &str =
    "Respond with interleaved text and audio.";

pub use model::{
    Lfm25AudioGenerationOutput, Lfm25AudioModel, Lfm25AudioStreamConfig, Lfm25AudioTextOutput,
};
pub use sampling::{Lfm25AudioGenerationConfig, Lfm25SamplingConfig};

fn normalize_speaker_key(speaker: Option<&str>) -> String {
    speaker
        .unwrap_or("")
        .chars()
        .filter(|ch| ch.is_ascii_alphanumeric())
        .map(|ch| ch.to_ascii_lowercase())
        .collect::<String>()
}

fn canonical_speaker_label(speaker: Option<&str>) -> &'static str {
    let normalized = normalize_speaker_key(speaker);

    if normalized.contains("ukmale")
        || normalized == "dylan"
        || normalized == "unclefu"
        || normalized == "ukm"
    {
        return "UK Male";
    }

    if normalized.contains("ukfemale") || normalized == "vivian" {
        return "UK Female";
    }

    if normalized.contains("usmale")
        || normalized == "ryan"
        || normalized == "aiden"
        || normalized == "eric"
        || normalized == "male"
    {
        return "US Male";
    }

    if normalized.contains("usfemale")
        || normalized == "serena"
        || normalized == "sohee"
        || normalized == "onoanna"
        || normalized == "anna"
        || normalized == "default"
    {
        return "US Female";
    }

    "US Female"
}

pub fn lfm25_audio_voice_instruction(speaker: Option<&str>) -> &'static str {
    match canonical_speaker_label(speaker) {
        "US Male" => "Use the US male voice.",
        "UK Female" => "Use the UK female voice.",
        "UK Male" => "Use the UK male voice.",
        _ => "Use the US female voice.",
    }
}

pub fn lfm25_audio_tts_system_prompt(speaker: Option<&str>) -> &'static str {
    match canonical_speaker_label(speaker) {
        "US Male" => "Perform TTS. Use the US male voice.",
        "UK Female" => "Perform TTS. Use the UK female voice.",
        "UK Male" => "Perform TTS. Use the UK male voice.",
        _ => "Perform TTS. Use the US female voice.",
    }
}

pub fn lfm25_audio_interleaved_system_prompt(
    system_prompt: Option<&str>,
    speaker: Option<&str>,
) -> String {
    let voice_instruction = lfm25_audio_voice_instruction(speaker);
    let base_prompt = system_prompt
        .map(str::trim)
        .filter(|value| !value.is_empty());

    let Some(base_prompt) = base_prompt else {
        return format!(
            "{} {}",
            LFM25_AUDIO_DEFAULT_INTERLEAVED_SYSTEM_PROMPT, voice_instruction
        );
    };

    let normalized_base = base_prompt.to_ascii_lowercase();
    let normalized_default = LFM25_AUDIO_DEFAULT_INTERLEAVED_SYSTEM_PROMPT.to_ascii_lowercase();
    let normalized_voice = voice_instruction.to_ascii_lowercase();

    let mut parts = Vec::new();
    if !normalized_base.contains(normalized_default.as_str()) {
        parts.push(LFM25_AUDIO_DEFAULT_INTERLEAVED_SYSTEM_PROMPT.to_string());
    }
    if !normalized_base.contains(normalized_voice.as_str()) {
        parts.push(voice_instruction.to_string());
    }
    parts.push(base_prompt.to_string());
    parts.join(" ")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn speaker_mapping_resolves_known_presets_and_legacy_aliases() {
        assert_eq!(
            lfm25_audio_tts_system_prompt(Some("US Female")),
            "Perform TTS. Use the US female voice."
        );
        assert_eq!(
            lfm25_audio_tts_system_prompt(Some("Ryan")),
            "Perform TTS. Use the US male voice."
        );
        assert_eq!(
            lfm25_audio_tts_system_prompt(Some("Vivian")),
            "Perform TTS. Use the UK female voice."
        );
        assert_eq!(
            lfm25_audio_tts_system_prompt(Some("Dylan")),
            "Perform TTS. Use the UK male voice."
        );
    }

    #[test]
    fn interleaved_prompt_adds_audio_and_voice_instructions() {
        assert_eq!(
            lfm25_audio_interleaved_system_prompt(
                Some("You are a concise assistant."),
                Some("UK Female")
            ),
            "Respond with interleaved text and audio. Use the UK female voice. You are a concise assistant."
        );
    }
}
