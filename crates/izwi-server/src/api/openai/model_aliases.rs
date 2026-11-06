//! OpenAI model names an operator maps to Izwi models
//! (`IZWI_OPENAI_MODEL_ALIASES`), so OpenAI SDK code that names `tts-1`,
//! `gpt-4o-mini-tts` or `whisper-1` runs unchanged against Izwi. No name is
//! mapped by default: silently picking a model for `tts-1` would mislead.

/// The Izwi model `name` stands for: its alias target when the operator
/// mapped it (names match without regard to case), else `name` itself.
pub(crate) fn resolve_model_alias(name: &str) -> &str {
    resolve_in(&crate::env_config::tuning().speech_api.model_aliases, name)
}

fn resolve_in<'a>(aliases: &'a [(String, String)], name: &'a str) -> &'a str {
    let wanted = name.trim();
    aliases
        .iter()
        .find(|(alias, _)| alias.eq_ignore_ascii_case(wanted))
        .map(|(_, model)| model.as_str())
        .unwrap_or(name)
}

#[cfg(test)]
mod tests {
    use super::resolve_in;

    #[test]
    fn mapped_names_resolve_and_others_pass_through() {
        let aliases = vec![
            ("tts-1".to_string(), "Kokoro-82M".to_string()),
            ("whisper-1".to_string(), "Parakeet-TDT-0.6B-v3".to_string()),
        ];
        assert_eq!(resolve_in(&aliases, "TTS-1"), "Kokoro-82M");
        assert_eq!(resolve_in(&aliases, "whisper-1"), "Parakeet-TDT-0.6B-v3");
        assert_eq!(resolve_in(&aliases, "Qwen3-TTS"), "Qwen3-TTS");
        assert_eq!(resolve_in(&[], "tts-1"), "tts-1");
    }
}
