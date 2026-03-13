//! Model variant capability helpers and parser utilities.

use std::fmt;

use super::ModelVariant;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelFamily {
    Qwen3Tts,
    KokoroTts,
    Qwen3Asr,
    ParakeetAsr,
    WhisperAsr,
    SortformerDiarization,
    Qwen3Chat,
    Lfm2Chat,
    Gemma3Chat,
    Qwen3ForcedAligner,
    Voxtral,
    Lfm2Audio,
    Tokenizer,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelTask {
    Tts,
    Asr,
    Diarization,
    Chat,
    ForcedAlign,
    AudioChat,
    Tokenizer,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InferenceBackendHint {
    CandleNative,
}

#[derive(Debug, Clone)]
pub struct ParseModelVariantError {
    input: String,
}

impl ParseModelVariantError {
    fn new(input: impl Into<String>) -> Self {
        Self {
            input: input.into(),
        }
    }
}

impl fmt::Display for ParseModelVariantError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Unsupported model identifier: {}",
            self.input.trim().if_empty("<empty>")
        )
    }
}

impl std::error::Error for ParseModelVariantError {}

trait EmptyFallback {
    fn if_empty(self, fallback: &str) -> String;
}

impl EmptyFallback for &str {
    fn if_empty(self, fallback: &str) -> String {
        if self.trim().is_empty() {
            fallback.to_string()
        } else {
            self.to_string()
        }
    }
}

impl ModelVariant {
    pub fn family(&self) -> ModelFamily {
        use ModelVariant::*;

        match self {
            Qwen3Tts12Hz06BBase
            | Qwen3Tts12Hz06BBase4Bit
            | Qwen3Tts12Hz06BBase8Bit
            | Qwen3Tts12Hz06BBaseBf16
            | Qwen3Tts12Hz06BCustomVoice
            | Qwen3Tts12Hz06BCustomVoice4Bit
            | Qwen3Tts12Hz06BCustomVoice8Bit
            | Qwen3Tts12Hz06BCustomVoiceBf16
            | Qwen3Tts12Hz17BBase
            | Qwen3Tts12Hz17BBase4Bit
            | Qwen3Tts12Hz17BCustomVoice
            | Qwen3Tts12Hz17BCustomVoice4Bit
            | Qwen3Tts12Hz17BVoiceDesign
            | Qwen3Tts12Hz17BVoiceDesign4Bit
            | Qwen3Tts12Hz17BVoiceDesign8Bit
            | Qwen3Tts12Hz17BVoiceDesignBf16 => ModelFamily::Qwen3Tts,
            Kokoro82M => ModelFamily::KokoroTts,
            Qwen3TtsTokenizer12Hz => ModelFamily::Tokenizer,
            Lfm25Audio15B | Lfm25Audio15B4Bit => ModelFamily::Lfm2Audio,
            Qwen3Asr06B | Qwen3Asr06B4Bit | Qwen3Asr06B8Bit | Qwen3Asr06BBf16 | Qwen3Asr17B
            | Qwen3Asr17B4Bit | Qwen3Asr17B8Bit | Qwen3Asr17BBf16 => ModelFamily::Qwen3Asr,
            ParakeetTdt06BV2 | ParakeetTdt06BV3 => ModelFamily::ParakeetAsr,
            WhisperLargeV3Turbo => ModelFamily::WhisperAsr,
            DiarStreamingSortformer4SpkV21 => ModelFamily::SortformerDiarization,
            Qwen306B | Qwen306B4Bit | Qwen306BGguf | Qwen317B | Qwen317B4Bit | Qwen317BGguf
            | Qwen34BGguf | Qwen38BGguf | Qwen314BGguf => ModelFamily::Qwen3Chat,
            Lfm2512BInstructGguf | Lfm2512BThinkingGguf => ModelFamily::Lfm2Chat,
            Gemma31BIt | Gemma34BIt => ModelFamily::Gemma3Chat,
            Qwen3ForcedAligner06B | Qwen3ForcedAligner06B4Bit => ModelFamily::Qwen3ForcedAligner,
            VoxtralMini4BRealtime2602 => ModelFamily::Voxtral,
        }
    }

    pub fn primary_task(&self) -> ModelTask {
        match self.family() {
            ModelFamily::Qwen3Tts | ModelFamily::KokoroTts => ModelTask::Tts,
            ModelFamily::Qwen3Asr | ModelFamily::ParakeetAsr | ModelFamily::WhisperAsr => {
                ModelTask::Asr
            }
            ModelFamily::SortformerDiarization => ModelTask::Diarization,
            ModelFamily::Qwen3Chat | ModelFamily::Lfm2Chat | ModelFamily::Gemma3Chat => {
                ModelTask::Chat
            }
            ModelFamily::Qwen3ForcedAligner => ModelTask::ForcedAlign,
            ModelFamily::Voxtral => ModelTask::AudioChat,
            ModelFamily::Lfm2Audio => ModelTask::AudioChat,
            ModelFamily::Tokenizer => ModelTask::Tokenizer,
        }
    }

    pub fn backend_hint(&self) -> InferenceBackendHint {
        InferenceBackendHint::CandleNative
    }
}

pub fn parse_model_variant(input: &str) -> Result<ModelVariant, ParseModelVariantError> {
    let trimmed = input.trim();
    if trimmed.is_empty() {
        return Err(ParseModelVariantError::new(input));
    }

    let normalized = normalize_identifier(trimmed);

    if let Some(found) = ModelVariant::all()
        .iter()
        .copied()
        .find(|variant| matches_variant_alias(*variant, trimmed, &normalized))
    {
        return Ok(found);
    }

    resolve_by_heuristic(&normalized).ok_or_else(|| ParseModelVariantError::new(input))
}

pub fn parse_tts_model_variant(input: &str) -> Result<ModelVariant, ParseModelVariantError> {
    let variant = parse_model_variant(input)?;
    if variant.is_tts() || variant.is_lfm2() {
        Ok(variant)
    } else {
        Err(ParseModelVariantError::new(input))
    }
}

pub fn parse_chat_model_variant(
    input: Option<&str>,
) -> Result<ModelVariant, ParseModelVariantError> {
    match input.unwrap_or("Qwen3-8B-GGUF") {
        id => {
            let variant = parse_model_variant(id)?;
            if variant.is_chat() {
                Ok(variant)
            } else {
                Err(ParseModelVariantError::new(id))
            }
        }
    }
}

/// Resolve the LLM variant for diarization transcript refinement.
/// Defaults to Qwen3-1.7B-GGUF and only accepts that variant.
pub fn resolve_diarization_llm_variant(
    input: Option<&str>,
) -> Result<ModelVariant, ParseModelVariantError> {
    match input.unwrap_or("Qwen3-1.7B-GGUF") {
        id => {
            let variant = parse_model_variant(id)?;
            if variant == ModelVariant::Qwen317BGguf {
                Ok(variant)
            } else {
                Err(ParseModelVariantError::new(id))
            }
        }
    }
}

pub fn resolve_asr_model_variant(input: Option<&str>) -> ModelVariant {
    use ModelVariant::*;

    let Some(raw) = input else {
        return Qwen3Asr06B;
    };

    match parse_model_variant(raw) {
        Ok(Qwen3Asr06B4Bit | Qwen3Asr06B8Bit) => Qwen3Asr06B,
        Ok(variant) if variant.is_asr() || variant.is_voxtral() || variant.is_lfm2() => variant,
        Ok(_) => Qwen3Asr06B,
        Err(_) => {
            let normalized = normalize_identifier(raw);
            if normalized.contains("voxtral") {
                VoxtralMini4BRealtime2602
            } else if normalized.contains("whisper")
                && normalized.contains("largev3")
                && normalized.contains("turbo")
            {
                WhisperLargeV3Turbo
            } else if let Some(lfm2_variant) = resolve_lfm2_audio_variant(&normalized) {
                lfm2_variant
            } else if normalized.contains("parakeet") {
                if normalized.contains("v3") {
                    ParakeetTdt06BV3
                } else {
                    ParakeetTdt06BV2
                }
            } else if normalized.contains("17") {
                Qwen3Asr17B
            } else {
                Qwen3Asr06B
            }
        }
    }
}

pub fn resolve_diarization_model_variant(input: Option<&str>) -> ModelVariant {
    use ModelVariant::*;

    let Some(raw) = input else {
        return DiarStreamingSortformer4SpkV21;
    };

    match parse_model_variant(raw) {
        Ok(variant) if variant.is_diarization() => variant,
        Ok(_) => DiarStreamingSortformer4SpkV21,
        Err(_) => {
            let normalized = normalize_identifier(raw);
            if normalized.contains("sortformer") || normalized.contains("diar") {
                DiarStreamingSortformer4SpkV21
            } else {
                DiarStreamingSortformer4SpkV21
            }
        }
    }
}

fn resolve_by_heuristic(normalized: &str) -> Option<ModelVariant> {
    use ModelVariant::*;

    if normalized.contains("voxtral") {
        return Some(VoxtralMini4BRealtime2602);
    }

    if normalized.contains("sortformer") && normalized.contains("diar") {
        return Some(DiarStreamingSortformer4SpkV21);
    }

    if normalized.contains("parakeet") && normalized.contains("tdt") {
        if normalized.contains("v3") {
            return Some(ParakeetTdt06BV3);
        }
        return Some(ParakeetTdt06BV2);
    }

    if normalized.contains("whisper")
        && normalized.contains("largev3")
        && normalized.contains("turbo")
    {
        return Some(WhisperLargeV3Turbo);
    }

    if normalized.contains("forcedaligner") {
        if normalized.contains("4bit") || normalized.contains("int4") {
            return Some(Qwen3ForcedAligner06B4Bit);
        }
        return Some(Qwen3ForcedAligner06B);
    }

    if normalized.contains("qwen3") && normalized.contains("asr") {
        let is_17b = normalized.contains("17b") || normalized.contains("17");
        let q4 = normalized.contains("4bit") || normalized.contains("int4");
        let q8 = normalized.contains("8bit") || normalized.contains("int8");
        let bf16 = normalized.contains("bf16") || normalized.contains("bfloat16");

        return Some(match (is_17b, q4, q8, bf16) {
            (true, true, _, _) => Qwen3Asr17B4Bit,
            (true, _, true, _) => Qwen3Asr17B8Bit,
            (true, _, _, true) => Qwen3Asr17BBf16,
            (true, _, _, _) => Qwen3Asr17B,
            (false, true, _, _) => Qwen3Asr06B,
            (false, _, true, _) => Qwen3Asr06B,
            (false, _, _, true) => Qwen3Asr06BBf16,
            (false, _, _, _) => Qwen3Asr06B,
        });
    }

    if normalized.contains("qwen3") && normalized.contains("tts") {
        let is_17b = normalized.contains("17b") || normalized.contains("17");
        let q4 = normalized.contains("4bit") || normalized.contains("int4");
        let q8 = normalized.contains("8bit") || normalized.contains("int8");
        let bf16 = normalized.contains("bf16") || normalized.contains("bfloat16");

        if normalized.contains("tokenizer") {
            return Some(Qwen3TtsTokenizer12Hz);
        }

        if is_17b && normalized.contains("voicedesign") {
            return Some(match (q4, q8, bf16) {
                (true, _, _) => Qwen3Tts12Hz17BVoiceDesign4Bit,
                (_, true, _) => Qwen3Tts12Hz17BVoiceDesign8Bit,
                (_, _, true) => Qwen3Tts12Hz17BVoiceDesignBf16,
                _ => Qwen3Tts12Hz17BVoiceDesign,
            });
        }

        if is_17b && normalized.contains("customvoice") {
            return Some(match (q4, q8, bf16) {
                (true, _, _) => Qwen3Tts12Hz17BCustomVoice4Bit,
                _ => Qwen3Tts12Hz17BCustomVoice,
            });
        }

        if is_17b {
            return Some(match (q4, q8, bf16) {
                (true, _, _) => Qwen3Tts12Hz17BBase4Bit,
                _ => Qwen3Tts12Hz17BBase,
            });
        }

        if normalized.contains("customvoice") {
            return Some(match (q4, q8, bf16) {
                (true, _, _) => Qwen3Tts12Hz06BCustomVoice4Bit,
                (_, true, _) => Qwen3Tts12Hz06BCustomVoice8Bit,
                (_, _, true) => Qwen3Tts12Hz06BCustomVoiceBf16,
                _ => Qwen3Tts12Hz06BCustomVoice,
            });
        }

        return Some(match (q4, q8, bf16) {
            (true, _, _) => Qwen3Tts12Hz06BBase4Bit,
            (_, true, _) => Qwen3Tts12Hz06BBase8Bit,
            (_, _, true) => Qwen3Tts12Hz06BBaseBf16,
            _ => Qwen3Tts12Hz06BBase,
        });
    }

    if normalized.contains("kokoro") && normalized.contains("82m") {
        return Some(Kokoro82M);
    }

    if normalized.contains("qwen3") && !normalized.contains("asr") && !normalized.contains("tts") {
        let is_14b = normalized.contains("14b");
        let is_17b = normalized.contains("17b") || normalized.contains("17");
        let is_8b = normalized.contains("8b");
        let is_4b = normalized.contains("qwen34b") || normalized.contains("4b");
        let q4 = normalized.contains("4bit") || normalized.contains("int4");
        let gguf =
            normalized.contains("gguf") || normalized.contains("q80") || normalized.contains("q8");

        if is_14b {
            return if gguf { Some(Qwen314BGguf) } else { None };
        }
        if is_4b {
            return if gguf { Some(Qwen34BGguf) } else { None };
        }
        if is_17b {
            return Some(if q4 {
                Qwen317B4Bit
            } else if gguf {
                Qwen317BGguf
            } else {
                Qwen317B
            });
        }
        if is_8b {
            return if gguf { Some(Qwen38BGguf) } else { None };
        }
        if normalized.contains("06b") || normalized.contains("0dot6b") || normalized.contains("06")
        {
            return Some(if q4 {
                Qwen306B4Bit
            } else if gguf {
                Qwen306BGguf
            } else {
                Qwen306B
            });
        }
    }

    if normalized.contains("gemma3") || (normalized.contains("gemma") && normalized.contains("it"))
    {
        if normalized.contains("1b") {
            return Some(Gemma31BIt);
        }
        if normalized.contains("4b") {
            return Some(Gemma34BIt);
        }
    }

    if let Some(lfm2_variant) = resolve_lfm2_chat_variant(normalized) {
        return Some(lfm2_variant);
    }

    if let Some(lfm2_variant) = resolve_lfm2_audio_variant(normalized) {
        return Some(lfm2_variant);
    }

    None
}

fn resolve_lfm2_chat_variant(normalized: &str) -> Option<ModelVariant> {
    use ModelVariant::*;

    if normalized.contains("audio") {
        return None;
    }

    if !normalized.contains("lfm25") && !normalized.contains("lfm2dot5") {
        return None;
    }

    if !(normalized.contains("12b") || normalized.contains("12")) {
        return None;
    }

    if !normalized.contains("gguf") {
        return None;
    }

    if normalized.contains("instruct") {
        return Some(Lfm2512BInstructGguf);
    }

    if normalized.contains("thinking") {
        return Some(Lfm2512BThinkingGguf);
    }

    None
}

fn resolve_lfm2_audio_variant(normalized: &str) -> Option<ModelVariant> {
    use ModelVariant::*;

    if !normalized.contains("audio") {
        return None;
    }

    if normalized.contains("lfm25") || normalized.contains("lfm2dot5") {
        if normalized.contains("4bit") || normalized.contains("int4") {
            return Some(Lfm25Audio15B4Bit);
        }
        if normalized.contains("gguf") {
            return None;
        }
        return Some(Lfm25Audio15B);
    }

    None
}

fn matches_variant_alias(variant: ModelVariant, raw: &str, normalized: &str) -> bool {
    let repo = variant.repo_id();
    let repo_tail = repo.rsplit('/').next().unwrap_or(repo);

    let aliases = [
        variant.dir_name(),
        variant.repo_id(),
        repo_tail,
        variant.display_name(),
    ];

    if aliases
        .iter()
        .any(|alias| normalize_identifier(alias) == normalized)
    {
        return true;
    }

    let maybe_compact = raw
        .trim()
        .replace('_', "-")
        .replace(' ', "-")
        .to_ascii_lowercase();

    variant.dir_name().eq_ignore_ascii_case(&maybe_compact)
        || variant.repo_id().eq_ignore_ascii_case(&maybe_compact)
}

fn normalize_identifier(input: &str) -> String {
    input
        .chars()
        .filter(|ch| ch.is_ascii_alphanumeric())
        .map(|ch| ch.to_ascii_lowercase())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_by_repo_tail() {
        let parsed = parse_model_variant("Qwen3-ASR-0.6B").unwrap();
        assert_eq!(parsed, ModelVariant::Qwen3Asr06B);
    }

    #[test]
    fn parse_by_display_name() {
        let parsed = parse_model_variant("Qwen3-TTS 0.6B Base 4-bit").unwrap();
        assert_eq!(parsed, ModelVariant::Qwen3Tts12Hz06BBase4Bit);
    }

    #[test]
    fn parse_tts_rejects_non_tts() {
        assert!(parse_tts_model_variant("Qwen3-ASR-0.6B").is_err());
    }

    #[test]
    fn resolve_asr_fallback_defaults_to_06b() {
        let resolved = resolve_asr_model_variant(Some("not-a-real-model"));
        assert_eq!(resolved, ModelVariant::Qwen3Asr06B);
    }

    #[test]
    fn resolve_asr_demotes_removed_qwen3_06b_quantized_variants() {
        let q4 = resolve_asr_model_variant(Some("Qwen3-ASR-0.6B-4bit"));
        let q8 = resolve_asr_model_variant(Some("Qwen3-ASR-0.6B-8bit"));
        assert_eq!(q4, ModelVariant::Qwen3Asr06B);
        assert_eq!(q8, ModelVariant::Qwen3Asr06B);
    }

    #[test]
    fn parse_tts_rejects_legacy_lfm2_audio() {
        assert!(parse_tts_model_variant("LFM2-Audio-1.5B").is_err());
    }

    #[test]
    fn parse_tts_accepts_lfm25_audio() {
        let parsed = parse_tts_model_variant("LFM2.5-Audio-1.5B").unwrap();
        assert_eq!(parsed, ModelVariant::Lfm25Audio15B);
    }

    #[test]
    fn resolve_asr_rejects_legacy_lfm2_audio() {
        let resolved = resolve_asr_model_variant(Some("LFM2-Audio-1.5B"));
        assert_eq!(resolved, ModelVariant::Qwen3Asr06B);
    }

    #[test]
    fn resolve_asr_accepts_lfm25_audio() {
        let resolved = resolve_asr_model_variant(Some("LFM2.5-Audio-1.5B"));
        assert_eq!(resolved, ModelVariant::Lfm25Audio15B);
    }

    #[test]
    fn parse_gemma_by_repo_tail() {
        let parsed = parse_model_variant("gemma-3-4b-it").unwrap();
        assert_eq!(parsed, ModelVariant::Gemma34BIt);
    }

    #[test]
    fn parse_chat_accepts_gemma() {
        let parsed = parse_chat_model_variant(Some("google/gemma-3-1b-it")).unwrap();
        assert_eq!(parsed, ModelVariant::Gemma31BIt);
    }

    #[test]
    fn parse_parakeet_by_repo_tail() {
        let parsed = parse_model_variant("parakeet-tdt-0.6b-v3").unwrap();
        assert_eq!(parsed, ModelVariant::ParakeetTdt06BV3);
    }

    #[test]
    fn resolve_asr_accepts_parakeet() {
        let resolved = resolve_asr_model_variant(Some("nvidia/parakeet-tdt-0.6b-v2"));
        assert_eq!(resolved, ModelVariant::ParakeetTdt06BV2);
    }

    #[test]
    fn parse_whisper_turbo_repo_alias() {
        let parsed = parse_model_variant("openai/whisper-large-v3-turbo").unwrap();
        assert_eq!(parsed, ModelVariant::WhisperLargeV3Turbo);
        assert_eq!(parsed.family(), ModelFamily::WhisperAsr);
    }

    #[test]
    fn resolve_asr_accepts_whisper_turbo() {
        let resolved = resolve_asr_model_variant(Some("whisper-large-v3-turbo"));
        assert_eq!(resolved, ModelVariant::WhisperLargeV3Turbo);
    }

    #[test]
    fn parse_mlx_parakeet_repo() {
        let parsed = parse_model_variant("mlx-community/parakeet-tdt-0.6b-v3").unwrap();
        assert_eq!(parsed, ModelVariant::ParakeetTdt06BV3);
    }

    #[test]
    fn parse_deprecated_parakeet_4bit_alias_demotes_to_nemo_variant() {
        let parsed = parse_model_variant("Parakeet-TDT-0.6B-v3-4bit").unwrap();
        assert_eq!(parsed, ModelVariant::ParakeetTdt06BV3);
    }

    #[test]
    fn parse_lfm25_audio_4bit() {
        let parsed = parse_model_variant("LFM2.5-Audio-1.5B-4bit").unwrap();
        assert_eq!(parsed, ModelVariant::Lfm25Audio15B4Bit);
    }

    #[test]
    fn parse_qwen_chat_17b_4bit() {
        let parsed = parse_chat_model_variant(Some("Qwen3-1.7B-4bit")).unwrap();
        assert_eq!(parsed, ModelVariant::Qwen317B4Bit);
    }

    #[test]
    fn parse_qwen_chat_06b() {
        let parsed = parse_chat_model_variant(Some("Qwen3-0.6B")).unwrap();
        assert_eq!(parsed, ModelVariant::Qwen306B);
    }

    #[test]
    fn parse_qwen_chat_06b_repo() {
        let parsed = parse_chat_model_variant(Some("Qwen/Qwen3-0.6B")).unwrap();
        assert_eq!(parsed, ModelVariant::Qwen306B);
    }

    #[test]
    fn parse_qwen_chat_06b_lowercase_alias() {
        let parsed = parse_model_variant("qwen3-0.6b").unwrap();
        assert_eq!(parsed, ModelVariant::Qwen306B);
    }

    #[test]
    fn parse_qwen_chat_06b_4bit() {
        let parsed = parse_chat_model_variant(Some("Qwen3-0.6B-4bit")).unwrap();
        assert_eq!(parsed, ModelVariant::Qwen306B4Bit);
    }

    #[test]
    fn parse_qwen_chat_06b_gguf() {
        let parsed = parse_chat_model_variant(Some("Qwen3-0.6B-GGUF")).unwrap();
        assert_eq!(parsed, ModelVariant::Qwen306BGguf);
    }

    #[test]
    fn parse_qwen_chat_17b_gguf_repo() {
        let parsed = parse_chat_model_variant(Some("Qwen/Qwen3-1.7B-GGUF")).unwrap();
        assert_eq!(parsed, ModelVariant::Qwen317BGguf);
    }

    #[test]
    fn parse_chat_defaults_to_qwen3_8b_gguf() {
        let parsed = parse_chat_model_variant(None).unwrap();
        assert_eq!(parsed, ModelVariant::Qwen38BGguf);
    }

    #[test]
    fn parse_lfm25_chat_instruct_repo() {
        let parsed = parse_chat_model_variant(Some("LiquidAI/LFM2.5-1.2B-Instruct-GGUF")).unwrap();
        assert_eq!(parsed, ModelVariant::Lfm2512BInstructGguf);
        assert_eq!(parsed.family(), ModelFamily::Lfm2Chat);
    }

    #[test]
    fn parse_lfm25_chat_thinking_repo() {
        let parsed = parse_chat_model_variant(Some("LiquidAI/LFM2.5-1.2B-Thinking-GGUF")).unwrap();
        assert_eq!(parsed, ModelVariant::Lfm2512BThinkingGguf);
        assert_eq!(parsed.family(), ModelFamily::Lfm2Chat);
    }

    #[test]
    fn parse_lfm25_chat_q4_gguf_file_alias() {
        let parsed = parse_chat_model_variant(Some("LFM2.5-1.2B-Instruct-Q4_K_M.gguf")).unwrap();
        assert_eq!(parsed, ModelVariant::Lfm2512BInstructGguf);
    }

    #[test]
    fn parse_lfm25_non_gguf_repo_is_rejected_for_chat() {
        assert!(parse_chat_model_variant(Some("LiquidAI/LFM2.5-1.2B-Instruct")).is_err());
    }

    #[test]
    fn resolve_diarization_llm_defaults_to_qwen_17b_gguf() {
        let resolved = resolve_diarization_llm_variant(None).unwrap();
        assert_eq!(resolved, ModelVariant::Qwen317BGguf);
    }

    #[test]
    fn resolve_diarization_llm_accepts_qwen_17b_gguf_repo_alias() {
        let resolved = resolve_diarization_llm_variant(Some("Qwen/Qwen3-1.7B-GGUF")).unwrap();
        assert_eq!(resolved, ModelVariant::Qwen317BGguf);
    }

    #[test]
    fn resolve_diarization_llm_rejects_other_chat_models() {
        assert!(resolve_diarization_llm_variant(Some("Qwen3-1.7B")).is_err());
        assert!(resolve_diarization_llm_variant(Some("google/gemma-3-1b-it")).is_err());
    }

    #[test]
    fn parse_qwen_chat_4b_repo_is_rejected() {
        assert!(parse_chat_model_variant(Some("Qwen/Qwen3-4B")).is_err());
    }

    #[test]
    fn parse_qwen_chat_4b_gguf_file_alias() {
        let parsed = parse_chat_model_variant(Some("Qwen3-4B-Q4_K_M.gguf")).unwrap();
        assert_eq!(parsed, ModelVariant::Qwen34BGguf);
    }

    #[test]
    fn parse_qwen_chat_8b_gguf_file_alias() {
        let parsed = parse_chat_model_variant(Some("Qwen3-8B-Q4_K_M.gguf")).unwrap();
        assert_eq!(parsed, ModelVariant::Qwen38BGguf);
    }

    #[test]
    fn parse_qwen_chat_14b_gguf_file_alias() {
        let parsed = parse_chat_model_variant(Some("Qwen3-14B-Q4_K_M.gguf")).unwrap();
        assert_eq!(parsed, ModelVariant::Qwen314BGguf);
    }

    #[test]
    fn parse_lfm2_audio_gguf_is_rejected() {
        assert!(parse_tts_model_variant("LFM2-Audio-1.5B-GGUF").is_err());
    }

    #[test]
    fn parse_kokoro_by_repo_id() {
        let parsed = parse_tts_model_variant("hexgrad/Kokoro-82M").unwrap();
        assert_eq!(parsed, ModelVariant::Kokoro82M);
    }

    #[test]
    fn parse_kokoro_by_display_name() {
        let parsed = parse_model_variant("Kokoro 82M").unwrap();
        assert_eq!(parsed, ModelVariant::Kokoro82M);
    }

    #[test]
    fn parse_lfm25_audio_gguf_is_rejected() {
        assert!(parse_tts_model_variant("LiquidAI/LFM2.5-Audio-1.5B-GGUF").is_err());
    }

    #[test]
    fn parse_forced_aligner_4bit() {
        let parsed = parse_model_variant("Qwen3-ForcedAligner-0.6B-4bit").unwrap();
        assert_eq!(parsed, ModelVariant::Qwen3ForcedAligner06B4Bit);
    }

    #[test]
    fn parse_diarization_by_repo_tail() {
        let parsed = parse_model_variant("diar_streaming_sortformer_4spk-v2.1").unwrap();
        assert_eq!(parsed, ModelVariant::DiarStreamingSortformer4SpkV21);
    }

    #[test]
    fn resolve_diarization_defaults_to_sortformer() {
        let resolved = resolve_diarization_model_variant(Some("unknown-model"));
        assert_eq!(resolved, ModelVariant::DiarStreamingSortformer4SpkV21);
    }
}
