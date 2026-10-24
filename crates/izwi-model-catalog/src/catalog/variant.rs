//! Model variant capability helpers and parser utilities.

use std::fmt;

use super::ModelVariant;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelFamily {
    Qwen3Tts,
    KokoroTts,
    VoxtralTts,
    VibeVoiceTts,
    FishS2Tts,
    ParakeetAsr,
    WhisperAsr,
    Qwen3Asr,
    VibeVoiceAsr,
    NemotronAsr,
    GraniteSpeechAsr,
    /// Granite Speech 5.0: the 4.1 encoder, projector and Granite LM shapes
    /// with a YaRN language model and its own prompt contract.
    GraniteSpeech5Asr,
    SortformerDiarization,
    Qwen3Chat,
    Qwen3MoeChat,
    Qwen35Chat,
    Qwen35MoeChat,
    Qwen38Chat,
    Lfm2Chat,
    Lfm25Audio,
    Gemma3Chat,
    Qwen3ForcedAligner,
    Voxtral,
    Tokenizer,
}

impl ModelFamily {
    /// Whether this family's ASR decoder can emit text for input that holds
    /// no speech. Generative decoders continue a language-model prompt, so
    /// silence or noise can come back as a plausible phrase ("you", "Okay.").
    /// Transducers emit blanks for frames without speech. Families without an
    /// offline ASR decoder report `false`.
    pub fn hallucinates_on_non_speech(self) -> bool {
        match self {
            Self::WhisperAsr
            | Self::Qwen3Asr
            | Self::VibeVoiceAsr
            | Self::GraniteSpeechAsr
            | Self::GraniteSpeech5Asr
            | Self::Lfm25Audio
            | Self::Voxtral => true,
            Self::ParakeetAsr | Self::NemotronAsr => false,
            Self::Qwen3Tts
            | Self::KokoroTts
            | Self::VoxtralTts
            | Self::VibeVoiceTts
            | Self::FishS2Tts
            | Self::SortformerDiarization
            | Self::Qwen3Chat
            | Self::Qwen3MoeChat
            | Self::Qwen35Chat
            | Self::Qwen35MoeChat
            | Self::Qwen38Chat
            | Self::Lfm2Chat
            | Self::Gemma3Chat
            | Self::Qwen3ForcedAligner
            | Self::Tokenizer => false,
        }
    }

    /// Whether a long-form transcription of this family runs one chunk per
    /// engine dispatch, so the device serves other work between the chunks
    /// of one file instead of holding it for the whole file. The chunks run
    /// on the chunk executor, whose state between chunks is host-only.
    /// VibeVoice's load-sealed state spec authenticates its long-form graph
    /// as one atomic stage; offline Voxtral and Nemotron plan their chunks
    /// inside the dispatch (no prepared long-form route); Parakeet never
    /// chunks. Those keep one dispatch per file.
    pub fn dispatches_long_form_asr_by_chunk(self) -> bool {
        matches!(
            self,
            Self::WhisperAsr
                | Self::Qwen3Asr
                | Self::GraniteSpeechAsr
                | Self::GraniteSpeech5Asr
                | Self::Lfm25Audio
        )
    }

    /// Whether this family's ASR encoder pads every input to one fixed window
    /// (Whisper's 30 s mel window). A shorter chunk then costs a full window,
    /// so the route planner never cuts such a model's audio below its window
    /// for latency.
    pub fn asr_fixed_encoder_window(self) -> bool {
        matches!(self, Self::WhisperAsr)
    }

    /// Whether one ASR invocation is meant to cover the model's whole audio
    /// window (VibeVoice's hour-long context, Parakeet's windowed encoder),
    /// rather than the shared ~30 s long-form chunk.
    pub fn asr_single_pass_spans_model_window(self) -> bool {
        matches!(self, Self::VibeVoiceAsr | Self::ParakeetAsr)
    }

    /// Whether the chunked ASR route can stream this family's text token by
    /// token while a chunk decodes. Granite's chunk decode replays earlier
    /// text as a prefix and may re-decode a looping chunk, so its chunk text is
    /// known only when the chunk finishes; Granite Speech 5.0 re-decodes a
    /// looping or truncated chunk the same way.
    pub fn asr_streams_within_chunk(self) -> bool {
        matches!(
            self,
            Self::WhisperAsr
                | Self::Qwen3Asr
                | Self::VibeVoiceAsr
                | Self::ParakeetAsr
                | Self::NemotronAsr
                | Self::Lfm25Audio
                | Self::Voxtral
        )
    }
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

impl ModelTask {
    /// Stable wire name used by runtime diagnostics and the admin API.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Tts => "tts",
            Self::Asr => "asr",
            Self::Diarization => "diarization",
            Self::Chat => "chat",
            Self::ForcedAlign => "forced_align",
            Self::AudioChat => "audio_chat",
            Self::Tokenizer => "tokenizer",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InferenceBackendHint {
    CandleNative,
}

#[derive(Debug, Clone)]
pub struct ParseModelVariantError {
    input: String,
    removed: Option<&'static RemovedModelVariant>,
}

impl ParseModelVariantError {
    fn new(input: impl Into<String>) -> Self {
        Self {
            input: input.into(),
            removed: None,
        }
    }

    fn removed(input: impl Into<String>, removed: &'static RemovedModelVariant) -> Self {
        Self {
            input: input.into(),
            removed: Some(removed),
        }
    }

    /// The removed variant the input named, if it named one.
    pub fn removed_variant(&self) -> Option<&'static RemovedModelVariant> {
        self.removed
    }
}

impl fmt::Display for ParseModelVariantError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.removed {
            Some(removed) => write!(
                f,
                "Model {} is no longer supported: {}. Use {} instead; {}",
                removed.id,
                removed.reason,
                removed.replacement.dir_name(),
                removed.not_substituted
            ),
            None => write!(
                f,
                "Unsupported model identifier: {}",
                self.input.trim().if_empty("<empty>")
            ),
        }
    }
}

/// A catalog variant whose support was removed. Requests that name it are
/// refused with its replacement, stored references to it normalize to the
/// replacement on read, and a leftover download stays listed (and
/// deletable) until the user removes it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RemovedModelVariant {
    /// The former variant id, which was also its model directory name.
    pub id: &'static str,
    pub repo_id: &'static str,
    pub display_name: &'static str,
    /// The supported variant that serves the same purpose.
    pub replacement: ModelVariant,
    pub reason: &'static str,
    /// Why a request naming it is refused rather than served by the
    /// replacement.
    pub not_substituted: &'static str,
}

const MLX_4BIT_QWEN3_TTS_REMOVAL_REASON: &str = "the mlx 4-bit Qwen3-TTS conversions never reach end of speech and produce unintelligible audio";

const MLX_4BIT_QWEN3_TTS_NOT_SUBSTITUTED: &str =
    "it needs more memory, so it is not substituted automatically";

/// The five mlx 4-bit Qwen3-TTS conversions (QTTS-22, decision 7) and the
/// mlx 4-bit forced aligner.
pub const REMOVED_MODEL_VARIANTS: &[RemovedModelVariant] = &[
    RemovedModelVariant {
        id: "Qwen3-TTS-12Hz-0.6B-Base-4bit",
        repo_id: "mlx-community/Qwen3-TTS-12Hz-0.6B-Base-4bit",
        display_name: "Qwen3-TTS 0.6B Base 4-bit",
        replacement: ModelVariant::Qwen3Tts12Hz06BBase,
        reason: MLX_4BIT_QWEN3_TTS_REMOVAL_REASON,
        not_substituted: MLX_4BIT_QWEN3_TTS_NOT_SUBSTITUTED,
    },
    RemovedModelVariant {
        id: "Qwen3-TTS-12Hz-0.6B-CustomVoice-4bit",
        repo_id: "mlx-community/Qwen3-TTS-12Hz-0.6B-CustomVoice-4bit",
        display_name: "Qwen3-TTS 0.6B CustomVoice 4-bit",
        replacement: ModelVariant::Qwen3Tts12Hz06BCustomVoice,
        reason: MLX_4BIT_QWEN3_TTS_REMOVAL_REASON,
        not_substituted: MLX_4BIT_QWEN3_TTS_NOT_SUBSTITUTED,
    },
    RemovedModelVariant {
        id: "Qwen3-TTS-12Hz-1.7B-Base-4bit",
        repo_id: "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-4bit",
        display_name: "Qwen3-TTS 1.7B Base 4-bit",
        replacement: ModelVariant::Qwen3Tts12Hz17BBase,
        reason: MLX_4BIT_QWEN3_TTS_REMOVAL_REASON,
        not_substituted: MLX_4BIT_QWEN3_TTS_NOT_SUBSTITUTED,
    },
    RemovedModelVariant {
        id: "Qwen3-TTS-12Hz-1.7B-CustomVoice-4bit",
        repo_id: "mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-4bit",
        display_name: "Qwen3-TTS 1.7B CustomVoice 4-bit",
        replacement: ModelVariant::Qwen3Tts12Hz17BCustomVoice,
        reason: MLX_4BIT_QWEN3_TTS_REMOVAL_REASON,
        not_substituted: MLX_4BIT_QWEN3_TTS_NOT_SUBSTITUTED,
    },
    RemovedModelVariant {
        id: "Qwen3-TTS-12Hz-1.7B-VoiceDesign-4bit",
        repo_id: "mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-4bit",
        display_name: "Qwen3-TTS 1.7B VoiceDesign 4-bit",
        replacement: ModelVariant::Qwen3Tts12Hz17BVoiceDesign,
        reason: MLX_4BIT_QWEN3_TTS_REMOVAL_REASON,
        not_substituted: MLX_4BIT_QWEN3_TTS_NOT_SUBSTITUTED,
    },
    RemovedModelVariant {
        id: "Qwen3-ForcedAligner-0.6B-4bit",
        repo_id: "mlx-community/Qwen3-ForcedAligner-0.6B-4bit",
        display_name: "Qwen3-ForcedAligner 0.6B 4-bit",
        replacement: ModelVariant::Qwen3ForcedAligner06B,
        reason: "the mlx 4-bit forced aligner is dequantized at load, so it saves download size but no memory over the full-precision aligner",
        not_substituted: "it is a larger download, so it is not substituted automatically",
    },
];

/// The removed variant `input` names: an exact former id, repo id or
/// display name, or any 4-bit Qwen3-TTS or forced-aligner spelling.
pub fn removed_model_variant(input: &str) -> Option<&'static RemovedModelVariant> {
    let normalized = normalize_identifier(input.trim());
    if normalized.is_empty() {
        return None;
    }
    if let Some(removed) = REMOVED_MODEL_VARIANTS.iter().find(|removed| {
        [removed.id, removed.repo_id, removed.display_name]
            .iter()
            .any(|alias| normalize_identifier(alias) == normalized)
    }) {
        return Some(removed);
    }
    let is_4bit = normalized.contains("4bit") || normalized.contains("int4");
    if normalized.contains("forcedaligner") && is_4bit {
        return REMOVED_MODEL_VARIANTS
            .iter()
            .find(|removed| removed.replacement == ModelVariant::Qwen3ForcedAligner06B);
    }
    let qwen_tts_4bit = normalized.contains("qwen3")
        && normalized.contains("tts")
        && !normalized.contains("tokenizer")
        && is_4bit;
    if !qwen_tts_4bit {
        return None;
    }
    let is_17b = normalized.contains("17b") || normalized.contains("17");
    let replacement = match (
        is_17b,
        normalized.contains("customvoice"),
        normalized.contains("voicedesign"),
    ) {
        (true, _, true) => ModelVariant::Qwen3Tts12Hz17BVoiceDesign,
        (true, true, _) => ModelVariant::Qwen3Tts12Hz17BCustomVoice,
        (true, _, _) => ModelVariant::Qwen3Tts12Hz17BBase,
        (false, true, _) => ModelVariant::Qwen3Tts12Hz06BCustomVoice,
        (false, _, _) => ModelVariant::Qwen3Tts12Hz06BBase,
    };
    REMOVED_MODEL_VARIANTS
        .iter()
        .find(|removed| removed.replacement == replacement)
}

/// A stored model id with any removed variant replaced by its supported
/// counterpart (the replacement's id). Stored settings, Studio projects and
/// queued jobs read through this, so data written before a removal keeps
/// working; history keeps its original label and does not.
pub fn normalize_stored_model_id(id: &str) -> std::borrow::Cow<'_, str> {
    match removed_model_variant(id) {
        Some(removed) => std::borrow::Cow::Borrowed(removed.replacement.dir_name()),
        None => std::borrow::Cow::Borrowed(id),
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
            | Qwen3Tts12Hz06BBase8Bit
            | Qwen3Tts12Hz06BBaseBf16
            | Qwen3Tts12Hz06BCustomVoice
            | Qwen3Tts12Hz06BCustomVoice8Bit
            | Qwen3Tts12Hz06BCustomVoiceBf16
            | Qwen3Tts12Hz17BBase
            | Qwen3Tts12Hz17BCustomVoice
            | Qwen3Tts12Hz17BVoiceDesign
            | Qwen3Tts12Hz17BVoiceDesign8Bit
            | Qwen3Tts12Hz17BVoiceDesignBf16 => ModelFamily::Qwen3Tts,
            Kokoro82M => ModelFamily::KokoroTts,
            Voxtral4BTts2603 => ModelFamily::VoxtralTts,
            VibeVoice15BTts => ModelFamily::VibeVoiceTts,
            FishAudioS2Pro => ModelFamily::FishS2Tts,
            Qwen3TtsTokenizer12Hz => ModelFamily::Tokenizer,
            ParakeetTdt06BV3 => ModelFamily::ParakeetAsr,
            WhisperLargeV3Turbo => ModelFamily::WhisperAsr,
            Qwen3Asr06BGguf | Qwen3Asr17BGguf => ModelFamily::Qwen3Asr,
            VibeVoiceAsr => ModelFamily::VibeVoiceAsr,
            Nemotron35AsrStreaming06B => ModelFamily::NemotronAsr,
            GraniteSpeech412B | GraniteSpeech412BPlus => ModelFamily::GraniteSpeechAsr,
            GraniteSpeech502B | GraniteSpeech502BGguf => ModelFamily::GraniteSpeech5Asr,
            DiarStreamingSortformer4SpkV21 | Nemotron3Diarization => {
                ModelFamily::SortformerDiarization
            }
            Qwen306B | Qwen306B4Bit | Qwen317B | Qwen317B4Bit | Qwen314BGguf => {
                ModelFamily::Qwen3Chat
            }
            Qwen3Moe30bA3bGguf => ModelFamily::Qwen3MoeChat,
            Qwen3508BGguf | Qwen352BGguf | Qwen354BGguf | Qwen359BGguf => ModelFamily::Qwen35Chat,
            Qwen36Moe35BA3BFp8 => ModelFamily::Qwen35MoeChat,
            Qwen3827BFp8 => ModelFamily::Qwen38Chat,
            Lfm2512BInstructGguf | Lfm2512BThinkingGguf => ModelFamily::Lfm2Chat,
            Lfm25Audio15BGguf => ModelFamily::Lfm25Audio,
            Gemma34BIt => ModelFamily::Gemma3Chat,
            Qwen3ForcedAligner06B => ModelFamily::Qwen3ForcedAligner,
            VoxtralMini4BRealtime2602 => ModelFamily::Voxtral,
        }
    }

    pub fn primary_task(&self) -> ModelTask {
        match self.family() {
            ModelFamily::Qwen3Tts
            | ModelFamily::KokoroTts
            | ModelFamily::VoxtralTts
            | ModelFamily::VibeVoiceTts
            | ModelFamily::FishS2Tts => ModelTask::Tts,
            ModelFamily::ParakeetAsr
            | ModelFamily::WhisperAsr
            | ModelFamily::Qwen3Asr
            | ModelFamily::VibeVoiceAsr
            | ModelFamily::NemotronAsr
            | ModelFamily::GraniteSpeechAsr
            | ModelFamily::GraniteSpeech5Asr => ModelTask::Asr,
            ModelFamily::SortformerDiarization => ModelTask::Diarization,
            ModelFamily::Qwen3Chat
            | ModelFamily::Qwen3MoeChat
            | ModelFamily::Qwen35Chat
            | ModelFamily::Qwen35MoeChat
            | ModelFamily::Qwen38Chat
            | ModelFamily::Lfm2Chat
            | ModelFamily::Gemma3Chat => ModelTask::Chat,
            ModelFamily::Qwen3ForcedAligner => ModelTask::ForcedAlign,
            ModelFamily::Voxtral => ModelTask::Asr,
            ModelFamily::Lfm25Audio => ModelTask::AudioChat,
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

    if let Some(removed) = removed_model_variant(trimmed) {
        return Err(ParseModelVariantError::removed(input, removed));
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
    if variant.is_tts() || variant.is_audio_chat() {
        Ok(variant)
    } else {
        Err(ParseModelVariantError::new(input))
    }
}

pub fn parse_chat_model_variant(
    input: Option<&str>,
) -> Result<ModelVariant, ParseModelVariantError> {
    let id = input.unwrap_or("Qwen3.5-4B");
    let variant = parse_model_variant(id)?;
    if variant.is_chat() {
        Ok(variant)
    } else {
        Err(ParseModelVariantError::new(id))
    }
}

/// Resolve the LLM variant that summarizes a diarization transcript. Any
/// chat-capable catalog model is accepted: the summary prompt is
/// model-agnostic. Defaults to Qwen3.5-4B. Legacy records may still
/// carry "Qwen3-1.7B-GGUF" (removed from the catalog); that id maps onto the
/// current default instead of erroring on previously stored settings.
pub fn resolve_diarization_llm_variant(
    input: Option<&str>,
) -> Result<ModelVariant, ParseModelVariantError> {
    let legacy_removed = input.is_some_and(|raw| {
        let trimmed = raw.trim();
        let tail = trimmed.rsplit('/').next().unwrap_or(trimmed);
        normalize_identifier(tail) == normalize_identifier("Qwen3-1.7B-GGUF")
    });
    if legacy_removed {
        return parse_chat_model_variant(Some("Qwen3.5-4B"));
    }
    parse_chat_model_variant(input)
}

pub fn resolve_asr_model_variant(input: Option<&str>) -> ModelVariant {
    use ModelVariant::*;

    let Some(raw) = input else {
        return ParakeetTdt06BV3;
    };

    match parse_model_variant(raw) {
        Ok(variant) if variant.is_asr() || variant.is_voxtral() || variant.is_audio_chat() => {
            variant
        }
        Ok(_) => ParakeetTdt06BV3,
        Err(_) => {
            let normalized = normalize_identifier(raw);
            if normalized.contains("voxtral") {
                VoxtralMini4BRealtime2602
            } else if normalized.contains("nemotron") && normalized.contains("asr") {
                Nemotron35AsrStreaming06B
            } else if let Some(granite_speech) = resolve_granite_speech_variant(&normalized) {
                granite_speech
            } else if normalized.contains("vibevoice") && normalized.contains("asr") {
                VibeVoiceAsr
            } else if normalized.contains("whisper")
                && normalized.contains("largev3")
                && normalized.contains("turbo")
            {
                WhisperLargeV3Turbo
            } else if let Some(qwen_asr) = resolve_qwen3_asr_variant(&normalized) {
                qwen_asr
            } else {
                ParakeetTdt06BV3
            }
        }
    }
}

/// Resolve the ASR model of the diarization transcript pipeline. The pipeline
/// transcribes through the native ASR registry (or its Voxtral route) with an
/// atomic long-form graph, so audio-chat models are not accepted. An explicit
/// id that is unknown or not such a model is an error, never a silent
/// fallback to the default.
pub fn resolve_diarization_asr_model_variant(input: Option<&str>) -> Result<ModelVariant, String> {
    let Some(raw) = input else {
        return Ok(ModelVariant::ParakeetTdt06BV3);
    };
    match parse_model_variant(raw) {
        Ok(variant) if variant.is_asr() || variant.is_voxtral() => Ok(variant),
        Ok(variant) => Err(format!(
            "Model '{raw}' cannot transcribe in the diarization pipeline (resolved to {variant:?}); choose an ASR model."
        )),
        Err(_) => Err(format!("Unknown diarization ASR model id '{raw}'.")),
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
        Err(_) => DiarStreamingSortformer4SpkV21,
    }
}

/// Refuses diarization speaker limits a model with `channels` speaker
/// channels cannot meet (PIPE-18): a bound above the channel count, which
/// used to be clamped silently, or `min_speakers` above `max_speakers`.
pub fn validate_diarization_speaker_limits(
    min_speakers: Option<usize>,
    max_speakers: Option<usize>,
    channels: usize,
) -> Result<(), String> {
    for (field, value) in [
        ("min_speakers", min_speakers),
        ("max_speakers", max_speakers),
    ] {
        if let Some(value) = value.filter(|value| *value > channels) {
            return Err(format!(
                "`{field}` = {value} exceeds the {channels} speaker channels of this diarization model."
            ));
        }
    }
    if let (Some(min), Some(max)) = (min_speakers, max_speakers) {
        if min > max {
            return Err("`min_speakers` cannot be greater than `max_speakers`.".to_string());
        }
    }
    Ok(())
}

/// Strict resolution for an explicitly provided diarization model id: the
/// caller's pick is the source of truth, so an id that does not parse to a
/// diarization variant is an error instead of a silent fallback to the
/// legacy default.
pub fn resolve_diarization_model_variant_strict(
    input: &str,
) -> Result<ModelVariant, String> {
    match parse_model_variant(input) {
        Ok(variant) if variant.is_diarization() => Ok(variant),
        Ok(variant) => Err(format!(
            "Model '{input}' is not a diarization model (resolved to {variant:?})."
        )),
        Err(_) => Err(format!("Unknown diarization model id '{input}'.")),
    }
}

fn resolve_by_heuristic(normalized: &str) -> Option<ModelVariant> {
    use ModelVariant::*;

    if normalized.contains("voxtral") {
        if normalized.contains("tts")
            || normalized.contains("texttospeech")
            || normalized.contains("2603")
        {
            return Some(Voxtral4BTts2603);
        }
        return Some(VoxtralMini4BRealtime2602);
    }

    if normalized.contains("vibevoice") {
        if normalized.contains("asr") {
            return Some(VibeVoiceAsr);
        }
        if normalized.contains("15b")
            || normalized.contains("1dot5b")
            || normalized.contains("tts")
            || normalized.contains("texttospeech")
        {
            return Some(VibeVoice15BTts);
        }
    }

    if normalized.contains("fishaudio")
        || normalized.contains("fishspeech")
        || (normalized.contains("fish") && normalized.contains("s2"))
    {
        return Some(FishAudioS2Pro);
    }

    if normalized.contains("nemotron") && normalized.contains("asr") {
        return Some(Nemotron35AsrStreaming06B);
    }

    if normalized.contains("nemotron") && normalized.contains("diar") {
        return Some(Nemotron3Diarization);
    }

    if let Some(granite_speech) = resolve_granite_speech_variant(normalized) {
        return Some(granite_speech);
    }

    // `sortformer4spk` covers the short id older CLIs sent (PIPE-3).
    if normalized.contains("sortformer")
        && (normalized.contains("diar") || normalized == "sortformer4spk")
    {
        return Some(DiarStreamingSortformer4SpkV21);
    }

    if normalized.contains("parakeet") && normalized.contains("tdt") {
        return Some(ParakeetTdt06BV3);
    }

    if normalized.contains("whisper")
        && normalized.contains("largev3")
        && normalized.contains("turbo")
    {
        return Some(WhisperLargeV3Turbo);
    }

    if normalized.contains("forcedaligner") {
        return Some(Qwen3ForcedAligner06B);
    }

    if let Some(qwen_asr) = resolve_qwen3_asr_variant(normalized) {
        return Some(qwen_asr);
    }

    if normalized.contains("qwen3") && normalized.contains("tts") {
        let is_17b = normalized.contains("17b") || normalized.contains("17");
        // 4-bit ids name removed variants; `removed_model_variant` refuses
        // them before this heuristic, and none may resolve to a dense one.
        if normalized.contains("4bit") || normalized.contains("int4") {
            return None;
        }
        let q8 = normalized.contains("8bit") || normalized.contains("int8");
        let bf16 = normalized.contains("bf16") || normalized.contains("bfloat16");

        if normalized.contains("tokenizer") {
            return Some(Qwen3TtsTokenizer12Hz);
        }

        if is_17b && normalized.contains("voicedesign") {
            return Some(match (q8, bf16) {
                (true, _) => Qwen3Tts12Hz17BVoiceDesign8Bit,
                (_, true) => Qwen3Tts12Hz17BVoiceDesignBf16,
                _ => Qwen3Tts12Hz17BVoiceDesign,
            });
        }

        if is_17b && normalized.contains("customvoice") {
            return Some(Qwen3Tts12Hz17BCustomVoice);
        }

        if is_17b {
            return Some(Qwen3Tts12Hz17BBase);
        }

        if normalized.contains("customvoice") {
            return Some(match (q8, bf16) {
                (true, _) => Qwen3Tts12Hz06BCustomVoice8Bit,
                (_, true) => Qwen3Tts12Hz06BCustomVoiceBf16,
                _ => Qwen3Tts12Hz06BCustomVoice,
            });
        }

        return Some(match (q8, bf16) {
            (true, _) => Qwen3Tts12Hz06BBase8Bit,
            (_, true) => Qwen3Tts12Hz06BBaseBf16,
            _ => Qwen3Tts12Hz06BBase,
        });
    }

    if normalized.contains("kokoro") && normalized.contains("82m") {
        return Some(Kokoro82M);
    }

    if let Some(qwen35_variant) = resolve_qwen35_chat_variant(normalized) {
        return Some(qwen35_variant);
    }

    if normalized.contains("qwen38")
        && normalized.contains("27b")
        && !normalized.contains("asr")
        && !normalized.contains("tts")
    {
        return Some(Qwen3827BFp8);
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
        // The 0.6B/1.7B/4B/8B Qwen3 GGUF variants were removed from the
        // catalog: ids that only match them must fail to parse rather than
        // re-resolve to a different (e.g. MLX native) variant.
        if is_4b {
            return None;
        }
        if is_17b {
            return if gguf {
                None
            } else if q4 {
                Some(Qwen317B4Bit)
            } else {
                Some(Qwen317B)
            };
        }
        if is_8b {
            return None;
        }
        if normalized.contains("06b") || normalized.contains("0dot6b") || normalized.contains("06")
        {
            return if gguf {
                None
            } else if q4 {
                Some(Qwen306B4Bit)
            } else {
                Some(Qwen306B)
            };
        }
    }

    if normalized.contains("gemma3") || (normalized.contains("gemma") && normalized.contains("it"))
    {
        // Gemma-3-1b-it was removed from the catalog: its ids must fail to
        // parse rather than re-resolve to another variant.
        if normalized.contains("1b") {
            return None;
        }
        if normalized.contains("4b") {
            return Some(Gemma34BIt);
        }
    }

    if let Some(lfm2_variant) = resolve_lfm2_chat_variant(normalized) {
        return Some(lfm2_variant);
    }

    if let Some(lfm25_audio_variant) = resolve_lfm25_audio_variant(normalized) {
        return Some(lfm25_audio_variant);
    }

    None
}

fn resolve_qwen3_asr_variant(normalized: &str) -> Option<ModelVariant> {
    use ModelVariant::*;

    if !normalized.contains("qwen3") || !normalized.contains("asr") {
        return None;
    }
    if normalized.contains("forcedaligner") || normalized.contains("tts") {
        return None;
    }

    if normalized.contains("17b")
        || normalized.contains("1dot7b")
        || normalized.contains("17")
        || normalized.contains("qwen3asr17")
    {
        return Some(Qwen3Asr17BGguf);
    }

    if normalized.contains("06b")
        || normalized.contains("0dot6b")
        || normalized.contains("06")
        || normalized.contains("qwen3asr06")
    {
        return Some(Qwen3Asr06BGguf);
    }

    Some(Qwen3Asr06BGguf)
}

fn resolve_granite_speech_variant(normalized: &str) -> Option<ModelVariant> {
    use ModelVariant::*;

    if !normalized.contains("granite") || !normalized.contains("speech") {
        return None;
    }

    if normalized.contains("chat") || normalized.contains("code") || normalized.contains("vision") {
        return None;
    }

    // Granite Speech 5.0 is its own family, so a 5.0 id must never fall
    // through to the 4.1 checkpoints below. 5.0's Turbo CTC sibling is a
    // different model that the catalog does not carry.
    if normalized.contains("speech5") {
        if ["turbo", "ctc", "470m"]
            .iter()
            .any(|other| normalized.contains(other))
        {
            return None;
        }
        // Of IBM's GGUF quantizations the catalog carries Q8_0 only.
        if normalized.contains("gguf") || normalized.contains("q80") {
            return Some(GraniteSpeech502BGguf);
        }
        if ["q2k", "q3k", "q4", "q5", "q6k", "bf16"]
            .iter()
            .any(|quant| normalized.contains(quant))
        {
            return None;
        }
        return Some(GraniteSpeech502B);
    }

    // IBM's library resolves local checkpoints the same way: "plus" names the
    // rich-transcription model, anything else the punctuating base model,
    // which is IBM's default.
    if normalized.contains("plus") {
        return Some(GraniteSpeech412BPlus);
    }

    Some(GraniteSpeech412B)
}

fn resolve_qwen35_chat_variant(normalized: &str) -> Option<ModelVariant> {
    use ModelVariant::*;

    // "qwen36" resolves here too: Qwen3.6-35B-A3B-FP8 shares the qwen3_5_moe
    // architecture family with the Qwen3.5 chat GGUF variants.
    if !normalized.contains("qwen35") && !normalized.contains("qwen36") {
        return None;
    }

    if normalized.contains("asr")
        || normalized.contains("tts")
        || normalized.contains("forcedaligner")
    {
        return None;
    }

    if normalized.contains("qwen36") {
        // The Qwen3.6 line only ships the 35B-A3B MoE checkpoint.
        return if normalized.contains("35b") {
            Some(Qwen36Moe35BA3BFp8)
        } else {
            None
        };
    }

    // The Qwen3.5-35B-A3B-FP8 variant was replaced by Qwen3.6-35B-A3B-FP8:
    // legacy 3.5 35B ids must fail to parse rather than silently resolve to
    // the replacement checkpoint.
    if normalized.contains("35b") {
        return None;
    }
    if normalized.contains("09b") || normalized.contains("9b") {
        return Some(Qwen359BGguf);
    }
    if normalized.contains("08b") || normalized.contains("08") {
        return Some(Qwen3508BGguf);
    }
    if normalized.contains("2b") {
        return Some(Qwen352BGguf);
    }
    if normalized.contains("4b") {
        return Some(Qwen354BGguf);
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

fn resolve_lfm25_audio_variant(normalized: &str) -> Option<ModelVariant> {
    use ModelVariant::*;

    if !normalized.contains("audio") {
        return None;
    }

    if !normalized.contains("lfm25") && !normalized.contains("lfm2dot5") {
        return None;
    }

    if !(normalized.contains("15b") || normalized.contains("15")) {
        return None;
    }

    if !(normalized.contains("gguf") || normalized.contains("q40")) {
        return None;
    }

    Some(Lfm25Audio15BGguf)
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

    let maybe_compact = raw.trim().replace(['_', ' '], "-").to_ascii_lowercase();

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
    fn diarization_asr_resolution_rejects_unknown_and_non_native_ids() {
        assert_eq!(
            resolve_diarization_asr_model_variant(None),
            Ok(ModelVariant::ParakeetTdt06BV3)
        );
        assert_eq!(
            resolve_diarization_asr_model_variant(Some("parakeet-tdt-0.6b-v3")),
            Ok(ModelVariant::ParakeetTdt06BV3)
        );
        assert_eq!(
            resolve_diarization_asr_model_variant(Some("Whisper-Large-v3-Turbo")),
            Ok(ModelVariant::WhisperLargeV3Turbo)
        );
        assert_eq!(
            resolve_diarization_asr_model_variant(Some("Voxtral-Mini-4B-Realtime-2602")),
            Ok(ModelVariant::VoxtralMini4BRealtime2602)
        );
        for rejected in [
            "bogus",
            "LFM2.5-Audio-1.5B-GGUF",
            "Qwen3-ForcedAligner-0.6B",
            "Qwen3.5-4B",
        ] {
            assert!(
                resolve_diarization_asr_model_variant(Some(rejected)).is_err(),
                "{rejected} must not silently become the default pipeline ASR"
            );
        }
    }

    #[test]
    fn parse_by_repo_tail() {
        let parsed = parse_model_variant("Parakeet-TDT-0.6B-v3").unwrap();
        assert_eq!(parsed, ModelVariant::ParakeetTdt06BV3);
    }

    #[test]
    fn parse_by_display_name() {
        let parsed = parse_model_variant("Qwen3-TTS 0.6B Base").unwrap();
        assert_eq!(parsed, ModelVariant::Qwen3Tts12Hz06BBase);
    }

    /// The UI's route preference lists name only catalog variants that are
    /// enabled, by their exact id: a removed or catalog-disabled id there
    /// silently falls through to a later choice.
    #[test]
    fn ui_route_preferred_models_exist_in_the_catalog_and_are_enabled() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../ui/src/features/models/catalog/routeModelCatalog.ts");
        let source = std::fs::read_to_string(&path).expect("UI route catalog");
        let mut checked = 0;
        for block in source.split("export const ").skip(1) {
            let Some((name, rest)) = block.split_once(" = [") else {
                continue;
            };
            if !name.contains("PREFERRED") {
                continue;
            }
            let list = rest.split_once("] as const").expect(name).0;
            for id in list.split('"').skip(1).step_by(2) {
                let variant = parse_model_variant(id)
                    .unwrap_or_else(|error| panic!("{name} names {id}: {error}"));
                assert_eq!(variant.dir_name(), id, "{name} must use the exact id");
                assert!(variant.is_enabled(), "{name} names disabled {id}");
                checked += 1;
            }
        }
        assert!(checked >= 10, "parsed only {checked} preferred ids");
    }

    /// P1.0: the mlx 4-bit Qwen3-TTS variants are removed. Every spelling of
    /// them is refused with its dense replacement named, none resolves to a
    /// catalog variant, and stored ids normalize to the replacement.
    #[test]
    fn removed_mlx_4bit_qwen_tts_ids_are_refused_with_their_dense_replacement() {
        let cases = [
            (
                "Qwen3-TTS-12Hz-0.6B-Base-4bit",
                ModelVariant::Qwen3Tts12Hz06BBase,
            ),
            (
                "mlx-community/Qwen3-TTS-12Hz-0.6B-CustomVoice-4bit",
                ModelVariant::Qwen3Tts12Hz06BCustomVoice,
            ),
            (
                "Qwen3-TTS 1.7B Base 4-bit",
                ModelVariant::Qwen3Tts12Hz17BBase,
            ),
            (
                "qwen3-tts-1.7b-customvoice-int4",
                ModelVariant::Qwen3Tts12Hz17BCustomVoice,
            ),
            (
                "Qwen3-TTS-12Hz-1.7B-VoiceDesign-4bit",
                ModelVariant::Qwen3Tts12Hz17BVoiceDesign,
            ),
        ];
        for (input, replacement) in cases {
            let error = parse_model_variant(input).expect_err(input);
            assert_eq!(
                error.removed_variant().unwrap().replacement,
                replacement,
                "{input}"
            );
            assert!(
                error.to_string().contains(replacement.dir_name()),
                "{error}"
            );
            assert!(parse_tts_model_variant(input).is_err(), "{input}");
            assert_eq!(normalize_stored_model_id(input), replacement.dir_name());
        }
        for removed in REMOVED_MODEL_VARIANTS {
            assert!(parse_model_variant(removed.id).is_err());
            assert!(removed.replacement.is_enabled());
        }
        assert!(ModelVariant::all().iter().all(|variant| {
            !(variant.family() == ModelFamily::Qwen3Tts && variant.dir_name().contains("4bit"))
                && removed_model_variant(variant.dir_name()).is_none()
        }));
        // Supported ids are untouched.
        assert_eq!(normalize_stored_model_id("Kokoro-82M"), "Kokoro-82M");
    }

    /// The mlx 4-bit forced aligner is removed: every spelling is refused
    /// with the full-precision aligner named (without the TTS memory
    /// note), and stored ids normalize to it.
    #[test]
    fn removed_4bit_forced_aligner_ids_are_refused_with_the_full_aligner() {
        for input in [
            "Qwen3-ForcedAligner-0.6B-4bit",
            "mlx-community/Qwen3-ForcedAligner-0.6B-4bit",
            "Qwen3-ForcedAligner 0.6B 4-bit",
            "qwen3-forcedaligner-0.6b-int4",
        ] {
            let error = parse_model_variant(input).expect_err(input);
            assert_eq!(
                error.removed_variant().unwrap().replacement,
                ModelVariant::Qwen3ForcedAligner06B,
                "{input}"
            );
            let message = error.to_string();
            assert!(message.contains("Use Qwen3-ForcedAligner-0.6B instead"), "{message}");
            assert!(!message.contains("needs more memory"), "{message}");
            assert_eq!(normalize_stored_model_id(input), "Qwen3-ForcedAligner-0.6B");
        }
        assert_eq!(
            parse_model_variant("Qwen3-ForcedAligner-0.6B").unwrap(),
            ModelVariant::Qwen3ForcedAligner06B
        );
        assert_eq!(
            normalize_stored_model_id("Qwen/Qwen3-ForcedAligner-0.6B"),
            "Qwen/Qwen3-ForcedAligner-0.6B"
        );
    }

    #[test]
    fn parse_tts_rejects_non_tts() {
        assert!(parse_tts_model_variant("Parakeet-TDT-0.6B-v3").is_err());
    }

    #[test]
    fn parse_tts_accepts_lfm25_audio() {
        let parsed = parse_tts_model_variant("LiquidAI/LFM2.5-Audio-1.5B-GGUF")
            .expect("lfm25 audio should parse for tts");
        assert_eq!(parsed, ModelVariant::Lfm25Audio15BGguf);
    }

    #[test]
    fn parse_tts_accepts_voxtral_tts_repo_alias() {
        let parsed = parse_tts_model_variant("mistralai/Voxtral-4B-TTS-2603")
            .expect("Voxtral TTS should parse for tts");
        assert_eq!(parsed, ModelVariant::Voxtral4BTts2603);
        assert_eq!(parsed.family(), ModelFamily::VoxtralTts);
    }

    #[test]
    fn parse_tts_accepts_vibevoice_repo_alias() {
        let parsed = parse_tts_model_variant("microsoft/VibeVoice-1.5B")
            .expect("VibeVoice TTS should parse for tts");
        assert_eq!(parsed, ModelVariant::VibeVoice15BTts);
        assert_eq!(parsed.family(), ModelFamily::VibeVoiceTts);
    }

    #[test]
    fn parse_tts_accepts_fish_s2_aliases() {
        for alias in [
            "FishAudio-S2-Pro",
            "fishaudio/s2-pro",
            "s2-pro",
            "Fish Audio S2 Pro",
        ] {
            let parsed =
                parse_tts_model_variant(alias).expect("Fish Audio S2 Pro should parse for tts");
            assert_eq!(parsed, ModelVariant::FishAudioS2Pro);
            assert_eq!(parsed.family(), ModelFamily::FishS2Tts);
        }
    }

    #[test]
    fn resolve_asr_fallback_defaults_to_parakeet_v3() {
        let resolved = resolve_asr_model_variant(Some("not-a-real-model"));
        assert_eq!(resolved, ModelVariant::ParakeetTdt06BV3);
    }

    #[test]
    fn resolve_asr_accepts_qwen3_asr_family_alias() {
        let resolved = resolve_asr_model_variant(Some("Qwen3-ASR-1.7B"));
        assert_eq!(resolved, ModelVariant::Qwen3Asr17BGguf);
    }

    #[test]
    fn resolve_asr_accepts_vibevoice_aliases() {
        let resolved = resolve_asr_model_variant(Some("microsoft/VibeVoice-ASR"));
        assert_eq!(resolved, ModelVariant::VibeVoiceAsr);
        assert_eq!(resolved.family(), ModelFamily::VibeVoiceAsr);
    }

    #[test]
    fn parse_nemotron_asr_repo_alias() {
        let parsed = parse_model_variant("nvidia/nemotron-3.5-asr-streaming-0.6b").unwrap();
        assert_eq!(parsed, ModelVariant::Nemotron35AsrStreaming06B);
        assert_eq!(parsed.family(), ModelFamily::NemotronAsr);
    }

    #[test]
    fn resolve_asr_accepts_nemotron_family_alias() {
        let resolved = resolve_asr_model_variant(Some("nemotron asr streaming"));
        assert_eq!(resolved, ModelVariant::Nemotron35AsrStreaming06B);
    }

    #[test]
    fn parse_granite_speech_repo_alias() {
        let parsed = parse_model_variant("ibm-granite/granite-speech-4.1-2b-plus").unwrap();
        assert_eq!(parsed, ModelVariant::GraniteSpeech412BPlus);
        assert_eq!(parsed.family(), ModelFamily::GraniteSpeechAsr);
    }

    #[test]
    fn resolve_asr_accepts_granite_speech_family_alias() {
        let resolved = resolve_asr_model_variant(Some("granite speech plus"));
        assert_eq!(resolved, ModelVariant::GraniteSpeech412BPlus);
    }

    #[test]
    fn granite_speech_names_resolve_to_base_unless_they_say_plus() {
        for (input, expected) in [
            (
                "ibm-granite/granite-speech-4.1-2b",
                ModelVariant::GraniteSpeech412B,
            ),
            ("granite-speech-4.1-2b", ModelVariant::GraniteSpeech412B),
            ("Granite-Speech-4.1-2B", ModelVariant::GraniteSpeech412B),
            ("Granite Speech 4.1 2B", ModelVariant::GraniteSpeech412B),
            (
                "ibm-granite/granite-speech-4.1-2b-plus",
                ModelVariant::GraniteSpeech412BPlus,
            ),
            (
                "Granite-Speech-4.1-2B-Plus",
                ModelVariant::GraniteSpeech412BPlus,
            ),
        ] {
            assert_eq!(parse_model_variant(input).unwrap(), expected, "{input}");
        }
        for (input, expected) in [
            ("granite speech", ModelVariant::GraniteSpeech412B),
            ("granite-speech-4.1", ModelVariant::GraniteSpeech412B),
            ("granite_speech_2b", ModelVariant::GraniteSpeech412B),
            ("granite speech plus", ModelVariant::GraniteSpeech412BPlus),
            (
                "granite-speech-4.1-2b-plus-local",
                ModelVariant::GraniteSpeech412BPlus,
            ),
        ] {
            assert_eq!(resolve_asr_model_variant(Some(input)), expected, "{input}");
        }
    }

    #[test]
    fn granite_speech_5_names_never_resolve_to_a_4_1_checkpoint() {
        for input in [
            "ibm-granite/granite-speech-5.0-2b",
            "granite-speech-5.0-2b",
            "Granite-Speech-5.0-2B",
            "Granite Speech 5.0 2B",
        ] {
            assert_eq!(
                parse_model_variant(input).unwrap(),
                ModelVariant::GraniteSpeech502B,
                "{input}"
            );
        }
        for input in [
            "granite speech 5",
            "granite-speech-5.0",
            "granite_speech_5_2b",
            "granite-speech-5.0-2b-local",
        ] {
            assert_eq!(
                resolve_asr_model_variant(Some(input)),
                ModelVariant::GraniteSpeech502B,
                "{input}"
            );
        }
        // The Turbo CTC sibling is not this model.
        assert!(parse_model_variant("granite-speech-5.0-470m-turboctc").is_err());
        // The GGUF release is its own variant of the same family; only its
        // Q8_0 quantization is carried.
        for input in [
            "Granite-Speech-5.0-2B-GGUF",
            "ibm-granite/granite-speech-5.0-2b-GGUF",
            "granite-speech-5.0-2b-Q8_0.gguf",
        ] {
            assert_eq!(
                parse_model_variant(input).unwrap(),
                ModelVariant::GraniteSpeech502BGguf,
                "{input}"
            );
        }
        assert!(parse_model_variant("granite-speech-5.0-2b-Q4_K_M").is_err());
        assert_eq!(
            ModelVariant::GraniteSpeech502BGguf.family(),
            ModelFamily::GraniteSpeech5Asr
        );
        // 4.1 ids keep resolving to 4.1.
        assert_eq!(
            resolve_asr_model_variant(Some("granite-speech-4.1")),
            ModelVariant::GraniteSpeech412B
        );
        assert_eq!(
            ModelVariant::GraniteSpeech502B.family(),
            ModelFamily::GraniteSpeech5Asr
        );
        assert!(ModelFamily::GraniteSpeech5Asr.hallucinates_on_non_speech());
        assert!(ModelFamily::GraniteSpeech5Asr.dispatches_long_form_asr_by_chunk());
        assert!(!ModelFamily::GraniteSpeech5Asr.asr_streams_within_chunk());
    }

    #[test]
    fn parse_gemma_by_repo_tail() {
        let parsed = parse_model_variant("gemma-3-4b-it").unwrap();
        assert_eq!(parsed, ModelVariant::Gemma34BIt);
    }

    #[test]
    fn parse_qwen35_chat_repo_alias() {
        let parsed = parse_chat_model_variant(Some("unsloth/Qwen3.5-4B-GGUF")).unwrap();
        assert_eq!(parsed, ModelVariant::Qwen354BGguf);
        assert_eq!(parsed.family(), ModelFamily::Qwen35Chat);
    }

    #[test]
    fn parse_qwen35_chat_base_name() {
        let parsed = parse_chat_model_variant(Some("Qwen3.5-0.8B")).unwrap();
        assert_eq!(parsed, ModelVariant::Qwen3508BGguf);
    }

    #[test]
    fn parse_qwen35_chat_q4_file_alias() {
        let parsed = parse_chat_model_variant(Some("Qwen3.5-9B-Q4_K_M.gguf")).unwrap();
        assert_eq!(parsed, ModelVariant::Qwen359BGguf);
    }

    #[test]
    fn parse_qwen36_moe_fp8_chat_aliases() {
        for alias in [
            "Qwen3.6-35B-A3B-FP8",
            "Qwen/Qwen3.6-35B-A3B-FP8",
            "Qwen3.6 35B-A3B FP8",
            "Qwen3.6-35B-A3B",
        ] {
            let parsed = parse_chat_model_variant(Some(alias)).expect("Qwen3.6 MoE chat alias");
            assert_eq!(parsed, ModelVariant::Qwen36Moe35BA3BFp8, "alias {alias}");
            assert_eq!(parsed.family(), ModelFamily::Qwen35MoeChat);
            assert_eq!(parsed.primary_task(), ModelTask::Chat);
        }
    }

    #[test]
    fn parse_rejected_qwen35_moe_ids_fail_closed() {
        // Qwen3.5-35B-A3B-FP8 was replaced by the Qwen3.6 checkpoint: its ids
        // must fail to parse rather than re-resolve to the replacement model.
        for removed in [
            "Qwen3.5-35B-A3B-FP8",
            "Qwen/Qwen3.5-35B-A3B-FP8",
            "Qwen3.5 35B-A3B FP8",
            "Qwen3.5-35B-A3B",
        ] {
            assert!(
                parse_chat_model_variant(Some(removed)).is_err(),
                "removed id {removed} must not parse"
            );
        }
    }

    #[test]
    fn parse_qwen38_fp8_chat_aliases() {
        for alias in [
            "Qwen3.8-27B-FP8",
            "Qwen/Qwen3.8-27B-FP8",
            "Qwen3.8 27B FP8",
            "Qwen3.8-27B",
        ] {
            let parsed = parse_chat_model_variant(Some(alias)).expect("Qwen3.8 chat alias");
            assert_eq!(parsed, ModelVariant::Qwen3827BFp8, "alias {alias}");
            assert_eq!(parsed.family(), ModelFamily::Qwen38Chat);
            assert_eq!(parsed.primary_task(), ModelTask::Chat);
        }
    }

    #[test]
    fn parse_chat_rejects_removed_gemma_1b() {
        // Gemma-3-1b-it left the catalog: its ids must fail to parse.
        assert!(parse_chat_model_variant(Some("Gemma-3-1b-it")).is_err());
        assert!(parse_chat_model_variant(Some("google/gemma-3-1b-it")).is_err());
        assert!(parse_model_variant("Gemma 3 1B Instruct").is_err());
    }

    #[test]
    fn parse_parakeet_by_repo_tail() {
        let parsed = parse_model_variant("parakeet-tdt-0.6b-v3").unwrap();
        assert_eq!(parsed, ModelVariant::ParakeetTdt06BV3);
    }

    #[test]
    fn resolve_asr_accepts_unversioned_parakeet_family_name() {
        let resolved = resolve_asr_model_variant(Some("parakeet-tdt-0.6b"));
        assert_eq!(resolved, ModelVariant::ParakeetTdt06BV3);
    }

    #[test]
    fn parse_whisper_turbo_repo_alias() {
        let parsed = parse_model_variant("openai/whisper-large-v3-turbo").unwrap();
        assert_eq!(parsed, ModelVariant::WhisperLargeV3Turbo);
        assert_eq!(parsed.family(), ModelFamily::WhisperAsr);
    }

    #[test]
    fn parse_voxtral_repo_alias() {
        let parsed = parse_model_variant("mistralai/Voxtral-Mini-4B-Realtime-2602").unwrap();
        assert_eq!(parsed, ModelVariant::VoxtralMini4BRealtime2602);
        assert_eq!(parsed.family(), ModelFamily::Voxtral);
    }

    #[test]
    fn parse_voxtral_tts_repo_alias() {
        let parsed = parse_model_variant("mistralai/Voxtral-4B-TTS-2603").unwrap();
        assert_eq!(parsed, ModelVariant::Voxtral4BTts2603);
        assert_eq!(parsed.family(), ModelFamily::VoxtralTts);
    }

    #[test]
    fn resolve_asr_accepts_voxtral_display_name() {
        let resolved = resolve_asr_model_variant(Some("Voxtral-Mini-4B-Realtime-2602"));
        assert_eq!(resolved, ModelVariant::VoxtralMini4BRealtime2602);
    }

    #[test]
    fn resolve_asr_accepts_voxtral_family_alias() {
        let resolved = resolve_asr_model_variant(Some("voxtral-mini"));
        assert_eq!(resolved, ModelVariant::VoxtralMini4BRealtime2602);
    }

    #[test]
    fn resolve_asr_accepts_whisper_turbo() {
        let resolved = resolve_asr_model_variant(Some("whisper-large-v3-turbo"));
        assert_eq!(resolved, ModelVariant::WhisperLargeV3Turbo);
    }

    #[test]
    fn resolve_asr_accepts_lfm25_audio() {
        let resolved = resolve_asr_model_variant(Some("LFM2.5-Audio-1.5B-GGUF"));
        assert_eq!(resolved, ModelVariant::Lfm25Audio15BGguf);
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
    fn parse_removed_qwen3_dense_gguf_ids_are_rejected() {
        // The dense Qwen3 GGUF variants were removed from the catalog: every
        // spelling must fail instead of re-resolving to another variant.
        for id in [
            "Qwen3-0.6B-GGUF",
            "Qwen3-1.7B-GGUF",
            "Qwen/Qwen3-1.7B-GGUF",
            "Qwen3-4B-GGUF",
            "Qwen3-4B-Q4_K_M.gguf",
            "Qwen3-8B-GGUF",
            "Qwen3-8B-Q4_K_M.gguf",
        ] {
            assert!(
                parse_chat_model_variant(Some(id)).is_err(),
                "removed variant {id} must not parse"
            );
            assert!(
                parse_model_variant(id).is_err(),
                "removed variant {id} must not parse"
            );
        }
    }

    #[test]
    fn parse_chat_defaults_to_qwen35_4b() {
        let parsed = parse_chat_model_variant(None).unwrap();
        assert_eq!(parsed, ModelVariant::Qwen354BGguf);
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
    fn parse_lfm25_audio_repo_alias() {
        let parsed = parse_model_variant("LiquidAI/LFM2.5-Audio-1.5B-GGUF").unwrap();
        assert_eq!(parsed, ModelVariant::Lfm25Audio15BGguf);
        assert_eq!(parsed.family(), ModelFamily::Lfm25Audio);
    }

    #[test]
    fn parse_lfm25_audio_q40_file_alias() {
        let parsed = parse_model_variant("LFM2.5-Audio-1.5B-Q4_0.gguf").unwrap();
        assert_eq!(parsed, ModelVariant::Lfm25Audio15BGguf);
    }

    #[test]
    fn parse_chat_rejects_lfm25_audio_bundle() {
        assert!(parse_chat_model_variant(Some("LiquidAI/LFM2.5-Audio-1.5B-GGUF")).is_err());
    }

    #[test]
    fn resolve_diarization_llm_defaults_to_qwen_35_4b_gguf() {
        let resolved = resolve_diarization_llm_variant(None).unwrap();
        assert_eq!(resolved, ModelVariant::Qwen354BGguf);
    }

    #[test]
    fn resolve_diarization_llm_accepts_qwen_35_4b_repo_alias() {
        let resolved = resolve_diarization_llm_variant(Some("unsloth/Qwen3.5-4B-GGUF")).unwrap();
        assert_eq!(resolved, ModelVariant::Qwen354BGguf);
    }

    #[test]
    fn resolve_diarization_llm_maps_legacy_qwen_17b_gguf_to_default() {
        // "Qwen3-1.7B-GGUF" left the catalog but persists in stored
        // diarization records; it must resolve to the current default.
        let resolved = resolve_diarization_llm_variant(Some("Qwen/Qwen3-1.7B-GGUF")).unwrap();
        assert_eq!(resolved, ModelVariant::Qwen354BGguf);
        let resolved = resolve_diarization_llm_variant(Some("Qwen3-1.7B-GGUF")).unwrap();
        assert_eq!(resolved, ModelVariant::Qwen354BGguf);
    }

    #[test]
    fn resolve_diarization_llm_accepts_any_chat_model_and_rejects_non_chat() {
        let resolved = resolve_diarization_llm_variant(Some("Qwen3.5-9B")).unwrap();
        assert_eq!(resolved, ModelVariant::Qwen359BGguf);
        let resolved = resolve_diarization_llm_variant(Some("Gemma-3-4b-it")).unwrap();
        assert_eq!(resolved, ModelVariant::Gemma34BIt);
        assert!(resolve_diarization_llm_variant(Some("Whisper-Large-v3-Turbo")).is_err());
        assert!(resolve_diarization_llm_variant(Some("Nemotron-3-Diarization")).is_err());
        assert!(resolve_diarization_llm_variant(Some("Kokoro-82M")).is_err());
    }

    #[test]
    fn parse_qwen_chat_4b_repo_is_rejected() {
        assert!(parse_chat_model_variant(Some("Qwen/Qwen3-4B")).is_err());
    }

    #[test]
    fn parse_qwen_chat_14b_gguf_file_alias() {
        let parsed = parse_chat_model_variant(Some("Qwen3-14B-Q4_K_M.gguf")).unwrap();
        assert_eq!(parsed, ModelVariant::Qwen314BGguf);
    }

    #[test]
    fn parse_qwen3_asr_06b_gguf_filename_alias() {
        let parsed = parse_model_variant("qwen3_asr_0.6b_q8_0.gguf").unwrap();
        assert_eq!(parsed, ModelVariant::Qwen3Asr06BGguf);
    }

    #[test]
    fn parse_qwen3_asr_17b_alias() {
        let parsed = parse_model_variant("qwen3-asr-1.7b").unwrap();
        assert_eq!(parsed, ModelVariant::Qwen3Asr17BGguf);
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
    fn parse_diarization_by_repo_tail() {
        let parsed = parse_model_variant("diar_streaming_sortformer_4spk-v2.1").unwrap();
        assert_eq!(parsed, ModelVariant::DiarStreamingSortformer4SpkV21);
    }

    #[test]
    fn parse_nemotron3_diarization_by_all_identifier_forms() {
        for raw in [
            "Nemotron-3-Diarization",
            "nvidia/Nemotron-3-Diarization",
            "nemotron-3-diarization",
            "nemotron3-diarization",
            "Nemotron 3 Diarization",
        ] {
            let parsed = parse_model_variant(raw).unwrap();
            assert_eq!(parsed, ModelVariant::Nemotron3Diarization, "input: {raw}");
        }
        // The fuzzy heuristic resolves free-form mentions.
        let heuristic = parse_model_variant("the new nemotron diarization model").unwrap();
        assert_eq!(heuristic, ModelVariant::Nemotron3Diarization);
    }

    #[test]
    fn resolve_diarization_selects_nemotron3_by_name_but_keeps_v21_default() {
        let resolved = resolve_diarization_model_variant(Some("Nemotron-3-Diarization"));
        assert_eq!(resolved, ModelVariant::Nemotron3Diarization);
        let resolved = resolve_diarization_model_variant(Some("unknown-model"));
        assert_eq!(resolved, ModelVariant::DiarStreamingSortformer4SpkV21);
        let resolved = resolve_diarization_model_variant(None);
        assert_eq!(resolved, ModelVariant::DiarStreamingSortformer4SpkV21);
    }

    /// PIPE-18: speaker limits above a diarizer's channels (4 for v2.1, 8
    /// for Nemotron-3) are refused instead of clamped.
    #[test]
    fn diarization_speaker_limits_above_the_channel_count_are_refused() {
        let v21 = ModelVariant::DiarStreamingSortformer4SpkV21
            .diarization_speaker_channels()
            .unwrap();
        let nemotron = ModelVariant::Nemotron3Diarization
            .diarization_speaker_channels()
            .unwrap();
        assert_eq!((v21, nemotron), (4, 8));
        assert_eq!(ModelVariant::Qwen306B.diarization_speaker_channels(), None);
        assert!(validate_diarization_speaker_limits(None, None, v21).is_ok());
        assert!(validate_diarization_speaker_limits(Some(1), Some(4), v21).is_ok());
        assert!(validate_diarization_speaker_limits(Some(8), Some(8), nemotron).is_ok());
        for (min, max, channels, field) in [
            (Some(5), None, v21, "min_speakers"),
            (None, Some(5), v21, "max_speakers"),
            (None, Some(9), nemotron, "max_speakers"),
            (Some(3), Some(2), v21, "min_speakers"),
        ] {
            let err = validate_diarization_speaker_limits(min, max, channels).unwrap_err();
            assert!(err.contains(field), "{err}");
        }
    }

    #[test]
    fn resolve_diarization_strict_accepts_diarization_ids() {
        let resolved =
            resolve_diarization_model_variant_strict("Nemotron-3-Diarization").unwrap();
        assert_eq!(resolved, ModelVariant::Nemotron3Diarization);
        let resolved = resolve_diarization_model_variant_strict(
            "nvidia/diar_streaming_sortformer_4spk-v2.1",
        )
        .unwrap();
        assert_eq!(resolved, ModelVariant::DiarStreamingSortformer4SpkV21);
    }

    /// PIPE-3: the short Sortformer ids older CLIs sent resolve to v2.1.
    #[test]
    fn resolve_diarization_strict_accepts_short_sortformer_aliases() {
        for raw in ["sortformer-4spk", "sortformer4spk", "Sortformer_4spk"] {
            assert_eq!(
                resolve_diarization_model_variant_strict(raw),
                Ok(ModelVariant::DiarStreamingSortformer4SpkV21),
                "input: {raw}"
            );
        }
        assert!(resolve_diarization_model_variant_strict("nvidia/sortformer").is_err());
    }

    #[test]
    fn resolve_diarization_strict_rejects_non_diarization_and_unknown_ids() {
        let chat_variant = resolve_diarization_model_variant_strict("Qwen3.5-4B")
            .expect_err("chat model must not resolve as diarization");
        assert!(chat_variant.contains("not a diarization model"));
        let unknown = resolve_diarization_model_variant_strict("Not-A-Real-Model")
            .expect_err("unknown id must not resolve");
        assert!(unknown.contains("Unknown diarization model id"));
    }
}
