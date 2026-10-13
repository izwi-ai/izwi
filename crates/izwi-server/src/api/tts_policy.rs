use izwi_core::catalog::ModelFamily;
pub(crate) use izwi_core::runtime_models::architectures::qwen3::tts::{
    qwen3_tts_auto_max_frames_for_text as qwen_tts_auto_max_frames_for_text,
};
use izwi_core::runtime_models::architectures::qwen3::tts::qwen3_tts_auto_max_output_frames;
use izwi_core::runtime_models::architectures::vibevoice::tts::{
    vibevoice_tts_auto_max_frames_for_text, vibevoice_tts_auto_max_output_frames,
};
use izwi_core::runtime_models::architectures::voxtral::tts::voxtral_tts_auto_max_frames_for_text;
use izwi_core::ModelVariant;

/// The longest speech input one request may carry, in UTF-8 bytes
/// (`IZWI_TTS_MAX_TEXT_BYTES`, default 1 MiB). Long form splits text into
/// model-sized calls, so this bounds only the request itself (TPIPE-14).
pub(crate) fn tts_max_text_bytes() -> usize {
    std::env::var("IZWI_TTS_MAX_TEXT_BYTES")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(1024 * 1024)
}

/// Refuse an over-limit speech input with 413 before any model loads.
pub(crate) fn check_tts_input_length(text: &str) -> Result<(), crate::error::ApiError> {
    let limit = tts_max_text_bytes();
    if text.len() > limit {
        return Err(crate::error::ApiError::payload_too_large(format!(
            "Speech input is {} bytes, above the {limit}-byte limit; send it as several requests",
            text.len()
        )));
    }
    Ok(())
}

/// Optional speech-sampling fields a request may set beside `temperature`
/// and `top_k`. Anything left unset keeps the model family's reference
/// default (TPIPE-24): Qwen3-TTS generation_config, LFM 0.8 / top-k 64,
/// VibeVoice CFG 1.3 with 10 steps, fish-speech 0.7 / 0.7, Voxtral greedy.
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize, utoipa::ToSchema)]
pub struct SpeechSamplingRequest {
    /// Nucleus sampling threshold, in (0, 1].
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    /// Sampling seed (Qwen3-TTS, LFM2.5-Audio, Fish S2, Voxtral noise).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub seed: Option<u64>,
    /// Repetition penalty (Qwen3-TTS), >= 1.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub repetition_penalty: Option<f32>,
    /// Qwen3-TTS residual-codebook sampling.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub subtalker_do_sample: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub subtalker_temperature: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub subtalker_top_k: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub subtalker_top_p: Option<f32>,
    /// VibeVoice classifier-free guidance scale.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cfg_scale: Option<f32>,
    /// VibeVoice diffusion steps per frame.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub diffusion_steps: Option<usize>,
    /// Speak numbers, currency, dates, times and units as words (English).
    /// Unset keeps the model's default: on for Kokoro, off for models that
    /// read them themselves.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub normalize_text: Option<bool>,
}

impl SpeechSamplingRequest {
    /// The engine's speech sampling for a request that also set `temperature`
    /// and `top_k` (each `None` when the client left it unset).
    pub(crate) fn options(
        &self,
        temperature: Option<f32>,
        top_k: Option<usize>,
    ) -> izwi_core::SpeechSamplingOptions {
        izwi_core::SpeechSamplingOptions {
            temperature,
            top_p: self.top_p,
            top_k,
            repetition_penalty: self.repetition_penalty,
            seed: self.seed,
            subtalker_do_sample: self.subtalker_do_sample,
            subtalker_temperature: self.subtalker_temperature,
            subtalker_top_k: self.subtalker_top_k,
            subtalker_top_p: self.subtalker_top_p,
            cfg_scale: self.cfg_scale,
            diffusion_steps: self.diffusion_steps,
            normalize_text: self.normalize_text,
        }
    }
}

pub(crate) fn resolve_tts_output_frames(
    variant: ModelVariant,
    text: &str,
    requested_frames: Option<usize>,
) -> Option<usize> {
    let model_max_frames = variant.tts_max_output_frames_hint()?;
    match requested_frames {
        Some(value) if value > 0 => Some(value.clamp(1, model_max_frames)),
        Some(0) | None if variant == ModelVariant::Voxtral4BTts2603 => {
            Some(voxtral_tts_auto_max_frames_for_text(text).min(model_max_frames))
        }
        Some(0) | None if variant == ModelVariant::VibeVoice15BTts => {
            Some(vibevoice_tts_auto_max_frames_for_text(text).min(model_max_frames))
        }
        Some(0) | None if variant == ModelVariant::FishAudioS2Pro => {
            Some(fish_s2_tts_auto_max_frames_for_text(text).min(model_max_frames))
        }
        Some(0) | None if variant.family() == ModelFamily::Qwen3Tts => {
            Some(qwen_tts_auto_max_frames_for_text(text).min(model_max_frames))
        }
        Some(0) | None => Some(model_max_frames),
        Some(_) => unreachable!("positive requested frames handled above"),
    }
}

/// Whether `text` is too long for one automatic request of `variant`: its
/// automatic budget reaches the per-request cap, so one request would cut
/// it. Only families that advertise automatic long form split; Fish plans
/// its own segments.
pub(crate) fn tts_text_exceeds_one_request(variant: ModelVariant, text: &str) -> bool {
    if !variant
        .speech_capabilities()
        .is_some_and(|capabilities| capabilities.supports_auto_long_form)
    {
        return false;
    }
    let cap = match variant.family() {
        ModelFamily::Qwen3Tts => qwen3_tts_auto_max_output_frames(),
        ModelFamily::VibeVoiceTts => vibevoice_tts_auto_max_output_frames(),
        _ => return false,
    };
    resolve_tts_output_frames(variant, text, None).is_some_and(|frames| frames >= cap)
}

fn fish_s2_tts_auto_max_frames_for_text(text: &str) -> usize {
    // Whitespace alone treats an entire CJK paragraph as one word. Include a
    // character estimate and count CJK syllabic characters separately; leave
    // time for pauses and EOS rather than truncating at the speaking estimate.
    let non_space = text.chars().filter(|ch| !ch.is_whitespace()).count();
    let cjk = text
        .chars()
        .filter(|ch| {
            matches!(*ch as u32,
                0x3040..=0x30ff | 0x3400..=0x4dbf | 0x4e00..=0x9fff |
                0xac00..=0xd7af | 0x20000..=0x3134f
            )
        })
        .count();
    let estimated_words = text
        .split_whitespace()
        .count()
        .max(non_space.saturating_sub(cjk).div_ceil(5));
    // This is a per-segment ceiling, not a total-recording limit. Budget for
    // slower narration and pauses; the model's EOS ends normal speech earlier.
    let estimated_secs = (estimated_words as f32 / 1.8 + cjk as f32 / 2.5 + 4.0).clamp(6.0, 120.0);
    let frames = (estimated_secs * ModelVariant::FISH_S2_PRO_FRAME_RATE_HZ).ceil() as usize;
    frames.clamp(96, ModelVariant::FISH_S2_PRO_MAX_OUTPUT_FRAMES)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn qwen_auto_budget_is_text_sized_and_bounded() {
        let short =
            resolve_tts_output_frames(ModelVariant::Qwen3Tts12Hz06BCustomVoice, "hello", None)
                .expect("Qwen frame hint");
        let longer = resolve_tts_output_frames(
            ModelVariant::Qwen3Tts12Hz06BCustomVoice,
            "This sentence is intentionally much longer than hello and needs more speech frames.",
            None,
        )
        .expect("Qwen frame hint");

        assert!(short < ModelVariant::QWEN3_TTS_MAX_OUTPUT_FRAMES);
        assert!(longer > short);
        assert!(longer <= ModelVariant::QWEN3_TTS_MAX_OUTPUT_FRAMES);
    }

    #[test]
    fn fish_s2_auto_budget_handles_unspaced_multilingual_text() {
        let short = fish_s2_tts_auto_max_frames_for_text("Hello");
        for text in [
            "你好世界".repeat(30),
            "こんにちは世界".repeat(20),
            "안녕하세요".repeat(24),
            "longunspacedtext".repeat(20),
        ] {
            let budget = fish_s2_tts_auto_max_frames_for_text(&text);
            assert!(budget > short, "{text}");
            assert!(budget <= ModelVariant::FISH_S2_PRO_MAX_OUTPUT_FRAMES);
        }
        assert_eq!(
            resolve_tts_output_frames(ModelVariant::FishAudioS2Pro, "你好", Some(10)),
            Some(10)
        );
        assert_eq!(
            resolve_tts_output_frames(ModelVariant::FishAudioS2Pro, "你好", Some(usize::MAX)),
            Some(ModelVariant::FISH_S2_PRO_MAX_OUTPUT_FRAMES)
        );
        assert_eq!(
            resolve_tts_output_frames(ModelVariant::FishAudioS2Pro, "你好", Some(0)),
            resolve_tts_output_frames(ModelVariant::FishAudioS2Pro, "你好", None)
        );
    }

    #[test]
    fn explicit_budget_is_clamped_for_every_route() {
        assert_eq!(
            resolve_tts_output_frames(
                ModelVariant::Qwen3Tts12Hz06BCustomVoice,
                "hello",
                Some(usize::MAX),
            ),
            Some(ModelVariant::QWEN3_TTS_MAX_OUTPUT_FRAMES)
        );
    }
}
