//! ASR runtime methods routed through the unified core engine.

use crate::catalog::{parse_model_variant, resolve_asr_model_variant};
use crate::engine::EngineCoreRequest;
use crate::error::{Error, Result};
use crate::model::ModelVariant;
use crate::runtime::audio_io::{base64_decode, decode_audio_bytes};
use crate::runtime::service::RuntimeService;
use crate::runtime::types::AsrTranscription;

enum AsrAudioInput<'a> {
    Base64(&'a str),
    Bytes(&'a [u8]),
}

impl RuntimeService {
    async fn asr_transcribe_audio_chat_samples<F>(
        &self,
        variant: ModelVariant,
        samples: Vec<f32>,
        sample_rate: u32,
        mut on_delta: F,
    ) -> Result<AsrTranscription>
    where
        F: FnMut(String),
    {
        self.load_model(variant).await?;
        let model = self
            .model_registry
            .get_audio_chat(variant)
            .await
            .ok_or_else(|| {
                Error::ModelNotFound(format!("Audio-chat model {variant} is not loaded"))
            })?;

        let mut delta_sink = |delta: &str| {
            if !delta.is_empty() {
                on_delta(delta.to_string());
            }
        };
        let output = model.transcribe_with_callback(&samples, sample_rate, &mut delta_sink)?;

        Ok(AsrTranscription {
            text: output.text,
            language: output.language,
            duration_secs: if sample_rate > 0 {
                samples.len() as f32 / sample_rate as f32
            } else {
                0.0
            },
        })
    }

    async fn asr_transcribe_audio_chat_base64<F>(
        &self,
        variant: ModelVariant,
        audio_base64: &str,
        on_delta: F,
    ) -> Result<AsrTranscription>
    where
        F: FnMut(String),
    {
        let audio_bytes = base64_decode(audio_base64)?;
        let (samples, sample_rate) = decode_audio_bytes(&audio_bytes)?;
        self.asr_transcribe_audio_chat_samples(variant, samples, sample_rate, on_delta)
            .await
    }

    async fn asr_transcribe_audio_chat_bytes<F>(
        &self,
        variant: ModelVariant,
        audio_bytes: &[u8],
        on_delta: F,
    ) -> Result<AsrTranscription>
    where
        F: FnMut(String),
    {
        let (samples, sample_rate) = decode_audio_bytes(audio_bytes)?;
        self.asr_transcribe_audio_chat_samples(variant, samples, sample_rate, on_delta)
            .await
    }

    async fn build_asr_request(
        &self,
        variant: ModelVariant,
        audio_input: AsrAudioInput<'_>,
        language: Option<&str>,
        correlation_id: Option<&str>,
    ) -> Result<EngineCoreRequest> {
        self.load_model(variant).await?;

        let mut request = match audio_input {
            AsrAudioInput::Base64(audio_base64) => EngineCoreRequest::asr(audio_base64.to_string()),
            AsrAudioInput::Bytes(audio_bytes) => EngineCoreRequest::asr_bytes(audio_bytes.to_vec()),
        };
        request.model_variant = Some(variant);
        request.language = language.map(|s| s.to_string());
        request.correlation_id = correlation_id.map(|s| s.to_string());
        Ok(request)
    }

    pub(crate) async fn asr_transcribe_with_variant(
        &self,
        variant: ModelVariant,
        audio_base64: &str,
        language: Option<&str>,
        correlation_id: Option<&str>,
    ) -> Result<AsrTranscription> {
        if variant.is_audio_chat() {
            return self
                .asr_transcribe_audio_chat_base64(variant, audio_base64, |_delta| {})
                .await;
        }

        let request = self
            .build_asr_request(
                variant,
                AsrAudioInput::Base64(audio_base64),
                language,
                correlation_id,
            )
            .await?;
        let output = self.run_request(request).await?;
        let text = output.text.unwrap_or_default();

        Ok(AsrTranscription {
            text,
            language: language.map(|s| s.to_string()),
            duration_secs: output.audio.duration_secs,
        })
    }

    pub(crate) async fn asr_transcribe_with_variant_streaming<F>(
        &self,
        variant: ModelVariant,
        audio_base64: &str,
        language: Option<&str>,
        correlation_id: Option<&str>,
        mut on_delta: F,
    ) -> Result<AsrTranscription>
    where
        F: FnMut(String) + Send + 'static,
    {
        if variant.is_audio_chat() {
            return self
                .asr_transcribe_audio_chat_base64(variant, audio_base64, on_delta)
                .await;
        }

        let request = self
            .build_asr_request(
                variant,
                AsrAudioInput::Base64(audio_base64),
                language,
                correlation_id,
            )
            .await?;
        let mut streamed_text = String::new();
        let output = self
            .run_streaming_request(request, |chunk| {
                if let Some(delta) = chunk.text {
                    if !delta.is_empty() {
                        streamed_text.push_str(&delta);
                        on_delta(delta);
                    }
                }
                std::future::ready(Ok(()))
            })
            .await?;
        let text = output.text.unwrap_or(streamed_text);

        Ok(AsrTranscription {
            text,
            language: language.map(|s| s.to_string()),
            duration_secs: output.audio.duration_secs,
        })
    }

    pub(crate) async fn asr_transcribe_bytes_with_variant(
        &self,
        variant: ModelVariant,
        audio_bytes: &[u8],
        language: Option<&str>,
        correlation_id: Option<&str>,
    ) -> Result<AsrTranscription> {
        if variant.is_audio_chat() {
            return self
                .asr_transcribe_audio_chat_bytes(variant, audio_bytes, |_delta| {})
                .await;
        }

        let request = self
            .build_asr_request(
                variant,
                AsrAudioInput::Bytes(audio_bytes),
                language,
                correlation_id,
            )
            .await?;
        let output = self.run_request(request).await?;
        let text = output.text.unwrap_or_default();

        Ok(AsrTranscription {
            text,
            language: language.map(|s| s.to_string()),
            duration_secs: output.audio.duration_secs,
        })
    }

    pub(crate) async fn asr_transcribe_bytes_with_variant_streaming<F>(
        &self,
        variant: ModelVariant,
        audio_bytes: &[u8],
        language: Option<&str>,
        correlation_id: Option<&str>,
        mut on_delta: F,
    ) -> Result<AsrTranscription>
    where
        F: FnMut(String) + Send + 'static,
    {
        if variant.is_audio_chat() {
            return self
                .asr_transcribe_audio_chat_bytes(variant, audio_bytes, on_delta)
                .await;
        }

        let request = self
            .build_asr_request(
                variant,
                AsrAudioInput::Bytes(audio_bytes),
                language,
                correlation_id,
            )
            .await?;
        let mut streamed_text = String::new();
        let output = self
            .run_streaming_request(request, |chunk| {
                if let Some(delta) = chunk.text {
                    if !delta.is_empty() {
                        streamed_text.push_str(&delta);
                        on_delta(delta);
                    }
                }
                std::future::ready(Ok(()))
            })
            .await?;
        let text = output.text.unwrap_or(streamed_text);

        Ok(AsrTranscription {
            text,
            language: language.map(|s| s.to_string()),
            duration_secs: output.audio.duration_secs,
        })
    }

    /// Transcribe audio with Voxtral Realtime.
    pub async fn voxtral_transcribe(
        &self,
        audio_base64: &str,
        language: Option<&str>,
    ) -> Result<AsrTranscription> {
        self.voxtral_transcribe_streaming(audio_base64, language, |_delta| {})
            .await
    }

    /// Transcribe audio with Voxtral Realtime and emit incremental deltas.
    pub async fn voxtral_transcribe_streaming<F>(
        &self,
        audio_base64: &str,
        language: Option<&str>,
        on_delta: F,
    ) -> Result<AsrTranscription>
    where
        F: FnMut(String) + Send + 'static,
    {
        self.asr_transcribe_with_variant_streaming(
            ModelVariant::VoxtralMini4BRealtime2602,
            audio_base64,
            language,
            None,
            on_delta,
        )
        .await
    }

    /// Transcribe audio with native ASR models.
    pub async fn asr_transcribe(
        &self,
        audio_base64: &str,
        model_id: Option<&str>,
        language: Option<&str>,
    ) -> Result<AsrTranscription> {
        let variant = resolve_asr_model_variant(model_id);
        self.asr_transcribe_with_variant(variant, audio_base64, language, None)
            .await
    }

    pub async fn asr_transcribe_bytes(
        &self,
        audio_bytes: &[u8],
        model_id: Option<&str>,
        language: Option<&str>,
    ) -> Result<AsrTranscription> {
        let variant = resolve_asr_model_variant(model_id);
        self.asr_transcribe_bytes_with_variant(variant, audio_bytes, language, None)
            .await
    }

    /// Transcribe audio and emit deltas.
    pub async fn asr_transcribe_streaming<F>(
        &self,
        audio_base64: &str,
        model_id: Option<&str>,
        language: Option<&str>,
        on_delta: F,
    ) -> Result<AsrTranscription>
    where
        F: FnMut(String) + Send + 'static,
    {
        self.asr_transcribe_streaming_with_correlation(
            audio_base64,
            model_id,
            language,
            None,
            on_delta,
        )
        .await
    }

    pub async fn asr_transcribe_streaming_bytes<F>(
        &self,
        audio_bytes: &[u8],
        model_id: Option<&str>,
        language: Option<&str>,
        on_delta: F,
    ) -> Result<AsrTranscription>
    where
        F: FnMut(String) + Send + 'static,
    {
        self.asr_transcribe_streaming_bytes_with_correlation(
            audio_bytes,
            model_id,
            language,
            None,
            on_delta,
        )
        .await
    }

    /// Transcribe audio and emit deltas with request correlation metadata.
    pub async fn asr_transcribe_streaming_with_correlation<F>(
        &self,
        audio_base64: &str,
        model_id: Option<&str>,
        language: Option<&str>,
        correlation_id: Option<&str>,
        on_delta: F,
    ) -> Result<AsrTranscription>
    where
        F: FnMut(String) + Send + 'static,
    {
        let variant = resolve_asr_model_variant(model_id);
        self.asr_transcribe_with_variant_streaming(
            variant,
            audio_base64,
            language,
            correlation_id,
            on_delta,
        )
        .await
    }

    pub async fn asr_transcribe_streaming_bytes_with_correlation<F>(
        &self,
        audio_bytes: &[u8],
        model_id: Option<&str>,
        language: Option<&str>,
        correlation_id: Option<&str>,
        on_delta: F,
    ) -> Result<AsrTranscription>
    where
        F: FnMut(String) + Send + 'static,
    {
        let variant = resolve_asr_model_variant(model_id);
        self.asr_transcribe_bytes_with_variant_streaming(
            variant,
            audio_bytes,
            language,
            correlation_id,
            on_delta,
        )
        .await
    }

    /// Force alignment remains a specialized operation not expressed by the
    /// generic engine output type.
    pub async fn force_align(
        &self,
        audio_base64: &str,
        reference_text: &str,
    ) -> Result<Vec<(String, u32, u32)>> {
        self.force_align_with_model_and_language(audio_base64, reference_text, None, None)
            .await
    }

    pub async fn force_align_with_model(
        &self,
        audio_base64: &str,
        reference_text: &str,
        model_id: Option<&str>,
    ) -> Result<Vec<(String, u32, u32)>> {
        self.force_align_with_model_and_language(audio_base64, reference_text, None, model_id)
            .await
    }

    pub async fn force_align_with_model_and_language(
        &self,
        audio_base64: &str,
        reference_text: &str,
        language: Option<&str>,
        model_id: Option<&str>,
    ) -> Result<Vec<(String, u32, u32)>> {
        let audio_bytes = base64_decode(audio_base64)?;
        self.force_align_bytes_with_model_and_language(
            &audio_bytes,
            reference_text,
            language,
            model_id,
        )
        .await
    }

    pub async fn force_align_bytes_with_model_and_language(
        &self,
        audio_bytes: &[u8],
        reference_text: &str,
        language: Option<&str>,
        model_id: Option<&str>,
    ) -> Result<Vec<(String, u32, u32)>> {
        let variant = resolve_forced_aligner_variant(model_id)?;
        self.load_model(variant).await?;

        let model = self
            .model_registry
            .get_asr(variant)
            .await
            .ok_or_else(|| Error::ModelNotFound(variant.to_string()))?;

        let (samples, sample_rate) = decode_audio_bytes(audio_bytes)?;
        model.force_align(&samples, sample_rate, reference_text, language)
    }
}

pub(crate) fn resolve_forced_aligner_variant(model_id: Option<&str>) -> Result<ModelVariant> {
    let variant = match model_id {
        Some(raw) => {
            parse_model_variant(raw).map_err(|err| Error::InvalidInput(err.to_string()))?
        }
        None => ModelVariant::Qwen3ForcedAligner06B,
    };

    if !variant.is_forced_aligner() {
        return Err(Error::InvalidInput(format!(
            "Model {} is not a forced aligner model",
            variant.dir_name()
        )));
    }

    Ok(variant)
}

#[cfg(all(test, unix))]
mod tests {
    use super::*;
    use crate::backends::BackendPreference;
    use crate::config::EngineConfig;
    use std::sync::{Mutex, OnceLock};
    use uuid::Uuid;

    fn env_lock() -> &'static Mutex<()> {
        static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        LOCK.get_or_init(|| Mutex::new(()))
    }

    #[tokio::test]
    async fn parakeet_load_rejects_invalid_nemo_archive() {
        let _guard = env_lock().lock().expect("env lock poisoned");

        let root = std::env::temp_dir().join(format!("izwi-parakeet-runtime-{}", Uuid::new_v4()));
        let model_dir = root.join("Parakeet-TDT-0.6B-v3");
        std::fs::create_dir_all(&model_dir).unwrap();
        std::fs::write(model_dir.join("parakeet-tdt-0.6b-v3.nemo"), b"mock-nemo").unwrap();

        let mut config = EngineConfig::default();
        config.models_dir = root.clone();
        config.backend = BackendPreference::Cpu;

        let engine = RuntimeService::new(config).unwrap();
        let err = engine
            .load_model(ModelVariant::ParakeetTdt06BV3)
            .await
            .expect_err("invalid .nemo archive should fail to load");
        let msg = err.to_string();
        assert!(
            msg.contains(".nemo")
                || msg.contains("archive")
                || msg.contains("Failed to load")
                || msg.contains("invalid")
        );

        let _ = std::fs::remove_dir_all(root);
    }
}
